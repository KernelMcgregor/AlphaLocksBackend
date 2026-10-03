"""Build models/ufc/grade_table.json from walk-forward backtests (see app/services/ufc/grading.py).

Bets are rebuilt from data in the repo + DB only, so this runs in CI after each event:
  probabilities  ensemble OOF (models/ufc/h2h/ensemble_oof.csv) with the ensemble's market
                 stacks; method_v2 OOF (models/ufc/method/method_oof.csv); rounds OOF
                 (models/ufc/method/rounds_oof.csv, anchored to method_v2's decision prob)
  prices         moneyline open/close: ufc_fight_odds_open_close (opening = the consensus
                 price; closing = best across sportsbooks); props: ufc_prop_odds_history
                 rows with source 'bfo_close' (best and median American price)
  results        ufc_fights

Rule (identical for every family): per fight and market, the side with the highest EV at the
typical book's price if EV > 0; flat 1 unit; ROI graded at that typical price. Prop
probabilities are the model blended 50/50 (log-odds) with the de-vigged closing consensus,
as the picks API serves them; moneyline probabilities are the ensemble's market stack.

Usage:
    DATABASE_URL=... python -m scripts.build_grade_table
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from app.config import settings
from app.services.ufc.grading import blend_prob, decimal, family_table, save_table
from app.services.ufc.method_ratings import method_class
from app.services.ufc.round_hazard import anchor

log = logging.getLogger("build_grade_table")
ROOT = Path(__file__).resolve().parents[1]
CELLS = ("red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec")
EXCHANGES = ("polymarket", "kalshi", "consensus", "mean")


def _results(engine) -> pd.DataFrame:
    with engine.connect() as c:
        rows = c.execute(text("""select id::text, method, details, winner_id, red_fighter_id,
                                        blue_fighter_id, fight_time_seconds, max_fight_time_seconds
                                 from ufc.ufc_fights where method is not null and method <> ''""")).all()
    out = []
    for fid, method, details, w, r, b, secs, sched in rows:
        k = method_class(method, details, w)
        out.append({"fight_id": fid, "cls": k, "winner": "red" if w == r else ("blue" if w == b else None),
                    "t": (secs or 0) / 60.0, "sched": 25.0 if (sched or 0) >= 1500 else 15.0})
    return pd.DataFrame(out)


def _bets(df: pd.DataFrame, sides: list[tuple[str, str, str]], family_of, blend: bool = False) -> pd.DataFrame:
    """df has p_<side>, best_<side>, med_<side>, won_<side> and, for props, q_<side> (the
    de-vigged closing consensus). The probability is blend_prob(p, q) when blend=True, as the
    picks API serves props. Each fight bets the side with the highest EV at the TYPICAL book's
    price (median; best if no median) if that EV > 0. Returns one row per bet with profit at
    the typical price (graded) and at the best price (shown)."""
    out = []
    for r in df.itertuples(index=False):
        rd = r._asdict()
        best = None
        for side, _, _ in sides:
            p = rd.get(f"p_{side}")
            if p is None or p != p:
                continue
            if blend:
                p = blend_prob(p, rd.get(f"q_{side}"))
            d_typ = decimal(rd.get(f"med_{side}")) or decimal(rd.get(f"best_{side}"))
            if d_typ is None:
                continue
            e = p * d_typ - 1
            if e > 0 and (best is None or e > best[1]):
                best = (side, e, d_typ)
        if best is None:
            continue
        side, e, d_typ = best
        won = bool(rd[f"won_{side}"])
        code = family_of.__code__
        fam = family_of(side, rd) if "rd" in code.co_varnames[:code.co_argcount] else family_of(side)
        d_best = decimal(rd.get(f"best_{side}")) or d_typ
        out.append({"family": fam, "ev": e, "decimal": d_typ,
                    "profit_best": d_typ - 1 if won else -1.0,        # graded: typical book
                    "profit_median": d_best - 1 if won else -1.0})    # shown: best price
    return pd.DataFrame(out)


def winner_bets(engine, res: pd.DataFrame) -> pd.DataFrame:
    from app.services.ufc import ensemble
    ens = ensemble.load()
    oof = pd.read_csv(ROOT / "models/ufc/h2h/ensemble_oof.csv", dtype={"fight_id": str})
    with engine.connect() as c:
        oc = pd.DataFrame(c.execute(text("""select fight_id::text fight_id, bookmaker, red_open, blue_open,
                                                   red_close, blue_close, red_open_prob, red_close_prob
                                            from ufc.ufc_fight_odds_open_close""")).all(),
                          columns=["fight_id", "book", "red_open", "blue_open", "red_close", "blue_close",
                                   "q_open", "q_close"])
    cons = oc[oc["book"] == "Consensus"].drop_duplicates("fight_id").set_index("fight_id")
    books = oc[~oc["book"].str.lower().str.contains("|".join(EXCHANGES))]
    # median book = median DECIMAL price, back to American (a median of American odds is
    # meaningless across even money: -110 and +100 would give -5)
    def _am_median(x):
        d = [decimal(v) for v in x if decimal(v)]
        if not d:
            return np.nan
        m = float(np.median(d))
        return (m - 1) * 100 if m >= 2 else -100 / (m - 1)
    best_close = books.groupby("fight_id").agg(best_red=("red_close", "max"), best_blue=("blue_close", "max"),
                                               med_red=("red_close", _am_median), med_blue=("blue_close", _am_median))
    d = oof.merge(res, on="fight_id").join(cons, on="fight_id", how="inner").join(best_close, on="fight_id")
    d = d[d["winner"].notna()]
    m = d["model_prob"].to_numpy(float)
    out = []
    for fam, q, br, bb, mr, mb in (
            ("winner_open", "q_open", "red_open", "blue_open", "red_open", "blue_open"),
            ("winner_close", "q_close", "best_red", "best_blue", "med_red", "med_blue")):
        p_red = ens.final_prob(m, d[q].to_numpy(float), opening=(fam == "winner_open")) if ens else m
        x = pd.DataFrame({"p_red": p_red, "p_blue": 1 - p_red,
                          "best_red": d[br].fillna(d["red_close"]).to_numpy(float) if fam == "winner_close" else d[br].to_numpy(float),
                          "best_blue": d[bb].fillna(d["blue_close"]).to_numpy(float) if fam == "winner_close" else d[bb].to_numpy(float),
                          "med_red": d[mr].to_numpy(float), "med_blue": d[mb].to_numpy(float),
                          "won_red": (d["winner"] == "red").to_numpy(), "won_blue": (d["winner"] == "blue").to_numpy()})
        out.append(_bets(x, [("red", "", ""), ("blue", "", "")], lambda s, f=fam: f))
    return pd.concat(out, ignore_index=True)


def prop_prices(engine) -> pd.DataFrame:
    with engine.connect() as c:
        df = pd.DataFrame(c.execute(text("""select fight_id::text fight_id, market, best_american, median_american, prob
                                            from ufc.ufc_prop_odds_history
                                            where source = 'bfo_close' and best_american is not null""")).all(),
                          columns=["fight_id", "market", "best", "med", "prob"])
    best = df.pivot_table(index="fight_id", columns="market", values="best", aggfunc="last").add_prefix("best_")
    med = df.pivot_table(index="fight_id", columns="market", values="med", aggfunc="last").add_prefix("med_")
    q = df.pivot_table(index="fight_id", columns="market", values="prob", aggfunc="last").add_prefix("q_")
    return best.join(med).join(q).reset_index()


def prop_bets(engine, res: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    from app.services.ufc import ensemble
    ens = ensemble.load()
    oof = pd.read_csv(ROOT / "models/ufc/method/method_oof.csv", dtype={"fight_id": str})
    e_oof = pd.read_csv(ROOT / "models/ufc/h2h/ensemble_oof.csv", dtype={"fight_id": str})
    d = oof.merge(e_oof, on="fight_id").merge(res, on="fight_id").merge(px, on="fight_id")
    d = d[d["cls"].notna()]
    q = d["odds_red_prob"].to_numpy(float)
    p_red = ens.final_prob(d["model_prob"].to_numpy(float), q) if ens else d["model_prob"].to_numpy(float)
    d = d.assign(p_red=p_red)   # served-style winner prob (market blend), reused by round_bets
    six = np.column_stack([p_red * d[f"red_{c}"] for c in ("ko", "sub", "dec")] +
                          [(1 - p_red) * d[f"blue_{c}"] for c in ("ko", "sub", "dec")])
    out = []
    # winner x method, graded per method: the market misprices them differently (across
    # 2,831 priced fights "favourite by decision" is de-vigged at 27.7% but happens 32.8%,
    # while "favourite by KO" is 23.9% vs 22.4%). Per fight and method, the better corner.
    x = pd.DataFrame({f"p_{c}": six[:, j] for j, c in enumerate(CELLS)})
    for c in CELLS:
        x[f"best_{c}"] = d.get(f"best_{c}"); x[f"med_{c}"] = d.get(f"med_{c}"); x[f"q_{c}"] = d.get(f"q_{c}")
        side, m = c.split("_")
        x[f"won_{c}"] = ((d["winner"] == side) & (d["cls"] == m)).to_numpy()
    # ...and separately for the favourite and the underdog: across 2,831 priced fights
    # "favourite by decision" is de-vigged at 27.7% but happens 32.8%, while "underdog by
    # decision" is priced about right (15.8% vs 16.4%).
    x["fav"] = np.where(p_red >= 0.5, "red", "blue")
    for m in ("ko", "sub", "dec"):
        out.append(_bets(x, [(f"red_{m}", "", ""), (f"blue_{m}", "", "")],
                         lambda s, rd, m=m: f"sixway_{m}_{'fav' if s.startswith(rd['fav']) else 'dog'}",
                         blend=True))
    # decision yes / no
    dec = six[:, 2] + six[:, 5]
    x = pd.DataFrame({"p_dec_yes": dec, "p_dec_no": 1 - dec,
                      "best_dec_yes": d.get("best_dec_yes"), "best_dec_no": d.get("best_dec_no"),
                      "med_dec_yes": d.get("med_dec_yes"), "med_dec_no": d.get("med_dec_no"),
                      "q_dec_yes": d.get("q_dec_yes"), "q_dec_no": d.get("q_dec_no"),
                      "won_dec_yes": (d["cls"] == "dec").to_numpy(), "won_dec_no": (d["cls"] != "dec").to_numpy()})
    out.append(_bets(x, [("dec_yes", "", ""), ("dec_no", "", "")],
                     lambda s: "decision_yes" if s == "dec_yes" else "decision_no", blend=True))
    # inside the distance, per fighter
    for corner, cols in (("red", [0, 1]), ("blue", [3, 4])):
        itd = six[:, cols].sum(axis=1)
        hit = ((d["winner"] == corner) & d["cls"].isin(["ko", "sub"])).to_numpy()
        x = pd.DataFrame({"p_y": itd, "p_n": 1 - itd,
                          "best_y": d.get(f"best_itd_{corner}_yes"), "best_n": d.get(f"best_itd_{corner}_no"),
                          "med_y": d.get(f"med_itd_{corner}_yes"), "med_n": d.get(f"med_itd_{corner}_no"),
                          "q_y": d.get(f"q_itd_{corner}_yes"), "q_n": d.get(f"q_itd_{corner}_no"),
                          "won_y": hit, "won_n": ~hit})
        out.append(_bets(x, [("y", "", ""), ("n", "", "")], lambda s: "itd_yes" if s == "y" else "itd_no",
                         blend=True))
    return pd.concat(out, ignore_index=True), d


def round_bets(engine, res: pd.DataFrame, px: pd.DataFrame, method_d: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "models/ufc/method/rounds_oof.csv"
    if not path.exists():
        log.warning("  no rounds OOF; skipping round families")
        return pd.DataFrame()
    S = pd.read_csv(path, dtype={"fight_id": str})
    d = S.merge(res, on="fight_id").merge(px, on="fight_id")
    # anchor to method_v2's decision probability, as served
    md = method_d.drop_duplicates("fight_id").set_index("fight_id")
    pr = d["fight_id"].map(md["p_red"])
    dec_p = pr * d["fight_id"].map(md["red_dec"]) + (1 - pr) * d["fight_id"].map(md["blue_dec"])
    curves = d[[f"S_{e:g}" for e in np.arange(0, 25.01, 2.5)]].to_numpy(float)
    A = anchor(curves, dec_p.to_numpy(float), d["sched"].to_numpy(float))
    t = d["t"].to_numpy(float)
    is_dec = (d["cls"] == "dec").to_numpy()
    out = []
    for name, line, k, over_key, under_key in (("ou_1_5", 7.5, 3, "ou_1.5_over", "ou_1.5_under"),
                                               ("ou_2_5", 12.5, 5, "ou_2.5_over", "ou_2.5_under"),
                                               ("starts_r2", 5.0, 2, "sr_2", "sr_2_no"),
                                               ("starts_r3", 10.0, 4, "sr_3", "sr_3_no")):
        alive = (t > line + 1e-9) | (is_dec & (t >= line - 1e-9))
        x = pd.DataFrame({"p_o": A[:, k], "p_u": 1 - A[:, k],
                          "best_o": d.get(f"best_{over_key}"), "best_u": d.get(f"best_{under_key}"),
                          "med_o": d.get(f"med_{over_key}"), "med_u": d.get(f"med_{under_key}"),
                          "q_o": d.get(f"q_{over_key}"), "q_u": d.get(f"q_{under_key}"),
                          "won_o": alive, "won_u": ~alive})
        yes, no = ("over", "under") if name.startswith("ou") else ("yes", "no")
        out.append(_bets(x, [("o", "", ""), ("u", "", "")],
                         lambda s, n=name, y=yes, nn=no: f"{n}_{y if s == 'o' else nn}", blend=True))
    return pd.concat(out, ignore_index=True)


def forward_record() -> dict:
    out = {}
    p = ROOT / "forward_log_v2.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        settled = [r for r in rows if r.get("bet") and r.get("profit") is not None]
        if settled:
            prof = np.array([r["profit"] for r in settled], float) / 100.0  # logged per $100 stake
            out["winner_open"] = {"n": len(settled), "roi": float(prof.mean())}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    engine = create_engine(settings.DATABASE_URL)
    res = _results(engine)
    px = prop_prices(engine)
    bets = [winner_bets(engine, res)]
    pb, method_d = prop_bets(engine, res, px)
    bets.append(pb)
    bets.append(round_bets(engine, res, px, method_d))
    allb = pd.concat([b for b in bets if len(b)], ignore_index=True)
    families = {}
    for fam, g in allb.groupby("family"):
        families[fam] = family_table(g["ev"], g["profit_best"], g["profit_median"], g["decimal"])
        f = families[fam]
        log.info(f"  {fam:14s} n={f['n']:5d} ROI {f['roi']:+6.1%} (exp {f['expected_roi']:+.1%}) | " + " ".join(
            f"[{b['lo']:.0%}+ n={b['n']} raw {'—' if b['roi'] is None else format(b['roi'], '+.0%')} "
            f"exp {b['expected_roi']:+.1%} {b['grade']}]" for b in f["buckets"]))
    table = save_table(families, forward_record(), {
        "ensemble_oof": "models/ufc/h2h/ensemble_oof.csv", "method_oof": "models/ufc/method/method_oof.csv",
        "rounds_oof": "models/ufc/method/rounds_oof.csv", "props": "ufc_prop_odds_history (bfo_close)",
        "moneyline": "ufc_fight_odds_open_close"})
    log.info(f"wrote grade table version {table['version']} ({len(families)} families)")


if __name__ == "__main__":
    main()
