"""Every bet the walk-forward backtest would have made, one row per bet.

Shared by scripts/build_grade_table.py (grades: per fight and market, only the highest-EV
side, `is_top`) and scripts/build_backtest_ledger.py (the Past Model Results page: every side
with EV > 0). Built from data in the repo + DB only:
  probabilities  ensemble OOF (models/ufc/h2h/ensemble_oof.csv) with the ensemble's market
                 stacks; method_v2 OOF (models/ufc/method/method_oof.csv); rounds OOF
                 (models/ufc/method/rounds_oof.csv, anchored to the market's decision price when priced,
                  else method_v2's decision prob)
  prices         moneyline open/close: ufc_fight_odds_open_close (opening = the consensus
                 price; closing = best across sportsbooks); props: ufc_prop_odds_history
                 rows with source 'bfo_close' (best and median American price) -- props have
                 no historical opening price, so every prop bet is at the close
  results        ufc_fights

Prop probabilities are the model blended 50/50 (log-odds) with the de-vigged closing
consensus, as the picks API serves them; moneyline probabilities are the ensemble's market
stack (opening stack against the opener, closing stack against the close).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.services.ufc.grading import blend_prob, decimal
from app.services.ufc.method_ratings import method_class
from app.services.ufc.round_hazard import anchor

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
LEDGER_PATH = ROOT / "models" / "ufc" / "backtest_ledger.csv.gz"
CELLS = ("red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec")
EXCHANGES = ("polymarket", "kalshi", "consensus", "mean")
#: Market group of each family prefix (the page's market filter).
GROUPS = {"winner": "winner", "sixway": "method", "decision": "distance", "itd": "itd",
          "ou": "rounds", "starts": "rounds"}


def group_of(family: str) -> str:
    return GROUPS[family.split("_")[0]]


def results(engine) -> pd.DataFrame:
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


def bets(df: pd.DataFrame, sides: list[tuple[str, str]], family_of, blend: bool = False,
         basis: str = "close", keep_all: bool = False) -> pd.DataFrame:
    """df has fight_id and, per side, p_<side>, best_<side>, med_<side>, won_<side> and
    q_<side> (the de-vigged market). sides = [(side, market key)]. The probability is
    blend_prob(p, q) when blend=True, as the picks API serves props. EV is at the TYPICAL
    book's price (median; best if no median). Every side with EV > 0 is a bet (`bet`);
    `is_top` marks the highest-EV one per fight (the grade table's rule). Flat 1 unit.
    keep_all=True also keeps the priced sides that are not bets, so a page can show the
    model's number on every market of a fight."""
    out = []
    code = family_of.__code__
    takes_row = "rd" in code.co_varnames[:code.co_argcount]
    for r in df.itertuples(index=False):
        rd = r._asdict()
        cands = []
        for side, market in sides:
            p_raw = rd.get(f"p_{side}")
            if p_raw is None or p_raw != p_raw:
                continue
            q = rd.get(f"q_{side}")
            p = blend_prob(p_raw, q) if blend else p_raw
            d_typ = decimal(rd.get(f"med_{side}")) or decimal(rd.get(f"best_{side}"))
            if d_typ is None:
                continue
            e = p * d_typ - 1
            if e > 0 or keep_all:
                cands.append((side, market, e, d_typ, float(p_raw), float(p),
                              None if q is None or q != q else float(q)))
        if not cands:
            continue
        top = max(cands, key=lambda c: c[2])
        for side, market, e, d_typ, p_raw, p, q in cands:
            won = bool(rd[f"won_{side}"])
            fam = family_of(side, rd) if takes_row else family_of(side)
            d_best = decimal(rd.get(f"best_{side}")) or d_typ
            out.append({"fight_id": rd["fight_id"], "family": fam, "market": market, "basis": basis,
                        "p_raw": p_raw, "p": p, "q": q, "ev": e, "decimal": d_typ, "decimal_best": d_best,
                        "won": won, "bet": e > 0, "is_top": e > 0 and side == top[0],
                        "profit_typical": d_typ - 1 if won else -1.0,
                        "profit_best": d_best - 1 if won else -1.0})
    return pd.DataFrame(out)


def _am_median(x):
    """Median book = median DECIMAL price, back to American (a median of American odds is
    meaningless across even money: -110 and +100 would give -5)."""
    d = [decimal(v) for v in x if decimal(v)]
    if not d:
        return np.nan
    m = float(np.median(d))
    return (m - 1) * 100 if m >= 2 else -100 / (m - 1)


def _norm(a, b) -> float:
    """De-vigged probability of side a from two American prices."""
    pa, pb = 1 / decimal(a), 1 / decimal(b)
    return pa / (pa + pb)


# ---------- inputs ----------
#
# The base inputs are the historical BFO backfills (ufc_fight_odds_open_close, bfo_close
# props) and the walk-forward OOF files: what the grade table is built from. extended=True
# (the Past Model Results ledger) carries every completed fight past them:
#   prices       the line watcher's snapshots (line_watcher.py, every two hours): a book's
#                first snapshot is its opening price, its last before the bout's cutoff
#                (prediction_markets.common.cutoffs_for_fights) its closing price
#   predictions  models/ufc/forward_predictions.csv (forward_predictions.py): live frozen
#                pre-fight predictions, or ones from a model trained before the fight
# A backfill/OOF row always wins over its fallback, except that a LIVE prediction (what the
# site actually served) wins over a walk-forward one.

def _cutoffs(engine, fight_ids) -> dict[str, object]:
    from sqlalchemy.orm import Session
    from app.services.ufc.prediction_markets.common import cutoffs_for_fights
    with Session(engine) as db:
        return {str(k): v for k, v in cutoffs_for_fights(db, [int(f) for f in fight_ids]).items()}


def _before_cutoff(df: pd.DataFrame, cut: dict, dates: dict) -> pd.DataFrame:
    """Rows captured before the bout's cutoff; with no cutoff, before the end of fight day."""
    lim = df["fight_id"].map(lambda f: cut.get(f) or (pd.Timestamp(dates[f]) + pd.Timedelta(days=1)))
    return df[df["captured_at"] < pd.to_datetime(lim)]


def moneyline(engine, extended: bool = False, need: set[str] | None = None):
    """(cons, best_close): per fight the consensus open/close American prices and de-vigged
    probabilities, and the best/median sportsbook close."""
    with engine.connect() as c:
        oc = pd.DataFrame(c.execute(text("""select fight_id::text fight_id, bookmaker, red_open, blue_open,
                                                   red_close, blue_close, red_open_prob, red_close_prob
                                            from ufc.ufc_fight_odds_open_close""")).all(),
                          columns=["fight_id", "book", "red_open", "blue_open", "red_close", "blue_close",
                                   "q_open", "q_close"])
    cons = oc[oc["book"] == "Consensus"].drop_duplicates("fight_id").set_index("fight_id")
    cons = cons.drop(columns="book").assign(price_source="bfo")
    books = oc[~oc["book"].str.lower().str.contains("|".join(EXCHANGES))]
    best_close = books.groupby("fight_id").agg(best_red=("red_close", "max"), best_blue=("blue_close", "max"),
                                               med_red=("red_close", _am_median), med_blue=("blue_close", _am_median))
    if not extended or not need:
        return cons, best_close
    missing = sorted(need - set(cons.index))
    if not missing:
        return cons, best_close
    with engine.connect() as c:
        h = pd.DataFrame(c.execute(text("""
            select h.fight_id::text, h.bookmaker, h.red_odds, h.blue_odds, h.captured_at, e.date
            from ufc.ufc_fight_odds_history h
            join ufc.ufc_fights f on f.id = h.fight_id join ufc.ufc_events e on e.id = f.event_id
            where h.bookmaker like 'BFO:%' and h.fight_id::text = any(:ids)
            order by h.captured_at"""), {"ids": missing}).all(),
            columns=["fight_id", "book", "red", "blue", "captured_at", "date"])
    h = h[~h["book"].str.lower().str.contains("|".join(EXCHANGES))]
    if h.empty:
        return cons, best_close
    dates = h.groupby("fight_id")["date"].first().to_dict()
    pre = _before_cutoff(h, _cutoffs(engine, h["fight_id"].unique()), dates)
    first = h.groupby(["fight_id", "book"]).first().reset_index()
    last = pre.groupby(["fight_id", "book"]).last().reset_index()
    o = first.groupby("fight_id").agg(red_open=("red", _am_median), blue_open=("blue", _am_median))
    cl = last.groupby("fight_id").agg(red_close=("red", _am_median), blue_close=("blue", _am_median),
                                      best_red=("red", "max"), best_blue=("blue", "max"))
    w = o.join(cl, how="inner").dropna()
    w["q_open"] = [_norm(a, b) for a, b in zip(w["red_open"], w["blue_open"])]
    w["q_close"] = [_norm(a, b) for a, b in zip(w["red_close"], w["blue_close"])]
    cons = pd.concat([cons, w[["red_open", "blue_open", "red_close", "blue_close", "q_open", "q_close"]]
                      .assign(price_source="watch")])
    best_close = pd.concat([best_close, w[["best_red", "best_blue"]].assign(
        med_red=w["red_close"], med_blue=w["blue_close"])])
    log.info(f"  moneyline: {len(w)} fights priced from line-watcher snapshots")
    return cons, best_close


def prop_prices(engine, extended: bool = False, need: set[str] | None = None) -> pd.DataFrame:
    with engine.connect() as c:
        df = pd.DataFrame(c.execute(text("""select fight_id::text fight_id, market, best_american, median_american, prob
                                            from ufc.ufc_prop_odds_history
                                            where source = 'bfo_close' and best_american is not null""")).all(),
                          columns=["fight_id", "market", "best", "med", "prob"])
    df["price_source"] = "bfo"
    missing = sorted((need or set()) - set(df["fight_id"])) if extended else []
    if missing:
        with engine.connect() as c:
            w = pd.DataFrame(c.execute(text("""
                select p.fight_id::text, p.market, p.best_american, p.median_american, p.prob, p.captured_at, e.date
                from ufc.ufc_prop_odds_history p
                join ufc.ufc_fights f on f.id = p.fight_id join ufc.ufc_events e on e.id = f.event_id
                where p.source = 'bfo_watch' and p.best_american is not null and p.fight_id::text = any(:ids)
                order by p.captured_at"""), {"ids": missing}).all(),
                columns=["fight_id", "market", "best", "med", "prob", "captured_at", "date"])
        if len(w):
            dates = w.groupby("fight_id")["date"].first().to_dict()
            w = _before_cutoff(w, _cutoffs(engine, w["fight_id"].unique()), dates)
            w = w.groupby(["fight_id", "market"]).last().reset_index()
            df = pd.concat([df, w[["fight_id", "market", "best", "med", "prob"]].assign(price_source="watch")])
            log.info(f"  props: {w['fight_id'].nunique()} fights priced from line-watcher snapshots")
    best = df.pivot_table(index="fight_id", columns="market", values="best", aggfunc="last").add_prefix("best_")
    med = df.pivot_table(index="fight_id", columns="market", values="med", aggfunc="last").add_prefix("med_")
    q = df.pivot_table(index="fight_id", columns="market", values="prob", aggfunc="last").add_prefix("q_")
    src = df.groupby("fight_id")["price_source"].last().rename("price_source")
    return best.join(med).join(q).join(src).reset_index()


def predictions(extended: bool = False, q_close: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """winner (fight_id, model_prob, odds_red_prob), method (conditional cells), rounds
    (S_ grid), each with `source`: backtest | live | forward."""
    from app.services.ufc import forward_predictions as fp
    out = {"winner": pd.read_csv(ROOT / "models/ufc/h2h/ensemble_oof.csv", dtype={"fight_id": str}),
           "method": pd.read_csv(ROOT / "models/ufc/method/method_oof.csv", dtype={"fight_id": str})}
    rp = ROOT / "models/ufc/method/rounds_oof.csv"
    out["rounds"] = pd.read_csv(rp, dtype={"fight_id": str}) if rp.exists() else pd.DataFrame(columns=["fight_id"])
    for k in out:
        out[k] = out[k].assign(source="backtest")
    if not extended:
        return out
    fwd = fp.read()
    cols = {"winner": ["model_prob"], "method": list(fp.CELLS), "rounds": fp.S_COLS}
    for k, base in out.items():
        f = fwd[fwd["model"] == k][["fight_id", "source", *cols[k]]]
        if f.empty:
            continue
        live = f[f["source"] == "live"]
        keep = base[~base["fight_id"].isin(live["fight_id"])]
        add = pd.concat([live, f[(f["source"] != "live") & ~f["fight_id"].isin(keep["fight_id"])]])
        if k == "winner" and q_close is not None:
            add = add.assign(odds_red_prob=add["fight_id"].map(q_close))
        out[k] = pd.concat([keep, add], ignore_index=True)
    return out


# ---------- bets ----------

def winner_bets(res: pd.DataFrame, ml, wpred: pd.DataFrame, keep_all: bool = False) -> pd.DataFrame:
    from app.services.ufc import ensemble
    ens = ensemble.load()
    cons, best_close = ml
    d = (wpred[["fight_id", "model_prob", "source"]].merge(res, on="fight_id")
         .join(cons, on="fight_id", how="inner").join(best_close, on="fight_id"))
    d = d[d["winner"].notna()]
    m = d["model_prob"].to_numpy(float)
    out = []
    for fam, q, br, bb, mr, mb in (
            ("winner_open", "q_open", "red_open", "blue_open", "red_open", "blue_open"),
            ("winner_close", "q_close", "best_red", "best_blue", "med_red", "med_blue")):
        p_red = ens.final_prob(m, d[q].to_numpy(float), opening=(fam == "winner_open")) if ens else m
        qr = d[q].to_numpy(float)
        x = pd.DataFrame({"fight_id": d["fight_id"].to_numpy(),
                          "p_red": p_red, "p_blue": 1 - p_red, "q_red": qr, "q_blue": 1 - qr,
                          "best_red": d[br].fillna(d["red_close"]).to_numpy(float) if fam == "winner_close" else d[br].to_numpy(float),
                          "best_blue": d[bb].fillna(d["blue_close"]).to_numpy(float) if fam == "winner_close" else d[bb].to_numpy(float),
                          "med_red": d[mr].to_numpy(float), "med_blue": d[mb].to_numpy(float),
                          "won_red": (d["winner"] == "red").to_numpy(), "won_blue": (d["winner"] == "blue").to_numpy()})
        b = bets(x, [("red", "ml_red"), ("blue", "ml_blue")], lambda s, f=fam: f,
                 basis="open" if fam == "winner_open" else "close", keep_all=keep_all)
        if len(b):
            # model-only probability, and closing line value of an opening bet: the de-vigged
            # close minus the de-vigged open on the side bet (> 0 = the line moved our way)
            mm = d.set_index("fight_id")
            red = b["market"] == "ml_red"
            b["p_raw"] = np.where(red, b["fight_id"].map(mm["model_prob"]), 1 - b["fight_id"].map(mm["model_prob"]))
            qo, qc = b["fight_id"].map(mm["q_open"]), b["fight_id"].map(mm["q_close"])
            b["clv"] = np.where(red, qc - qo, qo - qc) if fam == "winner_open" else np.nan
            b["source"] = b["fight_id"].map(mm["source"])
            b["price_source"] = b["fight_id"].map(mm["price_source"])
        out.append(b)
    return pd.concat(out, ignore_index=True)


def prop_bets(res: pd.DataFrame, px: pd.DataFrame, wpred: pd.DataFrame, mpred: pd.DataFrame,
              keep_all: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    from app.services.ufc import ensemble
    ens = ensemble.load()
    d = (mpred.merge(wpred[["fight_id", "model_prob", "odds_red_prob"]], on="fight_id")
         .merge(res, on="fight_id").merge(px, on="fight_id"))
    d = d[d["cls"].notna()]
    q = d["odds_red_prob"].to_numpy(float)
    p_red = ens.final_prob(d["model_prob"].to_numpy(float), q) if ens else d["model_prob"].to_numpy(float)
    d = d.assign(p_red=p_red)   # served-style winner prob (market blend), reused by round_bets
    six = np.column_stack([p_red * d[f"red_{c}"] for c in ("ko", "sub", "dec")] +
                          [(1 - p_red) * d[f"blue_{c}"] for c in ("ko", "sub", "dec")])
    fid = d["fight_id"].to_numpy()
    out = []
    # winner x method, graded per method: the market misprices them differently (across
    # 2,831 priced fights "favourite by decision" is de-vigged at 27.7% but happens 32.8%,
    # while "favourite by KO" is 23.9% vs 22.4%). Per fight and method, the better corner.
    x = pd.DataFrame({"fight_id": fid, **{f"p_{c}": six[:, j] for j, c in enumerate(CELLS)}})
    for c in CELLS:
        x[f"best_{c}"] = d.get(f"best_{c}"); x[f"med_{c}"] = d.get(f"med_{c}"); x[f"q_{c}"] = d.get(f"q_{c}")
        side, m = c.split("_")
        x[f"won_{c}"] = ((d["winner"] == side) & (d["cls"] == m)).to_numpy()
    # ...and separately for the favourite and the underdog: across 2,831 priced fights
    # "favourite by decision" is de-vigged at 27.7% but happens 32.8%, while "underdog by
    # decision" is priced about right (15.8% vs 16.4%).
    x["fav"] = np.where(p_red >= 0.5, "red", "blue")
    for m in ("ko", "sub", "dec"):
        out.append(bets(x, [(f"red_{m}", f"red_{m}"), (f"blue_{m}", f"blue_{m}")],
                        lambda s, rd, m=m: f"sixway_{m}_{'fav' if s.startswith(rd['fav']) else 'dog'}",
                        blend=True, keep_all=keep_all))
    # decision yes / no
    dec = six[:, 2] + six[:, 5]
    x = pd.DataFrame({"fight_id": fid, "p_dec_yes": dec, "p_dec_no": 1 - dec,
                      "best_dec_yes": d.get("best_dec_yes"), "best_dec_no": d.get("best_dec_no"),
                      "med_dec_yes": d.get("med_dec_yes"), "med_dec_no": d.get("med_dec_no"),
                      "q_dec_yes": d.get("q_dec_yes"), "q_dec_no": d.get("q_dec_no"),
                      "won_dec_yes": (d["cls"] == "dec").to_numpy(), "won_dec_no": (d["cls"] != "dec").to_numpy()})
    out.append(bets(x, [("dec_yes", "dec_yes"), ("dec_no", "dec_no")],
                    lambda s: "decision_yes" if s == "dec_yes" else "decision_no", blend=True, keep_all=keep_all))
    # inside the distance, per fighter
    for corner, cols in (("red", [0, 1]), ("blue", [3, 4])):
        itd = six[:, cols].sum(axis=1)
        hit = ((d["winner"] == corner) & d["cls"].isin(["ko", "sub"])).to_numpy()
        x = pd.DataFrame({"fight_id": fid, "p_y": itd, "p_n": 1 - itd,
                          "best_y": d.get(f"best_itd_{corner}_yes"), "best_n": d.get(f"best_itd_{corner}_no"),
                          "med_y": d.get(f"med_itd_{corner}_yes"), "med_n": d.get(f"med_itd_{corner}_no"),
                          "q_y": d.get(f"q_itd_{corner}_yes"), "q_n": d.get(f"q_itd_{corner}_no"),
                          "won_y": hit, "won_n": ~hit})
        out.append(bets(x, [("y", f"itd_{corner}_yes"), ("n", f"itd_{corner}_no")],
                        lambda s: "itd_yes" if s == "y" else "itd_no", blend=True, keep_all=keep_all))
    b = pd.concat(out, ignore_index=True)
    mm = d.drop_duplicates("fight_id").set_index("fight_id")
    b["source"] = b["fight_id"].map(mm["source"])
    b["price_source"] = b["fight_id"].map(mm["price_source"])
    return b, d


def round_bets(res: pd.DataFrame, px: pd.DataFrame, rpred: pd.DataFrame, method_d: pd.DataFrame,
               keep_all: bool = False) -> pd.DataFrame:
    if rpred.empty or "S_0" not in rpred:
        log.warning("  no rounds OOF; skipping round families")
        return pd.DataFrame()
    d = rpred.merge(res, on="fight_id").merge(px, on="fight_id")
    # anchor to method_v2's decision probability, as served
    md = method_d.drop_duplicates("fight_id").set_index("fight_id")
    pr = d["fight_id"].map(md["p_red"])
    dec_p = pr * d["fight_id"].map(md["red_dec"]) + (1 - pr) * d["fight_id"].map(md["blue_dec"])
    # as served (picks_v2._round_s_market): the market's P(decision) sets the finish mass when
    # priced, the rounds model only the timing
    if "q_dec_yes" in d:
        dec_p = d["q_dec_yes"].where(d["q_dec_yes"].notna(), dec_p)
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
        x = pd.DataFrame({"fight_id": d["fight_id"].to_numpy(), "p_o": A[:, k], "p_u": 1 - A[:, k],
                          "best_o": d.get(f"best_{over_key}"), "best_u": d.get(f"best_{under_key}"),
                          "med_o": d.get(f"med_{over_key}"), "med_u": d.get(f"med_{under_key}"),
                          "q_o": d.get(f"q_{over_key}"), "q_u": d.get(f"q_{under_key}"),
                          "won_o": alive, "won_u": ~alive})
        yes, no = ("over", "under") if name.startswith("ou") else ("yes", "no")
        out.append(bets(x, [("o", over_key), ("u", under_key)],
                        lambda s, n=name, y=yes, nn=no: f"{n}_{y if s == 'o' else nn}", blend=True, keep_all=keep_all))
    b = pd.concat(out, ignore_index=True)
    mm = d.drop_duplicates("fight_id").set_index("fight_id")
    b["source"] = b["fight_id"].map(mm["source"])
    b["price_source"] = b["fight_id"].map(mm["price_source"])
    return b


def build(engine, keep_all: bool = False, extended: bool = False) -> pd.DataFrame:
    """Every +EV bet across all families (keep_all: every priced side, `bet` marks the +EV
    ones); `is_top` = the grade table's one-per-market rule. extended: also every completed
    fight past the backfills / OOF files (line-watcher prices, forward predictions)."""
    res = results(engine)
    need = set(res.loc[res["cls"].notna() | res["winner"].notna(), "fight_id"]) if extended else None
    if extended:   # only fights some prediction covers are worth pricing
        from app.services.ufc import forward_predictions as fp
        need &= set(fp.read()["fight_id"]) | set(pd.read_csv(
            ROOT / "models/ufc/h2h/ensemble_oof.csv", usecols=["fight_id"], dtype={"fight_id": str})["fight_id"])
    ml = moneyline(engine, extended, need)
    px = prop_prices(engine, extended, need)
    preds = predictions(extended, ml[0]["q_close"])
    parts = [winner_bets(res, ml, preds["winner"], keep_all)]
    pb, method_d = prop_bets(res, px, preds["winner"], preds["method"], keep_all)
    parts.append(pb)
    parts.append(round_bets(res, px, preds["rounds"], method_d, keep_all))
    allb = pd.concat([b for b in parts if len(b)], ignore_index=True)
    allb["group"] = allb["family"].map(group_of)
    return allb
