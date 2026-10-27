"""Past Model Results: profitability of every backtested pick, per event and across splits.

Reads models/ufc/backtest_ledger.csv.gz (scripts/build_backtest_ledger.py): one row per priced
side of every market on every walk-forward fight, `bet` = EV > 0 at the typical book's price.
Everything here is pure pandas over that ledger, so it is unit-testable without a database.

Conventions the page explains to readers:
  - flat 1 unit per bet; profit at the TYPICAL (median) book's price, or the best book's
  - basis 'open' bets the moneyline at the opening consensus line and 'close' at the close;
    props have no historical opening price, so prop bets are always at the close and are
    flagged `close_only` when the basis is 'open'
  - every side with EV > 0 is a bet, so several bets on one fight are correlated
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.services.ufc.backtest_ledger import LEDGER_PATH
from app.services.ufc.grading import SMALL_SAMPLE_N

GROUP_ORDER = ["winner", "method", "distance", "itd", "rounds"]
GROUP_LABELS = {"winner": "Moneyline", "method": "Method", "distance": "Goes the distance",
                "itd": "Inside the distance", "rounds": "Rounds"}
EV_BANDS = [0.0, 0.03, 0.05, 0.08, 0.12, 0.20, math.inf]
DISAGREE_BANDS = [0.0, 0.03, 0.06, 0.10, math.inf]   # |model-only p - market q|
PRICE_BANDS = [(1.0, 1.5, "-200 or shorter"), (1.5, 2.0, "-199 to -101"), (2.0, 3.0, "+100 to +199"),
               (3.0, 5.0, "+200 to +399"), (5.0, math.inf, "+400 or longer")]
N_BOOT = 1000

_ledger: tuple[float, pd.DataFrame] | None = None


def load_ledger() -> pd.DataFrame | None:
    """The ledger, re-read when the file changes (the post-event job commits a new one)."""
    global _ledger
    if not LEDGER_PATH.exists():
        return None
    mtime = LEDGER_PATH.stat().st_mtime
    if _ledger is None or _ledger[0] != mtime:
        df = pd.read_csv(LEDGER_PATH, dtype={"fight_id": str, "event_id": str})
        _ledger = (mtime, prepare(df))
    return _ledger[1]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns used by filters and splits."""
    df = df.copy()
    # ids are 64-bit; sent as strings like every other id in the API (JS numbers lose them)
    df["event_id"] = df["event_id"].astype(str)
    for c in ("won", "bet", "is_top"):
        df[c] = df[c].astype(str).str.lower().isin(["true", "1"])
    wc = df["weight_class"].fillna("")
    df["title"] = wc.str.contains("Title", case=False)
    df["gender"] = np.where(wc.str.contains("Women", case=False), "Women", "Men")
    df["division"] = wc.map(_division)
    pos = df["card_position"]
    df["section"] = np.select([pos == 0, pos.between(1, 4), pos >= 5], ["Main event", "Main card", "Prelims"],
                              "Unknown")
    df["year"] = df["date"].str[:4]
    # where each row's prediction and price came from (backtest_ledger.py, extended inputs)
    df["source"] = df["source"].fillna("backtest") if "source" in df else "backtest"
    df["price_source"] = df["price_source"].fillna("bfo") if "price_source" in df else "bfo"
    return df


def _division(wc: str) -> str:
    s = wc.replace("UFC ", "").replace("Interim ", "").replace(" Title", "").replace(" Bout", "")
    s = s.replace("Tournament", "").strip()
    for d in ("Light Heavyweight", "Heavyweight", "Middleweight", "Welterweight", "Lightweight",
              "Featherweight", "Bantamweight", "Flyweight", "Strawweight", "Catch Weight"):
        if d in s:
            return ("Women's " if "Women" in s else "") + d
    return s or "Unknown"


@dataclass
class Filters:
    start: str | None = None          # inclusive ISO dates
    end: str | None = None
    basis: str = "close"              # 'open' | 'close' (moneyline price; props are close-only)
    price: str = "typical"            # 'typical' | 'best'
    min_ev: float = 0.0
    groups: list[str] = field(default_factory=lambda: list(GROUP_ORDER))
    props_at_open: bool = True        # basis 'open': include close-only prop bets
    top_only: bool = False            # one bet per fight and market (the grade table's rule)

    def key(self) -> str:
        return (f"{self.start}:{self.end}:{self.basis}:{self.price}:{self.min_ev:.4f}:"
                f"{','.join(sorted(self.groups))}:{int(self.props_at_open)}:{int(self.top_only)}")


def priced(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Every priced side in scope (bets or not): date range, basis, market groups."""
    m = pd.Series(True, index=df.index)
    if f.start:
        m &= df["date"] >= f.start
    if f.end:
        m &= df["date"] <= f.end
    is_ml = df["group"] == "winner"
    m &= np.where(is_ml, df["basis"] == f.basis, True)
    if f.basis == "open" and not f.props_at_open:
        m &= is_ml
    m &= df["group"].isin(f.groups)
    out = df[m].copy()
    out["profit"] = out["profit_best"] if f.price == "best" else out["profit_typical"]
    out["dec"] = out["decimal_best"] if f.price == "best" else out["decimal"]
    out["close_only"] = (f.basis == "open") & (out["group"] != "winner")
    return out


def bets_of(rows: pd.DataFrame, f: Filters) -> pd.DataFrame:
    b = rows[rows["bet"] & (rows["ev"] >= f.min_ev)]
    return b[b["is_top"]] if f.top_only else b


# ---------- numbers ----------

def american(dec: float | None) -> int | None:
    if dec is None or dec != dec or dec <= 1:
        return None
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def _f(x) -> float | None:
    return None if x is None or x != x else round(float(x), 5)


def totals(b: pd.DataFrame, ci: bool = False, seed: int = 0) -> dict:
    n = len(b)
    if n == 0:
        return {"n": 0, "wins": 0, "profit": 0.0, "roi": None, "hit": None, "avg_ev": None,
                "avg_price": None, "ci": None, "small_sample": True}
    prof = b["profit"].to_numpy(float)
    out = {"n": n, "wins": int(b["won"].sum()), "profit": round(float(prof.sum()), 3),
           "roi": _f(prof.mean()), "hit": _f(b["won"].mean()), "avg_ev": _f(b["ev"].mean()),
           "avg_price": american(float(np.median(b["dec"]))), "small_sample": n < SMALL_SAMPLE_N, "ci": None}
    if ci and n >= 2:
        rng = np.random.default_rng(seed)
        boot = prof[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
        out["ci"] = [_f(np.percentile(boot, 2.5)), _f(np.percentile(boot, 97.5))]
    return out


# ---------- labels ----------

def family_label(fam: str) -> str:
    """'sixway_dec_fav' -> 'Favourite by decision', 'ou_1_5_under' -> 'Under 1.5 rounds'."""
    p = fam.split("_")
    if p[0] == "winner":
        return f"Moneyline ({p[1]})"
    if p[0] == "sixway":
        how = {"ko": "KO/TKO", "sub": "submission", "dec": "decision"}[p[1]]
        return f"{'Favourite' if p[2] == 'fav' else 'Underdog'} by {how}"
    if p[0] == "decision":
        return "Goes to decision" if p[1] == "yes" else "Doesn't go to decision"
    if p[0] == "itd":
        return "Wins inside distance" if p[1] == "yes" else "Doesn't win inside distance"
    if p[0] == "ou":
        return f"{p[3].capitalize()} {p[1]}.{p[2]} rounds"
    if p[0] == "starts":
        return f"Starts round {p[1][1:]}" + ("" if p[2] == "yes" else ": no")
    return fam


def market_name(market: str, red: str, blue: str) -> str:
    if market.startswith("ml_"):
        return "Moneyline"
    if market.split("_")[0] in ("red", "blue"):
        return "Method of victory"
    if market.startswith("dec_"):
        return "Goes the distance"
    if market.startswith("itd_"):
        return f"{red if '_red_' in market else blue} inside the distance"
    if market.startswith("ou_"):
        return f"Total rounds {market.split('_')[1]}"
    if market.startswith("sr_"):
        return f"Fight starts round {market.split('_')[1]}"
    return market


def side_label(market: str, red: str, blue: str) -> str:
    who = lambda c: red if c == "red" else blue  # noqa: E731
    parts = market.split("_")
    if market.startswith("ml_"):
        return who(parts[1])
    if parts[0] in ("red", "blue"):
        return f"{who(parts[0])} by " + {"ko": "KO/TKO", "sub": "submission", "dec": "decision"}[parts[1]]
    if market == "dec_yes":
        return "Goes to decision"
    if market == "dec_no":
        return "Doesn't go to decision"
    if market.startswith("itd_"):
        return f"{who(parts[1])} wins inside distance" + ("" if parts[2] == "yes" else ": no")
    if market.startswith("ou_"):
        return f"{parts[2].capitalize()} {parts[1]} rounds"
    if market.startswith("sr_"):
        return f"Starts round {parts[1]}" + (": no" if market.endswith("_no") else "")
    return market


def result_text(r) -> str:
    if not isinstance(r.get("winner_side"), str):
        return r.get("result_method") or "No result"
    who = r["red_name"] if r["winner_side"] == "red" else r["blue_name"]
    how = {"ko": "KO/TKO", "sub": "Submission", "dec": "Decision"}.get(r.get("result_cls"), r.get("result_method"))
    rnd = r.get("finish_round")
    when = f", R{int(rnd)} {r.get('finish_time') or ''}".rstrip() if how != "Decision" and rnd == rnd and rnd else ""
    return f"{who} by {how}{when}"


# ---------- views ----------

def coverage(df: pd.DataFrame) -> dict:
    return {"from": df["date"].min(), "to": df["date"].max(), "events": int(df["event_id"].nunique()),
            "fights": int(df["fight_id"].nunique())}


def events_summary(df: pd.DataFrame, f: Filters) -> dict:
    rows = priced(df, f)
    b = bets_of(rows, f)
    by_group = {g: totals(b[b["group"] == g]) for g in GROUP_ORDER if g in f.groups}
    events = []
    for eid, g in rows.groupby("event_id", sort=False):
        eb = b[b["event_id"] == eid]
        t = totals(eb)
        events.append({"event_id": str(eid), "name": g["event_name"].iat[0], "date": g["date"].iat[0],
                       "fights": int(g["fight_id"].nunique()), **t,
                       "sources": sorted(set(g["source"])), "watch_prices": bool((g["price_source"] == "watch").any()),
                       "by_group": {k: round(float(v), 3) for k, v in eb.groupby("group")["profit"].sum().items()}})
    events.sort(key=lambda e: (e["date"], e["event_id"]))
    cum, series = 0.0, []
    for e in events:
        cum += e["profit"]
        series.append({"date": e["date"], "event_id": e["event_id"], "name": e["name"],
                       "profit": e["profit"], "cumulative": round(cum, 3)})
    return {"filters": vars(f), "coverage": coverage(df), "totals": totals(b, ci=True),
            "by_group": by_group, "events": events[::-1], "cumulative": series,
            "props_close_only": f.basis == "open" and f.props_at_open}


def event_detail(df: pd.DataFrame, event_id: str, f: Filters) -> dict | None:
    ev_rows = df[df["event_id"] == str(event_id)]
    if ev_rows.empty:
        return None
    f2 = Filters(**{**vars(f), "start": None, "end": None})
    rows = priced(ev_rows, f2)
    b = bets_of(rows, f2)
    bet_idx = set(b.index)
    fights = []
    order = rows.drop_duplicates("fight_id").sort_values("card_position", na_position="last")
    for _, fr in order.iterrows():
        fid = fr["fight_id"]
        fx = rows[rows["fight_id"] == fid]
        red, blue = fr["red_name"], fr["blue_name"]
        markets: dict[str, dict] = {}
        for idx, r in fx.iterrows():
            name = market_name(r["market"], red, blue)
            mk = markets.setdefault(name, {"market": name, "group": r["group"], "sides": []})
            mk["sides"].append({
                "label": side_label(r["market"], red, blue), "market_key": r["market"], "family": r["family"],
                "price": american(r["dec"]), "p": _f(r["p"]), "p_model": _f(r["p_raw"]), "q": _f(r["q"]),
                "ev": _f(r["ev"]), "bet": idx in bet_idx, "won": bool(r["won"]),
                "profit": _f(r["profit"]) if idx in bet_idx else None, "close_only": bool(r["close_only"]),
                "clv": _f(r.get("clv")) if r["group"] == "winner" else None})
        for mk in markets.values():
            mk["sides"].sort(key=lambda s: -(s["ev"] or -9))
        fb = b[b["fight_id"] == fid]
        fights.append({"fight_id": fid, "red": red, "blue": blue, "weight_class": fr["weight_class"],
                       "card_position": None if fr["card_position"] != fr["card_position"] else int(fr["card_position"]),
                       "section": fr["section"], "rounds": int(fr["rounds"]), "winner_side": fr["winner_side"]
                       if isinstance(fr["winner_side"], str) else None,
                       "result": result_text(fr), **{k: v for k, v in totals(fb).items() if k in ("n", "profit", "roi")},
                       "sources": sorted(set(fx["source"])),
                       "markets": sorted(markets.values(), key=lambda m: GROUP_ORDER.index(m["group"]))})
    return {"event_id": str(event_id), "name": ev_rows["event_name"].iat[0], "date": ev_rows["date"].iat[0],
            "sources": sorted(set(rows["source"])), "watch_prices": bool((rows["price_source"] == "watch").any()),
            "filters": vars(f2), "totals": totals(b), "by_group": {g: totals(b[b["group"] == g])
                                                                   for g in GROUP_ORDER if g in f.groups},
            "fights": fights, "props_close_only": f.basis == "open" and f.props_at_open}


def _band(x: pd.Series, edges: list[float], fmt) -> pd.Series:
    labels = [fmt(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    return pd.cut(x, edges, right=False, labels=labels).astype(str)


def _pct(lo, hi):
    return f"{lo:.0%}+" if hi == math.inf else f"{lo:.0%}–{hi:.0%}"


def _pts(lo, hi):
    if lo < 0:
        return "No market price"
    return f"{lo * 100:.0f}+ pts" if hi == math.inf else f"{lo * 100:.0f}–{hi * 100:.0f} pts"


def split_columns(b: pd.DataFrame) -> dict[str, pd.Series]:
    price = pd.Series("Unknown", index=b.index)
    for lo, hi, lab in PRICE_BANDS:
        price[(b["dec"] >= lo) & (b["dec"] < hi)] = lab
    return {
        "ev_band": _band(b["ev"], EV_BANDS, _pct),
        "market": b["group"].map(GROUP_LABELS),
        "family": b["family"].map(family_label),
        "price": price,
        "disagreement": _band((b["p_raw"] - b["q"]).abs().fillna(-1), [-1.0] + DISAGREE_BANDS, _pts),
        "section": b["section"],
        "division": b["division"],
        "gender": b["gender"],
        "rounds": b["rounds"].map(lambda r: f"{int(r)} rounds"),
        "title": np.where(b["title"], "Title fight", "Non-title").astype(str),
        "year": b["year"],
        "source": b["source"].map(SOURCE_LABELS).fillna(b["source"]),
    }


SOURCE_LABELS = {"backtest": "Walk-forward backtest", "forward": "Model trained before the card",
                 "live": "Live: served before the fight"}
SPLIT_LABELS = {"source": "Prediction source", "ev_band": "Edge (EV) band", "market": "Market", "family": "Market family",
                "price": "Price range", "disagreement": "Model vs market gap", "section": "Card section",
                "division": "Division", "gender": "Gender", "rounds": "Scheduled rounds",
                "title": "Title fights", "year": "Year"}
#: splits whose rows have a natural order (others sort by number of bets)
_ORDERED = {"ev_band": [_pct(lo, hi) for lo, hi in zip(EV_BANDS[:-1], EV_BANDS[1:])],
            "price": [lab for *_, lab in PRICE_BANDS], "market": [GROUP_LABELS[g] for g in GROUP_ORDER],
            "section": ["Main event", "Main card", "Prelims", "Unknown"],
            "source": list(SOURCE_LABELS.values()),
            "disagreement": [_pts(lo, hi) for lo, hi in zip(DISAGREE_BANDS[:-1], DISAGREE_BANDS[1:])]
            + ["No market price"]}


def splits(df: pd.DataFrame, f: Filters) -> dict:
    rows = priced(df, f)
    b = bets_of(rows, f)
    out = {}
    cols = split_columns(b) if len(b) else {}
    for name, label in SPLIT_LABELS.items():
        if not len(b):
            out[name] = {"label": label, "rows": []}
            continue
        key = pd.Series(cols[name], index=b.index)
        rws = [{"key": str(k), **totals(g, ci=True, seed=zlib.crc32(str(k).encode()))} for k, g in b.groupby(key)]
        if name in _ORDERED:
            rws.sort(key=lambda r: _ORDERED[name].index(r["key"]) if r["key"] in _ORDERED[name] else 99)
        elif name == "year":
            rws.sort(key=lambda r: r["key"])
        else:
            rws.sort(key=lambda r: -r["n"])
        out[name] = {"label": label, "rows": rws}
    return {"filters": vars(f), "coverage": coverage(df), "totals": totals(b, ci=True), "splits": out,
            "calibration": calibration(rows), "accuracy": accuracy(rows), "clv": clv(b),
            "cumulative": cumulative_by_group(b), "props_close_only": f.basis == "open" and f.props_at_open}


def calibration(rows: pd.DataFrame, bins: int = 10) -> list[dict]:
    """Predicted probability vs how often it happened, over every priced side in scope
    (bets or not): the model alone (before any market blend), the served probability (the
    blend the bets use) and the de-vigged market."""
    r = rows[rows["q"].notna()]
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi >= 1
        row = {"lo": round(lo, 2), "hi": round(hi, 2)}
        for name, col in (("model", "p_raw"), ("served", "p"), ("market", "q")):
            m = r[(r[col] >= lo) & ((r[col] <= hi) if last else (r[col] < hi))]
            row[name] = {"n": len(m), "pred": _f(m[col].mean()), "actual": _f(m["won"].mean())}
        out.append(row)
    return out


def accuracy(rows: pd.DataFrame) -> list[dict]:
    """Brier score of the model alone, the served blend and the de-vigged market, per market
    group, on the same priced sides (lower is better; negative diff = the served probability
    was more accurate than the market)."""
    out = []
    r = rows[rows["q"].notna()]
    for g in GROUP_ORDER:
        x = r[r["group"] == g]
        if not len(x):
            continue
        y = x["won"].astype(float)
        br = float(((x["p_raw"] - y) ** 2).mean())
        bm, bq = float(((x["p"] - y) ** 2).mean()), float(((x["q"] - y) ** 2).mean())
        out.append({"group": g, "label": GROUP_LABELS[g], "n": len(x), "brier_model_only": _f(br),
                    "brier_model": _f(bm), "brier_market": _f(bq), "diff": _f(bm - bq)})
    return out


def clv(b: pd.DataFrame) -> dict | None:
    """Closing line value of opening moneyline bets: de-vigged close minus open on the side bet."""
    x = b[(b["group"] == "winner") & (b["basis"] == "open") & b["clv"].notna()] if "clv" in b else b.iloc[0:0]
    if not len(x):
        return None
    pos = x[x["clv"] > 0]
    neg = x[x["clv"] <= 0]
    return {"n": len(x), "avg_clv": _f(x["clv"].mean()), "beat_close": _f((x["clv"] > 0).mean()),
            "roi_beat": totals(pos)["roi"], "roi_not": totals(neg)["roi"], "n_beat": len(pos), "n_not": len(neg)}


def cumulative_by_group(b: pd.DataFrame) -> list[dict]:
    if not len(b):
        return []
    p = b.groupby(["date", "event_id", "group"])["profit"].sum().unstack("group").fillna(0.0).sort_index()
    c = p.cumsum()
    out = []
    for (d, eid), r in c.iterrows():
        out.append({"date": d, "event_id": str(eid), **{g: round(float(r[g]), 3) for g in c.columns},
                    "all": round(float(r.sum()), 3)})
    return out
