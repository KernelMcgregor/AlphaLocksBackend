"""Closing line value of the model's moneyline view, over time (GET /ufc/model/clv).

Per fight in the backtest ledger (backtest_ledger.py): the model's own probability (`p_raw`, before
any market blend) against the de-vigged opening line picks a side -- the corner the model rates
above the market. CLV is how far the de-vigged close moved toward that side. A model that sees
something real is one the market keeps moving toward after the open, whatever the results were;
it converges in hundreds of fights where ROI needs thousands.

The opening line stands in for "the line when the model published": the model prices a bout as
soon as it is on a card, which is when books open it, and the ledger's open is the first
observed price. Live rows (`source == 'live'`) are the predictions the site actually served,
frozen at the cutoff (forward_predictions.py).

Pure pandas over the ledger frame, so unit-testable without a database.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EDGE_BANDS = [0.0, 0.03, 0.06, 0.10, math.inf]
#: A close within this of the open has not moved either way.
FLAT = 0.005
RECENT = 60


def _f(x, d: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)


def fight_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fight: the model's side against the open, and where the close went."""
    w = df[df["group"] == "winner"]
    if not len(w):
        return pd.DataFrame()
    keep = ["fight_id", "market", "p_raw", "q", "won", "source", "date", "event_id", "event_name",
            "red_name", "blue_name"]
    o = w[w["basis"] == "open"][keep]
    c = w[w["basis"] == "close"][["fight_id", "market", "q"]].rename(columns={"q": "q_close"})
    x = o.rename(columns={"q": "q_open", "p_raw": "p"}).merge(c, on=["fight_id", "market"])
    x = x.dropna(subset=["p", "q_open", "q_close"])
    x["edge"] = x["p"] - x["q_open"]
    # the side the model rates above the open; ties carry no view
    x = x[x["edge"] > 0].sort_values("edge", ascending=False).drop_duplicates("fight_id")
    x["clv"] = x["q_close"] - x["q_open"]
    x["toward"] = x["clv"] > FLAT
    x["against"] = x["clv"] < -FLAT
    x["side"] = np.where(x["market"] == "ml_red", "red", "blue")
    return x.sort_values(["date", "fight_id"]).reset_index(drop=True)


def _summary(x: pd.DataFrame) -> dict:
    n = len(x)
    if not n:
        return {"n": 0}
    return {"n": n, "avg_edge": _f(x["edge"].mean()), "avg_clv": _f(x["clv"].mean()),
            "toward": _f(x["toward"].mean()), "against": _f(x["against"].mean()),
            "win_rate": _f(x["won"].mean()), "exp_win_rate": _f(x["q_close"].mean())}


def _band_label(lo: float, hi: float) -> str:
    return f"{lo * 100:.0f}%+" if hi == math.inf else f"{lo * 100:.0f}–{hi * 100:.0f}%"


def by_edge(x: pd.DataFrame) -> list[dict]:
    out = []
    for lo, hi in zip(EDGE_BANDS[:-1], EDGE_BANDS[1:]):
        b = x[(x["edge"] >= lo) & (x["edge"] < hi)]
        out.append({"band": _band_label(lo, hi), "lo": lo, **_summary(b)})
    return out


def timeline(x: pd.DataFrame) -> list[dict]:
    """Per month: fights, average CLV, share the close moved toward the model; plus the running
    average CLV, which is the line that says whether the model has kept beating the close."""
    if not len(x):
        return []
    m = x.assign(month=x["date"].str[:7])
    out, tot, n = [], 0.0, 0
    for month, g in m.groupby("month", sort=True):
        tot += float(g["clv"].sum())
        n += len(g)
        out.append({"month": month, "n": len(g), "avg_clv": _f(g["clv"].mean()),
                    "toward": _f(g["toward"].mean()), "cum_avg_clv": _f(tot / n)})
    return out


def recent(x: pd.DataFrame, k: int = RECENT) -> list[dict]:
    out = []
    for r in x.tail(k).iloc[::-1].itertuples():
        out.append({"fight_id": str(r.fight_id), "event_id": str(r.event_id), "event_name": r.event_name,
                    "date": r.date, "side": r.side, "red_name": r.red_name, "blue_name": r.blue_name,
                    "pick_name": r.red_name if r.side == "red" else r.blue_name,
                    "p": _f(r.p), "q_open": _f(r.q_open), "q_close": _f(r.q_close), "edge": _f(r.edge),
                    "clv": _f(r.clv), "won": bool(r.won), "source": r.source})
    return out


def clv_report(df: pd.DataFrame, source: str = "all", start: str | None = None, end: str | None = None,
               min_edge: float = 0.0) -> dict:
    x = fight_rows(df)
    if len(x):
        if source == "live":
            x = x[x["source"] != "backtest"]
        if start:
            x = x[x["date"] >= start]
        if end:
            x = x[x["date"] <= end]
        if min_edge > 0:
            x = x[x["edge"] >= min_edge]
    return {
        "source": source,
        "coverage": {"from": x["date"].min() if len(x) else None, "to": x["date"].max() if len(x) else None},
        "summary": _summary(x),
        "by_edge": by_edge(x) if len(x) else [],
        "timeline": timeline(x),
        "recent": recent(x) if len(x) else [],
    }
