"""Tapology-clone rank history, as of before each fight.

`ufc_ranking_history` has a row per fighter per division at every event date. The row
stamped on an event date INCLUDES that card's results (rank_history_backfill bounds at
as_of + 1 day), so a fight on date d may only read rows with as_of < d.

Features: rk_score (0-1000 within division), rk_pct (rank / division size; lower = better),
rk_sos (Tapology strength of schedule, 1-99), rk_delta_180 (score change over the previous
~6 months). The row is the fighter's standing in the fight's division if they have one,
otherwise their latest standing in any division.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.ufc.fighter_registry import classify_weight_class

PREFIX = "rk_"
FIGHT_LEVEL = ()
RAW = ()


def features_from(hist: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """hist: fighter_id, as_of, weight_class, rank, score, total_ranked, sos.
    fighters: fight_id, stats_fighter_id, date, weight_class."""
    h = hist.copy()
    h["as_of"] = pd.to_datetime(h["as_of"]).astype("datetime64[ns]")
    h["pct"] = h["rank"] / h["total_ranked"]
    h = h.sort_values("as_of")
    q = fighters[["fight_id", "stats_fighter_id", "date", "weight_class"]].copy()
    q["date"] = pd.to_datetime(q["date"]).astype("datetime64[ns]")
    q["division"] = q["weight_class"].map(classify_weight_class)
    q = q.rename(columns={"stats_fighter_id": "fighter_id"}).sort_values("date")
    q["fighter_id"] = q["fighter_id"].astype("int64")
    h["fighter_id"] = h["fighter_id"].astype("int64")
    cols = ["score", "pct", "sos"]

    def asof(left, right, by, shift_days=0):
        left = left.copy()
        left["_t"] = left["date"] - pd.Timedelta(days=shift_days)
        left = left.sort_values("_t")
        return pd.merge_asof(left, right.rename(columns={"as_of": "_t"}), on="_t", by=by,
                             allow_exact_matches=False, direction="backward")

    same = asof(q, h[["fighter_id", "weight_class", "as_of", *cols]].rename(columns={"weight_class": "division"}),
                ["fighter_id", "division"])
    anyd = asof(q, h[["fighter_id", "as_of", *cols]], "fighter_id")
    past = asof(q, h[["fighter_id", "as_of", "score"]].rename(columns={"score": "score_180"}),
                "fighter_id", shift_days=180)
    key = ["fight_id", "fighter_id"]
    out = same[key + cols].merge(anyd[key + cols], on=key, suffixes=("", "_any"))
    for c in cols:
        out[c] = out[c].fillna(out[f"{c}_any"])
    out = out.merge(past[key + ["score_180"]], on=key, how="left")
    res = pd.DataFrame({"fight_id": out["fight_id"], "fighter_id": out["fighter_id"],
                        "rk_score": out["score"], "rk_pct": out["pct"], "rk_sos": out["sos"],
                        "rk_delta_180": out["score"] - out["score_180"]})
    return res.replace([np.inf, -np.inf], np.nan)


def compute(ctx) -> pd.DataFrame:
    from sqlalchemy import text
    from app.models.ufc import UFC_SCHEMA
    schema = f"{UFC_SCHEMA}." if UFC_SCHEMA else ""
    db = ctx.session()
    try:
        hist = pd.read_sql(text(f"SELECT fighter_id, as_of, weight_class, rank, score, total_ranked, sos "
                                f"FROM {schema}ufc_ranking_history"), db.bind)
    finally:
        db.close()
    return features_from(hist, ctx.fighters)
