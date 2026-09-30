"""Whole-career rating from every pro bout on Sherdog (all promotions), point-in-time.

A UFC debutant has no UFC history, so every UFC-derived feature is empty or a prior. But
most debutants have 10-30 pro fights across regional promotions. This runs one Elo over
all of those bouts (UFC included, so the scale is shared) and reads each fighter's rating
strictly before the date of each UFC fight. A debutant who beat well-rated regional
opponents arrives with a real rating; one who padded a record against 0-3 opponents
does not.

It also gives point-in-time pro record counts, replacing pre_ufc_* (lifetime record
minus UFC results), which used today's record and so leaked future non-UFC results.

Features (per fighter, before the fight):
  pro_elo            rating on the shared all-promotions scale
  pro_fights         pro bouts before this date
  pro_win_pct        shrunk win share, (W + 2) / (n + 4)
  pro_opp_elo        mean pre-bout rating of opponents faced (strength of schedule)
  pro_nonufc_fights  bouts outside the UFC
  pro_known          1 if the fighter is linked to a Sherdog profile, else 0
"""
from __future__ import annotations

import logging
import zlib
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import pandas as pd

log = logging.getLogger("career_rating")

K = 48.0  # tuned 2015-21 on next-UFC-fight log loss (16..96 grid); differences are small
INIT = 1500.0
FEATURES = ("pro_elo", "pro_fights", "pro_win_pct", "pro_opp_elo", "pro_nonufc_fights",
            "pro_known")


def load_bouts(db) -> tuple[pd.DataFrame, dict[int, int]]:
    """Canonical pro bouts (one row per bout) and the ufc_fighter_id -> sherdog_id map."""
    from app.models.ufc import SherdogBout, SherdogFighter

    link = {u: s for s, u in db.query(SherdogFighter.sherdog_id, SherdogFighter.ufc_fighter_id)
            if u is not None}
    rows = db.query(SherdogBout.fighter_sherdog_id, SherdogBout.opponent_sherdog_id,
                    SherdogBout.opponent_key, SherdogBout.date, SherdogBout.result,
                    SherdogBout.promotion).all()
    b = pd.DataFrame(rows, columns=["a", "b", "b_key", "date", "result", "promotion"])
    b = b[b["date"].notna() & b["result"].isin(["W", "L", "D"])]
    # Opponents without a Sherdog id get a synthetic negative id from their name key so
    # they still carry (and receive) a rating.
    synthetic = np.array([-(zlib.crc32(str(k).encode()) % 10**9) - 1 for k in b["b_key"]],
                         dtype=np.int64)
    b["b"] = np.where(b["b"].isna(), synthetic, b["b"].fillna(0)).astype(np.int64)
    b["lo"] = np.minimum(b["a"], b["b"])
    b["hi"] = np.maximum(b["a"], b["b"])
    b = b.drop_duplicates(["lo", "hi", "date"]).sort_values(["date", "lo", "hi"])
    score = b["result"].map({"W": 1.0, "L": 0.0, "D": 0.5})
    b["a_score"] = score.values
    return b[["a", "b", "date", "a_score", "promotion"]].reset_index(drop=True), link


def run(bouts: pd.DataFrame, k: float = None) -> dict[int, dict[str, list]]:
    """Per sherdog id: parallel lists of bout dates and the state AFTER each bout."""
    k = K if k is None else k
    rating: dict[int, float] = defaultdict(lambda: INIT)
    tally: dict[int, list] = defaultdict(lambda: [0, 0, 0.0, 0])  # n, wins, sum opp, nonufc
    hist: dict[int, dict[str, list]] = defaultdict(lambda: {"date": [], "state": []})
    for a, b, d, s, promo in bouts.itertuples(index=False):
        ra, rb = rating[a], rating[b]
        e = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        rating[a] = ra + k * (s - e)
        rating[b] = rb - k * (s - e)
        nonufc = 0 if promo == "UFC" else 1
        for me, opp_r, won in ((a, rb, s), (b, ra, 1 - s)):
            t = tally[me]
            t[0] += 1; t[1] += won; t[2] += opp_r; t[3] += nonufc
            hist[me]["date"].append(d)
            hist[me]["state"].append((rating[me], t[0], t[1], t[2], t[3]))
    return hist


def features_for(rows: pd.DataFrame, hist, link: dict[int, int]) -> pd.DataFrame:
    """rows: stats_fighter_id, date. Returns FEATURES aligned to rows (before `date`)."""
    out = np.full((len(rows), len(FEATURES)), np.nan)
    for i, (fid, d) in enumerate(zip(rows["stats_fighter_id"], rows["date"])):
        sid = link.get(fid)
        if sid is None:
            out[i, 5] = 0.0
            continue
        out[i, 5] = 1.0
        h = hist.get(sid)
        k = bisect_left(h["date"], d) if h else 0  # bouts strictly before d
        if k == 0:
            out[i, :5] = (INIT, 0.0, 0.5, np.nan, 0.0)
            continue
        r, n, w, opp_sum, nonufc = h["state"][k - 1]
        out[i, :5] = (r, n, (w + 2) / (n + 4), opp_sum / n, nonufc)
    return pd.DataFrame(out, columns=FEATURES, index=rows.index)


def compute(df: pd.DataFrame) -> pd.DataFrame | None:
    """Career features for a load_fight_data frame, or None if no Sherdog data exists."""
    from sqlalchemy import inspect

    from app.database import SessionLocal, engine
    from app.models.ufc import SherdogBout

    tbl = SherdogBout.__table__
    if not inspect(engine).has_table(tbl.name, schema=tbl.schema):
        return None
    db = SessionLocal()
    try:
        bouts, link = load_bouts(db)
    finally:
        db.close()
    if bouts.empty:
        return None
    hist = run(bouts)
    rows = df[["stats_fighter_id", "date"]].copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.date
    feats = features_for(rows, hist, link)
    log.info(f"  Career ratings: {len(bouts)} pro bouts, {len(link)} linked UFC fighters, "
             f"{int(feats['pro_known'].sum())}/{len(feats)} rows covered")
    return feats
