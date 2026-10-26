"""Deserve-to-win, summed over a fighter's EARLIER fights.

`ufc_deserve_to_win` describes each fight after the fact (share of 10,000 replays each corner
deserved to win), so the fight's own row is the label and is never read. Over prior fights:

  dtw_luck        sum of (won - deserved): results that ran ahead of performance, which should
                  even out (positive = has been lucky)
  dtw_mean        mean deserved share, shrunk toward 0.5 with K pseudo-fights
  dtw_robbed      prior fights lost while deserving >= 0.70
  dtw_gifted      prior fights won while deserving <= 0.30

Caveat (first pass): the stored model's round fits and judge variances use every year,
including later ones, so this is mildly leaky. Only if it clears the gate is the
quarterly-refit version worth building.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

PREFIX = "dtw_"
FIGHT_LEVEL = ()
RAW = ()
K = 3.0


def features_from(dtw: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """dtw: fight_id, p_red, p_blue. fighters: fight_id, stats_fighter_id, corner, date, won, winner_id."""
    p = dtw.drop_duplicates("fight_id").set_index("fight_id")
    fr = fighters.copy()
    fr["date"] = pd.to_datetime(fr["date"])
    fr = fr.sort_values(["date", "fight_id"])
    fr["deserved"] = np.where(fr["corner"] == "red", fr["fight_id"].map(p["p_red"]), fr["fight_id"].map(p["p_blue"]))
    state = defaultdict(lambda: [0.0, 0.0, 0, 0, 0])     # luck, deserved sum, n, robbed, gifted
    out = []
    # snapshot every fight of a date before folding any of that date's results in
    for _, day in fr.groupby("date", sort=True):
        for r in day.itertuples(index=False):
            s = state[r.stats_fighter_id]
            row = {"fight_id": r.fight_id, "fighter_id": r.stats_fighter_id}
            if s[2]:
                row.update(dtw_luck=s[0], dtw_mean=(s[1] + 0.5 * K) / (s[2] + K), dtw_robbed=float(s[3]),
                           dtw_gifted=float(s[4]))
            out.append(row)
        for r in day.itertuples(index=False):
            if pd.isna(r.deserved) or pd.isna(r.winner_id):
                continue
            s = state[r.stats_fighter_id]
            won = float(r.won)
            s[0] += won - r.deserved
            s[1] += r.deserved
            s[2] += 1
            s[3] += int(won == 0 and r.deserved >= 0.70)
            s[4] += int(won == 1 and r.deserved <= 0.30)
    return pd.DataFrame(out)


def compute(ctx) -> pd.DataFrame:
    from sqlalchemy import text
    from app.models.ufc import UFC_SCHEMA
    schema = f"{UFC_SCHEMA}." if UFC_SCHEMA else ""
    db = ctx.session()
    try:
        dtw = pd.read_sql(text(f"SELECT fight_id, p_red, p_blue FROM {schema}ufc_deserve_to_win"), db.bind)
    finally:
        db.close()
    return features_from(dtw, ctx.fighters)
