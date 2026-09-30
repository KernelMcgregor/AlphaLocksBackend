"""Matchmaking signals: where the UFC puts a fight on the card, and a fighter's trajectory.

Card placement is the promotion's own read on a fight. Main events and co-mains go to the
fighters matchmakers rate and want to build; a fighter bumped from the prelims to the main
card has been judged ready. Everything here is known before the fight (the card order is
published with the event).

Fight-level (same value for both corners):
  mm_main_event     1 if the bout headlines the card (card_position 0)
  mm_co_main        1 if it is the co-main event (card_position 1)
  mm_main_card      1 if it is in the top five bouts (the usual main card)
  mm_card_depth     position / (bouts on card - 1): 0 = top of the card, 1 = first prelim

Fighter-level (previous bouts only):
  mm_prev_depth     card depth of this fighter's previous UFC bout
  mm_avg_depth_3    mean card depth over their previous three bouts
  mm_depth_change   previous depth minus this bout's depth: positive = moved up the card

card_position comes from the ufcstats event page (scripts/backfill_card_positions.py for
history, the scraper for new events). Bouts without it get NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIGHT_LEVEL = ("mm_main_event", "mm_co_main", "mm_main_card", "mm_card_depth")
FIGHTER_LEVEL = ("mm_prev_depth", "mm_avg_depth_3", "mm_depth_change")
FEATURES = FIGHT_LEVEL + FIGHTER_LEVEL


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """df: load_fight_data frame (one row per fighter per fight) with event_id, card_position."""
    out = pd.DataFrame(index=df.index)
    pos = pd.to_numeric(df.get("card_position"), errors="coerce")
    # Bouts on the card = distinct fights per event among our rows (cancelled bouts are
    # already removed, so this is the card as fought).
    n_on_card = df.groupby("event_id")["fight_id"].transform("nunique")
    depth = pos / (n_on_card - 1).clip(lower=1)
    known = pos.notna()
    out["mm_main_event"] = np.where(known, (pos == 0).astype(float), np.nan)
    out["mm_co_main"] = np.where(known, (pos == 1).astype(float), np.nan)
    out["mm_main_card"] = np.where(known, (pos <= 4).astype(float), np.nan)
    out["mm_card_depth"] = depth

    order = df.assign(_depth=depth).sort_values(["stats_fighter_id", "date", "fight_id"],
                                                kind="stable")
    g = order.groupby("stats_fighter_id")["_depth"]
    prev = g.shift(1)
    avg3 = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    out.loc[order.index, "mm_prev_depth"] = prev.values
    out.loc[order.index, "mm_avg_depth_3"] = avg3.values
    out["mm_depth_change"] = out["mm_prev_depth"] - out["mm_card_depth"]
    return out
