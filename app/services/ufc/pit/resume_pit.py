"""A working resume score, as of before each fight (item E).

model.build_features' `resume_score` is identically 0: its recursion is homogeneous and
seeded at 0 (see the KNOWN DEFECT note there). This gives it the source term that note
suggests: credit by the opponent's PRE-fight Elo.

  q = (opponent's pre-fight Elo - 1500) / 400
  rs_mean       mean over prior decided bouts of  q * (1.3 if finish else 1) for a win,
                0.3 * (q - 1) for a loss  (a loss to a strong opponent costs less)
  rs_best_win   best q among prior wins (NaN with no wins)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

PREFIX = "rs_"
FIGHT_LEVEL = ()
RAW = ()


def features_from(fighters: pd.DataFrame) -> pd.DataFrame:
    """fighters: fight_id, stats_fighter_id, red/blue_fighter_id, date, won, winner_id, elo, method."""
    fr = fighters.copy()
    fr["date"] = pd.to_datetime(fr["date"])
    elo = fr.set_index(["fight_id", "stats_fighter_id"])["elo"]
    opp = np.where(fr.stats_fighter_id == fr.red_fighter_id, fr.blue_fighter_id, fr.red_fighter_id)
    fr["q"] = (pd.Series(list(zip(fr.fight_id, opp))).map(elo).to_numpy() - 1500.0) / 400.0
    sums, ns, best = defaultdict(float), defaultdict(int), {}
    out = []
    for _, day in fr.sort_values(["date", "fight_id"]).groupby("date", sort=True):
        for r in day.itertuples(index=False):
            f = r.stats_fighter_id
            row = {"fight_id": r.fight_id, "fighter_id": f}
            if ns[f]:
                row["rs_mean"] = sums[f] / ns[f]
            if f in best:
                row["rs_best_win"] = best[f]
            out.append(row)
        for r in day.itertuples(index=False):
            if pd.isna(r.winner_id) or pd.isna(r.q):
                continue
            f = r.stats_fighter_id
            m = str(r.method or "")
            if r.won == 1:
                sums[f] += r.q * (1.3 if ("KO" in m or "Sub" in m) else 1.0)
                best[f] = max(best.get(f, -np.inf), r.q)
            else:
                sums[f] += 0.3 * (r.q - 1.0)
            ns[f] += 1
    return pd.DataFrame(out)


def compute(ctx) -> pd.DataFrame:
    return features_from(ctx.fighters)
