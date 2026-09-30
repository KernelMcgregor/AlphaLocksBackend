"""Withdrawal history: how often a fighter has pulled out of booked bouts.

A fighter who keeps withdrawing is often carrying injuries or struggling with weight;
both matter for the next fight. Built from ufc_cancelled_bouts (BestFightOdds history,
backfilled with include_scrapped, plus every bout reconcile_service removes).

For each pulled bout, who withdrew:
  - if exactly one of its fighters fought someone else on that card (within 10 days),
    the OTHER one withdrew;
  - if neither fought, the bout was scrapped and it is unknown who pulled out; both are
    credited with a "scrapped" bout instead.

Features per fighter, counting only pulled bouts on cards BEFORE this fight:
  wd_withdrawals_3y   withdrawals in the previous three years
  wd_scrapped_3y      scrapped bouts they were part of in the previous three years
  wd_days_since       days since their last withdrawal (NaN if none on record)

Coverage follows the source: BestFightOdds kept few cancelled bouts before ~2020.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

FEATURES = ("wd_withdrawals_3y", "wd_scrapped_3y", "wd_days_since")
WINDOW_DAYS = 10
LOOKBACK = timedelta(days=3 * 365)


def classify(pulled_rows, fought_on: dict[int, list]) -> tuple[dict, dict]:
    """-> (fighter -> sorted withdrawal dates, fighter -> sorted scrapped dates)."""
    def fought_near(fid, d):
        return any(abs((x - d).days) <= WINDOW_DAYS for x in fought_on.get(fid, []))

    withdrew, scrapped = defaultdict(list), defaultdict(list)
    for d, r, b in pulled_rows:
        if d is None:
            continue
        r_f = r is not None and fought_near(r, d)
        b_f = b is not None and fought_near(b, d)
        if r_f and not b_f and b is not None:
            withdrew[b].append(d)
        elif b_f and not r_f and r is not None:
            withdrew[r].append(d)
        elif not r_f and not b_f:
            for f in (r, b):
                if f is not None:
                    scrapped[f].append(d)
    for m in (withdrew, scrapped):
        for v in m.values():
            v.sort()
    return withdrew, scrapped


def features_from_rows(df: pd.DataFrame, pulled_rows) -> pd.DataFrame:
    """df needs date, stats_fighter_id (a load_fight_data frame: its rows ARE the fights)."""
    dates = pd.to_datetime(df["date"]).dt.date
    fought_on = defaultdict(list)
    for fid, d in zip(df["stats_fighter_id"], dates):
        fought_on[int(fid)].append(d)
    withdrew, scrapped = classify(pulled_rows, fought_on)

    def count(lst, d):
        if not lst:
            return 0.0
        return float(bisect_left(lst, d) - bisect_left(lst, d - LOOKBACK))

    w3, s3, since = [], [], []
    for fid, d in zip(df["stats_fighter_id"], dates):
        wl, sl = withdrew.get(int(fid), []), scrapped.get(int(fid), [])
        w3.append(count(wl, d))
        s3.append(count(sl, d))
        k = bisect_left(wl, d)
        since.append(float((d - wl[k - 1]).days) if k else np.nan)
    return pd.DataFrame({"wd_withdrawals_3y": w3, "wd_scrapped_3y": s3,
                         "wd_days_since": since}, index=df.index)


def compute(df: pd.DataFrame) -> pd.DataFrame | None:
    from sqlalchemy import inspect

    from app.database import SessionLocal, engine
    from app.models.ufc import UFCCancelledBout

    t = UFCCancelledBout.__table__
    if not inspect(engine).has_table(t.name, schema=t.schema):
        return None
    db = SessionLocal()
    try:
        rows = db.query(UFCCancelledBout.event_date, UFCCancelledBout.red_fighter_id,
                        UFCCancelledBout.blue_fighter_id).all()
    finally:
        db.close()
    return features_from_rows(df, rows)
