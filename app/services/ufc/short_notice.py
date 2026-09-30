"""Short-notice replacement / opponent-change features, from ufc_cancelled_bouts.

If A was booked against B, B withdrew and C stepped in, the card's history contains a
pulled bout "A vs B" and the fought bout "A vs C". Per fighter in a fought bout:

  sn_opp_changed  1 if this fighter had a different opponent pulled from the same card
                  (A: the camp was built for someone else)
  sn_replacement  1 if this fighter had no pulled bout but their opponent did
                  (C: stepped in on short notice)

Walk-forward 2022-26: replacement fighters won 33.6% where the model (without these)
said 43.1% and the closing line 39.8%; fighters whose opponent changed won 64.9% vs 56.3%.

ufc_cancelled_bouts is filled two ways:
  - reconcile_service records every bout it removes from an upcoming card (live), and
  - backfill_from_bfo() loads historical pulled bouts from the BestFightOdds crawl, which
    keeps cancelled matchups on its event pages (mostly 2020 onward).
So training (history) and serving (upcoming cards) read the same table.
"""
from __future__ import annotations

import logging
from bisect import bisect_left
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("short_notice")

FEATURES = ("sn_replacement", "sn_opp_changed")
WINDOW_DAYS = 10  # a pulled bout counts for fights within this many days of its card


def _pulled_by_fighter(rows) -> dict[int, list[tuple[date, int | None]]]:
    """fighter id -> sorted [(card date, opponent id)] of pulled bouts."""
    out: dict[int, list] = defaultdict(list)
    for d, r, b in rows:
        if d is None:
            continue
        if r is not None:
            out[int(r)].append((d, None if b is None else int(b)))
        if b is not None:
            out[int(b)].append((d, None if r is None else int(r)))
    for v in out.values():
        v.sort(key=lambda x: (x[0], x[1] or 0))
    return out


def _had_pulled(pulled, fighter, fight_date, current_opp) -> bool:
    lst = pulled.get(fighter)
    if not lst:
        return False
    lo = bisect_left([d for d, _ in lst], fight_date - timedelta(days=WINDOW_DAYS))
    for d, opp in lst[lo:]:
        if d > fight_date + timedelta(days=WINDOW_DAYS):
            break
        if opp != current_opp:
            return True
    return False


def features_from_rows(df: pd.DataFrame, pulled_rows) -> pd.DataFrame:
    """df needs date, stats_fighter_id, red_fighter_id, blue_fighter_id."""
    pulled = _pulled_by_fighter(pulled_rows)
    dates = pd.to_datetime(df["date"]).dt.date.to_numpy()
    me = df["stats_fighter_id"].to_numpy()
    red, blue = df["red_fighter_id"].to_numpy(), df["blue_fighter_id"].to_numpy()
    opp = np.where(me == red, blue, red)
    changed_me = np.array([_had_pulled(pulled, int(m), d, int(o))
                           for m, o, d in zip(me, opp, dates)], dtype=bool)
    changed_opp = np.array([_had_pulled(pulled, int(o), d, int(m))
                            for m, o, d in zip(me, opp, dates)], dtype=bool)
    return pd.DataFrame({
        "sn_replacement": (changed_opp & ~changed_me).astype(float),
        "sn_opp_changed": changed_me.astype(float),
    }, index=df.index)


def compute(df: pd.DataFrame) -> pd.DataFrame | None:
    """Features for a load_fight_data frame, or None if the table does not exist."""
    from sqlalchemy import inspect

    from app.database import SessionLocal, engine
    from app.models.ufc import UFCCancelledBout

    tbl = UFCCancelledBout.__table__
    if not inspect(engine).has_table(tbl.name, schema=tbl.schema):
        return None
    db = SessionLocal()
    try:
        rows = db.query(UFCCancelledBout.event_date, UFCCancelledBout.red_fighter_id,
                        UFCCancelledBout.blue_fighter_id).all()
    finally:
        db.close()
    out = features_from_rows(df, rows)
    log.info(f"  Short-notice: {len(rows)} pulled bouts; {int(out['sn_replacement'].sum())} "
             f"replacement rows, {int(out['sn_opp_changed'].sum())} opponent-changed rows")
    return out


def backfill_from_bfo(csv_path: Path = Path("data/bfo/odds.csv"),
                      include_scrapped: bool = False) -> int:
    """Load historical pulled bouts from the BFO crawl into ufc_cancelled_bouts.

    A BFO matchup that never matched a fought bout, where at least one of its fighters
    fought someone else on the same card (within WINDOW_DAYS), is a pulled bout. With
    include_scrapped, bouts scrapped outright (neither fighter fought that card) are loaded
    too; withdrawals.py needs them. Unmatched pairs that DID fight each other that week are
    matching failures and always skipped. Fighters are mapped to our ids through their BFO
    slugs in matched bouts; bouts with neither fighter mapped (e.g. TUF exhibitions) skip.
    """
    from app.database import SessionLocal, engine
    from app.models.ufc import UFCCancelledBout, UFCFight
    from app.services.ufc.external_records import _upsert

    UFCCancelledBout.__table__.create(bind=engine, checkfirst=True)
    bfo = pd.read_csv(csv_path, low_memory=False, dtype={"db_fight_id": str})
    c = bfo[bfo["bookmaker"] == "Consensus"].copy()
    c["date"] = pd.to_datetime(c["event_date"]).dt.date
    matched = c[c["db_fight_id"].notna()]

    db = SessionLocal()
    try:
        # Fight ids are ~1.2e18: parse from text, never via float.
        ids = [int(str(v).split(".")[0]) for v in matched["db_fight_id"]]
        corners = {f.id: (f.red_fighter_id, f.blue_fighter_id)
                   for f in db.query(UFCFight).filter(UFCFight.id.in_(ids))}
        slug_to_id: dict[str, int] = {}
        fought_on: dict[int, list] = defaultdict(list)
        for (_, r), fid in zip(matched.iterrows(), ids):
            if fid not in corners:
                continue
            red, blue = corners[fid]
            a, b = (blue, red) if str(r["db_swapped"]) == "True" else (red, blue)
            slug_to_id[r["fighter_a_bfo_slug"]] = a
            slug_to_id[r["fighter_b_bfo_slug"]] = b
            fought_on[a].append(r["date"])
            fought_on[b].append(r["date"])

        def fought_near(fid, d):
            return any(abs((x - d).days) <= WINDOW_DAYS for x in fought_on.get(fid, []))

        # Every pair that actually fought, from our own fights table (not BFO's matches:
        # a matching failure is exactly a bout BFO did not match).
        pair_dates: dict[frozenset, list] = defaultdict(list)
        for fr, fb, fd in db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id,
                                   UFCFight.date).filter(UFCFight.date.isnot(None)):
            pair_dates[frozenset((fr, fb))].append(fd)

        rows = []
        for _, r in c[c["db_fight_id"].isna()].iterrows():
            a = slug_to_id.get(r["fighter_a_bfo_slug"])
            b = slug_to_id.get(r["fighter_b_bfo_slug"])
            if a and b and any(abs((x - r["date"]).days) <= WINDOW_DAYS
                               for x in pair_dates.get(frozenset((a, b)), [])):
                continue  # they did fight each other: a matching failure, not a pull
            replaced = (a and fought_near(a, r["date"])) or (b and fought_near(b, r["date"]))
            if not replaced and not (include_scrapped and (a or b)):
                continue
            opened = (pd.to_datetime(r["open_ts_a"], utc=True).tz_localize(None).to_pydatetime()
                      if isinstance(r["open_ts_a"], str) else None)
            rows.append({
                "ufcstats_id": f"bfo:{int(r['bfo_matchup_id'])}", "event_id": None,
                "event_date": r["date"], "red_fighter_id": a, "blue_fighter_id": b,
                "weight_class": None, "booked_at": opened,
                "removed_at": datetime.combine(r["date"], datetime.min.time()),
            })
        uniq = list({x["ufcstats_id"]: x for x in rows}.values())
        _upsert(db, UFCCancelledBout, uniq, ["ufcstats_id"], update=False)
    finally:
        db.close()
    log.info(f"Backfilled {len(uniq)} pulled bouts from BFO")
    return len(uniq)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill-bfo", type=Path, nargs="?", const=Path("data/bfo/odds.csv"))
    ap.add_argument("--include-scrapped", action="store_true")
    a = ap.parse_args()
    if a.backfill_bfo:
        backfill_from_bfo(a.backfill_bfo, include_scrapped=a.include_scrapped)
