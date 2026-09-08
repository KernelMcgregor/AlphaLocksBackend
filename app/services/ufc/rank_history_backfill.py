"""Backfill `ufc_ranking_history` — divisional rank at every past event date.

Why this exists
---------------
`ufc_fighter_rankings` is truncated and rewritten on every publish, so it only ever
holds the present standings. Nothing records what a fighter's rank was after a given
bout, which is what a rank-over-time chart needs.

Ranks cannot be computed per request. `PointsEloRanker.rank()` reloads the whole fight
table and replays Elo across all of UFC history on each call, so one call per fight date
is seconds-to-minutes of work. This module does that replay once, offline, and writes
the result down.

Correctness note
----------------
The registry is rebuilt with `as_of=D` for each date. Passing today's registry would
apply today's eligibility (division, activity, fight/round counts) to a historical
ranking — excluding fighters who have since retired and admitting ones who had not yet
debuted. The ranks would look plausible and be wrong.

Run:
    python -m app.services.ufc.rank_history_backfill              # every event date
    python -m app.services.ufc.rank_history_backfill --since 2015-01-01
    python -m app.services.ufc.rank_history_backfill --resume     # skip dates already written
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import date

from sqlalchemy import delete, func, select

from app.models.ufc import UFCEvent, UFCFight, UFCRankingHistory
from app.services.ufc.fighter_registry import Eligibility, build_fighter_registry
from app.services.ufc.points_ranking_service import PointsEloRanker

log = logging.getLogger("rank_history_backfill")


def event_dates(db, since: date | None = None) -> list[date]:
    """Every date on which a UFC bout actually happened, oldest first.

    Ranks only move when someone fights, so event dates are the natural sample points —
    a daily or weekly grid would multiply the work for identical rows.
    """
    q = select(UFCEvent.date).join(UFCFight, UFCFight.event_id == UFCEvent.id).distinct()
    if since is not None:
        q = q.where(UFCEvent.date >= since)
    return sorted(d for (d,) in db.execute(q).all() if d is not None)


def ranks_as_of(db, as_of: date, crit: Eligibility, ranker: PointsEloRanker) -> list[dict]:
    """Rank every division as it stood on `as_of`. Mirrors ranking_publisher's
    normalisation so the stored score matches what the live rankings show."""
    registry = build_fighter_registry(db, as_of=as_of)
    result = ranker.rank(db, registry, as_of, crit)

    rows: list[dict] = []
    for division, fids in (result.order or {}).items():
        if len(fids) < 2:
            continue
        vals = [result.scores.get(f, 0.0) for f in fids]
        s_min, s_max = min(vals), max(vals)
        span = (s_max - s_min) or 1.0
        for rank, fid in enumerate(fids, 1):
            rows.append({
                "fighter_id": int(fid),
                "as_of": as_of,
                "weight_class": division,
                "rank": rank,
                "score": round((result.scores.get(fid, 0.0) - s_min) / span * 1000, 1),
                "total_ranked": len(fids),
            })
    return rows


def backfill(db, since: date | None = None, resume: bool = False) -> int:
    crit = Eligibility()
    ranker = PointsEloRanker()

    dates = event_dates(db, since)
    if resume:
        done = {d for (d,) in db.execute(select(UFCRankingHistory.as_of).distinct()).all()}
        dates = [d for d in dates if d not in done]

    log.info(f"Backfilling {len(dates)} event dates")
    written = 0

    for i, d in enumerate(dates, 1):
        rows = ranks_as_of(db, d, crit, ranker)
        if not rows:
            log.info(f"  [{i}/{len(dates)}] {d}  no rankable divisions")
            continue

        # Idempotent per date: a re-run replaces that date rather than duplicating it.
        db.execute(delete(UFCRankingHistory).where(UFCRankingHistory.as_of == d))
        db.bulk_insert_mappings(UFCRankingHistory, rows)
        db.commit()

        written += len(rows)
        log.info(f"  [{i}/{len(dates)}] {d}  {len(rows)} rows")

    log.info(f"Done — {written} rows across {len(dates)} dates")
    return written


def record_rank_history(db) -> int:
    """Append the freshly-published ranking to `ufc_ranking_history`.

    Runs in POST_EVENT_CHAIN immediately after `publish_rankings`, which has just
    overwritten `ufc_fighter_rankings` with the current standings. This copies those
    rows out under the completed card's date before the next publish destroys them —
    that copy is the whole reason history exists.

    Cheap by construction: it reads the ranking that was just computed rather than
    replaying Elo, so it costs one query, unlike the backfill.
    """
    from app.models.ufc import UFCFighterRanking

    # Stamp with the latest card that has actually happened, not today — the ranking
    # describes the state *after* that event, and the chart joins on fight dates.
    as_of = db.execute(
        select(func.max(UFCEvent.date))
        .join(UFCFight, UFCFight.event_id == UFCEvent.id)
        .where(UFCEvent.date <= date.today())
    ).scalar()
    if as_of is None:
        log.warning("record_rank_history: no completed events, nothing to stamp")
        return 0

    current = db.query(UFCFighterRanking).all()
    if not current:
        log.warning("record_rank_history: ufc_fighter_rankings is empty, nothing to record")
        return 0

    per_division: dict[str, int] = defaultdict(int)
    for r in current:
        per_division[r.weight_class] += 1

    rows = [{
        "fighter_id": r.fighter_id,
        "as_of": as_of,
        "weight_class": r.weight_class,
        "rank": r.rank,
        "score": r.score,
        "total_ranked": per_division[r.weight_class],
    } for r in current]

    # Idempotent: re-running the chain for the same card replaces that date.
    db.execute(delete(UFCRankingHistory).where(UFCRankingHistory.as_of == as_of))
    db.bulk_insert_mappings(UFCRankingHistory, rows)
    db.commit()
    log.info(f"record_rank_history: wrote {len(rows)} rows for {as_of}")
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=date.fromisoformat, default=None,
                    help="only backfill event dates on or after this ISO date")
    ap.add_argument("--resume", action="store_true",
                    help="skip dates already present in ufc_ranking_history")
    args = ap.parse_args()

    from app.database import SessionLocal

    _db = SessionLocal()
    try:
        backfill(_db, since=args.since, resume=args.resume)
    finally:
        _db.close()
