"""Backfill `ufc_ranking_history` — divisional rank at every past event date.

Why this exists
---------------
`ufc_fighter_rankings` is truncated and rewritten on every publish, so it only ever
holds the present standings. Nothing records what a fighter's rank was after a given
bout, which is what a rank-over-time chart needs.

Ranks cannot be computed per request: resolving them means loading every UFC bout and
running the opponent-tier recursion over all of it. This module does that work once,
offline, and writes the result down.

The history context is date-independent by construction — an opponent's tier is fixed at
the time of the bout, so it never changes with when you ask — which means ONE build
serves every date. `score(ctx, D)` then bounds its reads at D. That turns the backfill
from ~5 seconds per date into ~0.02, and it is why `build_history` no longer truncates.

Correctness note
----------------
Everything `score` reads is bounded at the date being written: the six-bout window, the
activity clock, the division window, and the reigning champion. Roster STATUS is the one
fact that cannot be bounded — it says who is retired today, not who was then — so it is
applied only to near-present dates. Without that, Khabib Nurmagomedov, the reigning
lightweight champion in January 2020, was absent from the January 2020 ranking.

Run:
    python -m app.services.ufc.rank_history_backfill              # every event date
    python -m app.services.ufc.rank_history_backfill --since 2015-01-01
    python -m app.services.ufc.rank_history_backfill --resume     # skip dates already written
    python -m app.services.ufc.rank_history_backfill --rebuild    # wipe first (ranker changed)
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import date

from sqlalchemy import delete, func, select

from app.models.ufc import UFCEvent, UFCFight, UFCRankingHistory
from app.services.ufc.tapology_rankings import TapologyRanker, build_history

log = logging.getLogger("rank_history_backfill")


def event_dates(db, since: date | None = None,
                until: date | None = None) -> list[date]:
    """Every date on which a UFC bout actually happened, oldest first.

    Ranks only move when someone fights, so event dates are the natural sample points —
    a daily or weekly grid would multiply the work for identical rows.

    Capped at today by default. `ufc_events` carries ANNOUNCED future cards, and a
    "historical" ranking stamped with next month's date is not history: it is today's
    standings wearing a future date, which a rank-over-time chart would draw as a real
    movement that has not happened.
    """
    q = select(UFCEvent.date).join(UFCFight, UFCFight.event_id == UFCEvent.id).distinct()
    if since is not None:
        q = q.where(UFCEvent.date >= since)
    q = q.where(UFCEvent.date <= (until or date.today()))
    return sorted(d for (d,) in db.execute(q).all() if d is not None)


def ranks_as_of(ctx: dict, as_of: date, ranker: TapologyRanker) -> list[dict]:
    """Rank every division as it stood on `as_of`.

    Mirrors ranking_publisher's normalisation so a stored historical score is on the same
    0-1000 scale as the live one, and carries the Strength of Schedule the fighter held on
    that date — SoS is computed from each opponent's standing at the time of the bout, so
    it cannot be recovered later from the fighter's record alone.
    """
    result = ranker.score(ctx, as_of)

    def _score(f, d):
        if result.division_scores:
            return result.division_scores.get((f, d), result.scores.get(f, 0.0))
        return result.scores.get(f, 0.0)

    rows: list[dict] = []
    for division, fids in (result.order or {}).items():
        if len(fids) < 2:
            continue
        vals = [_score(f, division) for f in fids]
        s_min, s_max = min(vals), max(vals)
        span = (s_max - s_min) or 1.0
        for rank, fid in enumerate(fids, 1):
            rows.append({
                "fighter_id": int(fid),
                "as_of": as_of,
                "weight_class": division,
                "rank": rank,
                "score": round((_score(fid, division) - s_min) / span * 1000, 1),
                "total_ranked": len(fids),
                "sos": result.extras.get(fid, {}).get("sos"),
            })
    return rows


#: Dates per transaction when rebuilding. The compute is ~0.02s a date; the cost is
#: network round trips to the database, so batching is what makes a full rebuild minutes
#: rather than tens of minutes.
BATCH_DATES = 40


def backfill(db, since: date | None = None, resume: bool = False,
             rebuild: bool = False) -> int:
    ranker = TapologyRanker()

    dates = event_dates(db, since)
    if resume:
        done = {d for (d,) in db.execute(select(UFCRankingHistory.as_of).distinct()).all()}
        dates = [d for d in dates if d not in done]

    log.info(f"Backfilling {len(dates)} event dates")
    log.info("  Building history (every UFC bout + the tier recursion), once...")
    ctx = build_history(db)
    written = 0

    if rebuild:
        # The whole table is derived from the fight table by the current ranker, so a
        # change of ranker invalidates every row in it at once. Clearing up front beats
        # a per-date DELETE: one statement instead of one per date, and no window in
        # which old and new rows coexist and a rank-over-time chart mixes the two.
        existing = db.execute(select(func.count()).select_from(UFCRankingHistory)).scalar()
        log.info(f"  --rebuild: clearing {existing} existing rows")
        db.execute(delete(UFCRankingHistory))
        db.commit()

    pending: list[dict] = []
    for i, d in enumerate(dates, 1):
        rows = ranks_as_of(ctx, d, ranker)
        if not rows:
            continue

        if not rebuild:
            # Idempotent per date: a re-run replaces that date rather than duplicating it.
            db.execute(delete(UFCRankingHistory).where(UFCRankingHistory.as_of == d))
        pending.extend(rows)
        written += len(rows)

        if len(pending) >= BATCH_DATES * 200 or i == len(dates):
            db.bulk_insert_mappings(UFCRankingHistory, pending)
            db.commit()
            pending = []
            log.info(f"  [{i}/{len(dates)}] {d}  {written} rows written")

    if pending:
        db.bulk_insert_mappings(UFCRankingHistory, pending)
        db.commit()

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

    def _sos(r) -> int | None:
        """Strength of Schedule, lifted out of the published feature_profile blob.

        Recomputing it here would mean replaying the whole tier recursion; the publish
        that just ran already did that work and wrote the answer down.
        """
        try:
            return json.loads(r.feature_profile or "{}").get("sos")
        except (json.JSONDecodeError, TypeError):
            return None

    rows = [{
        "fighter_id": r.fighter_id,
        "as_of": as_of,
        "weight_class": r.weight_class,
        "rank": r.rank,
        "score": r.score,
        "total_ranked": per_division[r.weight_class],
        "sos": _sos(r),
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
    ap.add_argument("--rebuild", action="store_true",
                    help="clear the table first — use after a ranker change, which "
                         "invalidates every existing row")
    args = ap.parse_args()

    from app.database import SessionLocal

    _db = SessionLocal()
    try:
        backfill(_db, since=args.since, resume=args.resume, rebuild=args.rebuild)
    finally:
        _db.close()
