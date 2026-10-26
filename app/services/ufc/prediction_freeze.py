"""Keep the prediction a fight was actually served with once the fight has started.

Every prediction job (winner, method, rounds) rebuilds its table from scratch on each run,
completed fights included. Without this, a finished fight's page shows whatever the model of the
day says about it rather than what the site said going in: UFC 333's Volkanovski vs Evloev was
served before the fight, then rewritten on 26 Oct 2026 -- two days later -- by a run whose features
differ for a bout that is now in the past, and its pick flipped.

A row is frozen when it was written before its bout's cutoff and the cutoff has passed. Rows for
older fights were all written after their events (by the rebuilds before this existed), so they
are not frozen and keep being refreshed with each retrain's out-of-sample predictions.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from app.models.ufc import UFCEvent, UFCFight

#: Only rows written by this point after the event date can be pre-fight, so only they need a
#: cutoff computed. Saves evaluating every card in history on each run.
_CANDIDATE_SLACK = dt.timedelta(days=2)


def frozen_fight_ids(db: Session, model) -> set[int]:
    """Fights whose stored `model` row is a pre-fight prediction for a bout that has started."""
    from app.services.ufc.prediction_markets.common import cutoffs_for_fights, fight_started

    rows = (
        db.query(model.fight_id, model.created_at, UFCEvent.date)
        .join(UFCFight, UFCFight.id == model.fight_id)
        .join(UFCEvent, UFCEvent.id == UFCFight.event_id)
        .all()
    )
    candidates = {
        fid: created for fid, created, date in rows
        if created is not None and date is not None
        and created < dt.datetime.combine(date, dt.time.min) + _CANDIDATE_SLACK
    }
    if not candidates:
        return set()
    cutoffs = cutoffs_for_fights(db, candidates)
    return {
        fid for fid, created in candidates.items()
        if (c := cutoffs.get(fid)) is not None and created < c and fight_started(c)
    }


def clear_unfrozen(db: Session, model, frozen: set[int]) -> None:
    """Delete every row of `model` except the frozen fights'. Caller commits."""
    query = db.query(model)
    if frozen:
        query = query.filter(~model.fight_id.in_(list(frozen)))
    query.delete(synchronize_session=False)
