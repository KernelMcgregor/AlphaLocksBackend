"""/ufc endpoints for the fightflow feature track (included into app.routers.ufc.router)."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.services import response_cache
from app.services.response_cache import with_session

router = APIRouter()

#: Pace norms move only when a card's stats land; the scan behind them is ~3s.
NORMS_TTL = 12 * 3600
PACE_TTL = 3600
CLUTCH_TTL = 3600


def _norms():
    from app.services.ufc.fightflow_pace import division_norms
    return response_cache.cached("fightflow:pace-norms", with_session(division_norms), NORMS_TTL,
                                 max_stale=2 * NORMS_TTL)


def _build_pace(db: Session, fight_id: int):
    from app.services.ufc.fightflow_pace import fight_pace_payload
    out = fight_pace_payload(db, fight_id, _norms())
    if out is None:
        raise HTTPException(status_code=404, detail="Fight not found")
    return out


@router.get("/fights/{fight_id}/pace")
def get_fight_pace(fight_id: int):
    """Both corners' sig. strikes per minute by round — this bout and career through it —
    against the division's per-round norm, with a fader / closer / steady label."""
    return response_cache.cached(f"fightflow:pace:{fight_id}", with_session(_build_pace, fight_id), PACE_TTL)


def _build_clutch(db: Session, fighter_id: int):
    from app.services.ufc.fightflow_clutch import payload
    out = payload(db, fighter_id)
    if out is None:
        raise HTTPException(status_code=404, detail="No clutch rating for this fighter")
    return out


@router.get("/fighters/{fighter_id}/clutch")
def get_fighter_clutch(fighter_id: int):
    """The fighter's clutch rating (services/ufc/fightflow_clutch): late rounds and rounds
    after a lost one, against their own other rounds in the same bouts."""
    return response_cache.cached(f"fightflow:clutch:{fighter_id}", with_session(_build_clutch, fighter_id),
                                 CLUTCH_TTL)
