"""/ufc endpoints for the fighter_extras feature track (included into app.routers.ufc.router).

Percentiles, rating history and splits are computed per request (cached); the age
curves and the camp registry are precomputed (fighter_extras_age / fighter_extras_camps).
All of it is display only.
"""
from fastapi import APIRouter, Query

from app.services import response_cache
from app.services.response_cache import with_session

router = APIRouter()

EXTRAS_TTL = 30 * 60


@router.get("/fighters/{fighter_id}/percentiles")
def get_fighter_percentiles(fighter_id: int):
    """Career stats as percentiles within the fighter's current division."""
    from app.services.ufc.fighter_extras_profile import fighter_percentiles, percentile_pools

    pools = response_cache.cached("fx:percentile-pools", with_session(percentile_pools), EXTRAS_TTL)
    return response_cache.cached(
        f"fx:percentiles:{fighter_id}", with_session(fighter_percentiles, fighter_id, pools), EXTRAS_TTL,
    )


@router.get("/fighters/{fighter_id}/rating-history")
def get_fighter_rating_history(fighter_id: int):
    """Pre-fight Glicko ratings per bout and the change each bout produced."""
    from app.services.ufc.fighter_extras_profile import rating_history
    return response_cache.cached(
        f"fx:rating-history:{fighter_id}", with_session(rating_history, fighter_id), EXTRAS_TTL,
    )


@router.get("/fighters/{fighter_id}/splits")
def get_fighter_splits(fighter_id: int):
    """Record and striking by situation: short notice, layoff, odds, rounds, altitude."""
    from app.services.ufc.fighter_extras_profile import fighter_splits
    return response_cache.cached(
        f"fx:splits:{fighter_id}", with_session(fighter_splits, fighter_id), EXTRAS_TTL,
    )


@router.get("/fighters/{fighter_id}/age-curve")
def get_fighter_age_curve(fighter_id: int):
    """Division win rate by age (precomputed) with this fighter's bouts by age."""
    from app.services.ufc.fighter_extras_age import fighter_age_curve
    return response_cache.cached(
        f"fx:age-curve:{fighter_id}", with_session(fighter_age_curve, fighter_id), EXTRAS_TTL,
    )


@router.get("/fighters/{fighter_id}/camp")
def get_fighter_camp(fighter_id: int):
    """The fighter's camp and teammates; camp null when none is on record."""
    from app.services.ufc.fighter_extras_camps import fighter_camp
    return response_cache.cached(
        f"fx:fighter-camp:{fighter_id}", with_session(fighter_camp, fighter_id), EXTRAS_TTL,
    )


@router.get("/camps")
def list_camps(min_fighters: int = Query(default=3, ge=1, le=50)):
    """Camp leaderboard: roster size, active fighters, ranked fighters and UFC record."""
    from app.services.ufc.fighter_extras_camps import camps_leaderboard
    return response_cache.cached(
        f"fx:camps:{min_fighters}", with_session(camps_leaderboard, min_fighters), EXTRAS_TTL,
    )


@router.get("/camps/{camp_ref}")
def get_camp(camp_ref: str):
    """One camp by id or slug, with its full roster."""
    from app.services.ufc.fighter_extras_camps import camp_detail
    return response_cache.cached(f"fx:camp:{camp_ref}", with_session(camp_detail, camp_ref), EXTRAS_TTL)


@router.get("/rankings/timelapse")
def get_rankings_timelapse(
    division: str = Query(default="lightweight"),
    step: str = Query(default="quarter"),
    top: int = Query(default=15, ge=5, le=30),
):
    """Top-N divisional rankings through time, one table per period."""
    from app.services.ufc.fighter_extras_timelapse import timelapse
    return response_cache.cached(
        f"fx:timelapse:{division}:{step}:{top}", with_session(timelapse, division, step, top), EXTRAS_TTL,
    )
