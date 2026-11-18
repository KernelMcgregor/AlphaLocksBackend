"""/ufc endpoints for the markets feature track (included into app.routers.ufc.router)."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.services import response_cache
from app.services.response_cache import with_session

router = APIRouter()

#: Sportsbook lines are re-read every two hours; a fight's movement chart can lag by part of that.
BOOK_TTL = 20 * 60
CLV_TTL = 60 * 60


@router.get("/fights/{fight_id}/book-history")
def get_fight_book_history(fight_id: int):
    """Per-book moneyline curves (vig-free, red corner) up to the bout's cutoff, Pinnacle against
    the soft-book median with Pinnacle's max stake, and dated markers (weigh-ins, opponent
    changes, news spikes, model publish). See app/services/ufc/markets_book_history.py."""
    from app.services.ufc.markets_book_history import book_history
    out = response_cache.cached(f"book-history:{fight_id}", with_session(book_history, fight_id), BOOK_TTL)
    if out is None:
        raise HTTPException(status_code=404, detail="Fight not found")
    return out


@router.get("/fights/{fight_id}/crowd")
def get_fight_crowd(fight_id: int):
    """Fan scorecards (mmadecisions), the model and the vig-free close for one fight."""
    from app.services.ufc.markets_crowd import crowd
    out = response_cache.cached(f"fight-crowd:{fight_id}", with_session(crowd, fight_id), BOOK_TTL)
    if out is None:
        raise HTTPException(status_code=404, detail="Fight not found")
    return out


@router.get("/model/clv")
def get_model_clv(source: str = Query(default="all"), start: str | None = None, end: str | None = None,
                  min_edge: float = Query(default=0.0, ge=0.0, le=0.5)):
    """Closing line value of the model's moneyline side against the open, over time. source=live
    keeps only predictions the site served before the fight. See markets_clv.py."""
    from app.services.ufc.markets_clv import clv_report
    from app.services.ufc.model_results import load_ledger
    if source not in ("all", "live"):
        raise HTTPException(status_code=400, detail="source must be all|live")
    df = load_ledger()
    if df is None:
        raise HTTPException(status_code=503, detail="backtest ledger not built")
    # id(df) changes when the ledger file is re-read, retiring answers built from the old one
    return response_cache.cached(f"model-clv:{id(df)}:{source}:{start}:{end}:{min_edge}",
                                 lambda: clv_report(df, source, start, end, min_edge), CLV_TTL)
