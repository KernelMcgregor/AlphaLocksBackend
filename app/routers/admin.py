from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.shared import ModelRun, OddsSnapshot, Prediction
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFightStats

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    return {
        "ufc": {
            "fighters": db.query(UFCFighter).count(),
            "events": db.query(UFCEvent).count(),
            "fights": db.query(UFCFight).count(),
            "fight_stats": db.query(UFCFightStats).count(),
        },
        "shared": {
            "predictions": db.query(Prediction).count(),
            "model_runs": db.query(ModelRun).count(),
            "odds_snapshots": db.query(OddsSnapshot).count(),
        },
    }


@router.post("/scrape")
def trigger_scrape(
    background_tasks: BackgroundTasks,
    mode: str = Query(default="full", pattern="^(full|update)$"),
):
    from app.services.scraper import run_scrape

    background_tasks.add_task(run_scrape, mode=mode)
    return {"message": f"Scrape started in background (mode={mode})"}


@router.post("/scrape-upcoming")
def trigger_upcoming_scrape(background_tasks: BackgroundTasks):
    from app.services.scraper import scrape_upcoming

    background_tasks.add_task(scrape_upcoming)
    return {"message": "Upcoming scrape started in background"}


@router.post("/scrape-recent")
def trigger_recent_update(background_tasks: BackgroundTasks):
    from app.services.scraper import run_recent_update

    background_tasks.add_task(run_recent_update)
    return {"message": "Recent update started in background"}


@router.post("/scrape-live-odds")
def trigger_live_odds_scrape(background_tasks: BackgroundTasks):
    from app.services.odds_scraper import run_live_odds_scrape

    background_tasks.add_task(run_live_odds_scrape)
    return {"message": "Live odds scrape started in background"}


@router.post("/generate-preview/{fight_id}")
def trigger_preview_generation(
    fight_id: int,
    force: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    from app.services.preview_service import generate_preview

    preview = generate_preview(fight_id, db, force=force)
    if not preview:
        return {"message": "Preview generation failed — check API key and fight data"}
    return {"message": "Preview generated", "fight_id": fight_id}


@router.post("/generate-all-previews")
def trigger_all_previews(
    background_tasks: BackgroundTasks,
    force: bool = Query(default=False),
):
    from app.services.preview_service import generate_all_upcoming_previews

    background_tasks.add_task(generate_all_upcoming_previews, force=force)
    return {"message": "Preview generation started for all upcoming fights"}
