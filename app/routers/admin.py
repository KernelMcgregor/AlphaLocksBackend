from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.shared import ModelRun, OddsSnapshot, Prediction
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFightStats

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin_key(x_admin_key: str = Header(...)):
    if not settings.ADMIN_API_KEY:
        raise HTTPException(503, "ADMIN_API_KEY not configured on server")
    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(403, "Invalid admin key")


# ---------------------------------------------------------------------------
# Stats & scheduler
# ---------------------------------------------------------------------------

@router.get("/stats", dependencies=[Depends(require_admin_key)])
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


@router.get("/scheduler", dependencies=[Depends(require_admin_key)])
def get_scheduler_status():
    from app.main import scheduler

    jobs = []
    for job in scheduler.get_jobs():
        next_run = job.next_run_time
        jobs.append({
            "id": job.id,
            "name": job.name or job.id,
            "next_run": next_run.isoformat() if next_run else None,
            "trigger": str(job.trigger),
        })
    return {"jobs": jobs}


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

@router.post("/scrape", dependencies=[Depends(require_admin_key)])
def trigger_scrape(
    background_tasks: BackgroundTasks,
    mode: str = Query(default="full", pattern="^(full|update)$"),
):
    from app.services.scraper import run_scrape

    background_tasks.add_task(run_scrape, mode=mode)
    return {"message": f"Scrape started in background (mode={mode})"}


@router.post("/scrape-upcoming", dependencies=[Depends(require_admin_key)])
def trigger_upcoming_scrape(background_tasks: BackgroundTasks):
    from app.services.scraper import scrape_upcoming

    background_tasks.add_task(scrape_upcoming)
    return {"message": "Upcoming scrape started in background"}


@router.post("/scrape-recent", dependencies=[Depends(require_admin_key)])
def trigger_recent_update(background_tasks: BackgroundTasks):
    from app.services.scraper import run_recent_update

    background_tasks.add_task(run_recent_update)
    return {"message": "Recent update started in background"}


@router.post("/scrape-live-odds", dependencies=[Depends(require_admin_key)])
def trigger_live_odds_scrape(background_tasks: BackgroundTasks):
    from app.services.odds_scraper import run_live_odds_scrape

    background_tasks.add_task(run_live_odds_scrape)
    return {"message": "Live odds scrape started in background"}


@router.post("/scrape-historical-odds", dependencies=[Depends(require_admin_key)])
def trigger_historical_odds_scrape(
    background_tasks: BackgroundTasks,
    since: int = Query(default=2022),
):
    from app.services.odds_scraper import run_odds_scrape

    background_tasks.add_task(run_odds_scrape, since)
    return {"message": f"Historical odds scrape started (since={since})"}


@router.post("/scrape-bovada", dependencies=[Depends(require_admin_key)])
def trigger_bovada_scrape(background_tasks: BackgroundTasks):
    from app.services.bovada_scraper import scrape_bovada_method_odds

    background_tasks.add_task(scrape_bovada_method_odds)
    return {"message": "Bovada method odds scrape started in background"}


@router.post("/scrape-profiles", dependencies=[Depends(require_admin_key)])
def trigger_profile_scrape(
    background_tasks: BackgroundTasks,
    images: bool = Query(default=True),
):
    from app.services.ufc_profile_scraper import run as run_profiles

    background_tasks.add_task(run_profiles, images=images)
    return {"message": f"Profile scrape started (images={images})"}


# ---------------------------------------------------------------------------
# Model training & predictions
# ---------------------------------------------------------------------------

@router.post("/train-model", dependencies=[Depends(require_admin_key)])
def trigger_model_training(background_tasks: BackgroundTasks):
    from app.services.model import run as run_training

    background_tasks.add_task(run_training)
    return {"message": "Winner model training started in background"}


@router.post("/train-method-model", dependencies=[Depends(require_admin_key)])
def trigger_method_model_training(background_tasks: BackgroundTasks):
    from app.services.method_model import run as run_method_training

    background_tasks.add_task(run_method_training)
    return {"message": "Method model training started in background"}


@router.post("/generate-predictions", dependencies=[Depends(require_admin_key)])
def trigger_predictions(background_tasks: BackgroundTasks):
    from app.services.model import generate_predictions

    background_tasks.add_task(generate_predictions)
    return {"message": "Winner predictions generation started in background"}


@router.post("/generate-method-predictions", dependencies=[Depends(require_admin_key)])
def trigger_method_predictions(background_tasks: BackgroundTasks):
    from app.services.method_model import generate_method_predictions

    background_tasks.add_task(generate_method_predictions)
    return {"message": "Method predictions generation started in background"}


@router.post("/generate-rankings", dependencies=[Depends(require_admin_key)])
def trigger_rankings(background_tasks: BackgroundTasks):
    from app.services.ranking_service import generate_rankings

    background_tasks.add_task(generate_rankings)
    return {"message": "Rankings generation started in background"}


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

@router.post("/generate-preview/{fight_id}", dependencies=[Depends(require_admin_key)])
def trigger_preview_generation(
    fight_id: int,
    force: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    from app.services.preview_service import generate_preview

    preview = generate_preview(fight_id, db, force=force)
    if not preview:
        return {"message": "Preview generation failed -- check API key and fight data"}
    return {"message": "Preview generated", "fight_id": fight_id}


@router.post("/generate-all-previews", dependencies=[Depends(require_admin_key)])
def trigger_all_previews(
    background_tasks: BackgroundTasks,
    force: bool = Query(default=False),
):
    from app.services.preview_service import generate_all_upcoming_previews

    background_tasks.add_task(generate_all_upcoming_previews, force=force)
    return {"message": "Preview generation started for all upcoming fights"}


# ---------------------------------------------------------------------------
# Full pipeline (same as scheduled job)
# ---------------------------------------------------------------------------

@router.post("/run-full-pipeline", dependencies=[Depends(require_admin_key)])
def trigger_full_pipeline(background_tasks: BackgroundTasks):
    from app.main import scheduled_scrape

    background_tasks.add_task(scheduled_scrape)
    return {"message": "Full pipeline started (scrape + predictions + rankings + previews)"}
