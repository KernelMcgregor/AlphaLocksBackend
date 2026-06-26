import logging
import re
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Body, Depends, Header, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.shared import ModelRun, OddsSnapshot, Prediction
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFightStats

log = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# ---------------------------------------------------------------------------
# Task tracking (in-memory)
# ---------------------------------------------------------------------------

_tasks: dict[str, dict] = {}
_last_runs: dict[str, dict] = {}


def record_run(label: str, status: str, error: str | None = None):
    """Record when an action last ran. Called by both manual triggers and scheduled jobs."""
    _last_runs[label] = {
        "status": status,
        "finished": datetime.utcnow().isoformat(),
        "error": error,
    }


def _tracked_task(task_id: str, label: str, fn, *args, **kwargs):
    """Wrapper that runs fn and records completion/failure in _tasks."""
    _tasks[task_id] = {"label": label, "status": "running", "started": datetime.utcnow().isoformat()}
    try:
        fn(*args, **kwargs)
        _tasks[task_id]["status"] = "done"
        _tasks[task_id]["finished"] = datetime.utcnow().isoformat()
        record_run(label, "done")
    except Exception as e:
        log.exception(f"Task {label} failed")
        _tasks[task_id]["status"] = "error"
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["finished"] = datetime.utcnow().isoformat()
        record_run(label, "error", str(e))


def _start_task(background_tasks: BackgroundTasks, label: str, fn, *args, **kwargs) -> dict:
    task_id = uuid.uuid4().hex[:12]
    background_tasks.add_task(_tracked_task, task_id, label, fn, *args, **kwargs)
    return {"message": f"{label} started in background", "task_id": task_id}


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
# Task status
# ---------------------------------------------------------------------------

@router.get("/task-status/{task_id}", dependencies=[Depends(require_admin_key)])
def get_task_status(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


@router.get("/last-runs", dependencies=[Depends(require_admin_key)])
def get_last_runs():
    return _last_runs


# ---------------------------------------------------------------------------
# Database query (read-only)
# ---------------------------------------------------------------------------

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE,
)

MAX_QUERY_ROWS = 200


@router.post("/query", dependencies=[Depends(require_admin_key)])
def run_query(sql: str = Body(..., embed=True), db: Session = Depends(get_db)):
    """Run a read-only SQL query against the production database."""
    stripped = sql.strip().rstrip(";")
    if _FORBIDDEN_KEYWORDS.search(stripped):
        raise HTTPException(400, "Only SELECT queries are allowed")

    try:
        result = db.execute(text(stripped))
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.fetchmany(MAX_QUERY_ROWS)]
        return {"columns": columns, "rows": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(400, f"Query error: {e}")


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

@router.post("/scrape", dependencies=[Depends(require_admin_key)])
def trigger_scrape(
    background_tasks: BackgroundTasks,
    mode: str = Query(default="full", pattern="^(full|update)$"),
):
    from app.services.scraper import run_scrape

    return _start_task(background_tasks, f"Scrape ({mode})", run_scrape, mode=mode)


@router.post("/scrape-upcoming", dependencies=[Depends(require_admin_key)])
def trigger_upcoming_scrape(background_tasks: BackgroundTasks):
    from app.services.scraper import scrape_upcoming

    return _start_task(background_tasks, "Scrape Upcoming", scrape_upcoming)


@router.post("/scrape-recent", dependencies=[Depends(require_admin_key)])
def trigger_recent_update(background_tasks: BackgroundTasks):
    from app.services.scraper import run_recent_update

    return _start_task(background_tasks, "Recent Update", run_recent_update)


@router.post("/scrape-last-event", dependencies=[Depends(require_admin_key)])
def trigger_scrape_last_event(background_tasks: BackgroundTasks):
    from app.services.scraper import run_scrape_last_event

    return _start_task(background_tasks, "Scrape Last Event", run_scrape_last_event)


@router.post("/scrape-live-odds", dependencies=[Depends(require_admin_key)])
def trigger_live_odds_scrape(background_tasks: BackgroundTasks):
    from app.services.odds_scraper import run_live_odds_scrape

    return _start_task(background_tasks, "Live Odds", run_live_odds_scrape)


@router.post("/scrape-historical-odds", dependencies=[Depends(require_admin_key)])
def trigger_historical_odds_scrape(
    background_tasks: BackgroundTasks,
    since: int = Query(default=2022),
):
    from app.services.odds_scraper import run_odds_scrape

    return _start_task(background_tasks, f"Historical Odds (since {since})", run_odds_scrape, since)


@router.post("/scrape-bovada", dependencies=[Depends(require_admin_key)])
def trigger_bovada_scrape(background_tasks: BackgroundTasks):
    from app.services.bovada_scraper import scrape_bovada_method_odds

    return _start_task(background_tasks, "Bovada Odds", scrape_bovada_method_odds)


@router.post("/scrape-profiles", dependencies=[Depends(require_admin_key)])
def trigger_profile_scrape(
    background_tasks: BackgroundTasks,
    images: bool = Query(default=True),
):
    from app.services.ufc_profile_scraper import run as run_profiles

    return _start_task(background_tasks, "Fighter Profiles", run_profiles, images=images)


# ---------------------------------------------------------------------------
# Model training & predictions
# ---------------------------------------------------------------------------

@router.post("/train-model", dependencies=[Depends(require_admin_key)])
def trigger_model_training(background_tasks: BackgroundTasks):
    from app.services.model import run as run_training

    return _start_task(background_tasks, "Train Winner Model", run_training)


@router.post("/train-method-model", dependencies=[Depends(require_admin_key)])
def trigger_method_model_training(background_tasks: BackgroundTasks):
    from app.services.method_model import run as run_method_training

    return _start_task(background_tasks, "Train Method Model", run_method_training)


@router.post("/generate-predictions", dependencies=[Depends(require_admin_key)])
def trigger_predictions(background_tasks: BackgroundTasks):
    from app.services.model import generate_predictions

    return _start_task(background_tasks, "Generate Predictions", generate_predictions)


@router.post("/generate-method-predictions", dependencies=[Depends(require_admin_key)])
def trigger_method_predictions(background_tasks: BackgroundTasks):
    from app.services.method_model import generate_method_predictions

    return _start_task(background_tasks, "Method Predictions", generate_method_predictions)


@router.post("/generate-glicko", dependencies=[Depends(require_admin_key)])
def trigger_glicko(background_tasks: BackgroundTasks):
    from app.services.glicko_service import compute_and_save_snapshots
    from app.database import SessionLocal

    def _run():
        db = SessionLocal()
        try:
            compute_and_save_snapshots(db)
        finally:
            db.close()

    return _start_task(background_tasks, "Generate Glicko Ratings", _run)


@router.post("/generate-rankings", dependencies=[Depends(require_admin_key)])
def trigger_rankings(background_tasks: BackgroundTasks):
    from app.services.ranking_service import generate_rankings
    from app.services.points_ranking_service import generate_rankings as generate_points_rankings

    def _run_all():
        generate_rankings()
        generate_points_rankings()

    return _start_task(background_tasks, "Generate Rankings", _run_all)


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

    return _start_task(background_tasks, "Generate All Previews", generate_all_upcoming_previews, force=force)


# ---------------------------------------------------------------------------
# Full pipeline (same as scheduled job)
# ---------------------------------------------------------------------------

@router.post("/run-full-pipeline", dependencies=[Depends(require_admin_key)])
def trigger_full_pipeline(background_tasks: BackgroundTasks):
    from app.main import scheduled_scrape

    return _start_task(background_tasks, "Full Pipeline", scheduled_scrape)
