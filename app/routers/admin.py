import contextvars
import json
import logging
import re
import threading
from datetime import date, datetime

from fastapi import APIRouter, BackgroundTasks, Body, Depends, Header, HTTPException, Query, Request
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.shared import AdminActionRun, ModelRun, OddsSnapshot, Prediction
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFightStats
from app.services import audit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request context: who is acting, and with what parameters
# ---------------------------------------------------------------------------

#: Set per request by admin_request_context. An async dependency runs in the request's
#: own context, and FastAPI copies that context into the threadpool for sync endpoints,
#: so _start_task can read it without every endpoint threading a Request through.
_request_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("admin_request_ctx", default=None)


async def admin_request_context(request: Request):
    # Free text typed into the dashboard's connect screen. Everyone shares one admin key,
    # so this is attribution for the audit log, not authentication.
    actor = (request.headers.get("x-admin-actor") or "").strip()[:120] or "unknown"
    _request_ctx.set({"actor": actor, "params": dict(request.query_params), "path": request.url.path})


router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(admin_request_context)])


def _actor() -> str:
    ctx = _request_ctx.get()
    return ctx["actor"] if ctx else "unknown"


# ---------------------------------------------------------------------------
# Task tracking (audit log)
# ---------------------------------------------------------------------------

# Labels with a task currently queued or running, guarded by _running_lock.
# Several actions (e.g. Generate Predictions) clear a table before rebuilding it,
# so two concurrent runs of the same label will corrupt or truncate the result.
_running_labels: set[str] = set()
_running_lock = threading.Lock()


def record_run(label: str, status: str, error: str | None = None, started: datetime | None = None):
    """Record a finished step. Called by pipeline steps whether they were started from
    the dashboard or by the scheduler; audit attaches it to whichever run is active."""
    audit.record_step(label, status, error, started)


def _tracked_task(run_id: int | None, label: str, actor: str, fn, *args, **kwargs):
    """Run fn under an already-opened audit row, storing a dict/str return as its summary."""
    try:
        with audit.track(label, source="manual", actor=actor, run_id=run_id) as frame:
            result = fn(*args, **kwargs)
            if isinstance(result, (dict, str, list)):
                frame["summary"] = result
    except Exception:
        log.exception(f"Task {label} failed")  # already recorded as an error by track()
    finally:
        with _running_lock:
            _running_labels.discard(label)
        # Every admin action writes something a cached view reads (odds, predictions,
        # fights, rankings), and a failed one may have written half of it.
        refresh_cached_views()


def refresh_cached_views():
    """Drop the API response cache and rebuild the hot views, so a job's writes show
    up on the next page load instead of after the cache TTL."""
    from app.services import response_cache
    response_cache.invalidate()
    response_cache.warm()


def _start_task(
    background_tasks: BackgroundTasks, label: str, fn, *args, target: str | None = None, **kwargs,
) -> dict:
    # Claim the label here rather than in _tracked_task: background tasks only run
    # after the response is sent, so two near-simultaneous requests would both get
    # past a check made inside the worker.
    with _running_lock:
        if label in _running_labels:
            raise HTTPException(409, f"{label} is already running")
        _running_labels.add(label)

    ctx = _request_ctx.get() or {}
    actor = ctx.get("actor", "unknown")
    params = {"endpoint": ctx.get("path"), **(ctx.get("params") or {})}
    try:
        # Opened now, not when the worker starts, so the caller has an id to poll and a
        # request that is accepted but never runs still leaves a trace.
        run_id = audit.start_run(label, "manual", actor, params, target)
        background_tasks.add_task(_tracked_task, run_id, label, actor, fn, *args, **kwargs)
    except Exception:
        with _running_lock:
            _running_labels.discard(label)
        raise
    return {"message": f"{label} started in background", "task_id": run_id}


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


#: What each in-app job does, keyed by APScheduler job id, and the audit label it runs under.
_JOB_INFO = {
    "ufc_scrape": ("Full Pipeline", "Nightly pipeline",
                   "Scrapes results for any card newer than the last one with results, then (only if a "
                   "card landed) rebuilds derived stats, career stats, rankings, rank history and "
                   "similarity. Always regenerates winner predictions (four-model ensemble blended "
                   "with the market), method predictions and fight previews."),
    "bovada_scrape": ("Bovada Odds", "Bovada method odds", "Method-of-victory odds for upcoming fights."),
    "prediction_markets": ("Prediction Markets", "Prediction markets",
                           "Kalshi and Polymarket quotes and price history for open fights."),
    "reconcile_fights": ("Reconcile Fights", "Reconcile fights",
                         "Records bouts that have been pulled from their card (ufc_cancelled_bouts), "
                         "then removes them and what depended on them."),
}

#: GitHub Actions workflows in the backend repo that write to the same database. They run
#: outside this process, so their schedule is read from here rather than from APScheduler.
#: Keep in sync with .github/workflows/*.yml.
_GITHUB_WORKFLOWS = [
    ("Post-Event Update", "post-event.yml", "0 6 * * *",
     "If a card finished in the last few days without results: scrape results, look up newly "
     "booked fighters on Sherdog, rebuild stats, rankings, similarity and predictions, fetch "
     "odds, settle forward-test rows against the closing line."),
    ("Odds Refresh", "post-event.yml", "0 6 * * *",
     "Tuesdays and Fridays only: live US bookmaker odds (The Odds API) and a Sherdog lookup "
     "for newly booked fighters."),
    ("Fight-Day Odds", "post-event.yml", "0 21 * * 6",
     "Saturdays: US bookmaker odds snapshot close to the card (backup closing line)."),
    ("Line Watcher", "line-watcher.yml", "17 */2 * * *",
     "Records new opening lines from BestFightOdds. Every ~6h also refreshes ufcstats bookings "
     "and records cancelled bouts. When a fight is priced for the first time: Sherdog lookup "
     "for new fighters, Glicko + predictions refresh, and the forward test logs the fight at "
     "its opening line."),
    ("Prediction Markets", "prediction-markets.yml", "0 */2 * * *",
     "Kalshi/Polymarket refresh, then removes cancelled fights."),
]


def _last_runs_by_action(db: Session, source: str | None = None) -> dict[str, dict]:
    """Latest run per action name, optionally from one source only: the backend scheduler
    and a GitHub workflow can share a name ("Prediction Markets") but are separate jobs."""
    q = db.query(AdminActionRun.action, func.max(AdminActionRun.id).label("id"))
    if source:
        q = q.filter(AdminActionRun.source == source)
    latest = q.group_by(AdminActionRun.action).subquery()
    rows = db.query(AdminActionRun).join(latest, AdminActionRun.id == latest.c.id).all()
    return {r.action: _run_dict(r) for r in rows}


@router.get("/schedule", dependencies=[Depends(require_admin_key)])
def get_schedule(db: Session = Depends(get_db)):
    """Everything that will run on its own, soonest first, with how its last run went."""
    from apscheduler.triggers.cron import CronTrigger
    from datetime import timezone
    from app.main import scheduler

    last_scheduled = _last_runs_by_action(db, "scheduled")
    last_github = _last_runs_by_action(db, "github")
    now = datetime.now(timezone.utc)
    items = []
    for job in scheduler.get_jobs():
        label, name, desc = _JOB_INFO.get(job.id, (job.name or job.id, job.name or job.id, ""))
        items.append({
            "id": job.id, "name": name, "action": label, "runner": "backend",
            "description": desc, "trigger": str(job.trigger),
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "last_run": last_scheduled.get(label),
        })
    for name, workflow, cron, desc in _GITHUB_WORKFLOWS:
        nxt = CronTrigger.from_crontab(cron, timezone="UTC").get_next_fire_time(None, now)
        items.append({
            "id": f"github:{name}", "name": name, "action": name, "runner": "github",
            "description": desc, "trigger": f"cron '{cron}' (UTC) · {workflow}",
            "next_run": nxt.isoformat() if nxt else None,
            "last_run": last_github.get(name),
        })
    items.sort(key=lambda i: i["next_run"] or "9999")
    return {"items": items, "now": now.isoformat()}


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _loads(v):
    if v is None:
        return None
    try:
        return json.loads(v)
    except ValueError:
        return v


def _run_dict(r: AdminActionRun) -> dict:
    return {
        "id": r.id, "parent_id": r.parent_id, "action": r.action, "source": r.source,
        "actor": r.actor, "status": r.status, "target": r.target,
        "params": _loads(r.params), "summary": _loads(r.summary), "error": r.error,
        # Stored as naive UTC; the suffix lets the browser convert to local time.
        "started_at": r.started_at.isoformat() + "Z" if r.started_at else None,
        "finished_at": r.finished_at.isoformat() + "Z" if r.finished_at else None,
    }


@router.get("/runs", dependencies=[Depends(require_admin_key)])
def list_runs(
    status: str | None = None,
    source: str | None = None,
    action: str | None = None,
    target: str | None = None,
    before_id: int | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Top-level runs, newest first, each with its steps. Page with before_id."""
    q = db.query(AdminActionRun).filter(AdminActionRun.parent_id.is_(None))
    if status:
        q = q.filter(AdminActionRun.status == status)
    if source:
        q = q.filter(AdminActionRun.source == source)
    if action:
        q = q.filter(AdminActionRun.action.ilike(f"%{action}%"))
    if target:
        q = q.filter(AdminActionRun.target == target)
    if before_id:
        q = q.filter(AdminActionRun.id < before_id)
    runs = q.order_by(AdminActionRun.id.desc()).limit(limit).all()

    steps: dict[int, list] = {}
    if runs:
        for c in (
            db.query(AdminActionRun)
            .filter(AdminActionRun.parent_id.in_([r.id for r in runs]))
            .order_by(AdminActionRun.id)
            .all()
        ):
            steps.setdefault(c.parent_id, []).append(_run_dict(c))
    return {
        "runs": [{**_run_dict(r), "steps": steps.get(r.id, [])} for r in runs],
        "has_more": len(runs) == limit,
    }


@router.get("/runs/{run_id}", dependencies=[Depends(require_admin_key)])
def get_run(run_id: int, db: Session = Depends(get_db)):
    r = db.get(AdminActionRun, run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    steps = db.query(AdminActionRun).filter(AdminActionRun.parent_id == run_id).order_by(AdminActionRun.id).all()
    return {**_run_dict(r), "steps": [_run_dict(c) for c in steps]}


@router.get("/task-status/{task_id}", dependencies=[Depends(require_admin_key)])
def get_task_status(task_id: int, db: Session = Depends(get_db)):
    """Kept for older dashboard builds; /runs/{id} carries the same plus the steps."""
    return get_run(task_id, db)


@router.get("/last-runs", dependencies=[Depends(require_admin_key)])
def get_last_runs(db: Session = Depends(get_db)):
    return _last_runs_by_action(db)


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
    from app.services.ufc.scraper import run_scrape

    return _start_task(background_tasks, f"Scrape ({mode})", run_scrape, mode=mode)


@router.post("/scrape-upcoming", dependencies=[Depends(require_admin_key)])
def trigger_upcoming_scrape(background_tasks: BackgroundTasks):
    from app.services.ufc.scraper import scrape_upcoming

    return _start_task(background_tasks, "Scrape Upcoming", scrape_upcoming)


@router.post("/scrape-recent", dependencies=[Depends(require_admin_key)])
def trigger_recent_update(background_tasks: BackgroundTasks):
    from app.services.ufc.scraper import run_recent_update

    return _start_task(background_tasks, "Recent Update", run_recent_update)


@router.post("/scrape-last-event", dependencies=[Depends(require_admin_key)])
def trigger_scrape_last_event(background_tasks: BackgroundTasks):
    from app.services.ufc.scraper import run_scrape_last_event

    return _start_task(background_tasks, "Scrape Last Event", run_scrape_last_event)


@router.post("/scrape-live-odds", dependencies=[Depends(require_admin_key)])
def trigger_live_odds_scrape(background_tasks: BackgroundTasks):
    from app.services.ufc.odds_scraper import run_live_odds_scrape

    return _start_task(background_tasks, "Live Odds", run_live_odds_scrape)


@router.post("/scrape-historical-odds", dependencies=[Depends(require_admin_key)])
def trigger_historical_odds_scrape(
    background_tasks: BackgroundTasks,
    since: int = Query(default=2022),
):
    from app.services.ufc.odds_scraper import run_odds_scrape

    return _start_task(background_tasks, f"Historical Odds (since {since})", run_odds_scrape, since)


@router.post("/scrape-bovada", dependencies=[Depends(require_admin_key)])
def trigger_bovada_scrape(background_tasks: BackgroundTasks):
    from app.services.ufc.bovada_scraper import scrape_bovada_method_odds

    return _start_task(background_tasks, "Bovada Odds", scrape_bovada_method_odds)


@router.post("/scrape-prediction-markets", dependencies=[Depends(require_admin_key)])
def trigger_prediction_markets(
    background_tasks: BackgroundTasks,
    venue: str = Query(default="both", pattern="^(kalshi|polymarket|both)$"),
    backfill: bool = Query(default=False),
    curves: str = Query(default="all", pattern="^(moneyline|all)$"),
):
    """Refresh Kalshi/Polymarket quotes and price curves.

    `backfill=true` walks every event the venues have ever listed, including settled ones, and
    rebuilds history from their candlestick endpoints. Safe to re-run: history writes are
    ON CONFLICT DO NOTHING on (market_id, captured_at).
    """
    from app.services.ufc.prediction_markets import run_backfill, run_live

    if backfill:
        return _start_task(background_tasks, f"Prediction Markets backfill ({venue})",
                           run_backfill, venue, curves)
    return _start_task(background_tasks, f"Prediction Markets ({venue})", run_live, venue, curves)


@router.post("/reconcile-fights", dependencies=[Depends(require_admin_key)])
def trigger_reconcile(
    background_tasks: BackgroundTasks,
    apply: bool = Query(default=False),
    previews: bool = Query(default=True),
):
    """Remove bouts that are no longer on their ufcstats event page.

    Defaults to a report-only pass -- `apply=true` is required to delete. Deletion is skipped for
    any fight with a recorded winner and for any event whose page fails to fetch, so a bad scrape
    cannot empty a card.
    """
    from app.services.ufc.reconcile_service import run_reconcile

    label = "Reconcile Fights" + ("" if apply else " (dry run)")
    return _start_task(background_tasks, label, run_reconcile, 30, 120, not apply, True, previews)


@router.post("/scrape-profiles", dependencies=[Depends(require_admin_key)])
def trigger_profile_scrape(
    background_tasks: BackgroundTasks,
    images: bool = Query(default=True),
    recent_days: int = Query(default=60, ge=0,
                             description="Fighters with a bout in the last N days or booked "
                                         "on an upcoming card. 0 = the whole roster (slow)."),
):
    from app.services.ufc.ufc_profile_scraper import ingest_fighter_profiles, scrape_profiles

    if recent_days == 0:
        # Whole roster: every fighter still missing a bio or photo, then cache photos.
        def _all():
            result = {"all_fighters": scrape_profiles()}
            if images:
                from scripts.cache_fighter_images import run as cache_images
                result["images"] = cache_images()
            return result
        return _start_task(background_tasks, "Fighter Profiles", _all)
    return _start_task(background_tasks, "Fighter Profiles", ingest_fighter_profiles,
                       recent_days=recent_days, images=images)


# ---------------------------------------------------------------------------
# Model training & predictions
# ---------------------------------------------------------------------------

@router.post("/train-model", dependencies=[Depends(require_admin_key)])
def trigger_model_training(background_tasks: BackgroundTasks):
    from app.services.ufc.model import run as run_training

    return _start_task(background_tasks, "Train Winner Model", run_training)


@router.post("/train-method-model", dependencies=[Depends(require_admin_key)])
def trigger_method_model_training(background_tasks: BackgroundTasks):
    from app.services.ufc.method_model import run as run_method_training

    return _start_task(background_tasks, "Train Method Model", run_method_training)


@router.post("/generate-predictions", dependencies=[Depends(require_admin_key)])
def trigger_predictions(background_tasks: BackgroundTasks):
    from app.services.ufc.model import generate_predictions

    return _start_task(background_tasks, "Generate Predictions", generate_predictions)


@router.post("/generate-method-predictions", dependencies=[Depends(require_admin_key)])
def trigger_method_predictions(background_tasks: BackgroundTasks):
    from app.services.ufc.method_model import generate_method_predictions

    return _start_task(background_tasks, "Method Predictions", generate_method_predictions)


@router.post("/generate-glicko", dependencies=[Depends(require_admin_key)])
def trigger_glicko(background_tasks: BackgroundTasks):
    from app.services.ufc.glicko_service import compute_and_save_snapshots
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
    from app.database import SessionLocal
    from app.services.ufc.ranking_publisher import publish_rankings

    def _run_all():
        db = SessionLocal()
        try:
            publish_rankings(db)
        finally:
            db.close()

    return _start_task(background_tasks, "Glicko Ratings + Rankings", _run_all)


@router.post("/generate-derived-stats", dependencies=[Depends(require_admin_key)])
def trigger_derived_stats(background_tasks: BackgroundTasks):
    from app.services.ufc.fight_stats_derived_service import compute_all_derived_stats

    return _start_task(background_tasks, "Derived Fight Stats", compute_all_derived_stats)


@router.post("/generate-career-stats", dependencies=[Depends(require_admin_key)])
def trigger_career_stats(background_tasks: BackgroundTasks):
    from app.services.ufc.career_stats_service import compute_all_career_stats

    return _start_task(background_tasks, "Career Stats", compute_all_career_stats)


@router.post("/generate-similarity", dependencies=[Depends(require_admin_key)])
def trigger_similarity(
    background_tasks: BackgroundTasks,
    refit: bool = Query(
        default=False,
        description="Refit the frozen style space instead of transforming through it. "
                    "Changes every stored similarity score and invalidates comparison "
                    "with previous runs — rerun style_eval afterwards.",
    ),
):
    from app.database import SessionLocal
    from app.services.ufc.style_service import compute_and_save_similarity

    def _run():
        db = SessionLocal()
        try:
            compute_and_save_similarity(db, refit=refit)
        finally:
            db.close()

    return _start_task(background_tasks, "Fighter Similarity", _run)


@router.post("/refresh-after-event", dependencies=[Depends(require_admin_key)])
def trigger_refresh_after_event(background_tasks: BackgroundTasks):
    """Rerun the whole stats chain: derived -> career -> rankings -> similarity.

    The same thing scheduled_scrape runs when a new card lands, exposed for the morning
    after an event when you have re-scraped results by hand and want everything
    downstream rebuilt in dependency order.
    """
    from app.main import refresh_after_event

    return _start_task(background_tasks, "Refresh After Event", refresh_after_event)


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

@router.post("/generate-preview/{fight_id}", dependencies=[Depends(require_admin_key)])
def trigger_preview_generation(
    fight_id: int,
    force: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    from app.services.ufc.preview_service import generate_preview

    with audit.track("Generate Preview", source="manual", actor=_actor(),
                     params={"fight_id": fight_id, "force": force}, target=f"fight:{fight_id}") as frame:
        preview = generate_preview(fight_id, db, force=force)
        if not preview:
            # Not an exception inside generate_preview, but still a failed action.
            raise HTTPException(502, "Preview generation failed -- check API key and fight data")
        frame["summary"] = {"fight_id": fight_id}
    return {"message": "Preview generated", "fight_id": fight_id}


@router.get("/pending-previews", dependencies=[Depends(require_admin_key)])
def list_pending_previews(
    force: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    """How many upcoming fights would `generate-all-previews` write, without writing."""
    from app.services.ufc.preview_service import pending_preview_fight_ids

    ids = pending_preview_fight_ids(db, force=force)
    return {"count": len(ids), "fight_ids": ids}


@router.post("/generate-all-previews", dependencies=[Depends(require_admin_key)])
def trigger_all_previews(
    background_tasks: BackgroundTasks,
    force: bool = Query(default=False),
    workers: int | None = Query(default=None, ge=1, le=16),
):
    from app.services.ufc.preview_service import generate_all_upcoming_previews

    return _start_task(
        background_tasks, "Generate All Previews", generate_all_upcoming_previews,
        force=force, workers=workers,
    )


# ---------------------------------------------------------------------------
# Full pipeline (same as scheduled job)
# ---------------------------------------------------------------------------

@router.post("/run-full-pipeline", dependencies=[Depends(require_admin_key)])
def trigger_full_pipeline(background_tasks: BackgroundTasks):
    from app.main import scheduled_scrape

    return _start_task(background_tasks, "Full Pipeline", scheduled_scrape)


# ---------------------------------------------------------------------------
# Event results (manual "get results & stats")
# ---------------------------------------------------------------------------

@router.get("/events/recent", dependencies=[Depends(require_admin_key)])
def recent_events(limit: int = Query(default=8, ge=1, le=50), db: Session = Depends(get_db)):
    """Past events, newest first, with how many of their fights have results."""
    events = (
        db.query(UFCEvent).filter(UFCEvent.date <= date.today())
        .order_by(UFCEvent.date.desc()).limit(limit).all()
    )
    ids = [e.id for e in events]
    counts: dict[int, dict] = {i: {"fights": 0, "with_result": 0} for i in ids}
    if ids:
        for event_id, total, finished in (
            db.query(
                UFCFight.event_id, func.count(UFCFight.id),
                # A finished bout always has a method; draws and no-contests have no winner.
                func.count(UFCFight.method),
            ).filter(UFCFight.event_id.in_(ids)).group_by(UFCFight.event_id).all()
        ):
            counts[event_id] = {"fights": total, "with_result": finished}

    last = {
        r.target: _run_dict(r) for r in (
            db.query(AdminActionRun)
            .filter(AdminActionRun.target.in_([f"event:{i}" for i in ids]), AdminActionRun.parent_id.is_(None))
            .order_by(AdminActionRun.id)
            .all()
        )
    } if ids else {}
    return [
        {
            "id": str(e.id), "name": e.name, "date": str(e.date), "location": e.location,
            **counts[e.id],
            "missing_results": counts[e.id]["fights"] > counts[e.id]["with_result"],
            "last_fetch": last.get(f"event:{e.id}"),
        }
        for e in events
    ]


@router.post("/events/{event_id}/fetch-results", dependencies=[Depends(require_admin_key)])
def trigger_event_results(event_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Scrape one past event's results and stats, then rebuild rankings and predictions."""
    from app.main import refresh_event_results

    event = db.get(UFCEvent, event_id)
    if not event:
        raise HTTPException(404, "Event not found")
    if event.date > date.today():
        raise HTTPException(400, f"{event.name} is on {event.date} and has not happened yet")
    # One label for every event: the rebuild steps are global (rankings, predictions),
    # so two events fetched at once would race on the same tables.
    return _start_task(
        background_tasks, "Get Event Results", refresh_event_results, event_id,
        target=f"event:{event_id}",
    )


# ---------------------------------------------------------------------------
# Fighter bio edits
# ---------------------------------------------------------------------------

#: Fields the dashboard may edit, with a validator returning the value to store.
def _text(max_len: int):
    def v(value):
        if value is None:
            return None
        value = str(value).strip()
        if len(value) > max_len:
            raise ValueError(f"must be at most {max_len} characters")
        return value or None
    return v


def _country_code(value):
    if value in (None, ""):
        return None
    value = str(value).strip().upper()
    # ISO 3166-1 alpha-2, or a UK home-nation subdivision (GB-ENG, GB-SCT, GB-WLS, GB-NIR).
    if not re.fullmatch(r"[A-Z]{2}(-[A-Z]{3})?", value):
        raise ValueError("must be a 2-letter ISO code like US or BR, or GB-ENG / GB-SCT / GB-WLS / GB-NIR")
    return value


def _date(value):
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError("must be a date as YYYY-MM-DD")


EDITABLE_FIGHTER_FIELDS = {
    "nickname": _text(200),
    "country_code": _country_code,
    "birthplace": _text(200),
    "birth_country": _text(100),
    "fighting_style": _text(100),
    "trains_at": _text(200),
    "dob": _date,
    "status": _text(20),
}


def _fighter_dict(f: UFCFighter) -> dict:
    return {
        "id": str(f.id), "first_name": f.first_name, "last_name": f.last_name,
        "record": f"{f.wins}-{f.losses}-{f.draws}", "image_url": f.image_url,
        "fields": {k: (str(getattr(f, k)) if getattr(f, k) is not None else None) for k in EDITABLE_FIGHTER_FIELDS},
        "locked_fields": sorted(f.locked()),
    }


@router.get("/fighters", dependencies=[Depends(require_admin_key)])
def search_fighters(q: str = Query(min_length=2), db: Session = Depends(get_db)):
    like = f"%{q.strip()}%"
    full = UFCFighter.first_name + " " + UFCFighter.last_name
    fighters = (
        db.query(UFCFighter)
        .filter((full.ilike(like)) | (UFCFighter.nickname.ilike(like)))
        .order_by(UFCFighter.last_name, UFCFighter.first_name)
        .limit(25)
        .all()
    )
    return [
        {"id": str(f.id), "name": f"{f.first_name} {f.last_name}", "nickname": f.nickname,
         "country_code": f.country_code, "record": f"{f.wins}-{f.losses}-{f.draws}"}
        for f in fighters
    ]


@router.get("/fighters/{fighter_id}", dependencies=[Depends(require_admin_key)])
def get_fighter(fighter_id: int, db: Session = Depends(get_db)):
    f = db.get(UFCFighter, fighter_id)
    if not f:
        raise HTTPException(404, "Fighter not found")
    history = (
        db.query(AdminActionRun)
        .filter(AdminActionRun.target == f"fighter:{fighter_id}")
        .order_by(AdminActionRun.id.desc()).limit(50).all()
    )
    return {**_fighter_dict(f), "history": [_run_dict(r) for r in history]}


@router.patch("/fighters/{fighter_id}", dependencies=[Depends(require_admin_key)])
def update_fighter(
    fighter_id: int,
    fields: dict = Body(default_factory=dict, embed=True),
    unlock: list[str] = Body(default_factory=list, embed=True),
    db: Session = Depends(get_db),
):
    """Edit bio fields. Every edited field is locked against the scrapers; `unlock`
    hands a field back to them (its value stays until the next scrape replaces it)."""
    f = db.get(UFCFighter, fighter_id)
    if not f:
        raise HTTPException(404, "Fighter not found")

    unknown = (set(fields) | set(unlock)) - set(EDITABLE_FIGHTER_FIELDS)
    if unknown:
        raise HTTPException(400, f"Not editable: {', '.join(sorted(unknown))}")

    changes: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for key, raw in fields.items():
        try:
            new = EDITABLE_FIGHTER_FIELDS[key](raw)
        except ValueError as e:
            errors[key] = str(e)
            continue
        old = getattr(f, key)
        if old != new:
            changes[key] = {"from": str(old) if old is not None else None,
                            "to": str(new) if new is not None else None}
            setattr(f, key, new)
    if errors:
        db.rollback()
        raise HTTPException(422, {"message": "Some fields are invalid", "fields": errors})

    locked_before = f.locked()
    locked = (locked_before | set(changes)) - set(unlock)
    if not changes and locked == locked_before:
        return {**_fighter_dict(f), "changed": {}}
    f.locked_fields = json.dumps(sorted(locked)) if locked else None
    db.commit()

    name = f"{f.first_name} {f.last_name}"
    audit.record_event(
        "Edit Fighter", "manual", _actor(), target=f"fighter:{fighter_id}",
        params={"fighter": name},
        summary={"changes": changes,
                 "locked": sorted(locked - locked_before), "unlocked": sorted(locked_before - locked)},
    )
    # Fighter pages and rankings are served from the response cache; rebuild it off
    # the request thread so the save returns immediately.
    threading.Thread(target=refresh_cached_views, name="cache-refresh", daemon=True).start()
    db.refresh(f)
    return {**_fighter_dict(f), "changed": changes}

