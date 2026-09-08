import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.models import *  # noqa: F401, F403 — ensure all models are registered
from app.models.ufc import FIGHTER_BIO_COLS, GLICKO_META_COLS
from app.routers import admin, predictions, ufc

scheduler = BackgroundScheduler()


def run_migrations():
    migrations_dir = Path(__file__).parent / "migrations"
    sql_files = sorted(migrations_dir.glob("*.sql"))

    if settings.DATABASE_URL.startswith("sqlite"):
        db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        for sql_file in sql_files:
            try:
                conn.executescript(sql_file.read_text())
            except Exception:
                pass  # Columns may already exist from prior runs
        # 008: ufc.com bio fields on ufc_fighters (idempotent)
        fighter_existing = {row[1] for row in conn.execute("PRAGMA table_info(ufc_fighters)").fetchall()}
        for col, ddl in FIGHTER_BIO_COLS:
            if col not in fighter_existing:
                conn.execute(f"ALTER TABLE ufc_fighters ADD COLUMN {col} {ddl}")
        # Add derived columns to ufc_fight_stats (idempotent)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(ufc_fight_stats)").fetchall()}
        derived_cols = [
            "fight_time_min", "est_standing_min", "est_ground_min",
            "slpm", "sapm", "sl_diff", "sig_acc", "sig_def", "tslpm",
            "head_pct", "head_pm", "head_acc", "head_abs_pct", "head_abs_pm", "head_def",
            "body_pct", "body_pm", "body_acc", "body_abs_pct", "body_abs_pm", "body_def",
            "leg_pct", "leg_pm", "leg_acc", "leg_abs_pct", "leg_abs_pm", "leg_def",
            "dist_pct", "dist_pm", "dist_acc", "dist_abs_pct", "dist_abs_pm", "dist_def",
            "clinch_pct", "clinch_pm", "clinch_acc", "clinch_abs_pct", "clinch_abs_pm", "clinch_def",
            "ground_pct", "ground_pm", "ground_acc", "ground_abs_pct", "ground_abs_pm", "ground_def",
            "gnp15g", "gnp_abs15g",
            "kd15", "kd15s", "kd_abs15", "kd_abs15s",
            "td15", "td15s", "td_acc", "td_abs15", "td_abs15s", "td_def",
            "ctrl15", "ctrl15g", "ctrl_abs15", "ctrl_abs15g",
            "sub_att15", "sub_att15g", "sub_abs15", "sub_abs15g",
            "rev15", "rev_abs15",
        ]
        for col in derived_cols:
            if col not in existing:
                conn.execute(f"ALTER TABLE ufc_fight_stats ADD COLUMN {col} REAL")
        # 006: rating-confidence columns on ufc_glicko_snapshots (idempotent)
        snap_existing = {row[1] for row in conn.execute("PRAGMA table_info(ufc_glicko_snapshots)").fetchall()}
        for col in GLICKO_META_COLS:
            if col not in snap_existing:
                conn.execute(f"ALTER TABLE ufc_glicko_snapshots ADD COLUMN {col} REAL")
        conn.commit()
        conn.close()
    else:
        from sqlalchemy import text, inspect
        # Create ufc schema and move tables if needed
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ufc"))
            # Move existing tables from public to ufc schema
            ufc_tables = [
                "ufc_fighters", "ufc_events", "ufc_fights", "ufc_fight_stats",
                "ufc_fight_odds", "ufc_fight_predictions", "ufc_method_predictions",
                "ufc_fight_shap_values", "ufc_fight_previews", "ufc_method_odds",
                "ufc_distance_predictions",
                "ufc_fighter_career_stats",
            ]
            for table in ufc_tables:
                try:
                    conn.execute(text(f"ALTER TABLE public.{table} SET SCHEMA ufc"))
                except Exception:
                    pass  # Already moved or doesn't exist

        Base.metadata.create_all(bind=engine)
        # Add columns that create_all won't add to existing tables
        insp = inspect(engine)
        existing = {c["name"] for c in insp.get_columns("ufc_fighters", schema="ufc")}
        with engine.begin() as conn:
            if "country_code" not in existing:
                conn.execute(text("ALTER TABLE ufc.ufc_fighters ADD COLUMN country_code VARCHAR(2)"))
            if "image_url" not in existing:
                conn.execute(text("ALTER TABLE ufc.ufc_fighters ADD COLUMN image_url VARCHAR(500)"))
            # 008: ufc.com bio fields. All nullable with no default, so Postgres treats
            # each as a metadata-only change — no table rewrite on the live roster.
            for col, ddl in FIGHTER_BIO_COLS:
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE ufc.ufc_fighters ADD COLUMN {col} {ddl}"))

        # Add derived columns to ufc_fight_stats
        stats_existing = {c["name"] for c in insp.get_columns("ufc_fight_stats", schema="ufc")}
        derived_cols = [
            "fight_time_min", "est_standing_min", "est_ground_min",
            "slpm", "sapm", "sl_diff", "sig_acc", "sig_def", "tslpm",
            "head_pct", "head_pm", "head_acc", "head_abs_pct", "head_abs_pm", "head_def",
            "body_pct", "body_pm", "body_acc", "body_abs_pct", "body_abs_pm", "body_def",
            "leg_pct", "leg_pm", "leg_acc", "leg_abs_pct", "leg_abs_pm", "leg_def",
            "dist_pct", "dist_pm", "dist_acc", "dist_abs_pct", "dist_abs_pm", "dist_def",
            "clinch_pct", "clinch_pm", "clinch_acc", "clinch_abs_pct", "clinch_abs_pm", "clinch_def",
            "ground_pct", "ground_pm", "ground_acc", "ground_abs_pct", "ground_abs_pm", "ground_def",
            "gnp15g", "gnp_abs15g",
            "kd15", "kd15s", "kd_abs15", "kd_abs15s",
            "td15", "td15s", "td_acc", "td_abs15", "td_abs15s", "td_def",
            "ctrl15", "ctrl15g", "ctrl_abs15", "ctrl_abs15g",
            "sub_att15", "sub_att15g", "sub_abs15", "sub_abs15g",
            "rev15", "rev_abs15",
        ]
        missing = [c for c in derived_cols if c not in stats_existing]
        if missing:
            with engine.begin() as conn:
                for col in missing:
                    conn.execute(text(f"ALTER TABLE ufc.ufc_fight_stats ADD COLUMN {col} FLOAT"))

        # 006: rating-confidence columns on ufc_glicko_snapshots
        snap_existing = {c["name"] for c in insp.get_columns("ufc_glicko_snapshots", schema="ufc")}
        snap_missing = [c for c in GLICKO_META_COLS if c not in snap_existing]
        if snap_missing:
            with engine.begin() as conn:
                for col in snap_missing:
                    conn.execute(text(f"ALTER TABLE ufc.ufc_glicko_snapshots ADD COLUMN {col} FLOAT"))


#: The stats chain, in dependency order. Each entry is (label, "module:function").
#:
#: Ordering is not cosmetic. Derived per-fight columns feed the career aggregates; the
#: career aggregates and the published Glicko percentiles are the two halves of the style
#: vector; so similarity has to run last and derived has to run first.
#:
#: Two of these steps were previously unreachable from the running app:
#:   * compute_all_career_stats had NO call site anywhere in app/ — ufc_fighter_career_stats
#:     was only ever refreshed by someone running the module CLI by hand.
#:   * compute_all_derived_stats was called only from the tail of run_scrape() (the FULL
#:     scrape). run_recent_update(), which is what actually runs the day after an event,
#:     never called it, so a new fight's ~65 derived columns stayed empty until somebody
#:     triggered a full re-scrape.
#: publish_rankings therefore ran every night on stats that no longer matched the fights
#: in the database. Fixing that is a precondition for style similarity, which reads both
#: tables, but it was a live bug in the ranking pipeline independent of this feature.
POST_EVENT_CHAIN = [
    ("Derived Fight Stats", "app.services.ufc.fight_stats_derived_service:compute_all_derived_stats"),
    ("Career Stats", "app.services.ufc.career_stats_service:compute_all_career_stats"),
    ("Generate Rankings", "app.services.ufc.ranking_publisher:publish_rankings"),
    # Must follow Generate Rankings: publish_rankings overwrites ufc_fighter_rankings,
    # so without this the previous standings are lost and rank history never grows.
    ("Record Rank History", "app.services.ufc.rank_history_backfill:record_rank_history"),
    ("Fighter Similarity", "app.services.ufc.style_service:compute_and_save_similarity"),
]

#: Steps that take a db session as their first positional argument.
_CHAIN_NEEDS_DB = {"Generate Rankings", "Record Rank History", "Fighter Similarity"}


def refresh_after_event() -> dict[str, str]:
    """Recompute everything downstream of a completed card.

    Runs the whole chain even if a step fails, so one broken stage does not silently
    strand the three after it — the same reasoning behind the per-step guards in
    scheduled_scrape. Returns {label: "done" | "error: ..."} for the caller to report.
    """
    import importlib
    import logging
    log = logging.getLogger("refresh_after_event")
    from app.routers.admin import record_run
    from app.database import SessionLocal as ChainSession

    results: dict[str, str] = {}
    for label, target in POST_EVENT_CHAIN:
        module_name, func_name = target.split(":")
        try:
            func = getattr(importlib.import_module(module_name), func_name)
            if label in _CHAIN_NEEDS_DB:
                db = ChainSession()
                try:
                    func(db)
                finally:
                    db.close()
            else:
                func()
            log.info(f"{label}: done")
            record_run(label, "done")
            results[label] = "done"
        except Exception as e:
            log.exception(f"{label} failed")
            record_run(label, "error", str(e))
            results[label] = f"error: {e}"
    return results


def scheduled_scrape():
    import logging
    log = logging.getLogger("scheduled_scrape")
    from app.routers.admin import record_run

    from app.services.ufc.scraper import run_recent_update
    new_events = []
    try:
        new_events = run_recent_update()
        record_run("Recent Update", "done")
    except Exception as e:
        log.error(f"Recent update failed: {e}")
        record_run("Recent Update", "error", str(e))

    # Only recompute when a card actually landed. On the ~29 nights in 30 with no new
    # event the inputs are unchanged, and a Glicko replay over all of history to produce
    # byte-identical output is the most expensive no-op in the pipeline.
    if new_events:
        names = ", ".join(e["name"] for e in new_events)
        log.info(f"{len(new_events)} new event(s) — running post-event chain: {names}")
        refresh_after_event()
    else:
        log.info("No new events; skipping the post-event stats chain")

    try:
        from app.services.ufc.model import generate_predictions
        generate_predictions()
        log.info("Winner predictions regenerated")
        record_run("Generate Predictions", "done")
    except Exception as e:
        log.error(f"Winner prediction generation failed: {e}")
        record_run("Generate Predictions", "error", str(e))

    try:
        from app.services.ufc.method_model import generate_method_predictions
        generate_method_predictions()
        log.info("Method predictions regenerated")
        record_run("Method Predictions", "done")
    except Exception as e:
        log.error(f"Method prediction generation failed: {e}")
        record_run("Method Predictions", "error", str(e))

    # On event nights refresh_after_event already published rankings, on fresh derived and
    # career stats. On quiet nights they still need republishing: eligibility and the
    # inactivity sigma inflation are both functions of today's date, so the table drifts
    # even when no one fights.
    #
    # One call, one transaction. These were two separately-guarded steps: the Glicko step
    # wrote rank=0 placeholders and the Points step replaced them, so a failure in the
    # second logged an error and left a table of zeros serving at the top of every
    # division. publish_rankings computes both and commits once or not at all.
    if not new_events:
        try:
            from app.database import SessionLocal as RankingSession
            from app.services.ufc.ranking_publisher import publish_rankings
            _rank_db = RankingSession()
            try:
                publish_rankings(_rank_db)
            finally:
                _rank_db.close()
            log.info("Glicko snapshots + fighter rankings published")
            record_run("Generate Rankings", "done")
        except Exception as e:
            log.error(f"Ranking publication failed: {e}")
            record_run("Generate Rankings", "error", str(e))

    try:
        from app.services.ufc.preview_service import generate_all_upcoming_previews
        generate_all_upcoming_previews()
        log.info("Fight previews generated")
        record_run("Generate All Previews", "done")
    except Exception as e:
        log.error(f"Fight preview generation failed: {e}")
        record_run("Generate All Previews", "error", str(e))

    record_run("Full Pipeline", "done")


def scheduled_bovada_scrape():
    import logging
    log = logging.getLogger("scheduled_bovada")
    from app.routers.admin import record_run
    try:
        from app.services.ufc.bovada_scraper import scrape_bovada_method_odds
        scrape_bovada_method_odds()
        log.info("Bovada method odds scraped")
        record_run("Bovada Odds", "done")
    except Exception as e:
        log.error(f"Bovada method odds scrape failed: {e}")
        record_run("Bovada Odds", "error", str(e))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    run_migrations()
    scheduler.add_job(scheduled_scrape, "interval", hours=24, id="ufc_scrape", replace_existing=True)
    scheduler.add_job(scheduled_bovada_scrape, "cron", day_of_week="thu", hour=12, id="bovada_scrape", replace_existing=True)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(
    title="ALocks Analytics API",
    description="Sports betting analytics — predictions, models, and stats",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ufc.router)
app.include_router(predictions.router)
app.include_router(admin.router)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "alocks-backend"}
