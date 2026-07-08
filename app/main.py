import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.models import *  # noqa: F401, F403 — ensure all models are registered
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


def scheduled_scrape():
    import logging
    log = logging.getLogger("scheduled_scrape")
    from app.routers.admin import record_run

    from app.services.ufc.scraper import run_recent_update
    try:
        run_recent_update()
        record_run("Recent Update", "done")
    except Exception as e:
        log.error(f"Recent update failed: {e}")
        record_run("Recent Update", "error", str(e))

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

    try:
        from app.services.ufc.ranking_service import generate_rankings
        generate_rankings()
        log.info("Glicko ratings computed (for prediction features + dimension profiles)")
    except Exception as e:
        log.error(f"Glicko rating computation failed: {e}")

    try:
        from app.services.ufc.points_ranking_service import generate_rankings as generate_points_rankings
        generate_points_rankings()
        log.info("Points + Elo fighter rankings generated")
        record_run("Generate Rankings", "done")
    except Exception as e:
        log.error(f"Points + Elo ranking generation failed: {e}")
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
