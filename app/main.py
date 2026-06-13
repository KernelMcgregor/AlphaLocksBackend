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
            conn.executescript(sql_file.read_text())
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


def scheduled_scrape():
    import logging
    log = logging.getLogger("scheduled_scrape")

    from app.services.scraper import run_recent_update
    run_recent_update()

    # Regenerate predictions with existing models (no retraining)
    try:
        from app.services.model import generate_predictions
        generate_predictions()
        log.info("Winner predictions regenerated")
    except Exception as e:
        log.error(f"Winner prediction generation failed: {e}")

    try:
        from app.services.method_model import generate_method_predictions
        generate_method_predictions()
        log.info("Method predictions regenerated")
    except Exception as e:
        log.error(f"Method prediction generation failed: {e}")

    try:
        from app.services.preview_service import generate_all_upcoming_previews
        generate_all_upcoming_previews()
        log.info("Fight previews generated")
    except Exception as e:
        log.error(f"Fight preview generation failed: {e}")



def scheduled_bovada_scrape():
    import logging
    log = logging.getLogger("scheduled_bovada")
    try:
        from app.services.bovada_scraper import scrape_bovada_method_odds
        scrape_bovada_method_odds()
        log.info("Bovada method odds scraped")
    except Exception as e:
        log.error(f"Bovada method odds scrape failed: {e}")


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
