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
        Base.metadata.create_all(bind=engine)
        # Add columns that create_all won't add to existing tables
        from sqlalchemy import text, inspect
        insp = inspect(engine)
        existing = {c["name"] for c in insp.get_columns("ufc_fighters")}
        with engine.begin() as conn:
            if "country_code" not in existing:
                conn.execute(text("ALTER TABLE ufc_fighters ADD COLUMN country_code VARCHAR(2)"))
            if "image_url" not in existing:
                conn.execute(text("ALTER TABLE ufc_fighters ADD COLUMN image_url VARCHAR(500)"))


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    run_migrations()
    scheduler.add_job(scheduled_scrape, "interval", hours=24, id="ufc_scrape", replace_existing=True)
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
