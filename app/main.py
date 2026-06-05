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


def scheduled_scrape():
    from app.services.scraper import run_scrape
    run_scrape(mode="update")


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
