from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import TimestampMixin


class Prediction(TimestampMixin, Base):
    __tablename__ = "predictions"

    sport: Mapped[str] = mapped_column(String(50), index=True)
    event_id: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(100))
    predicted_outcome: Mapped[str] = mapped_column(String(200))
    confidence: Mapped[float] = mapped_column(Float)
    actual_outcome: Mapped[str | None] = mapped_column(String(200), nullable=True)


class ModelRun(TimestampMixin, Base):
    __tablename__ = "model_runs"

    sport: Mapped[str] = mapped_column(String(50), index=True)
    model_name: Mapped[str] = mapped_column(String(100))
    run_date: Mapped[datetime] = mapped_column(DateTime)
    accuracy: Mapped[float] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)


class OddsSnapshot(TimestampMixin, Base):
    __tablename__ = "odds_snapshots"

    sport: Mapped[str] = mapped_column(String(50), index=True)
    event_id: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(100))
    home_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    draw_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    over_under: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
