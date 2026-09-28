from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Integer, String, Text, func
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


class AdminActionRun(Base):
    """One row per action that touched production: a button in the admin dashboard, a
    scheduled job, a GitHub Actions workflow, or a manual edit.

    This is the audit log, so rows are only ever inserted and then closed out — never
    deleted. A pipeline writes one parent row and one child row per step (parent_id), so
    "the full pipeline ran" and "which of its steps failed" are both answerable.

    Not TimestampMixin: its BigInteger primary key does not autoincrement on SQLite, and
    started_at/finished_at say more than created_at/updated_at would.
    """
    __tablename__ = "admin_action_runs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True,
    )
    parent_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(120), index=True)
    #: manual (admin dashboard) | scheduled (in-app APScheduler) | github (Actions) | cli
    source: Mapped[str] = mapped_column(String(20), index=True)
    #: Who ran it: the name entered in the admin dashboard, the GitHub actor, or "scheduler".
    actor: Mapped[str | None] = mapped_column(String(120), nullable=True)
    #: running | done | partial (finished, but a step failed) | error | interrupted
    status: Mapped[str] = mapped_column(String(20), index=True)
    #: What the action was aimed at, e.g. "fighter:123" or "event:456", so one record's
    #: history can be pulled without parsing params.
    target: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    params: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
