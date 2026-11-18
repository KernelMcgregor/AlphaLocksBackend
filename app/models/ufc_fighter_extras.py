"""Tables for the fighter_extras feature track. Same Base / schema conventions as app.models.ufc."""
import datetime as dt

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import TimestampMixin
from app.models.ufc import UFC_SCHEMA, _fk


class UFCCamp(TimestampMixin, Base):
    """A training camp: the canonical gym that ufc.com `trains_at` strings and Sherdog
    `association` strings resolve to. Rewritten whole by
    `python -m app.services.ufc.fighter_extras_camps --write`. Display only."""
    __tablename__ = "ufc_camps"
    __table_args__ = {"schema": UFC_SCHEMA}

    slug: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    #: Linked UFC fighters at write time, so the leaderboard can sort without a join.
    n_fighters: Mapped[int] = mapped_column(Integer, default=0)
    computed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class UFCCampAlias(TimestampMixin, Base):
    """Normalised gym string -> camp. `raw` keeps the most common spelling seen."""
    __tablename__ = "ufc_camp_aliases"
    __table_args__ = {"schema": UFC_SCHEMA}

    alias_key: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    camp_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_camps.id")), index=True)
    raw: Mapped[str] = mapped_column(String(200))
    n_seen: Mapped[int] = mapped_column(Integer, default=0)


class UFCFighterCamp(TimestampMixin, Base):
    """Fighter -> camp membership. `source` is "ufc" (ufc.com trains_at) or "sherdog"
    (association); `is_primary` marks the one camp shown on the fighter page."""
    __tablename__ = "ufc_fighter_camps"
    __table_args__ = (
        UniqueConstraint("fighter_id", "camp_id"),
        Index("ix_fighter_camps_camp", "camp_id"),
        {"schema": UFC_SCHEMA},
    )

    fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")), index=True)
    camp_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_camps.id")))
    source: Mapped[str] = mapped_column(String(10))
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)


class UFCAgeCurve(TimestampMixin, Base):
    """Division win rate by age at fight time, one row per (division, age year).
    `division` is a division key or "all". Rewritten whole by
    `python -m app.services.ufc.fighter_extras_age --write`. Display only."""
    __tablename__ = "ufc_age_curves"
    __table_args__ = (
        UniqueConstraint("division", "age"),
        {"schema": UFC_SCHEMA},
    )

    division: Mapped[str] = mapped_column(String(30), index=True)
    age: Mapped[int] = mapped_column(Integer)
    fights: Mapped[int] = mapped_column(Integer)
    wins: Mapped[float] = mapped_column(Float)          # draws count half
    win_rate: Mapped[float] = mapped_column(Float)
    #: Win rate pooled over age-1..age+1 (weights 1-2-1), the line the page draws.
    smoothed: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Mean pre-to-next-fight change in the Round Winning Glicko rating at this age.
    rating_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
