"""Tables for the fightflow feature track. Same Base / schema conventions as app.models.ufc."""
import datetime as dt

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import TimestampMixin
from app.models.ufc import UFC_SCHEMA, _fk


class UFCFighterClutch(Base):
    """Clutch rating (app/services/ufc/fightflow_clutch.py): a fighter's shrunk mean
    residual, in round-model logit points, in late rounds (3+) and rounds right after a
    lost one, against what the league-wide fit expects from how the bout had gone.
    Each run replaces a version's rows wholesale."""

    __tablename__ = "ufc_fighter_clutch"
    __table_args__ = (UniqueConstraint("fighter_id", "version", name="uq_fighter_clutch_fighter_version"),
                      {"schema": UFC_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fighter_id: Mapped[int] = mapped_column(BigInteger, index=True)
    version: Mapped[str] = mapped_column(String(20))
    clutch: Mapped[float] = mapped_column(Float)
    se: Mapped[float] = mapped_column(Float)
    #: Late rounds (3+) only.
    late: Mapped[float | None] = mapped_column(Float, nullable=True)
    late_rounds: Mapped[int] = mapped_column(Integer, default=0)
    #: Rounds right after a lost round only.
    bounce: Mapped[float | None] = mapped_column(Float, nullable=True)
    bounce_rounds: Mapped[int] = mapped_column(Integer, default=0)
    n_rounds: Mapped[int] = mapped_column(Integer)
    n_fights: Mapped[int] = mapped_column(Integer)
    #: Among fighters with >= MIN_FIGHTS bouts holding a clutch round; None below that.
    percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime)
