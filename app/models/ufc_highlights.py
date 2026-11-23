"""Tables for the best-of track (bonuses, excitement). Same Base / schema conventions as
app.models.ufc."""
import datetime as dt

from sqlalchemy import BigInteger, Boolean, Date, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import TimestampMixin
from app.models.ufc import UFC_SCHEMA, _fk


class UFCFightBonus(TimestampMixin, Base):
    """A post-fight bonus as ufcstats shows it (the icons beside the bout).

    bonus_type: fotn (Fight of the Night, both fighters), potn (Performance of the Night,
    2014-02 on), kotn / sotn (Knockout / Submission of the Night, before that). The icon
    marks the bout, not a fighter, so fighter_id is the winner for the performance awards
    and NULL for fotn (or when the bout has no winner)."""

    __tablename__ = "ufc_fight_bonuses"
    __table_args__ = (UniqueConstraint("fight_id", "bonus_type", name="uq_fight_bonuses_fight_type"),
                      {"schema": UFC_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True)
    bonus_type: Mapped[str] = mapped_column(String(10))
    fighter_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="ufcstats")


class UFCFightExcitement(TimestampMixin, Base):
    """How exciting each UFC bout was (app/services/ufc/fight_excitement.py), with the
    model's Fight / Performance of the Night odds. Each run replaces a version's rows."""

    __tablename__ = "ufc_fight_excitement"
    __table_args__ = (UniqueConstraint("fight_id", "model_version", name="uq_fight_excitement_fight_version"),
                      {"schema": UFC_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fight_id: Mapped[int] = mapped_column(BigInteger, index=True)
    model_version: Mapped[str] = mapped_column(String(20))
    event_id: Mapped[int] = mapped_column(BigInteger, index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    #: The fitted utility without the context terms, and its percentile (0-100) among all
    #: scored UFC bouts.
    score: Mapped[float] = mapped_column(Float)
    pct: Mapped[float] = mapped_column(Float, index=True)
    #: JSON {"groups": {action, suspense, swings, finish, upset}, "raw": {feature: value}}
    components: Mapped[str] = mapped_column(Text)
    p_fotn: Mapped[float] = mapped_column(Float)
    #: The winner's Performance of the Night odds (NULL without a winner).
    p_perf: Mapped[float | None] = mapped_column(Float, nullable=True)
    fotn: Mapped[bool] = mapped_column(Boolean, default=False)
    perf: Mapped[bool] = mapped_column(Boolean, default=False)
