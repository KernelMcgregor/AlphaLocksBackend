"""Tables for the officials feature track. Same Base / schema conventions as app.models.ufc."""
import datetime as dt

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import TimestampMixin
from app.models.ufc import UFC_SCHEMA, _fk

# Derived tables (app/services/ufc/officials_service.py). Each run replaces a version's
# rows wholesale, and names are copied in so the officials pages read these tables only.


class UFCJudgeProfile(TimestampMixin, Base):
    """One judge's scoring profile against the deserve-to-win round model. Sortable
    headline columns here; everything else (leans with SEs, axis weights, 10-8 rates,
    percentiles, fights by year) in `detail` JSON."""

    __tablename__ = "ufc_judge_profiles"
    __table_args__ = (UniqueConstraint("judge_id", "version", name="uq_judge_profiles_judge_version"),
                      {"schema": UFC_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    judge_id: Mapped[int] = mapped_column(BigInteger, index=True)
    version: Mapped[str] = mapped_column(String(20))
    name: Mapped[str] = mapped_column(String(120))
    eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    n_fights: Mapped[int] = mapped_column(Integer)
    n_rounds: Mapped[int] = mapped_column(Integer)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    acc_above_exp: Mapped[float | None] = mapped_column(Float, nullable=True)
    agree_above_exp: Mapped[float | None] = mapped_column(Float, nullable=True)
    deciding_cards: Mapped[int] = mapped_column(Integer, default=0)
    robbery_cards: Mapped[int] = mapped_column(Integer, default=0)
    first_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    last_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    #: Agreement-map position (classical MDS of excess pairwise disagreement).
    map_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    map_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    detail: Mapped[str] = mapped_column(Text)


class UFCJudgeRound(TimestampMixin, Base):
    """A judge's verdict on one round next to the model's P(red) (before deductions)."""

    __tablename__ = "ufc_judge_rounds"
    __table_args__ = (Index("ix_judge_rounds_judge_version", "judge_id", "version"),
                      {"schema": UFC_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(20))
    judge_id: Mapped[int] = mapped_column(BigInteger)
    fight_id: Mapped[int] = mapped_column(BigInteger, index=True)
    round: Mapped[int] = mapped_column(Integer)
    date: Mapped[dt.date] = mapped_column(Date)
    #: +1 red, -1 blue, 0 even (10-10).
    verdict: Mapped[int] = mapped_column(Integer)
    score: Mapped[str] = mapped_column(String(10))
    p_red: Mapped[float] = mapped_column(Float)
    red_name: Mapped[str] = mapped_column(String(200))
    blue_name: Mapped[str] = mapped_column(String(200))
    event_name: Mapped[str | None] = mapped_column(String(200), nullable=True)


class UFCJudgeCard(TimestampMixin, Base):
    """A judge's card for one decision (official totals, deductions included)."""

    __tablename__ = "ufc_judge_cards"
    __table_args__ = (Index("ix_judge_cards_judge_version", "judge_id", "version"),
                      {"schema": UFC_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(20))
    judge_id: Mapped[int] = mapped_column(BigInteger)
    judge_seq: Mapped[int] = mapped_column(Integer)
    fight_id: Mapped[int] = mapped_column(BigInteger, index=True)
    date: Mapped[dt.date] = mapped_column(Date)
    red_total: Mapped[int] = mapped_column(Integer)
    blue_total: Mapped[int] = mapped_column(Integer)
    #: The judge's winner and the official result: +1 red, -1 blue, 0 draw.
    side: Mapped[int] = mapped_column(Integer)
    result: Mapped[int] = mapped_column(Integer)
    pivotal: Mapped[bool] = mapped_column(Boolean, default=False)
    robbery_card: Mapped[bool] = mapped_column(Boolean, default=False)
    against_model: Mapped[bool] = mapped_column(Boolean, default=False)


class UFCDecisionReview(TimestampMixin, Base):
    """Every verified decision with the meter, the judges, the media and the fans side by
    side (the Robberies page). Fan/media shares are in DB corners."""

    __tablename__ = "ufc_decision_reviews"
    __table_args__ = (UniqueConstraint("fight_id", "version", name="uq_decision_reviews_fight_version"),
                      {"schema": UFC_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(20))
    fight_id: Mapped[int] = mapped_column(BigInteger, index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    event_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    weight_class: Mapped[str | None] = mapped_column(String(100), nullable=True)
    red_id: Mapped[int] = mapped_column(BigInteger)
    blue_id: Mapped[int] = mapped_column(BigInteger)
    red_name: Mapped[str] = mapped_column(String(200))
    blue_name: Mapped[str] = mapped_column(String(200))
    #: Official result: +1 red, -1 blue, 0 draw; kind ud | sd | md | draw.
    result: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str | None] = mapped_column(String(10), nullable=True)
    p_red: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_blue: Mapped[float | None] = mapped_column(Float, nullable=True)
    robbery_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    media_red: Mapped[int] = mapped_column(Integer, default=0)
    media_blue: Mapped[int] = mapped_column(Integer, default=0)
    media_draw: Mapped[int] = mapped_column(Integer, default=0)
    fan_red: Mapped[float | None] = mapped_column(Float, nullable=True)
    fan_blue: Mapped[float | None] = mapped_column(Float, nullable=True)
    fan_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: JSON [{"judge_id", "name", "red", "blue", "side"}] in panel order.
    cards: Mapped[str] = mapped_column(Text)


class UFCJudgePair(TimestampMixin, Base):
    """How often two judges agree on the same rounds vs what the round model expects."""

    __tablename__ = "ufc_judge_pairs"
    __table_args__ = (UniqueConstraint("judge_a", "judge_b", "version", name="uq_judge_pairs"),
                      {"schema": UFC_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(20))
    judge_a: Mapped[int] = mapped_column(BigInteger, index=True)
    judge_b: Mapped[int] = mapped_column(BigInteger, index=True)
    n_rounds: Mapped[int] = mapped_column(Integer)
    agree: Mapped[float] = mapped_column(Float)
    expected: Mapped[float] = mapped_column(Float)


class UFCOfficialsMeta(TimestampMixin, Base):
    """Pool-level numbers for a run (the average judge's leans, calibration, strengths)."""

    __tablename__ = "ufc_officials_meta"
    __table_args__ = (UniqueConstraint("kind", "version", name="uq_officials_meta"),
                      {"schema": UFC_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(20))           # judges | referees
    version: Mapped[str] = mapped_column(String(20))
    payload: Mapped[str] = mapped_column(Text)


class UFCReferee(TimestampMixin, Base):
    """Referee registry + profile. Names normalised from ufc_fights, mmad_decisions and
    sherdog_bouts; stats are UFC fights only (n_regional counts Sherdog-only bouts)."""

    __tablename__ = "ufc_referees"
    __table_args__ = {"schema": UFC_SCHEMA}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120))
    name_norm: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    n_fights: Mapped[int] = mapped_column(Integer, default=0)
    n_regional: Mapped[int] = mapped_column(Integer, default=0)
    finish_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Strike stoppages: the mean of log(1+strikes the loser took in the finishing round)
    #: minus log(1+expected) for a similar stoppage (time in round, round, weight class,
    #: era, knockdowns, pre-fight chin/durability/power), shrunk toward 0 for small
    #: samples. exp(x) - 1 = share more (or fewer) strikes than expected.
    stoppage_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    deductions_per_100: Mapped[float | None] = mapped_column(Float, nullable=True)
    dq_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    last_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class UFCRefereeFight(TimestampMixin, Base):
    """A fight a referee worked (UFC), with what the referee page lists."""

    __tablename__ = "ufc_referee_fights"
    __table_args__ = {"schema": UFC_SCHEMA}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    referee_id: Mapped[int] = mapped_column(Integer, index=True)
    fight_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    date: Mapped[dt.date] = mapped_column(Date)
    event_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    weight_class: Mapped[str | None] = mapped_column(String(100), nullable=True)
    red_name: Mapped[str] = mapped_column(String(200))
    blue_name: Mapped[str] = mapped_column(String(200))
    winner: Mapped[int] = mapped_column(Integer)                   # +1 red, -1 blue, 0 none
    method: Mapped[str | None] = mapped_column(String(60), nullable=True)
    finish_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finish_time: Mapped[str | None] = mapped_column(String(10), nullable=True)
    #: Strike stoppages: the loser's significant strikes in the finishing round, what a
    #: similar stoppage would expect, and log(1+absorbed) - log(1+expected).
    absorbed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    absorbed_expected: Mapped[float | None] = mapped_column(Float, nullable=True)
    absorbed_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    deductions: Mapped[int] = mapped_column(Integer, default=0)


class UFCFightOfficial(TimestampMixin, Base):
    """Assigned officials for a fight, entered in admin before the card (commissions post
    panels days out and no source we scrape carries them) or filled from results."""

    __tablename__ = "ufc_fight_officials"
    __table_args__ = (UniqueConstraint("fight_id", "role", "seq", name="uq_fight_officials"),
                      {"schema": UFC_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True)
    role: Mapped[str] = mapped_column(String(10))            # judge | referee
    seq: Mapped[int] = mapped_column(Integer, default=1)
    name: Mapped[str] = mapped_column(String(120))
    judge_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    referee_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(10), default="admin")
