import datetime as dt

from sqlalchemy import BigInteger, DateTime, Date, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import settings
from app.database import Base
from app.models.base import TimestampMixin

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
UFC_SCHEMA = None if _is_sqlite else "ufc"


def _fk(col: str) -> str:
    """Return a schema-qualified FK reference, e.g. 'ufc.ufc_fighters.id' or just 'ufc_fighters.id'."""
    return col if _is_sqlite else f"ufc.{col}"


class UFCFighter(TimestampMixin, Base):
    __tablename__ = "ufc_fighters"
    __table_args__ = {"schema": UFC_SCHEMA}

    ufcstats_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    nickname: Mapped[str | None] = mapped_column(String(200), nullable=True)
    height: Mapped[str | None] = mapped_column(String(20), nullable=True)
    weight: Mapped[str | None] = mapped_column(String(20), nullable=True)
    reach: Mapped[str | None] = mapped_column(String(20), nullable=True)
    stance: Mapped[str | None] = mapped_column(String(20), nullable=True)
    dob: Mapped[str | None] = mapped_column(Date, nullable=True)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    draws: Mapped[int] = mapped_column(Integer, default=0)
    country_code: Mapped[str | None] = mapped_column(String(2), nullable=True)  # ISO 3166-1 alpha-2
    image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)


class UFCEvent(TimestampMixin, Base):
    __tablename__ = "ufc_events"
    __table_args__ = {"schema": UFC_SCHEMA}

    ufcstats_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(300))
    date: Mapped[str] = mapped_column(Date)
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)

    fights: Mapped[list["UFCFight"]] = relationship(back_populates="event")


class UFCFight(TimestampMixin, Base):
    __tablename__ = "ufc_fights"
    __table_args__ = {"schema": UFC_SCHEMA}

    ufcstats_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    date: Mapped[str | None] = mapped_column(Date, nullable=True)
    event_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_events.id")))
    red_fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")))
    blue_fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")))
    winner_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")), nullable=True)
    red_result: Mapped[str | None] = mapped_column(String(10), nullable=True)
    blue_result: Mapped[str | None] = mapped_column(String(10), nullable=True)
    weight_class: Mapped[str | None] = mapped_column(String(100), nullable=True)
    method: Mapped[str | None] = mapped_column(String(100), nullable=True)
    details: Mapped[str | None] = mapped_column(String(300), nullable=True)
    referee: Mapped[str | None] = mapped_column(String(100), nullable=True)
    finish_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finish_time: Mapped[str | None] = mapped_column(String(10), nullable=True)
    time_format: Mapped[str | None] = mapped_column(String(50), nullable=True)
    fight_time_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_fight_time_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    event: Mapped["UFCEvent"] = relationship(back_populates="fights")
    red_fighter: Mapped["UFCFighter"] = relationship(foreign_keys=[red_fighter_id])
    blue_fighter: Mapped["UFCFighter"] = relationship(foreign_keys=[blue_fighter_id])
    winner: Mapped["UFCFighter | None"] = relationship(foreign_keys=[winner_id])
    stats: Mapped[list["UFCFightStats"]] = relationship(back_populates="fight")


class UFCFightStats(TimestampMixin, Base):
    __tablename__ = "ufc_fight_stats"
    __table_args__ = (
        UniqueConstraint("fight_id", "fighter_id", "round_number"),
        {"schema": UFC_SCHEMA},
    )

    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True)
    fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")), index=True)
    round_number: Mapped[int] = mapped_column(Integer, default=0)  # 0 = totals, 1+ = per round
    corner: Mapped[str] = mapped_column(String(4))
    kd: Mapped[int] = mapped_column(Integer, default=0)
    sig_str_landed: Mapped[int] = mapped_column(Integer, default=0)
    sig_str_attempted: Mapped[int] = mapped_column(Integer, default=0)
    total_str_landed: Mapped[int] = mapped_column(Integer, default=0)
    total_str_attempted: Mapped[int] = mapped_column(Integer, default=0)
    td_landed: Mapped[int] = mapped_column(Integer, default=0)
    td_attempted: Mapped[int] = mapped_column(Integer, default=0)
    sub_att: Mapped[int] = mapped_column(Integer, default=0)
    rev: Mapped[int] = mapped_column(Integer, default=0)
    ctrl_seconds: Mapped[int] = mapped_column(Integer, default=0)
    head_landed: Mapped[int] = mapped_column(Integer, default=0)
    head_attempted: Mapped[int] = mapped_column(Integer, default=0)
    body_landed: Mapped[int] = mapped_column(Integer, default=0)
    body_attempted: Mapped[int] = mapped_column(Integer, default=0)
    leg_landed: Mapped[int] = mapped_column(Integer, default=0)
    leg_attempted: Mapped[int] = mapped_column(Integer, default=0)
    distance_landed: Mapped[int] = mapped_column(Integer, default=0)
    distance_attempted: Mapped[int] = mapped_column(Integer, default=0)
    clinch_landed: Mapped[int] = mapped_column(Integer, default=0)
    clinch_attempted: Mapped[int] = mapped_column(Integer, default=0)
    ground_landed: Mapped[int] = mapped_column(Integer, default=0)
    ground_attempted: Mapped[int] = mapped_column(Integer, default=0)

    # -- Derived: fight context --
    fight_time_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    est_standing_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    est_ground_min: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: striking overall --
    slpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    sapm: Mapped[float | None] = mapped_column(Float, nullable=True)
    sl_diff: Mapped[float | None] = mapped_column(Float, nullable=True)
    sig_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    sig_def: Mapped[float | None] = mapped_column(Float, nullable=True)
    tslpm: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: head (offense + defense) --
    head_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: body (offense + defense) --
    body_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: legs (offense + defense) --
    leg_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: distance position (offense + defense) --
    dist_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: clinch position (offense + defense) --
    clinch_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: ground position (offense + defense + position-aware) --
    ground_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_def: Mapped[float | None] = mapped_column(Float, nullable=True)
    gnp15g: Mapped[float | None] = mapped_column(Float, nullable=True)
    gnp_abs15g: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: knockdowns --
    kd15: Mapped[float | None] = mapped_column(Float, nullable=True)
    kd15s: Mapped[float | None] = mapped_column(Float, nullable=True)
    kd_abs15: Mapped[float | None] = mapped_column(Float, nullable=True)
    kd_abs15s: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: takedowns --
    td15: Mapped[float | None] = mapped_column(Float, nullable=True)
    td15s: Mapped[float | None] = mapped_column(Float, nullable=True)
    td_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    td_abs15: Mapped[float | None] = mapped_column(Float, nullable=True)
    td_abs15s: Mapped[float | None] = mapped_column(Float, nullable=True)
    td_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: control time --
    ctrl15: Mapped[float | None] = mapped_column(Float, nullable=True)
    ctrl15g: Mapped[float | None] = mapped_column(Float, nullable=True)
    ctrl_abs15: Mapped[float | None] = mapped_column(Float, nullable=True)
    ctrl_abs15g: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: submissions --
    sub_att15: Mapped[float | None] = mapped_column(Float, nullable=True)
    sub_att15g: Mapped[float | None] = mapped_column(Float, nullable=True)
    sub_abs15: Mapped[float | None] = mapped_column(Float, nullable=True)
    sub_abs15g: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Derived: reversals --
    rev15: Mapped[float | None] = mapped_column(Float, nullable=True)
    rev_abs15: Mapped[float | None] = mapped_column(Float, nullable=True)

    fight: Mapped["UFCFight"] = relationship(back_populates="stats")
    fighter: Mapped["UFCFighter"] = relationship()


class UFCFightPrediction(TimestampMixin, Base):
    __tablename__ = "ufc_fight_predictions"
    __table_args__ = (
        UniqueConstraint("fight_id"),
        {"schema": UFC_SCHEMA},
    )

    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True)
    predicted_winner: Mapped[str] = mapped_column(String(4))  # 'red' or 'blue'
    confidence: Mapped[float] = mapped_column(Float)  # 0.0 to 0.5
    red_prob: Mapped[float] = mapped_column(Float)  # calibrated probability red wins
    va_prob_low: Mapped[float | None] = mapped_column(Float, nullable=True)  # Venn-Abers p0 (lower bound)
    va_prob_high: Mapped[float | None] = mapped_column(Float, nullable=True)  # Venn-Abers p1 (upper bound)

    fight: Mapped["UFCFight"] = relationship()


class UFCMethodPrediction(TimestampMixin, Base):
    __tablename__ = "ufc_method_predictions"
    __table_args__ = (
        UniqueConstraint("fight_id"),
        {"schema": UFC_SCHEMA},
    )

    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True)
    predicted_method: Mapped[str] = mapped_column(String(20))  # KO/TKO, Submission, Decision
    confidence: Mapped[float] = mapped_column(Float)  # max class probability
    ko_prob: Mapped[float] = mapped_column(Float)
    sub_prob: Mapped[float] = mapped_column(Float)
    dec_prob: Mapped[float] = mapped_column(Float)

    fight: Mapped["UFCFight"] = relationship()


class UFCFightOdds(TimestampMixin, Base):
    __tablename__ = "ufc_fight_odds"
    __table_args__ = (
        UniqueConstraint("fight_id", "bookmaker"),
        {"schema": UFC_SCHEMA},
    )

    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True)
    bookmaker: Mapped[str] = mapped_column(String(100))
    red_odds: Mapped[int] = mapped_column(Integer)  # American odds e.g. -150, +200
    blue_odds: Mapped[int] = mapped_column(Integer)
    red_implied_prob: Mapped[float] = mapped_column(Float)  # vig-removed
    blue_implied_prob: Mapped[float] = mapped_column(Float)

    fight: Mapped["UFCFight"] = relationship()


class UFCMethodOdds(TimestampMixin, Base):
    __tablename__ = "ufc_method_odds"
    __table_args__ = (
        UniqueConstraint("fight_id", "bookmaker"),
        {"schema": UFC_SCHEMA},
    )

    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True)
    bookmaker: Mapped[str] = mapped_column(String(100))
    # "How Will Fight End" market odds (American)
    ko_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sub_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dec_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Implied probabilities (vig-removed)
    ko_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    sub_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    dec_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    # "Method of Victory" per-fighter odds (American)
    red_ko_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    red_sub_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    red_dec_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blue_ko_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blue_sub_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blue_dec_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    fight: Mapped["UFCFight"] = relationship()


class UFCFightShapValue(TimestampMixin, Base):
    __tablename__ = "ufc_fight_shap_values"
    __table_args__ = (
        Index("ix_shap_fight_id", "fight_id"),
        {"schema": UFC_SCHEMA},
    )

    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")))
    feature_name: Mapped[str] = mapped_column(String(100))
    shap_value: Mapped[float] = mapped_column(Float)  # positive = favors red, negative = favors blue
    abs_value: Mapped[float] = mapped_column(Float)  # for sorting by importance
    feature_value: Mapped[float | None] = mapped_column(Float, nullable=True)


class UFCFighterRanking(TimestampMixin, Base):
    __tablename__ = "ufc_fighter_rankings"
    __table_args__ = (
        UniqueConstraint("fighter_id", "weight_class"),
        {"schema": UFC_SCHEMA},
    )

    fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")), index=True)
    weight_class: Mapped[str] = mapped_column(String(30))
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)  # expected win rate (0-1)
    expected_wins: Mapped[float] = mapped_column(Float)
    total_opponents: Mapped[int] = mapped_column(Integer)
    feature_profile: Mapped[str] = mapped_column(Text)  # JSON blob of feature values

    fighter: Mapped["UFCFighter"] = relationship()


class UFCGlickoSnapshot(Base):
    """Pre-fight Glicko dimension ratings for each fighter in each fight.
    Captured BEFORE the fight is processed — used as ML prediction features."""
    __tablename__ = "ufc_glicko_snapshots"
    __table_args__ = (
        UniqueConstraint("fight_id", "fighter_id"),
        Index("ix_glicko_snap_fighter", "fighter_id"),
        {"schema": UFC_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True)
    fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")))
    pts: Mapped[float | None] = mapped_column(Float, nullable=True)
    ko: Mapped[float | None] = mapped_column(Float, nullable=True)
    kod: Mapped[float | None] = mapped_column(Float, nullable=True)
    sub: Mapped[float | None] = mapped_column(Float, nullable=True)
    subd: Mapped[float | None] = mapped_column(Float, nullable=True)
    td: Mapped[float | None] = mapped_column(Float, nullable=True)
    tdd: Mapped[float | None] = mapped_column(Float, nullable=True)
    ctrl: Mapped[float | None] = mapped_column(Float, nullable=True)
    str_vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    str_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    str_def: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch: Mapped[float | None] = mapped_column(Float, nullable=True)
    gnd: Mapped[float | None] = mapped_column(Float, nullable=True)
    durability: Mapped[float | None] = mapped_column(Float, nullable=True)


class UFCMatchupPrediction(TimestampMixin, Base):
    __tablename__ = "ufc_matchup_predictions"
    __table_args__ = (
        UniqueConstraint("red_fighter_id", "blue_fighter_id"),
        Index("ix_matchup_red", "red_fighter_id"),
        Index("ix_matchup_blue", "blue_fighter_id"),
        {"schema": UFC_SCHEMA},
    )

    red_fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")))
    blue_fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")))
    red_win_prob: Mapped[float] = mapped_column(Float)
    ko_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    sub_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    dec_prob: Mapped[float | None] = mapped_column(Float, nullable=True)

    red_fighter: Mapped["UFCFighter"] = relationship(foreign_keys=[red_fighter_id])
    blue_fighter: Mapped["UFCFighter"] = relationship(foreign_keys=[blue_fighter_id])


class UFCFighterCareerStats(Base):
    __tablename__ = "ufc_fighter_career_stats"
    __table_args__ = (
        UniqueConstraint("fighter_id"),
        {"schema": UFC_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")), index=True)

    # -- Foundation --
    fight_count: Mapped[int] = mapped_column(Integer, default=0)
    total_fight_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    est_standing_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    est_ground_min: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Striking: Overall --
    slpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    sapm: Mapped[float | None] = mapped_column(Float, nullable=True)
    sl_diff: Mapped[float | None] = mapped_column(Float, nullable=True)
    sig_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    sig_def: Mapped[float | None] = mapped_column(Float, nullable=True)
    tslpm: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Striking: Head (offense + defense) --
    head_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    head_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Striking: Body (offense + defense) --
    body_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Striking: Legs (offense + defense) --
    leg_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Striking: Distance position (offense + defense) --
    dist_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Striking: Clinch position (offense + defense) --
    clinch_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Striking: Ground position (offense + defense + position-aware) --
    ground_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_abs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_abs_pm: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_def: Mapped[float | None] = mapped_column(Float, nullable=True)
    gnp15g: Mapped[float | None] = mapped_column(Float, nullable=True)
    gnp_abs15g: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Knockdowns --
    kd15: Mapped[float | None] = mapped_column(Float, nullable=True)
    kd15s: Mapped[float | None] = mapped_column(Float, nullable=True)
    kd_abs15: Mapped[float | None] = mapped_column(Float, nullable=True)
    kd_abs15s: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Takedowns --
    td15: Mapped[float | None] = mapped_column(Float, nullable=True)
    td15s: Mapped[float | None] = mapped_column(Float, nullable=True)
    td_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    td_abs15: Mapped[float | None] = mapped_column(Float, nullable=True)
    td_abs15s: Mapped[float | None] = mapped_column(Float, nullable=True)
    td_def: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Control time --
    ctrl15: Mapped[float | None] = mapped_column(Float, nullable=True)
    ctrl15g: Mapped[float | None] = mapped_column(Float, nullable=True)
    ctrl_abs15: Mapped[float | None] = mapped_column(Float, nullable=True)
    ctrl_abs15g: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Submissions --
    sub_att15: Mapped[float | None] = mapped_column(Float, nullable=True)
    sub_att15g: Mapped[float | None] = mapped_column(Float, nullable=True)
    sub_abs15: Mapped[float | None] = mapped_column(Float, nullable=True)
    sub_abs15g: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Reversals --
    rev15: Mapped[float | None] = mapped_column(Float, nullable=True)
    rev_abs15: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Outcomes --
    ko_wins: Mapped[int] = mapped_column(Integer, default=0)
    sub_wins: Mapped[int] = mapped_column(Integer, default=0)
    dec_wins: Mapped[int] = mapped_column(Integer, default=0)
    finish_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_fight_sec: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Metadata --
    computed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    fighter: Mapped["UFCFighter"] = relationship()


class UFCFightPreview(TimestampMixin, Base):
    __tablename__ = "ufc_fight_previews"
    __table_args__ = (
        UniqueConstraint("fight_id"),
        {"schema": UFC_SCHEMA},
    )

    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True)
    content: Mapped[str] = mapped_column(Text)
    model_used: Mapped[str] = mapped_column(String(50))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    fight: Mapped["UFCFight"] = relationship()
