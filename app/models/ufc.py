import datetime as dt

from sqlalchemy import BigInteger, Boolean, DateTime, Date, Float, ForeignKey, Index, Integer, LargeBinary, String, Text, UniqueConstraint
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
    #: ISO 3166-1 alpha-2 ("GB"), or a 3166-2 subdivision code for the UK home nations
    #: ("GB-ENG"/"GB-SCT"/"GB-WLS"/"GB-NIR") so England, Scotland and Wales render their
    #: own flags rather than the Union Jack. flag-icons ships both spellings.
    country_code: Mapped[str | None] = mapped_column(String(6), nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    #: Locally cached copy of image_url, filled by scripts/cache_fighter_images.py.
    #: UFC.com serves headshots through Drupal image styles whose URLs carry an `?itok=`
    #: signature — those rotate, so every stored URL is on a clock. Caching the bytes is
    #: what keeps a fighter's portrait working after that happens.
    image_data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    image_mime: Mapped[str | None] = mapped_column(String(40), nullable=True)
    image_fetched_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    # -- Bio, scraped from ufc.com athlete pages (see services/ufc/ufc_profile_scraper.py) --
    #: Raw UFC.com text, e.g. "Rochester, United States" — sometimes just "Germany".
    birthplace: Mapped[str | None] = mapped_column(String(200), nullable=True)
    #: Last comma-segment of birthplace; feeds country_code.
    birth_country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    fighting_style: Mapped[str | None] = mapped_column(String(100), nullable=True)
    trains_at: Mapped[str | None] = mapped_column(String(200), nullable=True)
    #: Text like "40.50", matching the ufcstats-sourced height/weight/reach above.
    leg_reach: Mapped[str | None] = mapped_column(String(20), nullable=True)
    octagon_debut: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str | None] = mapped_column(String(20), nullable=True)  # Active / Retired / ...


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
    #: Row index of this bout on the ufcstats event page: 0 = main event, ascending down
    #: the card. ufcstats exposes no card-segment labels (main/prelim), so position is the
    #: only signal for "this is the headline fight" — without it the API returns bouts in
    #: whatever order Postgres hands back and a title fight can render last.
    #: Nullable: rows scraped before this column existed keep NULL until re-scraped.
    card_position: Mapped[int | None] = mapped_column(Integer, nullable=True)

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


class UFCFightOddsHistory(Base):
    """Append-only snapshots of every odds observation, keyed by capture time.

    `UFCFightOdds` is unique on (fight_id, bookmaker) and is UPDATED in place on each
    scrape, so it holds only the most recent price -- effectively the closing line once
    a fight is over. Every intermediate observation is overwritten and lost.

    That loss is the binding constraint on this project's ability to measure anything.
    Without a price history there is no opening line and no line movement, so closing
    line value cannot be computed, and CLV is the only metric that converges fast enough
    to evaluate a betting model on a realistic timescale. Its absence is why the
    pre-registered picks rule needs ~150 settled bets and two seasons to reach a verdict
    (see PREREGISTRATION.md).

    This table therefore never updates and never deletes: one row per observation. The
    unique constraint is on the capture time as well, so re-running a scrape within the
    same second is idempotent while genuinely new observations always append.

    Nothing reads it yet, and that is expected -- the value is entirely in starting the
    record now, since history cannot be backfilled.
    """

    __tablename__ = "ufc_fight_odds_history"
    __table_args__ = (
        UniqueConstraint("fight_id", "bookmaker", "captured_at"),
        Index("ix_odds_history_fight_captured", "fight_id", "captured_at"),
        {"schema": UFC_SCHEMA},
    )

    # Declared explicitly rather than inherited from TimestampMixin: that mixin brings
    # an `updated_at` with onupdate=now(), and a column that tracks mutation has no
    # business on a table whose entire contract is that rows are never mutated.
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)

    fight_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True)
    bookmaker: Mapped[str] = mapped_column(String(100))
    red_odds: Mapped[int] = mapped_column(Integer)
    blue_odds: Mapped[int] = mapped_column(Integer)
    red_implied_prob: Mapped[float] = mapped_column(Float)   # normalised, vig removed
    blue_implied_prob: Mapped[float] = mapped_column(Float)
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)

    #: Days between capture and the fight date, denormalised at write time. The whole
    #: point of this table is analysing prices as a function of time-to-event, and the
    #: event date can change after the fact when a bout is rebooked.
    days_to_fight: Mapped[float | None] = mapped_column(Float, nullable=True)


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


class UFCPredictionMarket(TimestampMixin, Base):
    """One tradeable outcome on a prediction-market venue (Kalshi, Polymarket).

    Deliberately *not* stored in `ufc_fight_odds`. That table is queried unfiltered by the
    winner model's feature join (`model.py`), by the pre-registered picks generator
    (`scripts/generate_picks.py`), and by `ranking_baselines.py`. PREREGISTRATION.md registers
    the book set as FanDuel/DraftKings/BetMGM/Bovada and lists changing it as a rule change, so
    adding exchange prices there would both contaminate training features and silently invalidate
    the pre-registered rule. Keeping exchanges in their own tables makes that impossible rather
    than merely discouraged.

    Prices here live in probability space (0-1), not American odds: that is how both venues quote,
    and converting to American and back would only lose precision. Exchange quotes are also
    near-no-vig by construction (yes + no ~ 1), so the devig step that `ufc_fight_odds` bakes into
    `red_implied_prob` has no analogue -- normalise across the pair at read time instead.
    """

    __tablename__ = "ufc_prediction_markets"
    __table_args__ = (
        UniqueConstraint("platform", "external_market_id"),
        Index("ix_pred_market_fight_type", "fight_id", "market_type"),
        {"schema": UFC_SCHEMA},
    )

    #: Nullable: markets are recorded even when fuzzy name matching fails to tie them to a fight,
    #: so an unmatched venue market is a visible row to debug rather than a silent drop.
    fight_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey(_fk("ufc_fights.id")), index=True, nullable=True
    )
    platform: Mapped[str] = mapped_column(String(20), index=True)  # 'kalshi' | 'polymarket'

    #: Kalshi event_ticker (KXUFCFIGHT-26SEP19VANPAN) | Polymarket event slug
    external_event_id: Mapped[str] = mapped_column(String(200), index=True)
    #: Kalshi market ticker (…-VAN) | Polymarket CLOB token id
    external_market_id: Mapped[str] = mapped_column(String(200))

    #: 'moneyline'|'method'|'fighter_method'|'round_ou'|'fighter_round'|'distance'|'unknown'
    market_type: Mapped[str] = mapped_column(String(40), index=True)
    #: 'red'|'blue'|'ko_tko'|'submission'|'decision'|'ou_2.5'|'red_r1'|…
    outcome_key: Mapped[str] = mapped_column(String(60))
    #: Raw venue label, kept verbatim. Both venues add market types over time; when
    #: classification falls through to 'unknown' this is the only way to see what arrived.
    outcome_label: Mapped[str | None] = mapped_column(String(300), nullable=True)
    #: Which corner the outcome belongs to, once resolved against ufc_fights. Null for
    #: fight-level markets (method, distance, round totals) that belong to neither corner.
    side: Mapped[str | None] = mapped_column(String(10), nullable=True)

    status: Mapped[str] = mapped_column(String(20), default="open")  # open|settled|cancelled
    #: 1.0 / 0.0 / 0.5 once settled. Comparing this against ufc_fights.winner_id is the
    #: cheapest end-to-end check that matching and corner orientation are both correct.
    resolved_outcome: Mapped[float | None] = mapped_column(Float, nullable=True)

    fight: Mapped["UFCFight"] = relationship()


class UFCPredictionMarketQuote(TimestampMixin, Base):
    """Latest observation per market, updated in place -- the closing price once settled.

    Mirrors the role `UFCFightOdds` plays for sportsbooks: one row, always current. The full
    curve lives in `UFCPredictionMarketHistory`.
    """

    __tablename__ = "ufc_prediction_market_quotes"
    __table_args__ = (
        UniqueConstraint("market_id"),
        {"schema": UFC_SCHEMA},
    )

    market_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(_fk("ufc_prediction_markets.id")), index=True
    )
    price: Mapped[float] = mapped_column(Float)  # probability space, 0-1
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_interest: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)

    market: Mapped["UFCPredictionMarket"] = relationship()


class UFCPredictionMarketHistory(Base):
    """Append-only price curve, one row per market per observation timestamp.

    Unlike `UFCFightOddsHistory` -- which can only ever record what a scrape happened to observe,
    and so cannot be backfilled -- this table is populated from the venues' *own* history
    endpoints (Kalshi candlesticks, Polymarket /prices-history). Two consequences follow, and the
    ingestion job is built around both:

    1. History is backfillable. Curves exist for fights that settled long before this project
       ever called these APIs.
    2. History is self-healing. A missed or failed run leaves no permanent hole, because the next
       run re-requests the same window from the venue rather than recording only "now". This is
       why the refresh job pulls candles instead of polling snapshots, and why running it from
       both APScheduler and GitHub Actions is harmless.

    `captured_at` is therefore the venue's own period timestamp, not our wall clock. Combined with
    the unique constraint that makes re-ingesting a window a no-op.
    """

    __tablename__ = "ufc_prediction_market_history"
    __table_args__ = (
        UniqueConstraint("market_id", "captured_at"),
        Index("ix_pred_market_history_market_captured", "market_id", "captured_at"),
        {"schema": UFC_SCHEMA},
    )

    # Declared explicitly rather than inherited from TimestampMixin, for the same reason
    # UFCFightOddsHistory does: a column tracking mutation has no business on a table whose
    # entire contract is that rows are never mutated.
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)

    market_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(_fk("ufc_prediction_markets.id")), index=True
    )
    price: Mapped[float] = mapped_column(Float)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_interest: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)

    #: Days between this observation and the fight, denormalised at write time -- the whole point
    #: of the table is price as a function of time-to-event, and the event date can move when a
    #: bout is rebooked. Same rationale as UFCFightOddsHistory.days_to_fight.
    days_to_fight: Mapped[float | None] = mapped_column(Float, nullable=True)

    market: Mapped["UFCPredictionMarket"] = relationship()


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
    # 0-1000, min-max normalised WITHIN the division by ranking_publisher. Not a
    # probability and not comparable across divisions. (It was documented as "expected
    # win rate (0-1)" for a long time; it has never held that.)
    score: Mapped[float] = mapped_column(Float)
    expected_wins: Mapped[float] = mapped_column(Float)  # duplicate of `score`, kept for API compat
    total_opponents: Mapped[int] = mapped_column(Integer)
    feature_profile: Mapped[str] = mapped_column(Text)  # JSON blob of feature values

    fighter: Mapped["UFCFighter"] = relationship()


class UFCRankingHistory(Base):
    """Divisional rank for a fighter at a past date.

    `ufc_fighter_rankings` is overwritten on every publish, so it holds only the
    present standings. This table is append-only and keyed by (fighter, as_of), which
    is what makes a rank-over-time chart possible. Populated by
    `python -m app.services.ufc.rank_history_backfill`.
    """
    __tablename__ = "ufc_ranking_history"
    __table_args__ = (
        # weight_class is part of the key: a fighter is ranked both in their own
        # division and in p4p on the same date, so (fighter, date) collides.
        UniqueConstraint("fighter_id", "as_of", "weight_class"),
        Index("ix_rank_history_fighter_date", "fighter_id", "as_of"),
        {"schema": UFC_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")), index=True)
    as_of: Mapped[dt.date] = mapped_column(Date, index=True)
    weight_class: Mapped[str] = mapped_column(String(30))
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)          # 0-1000, normalised within division
    total_ranked: Mapped[int] = mapped_column(Integer)   # division size, so a rank can be read in context
    #: Tapology's Strength of Schedule, 1-99, AS OF this date. Stored rather than derived
    #: because it is computed from each opponent's standing at the time of the bout — it
    #: cannot be recovered later from the fighter's record alone.
    sos: Mapped[int | None] = mapped_column(Integer, nullable=True)


# Canonical names of the rating-confidence columns. glicko_service stores the same
# quantities in its in-memory snapshot dict under a leading underscore ("_meta_sigma"),
# so the dict key is always "_" + the column name.
GLICKO_META_COLS = ["meta_sigma", "meta_rounds_seen", "meta_fights_seen", "meta_days_since"]


# (column, DDL type) for the ufc.com bio fields on ufc_fighters. Kept alongside the model
# because create_all() only creates whole tables — existing tables need explicit ALTERs,
# which main.run_migrations() issues for both the Postgres and SQLite branches. The types
# below are spelled so the same list works verbatim on either.
#: (column, DDL type) added to ufc_ranking_history after its first release. Same pattern
#: as FIGHTER_BIO_COLS: create_all() only creates whole tables, so an existing history
#: table needs an explicit ALTER, spelled so one list works on Postgres and SQLite alike.
RANKING_HISTORY_COLS = [
    ("sos", "INTEGER"),
]


FIGHTER_BIO_COLS = [
    ("birthplace", "VARCHAR(200)"),
    ("birth_country", "VARCHAR(100)"),
    ("fighting_style", "VARCHAR(100)"),
    ("trains_at", "VARCHAR(200)"),
    ("leg_reach", "VARCHAR(20)"),
    ("octagon_debut", "DATE"),
    ("status", "VARCHAR(20)"),
]


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

    # Rating CONFIDENCE, not the rating itself. The model selects on these, so if they
    # are absent here the DB serving path silently feeds it constants that never appeared
    # in training. Nullable because rows written before 006 have no values.
    meta_sigma: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta_rounds_seen: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta_fights_seen: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta_days_since: Mapped[float | None] = mapped_column(Float, nullable=True)


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


class UFCFighterSimilarity(Base):
    """Top-K stylistic comparables for each fighter.

    DISPLAY-ONLY AND RETRODICTIVE. Built from career-to-date stats and the *final*
    Glicko ratings, so a fighter's vector here reflects fights that, for any given
    historical bout, had not happened yet. It carries the same hazard as `whr_ranker`
    and must never reach `model.build_features()`; `tests/test_leakage.py` asserts it.

    Stored top-K rather than pairwise: 4.5k fighters is ~20M pairs, and nothing in the
    product ever asks for the similarity of an arbitrary pair. Same reasoning as
    `ufc_matchup_predictions`, which stores only the matchups it will serve.
    """
    __tablename__ = "ufc_fighter_similarity"
    __table_args__ = (
        UniqueConstraint("fighter_id", "similar_fighter_id"),
        Index("ix_similarity_fighter", "fighter_id"),
        {"schema": UFC_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")), index=True)
    similar_fighter_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(_fk("ufc_fighters.id")))
    rank: Mapped[int] = mapped_column(Integer)
    similarity: Mapped[float] = mapped_column(Float)  # cosine in the frozen whitened space, 0-1
    same_division: Mapped[bool] = mapped_column(Boolean, default=False)

    #: JSON [{"feature": ..., "z": ...}, ...] — the traits both fighters share most
    #: strongly. Without this the panel is an oracle; with it the user can check the
    #: claim against the stat table on the same page.
    top_drivers: Mapped[str] = mapped_column(Text)

    #: Rank this pair held in the previous run; NULL means it is newly in the top-K.
    #: Carried across the delete-and-reinsert so the UI can mark what an event changed.
    previous_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)

    computed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    fighter: Mapped["UFCFighter"] = relationship(foreign_keys=[fighter_id])
    similar_fighter: Mapped["UFCFighter"] = relationship(foreign_keys=[similar_fighter_id])


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
