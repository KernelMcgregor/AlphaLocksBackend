"""Shared plumbing for the prediction-market ingesters (Kalshi, Polymarket).

Three concerns live here: rate-limited HTTP, matching a venue's free-text fighter names back
to a row in `ufc_fights`, and the DB writers for quotes and history.
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone

import httpx
from sqlalchemy import func
from sqlalchemy.orm import joinedload

from app.database import SessionLocal
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter,
    UFCPredictionMarket, UFCPredictionMarketHistory, UFCPredictionMarketQuote,
)

# Reused rather than re-implemented: these two are already the project's fighter-name matcher,
# and a fourth copy of them (there are three) would be a fourth thing to keep in sync.
from app.services.ufc.odds_scraper import _names_match, _normalize_name

log = logging.getLogger("prediction_markets")

USER_AGENT = "alocks-analytics/1.0 (+https://github.com/KernelMcgregor)"

#: Requests per second per host. Measured 2026-09-13, unauthenticated:
#: Kalshi 429s at ~10 rps and runs clean at 5.2 rps -- 4 leaves headroom, and Kalshi sends no
#: Retry-After or X-RateLimit-* headers, so there is no way to discover the ceiling except by
#: tripping it. Polymarket served 30 requests at ~14 rps with zero 429s on both Gamma and CLOB;
#: 10 is deliberately conservative against an undocumented and changeable limit.
RATE_LIMITS = {"kalshi": 4.0, "polymarket": 10.0}


class _TokenBucket:
    """Per-host request throttle.

    A bucket rather than a bare `time.sleep` between calls because a run interleaves both venues:
    sleeping after every request would make Polymarket pay Kalshi's much slower rate. Locked
    because the backfill may later want a thread pool, and an unsynchronised bucket silently
    stops throttling the moment it gets one.
    """

    def __init__(self, rate: float):
        self.min_interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def take(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval


_buckets = {host: _TokenBucket(rate) for host, rate in RATE_LIMITS.items()}
_client: httpx.Client | None = None


def http_client() -> httpx.Client:
    """One shared client. Gamma 403s without a User-Agent -- that is a UA check, not throttling."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    return _client


def fetch_json(host: str, url: str, params: dict | None = None, retries: int = 5):
    """GET JSON, throttled per host, with blind exponential backoff on 429/5xx.

    Backoff is blind because Kalshi returns no Retry-After header; parsing one that isn't there
    would just be decoration. Returns None rather than raising when the retries are exhausted:
    one unavailable market must not abort a refresh covering an entire card.
    """
    delay = 1.0
    for attempt in range(retries):
        _buckets[host].take()
        try:
            resp = http_client().get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                log.warning(f"  {host} {resp.status_code} on {url} (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                delay *= 2
                continue
            # 4xx other than 429 will not improve with a retry.
            log.warning(f"  {host} {resp.status_code} on {url}: {resp.text[:200]}")
            return None
        except httpx.HTTPError as e:
            log.warning(f"  {host} transport error on {url}: {e} (attempt {attempt + 1}/{retries})")
            time.sleep(delay)
            delay *= 2
    log.error(f"  {host} gave up on {url} after {retries} attempts")
    return None


# --------------------------------------------------------------------------- fight matching

def _fight_date(fight: UFCFight) -> date | None:
    return fight.event.date if fight.event else None


#: Candidate fights per date window, for the life of one run. Every fight on a card shares a date,
#: so without this a 13-fight card issues 13 identical windowed queries plus a lazy load of both
#: fighters per candidate -- which measured at ~2.4s per event against the hosted database, or
#: ~20 minutes of almost entirely redundant round trips for a full backfill.
_candidate_cache: dict[tuple[date, int], list] = {}
#: The session the cached fights are attached to. Cached rows are ORM instances, so they are only
#: usable while their session is open: a run that ingests both venues opens one session per venue,
#: and reusing Kalshi's fights under Polymarket's session raises DetachedInstanceError the moment
#: anything touches a lazy relationship. Tracking the owner and dropping the cache when it changes
#: is cheaper than re-querying, and safer than trusting callers to clear it.
_cache_owner: object | None = None


def clear_caches() -> None:
    """Drop the candidate cache. Call between runs so a long-lived process sees new fights."""
    global _cache_owner
    _candidate_cache.clear()
    _cache_owner = None


def _candidates(db, fight_date: date, window_days: int) -> list:
    global _cache_owner
    if _cache_owner is not db:
        _candidate_cache.clear()
        _cache_owner = db

    key = (fight_date, window_days)
    if key not in _candidate_cache:
        lo = fight_date - timedelta(days=window_days)
        hi = fight_date + timedelta(days=window_days)
        _candidate_cache[key] = (
            db.query(UFCFight)
            .options(joinedload(UFCFight.red_fighter), joinedload(UFCFight.blue_fighter))
            .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
            .filter(UFCEvent.date >= lo, UFCEvent.date <= hi)
            .all()
        )
    return _candidate_cache[key]


#: Generational suffixes. ufcstats records 'Sean King III' and 'Raul Rosas Jr.' where the
#: prediction-market venues bill plain 'Sean King' -- and because the suffix becomes the last
#: token, surname comparison ends up matching 'king' against 'iii' and fails.
_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def _fold(text: str) -> str:
    """Normalise for comparison: case, punctuation, diacritics, and generational suffixes.

    Extends `odds_scraper._normalize_name` with two things the prediction-market venues need and
    The Odds API did not. Both showed up as real misses in the dry run:

    * **Diacritics.** Polymarket writes 'Morgan Charrière', 'Édgar Cháirez', 'Joel Álvarez';
      ufcstats stores them unaccented. NFKD decomposition drops the combining marks so the two
      spellings compare equal.
    * **Suffixes.** See `_SUFFIXES` above.
    """
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = _normalize_name(folded)
    parts = [p for p in folded.split() if p not in _SUFFIXES]
    return " ".join(parts)


def _fold_match(a: str, b: str) -> bool:
    """`_names_match`, applied to diacritic- and suffix-folded names."""
    return _names_match(_fold(a), _fold(b))


def hint_agrees(hint: str | None, first: str | None) -> bool:
    """Does a slug-derived first-name prefix agree with this fighter's first name?

    Compared in both directions because the slug token is not always shorter than the name it
    abbreviates: `ufc-rod1-bon-2025-11-15` is Rodolfo Vieira vs **Bo** Nickal, where the token
    'bon' runs past the whole first name. A one-directional `first.startswith(hint)` reads that as
    a contradiction and throws away a correct match.

    Absent information counts as agreement -- this is only ever used to choose between candidates
    that already matched on name.
    """
    if not hint or not first:
        return True
    h, f = _fold(hint), _fold(first)
    return f.startswith(h) or h.startswith(f)


def name_matches(venue_name: str, first: str | None, last: str | None,
                 hint: str | None = None) -> bool:
    """Does a venue's rendering of a fighter's name refer to this fighter?

    `odds_scraper._names_match` alone is not enough here. It requires either a full-string match
    or a shared surname *plus* a matching first initial, and the prediction-market venues very
    often bill a fight by surname only -- 'Makhachev vs Della Maddalena', 'Naurdiev vs Loder'.
    Against a surname-only string that rule compares the surname to our fighter's first initial
    and rejects every genuine match; it accounted for essentially all 156 misses in the first
    dry run.

    So surname-only names are accepted here, which makes this matcher deliberately looser than
    the sportsbook one. The looseness is safe only because `match_fight` refuses to return an
    ambiguous result -- see there. Multi-token surnames ('Della Maddalena', 'Machado Garry',
    'Saint Denis') fall out of the same comparison for free.

    `hint` is a first-name prefix recovered from the venue's slug, used only to break ties.
    """
    if not last:
        return False
    venue = _fold(venue_name)
    full = f"{first or ''} {last or ''}".strip()

    if _fold_match(venue_name, full):
        return True

    # Surname-only billing ('Makhachev vs Della Maddalena'). Accepted here; `match_fight` is
    # responsible for refusing to act when more than one fighter answers to the surname.
    return venue == _fold(last)


def match_fight(db, name_a: str, name_b: str, fight_date: date, window_days: int = 2,
                hints: tuple[str | None, str | None] = (None, None)):
    """Resolve a venue's two fighter names + date to a fight, and say whether corners are swapped.

    Returns `(fight, is_swapped)` or `None`. `is_swapped` is True when the venue's *first* name is
    our blue corner, and callers must honour it: neither venue has any notion of red/blue and both
    list fighters in whatever order the bout is billed. Getting it wrong silently inverts every
    downstream probability while still looking entirely plausible, which is why this is returned
    explicitly rather than left for the caller to infer.

    **Ambiguity is a failure, not a coin flip.** Every candidate in the window is scored and a
    match is returned only when exactly one fight fits. Surname-only billing makes collisions
    genuinely possible -- two Nurmagomedovs or two Topurias on one card is not hypothetical -- and
    silently picking the first would be the single worst failure mode available here: a confident,
    plausible, wrong attribution. Slug hints are applied first as a tie-break, and anything still
    ambiguous is logged and skipped.

    The date window absorbs timezone skew -- a Saturday-night US card lands on the next UTC day --
    and nothing more. Two days rather than four is load bearing: both venues list Contender Series
    cards (Tuesdays) that ufcstats does not cover, and at four days those reach the neighbouring
    Saturday card, find no name match there, and report as ordinary match failures. At two days
    they find no candidates at all and raise `NoSuchCard`, which is what they actually are.
    """
    candidates = _candidates(db, fight_date, window_days)
    hint_a, hint_b = hints
    exact: list[tuple] = []
    loose: list[tuple] = []

    for fight in candidates:
        red, blue = fight.red_fighter, fight.blue_fighter
        if not red or not blue:
            continue

        for swapped, (fa, fb) in ((False, (red, blue)), (True, (blue, red))):
            if not (name_matches(name_a, fa.first_name, fa.last_name)
                    and name_matches(name_b, fb.first_name, fb.last_name)):
                continue
            # A full-name agreement on both fighters outranks a pair of surname-only hits, so a
            # precise match is never thrown away as "ambiguous" against a vaguer competitor.
            both_full = (
                _fold_match(name_a, f"{fa.first_name or ''} {fa.last_name or ''}".strip())
                and _fold_match(name_b, f"{fb.first_name or ''} {fb.last_name or ''}".strip())
            )
            (exact if both_full else loose).append((fight, swapped))

    for tier in (exact, loose):
        if not tier:
            continue
        # Both orderings of a genuine self-rematch resolve to the same fight; that is one match.
        if len({(f.id, s) for f, s in tier}) == 1:
            return tier[0]

        # Only now do slug hints come in, to choose among candidates that already agree on name.
        # Applying them earlier lets a hint reject the single correct match outright.
        narrowed = [
            (f, s) for f, s in tier
            if hint_agrees(hint_a, (f.blue_fighter if s else f.red_fighter).first_name)
            and hint_agrees(hint_b, (f.red_fighter if s else f.blue_fighter).first_name)
        ]
        if len({(f.id, s) for f, s in narrowed}) == 1:
            return narrowed[0]

        log.warning(
            f"  ambiguous: '{name_a}' vs '{name_b}' ({fight_date}) matched "
            f"{len({(f.id, s) for f, s in tier})} fights — skipping rather than guessing"
        )
        return None

    if not candidates:
        raise NoSuchCard(fight_date)
    return None


class NoSuchCard(LookupError):
    """No event exists in the window at all -- so there was nothing here to match against.

    Distinguished from an ordinary match failure because the two mean opposite things. Both venues
    tag Dana White's Contender Series cards `ufc`, and ufcstats does not cover DWCS, so those
    events have no fight rows and never will. Counting them as matching failures buries a genuine
    name-matching regression inside a constant ~15% of events that are correctly skipped.
    """

    def __init__(self, fight_date: date):
        super().__init__(f"no event within the window of {fight_date}")
        self.fight_date = fight_date


def split_versus(title: str) -> tuple[str, str] | None:
    """Pull two fighter names out of a venue's billing string.

    Polymarket event titles look like 'Noche UFC: Jessie Rosas vs. Sean King (Featherweight,
    Prelims)', so the prefix before ':' and the parenthetical suffix both have to go before the
    'vs.' split is meaningful.
    """
    text = title.split(":", 1)[-1]
    text = text.split("(", 1)[0].strip()
    for sep in (" vs. ", " vs ", " VS ", " Vs. "):
        if sep in text:
            a, b = text.split(sep, 1)
            a, b = a.strip(), b.strip()
            if a and b:
                return a, b
    return None


# --------------------------------------------------------------------------- writers

def upsert_market(db, *, platform: str, external_event_id: str, external_market_id: str,
                  market_type: str, outcome_key: str, outcome_label: str | None,
                  side: str | None, fight_id: int | None, status: str = "open",
                  resolved_outcome: float | None = None) -> UFCPredictionMarket:
    """Insert or refresh the catalog row, returning it. Keyed on (platform, external_market_id)."""
    row = (
        db.query(UFCPredictionMarket)
        .filter(
            UFCPredictionMarket.platform == platform,
            UFCPredictionMarket.external_market_id == external_market_id,
        )
        .first()
    )
    if row is None:
        row = UFCPredictionMarket(platform=platform, external_market_id=external_market_id)
        db.add(row)

    row.external_event_id = external_event_id
    row.market_type = market_type
    row.outcome_key = outcome_key
    row.outcome_label = outcome_label
    row.side = side
    row.status = status
    if fight_id is not None:
        row.fight_id = fight_id
    if resolved_outcome is not None:
        row.resolved_outcome = resolved_outcome

    db.flush()  # populate row.id for the quote/history writers
    return row


def upsert_quote(db, market_id: int, *, price: float, bid: float | None = None,
                 ask: float | None = None, volume: float | None = None,
                 open_interest: float | None = None, liquidity: float | None = None,
                 captured_at: datetime | None = None) -> None:
    """Replace the market's current quote in place -- the closing price once it settles."""
    captured_at = captured_at or datetime.now(timezone.utc).replace(tzinfo=None)
    row = (
        db.query(UFCPredictionMarketQuote)
        .filter(UFCPredictionMarketQuote.market_id == market_id)
        .first()
    )
    if row is None:
        row = UFCPredictionMarketQuote(market_id=market_id)
        db.add(row)
    row.price = price
    row.bid = bid
    row.ask = ask
    row.volume = volume
    row.open_interest = open_interest
    row.liquidity = liquidity
    row.captured_at = captured_at


def record_history(db, rows: list[dict]) -> int:
    """Append curve points, ignoring any that are already stored.

    Isolated and defensive for the same reason `odds_scraper._record_odds_history` is: the history
    append is a side effect of a job whose primary duty is refreshing current quotes, and a
    failure here -- a table missing on a DB that has not run create_all, a constraint collision --
    must not roll back the quotes.

    ON CONFLICT DO NOTHING is what makes the whole pull-don't-poll design work: every run
    re-requests overlapping candle windows, and re-ingesting a window has to be a no-op rather
    than an error for the history to be self-healing.
    """
    if not rows:
        return 0
    try:
        table = UFCPredictionMarketHistory.__table__
        if db.bind.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = pg_insert(table).values(rows).on_conflict_do_nothing(
                index_elements=["market_id", "captured_at"]
            )
        else:
            # SQLite honours the same semantics via OR IGNORE.
            stmt = table.insert().prefix_with("OR IGNORE").values(rows)
        result = db.execute(stmt)
        db.commit()
        return result.rowcount or 0
    except Exception as e:
        db.rollback()
        log.warning(f"  History not recorded ({e.__class__.__name__}: {e}). Quotes were saved.")
        return 0


def last_history_ts(db, market_id: int) -> datetime | None:
    """Newest stored curve point, so a refresh only asks the venue for the window it is missing."""
    return (
        db.query(func.max(UFCPredictionMarketHistory.captured_at))
        .filter(UFCPredictionMarketHistory.market_id == market_id)
        .scalar()
    )


def days_to_fight(captured_at: datetime, reference: date | datetime | None) -> float | None:
    """Days from an observation to the bout, positive before it.

    `reference` should be a *datetime* whenever the bout's own start time is known, and only fall
    back to the event date otherwise. The distinction decides whether the closing line is real:
    a card runs for six or seven hours, so measuring an early prelim against the event's headline
    time counts hours of post-result trading as "before the fight". Klose vs Gantt closed at
    19:49 UTC while the main event started at 03:40 the next day; against the event time its
    closing line read 0.01, which is the market knowing the result, not pricing it.
    """
    if reference is None:
        return None
    ref = reference if isinstance(reference, datetime) else datetime.combine(reference, datetime.min.time())
    if ref.tzinfo is not None:
        ref = ref.replace(tzinfo=None)
    if captured_at.tzinfo is not None:
        captured_at = captured_at.replace(tzinfo=None)
    return (ref - captured_at).total_seconds() / 86400.0


def open_session():
    return SessionLocal()


def align_fight_anchor(db, fight_id: int) -> int:
    """Put every venue's curve for one fight on a single time reference.

    Each venue stores `days_to_fight` against its own idea of when the bout is: Kalshi against a
    per-market `close_time` or the event's `occurrence_datetime`, Polymarket against midnight on
    the event's date, because Polymarket publishes no bout time at all. For UFC 331 those differ
    by 31 hours -- at the same instant Kalshi read 6.31 days out and Polymarket 5.00.

    That is not merely cosmetic. `days_to_fight` exists so price can be analysed as a function of
    time-to-event, and CLV compares a closing line against a bout time; a column that means
    something different depending on which venue wrote the row quietly corrupts both.

    The latest anchor wins, because it is the one derived from a real event time rather than a
    date rounded down to midnight. Returns the number of rows rewritten.
    """
    from sqlalchemy import func, literal, update

    # Latest observation per market, in two portable queries. The obvious single-query form needs
    # Postgres-only date arithmetic (make_interval / DISTINCT ON), and this module has to run on
    # the SQLite database used for local development too.
    latest = (
        db.query(UFCPredictionMarketHistory.market_id,
                 func.max(UFCPredictionMarketHistory.captured_at))
        .join(UFCPredictionMarket,
              UFCPredictionMarket.id == UFCPredictionMarketHistory.market_id)
        .filter(UFCPredictionMarket.fight_id == fight_id,
                UFCPredictionMarketHistory.days_to_fight.isnot(None))
        .group_by(UFCPredictionMarketHistory.market_id)
        .all()
    )
    if not latest:
        return 0

    anchors: dict[int, datetime] = {}
    for market_id, captured in latest:
        row = (
            db.query(UFCPredictionMarketHistory.days_to_fight)
            .filter(UFCPredictionMarketHistory.market_id == market_id,
                    UFCPredictionMarketHistory.captured_at == captured)
            .first()
        )
        if row and row[0] is not None:
            anchors[market_id] = captured + timedelta(days=row[0])
    if not anchors:
        return 0

    anchor = max(anchors.values())
    updated = 0
    for market_id, market_anchor in anchors.items():
        # A minute of slack: the anchors are reconstructed from a rounded column, and rewriting
        # thousands of rows to move them by seconds is pure churn.
        if abs((anchor - market_anchor).total_seconds()) < 60:
            continue
        result = db.execute(
            update(UFCPredictionMarketHistory)
            .where(UFCPredictionMarketHistory.market_id == market_id)
            .values(days_to_fight=(
                func.extract("epoch",
                             literal(anchor) - UFCPredictionMarketHistory.captured_at) / 86400.0
            ))
        )
        updated += result.rowcount or 0
    return updated
