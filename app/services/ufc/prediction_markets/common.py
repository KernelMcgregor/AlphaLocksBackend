"""Shared plumbing for the prediction-market ingesters (Kalshi, Polymarket).

Three concerns live here: rate-limited HTTP, matching a venue's free-text fighter names back
to a row in `ufc_fights`, and the DB writers for quotes and history.
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from dataclasses import dataclass
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
                  resolved_outcome: float | None = None,
                  venue_closed_at: datetime | None = None,
                  card_start_at: datetime | None = None) -> UFCPredictionMarket:
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
    if venue_closed_at is not None:
        row.venue_closed_at = _naive_utc(venue_closed_at)
    if card_start_at is not None:
        row.card_start_at = _naive_utc(card_start_at)

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


def _naive_utc(ts: datetime) -> datetime:
    """Every timestamp in these tables is naive UTC; normalise anything a venue hands back."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def parse_ts(raw: str | None) -> datetime | None:
    """A venue timestamp as naive UTC. Both '2026-09-19T21:00:00Z' and Polymarket's
    '2026-09-19 21:00:00+00' spellings occur, sometimes on the same object."""
    if not raw:
        return None
    text = raw.strip().replace("Z", "+00:00")
    if text.endswith("+00"):
        text += ":00"
    try:
        return _naive_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


# --------------------------------------------------------------------------- bout cutoffs

#: Subtracted from a bout's own close when nothing earlier on the card bounds its start. Kalshi
#: closes a UFC market a few minutes after the result, so close is bout *end*; a five-round fight
#: plus walkouts and the settlement delay runs past 40 minutes. Erring early is deliberate -- a
#: closing line taken slightly too soon is merely a little stale, while one taken slightly too late
#: is the market reading the result off the screen.
BOUT_LEAD = timedelta(minutes=45)

#: An earlier bout's close only bounds this bout's start if it is at least this far before this
#: bout's own close. Guards against a wrong `card_position` (a bout moved on the night) turning
#: this bout's own in-play window into "before the fight".
_MIN_BOUT = timedelta(minutes=10)


@dataclass(frozen=True)
class Bout:
    fight_id: int
    card_position: int | None      # 0 = main event, ascending *down* the card (= earlier)
    closed_at: datetime | None     # venue close of this bout's market (its end), if settled
    card_start: datetime | None    # the venue's card start, if published
    event_date: date | None
    resolved_by: datetime | None = None  # last normal price before the curve pinned at 0/1


def bout_cutoffs(bouts: list[Bout]) -> dict[int, datetime | None]:
    """The last moment each bout on one card can be priced as *pre-fight*.

    Neither venue publishes a per-bout start time, but the card itself bounds it: a bout cannot
    begin until the one before it has finished, and the one before it finished no later than its
    own market closed. So the cutoff for a bout is the latest close among the bouts below it on
    the card, floored at the card's published start. Nothing on the card can be in play before the
    card starts, so the first bout is cut at that start.

    Every fallback errs early:
      * nothing below it has closed and no card start is known, but it has closed itself:
        its own close minus `BOUT_LEAD`;
      * nothing at all (an upcoming Kalshi-only card): midnight UTC on the event date. That
        is before the first prelim for every venue the UFC uses, as far east as Japan.

    On top of all that, the curve itself caps the cutoff. Once the result is in, a market pins at
    0 or 1, so the last price before that tail is no later than the end of the bout, and the
    cutoff is held `BOUT_LEAD` before it. That catches bad venue timestamps the rules above
    cannot: older Kalshi markets often closed an hour or more after the result (one hit 0.99 at
    00:00 and closed at 02:48), and those late closes would otherwise bound the next bout's start.
    Measured on 610 settled fights, the cap takes the share of closing lines at >=95% on the
    eventual winner from 1.8% (Kalshi) / 0.7% (Polymarket) to zero.

    The old anchor was Kalshi's `occurrence_datetime`, which is a scheduled per-bout slot,
    not an observation: an early prelim on 19 Sep 2026 carried 03:00 UTC while its market had
    closed on the result at 22:00. Five hours of post-result trading counted as "before the fight".
    """
    out: dict[int, datetime | None] = {}
    card_start = max((b.card_start for b in bouts if b.card_start), default=None)
    for bout in bouts:
        own = bout.closed_at
        earlier = [
            b.closed_at for b in bouts
            if b.fight_id != bout.fight_id and b.closed_at is not None
            and b.card_position is not None and bout.card_position is not None
            and b.card_position > bout.card_position
            and (own is None or b.closed_at <= own - _MIN_BOUT)
        ]
        bounds = [t for t in (card_start, max(earlier, default=None)) if t is not None]
        if bounds:
            cutoff = max(bounds)
        elif own is not None:
            cutoff = own - BOUT_LEAD
        elif bout.event_date is not None:
            cutoff = datetime.combine(bout.event_date, datetime.min.time())
        else:
            cutoff = None
        # A card start that postdates the bout's own result is bad data, not a bound.
        if cutoff is not None and own is not None and cutoff > own - _MIN_BOUT:
            cutoff = own - BOUT_LEAD
        if bout.resolved_by is not None:
            cap = bout.resolved_by - BOUT_LEAD
            cutoff = cap if cutoff is None else min(cutoff, cap)
        out[bout.fight_id] = cutoff
    return out


def cutoffs_for_fights(db, fight_ids) -> dict[int, datetime | None]:
    """`bout_cutoffs` for these fights, read from stored venue timestamps -- no venue requests.

    Evaluated a whole card at a time, because each bout's cutoff depends on its neighbours.
    """
    fight_ids = list(fight_ids)
    if not fight_ids:
        return {}
    event_ids = {
        e for (e,) in db.query(UFCFight.event_id).filter(UFCFight.id.in_(fight_ids)).distinct()
    }
    card = (
        db.query(UFCFight.id, UFCFight.event_id, UFCFight.card_position, UFCEvent.date)
        .join(UFCEvent, UFCEvent.id == UFCFight.event_id)
        .filter(UFCFight.event_id.in_(event_ids))
        .all()
    )
    resolved = _resolved_by(db, [c.id for c in card])
    times: dict[int, tuple[datetime | None, datetime | None]] = {}
    for fid, closed, start in (
        db.query(UFCPredictionMarket.fight_id,
                 func.min(UFCPredictionMarket.venue_closed_at),
                 func.min(UFCPredictionMarket.card_start_at))
        .filter(UFCPredictionMarket.fight_id.in_([c.id for c in card]),
                UFCPredictionMarket.market_type == "moneyline")
        .group_by(UFCPredictionMarket.fight_id)
    ):
        times[fid] = (closed, start)

    by_event: dict[int, list[Bout]] = {}
    for c in card:
        closed, start = times.get(c.id, (None, None))
        by_event.setdefault(c.event_id, []).append(
            Bout(c.id, c.card_position, closed, start, c.date, resolved.get(c.id)))
    out: dict[int, datetime | None] = {}
    for bouts in by_event.values():
        out.update(bout_cutoffs(bouts))
    wanted = set(fight_ids)
    return {k: v for k, v in out.items() if k in wanted}


#: A price this close to 0 or 1 is the market reading a result, not pricing a bout. Pre-fight UFC
#: favourites essentially never trade past it (that is about -6500).
_PINNED = 0.015


def _resolved_by(db, fight_ids: list[int]) -> dict[int, datetime]:
    """Per fight, the last normal (unpinned) moneyline price before a curve that ends pinned.

    Per market: the newest point strictly inside (_PINNED, 1 - _PINNED), kept only if points
    follow it -- i.e. the curve ends in a pinned tail. A market whose losing side lingers at a
    stale mid price after settlement simply has no tail and contributes nothing. The earliest
    across the fight's markets wins.
    """
    from sqlalchemy import and_, case

    if not fight_ids:
        return {}
    H = UFCPredictionMarketHistory
    normal = and_(H.price > _PINNED, H.price < 1 - _PINNED)
    rows = (
        db.query(UFCPredictionMarket.fight_id,
                 func.max(case((normal, H.captured_at))),
                 func.max(H.captured_at))
        .join(H, H.market_id == UFCPredictionMarket.id)
        .filter(UFCPredictionMarket.fight_id.in_(fight_ids),
                UFCPredictionMarket.market_type == "moneyline")
        .group_by(UFCPredictionMarket.fight_id, UFCPredictionMarket.id)
        .all()
    )
    out: dict[int, datetime] = {}
    for fight_id, last_normal, last in rows:
        if last_normal is None or last is None or last <= last_normal:
            continue
        out[fight_id] = min(out.get(fight_id, last_normal), last_normal)
    return out


def fight_started(cutoff: datetime | None, now: datetime | None = None) -> bool:
    """Whether a live quote can no longer be trusted as a pre-fight price."""
    if cutoff is None:
        return False
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    return now >= cutoff


def realign_cards(db, fight_ids) -> int:
    """Rewrite `days_to_fight` so zero is each bout's cutoff, on every card these fights are on.

    `days_to_fight >= 0` is what every consumer (closing lines, CLV, the movement chart) uses to
    mean "before the fight", so the cutoff lives in this column rather than beside it. A whole card
    is realigned because one bout settling moves the cutoff of the next bout up the card.

    Skips a market whose stored anchor is already within a minute of the cutoff, so a refresh
    that changed nothing rewrites nothing. Returns rows rewritten. Caller commits.
    """
    from sqlalchemy import literal, update

    fight_ids = list(fight_ids)
    if not fight_ids:
        return 0
    event_ids = {
        e for (e,) in db.query(UFCFight.event_id).filter(UFCFight.id.in_(fight_ids)).distinct()
    }
    card_ids = [i for (i,) in db.query(UFCFight.id).filter(UFCFight.event_id.in_(event_ids))]
    cutoffs = cutoffs_for_fights(db, card_ids)

    # Current anchor per market, reconstructed from its newest point.
    newest = (
        db.query(UFCPredictionMarketHistory.market_id,
                 func.max(UFCPredictionMarketHistory.captured_at))
        .join(UFCPredictionMarket, UFCPredictionMarket.id == UFCPredictionMarketHistory.market_id)
        .filter(UFCPredictionMarket.fight_id.in_(card_ids))
        .group_by(UFCPredictionMarketHistory.market_id)
        .all()
    )
    fight_of = dict(
        db.query(UFCPredictionMarket.id, UFCPredictionMarket.fight_id)
        .filter(UFCPredictionMarket.fight_id.in_(card_ids))
    )
    updated = 0
    for market_id, captured in newest:
        cutoff = cutoffs.get(fight_of.get(market_id))
        if cutoff is None:
            continue
        dtf = (
            db.query(UFCPredictionMarketHistory.days_to_fight)
            .filter(UFCPredictionMarketHistory.market_id == market_id,
                    UFCPredictionMarketHistory.captured_at == captured)
            .scalar()
        )
        if dtf is not None and abs((captured + timedelta(days=dtf) - cutoff).total_seconds()) < 60:
            continue
        result = db.execute(
            update(UFCPredictionMarketHistory)
            .where(UFCPredictionMarketHistory.market_id == market_id)
            .values(days_to_fight=(
                func.extract("epoch",
                             literal(cutoff) - UFCPredictionMarketHistory.captured_at) / 86400.0
            ))
        )
        updated += result.rowcount or 0
    return updated
