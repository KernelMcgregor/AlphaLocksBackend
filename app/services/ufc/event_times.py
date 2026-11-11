"""Segment start times for upcoming cards, from ufc.com's events listing.

ufcstats (the source of ufc_events) publishes an event's date and city only. The
listing at ufc.com/events carries, per upcoming card, its link, headline ("Prochazka vs
Stirling") and each segment's start as a Unix timestamp:

    data-main-card-timestamp="1795284000" data-prelims-card-timestamp="1795273200"
    data-early-card-timestamp=""

Those are absolute instants, so no time-zone handling is needed here: they are stored as
naive UTC and every browser converts to its own zone.

Matching a listing card to one of our events:
  * numbered cards by number: our "UFC 334: Gane vs. Hokit" <-> /event/ufc-334 (also
    "/event/cryptocom-ufc-331" - sponsor prefixes come and go);
  * everything else by date and headline: the main card's date is within a day of ours
    (a Saturday-night card in Las Vegas starts on Sunday in UTC) and a headline surname
    appears in our event name.

Run weekly (main.py, Monday) and from the admin dashboard. Re-reads every upcoming card,
so newly announced cards and moved start times are both picked up.

    python -m app.services.ufc.event_times            # writes
    python -m app.services.ufc.event_times --dry-run  # prints matches only
"""

from __future__ import annotations

import datetime as dt
import html
import logging
import re
import unicodedata
from dataclasses import dataclass

import httpx

log = logging.getLogger("event_times")

LISTING_URL = "https://www.ufc.com/events"
#: The listing paginates; upcoming cards fill the first page or two.
MAX_PAGES = 3
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


@dataclass
class ListedCard:
    slug: str                       # "ufc-334", "ufc-fight-night-november-21-2026"
    headline: str                   # "Prochazka vs Stirling"
    main_card_at: dt.datetime | None
    prelims_at: dt.datetime | None
    early_prelims_at: dt.datetime | None


_CARD_RE = re.compile(
    r'c-card-event--result__headline">\s*<a href="/event/([^"#]+)">([^<]*)</a>.*?'
    r'data-main-card-timestamp="(\d*)"\s+data-prelims-card="[^"]*"\s+'
    r'data-prelims-card-timestamp="(\d*)"\s+data-early-card="[^"]*"\s+'
    r'data-early-card-timestamp="(\d*)"',
    re.S,
)


def _ts(v: str) -> dt.datetime | None:
    return dt.datetime.fromtimestamp(int(v), dt.timezone.utc).replace(tzinfo=None) if v else None


def parse_listing(page: str) -> list[ListedCard]:
    """Every card on one listing page that carries a main-card timestamp."""
    out = []
    for slug, headline, main, prelims, early in _CARD_RE.findall(page):
        if not main:
            continue
        out.append(ListedCard(slug, html.unescape(headline).strip(), _ts(main), _ts(prelims), _ts(early)))
    return out


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _number(text: str) -> int | None:
    m = re.search(r"\bufc[\s-]+(\d{1,4})\b", _fold(text))
    return int(m.group(1)) if m else None


def match_card(event_name: str, event_date: dt.date, cards: list[ListedCard]) -> ListedCard | None:
    """The listing card for one of our events, or None. Pure, so it is unit-tested."""
    num = _number(event_name.split(":")[0])
    if num is not None:
        for c in cards:
            if _number(c.slug) == num:
                return c
        return None
    title = _fold(event_name.split(":", 1)[-1])
    for c in cards:
        if c.main_card_at is None or _number(c.slug) is not None:
            continue
        if abs((c.main_card_at.date() - event_date).days) > 1:
            continue
        names = [w for w in re.split(r"\s+vs\.?\s+|\s+", _fold(c.headline)) if len(w) > 2 and w != "vs"]
        if any(re.search(rf"\b{re.escape(n)}\b", title) for n in names):
            return c
    return None


def fetch_listing(client: httpx.Client) -> list[ListedCard]:
    cards: list[ListedCard] = []
    seen: set[str] = set()
    for page in range(MAX_PAGES):
        resp = client.get(LISTING_URL, params={"page": page} if page else None)
        resp.raise_for_status()
        found = [c for c in parse_listing(resp.text) if c.slug not in seen]
        if not found:
            break
        seen.update(c.slug for c in found)
        cards.extend(found)
    return cards


def update_event_start_times(dry_run: bool = False) -> dict:
    """Fill the three start-time columns on every upcoming event ufc.com lists."""
    from app.database import SessionLocal
    from app.models.ufc import UFCEvent

    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        cards = fetch_listing(client)
    log.info(f"  {len(cards)} timed cards on ufc.com")

    db = SessionLocal()
    try:
        today = dt.date.today()
        upcoming = db.query(UFCEvent).filter(UFCEvent.date >= today - dt.timedelta(days=1)).all()
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        updated, unmatched = [], []
        for ev in upcoming:
            card = match_card(ev.name, ev.date, cards)
            if card is None:
                unmatched.append(ev.name)
                continue
            updated.append(f"{ev.name} -> /event/{card.slug} main {card.main_card_at:%Y-%m-%d %H:%M}Z")
            if not dry_run:
                ev.main_card_at = card.main_card_at
                ev.prelims_at = card.prelims_at
                ev.early_prelims_at = card.early_prelims_at
                ev.start_times_fetched_at = now
        if not dry_run:
            db.commit()
        for line in updated:
            log.info("  " + line)
        if unmatched:
            log.warning(f"  no ufc.com times for: {unmatched}")
        return {"updated": len(updated), "unmatched": unmatched, "matches": updated, "dry_run": dry_run}
    finally:
        db.close()


if __name__ == "__main__":
    import json
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(update_event_start_times(dry_run="--dry-run" in sys.argv), indent=1, default=str))
