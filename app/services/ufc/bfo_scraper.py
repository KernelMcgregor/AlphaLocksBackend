"""
BestFightOdds (bestfightodds.com) historical UFC moneyline scraper.

What BFO exposes (verified 2026-09-28):

* ``/robots.txt`` -- ``User-agent: * / Allow: /``. Scraping is permitted; we still
  throttle (>= 2 s between requests), send a descriptive User-Agent and cache every
  raw response on disk so a re-run never re-fetches.
* ``/sitemap-events.xml`` -- every event (~4.1k; ~800 past UFC cards after filtering) with
  ``<lastmod>`` equal to the event date. One request lists the whole archive back to
  2007. ``/archive`` only shows recent events, so the sitemap is the index.
* ``/events/<slug>-<id>`` -- the odds table. The header maps bookmaker ids to names
  (``<th data-b="21">FanDuel``). Each moneyline cell carries
  ``data-li="[book_id, side, matchup_id]"`` and the *latest* American price, which for
  a finished event is the closing line. Only bookmakers BFO currently lists get
  columns: 2019 and earlier pages have empty cells, 2024+ pages are populated (the
  exact cut-over year was not probed). A JSON-LD block gives the exact ``startDate``.
* ``/api/ggd?b=<book>&m=<matchup>&p=<side>`` -- the line-movement chart for one book
  and one side: a timestamped series of *decimal* odds. The first point is the opener
  and the last is the closer. Omitting ``b`` returns the cross-book "Mean" series.
  Responses are base64 of ROT47 text (mirrors ``notIn()`` in ``/js/bfo.min.js``).
  ``p`` is mandatory, so each series costs two requests (one per fighter).
  Historical per-book series survive only for a few legacy books (seen so far:
  1 = 5Dimes, 12 = "Ref", 19 = Bet365), plus current books for recent events.

Caveats encoded below:

* Prediction-market books (Polymarket id 28, Kalshi id 29) trade in-play, so their
  last price, and the BFO "Mean" series on events where they are listed, can reflect
  in-fight moves. By default they are excluded from per-book rows, and the consensus
  close uses the median of the sportsbook closes on the event page when available.
* Some books (5Dimes, "Ref") keep ticking into the evening of the card; without bout
  start times we cannot cut at walkout, so a close may include a few minutes of late
  (possibly in-play) movement. Moves seen were small (e.g. +165..+185).
* Chart series occasionally contain single-tick garbage (e.g. 5.5 in a run of 1.17).
  :func:`clean_series` drops isolated spikes before taking the open and close.

Usage::

    python -m app.services.ufc.bfo_scraper --events 5 --out data/bfo/odds.csv
    python -m app.services.ufc.bfo_scraper --event-slug ufc-145-jones-vs-evans-490 \
        --history books --out data/bfo/odds.csv --match
    python -m app.services.ufc.bfo_scraper --budget      # full-crawl request estimate

Matching (``--match``) reads the ufc_fights, ufc_events and ufc_fighters tables
read-only from ``--db-url`` (default ``postgresql://localhost/alocks_local``). Only a
localhost or SQLite URL is accepted: the app's ``.env`` points at production.
"""

from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("bfo_scraper")

BASE_URL = "https://www.bestfightodds.com"
USER_AGENT = (
    "ALocksResearchBot/0.1 (UFC odds research; cached, >=2s throttle)"
)
MIN_THROTTLE_S = 2.0
DEFAULT_CACHE_DIR = Path("data/bfo/cache")
DEFAULT_DB_URL = "postgresql://localhost/alocks_local"

#: Bookmakers that are exchanges / prediction markets and trade in-play.
EXCHANGE_BOOK_IDS = {28, 29}
#: Legacy books whose historical series still exist on /api/ggd (from probing ids 1-27
#: on 2012 and 2019 main events). Names are BFO's short labels.
LEGACY_BOOKS = {1: "5Dimes", 12: "Ref", 19: "Bet365"}
MEAN_BOOK_ID = 0
MEAN_BOOK_NAME = "BFO Mean"
CONSENSUS_BOOK_NAME = "Consensus"

_ROT47_ALPHABET = "".join(chr(c) for c in range(33, 127))  # '!'..'~', 94 chars


class BFOBlocked(RuntimeError):
    """The site refused us (403/429/503 or a challenge page). Stop; do not evade."""


# --------------------------------------------------------------------------- data


@dataclasses.dataclass
class BFOEventRef:
    slug: str
    bfo_event_id: int | None
    url: str
    date: dt.date | None


@dataclasses.dataclass
class BFOMatchup:
    matchup_id: int
    fighter_a: str
    fighter_b: str
    fighter_a_slug: str | None = None
    fighter_b_slug: str | None = None
    #: book_id -> [price_side1, price_side2] (American, as displayed = latest/closing)
    page_odds: dict[int, list[int | None]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class BFOEventPage:
    name: str
    date: dt.date | None
    books: dict[int, str]
    matchups: list[BFOMatchup]


@dataclasses.dataclass
class SeriesPoint:
    ts: dt.datetime
    decimal: float

    @property
    def american(self) -> int | None:
        return decimal_to_american(self.decimal)


# ------------------------------------------------------------------ odds helpers


def decimal_to_american(d: float | None) -> int | None:
    if d is None or not math.isfinite(d) or d <= 1.0:
        return None
    if d >= 2.0:
        return int(round((d - 1.0) * 100))
    return int(round(-100.0 / (d - 1.0)))


def american_to_decimal(a: int | None) -> float | None:
    if a is None or a == 0:
        return None
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def american_to_prob(a: int | None) -> float | None:
    d = american_to_decimal(a)
    return None if d is None else 1.0 / d


def devig_pair(a: int | None, b: int | None) -> tuple[float | None, float | None]:
    """Multiplicative (normalised) vig removal, matching UFCFightOdds.*_implied_prob."""
    pa, pb = american_to_prob(a), american_to_prob(b)
    if pa is None or pb is None:
        return None, None
    s = pa + pb
    return pa / s, pb / s


def parse_american(text: str | None) -> int | None:
    if not text:
        return None
    t = text.strip().replace("−", "-")
    if t.upper() in {"EV", "EVEN"}:
        return 100
    m = re.fullmatch(r"([+-]?\d+)", t)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------- decoding


def decode_ggd(raw: str) -> list[dict]:
    """Decode an /api/ggd response (base64 -> UTF-8 -> ROT47 -> JSON).

    Returns the list of Highcharts series dicts, e.g.
    ``[{"name": "FanDuel", "data": [{"x": ms_epoch, "y": decimal}, ...]}]``.
    An empty response ("[]" or blank) yields ``[]``.
    """
    s = (raw or "").strip()
    if not s or s == "[]":
        return []
    if s.startswith("["):  # already plain JSON (defensive)
        return json.loads(s)
    b64 = re.sub(r"[^A-Za-z0-9+/=]", "", s)
    text = base64.b64decode(b64 + "=" * (-len(b64) % 4)).decode("utf-8")
    n = len(_ROT47_ALPHABET)
    out = "".join(
        _ROT47_ALPHABET[(_ROT47_ALPHABET.index(ch) + n // 2) % n] if ch in _ROT47_ALPHABET else ch
        for ch in text
    )
    return json.loads(out)


def series_points(series: dict) -> list[SeriesPoint]:
    pts = []
    for p in series.get("data", []):
        x, y = p.get("x"), p.get("y")
        if x is None or y is None:
            continue
        pts.append(SeriesPoint(dt.datetime.fromtimestamp(x / 1000, tz=dt.timezone.utc), float(y)))
    pts.sort(key=lambda p: p.ts)
    return pts


def clean_series(points: list[SeriesPoint], spike_ratio: float = 2.0,
                 settle_ratio: float = 1.25) -> tuple[list[SeriesPoint], int]:
    """Drop isolated one-tick spikes. Returns (clean_points, n_dropped).

    A point is a spike when it differs from *both* neighbours by more than
    ``spike_ratio`` (in decimal-odds ratio terms) while the neighbours agree with each
    other within ``settle_ratio``. An opening point is a spike when it differs from the
    next point by ``spike_ratio`` and the next two points agree with each other.
    """
    if len(points) < 3:
        return list(points), 0
    lr = lambda a, b: abs(math.log(a / b))  # noqa: E731
    big, small = math.log(spike_ratio), math.log(settle_ratio)
    keep = [True] * len(points)
    d = [p.decimal for p in points]
    if lr(d[0], d[1]) > big and lr(d[1], d[2]) < small:
        keep[0] = False
    for i in range(1, len(d) - 1):
        if lr(d[i], d[i - 1]) > big and lr(d[i], d[i + 1]) > big and lr(d[i - 1], d[i + 1]) < small:
            keep[i] = False
    if lr(d[-1], d[-2]) > big and lr(d[-2], d[-3]) < small:
        keep[-1] = False
    clean = [p for p, k in zip(points, keep) if k]
    return clean, len(points) - len(clean)


# ---------------------------------------------------------------------- parsing

_SITEMAP_RE = re.compile(r"<url>\s*<loc>([^<]+)</loc>\s*(?:<lastmod>([^<]+)</lastmod>)?", re.S)
_UFC_SLUG_RE = re.compile(r"(^|-)ufc(-|$)")
#: Grappling cards and Road to UFC (not in ufcstats).
_NON_MMA_UFC_PREFIXES = ("ufc-bjj", "ufc-fight-pass-invitational", "ufc-grappling", "road-to-ufc")


def parse_sitemap_events(xml: str) -> list[BFOEventRef]:
    out = []
    for loc, lastmod in _SITEMAP_RE.findall(xml):
        loc = loc.strip()
        path = urlparse(loc).path
        if not path.startswith("/events/"):
            continue
        slug = path.rsplit("/", 1)[-1]
        m = re.search(r"-(\d+)$", slug)
        date = None
        if lastmod:
            try:
                date = dt.date.fromisoformat(lastmod.strip()[:10])
            except ValueError:
                pass
        out.append(BFOEventRef(slug, int(m.group(1)) if m else None, loc, date))
    return out


def is_ufc_event_slug(slug: str) -> bool:
    s = slug.lower()
    if not _UFC_SLUG_RE.search(s):
        return False
    # NB: bare slugs like "ufc-3159" are real cards (2023-24 naming) and must be kept.
    # A few are "UFC" placeholder buckets (e.g. ufc-3525, Jan 1 2026) holding rumoured
    # bouts; they simply fail to match. BFO also splits one card into two buckets a
    # day apart (UTC rollover: ufc-vegas-119-4225/4226), hence the +-2 day match window.
    return not s.startswith(_NON_MMA_UFC_PREFIXES)


def _fighter_slug(href: str | None) -> str | None:
    if href and href.startswith("/fighters/"):
        return href.split("/fighters/", 1)[1]
    return None


def _json_ld_date(html: str) -> dt.date | None:
    m = re.search(r'"startDate"\s*:\s*"(\d{4}-\d{2}-\d{2})', html)
    return dt.date.fromisoformat(m.group(1)) if m else None


def parse_event_page(html: str) -> BFOEventPage:
    """Parse an /events/ page: event name/date, book map, matchups + latest per-book odds."""
    # The full page is ~1 MB of prop rows. Slice to the main odds table so the pure-
    # Python parser only walks what we need.
    start = html.find('<table class="odds-table">')
    if start < 0:
        raise ValueError("odds table not found on event page")
    end = html.find("</table>", start)
    table = BeautifulSoup(html[start:end + len("</table>")], "html.parser")

    name = None
    m = re.search(r'<div class="table-header">.*?<h1>(.*?)</h1>', html, re.S)
    if m:
        name = re.sub(r"\s+Odds$", "", BeautifulSoup(m.group(1), "html.parser").get_text(strip=True))
    if not name:
        m = re.search(r'"@type"\s*:\s*"SportsEvent",\s*"name"\s*:\s*"([^"]+)"', html)
        name = m.group(1) if m else ""

    books: dict[int, str] = {}
    thead = table.find("thead")
    if thead:
        for th in thead.find_all("th", attrs={"data-b": True}):
            link = th.find(["a", "span"])
            label = (link.get_text(" ", strip=True) if link else th.get_text(" ", strip=True))
            books[int(th["data-b"])] = label.replace("\xa0", " ").strip()

    matchups: dict[int, BFOMatchup] = {}
    order: list[int] = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr", recursive=False):
        if "pr" in (tr.get("class") or []):
            continue  # prop row
        th = tr.find("th")
        a = th.find("a", href=re.compile(r"^/fighters/")) if th else None
        if not a:
            continue
        side_cell = tr.find("td", class_="but-si")
        side = mu_id = None
        if side_cell and side_cell.get("data-li"):
            side, mu_id = json.loads(side_cell["data-li"])[:2]
        # Fall back to moneyline cells if the index button is missing.
        cells = [td for td in tr.find_all("td") if td.get("data-li") and "but-sg" in (td.get("class") or [])]
        if mu_id is None and cells:
            _, side, mu_id = json.loads(cells[0]["data-li"])[:3]
        if mu_id is None:
            continue
        fname = a.get_text(" ", strip=True)
        fslug = _fighter_slug(a.get("href"))
        mu = matchups.get(mu_id)
        if mu is None:
            mu = BFOMatchup(matchup_id=mu_id, fighter_a="", fighter_b="")
            matchups[mu_id] = mu
            order.append(mu_id)
        if side == 1:
            mu.fighter_a, mu.fighter_a_slug = fname, fslug
        else:
            mu.fighter_b, mu.fighter_b_slug = fname, fslug
        for td in cells:
            li = json.loads(td["data-li"])
            if len(li) != 3:
                continue
            book_id, s, _ = li
            span = td.find("span")
            price = parse_american(span.get_text(strip=True) if span else None)
            pair = mu.page_odds.setdefault(int(book_id), [None, None])
            pair[int(s) - 1] = price
    return BFOEventPage(name=name, date=_json_ld_date(html), books=books,
                        matchups=[matchups[i] for i in order])


# ----------------------------------------------------------------------- client


class BFOClient:
    """Polite HTTP client: disk cache first, then throttled GET, stop on any block."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, throttle_s: float = MIN_THROTTLE_S,
                 max_requests: int | None = None, offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.throttle_s = max(MIN_THROTTLE_S, throttle_s)
        self.max_requests = max_requests
        self.offline = offline
        self.n_network = 0
        self.n_cache = 0
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en"})

    def _cache_path(self, path: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip("/")) or "root"
        if len(safe) > 150:
            safe = safe[:100] + "_" + hashlib.sha1(path.encode()).hexdigest()[:16]
        sub = "ggd" if path.startswith("/api/ggd") else "pages"
        return self.cache_dir / sub / f"{safe}.txt"

    def get(self, path: str, max_age_s: float | None = None, referer: str | None = None) -> str:
        cp = self._cache_path(path)
        if cp.exists() and (max_age_s is None or time.time() - cp.stat().st_mtime < max_age_s):
            self.n_cache += 1
            return cp.read_text(encoding="utf-8")
        if self.offline:
            raise FileNotFoundError(f"offline and not cached: {path}")
        if self.max_requests is not None and self.n_network >= self.max_requests:
            raise RuntimeError(f"request cap reached ({self.max_requests}); raise --max-requests")
        wait = self._last + self.throttle_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        headers = {"Referer": referer} if referer else {}
        # Transient connection resets happen (seen once in ~100 requests). Back off
        # hard and retry twice with the same identity; persistent failure = stop.
        for attempt, backoff in enumerate((30, 120, None)):
            try:
                resp = self.session.get(BASE_URL + path, headers=headers, timeout=30)
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                if backoff is None:
                    raise BFOBlocked(f"repeated connection failures on {path}: {e}") from e
                log.warning("connection error on %s (%s); backing off %ds", path, e, backoff)
                time.sleep(backoff)
        self._last = time.monotonic()
        self.n_network += 1
        body = resp.text
        if resp.status_code in (403, 429, 503) or re.search(
                r"<title>\s*Just a moment|cf-chl|challenge-platform|captcha", body[:5000], re.I):
            raise BFOBlocked(f"{resp.status_code} on {path}; stopping (not evading)")
        resp.raise_for_status()
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(body, encoding="utf-8")
        log.debug("GET %s -> %s (%d B)", path, resp.status_code, len(body))
        return body

    # -- endpoints --

    def robots_allows(self, path: str = "/events/") -> bool:
        txt = self.get("/robots.txt", max_age_s=7 * 86400)
        agent_all, disallowed = False, []
        for line in txt.splitlines():
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            k, v = (s.strip() for s in line.split(":", 1))
            if k.lower() == "user-agent":
                agent_all = v == "*"
            elif k.lower() == "disallow" and agent_all and v:
                disallowed.append(v)
        return not any(path.startswith(d) for d in disallowed)

    def list_events(self) -> list[BFOEventRef]:
        return parse_sitemap_events(self.get("/sitemap-events.xml", max_age_s=86400))

    def event_page(self, ref: BFOEventRef) -> BFOEventPage:
        # Finished events are immutable; recent/future ones may still move.
        recent = ref.date is None or ref.date >= dt.date.today() - dt.timedelta(days=3)
        html = self.get(urlparse(ref.url).path, max_age_s=6 * 3600 if recent else None)
        return parse_event_page(html)

    def chart(self, matchup_id: int, side: int, book_id: int | None = None,
              referer: str | None = None, recent: bool = False) -> list[dict]:
        q = f"/api/ggd?m={matchup_id}&p={side}" if not book_id else \
            f"/api/ggd?b={book_id}&m={matchup_id}&p={side}"
        return decode_ggd(self.get(q, max_age_s=6 * 3600 if recent else None, referer=referer))


def list_ufc_events(client: BFOClient, since: dt.date | None = None, until: dt.date | None = None,
                    include_future: bool = False) -> list[BFOEventRef]:
    today = dt.date.today()
    evs = [e for e in client.list_events() if is_ufc_event_slug(e.slug) and e.date]
    if not include_future:
        evs = [e for e in evs if e.date < today]
    if since:
        evs = [e for e in evs if e.date >= since]
    if until:
        evs = [e for e in evs if e.date <= until]
    # Dedupe by slug (sitemaps sometimes repeat) and sort newest first.
    seen, out = set(), []
    for e in sorted(evs, key=lambda e: e.date, reverse=True):
        if e.slug not in seen:
            seen.add(e.slug)
            out.append(e)
    return out


# ------------------------------------------------------------------- row builder

ROW_FIELDS = [
    "event_name", "event_date", "bfo_event_id", "bfo_event_slug", "bfo_matchup_id",
    "fighter_a", "fighter_b", "fighter_a_bfo_slug", "fighter_b_bfo_slug",
    "bookmaker", "bookmaker_id", "is_exchange",
    "open_a", "open_b", "open_ts_a", "open_ts_b",
    "close_a", "close_b", "close_ts_a", "close_ts_b",
    "open_prob_a", "close_prob_a", "close_source", "n_points_a", "n_points_b",
    "n_spikes_dropped", "flags",
]
MOVE_FIELDS = ["bfo_matchup_id", "bookmaker", "bookmaker_id", "side", "fighter",
               "ts", "decimal", "american"]


def _iso(ts: dt.datetime | None) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ") if ts else ""


def _summarise(points: list[SeriesPoint]) -> tuple[int | None, int | None, dt.datetime | None,
                                                   dt.datetime | None, int, int]:
    clean, dropped = clean_series(points)
    if not clean:
        return None, None, None, None, 0, dropped
    return clean[0].american, clean[-1].american, clean[0].ts, clean[-1].ts, len(clean), dropped


def build_rows(ref: BFOEventRef, page: BFOEventPage, client: BFOClient | None,
               history: str = "mean", legacy_books: dict[int, str] | None = None,
               include_exchanges: bool = False, max_matchups: int | None = None,
               ) -> tuple[list[dict], list[dict]]:
    """Turn a parsed event page (+ optional chart fetches) into odds and movement rows.

    history: "none"  -> event-page closing prices only (1 request/event)
             "mean"  -> + BFO Mean chart per fighter (opening, timestamps, movement)
             "books" -> + per-book charts for every sportsbook priced on the page and
                        every legacy book id (empty responses are cheap and cached)
    """
    legacy_books = LEGACY_BOOKS if legacy_books is None else legacy_books
    event_date = page.date or ref.date
    recent = event_date is None or event_date >= dt.date.today() - dt.timedelta(days=3)
    referer = ref.url
    rows: list[dict] = []
    moves: list[dict] = []
    book_names = dict(page.books)

    def base(mu: BFOMatchup) -> dict:
        return {
            "event_name": page.name, "event_date": event_date.isoformat() if event_date else "",
            "bfo_event_id": ref.bfo_event_id, "bfo_event_slug": ref.slug,
            "bfo_matchup_id": mu.matchup_id, "fighter_a": mu.fighter_a, "fighter_b": mu.fighter_b,
            "fighter_a_bfo_slug": mu.fighter_a_slug, "fighter_b_bfo_slug": mu.fighter_b_slug,
        }

    def fetch(mu: BFOMatchup, book_id: int | None, name: str) -> dict | None:
        """Fetch both sides of one series; record movement; return summary or None."""
        summ = {}
        for side in (1, 2):
            series = client.chart(mu.matchup_id, side, book_id or None, referer=referer, recent=recent)
            pts = series_points(series[0]) if series else []
            for p in pts:
                moves.append({"bfo_matchup_id": mu.matchup_id, "bookmaker": name,
                              "bookmaker_id": book_id or MEAN_BOOK_ID, "side": side,
                              "fighter": mu.fighter_a if side == 1 else mu.fighter_b,
                              "ts": _iso(p.ts), "decimal": p.decimal, "american": p.american})
            summ[side] = _summarise(pts)
        if not summ[1][4] and not summ[2][4]:
            return None
        return summ

    matchups = page.matchups[:max_matchups] if max_matchups else page.matchups
    for mu in matchups:
        per_book_closes: list[tuple[int, int]] = []
        exchange_on_page = False
        # --- per-book rows from event page cells (latest = closing for past events)
        page_rows: dict[int, dict] = {}
        for book_id, (pa, pb) in mu.page_odds.items():
            if pa is None and pb is None:
                continue
            is_ex = book_id in EXCHANGE_BOOK_IDS
            exchange_on_page |= is_ex
            if is_ex and not include_exchanges:
                continue
            if pa is not None and pb is not None and not is_ex:
                per_book_closes.append((pa, pb))
            r = base(mu) | {"bookmaker": book_names.get(book_id, f"book_{book_id}"),
                            "bookmaker_id": book_id, "is_exchange": is_ex,
                            "close_a": pa, "close_b": pb, "close_source": "event_page",
                            "flags": "exchange_inplay_risk" if is_ex else ""}
            page_rows[book_id] = r

        # --- per-book charts
        if history == "books" and client is not None:
            ids = [b for b in page_rows] + [b for b in legacy_books if b not in page_rows]
            for book_id in ids:
                name = book_names.get(book_id) or legacy_books.get(book_id, f"book_{book_id}")
                summ = fetch(mu, book_id, name)
                if summ is None:
                    continue
                r = page_rows.get(book_id) or (base(mu) | {
                    "bookmaker": name, "bookmaker_id": book_id,
                    "is_exchange": book_id in EXCHANGE_BOOK_IDS, "flags": ""})
                (oa, ca, ota, cta, na, da), (ob, cb, otb, ctb, nb, db) = summ[1], summ[2]
                r.update({"open_a": oa, "open_b": ob, "open_ts_a": _iso(ota), "open_ts_b": _iso(otb),
                          "close_ts_a": _iso(cta), "close_ts_b": _iso(ctb),
                          "n_points_a": na, "n_points_b": nb, "n_spikes_dropped": da + db})
                if r.get("close_a") is None or r.get("close_b") is None:
                    r.update({"close_a": ca, "close_b": cb, "close_source": "chart"})
                    if ca is not None and cb is not None and book_id not in EXCHANGE_BOOK_IDS:
                        per_book_closes.append((ca, cb))
                page_rows[book_id] = r
        rows.extend(page_rows.values())

        # --- consensus row: Mean chart for opening/movement; close from books if possible
        if history in ("mean", "books") and client is not None:
            summ = fetch(mu, None, MEAN_BOOK_NAME)
            if summ is not None:
                (oa, ca, ota, cta, na, da), (ob, cb, otb, ctb, nb, db) = summ[1], summ[2]
                flags = []
                close_src = "mean_chart"
                if per_book_closes:
                    # The (lower-)median sportsbook by de-vigged probability; keep its real
                    # (vigged) prices so open and close are on the same footing.
                    ranked = sorted(per_book_closes, key=lambda ab: devig_pair(*ab)[0])
                    ca, cb = ranked[(len(ranked) - 1) // 2]
                    close_src = f"median_of_{len(per_book_closes)}_books"
                    cta = ctb = None  # page cells carry no timestamp
                elif exchange_on_page:
                    flags.append("mean_close_may_include_inplay_exchange")
                r = base(mu) | {
                    "bookmaker": CONSENSUS_BOOK_NAME, "bookmaker_id": MEAN_BOOK_ID, "is_exchange": False,
                    "open_a": oa, "open_b": ob, "open_ts_a": _iso(ota), "open_ts_b": _iso(otb),
                    "close_a": ca, "close_b": cb, "close_ts_a": _iso(cta), "close_ts_b": _iso(ctb),
                    "close_source": close_src, "n_points_a": na, "n_points_b": nb,
                    "n_spikes_dropped": da + db, "flags": ";".join(flags)}
                rows.append(r)

    for r in rows:
        r.setdefault("is_exchange", False)
        r["open_prob_a"] = _round(devig_pair(r.get("open_a"), r.get("open_b"))[0])
        r["close_prob_a"] = _round(devig_pair(r.get("close_a"), r.get("close_b"))[0])
    return rows, moves


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


# --------------------------------------------------------------------- matching

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "junior"}
_FIRST_ALIASES = {
    "alex": "alexander", "alexandre": "alexander", "zach": "zachary", "zak": "zachary",
    "mike": "michael", "joe": "joseph", "chris": "christopher", "dan": "daniel",
    "danny": "daniel", "matt": "matthew", "nate": "nathan", "rob": "robert", "bob": "robert",
    "ben": "benjamin", "tony": "anthony", "jon": "jonathan", "josh": "joshua", "will": "william",
    "nick": "nicholas", "jim": "james", "jimmy": "james", "tom": "thomas", "tommy": "thomas",
    "sam": "samuel", "dave": "david", "steve": "steven", "stephen": "steven", "andy": "andrew",
    "drew": "andrew", "greg": "gregory", "jeff": "jeffrey", "rick": "richard", "ricky": "richard",
    "ed": "edward", "eddie": "edward", "pat": "patrick", "phil": "phillip", "philip": "phillip",
    "tim": "timothy", "vic": "victor", "abdul": "abdul", "manny": "manuel", "rafa": "rafael",
}


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes: "Raúl Rosas Jr." -> "raul rosas"."""
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("’", "'")
    s = re.sub(r"(?<=\w)['`](?=\w)", "", s)          # O'Malley -> omalley
    s = re.sub(r"\b([a-z])\.\s*(?=[a-z]\.)", r"\1", s)  # T.J. -> tj.
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t not in _SUFFIXES]
    if toks:
        toks[0] = _FIRST_ALIASES.get(toks[0], toks[0])
    return " ".join(toks)


def name_similarity(a: str, b: str) -> float:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = na.split(), nb.split()
    if sorted(ta) == sorted(tb):                      # "Song Yadong" vs "Yadong Song"
        return 0.99
    if na.replace(" ", "") == nb.replace(" ", ""):  # "De La Rosa" vs "Delarosa"
        return 0.98
    score = max(SequenceMatcher(None, na, nb).ratio(),
                SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio())
    # Single-name fighters ("Alatengheili") or dropped middle names.
    if len(ta) == 1 or len(tb) == 1:
        short, long_ = (ta, tb) if len(ta) == 1 else (tb, ta)
        if short[0] in long_ or short[0] == "".join(long_):
            score = max(score, 0.9)
    elif set(ta) <= set(tb) or set(tb) <= set(ta):
        score = max(score, 0.93)
    elif ta[-1] == tb[-1] and ta[0][0] == tb[0][0]:
        score = max(score, 0.9)                     # same surname + first initial
    elif ta[-1] == tb[-1] or ta[0] == tb[0]:
        score = max(score, min(score + 0.1, 0.85))
    return score


def pair_score(bfo_a: str, bfo_b: str, red: str, blue: str) -> tuple[float, bool]:
    """Best score over both corner orientations. Returns (score, swapped)."""
    s_same = min(name_similarity(bfo_a, red), name_similarity(bfo_b, blue))
    s_swap = min(name_similarity(bfo_a, blue), name_similarity(bfo_b, red))
    return (s_swap, True) if s_swap > s_same else (s_same, False)


def _safe_db_url(url: str) -> str:
    if url.startswith("sqlite"):
        return url
    host = urlparse(url).hostname or "localhost"
    if host not in ("localhost", "127.0.0.1", "::1"):
        raise SystemExit(f"refusing non-local DB URL (host={host}); the app .env is production")
    return url


def load_db_fights(db_url: str, start: dt.date, end: dt.date) -> list[dict]:
    """Read-only fetch of fights with fighter names in [start, end]."""
    from sqlalchemy import create_engine, text

    url = _safe_db_url(db_url)
    is_sqlite = url.startswith("sqlite")
    kw = {} if is_sqlite else {"connect_args": {"options": "-c default_transaction_read_only=on"}}
    engine = create_engine(url, **kw)
    sch = "" if is_sqlite else "ufc."
    sql = text(f"""
        SELECT f.id, COALESCE(f.date, e.date) AS fight_date, e.name AS event_name,
               rf.first_name || ' ' || rf.last_name AS red, bf.first_name || ' ' || bf.last_name AS blue,
               rf.nickname AS red_nick, bf.nickname AS blue_nick
        FROM {sch}ufc_fights f
        JOIN {sch}ufc_events e ON e.id = f.event_id
        JOIN {sch}ufc_fighters rf ON rf.id = f.red_fighter_id
        JOIN {sch}ufc_fighters bf ON bf.id = f.blue_fighter_id
        WHERE COALESCE(f.date, e.date) BETWEEN :s AND :e
    """)
    try:
        with engine.connect() as conn:
            out = []
            for r in conn.execute(sql, {"s": start, "e": end}).mappings():
                d = dict(r)
                if isinstance(d["fight_date"], str):
                    d["fight_date"] = dt.date.fromisoformat(d["fight_date"][:10])
                out.append(d)
            return out
    finally:
        engine.dispose()


def match_matchups(matchups: Iterable[dict], db_fights: list[dict], window_days: int = 2,
                   threshold: float = 0.82) -> dict[int, dict]:
    """Map BFO matchups -> DB fights.

    ``matchups``: dicts with bfo_matchup_id, fighter_a, fighter_b, event_date (date/iso).
    Returns {bfo_matchup_id: {"fight_id", "score", "swapped", "db_red", "db_blue"}}.
    ``swapped`` True means fighter_a is the DB *blue* corner.
    """
    by_date: dict[dt.date, list[dict]] = {}
    for f in db_fights:
        by_date.setdefault(f["fight_date"], []).append(f)
    result: dict[int, dict] = {}
    used: set[int] = set()
    cands_all = []
    for mu in matchups:
        d = mu["event_date"]
        d = dt.date.fromisoformat(d) if isinstance(d, str) else d
        best = None
        for off in range(-window_days, window_days + 1):
            for f in by_date.get(d + dt.timedelta(days=off), []):
                s, swapped = pair_score(mu["fighter_a"], mu["fighter_b"], f["red"], f["blue"])
                if s < threshold:
                    continue
                key = (s, -abs(off))
                if best is None or key > best[0]:
                    best = (key, f, swapped)
        if best:
            cands_all.append((best[0], mu["bfo_matchup_id"], best[1], best[2]))
    # Greedy one-to-one assignment, best scores first.
    for key, mid, f, swapped in sorted(cands_all, key=lambda c: c[0], reverse=True):
        if f["id"] in used or mid in result:
            continue
        used.add(f["id"])
        result[mid] = {"fight_id": f["id"], "score": round(key[0], 3), "swapped": swapped,
                       "db_red": f["red"], "db_blue": f["blue"], "db_event": f["event_name"]}
    return result


# ------------------------------------------------------------------------- CLI


def estimate_budget(client: BFOClient) -> str:
    """Request estimate for a full crawl (1 sitemap + per-event pages + charts)."""
    evs = list_ufc_events(client)
    n = len(evs)
    n_modern = sum(1 for e in evs if e.date.year >= 2023)  # current books priced on page
    avg_mu = 11.0  # sample mean bouts per BFO bucket (split cards pull it below ~13)
    # series per bout: mean=1; books: mean + 3 legacy (+ ~7 current books on 2023+ pages)
    modes = {
        "none": n * 1,
        "mean": n * (1 + 2 * avg_mu),
        "books": (n - n_modern) * (1 + 2 * avg_mu * 4) + n_modern * (1 + 2 * avg_mu * 11),
    }
    years = sorted({e.date.year for e in evs})
    lines = [f"Past UFC event buckets in sitemap: {n} ({years[0]}-{years[-1]}; {n_modern} from 2023+)"]
    for k, req in modes.items():
        lines.append(f"  history={k:<6} ~{req:>8,.0f} requests  ~{req * MIN_THROTTLE_S / 3600:5.1f} h at 2 s/request")
    return "\n".join(lines)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--events", type=int, default=3, help="N most recent past UFC events (after filters)")
    ap.add_argument("--event-slug", action="append", default=[], help="specific BFO event slug(s)")
    ap.add_argument("--since", type=dt.date.fromisoformat)
    ap.add_argument("--until", type=dt.date.fromisoformat)
    ap.add_argument("--history", choices=["none", "mean", "books"], default="mean")
    ap.add_argument("--legacy-books", default=",".join(map(str, LEGACY_BOOKS)),
                    help="legacy book ids to probe in --history books ('' to disable)")
    ap.add_argument("--include-exchanges", action="store_true", help="keep Polymarket/Kalshi rows")
    ap.add_argument("--max-matchups", type=int, help="dev: limit bouts per event for chart fetches")
    ap.add_argument("--out", type=Path, default=Path("data/bfo/odds.csv"))
    ap.add_argument("--movement-out", type=Path, help="default: <out>_movement.csv")
    ap.add_argument("--match", action="store_true", help="report match rate vs local DB")
    ap.add_argument("--db-url", default=os.environ.get("BFO_MATCH_DB_URL", DEFAULT_DB_URL))
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--throttle", type=float, default=MIN_THROTTLE_S)
    ap.add_argument("--max-requests", type=int, default=500, help="network request safety cap")
    ap.add_argument("--offline", action="store_true", help="cache only; never hit the network")
    ap.add_argument("--budget", action="store_true", help="print full-crawl request estimate and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    client = BFOClient(args.cache_dir, args.throttle, args.max_requests, args.offline)
    try:
        if not client.robots_allows("/events/") or not client.robots_allows("/api/ggd"):
            log.error("robots.txt disallows /events/ or /api/ggd -- stopping")
            return 2
        if args.budget:
            print(estimate_budget(client))
            return 0

        all_events = list_ufc_events(client, args.since, args.until)
        if args.event_slug:
            idx = {e.slug: e for e in client.list_events()}
            refs = []
            for s in args.event_slug:
                if s not in idx:
                    log.error("slug not in sitemap: %s", s)
                    return 2
                refs.append(idx[s])
        else:
            refs = all_events[: args.events]
        legacy = {int(b): LEGACY_BOOKS.get(int(b), f"book_{b}")
                  for b in args.legacy_books.split(",") if b.strip()}

        rows, moves = [], []
        for ref in refs:
            try:
                page = client.event_page(ref)
                r, m = build_rows(ref, page, client, args.history, legacy,
                                  args.include_exchanges, args.max_matchups)
            except ValueError as e:
                # Placeholder/cancelled event pages have no odds table. One odd page must
                # not end a 10-hour crawl; skip it and say so.
                log.warning("skipping %s (%s): %s", ref.slug, ref.date, e)
                continue
            except RuntimeError as e:
                if "request cap" not in str(e):
                    raise
                log.warning("%s; writing what was collected", e)
                break
            log.info("%s (%s): %d bouts, %d odds rows, %d movement pts [net=%d cache=%d]",
                     page.name, page.date or ref.date, len(page.matchups), len(r), len(m),
                     client.n_network, client.n_cache)
            rows += r
            moves += m
    except BFOBlocked as e:
        log.error("BLOCKED: %s", e)
        return 3

    mv_out = args.movement_out or args.out.with_name(args.out.stem + "_movement.csv")
    _write_csv(args.out, ROW_FIELDS + ["db_fight_id", "db_swapped", "match_score"], rows)
    _write_csv(mv_out, MOVE_FIELDS, moves)
    log.info("wrote %d rows -> %s, %d movement points -> %s", len(rows), args.out, len(moves), mv_out)

    if args.match and rows:
        mus = {r["bfo_matchup_id"]: r for r in rows}
        dates = [dt.date.fromisoformat(r["event_date"]) for r in mus.values() if r["event_date"]]
        db = load_db_fights(args.db_url, min(dates) - dt.timedelta(days=3), max(dates) + dt.timedelta(days=3))
        res = match_matchups(mus.values(), db)
        for r in rows:
            m = res.get(r["bfo_matchup_id"])
            if m:
                r.update({"db_fight_id": m["fight_id"], "db_swapped": m["swapped"], "match_score": m["score"]})
        _write_csv(args.out, ROW_FIELDS + ["db_fight_id", "db_swapped", "match_score"], rows)
        by_event: dict[str, list[int]] = {}
        for mid, r in mus.items():
            by_event.setdefault(f'{r["event_name"]} ({r["event_date"]})', []).append(mid)
        print(f"\nMatch report vs {urlparse(args.db_url).path or args.db_url}:")
        tot = hit = 0
        for ev, mids in by_event.items():
            h = sum(1 for m in mids if m in res)
            tot, hit = tot + len(mids), hit + h
            print(f"  {ev:<55} {h:>3}/{len(mids):<3}")
            for m in mids:
                if m not in res:
                    print(f"      UNMATCHED: {mus[m]['fighter_a']} vs {mus[m]['fighter_b']}")
        print(f"  TOTAL BFO bouts matched {hit}/{tot} = {hit / max(tot, 1):.1%}"
              "  (unmatched BFO bouts are usually cancelled/rebooked; BFO keeps them)")
        ev_dates = set(dates)
        in_win = [f for f in db if any(abs((f["fight_date"] - d).days) <= 1 for d in ev_dates)]
        got = {m["fight_id"] for m in res.values()}
        cov = sum(1 for f in in_win if f["id"] in got)
        print(f"  DB fights on sampled dates (+-1d) with BFO odds: {cov}/{len(in_win)}"
              f" = {cov / max(len(in_win), 1):.1%}")
        for f in in_win:
            if f["id"] not in got:
                print(f"      DB fight without BFO odds: {f['red']} vs {f['blue']} ({f['fight_date']})")
    print(f"\nrequests: network={client.n_network} cache={client.n_cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
