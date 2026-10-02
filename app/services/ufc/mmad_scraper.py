"""
mmadecisions.com judges' scorecards: round-by-round cards, media and fan scores.

What the site exposes (verified 2026-10-02; no robots.txt, old Tomcat server):

* ``/decisions-by-event/<year>/`` -- every event with at least one decision that year
  (1995+; UFC from UFC 30, 2001): date, ``event/<id>/<slug>`` link, number of decisions.
* ``/event/<id>/<slug>`` -- each decision on the card (``decision/<id>/<slug>``) with every
  judge's total, last names only.
* ``/decision/<id>/<slug>`` -- the fighters (``fighter/<id>``, winner first, "defeats" or
  "drew with"), decision type, event, date, location, referee, and per judge
  (``judge/<id>``, or "Unknown Judge" without a link) a ROUND | A | B table with a TOTAL
  row. Old cards show "-" for unknown rounds and placeholder totals (1-0 = scored for a
  fighter, score unknown; 1-1 = scored a draw; 0-0 = unknown). Also media scores
  (journalist, outlet, total, pick), fan scoring (the total-score distribution,
  per-round shares and the number of fan scorecards) and referee point deductions
  ("Figueiredo was deducted 1 point in round 3: Low blow"). Round scores on the page are
  AFTER deductions; ufc_judge_scorecards keeps them as posted and stores the deduction
  beside them (red_ded/blue_ded), so pts + ded is the judge's own verdict on the round.

Requests: plain browser User-Agent and nothing else, >= 3 s apart, every raw page cached
on disk so a re-run never re-fetches, and any 403/429/503 or challenge page stops the run
(no evasion).

Mapping to the DB (``load``): decisions are matched to ufc_fights by date +-1 day and both
names (bfo_scraper.match_matchups), then *verified*: the winner must agree with
winner_id, every judge's total must agree with the UFCStats totals in
ufc_fights.details (as a multiset), and each judge's rounds must add up to their total.
mmadecisions fighters are linked to ufc_fighters only through verified fights, and an id
that lands on two DB fighters (or the reverse) is a conflict and left unlinked.

Usage::

    python -m app.services.ufc.mmad_scraper --fetch [--years 2001-2026] [--max-requests N]
    python -m app.services.ufc.mmad_scraper --load [--review data/mmad/review.csv]
    python -m app.services.ufc.mmad_scraper --sync --days 30 --max-requests 150   # nightly

--fetch only fills the disk cache; --load parses the cache and writes; --sync (post-event
workflow) does both for recent decisions, and also credits the UFCStats judge totals of
recent decisions to judges, so a card has judges the morning after even before
mmadecisions posts its rounds.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from app.services.ufc.bfo_scraper import match_matchups, name_similarity, normalize_name
from app.services.ufc.scorecards import parse as parse_ufcstats_cards

log = logging.getLogger("mmad_scraper")

BASE_URL = "https://mmadecisions.com"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/129.0 Safari/537.36")
MIN_THROTTLE_S = 3.0
DEFAULT_CACHE_DIR = Path("data/mmad/cache")
FIRST_YEAR = 1995

#: Event names of other promotions. Anything else on a date with a DB UFC event is a
#: candidate (UFC cards have had many names: "Ortiz vs. Shamrock 3", "Ultimate Fight
#: Night", "TUF ... Finale", "Noche UFC" ...); the fight matcher has the final say.
NON_UFC_PREFIXES = (
    "bellator", "strikeforce", "wec", "pfl", "cw ", "cage warriors", "ksw", "wsof",
    "world series of fighting", "pride", "affliction", "dream", "elitexc", "elite xc",
    "one ", "one:", "invicta", "lfa", "rizin", "m-1", "dwcs", "boxing", "pbc", "top rank",
    "matchroom", "golden boy", "showtime", "dazn", "bkfc", "aca", "brave", "oktagon",
    "professional fighters league", "zuffa boxing", "power slap",
)
_UFC_NAME = re.compile(r"^(ufc|tuf|the ultimate fighter|ultimate fight night|noche ufc)\b", re.I)

#: Totals on old cards that stand for "no score known" rather than a real score.
PLACEHOLDER_TOTALS = {(0, 0), (1, 0), (0, 1), (1, 1)}
_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y")


class MMADBlocked(RuntimeError):
    """The site refused or challenged us. Stop; don't retry with another identity."""


# ------------------------------------------------------------------ dataclasses


@dataclasses.dataclass
class EventRef:
    mmad_event_id: int
    path: str
    name: str
    date: dt.date | None
    n_decisions: int | None


@dataclasses.dataclass
class EventPage:
    mmad_event_id: int | None
    name: str
    date: dt.date | None
    decision_paths: list[str]


@dataclasses.dataclass
class JudgeCard:
    seq: int                                   # 1..3, order on the page
    mmad_judge_id: int | None                  # None for "Unknown Judge"
    name: str
    rounds: list[tuple[int, int | None, int | None]]   # (round, a_pts, b_pts)
    total_a: int | None
    total_b: int | None

    @property
    def total_known(self) -> bool:
        if self.total_a is None or self.total_b is None:
            return False
        if (self.total_a, self.total_b) in PLACEHOLDER_TOTALS:
            return False
        return True

    @property
    def rounds_known(self) -> bool:
        return bool(self.rounds) and all(a is not None and b is not None for _, a, b in self.rounds)


@dataclasses.dataclass
class MediaScore:
    journalist: str
    outlet: str | None
    a_pts: int | None
    b_pts: int | None
    pick: str | None                           # a | b | draw


@dataclasses.dataclass
class Deduction:
    fighter: str | None                        # a | b (None if the name didn't resolve)
    points: int
    round: int | None
    reason: str | None


@dataclasses.dataclass
class FanScore:
    round: int                                 # 0 = total-score distribution
    score: str
    pick: str                                  # a | b | draw
    pct: float


@dataclasses.dataclass
class DecisionPage:
    mmad_decision_id: int
    fighter_a: str
    fighter_a_id: int | None
    fighter_b: str
    fighter_b_id: int | None
    outcome: str                               # win | draw | nc
    decision_type: str | None
    mmad_event_id: int | None
    event_name: str | None
    date: dt.date | None
    location: str | None
    referee: str | None
    judges: list[JudgeCard]
    media: list[MediaScore]
    fans: list[FanScore]
    deductions: list[Deduction] = dataclasses.field(default_factory=list)
    fan_n: int | None = None
    fan_a: int | None = None
    fan_b: int | None = None
    fan_draw: int | None = None


# --------------------------------------------------------------------- parsing


def _text(el) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", el.get_text(" ").replace("\xa0", " ")).strip()


def _int(s: str | None) -> int | None:
    s = (s or "").strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def _href_id(href: str | None, kind: str) -> int | None:
    m = re.search(rf"(?:^|/){kind}/(\d+)", (href or "").strip())
    return int(m.group(1)) if m else None


def _parse_date(s: str) -> dt.date | None:
    s = re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _find_date(text: str) -> dt.date | None:
    m = re.search(r"([A-Z][a-z]+\.? \d{1,2}, \d{4})", text)
    return _parse_date(m.group(1).replace(".", "")) if m else None


def is_ufc_event_name(name: str) -> bool:
    return bool(_UFC_NAME.match((name or "").strip()))


def is_other_promotion(name: str) -> bool:
    n = (name or "").strip().lower()
    return n.startswith(NON_UFC_PREFIXES)


def parse_year_index(html: str) -> list[EventRef]:
    soup = BeautifulSoup(html, "html.parser")
    out, seen = [], set()
    for a in soup.select('a[href^="event/"]'):
        ev_id = _href_id(a.get("href"), "event")
        if ev_id is None or ev_id in seen:
            continue
        tr = a.find_parent("tr")
        cells = tr.find_all("td") if tr else []
        date = _parse_date(_text(cells[0])) if len(cells) >= 3 else None
        n = _int(_text(cells[2])) if len(cells) >= 3 else None
        seen.add(ev_id)
        out.append(EventRef(ev_id, a.get("href").strip(), _text(a), date, n))
    return out


def parse_event(html: str) -> EventPage:
    soup = BeautifulSoup(html, "html.parser")
    title = _text(soup.title).split("::")[0].strip() if soup.title else ""
    paths, seen = [], set()
    for a in soup.select('a[href^="decision/"]'):
        d_id = _href_id(a.get("href"), "decision")
        if d_id is not None and d_id not in seen:
            seen.add(d_id)
            paths.append(a.get("href").strip())
    canon = soup.find("link", rel="canonical")
    ev_id = _href_id(canon.get("href") if canon else None, "event")
    return EventPage(ev_id, title, _find_date(_text(soup.body) if soup.body else ""), paths)


def _side_of(name: str, a_last: str, b_last: str) -> str | None:
    """Which fighter a last name (as the site prints it in tables) refers to."""
    n = normalize_name(name)
    if not n:
        return None
    if n == "draw":
        return "draw"
    sa, sb = name_similarity(name, a_last), name_similarity(name, b_last)
    if max(sa, sb) < 0.6 or sa == sb:
        return None
    return "a" if sa > sb else "b"


def _judge_cards(soup: BeautifulSoup) -> tuple[list[JudgeCard], str, str]:
    cards: list[JudgeCard] = []
    a_hdr = b_hdr = ""
    for seq, td in enumerate(soup.select("td.judge"), start=1):
        link = td.find("a", href=True)
        j_id = _href_id(link.get("href"), "judge") if link else None
        name = _text(link) if link else _text(td)
        table = td.find_parent("table")
        rounds: list[tuple[int, int | None, int | None]] = []
        total_a = total_b = None
        for tr in table.find_all("tr"):
            cells = [_text(c) for c in tr.find_all("td")]
            if len(cells) != 3:
                continue
            if cells[0].upper() == "ROUND":
                a_hdr, b_hdr = cells[1], cells[2]
            elif cells[0].upper() == "TOTAL":
                total_a, total_b = _int(cells[1]), _int(cells[2])
            elif re.fullmatch(r"\d+", cells[0]):
                rounds.append((int(cells[0]), _int(cells[1]), _int(cells[2])))
        cards.append(JudgeCard(seq, j_id, name, rounds, total_a, total_b))
    return cards, a_hdr, b_hdr


def _media(soup: BeautifulSoup, a_last: str, b_last: str) -> list[MediaScore]:
    head = next((td for td in soup.find_all("td") if _text(td) == "MEDIA SCORES"), None)
    if head is None:
        return []
    out = []
    for tr in head.find_parent("table").select("tr.decision"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        outlet_el = tds[0].find("i")
        outlet = _text(outlet_el) or None
        if outlet_el:
            outlet_el.extract()
        journalist = _text(tds[0])
        m = re.search(r"(\d+)\s*-\s*(\d+)", _text(tds[1]))
        pick = _side_of(_text(tds[2]), a_last, b_last)
        a_pts = b_pts = None
        if m:
            hi, lo = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
            if pick == "b":
                a_pts, b_pts = lo, hi
            elif pick in ("a", "draw"):
                a_pts, b_pts = hi, lo
        out.append(MediaScore(journalist, outlet, a_pts, b_pts, pick))
    return out


_DEDUCTION = re.compile(
    r"(?P<who>.+?)\s+(?:was|were)\s+deducted\s+(?P<pts>\d+)\s+points?"
    r"(?:\s+in\s+round\s+(?P<rnd>\d+))?\s*:?\s*(?P<why>.*)$", re.I)


def _deductions(soup: BeautifulSoup, a_last: str, b_last: str) -> list[Deduction]:
    out = []
    for td in soup.find_all("td"):
        if td.find("td") is not None or "deduct" not in td.get_text().lower():
            continue  # innermost cells only
        txt = _text(td)
        m = _DEDUCTION.search(txt)
        if not m:
            log.warning("unparsed deduction text: %r", txt[:200])
            continue
        out.append(Deduction(_side_of(m.group("who"), a_last, b_last), int(m.group("pts")),
                             int(m.group("rnd")) if m.group("rnd") else None,
                             m.group("why").strip() or None))
    return out


def _fans(soup: BeautifulSoup, html: str, a_name: str, b_name: str, a_last: str,
          b_last: str) -> tuple[list[FanScore], dict]:
    out: list[FanScore] = []
    box = soup.find(id="scorecard_totals")
    if box is not None:
        a_norm, b_norm = normalize_name(a_last), normalize_name(b_last)
        for tr in box.find_all("tr"):
            tds = tr.find_all("td", recursive=False)
            cells = [_text(c) for c in tds]
            # Total distribution: "Van defeats Pantoja" | "48 - 47" | "47.9%" | bar
            if len(cells) == 4 and cells[2].endswith("%") and re.search(r"\d+\s*-\s*\d+", cells[1]):
                first = normalize_name(re.split(r"\b(?:defeats|drew with|draws with)\b",
                                                cells[0])[0])
                pick = ("draw" if "drew" in cells[0] or "draws" in cells[0]
                        else "a" if first == a_norm else "b" if first == b_norm
                        else _side_of(first, a_last, b_last))
                if pick:
                    out.append(FanScore(0, re.sub(r"\s+", "", cells[1]), pick,
                                        float(cells[2].rstrip("%"))))
        for head in box.select("td.top-cell-small"):
            m = re.fullmatch(r"ROUND (\d+)", _text(head))
            if not m:
                continue
            rnd = int(m.group(1))
            for tr in head.find_parent("table").select("tr.decision"):
                cells = [_text(c) for c in tr.find_all("td")]
                if len(cells) == 3 and cells[2].endswith("%"):
                    pick = _side_of(cells[1], a_last, b_last)
                    if pick:
                        out.append(FanScore(rnd, cells[0], pick, float(cells[2].rstrip("%"))))
    counts: dict = {}
    m = re.search(r"<b>(\d+)</b>\s*SCORECARDS SUBMITTED", html)
    if m:
        counts["fan_n"] = int(m.group(1))
    m = re.search(r"data\.addRows\(\[\s*(\[.*?\])\s*\]\)", html, re.S)
    if m:
        for label, n in re.findall(r"\['((?:[^'\\]|\\.)*)',\s*(\d+)\]", m.group(1)):
            side = _side_of(label.replace("\\'", "'"), a_last, b_last)
            if side:
                counts[f"fan_{side}"] = int(n)
    return out, counts


def parse_decision(html: str, mmad_decision_id: int | None = None) -> DecisionPage:
    soup = BeautifulSoup(html, "html.parser")
    a_link = soup.select_one('td.decision-top a[href^="fighter/"]')
    b_link = soup.select_one('td.decision-bottom a[href^="fighter/"]')
    if a_link is None or b_link is None:
        raise ValueError("decision page without both fighters")
    middle = _text(soup.select_one("td.decision-middle")).lower()
    outcome = "draw" if "drew" in middle or "draw" in middle else (
        "win" if "defeat" in middle else "nc")
    ev_cell = soup.select_one("td.decision-top2")
    ev_link = ev_cell.find("a", href=True) if ev_cell else None
    ev_lines = [s.strip() for s in ev_cell.get_text("\n").replace("\xa0", " ").split("\n")
                if s.strip()] if ev_cell else []
    date = next((d for d in (_parse_date(s) for s in ev_lines) if d), None)
    location = ev_lines[-1] if ev_lines and _parse_date(ev_lines[-1]) is None and len(ev_lines) >= 3 else None
    ref = _text(soup.select_one("td.decision-bottom2"))
    ref = re.sub(r"^REFEREE:\s*", "", ref, flags=re.I).strip() or None
    if mmad_decision_id is None:
        canon = soup.find("link", rel="canonical")
        mmad_decision_id = _href_id(canon.get("href") if canon else None, "decision")
        if mmad_decision_id is None:
            m = re.search(r"readCookie\('scorecard(\d+)'\)", html)
            mmad_decision_id = int(m.group(1)) if m else None
    if mmad_decision_id is None:
        raise ValueError("decision id not found; pass it explicitly")

    a_name, b_name = _text(a_link), _text(b_link)
    judges, a_hdr, b_hdr = _judge_cards(soup)
    a_last = a_hdr or a_name.split()[-1]
    b_last = b_hdr or b_name.split()[-1]
    fans, counts = _fans(soup, html, a_name, b_name, a_last, b_last)
    return DecisionPage(
        mmad_decision_id=mmad_decision_id,
        fighter_a=a_name, fighter_a_id=_href_id(a_link.get("href"), "fighter"),
        fighter_b=b_name, fighter_b_id=_href_id(b_link.get("href"), "fighter"),
        outcome=outcome,
        decision_type=_text(soup.select_one("th.event2")) or None,
        mmad_event_id=_href_id(ev_link.get("href"), "event") if ev_link else None,
        event_name=_text(ev_link) if ev_link else None,
        date=date, location=location, referee=ref,
        judges=judges, media=_media(soup, a_last, b_last), fans=fans,
        deductions=_deductions(soup, a_last, b_last),
        fan_n=counts.get("fan_n"), fan_a=counts.get("fan_a"), fan_b=counts.get("fan_b"),
        fan_draw=counts.get("fan_draw"),
    )


# ---------------------------------------------------------------------- client


class MMADClient:
    """Disk cache first, then a throttled GET; stops on any block."""

    def __init__(self, cache_dir: Path | None = DEFAULT_CACHE_DIR,
                 throttle_s: float = MIN_THROTTLE_S, max_requests: int | None = None,
                 offline: bool = False):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.throttle_s = max(MIN_THROTTLE_S, throttle_s)
        self.max_requests = max_requests
        self.offline = offline
        self.n_network = 0
        self.n_cache = 0
        self._mem: dict[str, str] = {}
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _cache_path(self, path: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip("/")) or "root"
        if len(safe) > 150:
            safe = safe[:100] + "_" + hashlib.sha1(path.encode()).hexdigest()[:16]
        return self.cache_dir / f"{safe}.html"

    @staticmethod
    def _norm(path: str) -> str:
        return "/" + path.strip().lstrip("/")

    def cached(self, path: str) -> bool:
        path = self._norm(path)
        cp = self._cache_path(path)
        return path in self._mem or bool(cp and cp.exists())

    def get(self, path: str, max_age_s: float | None = None) -> str:
        path = self._norm(path)
        if path in self._mem:
            return self._mem[path]
        cp = self._cache_path(path)
        if cp and cp.exists() and (max_age_s is None or time.time() - cp.stat().st_mtime < max_age_s):
            self.n_cache += 1
            return cp.read_text(encoding="utf-8")
        if self.offline:
            raise FileNotFoundError(f"offline and not cached: {path}")
        if self.max_requests is not None and self.n_network >= self.max_requests:
            raise RuntimeError(f"request cap reached ({self.max_requests})")
        wait = self._last + self.throttle_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        for backoff in (30, 120, None):
            try:
                resp = self.session.get(BASE_URL + path, timeout=30)
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                if backoff is None:
                    raise MMADBlocked(f"repeated connection failures on {path}: {e}") from e
                log.warning("connection error on %s (%s); backing off %ds", path, e, backoff)
                time.sleep(backoff)
        self._last = time.monotonic()
        self.n_network += 1
        body = resp.text
        if resp.status_code in (403, 429, 503) or re.search(
                r"<title>\s*Just a moment|cf-chl|challenge-platform|captcha", body[:5000], re.I):
            raise MMADBlocked(f"{resp.status_code} on {path}; stopping")
        resp.raise_for_status()
        self._mem[path] = body
        if cp:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(body, encoding="utf-8")
        return body

    def year_index(self, year: int, max_age_s: float | None = None) -> list[EventRef]:
        # Past years are frozen; the current year grows, so callers pass a max age.
        return parse_year_index(self.get(f"/decisions-by-event/{year}/", max_age_s=max_age_s))

    def event(self, path: str, max_age_s: float | None = None) -> EventPage:
        return parse_event(self.get(path, max_age_s=max_age_s))

    def decision(self, path: str, max_age_s: float | None = None) -> DecisionPage:
        return parse_decision(self.get(path, max_age_s=max_age_s), _href_id(path, "decision"))


# ------------------------------------------------------------------- DB helpers


def _engine():
    from app.database import engine
    return engine


def _schema(engine) -> str:
    return "" if engine.dialect.name == "sqlite" else "ufc."


def ensure_tables(engine=None) -> None:
    """Create the scorecard tables if missing (checkfirst; no-op afterwards)."""
    from app.models.ufc import (
        MMADDecision, MMADDeduction, MMADFanScore, MMADFighter, MMADMediaScore, UFCJudge,
        UFCJudgeAlias, UFCJudgeScorecard,
    )
    engine = engine or _engine()
    for model in (UFCJudge, UFCJudgeAlias, MMADFighter, MMADDecision, UFCJudgeScorecard,
                  MMADMediaScore, MMADFanScore, MMADDeduction):
        model.__table__.create(bind=engine, checkfirst=True)


def load_db_decisions(engine, start: dt.date, end: dt.date) -> list[dict]:
    """UFC decision bouts (incl. decision draws) in [start, end] with names and totals."""
    from sqlalchemy import text
    sch = _schema(engine)
    sql = text(f"""
        SELECT f.id, COALESCE(f.date, e.date) AS fight_date, e.name AS event_name,
               f.red_fighter_id, f.blue_fighter_id, f.winner_id, f.method, f.details,
               f.referee,
               rf.first_name || ' ' || rf.last_name AS red,
               bf.first_name || ' ' || bf.last_name AS blue
        FROM {sch}ufc_fights f
        JOIN {sch}ufc_events e ON e.id = f.event_id
        JOIN {sch}ufc_fighters rf ON rf.id = f.red_fighter_id
        JOIN {sch}ufc_fighters bf ON bf.id = f.blue_fighter_id
        WHERE f.method LIKE 'Decision%'
          AND COALESCE(f.date, e.date) BETWEEN :s AND :e
    """)
    with engine.connect() as conn:
        out = []
        for r in conn.execute(sql, {"s": start, "e": end}).mappings():
            d = dict(r)
            if isinstance(d["fight_date"], str):
                d["fight_date"] = dt.date.fromisoformat(d["fight_date"][:10])
            out.append(d)
        return out


def db_ufc_event_dates(engine) -> set[dt.date]:
    from sqlalchemy import text
    sch = _schema(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT DISTINCT date FROM {sch}ufc_events WHERE date IS NOT NULL"))
        return {(dt.date.fromisoformat(d[:10]) if isinstance(d, str) else d) for (d,) in rows}


# --------------------------------------------------------------- verification


def _db_cards(details: str | None, draw: bool) -> list[tuple[int, int]]:
    cards = parse_ufcstats_cards(details)
    return [tuple(sorted((w, l), reverse=True)) if draw else (w, l) for _, w, l in cards]


def verify(page: DecisionPage, fight: dict, swapped: bool) -> tuple[str, list[str]]:
    """-> (match_status, flags) for a decision matched to a DB fight."""
    flags: list[str] = []
    a_corner = fight["blue_fighter_id"] if swapped else fight["red_fighter_id"]
    if page.outcome == "win":
        winner_ok = fight["winner_id"] == a_corner
    elif page.outcome == "draw":
        winner_ok = fight["winner_id"] is None
    else:
        winner_ok = False
    if not winner_ok:
        flags.append("winner")
    for j in page.judges:
        if j.rounds_known and j.total_known and (
                sum(r[1] for r in j.rounds) != j.total_a or sum(r[2] for r in j.rounds) != j.total_b):
            flags.append("round_sum")
            break
    if any(d.fighter is None for d in page.deductions):
        flags.append("deduction_side")
    draw = page.outcome == "draw"
    db = _db_cards(fight.get("details"), draw)
    known = [j for j in page.judges if j.total_known]
    site = [tuple(sorted((j.total_a, j.total_b), reverse=True)) if draw else (j.total_a, j.total_b)
            for j in known]
    totals_compared = bool(db) and len(known) == len(page.judges) and bool(known)
    if totals_compared and Counter(db) != Counter(site):
        flags.append("totals")
    if page.referee and fight.get("referee") and name_similarity(page.referee, fight["referee"]) < 0.8:
        flags.append("referee")  # informational: never decides the status
    hard = [f for f in flags if f != "referee"]
    if hard:
        return "review", flags
    return ("verified" if totals_compared else "name_only"), flags


# ------------------------------------------------------------------------ load


def _norm_judge(name: str) -> str:
    return normalize_name(name)


class _Judges:
    """In-memory view of ufc_judges + aliases, written back with the session."""

    def __init__(self, db):
        from app.models.ufc import UFCJudge, UFCJudgeAlias
        self.db = db
        self.by_mmad = {j.mmad_judge_id: j for j in db.query(UFCJudge).all() if j.mmad_judge_id}
        self.all = db.query(UFCJudge).all()
        self.aliases = {a.alias_norm: a for a in db.query(UFCJudgeAlias).all()}

    def for_mmad(self, mmad_id: int | None, name: str):
        from app.models.ufc import UFCJudge
        if mmad_id is None:
            return None
        j = self.by_mmad.get(mmad_id)
        if j is None:
            # A UFCStats-only row for the same person (created before mmadecisions posted)
            # is adopted rather than duplicated.
            nn = _norm_judge(name)
            j = next((x for x in self.all if x.mmad_judge_id is None and x.name_norm == nn), None)
            if j is None:
                j = UFCJudge(name=name, name_norm=nn)
                self.db.add(j)
                self.all.append(j)
            j.mmad_judge_id = mmad_id
            j.name = name
            self.db.flush()
            self.by_mmad[mmad_id] = j
        return j

    def learn_alias(self, alias: str, judge) -> None:
        from app.models.ufc import UFCJudgeAlias
        an = _norm_judge(alias)
        a = self.aliases.get(an)
        if a is None:
            a = UFCJudgeAlias(alias_norm=an, alias=alias, judge_id=judge.id, n_fights=0)
            self.db.add(a)
            self.aliases[an] = a
        if a.judge_id == judge.id:
            a.n_fights = (a.n_fights or 0) + 1

    def for_ufcstats(self, alias: str):
        """Alias first, then a near-exact name, else a new UFCStats-only judge."""
        from app.models.ufc import UFCJudge
        an = _norm_judge(alias)
        a = self.aliases.get(an)
        if a is not None:
            return self.db.get(UFCJudge, a.judge_id)
        best = max(self.all, key=lambda j: name_similarity(alias, j.name), default=None)
        if best is not None and name_similarity(alias, best.name) >= 0.95:
            return best
        j = UFCJudge(name=alias, name_norm=an)
        self.db.add(j)
        self.db.flush()
        self.all.append(j)
        return j


def _link_aliases(judges: _Judges, page: DecisionPage, fight: dict) -> None:
    """Within a verified fight, pair each UFCStats judge name with the mmad judge that
    gave the same total and has the most similar name."""
    if page.outcome != "win":
        return
    pool = [j for j in page.judges if j.mmad_judge_id and j.total_known]
    for alias, w, l in parse_ufcstats_cards(fight.get("details")):
        if not normalize_name(alias):
            continue
        cands = [j for j in pool if (j.total_a, j.total_b) == (w, l)]
        if not cands:
            continue
        best = max(cands, key=lambda j: name_similarity(alias, j.name))
        if name_similarity(alias, best.name) >= 0.75 or len(cands) == 1 and len(pool) == 3 and \
                name_similarity(alias, best.name) >= 0.5:
            judges.learn_alias(alias, judges.for_mmad(best.mmad_judge_id, best.name))
            pool.remove(best)


def write_ufcstats_cards(db, judges: _Judges, fights: list[dict], replace: bool = False) -> int:
    """Round-0 rows from ufc_fights.details for decided bouts (winner known)."""
    from app.models.ufc import UFCJudgeScorecard
    ids = [f["id"] for f in fights]
    have = {fid for (fid,) in db.query(UFCJudgeScorecard.fight_id).filter(
        UFCJudgeScorecard.source == "ufcstats", UFCJudgeScorecard.fight_id.in_(ids)).distinct()} \
        if ids else set()
    n = 0
    for f in fights:
        if f["winner_id"] is None:
            continue
        cards = parse_ufcstats_cards(f.get("details"))
        if not cards:
            continue
        if f["id"] in have:
            if not replace:
                continue
            db.query(UFCJudgeScorecard).filter(UFCJudgeScorecard.source == "ufcstats",
                                               UFCJudgeScorecard.fight_id == f["id"]).delete()
        red_won = f["winner_id"] == f["red_fighter_id"]
        for seq, (alias, w, l) in enumerate(cards, start=1):
            j = judges.for_ufcstats(alias) if normalize_name(alias) else None  # unnamed card
            db.add(UFCJudgeScorecard(
                source="ufcstats", fight_id=f["id"], judge_id=j.id if j else None,
                judge_seq=seq, round=0,
                red_pts=w if red_won else l, blue_pts=l if red_won else w))
            n += 1
    return n


def deductions_by_round(page: DecisionPage) -> dict[int, tuple[int, int]]:
    """{round: (a_points_lost, b_points_lost)}, round 0 = whole fight. A deduction with
    no round counts toward round 0 only."""
    out: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for d in page.deductions:
        if d.fighter not in ("a", "b"):
            continue
        k = 0 if d.fighter == "a" else 1
        out[0][k] += d.points
        if d.round:
            out[d.round][k] += d.points
    return {r: (v[0], v[1]) for r, v in out.items()}


def _decision_row(page: DecisionPage, is_ufc: bool, match: dict | None, status: str,
                  flags: list[str]) -> dict:
    return dict(
        mmad_decision_id=page.mmad_decision_id, mmad_event_id=page.mmad_event_id,
        event_name=page.event_name, is_ufc=is_ufc, date=page.date, location=page.location,
        fighter_a_mmad_id=page.fighter_a_id, fighter_b_mmad_id=page.fighter_b_id,
        fighter_a_name=page.fighter_a, fighter_b_name=page.fighter_b, outcome=page.outcome,
        decision_type=page.decision_type, referee=page.referee,
        fight_id=match["fight_id"] if match else None,
        swapped=match["swapped"] if match else None,
        match_score=match["score"] if match else None,
        match_status=status, flags=",".join(flags) or None,
        fan_n=page.fan_n, fan_a=page.fan_a, fan_b=page.fan_b, fan_draw=page.fan_draw,
    )


def load_pages(db, pages: list[tuple[DecisionPage, bool]], engine=None,
               window_days: int = 1, replace_ufcstats: bool = False) -> dict:
    """Match, verify and write decision pages. ``pages``: (page, is_ufc_event).
    replace_ufcstats rewrites existing UFCStats total rows (bulk reloads after a parser
    change); the nightly sync only adds missing ones."""
    from app.models.ufc import (
        MMADDecision, MMADDeduction, MMADFanScore, MMADMediaScore, UFCJudgeScorecard,
    )
    engine = engine or db.get_bind()
    stats: Counter = Counter()
    review: list[dict] = []
    dated = [p for p, _ in pages if p.date]
    fights: list[dict] = []
    if dated:
        lo = min(p.date for p in dated) - dt.timedelta(days=window_days)
        hi = max(p.date for p in dated) + dt.timedelta(days=window_days)
        fights = load_db_decisions(engine, lo, hi)
    by_id = {f["id"]: f for f in fights}
    ufc_pages = [p for p, u in pages if u and p.date]
    matches = match_matchups(
        [{"bfo_matchup_id": p.mmad_decision_id, "fighter_a": p.fighter_a,
          "fighter_b": p.fighter_b, "event_date": p.date} for p in ufc_pages],
        fights, window_days=window_days)
    # Never steal a fight already held by another decision id from an earlier load.
    taken = {fid: did for did, fid in db.query(MMADDecision.mmad_decision_id, MMADDecision.fight_id)
             .filter(MMADDecision.fight_id.isnot(None))}
    judges = _Judges(db)

    for page, is_ufc in pages:
        did = page.mmad_decision_id
        m = matches.get(did) if is_ufc else None
        if m and taken.get(m["fight_id"], did) != did:
            m = None
        if not is_ufc:
            status, flags = "non_ufc", []
        elif m is None:
            status, flags = "unmatched", []
        else:
            status, flags = verify(page, by_id[m["fight_id"]], m["swapped"])
        stats[status] += 1
        if status in ("review", "unmatched"):
            review.append({"mmad_decision_id": did, "date": page.date, "event": page.event_name,
                           "fighter_a": page.fighter_a, "fighter_b": page.fighter_b,
                           "status": status, "flags": ",".join(flags),
                           "fight_id": m["fight_id"] if m else None,
                           "db_red": m["db_red"] if m else None,
                           "db_blue": m["db_blue"] if m else None})
        row = _decision_row(page, is_ufc, m if status not in ("unmatched", "non_ufc") else None,
                            status, flags)
        rec = db.query(MMADDecision).filter_by(mmad_decision_id=did).one_or_none()
        if rec is None:
            db.add(MMADDecision(**row))
        else:
            for k, v in row.items():
                setattr(rec, k, v)

        # Children: replace wholesale (pages can be corrected on the site).
        for model in (UFCJudgeScorecard, MMADMediaScore, MMADFanScore, MMADDeduction):
            q = db.query(model).filter(model.mmad_decision_id == did)
            if model is UFCJudgeScorecard:
                q = q.filter(UFCJudgeScorecard.source == "mmad")
            q.delete(synchronize_session=False)
        db.flush()
        linked = status in ("verified", "name_only")
        fight_id = m["fight_id"] if (m and linked) else None
        swapped = bool(m["swapped"]) if (m and linked) else False
        ded = deductions_by_round(page)
        for j in page.judges:
            judge = judges.for_mmad(j.mmad_judge_id, j.name)
            rows = [(r, a, b) for r, a, b in j.rounds]
            rows.append((0, j.total_a if j.total_known else None,
                         j.total_b if j.total_known else None))
            for rnd, a, b in rows:
                da, db_ = ded.get(rnd, (0, 0))
                db.add(UFCJudgeScorecard(
                    source="mmad", mmad_decision_id=did, fight_id=fight_id,
                    judge_id=judge.id if judge else None, judge_seq=j.seq, round=rnd,
                    red_pts=b if swapped else a, blue_pts=a if swapped else b,
                    red_ded=db_ if swapped else da, blue_ded=da if swapped else db_))
        for d in page.deductions:
            db.add(MMADDeduction(mmad_decision_id=did, round=d.round, fighter=d.fighter,
                                 points=d.points, reason=(d.reason or None) and d.reason[:200]))
        for ms in page.media:
            db.add(MMADMediaScore(mmad_decision_id=did, journalist=ms.journalist[:120],
                                  outlet=(ms.outlet or None) and ms.outlet[:120],
                                  a_pts=ms.a_pts, b_pts=ms.b_pts, pick=ms.pick))
        for fs in page.fans:
            db.add(MMADFanScore(mmad_decision_id=did, round=fs.round, score=fs.score[:10],
                                pick=fs.pick, pct=fs.pct))
        if status == "verified":
            _link_aliases(judges, page, by_id[m["fight_id"]])
        db.flush()

    stats["ufcstats_rows"] = write_ufcstats_cards(db, judges, fights, replace=replace_ufcstats)
    db.flush()
    rebuild_fighter_links(db)
    return {"stats": dict(stats), "review": review}


def rebuild_fighter_links(db) -> Counter:
    """mmad fighter id -> ufc_fighters, from verified decisions only; conflicts unlinked."""
    from sqlalchemy import text
    from app.models.ufc import MMADDecision, MMADFighter
    sch = _schema(db.get_bind())
    rows = db.execute(text(f"""
        SELECT d.fighter_a_mmad_id, d.fighter_b_mmad_id, d.fighter_a_name, d.fighter_b_name,
               d.swapped, f.red_fighter_id, f.blue_fighter_id
        FROM {sch}mmad_decisions d JOIN {sch}ufc_fights f ON f.id = d.fight_id
        WHERE d.match_status = 'verified'
    """)).all()
    names: dict[int, str] = {}
    links: dict[int, Counter] = defaultdict(Counter)
    for a_id, b_id, a_name, b_name, swapped, red, blue in rows:
        a_db, b_db = (blue, red) if swapped else (red, blue)
        for mid, name, fid in ((a_id, a_name, a_db), (b_id, b_name, b_db)):
            if mid is not None:
                names[mid] = name
                links[mid][fid] += 1
    reverse: dict[int, set] = defaultdict(set)
    for mid, c in links.items():
        for fid in c:
            reverse[fid].add(mid)
    for d in db.query(MMADDecision.fighter_a_mmad_id, MMADDecision.fighter_a_name,
                      MMADDecision.fighter_b_mmad_id, MMADDecision.fighter_b_name):
        for mid, name in ((d[0], d[1]), (d[2], d[3])):
            if mid is not None:
                names.setdefault(mid, name)
    existing = {f.mmad_fighter_id: f for f in db.query(MMADFighter).all()}
    out: Counter = Counter()
    for mid, name in names.items():
        c = links.get(mid)
        if not c:
            status, fid, n = "unlinked", None, 0
        elif len(c) > 1 or any(len(reverse[f]) > 1 for f in c):
            status, fid, n = "conflict", None, sum(c.values())
        else:
            (fid, n), = c.items()
            status = "verified"
        out[status] += 1
        rec = existing.get(mid)
        if rec is None:
            db.add(MMADFighter(mmad_fighter_id=mid, name=name[:200], ufc_fighter_id=fid,
                               match_status=status, n_fights_matched=n))
        else:
            rec.name, rec.ufc_fighter_id, rec.match_status, rec.n_fights_matched = \
                name[:200], fid, status, n
    db.flush()
    return out


# -------------------------------------------------------------------- crawling


def candidate_events(refs: list[EventRef], ufc_dates: set[dt.date],
                     all_promotions: bool = False) -> list[tuple[EventRef, bool]]:
    """-> [(event, is_ufc)]. UFC by name, or any non-other-promotion card on a date
    (+-1 day) with a DB UFC event; with all_promotions every event is kept."""
    out = []
    for r in refs:
        near = r.date is not None and any(
            (r.date + dt.timedelta(days=k)) in ufc_dates for k in (-1, 0, 1))
        ufc = is_ufc_event_name(r.name) or (near and not is_other_promotion(r.name))
        if ufc or all_promotions:
            out.append((r, ufc))
    return out


def fetch(client: MMADClient, years: list[int], ufc_dates: set[dt.date],
          all_promotions: bool = False) -> dict:
    """Fill the cache: year indexes -> candidate events -> their decision pages."""
    today = dt.date.today()
    n_ev = n_dec = 0
    for y in years:
        refs = client.year_index(y, max_age_s=6 * 3600 if y >= today.year else None)
        for ref, _ in candidate_events(refs, ufc_dates, all_promotions):
            recent = ref.date is None or ref.date >= today - dt.timedelta(days=14)
            ev = client.event(ref.path, max_age_s=24 * 3600 if recent else None)
            n_ev += 1
            for p in ev.decision_paths:
                if not client.cached(p) or recent:
                    client.get(p, max_age_s=24 * 3600 if recent else None)
                n_dec += 1
        log.info("year %d done: %d events, %d decisions so far (%d network, %d cached)",
                 y, n_ev, n_dec, client.n_network, client.n_cache)
    return {"events": n_ev, "decisions": n_dec, "network": client.n_network}


def pages_from_cache(client: MMADClient, years: list[int], ufc_dates: set[dt.date],
                     all_promotions: bool = False) -> tuple[list[tuple[DecisionPage, bool]], int]:
    out, missing = [], 0
    for y in years:
        try:
            refs = client.year_index(y)
        except FileNotFoundError:
            missing += 1
            continue
        for ref, is_ufc in candidate_events(refs, ufc_dates, all_promotions):
            try:
                ev = client.event(ref.path)
            except FileNotFoundError:
                missing += 1
                continue
            for p in ev.decision_paths:
                try:
                    page = client.decision(p)
                except FileNotFoundError:
                    missing += 1
                    continue
                except ValueError as e:
                    log.warning("unparseable %s: %s", p, e)
                    continue
                if page.date is None:
                    page.date = ref.date
                out.append((page, is_ufc))
    return out, missing


def sync(db, client: MMADClient, days: int = 30, today: dt.date | None = None) -> dict:
    """Nightly: judge totals from UFCStats for recent decisions, then round cards from
    mmadecisions for recent decisions that don't have verified ones yet. Makes no
    request when nothing is pending."""
    from app.models.ufc import MMADDecision
    today = today or dt.date.today()
    engine = db.get_bind()
    ensure_tables(engine)
    fights = load_db_decisions(engine, today - dt.timedelta(days=days), today)
    judges = _Judges(db)
    n_ufcstats = write_ufcstats_cards(db, judges, fights)
    db.flush()
    ids = [f["id"] for f in fights]
    done = {fid for (fid,) in db.query(MMADDecision.fight_id).filter(
        MMADDecision.fight_id.in_(ids),
        MMADDecision.match_status.in_(("verified", "name_only")))} if ids else set()
    pending = [f for f in fights if f["id"] not in done]
    result = {"decisions_recent": len(fights), "ufcstats_rows": n_ufcstats,
              "pending": len(pending), "loaded": 0, "network": 0}
    if not pending:
        return result
    pend_dates = {f["fight_date"] for f in pending}
    years = sorted({d.year for d in pend_dates})
    pages: list[tuple[DecisionPage, bool]] = []
    try:
        for y in years:
            refs = client.year_index(y, max_age_s=0)
            for ref, is_ufc in candidate_events(refs, pend_dates):
                if not is_ufc:
                    continue
                ev = client.event(ref.path, max_age_s=24 * 3600)
                for p in ev.decision_paths:
                    try:
                        page = client.decision(p, max_age_s=24 * 3600)
                    except ValueError as e:
                        log.warning("unparseable %s: %s", p, e)
                        continue
                    if page.date is None:
                        page.date = ref.date
                    pages.append((page, True))
    except (RuntimeError, MMADBlocked) as e:
        # Request cap or a block: load what we have; the rest is retried tomorrow.
        log.warning("stopping early (%s); loading %d pages fetched so far", e, len(pages))
        result["stopped"] = str(e)[:200]
    if pages:
        res = load_pages(db, pages, engine)
        result["loaded"] = len(pages)
        result["status"] = res["stats"]
        result["review"] = res["review"]
    result["network"] = client.n_network
    stale = [f for f in pending if f["fight_date"] < today - dt.timedelta(days=days - 1)]
    for f in stale:
        log.warning("no mmadecisions card after %d days: fight %s (%s vs %s)", days, f["id"],
                    f["red"], f["blue"])
    return result


# ------------------------------------------------------------------------- CLI


def _years(spec: str | None) -> list[int]:
    if not spec:
        return list(range(FIRST_YEAR, dt.date.today().year + 1))
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _require_local(engine, allow_remote: bool) -> None:
    if engine.dialect.name == "sqlite" or allow_remote:
        return
    host = engine.url.host or "localhost"
    if host not in ("localhost", "127.0.0.1", "::1"):
        raise SystemExit(f"refusing to bulk-load into non-local DB (host={host}); "
                         "pass --allow-remote if that is really intended")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch", action="store_true", help="fill the page cache")
    mode.add_argument("--load", action="store_true", help="parse the cache and write the DB")
    mode.add_argument("--sync", action="store_true", help="nightly: recent decisions only")
    ap.add_argument("--years", help="e.g. 2001-2026 (default: all)")
    ap.add_argument("--days", type=int, default=30, help="--sync window")
    ap.add_argument("--all-promotions", action="store_true")
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--no-cache", action="store_true", help="in-memory only (CI)")
    ap.add_argument("--review", type=Path, default=Path("data/mmad/review.csv"))
    ap.add_argument("--allow-remote", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from app.database import SessionLocal
    engine = _engine()
    client = MMADClient(None if args.no_cache else args.cache_dir,
                        max_requests=args.max_requests, offline=args.load)
    years = _years(args.years)

    if args.fetch:
        res = fetch(client, years, db_ufc_event_dates(engine), args.all_promotions)
        print(f"fetched: {res}")
        return 0

    ensure_tables(engine)
    db = SessionLocal()
    try:
        if args.load:
            _require_local(engine, args.allow_remote)
            pages, missing = pages_from_cache(client, years, db_ufc_event_dates(engine),
                                              args.all_promotions)
            log.info("%d decision pages parsed (%d pages missing from cache)", len(pages), missing)
            res = load_pages(db, pages, engine, replace_ufcstats=True)
            db.commit()
            _write_csv(args.review, res["review"])
            print(f"status: {res['stats']}")
            print(f"fighters: {dict(rebuild_fighter_links(db))}")
            if res["review"]:
                print(f"review rows: {len(res['review'])} -> {args.review}")
        else:
            try:
                res = sync(db, client, days=args.days)
            except MMADBlocked as e:
                db.commit()  # keep the UFCStats totals written before the block
                log.error("mmadecisions blocked: %s", e)
                return 0
            db.commit()
            review = res.pop("review", [])
            print(f"sync: {res}")
            for r in review:
                print(f"  review: {r}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
