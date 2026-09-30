"""Sherdog Fight Finder scraper: full professional MMA records for UFC fighters.

Purpose: give every UFC fighter (debutants especially) a rating built from their whole
pro career -- who they beat / lost to, when, how, and in which promotion. Results only;
no striking/grappling stats.

Politeness (non-negotiable):
  * robots.txt is fetched once, cached, and checked before every request.
  * Descriptive User-Agent, >= 2 s between network requests (MIN_DELAY_S is a floor).
  * Every raw HTML response is cached under data/sherdog/cache/ -- a re-run never
    re-fetches a page it already has.
  * A hard per-run request budget (--max-requests, default 15) so nobody starts a full
    crawl by accident.
  * 403 / 429 / Cloudflare challenge / captcha => SherdogBlocked is raised and the run
    stops. We never try to evade a block.

This module never touches the production database. It reads our fighter list read-only
from --db-url (default postgresql://localhost/alocks_local) and refuses non-local hosts
unless --allow-remote-db is passed. It deliberately does NOT load app settings/.env.

Outputs (data/sherdog/):
  resolved.csv        our ufc_fighters row -> sherdog id, with match status/evidence
  fighters.jsonl      one parsed Sherdog profile per line (UFC fighters + opponents)
  bouts.csv           one row per (fighter, pro bout); both sides appear once each
                      fighter's page is fetched
  crawl_plan.json     output of --plan: frontier sizes and crawl-time estimate

CLI:
  python -m app.services.ufc.sherdog_scraper --resolve N [--names "Tom Aspinall,..."]
  python -m app.services.ufc.sherdog_scraper --fetch N
  python -m app.services.ufc.sherdog_scraper --expand-opponents --max-pages N
  python -m app.services.ufc.sherdog_scraper --plan
  python -m app.services.ufc.sherdog_scraper --reparse   # rebuild outputs from cache only
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import logging
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup

log = logging.getLogger("sherdog")

BASE_URL = "https://www.sherdog.com"
SEARCH_PATH = "/stats/fightfinder?SearchTxt="
USER_AGENT = (
    "ALocksResearchBot/0.1 (UFC fight-prediction research; low-rate, cached; "
    "respects robots.txt)"
)
MIN_DELAY_S = 2.0
DEFAULT_MAX_REQUESTS = 15
DEFAULT_DB_URL = "postgresql://localhost/alocks_local"

BACKEND_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = BACKEND_ROOT / "data" / "sherdog"
CACHE_DIR = DATA_DIR / "cache"

FIGHTER_ID_RE = re.compile(r"/fighter/[^/?#]*?-(\d+)/?$")
EVENT_ID_RE = re.compile(r"/events/[^/?#]*?-(\d+)/?$")


# --------------------------------------------------------------------------- errors


class SherdogBlocked(RuntimeError):
    """Sherdog refused us (403/429/challenge/captcha). Stop; do not retry around it."""


class RobotsDisallowed(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


# --------------------------------------------------------------------------- fetcher


def _cache_name(url: str) -> str:
    """Stable on-disk name. Fighter pages are keyed by numeric id (slugs get renamed)."""
    path = urlparse(url).path
    m = FIGHTER_ID_RE.search(path)
    if m:
        return f"fighter_{m.group(1)}.html"
    q = urlparse(url).query
    if path.rstrip("/") == "/stats/fightfinder" and q.startswith("SearchTxt="):
        return f"search_{q.split('=', 1)[1].lower()}.html"
    safe = re.sub(r"[^A-Za-z0-9._+-]+", "_", (path + ("_" + q if q else "")).strip("/"))
    return f"page_{safe[:180]}.html"


_BLOCK_MARKERS = (
    "cf-chl-",  # Cloudflare challenge assets
    "challenge-platform",
    "Just a moment...",
    "Attention Required! | Cloudflare",
    "g-recaptcha",
    "h-captcha",
    "px-captcha",
)


def looks_blocked(status: int, html: str) -> bool:
    if status in (403, 429, 503):
        return True
    head = html[:20000]
    return any(m in head for m in _BLOCK_MARKERS)


class PoliteFetcher:
    """Cache-first, robots-aware, throttled, budgeted HTTP GET."""

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        delay_s: float = MIN_DELAY_S,
        offline: bool = False,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_requests = max_requests
        self.delay_s = max(delay_s, MIN_DELAY_S)
        self.offline = offline
        self.requests_made = 0
        self._last_request = 0.0
        self._session = None
        self._robots: RobotFileParser | None = None

    # -- plumbing
    @property
    def session(self):
        if self._session is None:
            import requests

            s = requests.Session()
            s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html"})
            self._session = s
        return self._session

    def _throttle(self) -> None:
        wait = self._last_request + self.delay_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _raw_get(self, url: str):
        if self.offline:
            raise BudgetExhausted(f"offline mode, not cached: {url}")
        if self.requests_made >= self.max_requests:
            raise BudgetExhausted(f"request budget {self.max_requests} used up")
        self._throttle()
        self.requests_made += 1
        resp = self.session.get(url, timeout=30, allow_redirects=True)
        log.info("GET %s -> %s (%d/%d)", url, resp.status_code, self.requests_made, self.max_requests)
        return resp

    # -- robots
    def robots(self) -> RobotFileParser:
        if self._robots is None:
            path = self.cache_dir / "robots.txt"
            if not path.exists():
                resp = self._raw_get(BASE_URL + "/robots.txt")
                if looks_blocked(resp.status_code, resp.text):
                    raise SherdogBlocked(f"robots.txt fetch blocked: HTTP {resp.status_code}")
                path.write_text(resp.text if resp.status_code == 200 else "", encoding="utf-8")
            rp = RobotFileParser()
            rp.parse(path.read_text(encoding="utf-8").splitlines())
            self._robots = rp
        return self._robots

    def allowed(self, url: str) -> bool:
        return self.robots().can_fetch(USER_AGENT, url)

    # -- public
    def cached(self, url: str) -> str | None:
        p = self.cache_dir / _cache_name(url)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def get(self, url: str) -> str:
        url = urljoin(BASE_URL, url)
        hit = self.cached(url)
        if hit is not None:
            return hit
        if not self.allowed(url):
            raise RobotsDisallowed(url)
        rp = self.robots()
        crawl_delay = rp.crawl_delay(USER_AGENT)
        if crawl_delay:
            self.delay_s = max(self.delay_s, float(crawl_delay))
        # Transient network failures (read timeouts, connection resets) are retried with
        # a growing pause; after the last attempt the error propagates as before. A single
        # slow response ended an overnight crawl before this.
        import requests
        for attempt, pause in enumerate((30, 120, 300)):
            try:
                resp = self._raw_get(url)
                if resp.status_code < 500:
                    break
                log.warning("HTTP %d on %s; retrying in %ds", resp.status_code, url, pause)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                log.warning("network error on %s (%s); retrying in %ds", url, e.__class__.__name__, pause)
            time.sleep(pause)
        else:
            resp = self._raw_get(url)
        if looks_blocked(resp.status_code, resp.text):
            raise SherdogBlocked(f"HTTP {resp.status_code} / challenge on {url}; stopping")
        if resp.status_code == 404:
            html = ""
        elif resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} on {url}")
        else:
            html = resp.text
        # A renamed fighter slug redirects; the id (cache key) is what matters.
        (self.cache_dir / _cache_name(url)).write_text(html, encoding="utf-8")
        with open(self.cache_dir / "fetch_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"url": url, "final_url": resp.url, "status": resp.status_code,
                                 "fetched_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")}) + "\n")
        return html


# --------------------------------------------------------------------------- parsing


def sherdog_id_from_url(href: str | None) -> int | None:
    if not href:
        return None
    m = FIGHTER_ID_RE.search(urlparse(href).path)
    return int(m.group(1)) if m else None


def event_id_from_url(href: str | None) -> int | None:
    if not href:
        return None
    m = EVENT_ID_RE.search(urlparse(href).path)
    return int(m.group(1)) if m else None


def _text(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip() if el else ""


def _parse_date(s: str, fmts: Iterable[str]) -> str | None:
    s = re.sub(r"\s+", " ", (s or "").strip())
    for f in fmts:
        try:
            return dt.datetime.strptime(s, f).date().isoformat()
        except ValueError:
            continue
    return None


def parse_bout_date(s: str) -> str | None:
    """'Oct / 25 / 2025' -> '2025-10-25'."""
    return _parse_date(s, ("%b / %d / %Y", "%B / %d / %Y", "%m / %d / %Y"))


def parse_birth_date(s: str) -> str | None:
    """'Apr 11, 1993' -> '1993-04-11'. Sherdog uses 'N/A' when unknown."""
    return _parse_date(s, ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"))


# Longest/most-specific first. Matched against the event-name prefix (before " - ").
_PROMOTION_PATTERNS: list[tuple[str, str]] = [
    (r"^(UFC|The Ultimate Fighter|TUF)\b", "UFC"),
    (r"^Dana White'?s (Contender Series|Tuesday Night)", "DWCS"),
    (r"^Road to UFC", "Road to UFC"),
    (r"^(Bellator|BMMA)\b", "Bellator"),
    (r"^Rizin\b", "Rizin"),
    (r"^(PFL|Professional Fighters League)\b", "PFL"),
    (r"^WSOF\b|^World Series of Fighting", "PFL"),  # WSOF became PFL
    (r"^(CW|Cage Warriors)\b", "Cage Warriors"),
    (r"^(LFA|Legacy Fighting Alliance)\b", "LFA"),
    (r"^(Legacy FC|Legacy Fighting Championship)\b", "Legacy FC"),
    (r"^RFA\b|^Resurrection Fighting Alliance", "RFA"),
    (r"^ONE\b|^ONE Championship|^ONE FC\b", "ONE"),
    (r"^KSW\b", "KSW"),
    (r"^(ACA|ACB|Absolute Championship (Akhmat|Berkut|Berkov))\b", "ACA"),  # ACB renamed ACA
    (r"^(M-1|M1)\b", "M-1"),
    (r"^(Rizin|RIZIN)\b", "Rizin"),
    (r"^(Brave CF|Brave Combat Federation|BRAVE)\b", "Brave CF"),
    (r"^UAE Warriors\b", "UAE Warriors"),
    (r"^(Invicta|Invicta FC)\b", "Invicta FC"),
    (r"^(CFFC|Cage Fury)\b", "CFFC"),
    (r"^(Jungle Fight)\b", "Jungle Fight"),
    (r"^(Pancrase)\b", "Pancrase"),
    (r"^(Shooto)\b", "Shooto"),
    (r"^(Road FC|ROAD FC)\b", "Road FC"),
    (r"^(Fury FC|Fury Fighting)\b", "Fury FC"),
    (r"^(Strikeforce|SF)\b", "Strikeforce"),
    (r"^WEC\b|^World Extreme Cagefighting", "WEC"),
    (r"^(Pride|PRIDE)\b", "Pride"),
    (r"^(DREAM|Dream)\b", "DREAM"),
    (r"^(Oktagon|OKTAGON)\b", "Oktagon"),
    (r"^(Hexagone)\b", "Hexagone"),
    (r"^(ARES|Ares FC)\b", "Ares FC"),
    (r"^(Titan FC|Titan Fighting)\b", "Titan FC"),
]


def derive_promotion(event_name: str) -> str | None:
    """Best-effort promotion from a Sherdog event title.

    Sherdog titles look like '<Org> <number or subtitle> - <card name>'. We map known
    orgs via regex; otherwise return the prefix with trailing event numbers stripped
    ('MMA Versus UK 1' -> 'MMA Versus UK'). Authoritative org would need the event page
    (one extra request per event), which we avoid.
    """
    if not event_name:
        return None
    prefix, _, suffix = (p.strip() for p in event_name.partition(" - "))
    for pat, org in _PROMOTION_PATTERNS:
        if re.search(pat, prefix):
            return org
    # Regional titles are often 'ABBR n - Full Org Name n' ('BFC - Bellator Fighting
    # Championships 45', 'EFC - Eagle Fighting Championship'): when the prefix is a bare
    # abbreviation, the suffix names the org.
    if suffix and re.fullmatch(r"[A-Z0-9&%]{2,7}(\s+\d+)?", prefix):
        for pat, org in _PROMOTION_PATTERNS:
            if re.search(pat, suffix):
                return org
    stripped = re.sub(r"(\s+(#?\d+[A-Za-z]?|[IVXLC]+|Fight Night.*|FN\s*\d+))+\s*$", "", prefix).strip()
    return stripped or prefix


def classify_method(method: str) -> str:
    m = (method or "").lower()
    if m.startswith("no contest") or m == "nc":
        return "NC"
    if m.startswith("draw") or "draw" in m.split("(")[0]:
        return "DRAW"
    if "dq" in m or "disqualification" in m:
        return "DQ"
    if m.startswith("ko") or m.startswith("tko"):
        return "KO/TKO"
    if m.startswith("submission") or m.startswith("technical submission"):
        return "SUB"
    if m.startswith("decision") or m.startswith("technical decision"):
        return "DEC"
    return "OTHER"


_RESULT_MAP = {"win": "W", "loss": "L", "draw": "D", "nc": "NC", "no contest": "NC"}


@dataclasses.dataclass
class Bout:
    fighter_sherdog_id: int
    bout_index: int  # 0 = newest row on the page
    section: str  # "pro" (amateur rows are parsed but excluded from bouts.csv by default)
    result: str  # W / L / D / NC / raw text if unknown
    opponent_name: str
    opponent_sherdog_id: int | None
    opponent_url: str | None
    event_name: str
    event_sherdog_id: int | None
    promotion: str | None
    date: str | None
    method: str
    method_detail: str | None
    method_class: str
    referee: str | None
    round: int | None
    time: str | None


@dataclasses.dataclass
class FighterProfile:
    sherdog_id: int
    url: str
    name: str
    nickname: str | None
    birth_date: str | None
    nationality: str | None
    locality: str | None
    height: str | None
    weight: str | None
    association: str | None
    weight_class: str | None
    record: dict  # {"wins": 15, "losses": 3, "draws": 0, "nc": 1} (as displayed)
    n_pro_bouts: int
    parsed_at: str


def _section_tables(soup) -> list[tuple[str, object]]:
    """Return [(section_title_lower, table)] for each fight-history table."""
    out = []
    for t in soup.select("table.new_table.fighter"):
        title_el = t.find_previous(class_="slanted_title")
        out.append((_text(title_el).lower(), t))
    return out


def parse_fighter_page(html: str, url: str, include_amateur: bool = False) -> tuple[FighterProfile | None, list[Bout]]:
    if not html:
        return None, []
    soup = BeautifulSoup(html, "html.parser")
    sid = sherdog_id_from_url(url)
    if sid is None:
        canon = soup.find("link", rel="canonical")
        sid = sherdog_id_from_url(canon.get("href") if canon else None)
    if sid is None:
        raise ValueError(f"cannot determine sherdog id for {url}")

    info = soup.find(class_="fighter-info") or soup
    name = _text(info.select_one("h1 .fn")) or _text(info.find("h1"))
    nick_el = info.select_one(".fighter-line2 .nickname")
    nickname = _text(nick_el).strip('"“” ') or None if nick_el else None

    bd_el = info.find(itemprop="birthDate")
    birth_date = parse_birth_date(_text(bd_el)) if bd_el else None
    nat_el = info.find(itemprop="nationality")
    loc_el = info.find(itemprop="addressLocality")
    h_el = info.find(itemprop="height")
    w_el = info.find(itemprop="weight")
    assoc_el = info.select_one("a.association [itemprop=name]") or info.select_one("a.association")
    wc_el = info.select_one('.association-class a[href*="weightclass="]')

    record: dict[str, int] = {}
    for box in info.select(".winloses"):
        spans = box.find_all("span")
        if len(spans) >= 2:
            key = _text(spans[0]).lower().replace("/", "").replace(" ", "")
            key = {"wins": "wins", "losses": "losses", "draws": "draws", "nc": "nc"}.get(key, key)
            try:
                record[key] = int(_text(spans[1]))
            except ValueError:
                pass

    bouts: list[Bout] = []
    for title, table in _section_tables(soup):
        if "pro" in title and "amateur" not in title:
            section = "pro"
        elif "amateur" in title:
            section = "amateur"
            if not include_amateur:
                continue
        else:
            # e.g. "upcoming"/exhibition tables -- skip, results only
            continue
        idx = 0
        for tr in table.find_all("tr"):
            if "table_head" in (tr.get("class") or []):
                continue
            tds = tr.find_all("td")
            if len(tds) < 6:
                continue
            raw_result = _text(tds[0]).lower()
            result = _RESULT_MAP.get(raw_result, raw_result.upper() or "?")
            opp_a = tds[1].find("a")
            opp_href = opp_a.get("href") if opp_a else None
            ev_a = tds[2].find("a")
            ev_name = _text(ev_a) if ev_a else _text(tds[2].find(itemprop="award")) or ""
            date_el = tds[2].find(class_="sub_line")
            method_b = tds[3].find("b")
            method_full = _text(method_b) if method_b else ""
            mm = re.match(r"^(.*?)\s*\((.*)\)\s*$", method_full)
            method, detail = (mm.group(1), mm.group(2)) if mm else (method_full, None)
            ref_el = tds[3].find(class_="sub_line")
            referee = _text(ref_el) or None
            rnd = _text(tds[4])
            bouts.append(
                Bout(
                    fighter_sherdog_id=sid,
                    bout_index=idx,
                    section=section,
                    result=result,
                    opponent_name=_text(tds[1]),
                    opponent_sherdog_id=sherdog_id_from_url(opp_href),
                    opponent_url=urljoin(BASE_URL, opp_href) if opp_href else None,
                    event_name=ev_name,
                    event_sherdog_id=event_id_from_url(ev_a.get("href") if ev_a else None),
                    promotion=derive_promotion(ev_name),
                    date=parse_bout_date(_text(date_el)),
                    method=method,
                    method_detail=detail,
                    method_class=classify_method(method_full),
                    referee=referee,
                    round=int(rnd) if rnd.isdigit() else None,
                    time=_text(tds[5]) or None,
                )
            )
            idx += 1

    profile = FighterProfile(
        sherdog_id=sid,
        url=urljoin(BASE_URL, urlparse(url).path),
        name=name,
        nickname=nickname,
        birth_date=birth_date,
        nationality=_text(nat_el) or None,
        locality=_text(loc_el) or None,
        height=_text(h_el) or None,
        weight=_text(w_el) or None,
        association=_text(assoc_el) or None,
        weight_class=_text(wc_el) or None,
        record=record,
        n_pro_bouts=sum(1 for b in bouts if b.section == "pro"),
        parsed_at=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
    )
    return profile, bouts


@dataclasses.dataclass
class SearchCandidate:
    sherdog_id: int
    url: str
    name: str
    nickname: str | None
    height: str | None
    weight: str | None
    association: str | None


def parse_search_results(html: str) -> list[SearchCandidate]:
    soup = BeautifulSoup(html or "", "html.parser")
    table = soup.select_one("table.fightfinder_result")
    if table is None:
        return []
    out: list[SearchCandidate] = []
    for tr in table.find_all("tr"):
        if "table_head" in (tr.get("class") or []):
            continue
        tds = tr.find_all("td")
        a = tr.find("a", href=FIGHTER_ID_RE)
        if not a or len(tds) < 6:
            continue
        out.append(
            SearchCandidate(
                sherdog_id=sherdog_id_from_url(a["href"]),
                url=urljoin(BASE_URL, a["href"]),
                name=_text(a),
                nickname=_text(tds[2]).strip('"“” ') or None,
                height=_text(tds[3].find("strong")) or None,
                weight=_text(tds[4].find("strong")) or None,
                association=_text(tds[5]) or None,
            )
        )
    return out


# --------------------------------------------------------------------------- name matching

_TRANSLIT = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "Æ": "AE",
                           "đ": "d", "Đ": "D", "ı": "i", "œ": "oe", "þ": "th", "ð": "d"})


def fold(s: str | None) -> str:
    """Accent/case/punctuation-insensitive form: 'José Aldo Jr.' -> 'jose aldo jr'."""
    s = (s or "").translate(_TRANSLIT)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("'", "").replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "junior"}


def name_tokens(s: str | None) -> set[str]:
    return {t for t in fold(s).split() if t not in _SUFFIXES}


def name_score(first: str, last: str, nickname: str | None, cand_name: str, cand_nick: str | None) -> float:
    """1.0 = same token set (any order). Partial credit for subset / nickname-as-surname."""
    ours = name_tokens(f"{first} {last}")
    theirs = name_tokens(cand_name)
    if not ours or not theirs:
        return 0.0
    if ours == theirs:
        return 1.0
    inter = ours & theirs
    score = len(inter) / max(len(ours), len(theirs))
    # One side has extra middle/second surname (common for Brazilian/Hispanic names).
    if ours <= theirs or theirs <= ours:
        score = max(score, 0.85)
    # ufcstats sometimes stores ring name as surname ("Patricio Pitbull").
    nick_tokens = name_tokens(cand_nick) | name_tokens(nickname)
    if nick_tokens and (ours - theirs) and (ours - theirs) <= nick_tokens and inter:
        score = max(score, 0.8)
    # Surname match only, first name different spelling (e.g. 'Alex'/'Alexander').
    last_t = name_tokens(last)
    first_t = name_tokens(first)
    if last_t and last_t <= theirs and any(
        any(a.startswith(b[:4]) or b.startswith(a[:4]) for b in theirs) for a in first_t
    ):
        score = max(score, 0.75)
    return round(score, 3)


# --------------------------------------------------------------------------- resolver


@dataclasses.dataclass
class OurFighter:
    id: int
    ufcstats_id: str
    first_name: str
    last_name: str
    nickname: str | None
    dob: str | None
    ufc_fight_dates: list[str]  # ISO dates of completed/scheduled UFC bouts in ufc_fights

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


@dataclasses.dataclass
class Resolution:
    ufc_fighter_id: int
    ufcstats_id: str
    name: str
    dob: str | None
    sherdog_id: int | None
    sherdog_url: str | None
    sherdog_name: str | None
    sherdog_dob: str | None
    status: str  # matched_dob | matched_ufc_dates | matched_name_only | ambiguous | not_found | dob_mismatch | error
    name_score: float | None
    dob_match: bool | None
    ufc_date_overlap: int | None
    n_candidates: int
    queries: str
    note: str | None = None


MAX_CANDIDATE_PAGES = 3


def _search_queries(f: OurFighter) -> list[str]:
    qs = [f.full_name]
    # Surname-only fallback catches Sherdog spelling variants of the first name and
    # ufcstats ring-names. Skipped for very short / very common single tokens.
    if len(fold(f.last_name)) >= 4:
        qs.append(f.last_name)
    if f.nickname and len(f.first_name) >= 3:
        qs.append(f"{f.first_name} {f.nickname}")
    seen, out = set(), []
    for q in qs:
        k = fold(q)
        if k and k not in seen:
            seen.add(k)
            out.append(q)
    return out


def _ufc_overlap(bouts: list[Bout], ufc_dates: list[str]) -> int:
    ufc_days = {dt.date.fromisoformat(d) for d in ufc_dates if d}
    n = 0
    for b in bouts:
        if b.date and b.promotion == "UFC":
            d = dt.date.fromisoformat(b.date)
            if any(abs((d - u).days) <= 1 for u in ufc_days):
                n += 1
    return n


def _nick_bonus(f: OurFighter, c: SearchCandidate) -> float:
    """Tie-break among equal name scores: nickname agreement (ours may be a ring-name surname)."""
    cn = name_tokens(c.nickname)
    if not cn:
        return 0.0
    if cn & (name_tokens(f.nickname) | name_tokens(f.last_name)):
        return 0.05
    return 0.0


def resolve_fighter(f: OurFighter, fetcher: PoliteFetcher) -> tuple[Resolution, list[tuple[FighterProfile, list[Bout]]]]:
    """Search Sherdog, score candidates on name, verify lazily with DOB and UFC bout dates.

    Candidates are verified best-name-first and the search stops at the first strong
    match (DOB equal, or a Sherdog UFC bout on one of our ufc_fights dates), so the
    typical cost is 1 search + 1 fighter page. Fallback queries (surname only, first
    name + nickname) run only when nothing strong was found. Every page parsed along the
    way is returned (HTML is cached, so --fetch never re-downloads the winner).
    """
    queries_used: list[str] = []
    cands: dict[int, tuple[SearchCandidate, float]] = {}
    verified: dict[int, tuple] = {}
    parsed: list[tuple[FighterProfile, list[Bout]]] = []
    strong = None
    for q in _search_queries(f):
        queries_used.append(q)
        for c in parse_search_results(fetcher.get(SEARCH_PATH + quote_plus(q))):
            s = name_score(f.first_name, f.last_name, f.nickname, c.name, c.nickname)
            if s >= 0.75 and (c.sherdog_id not in cands or cands[c.sherdog_id][1] < s):
                cands[c.sherdog_id] = (c, s)
        ranked = sorted(cands.values(), key=lambda cs: -(cs[1] + _nick_bonus(f, cs[0])))
        for c, s in ranked:
            if strong or len(verified) >= MAX_CANDIDATE_PAGES:
                break
            if c.sherdog_id in verified:
                continue
            prof, bouts = parse_fighter_page(fetcher.get(c.url), c.url)
            if prof is None:
                verified[c.sherdog_id] = None
                continue
            parsed.append((prof, bouts))
            dob_match = (prof.birth_date == f.dob) if (prof.birth_date and f.dob) else None
            overlap = _ufc_overlap(bouts, f.ufc_fight_dates)
            evidence = (2 if dob_match else 0) + (1.5 if overlap >= 1 else 0) - (3 if dob_match is False else 0)
            verified[c.sherdog_id] = (evidence + s + _nick_bonus(f, c), s, dob_match, overlap, c, prof)
            if dob_match or (overlap >= 1 and dob_match is not False):
                strong = verified[c.sherdog_id]
        if strong:
            break

    base = dict(ufc_fighter_id=f.id, ufcstats_id=f.ufcstats_id, name=f.full_name, dob=f.dob,
                queries=" | ".join(queries_used), n_candidates=len(cands))
    scored = sorted((v for v in verified.values() if v), key=lambda x: -x[0])
    if not scored:
        return Resolution(**base, sherdog_id=None, sherdog_url=None, sherdog_name=None, sherdog_dob=None,
                          status="not_found", name_score=None, dob_match=None, ufc_date_overlap=None), parsed
    total, s, dob_match, overlap, c, prof = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else None
    if dob_match:
        status = "matched_dob"
    elif overlap >= 1 and dob_match is not False:
        status = "matched_ufc_dates"
    elif dob_match is False:
        status = "dob_mismatch"
    elif s == 1.0 and len(cands) == 1:
        # Debutant: no DOB on one side and no UFC bout on Sherdog yet; unique exact name.
        status = "matched_name_only"
    else:
        status = "ambiguous"
    unverified = len(cands) - len(verified)
    note = f"{unverified} name candidates not inspected (cap {MAX_CANDIDATE_PAGES})" if (unverified and not strong) else None
    if status == "ambiguous" and runner_up is not None:
        note = (note + "; " if note else "") + f"runner-up score {runner_up:.2f} vs {total:.2f}"
    return Resolution(**base, sherdog_id=prof.sherdog_id, sherdog_url=prof.url, sherdog_name=prof.name,
                      sherdog_dob=prof.birth_date, status=status, name_score=s, dob_match=dob_match,
                      ufc_date_overlap=overlap, note=note), parsed


# --------------------------------------------------------------------------- our fighters (read-only)


def load_our_fighters(db_url: str, allow_remote: bool = False, names: list[str] | None = None,
                      limit: int | None = None) -> list[OurFighter]:
    """Read ufc.ufc_fighters + fight dates from the LOCAL db, in a read-only transaction.

    Ordered by most recent UFC bout (upcoming debutants first), so --resolve N covers
    the fighters that matter for the next cards.
    """
    host = urlparse(db_url).hostname or "localhost"
    if host not in ("localhost", "127.0.0.1", "::1") and not allow_remote:
        raise SystemExit(f"refusing non-local database host {host!r} (production safety); "
                         "pass --allow-remote-db only if you are sure")
    from sqlalchemy import create_engine, text

    kwargs = {}
    if db_url.startswith("postgresql"):
        kwargs["connect_args"] = {"options": "-c default_transaction_read_only=on"}
    engine = create_engine(db_url, **kwargs)
    sql = """
        SELECT f.id, f.ufcstats_id, f.first_name, f.last_name, f.nickname, f.dob,
               array_remove(array_agg(DISTINCT x.date), NULL) AS dates,
               max(x.date) AS last_date
        FROM ufc.ufc_fighters f
        LEFT JOIN ufc.ufc_fights x ON f.id IN (x.red_fighter_id, x.blue_fighter_id)
        GROUP BY f.id
        ORDER BY max(x.date) DESC NULLS LAST, f.id
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    engine.dispose()
    out = []
    wanted = {fold(n) for n in names} if names else None
    for r in rows:
        name = f"{r['first_name']} {r['last_name']}"
        if wanted is not None and fold(name) not in wanted:
            continue
        out.append(OurFighter(id=r["id"], ufcstats_id=r["ufcstats_id"], first_name=r["first_name"] or "",
                              last_name=r["last_name"] or "", nickname=r["nickname"],
                              dob=r["dob"].isoformat() if r["dob"] else None,
                              ufc_fight_dates=[d.isoformat() for d in (r["dates"] or [])]))
    return out[:limit] if limit else out


# --------------------------------------------------------------------------- storage


RESOLVED_CSV = "resolved.csv"
FIGHTERS_JSONL = "fighters.jsonl"
BOUTS_CSV = "bouts.csv"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    tmp.replace(path)


class Store:
    """Idempotent upsert of scraper output into data/sherdog/ files."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.dir = data_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def save_resolutions(self, res: list[Resolution]) -> None:
        path = self.dir / RESOLVED_CSV
        fields = [f.name for f in dataclasses.fields(Resolution)]
        rows = {r["ufc_fighter_id"]: r for r in _read_csv(path)}
        for r in res:
            rows[str(r.ufc_fighter_id)] = dataclasses.asdict(r)
        _write_csv(path, list(rows.values()), fields)

    def resolutions(self) -> list[dict]:
        return _read_csv(self.dir / RESOLVED_CSV)

    def profiles(self) -> dict[int, dict]:
        path = self.dir / FIGHTERS_JSONL
        out: dict[int, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    out[int(d["sherdog_id"])] = d
        return out

    def save_fighters(self, parsed: list[tuple[FighterProfile, list[Bout]]], tier: dict[int, str]) -> None:
        profs = self.profiles()
        for prof, _ in parsed:
            d = dataclasses.asdict(prof)
            d["crawl_tier"] = tier.get(prof.sherdog_id) or profs.get(prof.sherdog_id, {}).get("crawl_tier")
            profs[prof.sherdog_id] = d
        with open(self.dir / FIGHTERS_JSONL, "w", encoding="utf-8") as fh:
            for d in profs.values():
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")

        path = self.dir / BOUTS_CSV
        fields = [f.name for f in dataclasses.fields(Bout)]
        replaced = {prof.sherdog_id for prof, _ in parsed}
        rows = [r for r in _read_csv(path) if int(r["fighter_sherdog_id"]) not in replaced]
        for _, bouts in parsed:
            rows.extend(dataclasses.asdict(b) for b in bouts if b.section == "pro")
        _write_csv(path, rows, fields)

    def bouts(self) -> list[dict]:
        return _read_csv(self.dir / BOUTS_CSV)


# --------------------------------------------------------------------------- crawl planner


def plan_opponent_frontier(store: Store) -> list[tuple[int, int, str, str]]:
    """Opponents (one hop from resolved UFC fighters) whose pages we do not have yet.

    Priority: opponents referenced by more of our fighters first, then most recent bout,
    so a small --max-pages budget buys the most strength-of-schedule signal. Opponents
    who are themselves UFC fighters resolve through --fetch, not here.
    """
    have = set(store.profiles())
    ufc_ids = {int(r["sherdog_id"]) for r in store.resolutions() if r.get("sherdog_id")}
    refs: Counter[int] = Counter()
    last_seen: dict[int, str] = {}
    names: dict[int, str] = {}
    urls: dict[int, str] = {}
    for b in store.bouts():
        if int(b["fighter_sherdog_id"]) not in ufc_ids or not b["opponent_sherdog_id"]:
            continue
        oid = int(b["opponent_sherdog_id"])
        if oid in have or oid in ufc_ids:
            continue
        refs[oid] += 1
        last_seen[oid] = max(last_seen.get(oid, ""), b["date"] or "")
        names[oid] = b["opponent_name"]
        urls[oid] = b["opponent_url"]
    items = [(oid, n, names[oid], urls[oid]) for oid, n in refs.items()]
    items.sort(key=lambda t: last_seen[t[0]], reverse=True)  # most recent first...
    items.sort(key=lambda t: -t[1])  # ...then (stable) most-referenced first
    return items


def crawl_estimate(store: Store, total_ufc_fighters: int, delay_s: float = MIN_DELAY_S) -> dict:
    res = [r for r in store.resolutions() if r.get("sherdog_id")]
    ufc_ids = {int(r["sherdog_id"]) for r in res}
    bouts = [b for b in store.bouts() if int(b["fighter_sherdog_id"]) in ufc_ids]
    per_fighter = Counter(int(b["fighter_sherdog_id"]) for b in bouts)
    opps = {int(b["opponent_sherdog_id"]) for b in bouts if b["opponent_sherdog_id"]}
    non_ufc_opps = opps - ufc_ids
    n = max(len(per_fighter), 1)
    uniq_per_fighter = len(non_ufc_opps) / n
    # Per-fighter request cost with lazy verification: ~1.1 searches + ~1.2 candidate pages
    # (observed on the development sample; ring-name/common-name fighters cost more).
    per_fighter_requests = 2.3
    resolve_req = round(total_ufc_fighters * per_fighter_requests)
    # Naive one-hop upper bound ignores that regional opponents are shared between UFC
    # fighters; use ~0.65x of non-UFC bout appearances as the realistic unique count.
    naive_opps = round(uniq_per_fighter * total_ufc_fighters)
    return {
        "sample_ufc_fighters_fetched": len(per_fighter),
        "avg_pro_bouts_per_fighter": round(len(bouts) / n, 1),
        "non_ufc_unique_opponents_in_sample": len(non_ufc_opps),
        "non_ufc_unique_opponents_per_fighter": round(uniq_per_fighter, 2),
        "total_ufc_fighters": total_ufc_fighters,
        "est_requests_resolve_and_fetch": resolve_req,
        "est_hours_resolve_and_fetch": round(resolve_req * delay_s / 3600, 1),
        "naive_one_hop_opponents_upper_bound": naive_opps,
        "est_hours_per_10k_opponent_pages": round(10000 * delay_s / 3600, 1),
        "delay_s": delay_s,
        "note": "sample is veteran-heavy; see report for DB-based estimate using ufc_fighters W+L+D",
    }


# --------------------------------------------------------------------------- orchestration


def _dump_parsed(parsed, tier, store: Store) -> None:
    if parsed:
        store.save_fighters(parsed, tier)


def cmd_resolve(args, fetcher: PoliteFetcher, store: Store) -> None:
    ours = load_our_fighters(args.db_url, args.allow_remote_db, args.names, None)
    done = {r["ufc_fighter_id"] for r in store.resolutions() if r["status"] != "error"} if not args.names else set()
    todo = [f for f in ours if str(f.id) not in done][: args.resolve]
    results, parsed_all, tier = [], [], {}
    try:
        for f in todo:
            try:
                res, parsed = resolve_fighter(f, fetcher)
            except (BudgetExhausted, SherdogBlocked, RobotsDisallowed):
                raise
            except Exception as e:  # parse bug on one fighter shouldn't kill the batch
                log.exception("resolve failed for %s", f.full_name)
                res, parsed = Resolution(f.id, f.ufcstats_id, f.full_name, f.dob, None, None, None, None,
                                         "error", None, None, None, 0, "", str(e)[:200]), []
            results.append(res)
            # Keep only the chosen profile; losing candidates stay in the HTML cache.
            for prof, bouts in parsed:
                if prof.sherdog_id == res.sherdog_id:
                    parsed_all.append((prof, bouts))
                    tier[prof.sherdog_id] = "ufc"
            log.info("%-28s -> %-18s %s", f.full_name, res.status, res.sherdog_url or "")
    finally:
        store.save_resolutions(results)
        _dump_parsed(parsed_all, tier, store)
    _print_resolutions(results)


def cmd_fetch(args, fetcher: PoliteFetcher, store: Store) -> None:
    have = set(store.profiles())
    ok = {"matched_dob", "matched_ufc_dates", "matched_name_only"}
    todo = [r for r in store.resolutions() if r.get("sherdog_id") and r["status"] in ok
            and int(r["sherdog_id"]) not in have][: args.fetch]
    parsed, tier = [], {}
    try:
        for r in todo:
            html = fetcher.get(r["sherdog_url"])
            prof, bouts = parse_fighter_page(html, r["sherdog_url"])
            if prof:
                parsed.append((prof, bouts))
                tier[prof.sherdog_id] = "ufc"
    finally:
        _dump_parsed(parsed, tier, store)
    print(f"fetched {len(parsed)} UFC fighter pages")


def cmd_expand(args, fetcher: PoliteFetcher, store: Store) -> None:
    frontier = plan_opponent_frontier(store)
    print(f"opponent frontier: {len(frontier)} un-fetched one-hop opponents; budget {args.max_pages}")
    parsed, tier = [], {}
    try:
        for oid, refs, name, url in frontier[: args.max_pages]:
            try:
                html = fetcher.get(url)
            except (BudgetExhausted, SherdogBlocked, RobotsDisallowed):
                raise
            except Exception as e:  # one broken page must not end the crawl
                log.warning("skipping %s: %s", url, e)
                continue
            prof, bouts = parse_fighter_page(html, url)
            if prof:
                parsed.append((prof, bouts))
                tier[oid] = "opponent_1hop"
                print(f"  {prof.name:<28} id={oid} refs={refs} pro_bouts={prof.n_pro_bouts}")
    finally:
        _dump_parsed(parsed, tier, store)


def cmd_reparse(args, store: Store) -> None:
    """Rebuild fighters.jsonl/bouts.csv from cached HTML only (after parser changes)."""
    profs = store.profiles()
    parsed = []
    for sid, d in profs.items():
        p = CACHE_DIR / f"fighter_{sid}.html"
        if p.exists():
            parsed.append(parse_fighter_page(p.read_text(encoding="utf-8"), d["url"]))
    tier = {sid: d.get("crawl_tier") for sid, d in profs.items()}
    store.save_fighters([x for x in parsed if x[0]], tier)
    print(f"reparsed {len(parsed)} cached fighter pages")


def cmd_plan(args, store: Store) -> None:
    total = args.total_ufc_fighters
    est = crawl_estimate(store, total)
    frontier = plan_opponent_frontier(store)
    est["current_opponent_frontier"] = len(frontier)
    path = DATA_DIR / "crawl_plan.json"
    path.write_text(json.dumps(est, indent=2), encoding="utf-8")
    print(json.dumps(est, indent=2))


def _print_resolutions(results: list[Resolution]) -> None:
    for r in results:
        print(f"{r.name:<26} {r.status:<18} id={r.sherdog_id} name={r.sherdog_name!r} "
              f"dob={r.dob}/{r.sherdog_dob} ufc_overlap={r.ufc_date_overlap} cands={r.n_candidates}"
              + (f" note={r.note}" if r.note else ""))
    if results:
        c = Counter(r.status for r in results)
        print("summary:", dict(c))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.services.ufc.sherdog_scraper", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolve", type=int, metavar="N", help="resolve Sherdog ids for N unresolved UFC fighters")
    ap.add_argument("--names", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                    help='restrict --resolve to these "First Last" names')
    ap.add_argument("--fetch", type=int, metavar="N", help="fetch/parse N resolved fighters' pages")
    ap.add_argument("--expand-opponents", action="store_true", help="fetch one-hop opponents' pages")
    ap.add_argument("--max-pages", type=int, default=10, help="page budget for --expand-opponents")
    ap.add_argument("--plan", action="store_true", help="print crawl-size estimate (no network)")
    ap.add_argument("--reparse", action="store_true", help="rebuild outputs from cached HTML (no network)")
    ap.add_argument("--total-ufc-fighters", type=int, default=4616)
    ap.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS,
                    help="hard cap on network requests this run (cache hits are free)")
    ap.add_argument("--delay", type=float, default=MIN_DELAY_S, help=f"seconds between requests (min {MIN_DELAY_S})")
    ap.add_argument("--offline", action="store_true", help="cache only; never touch the network")
    ap.add_argument("--db-url", default=DEFAULT_DB_URL)
    ap.add_argument("--allow-remote-db", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    fetcher = PoliteFetcher(max_requests=args.max_requests, delay_s=args.delay, offline=args.offline)
    store = Store()
    try:
        if args.resolve:
            cmd_resolve(args, fetcher, store)
        if args.fetch:
            cmd_fetch(args, fetcher, store)
        if args.expand_opponents:
            cmd_expand(args, fetcher, store)
        if args.reparse:
            cmd_reparse(args, store)
        if args.plan:
            cmd_plan(args, store)
    except SherdogBlocked as e:
        print(f"BLOCKED by Sherdog, stopping (not retrying): {e}", file=sys.stderr)
        return 3
    except RobotsDisallowed as e:
        print(f"robots.txt disallows {e}; stopping", file=sys.stderr)
        return 4
    except BudgetExhausted as e:
        print(f"stopped: {e}. Re-run to continue; cached pages are not re-fetched.", file=sys.stderr)
        return 2
    finally:
        print(f"network requests this run: {fetcher.requests_made}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
