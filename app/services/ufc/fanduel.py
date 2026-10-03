"""FanDuel Sportsbook: every UFC market, read from the book's own public JSON API.

BestFightOdds carries FanDuel's main props, but not the rest of what the book lists: winner x
round, method x round, alternate round ranges, strike ladders, specials. This reads the same
JSON the FanDuel website loads in a browser, with no login:

  1. content-managed-page?page=SPORT&eventTypeId=26420387     every MMA event (one request)
  2. event-page?eventId=<id>[&tab=<slug>]                      a bout's markets, one tab per call

then hands the bouts to book_markets.record, which matches them to our fights and appends every
selection whose price or line moved to ufc_book_market_history. Selections that map onto a
market this project prices get a market_key (see canonical_key), which is how
app/services/ufc/picks_v2.py picks FanDuel's prices up.

Politeness: one process, sequential requests, REQUEST_DELAY_S between them, a hard request cap,
and no retries on a refusal. If FanDuel starts refusing, this stops and says so. Beyond the
ordinary browser User-Agent the API expects, it does nothing to disguise itself -- no rotating
identities, proxies or fingerprint tricks.

Region: the API is served per state (sbapi.<state>.sportsbook.fanduel.com). FANDUEL_STATE picks
it (default "va"); prices are the same across most states. FANDUEL_AK overrides the public app
key the website itself sends.

Usage:
    python -m app.services.ufc.book_markets --books fanduel [--dry-run]
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import time

import requests

from app.services.ufc.book_markets import corners, ladder_line, side_of

log = logging.getLogger("fanduel")

BOOK = "FanDuel"
MMA_EVENT_TYPE_ID = 26420387
STATE = os.environ.get("FANDUEL_STATE", "va")
APP_KEY = os.environ.get("FANDUEL_AK", "FhMFpcPWXMeyZxOx")
BASE = f"https://sbapi.{STATE}.sportsbook.fanduel.com/api"
REQUEST_DELAY_S = 1.0
TIMEOUT_S = 20
HEADERS = {"Accept": "application/json",
           "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"}


class Refused(RuntimeError):
    """FanDuel answered with a block or an error page; the run stops rather than retrying."""


class Client:
    def __init__(self, max_requests: int = 250, delay_s: float = REQUEST_DELAY_S):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.max_requests, self.delay_s, self.n = max_requests, delay_s, 0

    def get(self, path: str, **params) -> dict:
        if self.n >= self.max_requests:
            raise Refused(f"request cap of {self.max_requests} reached")
        if self.n:
            time.sleep(self.delay_s)
        self.n += 1
        r = self.s.get(f"{BASE}/{path}", params={**params, "_ak": APP_KEY,
                                                 "timezone": "America/New_York"}, timeout=TIMEOUT_S)
        if r.status_code in (401, 403, 429):
            raise Refused(f"FanDuel returned {r.status_code} for {path}")
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# parsing (pure; unit-tested on saved responses)
# ---------------------------------------------------------------------------

def ufc_events(listing: dict) -> list[dict]:
    """[{event_id, name, fighter_a, fighter_b, open_date}] for every UFC bout in the MMA listing."""
    att = listing.get("attachments") or {}
    comps = {str(k): v.get("name", "") for k, v in (att.get("competitions") or {}).items()}
    out = []
    for e in (att.get("events") or {}).values():
        if "UFC" not in comps.get(str(e.get("competitionId")), "").upper():
            continue
        parts = re.split(r"\s+v(?:s\.?)?\s+", e.get("name", ""), maxsplit=1)
        if len(parts) != 2:
            continue
        out.append({"event_id": str(e["eventId"]), "name": e["name"], "fighter_a": parts[0].strip(),
                    "fighter_b": parts[1].strip(),
                    "open_date": dt.datetime.fromisoformat(e["openDate"].replace("Z", "+00:00"))})
    return out


def tab_slugs(event_page: dict) -> list[str]:
    """The non-default tabs worth fetching. Same-game-parlay tabs only repeat other markets."""
    tabs = (event_page.get("layout") or {}).get("tabs") or {}
    slugs = []
    for t in tabs.values():
        title = t.get("title") or ""
        if t.get("isSameGameMulti") or title.lower() == "popular":
            continue
        slugs.append(re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"))
    return slugs


def _american(runner: dict) -> int | None:
    if runner.get("runnerStatus", "ACTIVE") != "ACTIVE":
        return None
    a = ((runner.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}).get("americanOdds")
    return int(a) if a is not None else None


def canonical_key(market_type: str, selection: str, line: float | None, side: str | None) -> str | None:
    """This project's market key for a selection the models price, else None.

    side is 'red' / 'blue' (already resolved against our corners) for fighter-specific
    selections. Keys match prop_serving / picks_v2: moneyline_<c>, <c>_<ko|sub|dec>,
    dec_yes|dec_no, ou_<l>_over|under, sr_<n>|sr_<n>_no, er_<n>.
    """
    # five-round bouts carry the same markets with a "_(5_ROUNDS)" suffix
    mt = re.sub(r"_\(5_ROUNDS\)$", "", market_type.upper())
    sel = selection.strip().lower()
    if mt == "MATCH_BETTING" and side:
        return f"moneyline_{side}"
    if mt == "METHOD_OF_VICTORY" and side:
        for word, m in (("ko", "ko"), ("submission", "sub"), ("points", "dec"), ("decision", "dec")):
            if re.search(rf"\bby {word}", sel):
                return f"{side}_{m}"
        return None
    if mt.startswith("WILL_THE_FIGHT_GO_THE_DISTANCE"):
        return {"yes": "dec_yes", "no": "dec_no"}.get(sel)
    if mt == "TOTAL_ROUNDS" and line is not None and sel in ("over", "under"):
        return f"ou_{line:g}_{sel}"
    m = re.fullmatch(r"FIGHT_TO_START_ROUND_(\d)", mt)
    if m:
        return {"yes": f"sr_{m.group(1)}", "no": f"sr_{m.group(1)}_no"}.get(sel)
    if mt == "WHAT_ROUND_WILL_FIGHT_END":
        r = re.fullmatch(r"round (\d)", sel)
        return f"er_{r.group(1)}" if r else None
    return None


def parse_markets(pages: list[dict], fighter_a: str, fighter_b: str, swapped: bool) -> list[dict]:
    """Every selection on a bout's pages, deduplicated by (market, selection).

    swapped: fighter_a is our *blue* corner (bfo.match_matchups semantics).
    """
    corner = corners(swapped)
    seen, rows = set(), []
    for page in pages:
        for m in ((page.get("attachments") or {}).get("markets") or {}).values():
            mtype, mname = m.get("marketType") or "", m.get("marketName") or ""
            open_ = m.get("marketStatus", "OPEN") == "OPEN"
            market_side = side_of(mname, fighter_a, fighter_b)
            for r in m.get("runners") or []:
                key = (str(m.get("marketId")), str(r.get("selectionId")))
                if key in seen:
                    continue
                seen.add(key)
                sel = r.get("runnerName") or ""
                s = side_of(sel, fighter_a, fighter_b) or market_side
                side = corner[s] if s else None
                hcap = r.get("handicap")
                line = ladder_line(sel)
                if line is None and hcap not in (None, 0, 0.0):
                    line = float(hcap)
                rows.append({
                    "external_market_id": key[0], "external_selection_id": key[1],
                    "market_type": mtype, "market_name": mname, "selection": sel,
                    "side": side, "line": line,
                    "market_key": canonical_key(mtype, sel, line, side),
                    "american": _american(r) if open_ else None,
                })
    return rows


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def fetch_bout(client: Client, event_id: str) -> list[dict]:
    first = client.get("event-page", eventId=event_id)
    if not first:
        return []
    pages = [first]
    for slug in tab_slugs(first):
        page = client.get("event-page", eventId=event_id, tab=slug)
        if page:
            pages.append(page)
    return pages


def fetch_bouts(max_requests: int = 250) -> tuple[list[dict], dict]:
    """Every listed UFC bout with its pages, for book_markets.record. Stops at the first refusal
    and returns what it has, so a block mid-run still records the bouts fetched before it."""
    client = Client(max_requests=max_requests)
    info = {"requests": 0, "refused": None}
    events = ufc_events(client.get("content-managed-page", page="SPORT", eventTypeId=MMA_EVENT_TYPE_ID))
    bouts = []
    for e in events:
        try:
            pages = fetch_bout(client, e["event_id"])
        except Refused as err:
            info["refused"] = str(err)
            break
        bouts.append({"external_event_id": e["event_id"], "name": e["name"],
                      "fighter_a": e["fighter_a"], "fighter_b": e["fighter_b"],
                      # openDate is UTC; a late card can tip into the next day (matching allows +-2)
                      "date": e["open_date"].date(),
                      "parse": lambda swapped, pages=pages, e=e: parse_markets(
                          pages, e["fighter_a"], e["fighter_b"], swapped)})
    info["requests"] = client.n
    return bouts, info
