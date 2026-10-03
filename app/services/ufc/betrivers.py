"""BetRivers: every UFC market, from Kambi's public offering API (the platform BetRivers runs on).

  1. listView/ufc_mma.json                 every listed MMA bout (one request)
  2. betoffer/event/<id>.json              all of a bout's bet offers (one request per bout)

BetRivers' main props already arrive through BestFightOdds; reading Kambi directly adds what
BFO does not carry (winner x over/under rounds, winning round, round groups, finish-only and
decision-only lines) and BetRivers' own price history. Rows go to ufc_book_market_history via
book_markets.record.

Kambi serves one "customer" per operator and state; KAMBI_CUSTOMER picks it (default
"rsiuspa", BetRivers Pennsylvania).

Usage:
    python -m app.services.ufc.book_markets --books betrivers [--dry-run]
"""
from __future__ import annotations

import datetime as dt
import os
import re
import time

import httpx

from app.services.ufc.book_markets import corners, ladder_line, side_of

BOOK = "BetRivers"
CUSTOMER = os.environ.get("KAMBI_CUSTOMER", "rsiuspa")
BASE = f"https://eu-offering-api.kambicdn.com/offering/v2018/{CUSTOMER}"
PARAMS = {"lang": "en_US", "market": "US"}
REQUEST_DELAY_S = 1.0
MAX_REQUESTS = 80
HEADERS = {"Accept": "application/json",
           "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"}


def canonical_key(market: str, selection: str, line: float | None, side: str | None) -> str | None:
    """This project's key for a Kambi selection the models price (see fanduel.canonical_key)."""
    m, sel = market.strip().lower(), selection.strip().lower()
    if m == "bout odds" and side:
        return f"moneyline_{side}"
    if m == "winning method" and side:
        # match the method after "by": a fighter's own name can contain "ko" (Kopylov)
        for pat, method in ((r"\bby (?:ko|tko)\b", "ko"), (r"\bby submission\b", "sub"), (r"\bby decision\b", "dec")):
            if re.search(pat, sel):
                return f"{side}_{method}"
        return None
    if m == "to go the distance":
        return {"yes": "dec_yes", "no": "dec_no"}.get(sel)
    if m == "total rounds" and line is not None and sel in ("over", "under"):
        return f"ou_{line:g}_{sel}"
    r = re.fullmatch(r"fight to start round (\d)", m)
    if r:
        return {"yes": f"sr_{r.group(1)}", "no": f"sr_{r.group(1)}_no"}.get(sel)
    if re.fullmatch(r".+ to win by finish", m) and side:
        return {"yes": f"itd_{side}_yes", "no": f"itd_{side}_no"}.get(sel)
    r = re.fullmatch(r"any fighter to win in round (\d)", m)
    if r and sel == "yes":   # a fighter wins in round N == the fight ends in round N
        return f"er_{r.group(1)}"
    return None


def _american(o: dict) -> int | None:
    if o.get("status") != "OPEN" or o.get("oddsAmerican") in (None, ""):
        return None
    return int(o["oddsAmerican"])


def parse_offers(offers: list[dict], fighter_a: str, fighter_b: str, swapped: bool) -> list[dict]:
    corner = corners(swapped)
    rows = []
    for b in offers or []:
        crit = b.get("criterion") or {}
        mname = crit.get("englishLabel") or crit.get("label") or ""
        market_side = side_of(mname, fighter_a, fighter_b)
        for o in b.get("outcomes") or []:
            # match-type outcomes are labelled "1" / "2"; the fighter is in `participant`
            sel = o.get("participant") or o.get("englishLabel") or o.get("label") or ""
            s = side_of(sel, fighter_a, fighter_b) or market_side
            side = corner[s] if s else None
            line = o["line"] / 1000 if o.get("line") is not None else ladder_line(sel)
            rows.append({
                "external_market_id": str(b.get("id")), "external_selection_id": str(o.get("id")),
                "market_type": ((b.get("betOfferType") or {}).get("englishName") or "")[:120],
                "market_name": mname[:300], "selection": sel[:300], "side": side, "line": line,
                "market_key": canonical_key(mname, sel, line, side), "american": _american(o),
            })
    return rows


def ufc_events(listing: dict) -> list[dict]:
    out = []
    for item in (listing or {}).get("events") or []:
        e = item.get("event") or {}
        terms = {p.get("termKey") for p in e.get("path") or []}
        if "ufc" not in terms or e.get("state") != "NOT_STARTED":
            continue
        a, b = e.get("homeName"), e.get("awayName")
        if not (a and b):
            continue
        out.append({"id": str(e["id"]), "name": e.get("englishName") or e.get("name"), "fighter_a": a,
                    "fighter_b": b,
                    "date": dt.datetime.fromisoformat(e["start"].replace("Z", "+00:00")).date()})
    return out


def fetch_bouts() -> tuple[list[dict], dict]:
    n = 0
    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as c:
        def get(path):
            nonlocal n
            if n:
                time.sleep(REQUEST_DELAY_S)
            n += 1
            r = c.get(f"{BASE}/{path}", params=PARAMS)
            r.raise_for_status()
            return r.json()
        events = ufc_events(get("listView/ufc_mma.json"))
        bouts = []
        for e in events[:MAX_REQUESTS - 1]:
            offers = get(f"betoffer/event/{e['id']}.json").get("betOffers") or []
            bouts.append({"external_event_id": e["id"], "name": e["name"], "fighter_a": e["fighter_a"],
                          "fighter_b": e["fighter_b"], "date": e["date"],
                          "parse": lambda swapped, offers=offers, e=e: parse_offers(
                              offers, e["fighter_a"], e["fighter_b"], swapped)})
    return bouts, {"requests": n}
