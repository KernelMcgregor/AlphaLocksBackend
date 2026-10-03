"""Bovada: every UFC market, from the book's public coupon API (one request for the whole slate).

Bovada lists more fight props than any other US-facing book we read: method, round and method x
round, double chance, winner x total rounds, and per-fighter significant-strike and takedown
over/unders with alternate ladders. All of it goes to ufc_book_market_history via
book_markets.record; selections that map onto a market this project prices get a market_key
(canonical_key).

Separate from bovada_scraper.py, which keeps filling ufc_method_odds (one current row per fight)
for the admin page; this module keeps the full price history.

Usage:
    python -m app.services.ufc.book_markets --books bovada [--dry-run]
"""
from __future__ import annotations

import datetime as dt
import re

import httpx

from app.services.ufc.book_markets import corners, ladder_line, side_of

BOOK = "Bovada"
API = "https://www.bovada.lv/services/sports/event/v2/events/A/description/ufc-mma"
UFC_PATH = "/ufc-mma/ufc/"
SKIP_PATHS = ("/ufc-mma/ufc/potential-fights",)   # rumoured bouts, not booked cards
HEADERS = {"Accept": "application/json",
           "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"}


def _american(price: dict | None) -> int | None:
    a = (price or {}).get("american")
    if a is None:
        return None
    a = str(a).strip().upper()
    return 100 if a == "EVEN" else int(a)


def canonical_key(market: str, selection: str, line: float | None, side: str | None) -> str | None:
    """This project's key for a Bovada selection the models price (see fanduel.canonical_key)."""
    m, sel = market.strip().lower(), selection.strip().lower()
    if m == "fight winner" and side:
        return f"moneyline_{side}"
    if m == "method of victory" and side:
        # match the method after "by": a fighter's own name can contain "ko" (Kopylov)
        for pat, method in ((r"\bby (?:ko|tko)\b", "ko"), (r"\bby submission\b", "sub"), (r"\bby decision\b", "dec")):
            if re.search(pat, sel):
                return f"{side}_{method}"
        return None
    if m == "will the fight go the distance":
        return {"yes": "dec_yes", "no": "dec_no"}.get(sel)
    if m in ("total rounds over/under", "main total rounds over/under") and line is not None and sel in ("over", "under"):
        return f"ou_{line:g}_{sel}"
    r = re.fullmatch(r"fight to complete (\d) full rounds?", m)
    if r:   # completing round N == round N+1 starts
        n = int(r.group(1)) + 1
        return f"sr_{n}" if sel.startswith("yes") else f"sr_{n}_no" if sel.startswith("no") else None
    if m == "when will the fight end":
        e = re.fullmatch(r"round (\d)", sel)
        return f"er_{e.group(1)}" if e else None
    if m == "fight winner - inside the distance only" and side:
        return f"itd_{side}_yes"
    return None


def _line(sel: str, price: dict | None) -> float | None:
    rung = ladder_line(sel)
    if rung is not None:
        return rung
    h = (price or {}).get("handicap")
    if h not in (None, ""):
        return float(h)
    m = re.match(r"\s*(?:over|under)\s+(\d+(?:\.\d+)?)", sel, re.I)
    return float(m.group(1)) if m else None


def parse_event(e: dict, fighter_a: str, fighter_b: str, swapped: bool) -> list[dict]:
    corner = corners(swapped)
    rows = []
    for g in e.get("displayGroups") or []:
        for m in g.get("markets") or []:
            period = (m.get("period") or {})
            if not period.get("main", True):
                continue   # per-round sub-markets ("Round 1 winner") are not bout markets
            mname = m.get("description") or ""
            open_ = m.get("status") == "O"
            market_side = side_of(mname, fighter_a, fighter_b)
            for o in m.get("outcomes") or []:
                sel, price = o.get("description") or "", o.get("price") or {}
                s = side_of(sel, fighter_a, fighter_b) or market_side
                side = corner[s] if s else None
                line = _line(sel, price)
                rows.append({
                    "external_market_id": str(m.get("id")), "external_selection_id": str(o.get("id")),
                    "market_type": (m.get("key") or "")[:120], "market_name": mname[:300],
                    "selection": sel[:300], "side": side, "line": line,
                    "market_key": canonical_key(mname, sel, line, side),
                    "american": _american(price) if open_ and o.get("status") == "O" else None,
                })
    return rows


def ufc_bouts(listing: list) -> list[dict]:
    out = []
    for comp in listing or []:
        path = ((comp.get("path") or [{}])[0]).get("link", "")
        if not path.startswith(UFC_PATH) or path.startswith(SKIP_PATHS):
            continue
        for e in comp.get("events") or []:
            parts = re.split(r"\s+vs\.?\s+", e.get("description", ""), maxsplit=1)
            if len(parts) != 2 or e.get("live"):
                continue
            out.append({"event": e, "fighter_a": parts[0].strip(), "fighter_b": parts[1].strip(),
                        "date": dt.datetime.fromtimestamp(e["startTime"] / 1000, dt.timezone.utc).date()})
    return out


def fetch_bouts() -> tuple[list[dict], dict]:
    r = httpx.get(API, headers=HEADERS, timeout=30, follow_redirects=True)
    r.raise_for_status()
    bouts = [{"external_event_id": str(b["event"]["id"]), "name": b["event"]["description"],
              "fighter_a": b["fighter_a"], "fighter_b": b["fighter_b"], "date": b["date"],
              "parse": lambda swapped, b=b: parse_event(b["event"], b["fighter_a"], b["fighter_b"], swapped)}
             for b in ufc_bouts(r.json())]
    return bouts, {"requests": 1}
