"""Pinnacle: UFC moneylines, totals and fight props from the public guest API (two requests).

  1. leagues/1624/matchups           every UFC bout, plus its props as child "special" matchups
  2. leagues/1624/markets/straight   every price for all of them, with Pinnacle's stake limits

Pinnacle is the sharpest book we read: low margins and high limits mean its de-vigged price is
the closest thing to a true market probability, which is how app/services/ufc/picks_v2.py uses
it (q_market prefers Pinnacle where it prices the market). Its maximum stake per market is
stored too, as a read of how much it trusts that price.

Props are Yes/No specials ("Eric Nolan To Win Inside Distance"); only the Yes leg maps onto a
per-fighter market key, while fight-level props (decision, starts round N) map both legs.

Usage:
    python -m app.services.ufc.book_markets --books pinnacle [--dry-run]
"""
from __future__ import annotations

import datetime as dt
import re

import httpx

from app.services.ufc.book_markets import corners, side_of

BOOK = "Pinnacle"
UFC_LEAGUE_ID = 1624
BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
HEADERS = {"Accept": "application/json",
           "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"}

_PROP = [
    (re.compile(r"to win inside distance$", re.I), lambda side, sel: f"itd_{side}_{sel}" if side else None),
    (re.compile(r"to win by (?:tko/ko|ko/tko|ko|tko)$", re.I), lambda side, sel: f"{side}_ko" if side and sel == "yes" else None),
    (re.compile(r"to win by submission$", re.I), lambda side, sel: f"{side}_sub" if side and sel == "yes" else None),
    (re.compile(r"to win by decision$", re.I), lambda side, sel: f"{side}_dec" if side and sel == "yes" else None),
    (re.compile(r"^fight goes to decision$", re.I), lambda side, sel: {"yes": "dec_yes", "no": "dec_no"}.get(sel)),
]


def canonical_key(market: str, selection: str, line: float | None, side: str | None) -> str | None:
    """This project's key for a Pinnacle selection (see fanduel.canonical_key)."""
    m, sel = market.strip(), selection.strip().lower()
    if m == "Moneyline" and side:
        return f"moneyline_{side}"
    if m == "Total Rounds" and line is not None and sel in ("over", "under"):
        return f"ou_{line:g}_{sel}"
    r = re.fullmatch(r"fight starts round (\d)", m, re.I)
    if r:
        return {"yes": f"sr_{r.group(1)}", "no": f"sr_{r.group(1)}_no"}.get(sel)
    for pat, key in _PROP:
        if pat.search(m):
            return key(side, sel)
    return None


def bouts_from(matchups: list[dict], markets: list[dict]) -> list[dict]:
    """[{id, fighter_a (home), fighter_b (away), date, specials: [...], prices: {matchupId: [...]}}]"""
    prices: dict[int, list] = {}
    for p in markets or []:
        if p.get("period", 0) == 0 and not p.get("isAlternate"):
            prices.setdefault(p["matchupId"], []).append(p)
    fights, specials = {}, {}
    for m in matchups or []:
        if m.get("isLive"):
            continue
        if m.get("type") == "matchup":
            names = {p.get("alignment"): p.get("name") for p in m.get("participants") or []}
            if names.get("home") and names.get("away"):
                fights[m["id"]] = {"id": str(m["id"]), "fighter_a": names["home"], "fighter_b": names["away"],
                                   "date": dt.datetime.fromisoformat(m["startTime"].replace("Z", "+00:00")).date(),
                                   "matchup": m}
        elif m.get("type") == "special" and m.get("parentId"):
            specials.setdefault(m["parentId"], []).append(m)
    for fid, f in fights.items():
        f["specials"] = specials.get(fid, [])
        f["prices"] = {mid: prices.get(mid, []) for mid in [fid] + [s["id"] for s in f["specials"]]}
    return list(fights.values())


def _limit(p: dict) -> float | None:
    return next((l.get("amount") for l in p.get("limits") or [] if l.get("type") == "maxRiskStake"), None)


def parse_bout(f: dict, swapped: bool) -> list[dict]:
    a, b = f["fighter_a"], f["fighter_b"]
    corner = corners(swapped)
    rows = []

    def row(market_id, sel_id, mtype, mname, sel, side, line, american, stake):
        rows.append({"external_market_id": str(market_id), "external_selection_id": str(sel_id),
                     "market_type": mtype, "market_name": mname[:300], "selection": sel[:300],
                     "side": side, "line": line, "american": american, "max_stake": stake,
                     "market_key": canonical_key(mname, sel, line, side)})

    for p in f["prices"].get(int(f["id"]), []):
        open_ = p.get("status", "open") == "open"
        for pr in p.get("prices") or []:
            d = pr.get("designation")
            if p["type"] == "moneyline" and d in ("home", "away"):
                side = corner["a" if d == "home" else "b"]
                row(p["key"], f"{p['key']}:{d}", "moneyline", "Moneyline", a if d == "home" else b,
                    side, None, pr.get("price") if open_ else None, _limit(p))
            elif p["type"] == "total" and d in ("over", "under"):
                row(p["key"], f"{p['key']}:{d}", "total", "Total Rounds", d.capitalize(), None,
                    pr.get("points"), pr.get("price") if open_ else None, _limit(p))
    for s in f["specials"]:
        desc = (s.get("special") or {}).get("description") or ""
        s_side = side_of(desc, a, b)
        side = corner[s_side] if s_side else None
        names = {p["id"]: p.get("name") or "" for p in s.get("participants") or []}
        for p in f["prices"].get(s["id"], []):
            for pr in p.get("prices") or []:
                pid = pr.get("participantId")
                row(s["id"], pid, "special", desc, names.get(pid, ""), side, None, pr.get("price"), _limit(p))
    # moneyline / totals rows are keyed by Pinnacle's market key, which repeats across bouts;
    # scope them to the bout so change detection never compares two different fights
    for r in rows:
        if not r["external_market_id"].isdigit():
            r["external_market_id"] = f"{f['id']}:{r['external_market_id']}"
            r["external_selection_id"] = f"{f['id']}:{r['external_selection_id']}"
    return rows


def fetch_bouts() -> tuple[list[dict], dict]:
    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as c:
        mu = c.get(f"{BASE}/leagues/{UFC_LEAGUE_ID}/matchups"); mu.raise_for_status()
        mk = c.get(f"{BASE}/leagues/{UFC_LEAGUE_ID}/markets/straight"); mk.raise_for_status()
    bouts = [{"external_event_id": f["id"], "name": f"{f['fighter_a']} vs {f['fighter_b']}",
              "fighter_a": f["fighter_a"], "fighter_b": f["fighter_b"], "date": f["date"],
              "parse": lambda swapped, f=f: parse_bout(f, swapped)}
             for f in bouts_from(mu.json(), mk.json())]
    return bouts, {"requests": 2}
