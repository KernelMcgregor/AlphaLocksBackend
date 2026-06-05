"""
Historical UFC odds scraper using the-odds-api.com.

Fetches closing odds for UFC events and matches them to fights in our DB
by fighter name fuzzy matching.

Usage:
    python -m app.services.odds_scraper                # scrape all events in DB
    python -m app.services.odds_scraper --since 2022   # only events from 2022+
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta

import httpx
from sqlalchemy import func

from app.config import settings
from app.database import SessionLocal
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFightOdds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("odds_scraper")

API_BASE = "https://api.the-odds-api.com/v4"


def _normalize_name(name: str) -> str:
    """Normalize fighter name for matching."""
    return name.strip().lower().replace(".", "").replace("'", "").replace("-", " ")


def _names_match(a: str, b: str) -> bool:
    """Fuzzy match two fighter names."""
    a_norm, b_norm = _normalize_name(a), _normalize_name(b)
    if a_norm == b_norm:
        return True
    # Check last name match
    a_parts, b_parts = a_norm.split(), b_norm.split()
    if a_parts and b_parts and a_parts[-1] == b_parts[-1]:
        # Last names match — check first initial
        if a_parts[0][0] == b_parts[0][0]:
            return True
    return False


def _american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def fetch_historical_odds(event_date: datetime, api_key: str) -> list[dict]:
    """Fetch odds snapshot closest to an event's start time."""
    # Query 2 hours before event (close to closing line)
    query_time = (event_date - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = (
        f"{API_BASE}/historical/sports/mma_mixed_martial_arts/odds"
        f"?apiKey={api_key}&regions=us&markets=h2h&oddsFormat=american"
        f"&date={query_time}"
    )

    try:
        r = httpx.get(url, timeout=15)
        if r.status_code == 422:
            # Date out of range — try day-of at noon
            query_time = event_date.strftime("%Y-%m-%dT12:00:00Z")
            url = (
                f"{API_BASE}/historical/sports/mma_mixed_martial_arts/odds"
                f"?apiKey={api_key}&regions=us&markets=h2h&oddsFormat=american"
                f"&date={query_time}"
            )
            r = httpx.get(url, timeout=15)

        if r.status_code != 200:
            log.warning(f"  API returned {r.status_code} for {query_time}")
            return []

        data = r.json()
        remaining = r.headers.get("x-requests-remaining", "?")
        log.info(f"  API calls remaining: {remaining}")
        return data.get("data", [])

    except httpx.HTTPError as e:
        log.warning(f"  HTTP error: {e}")
        return []


def match_odds_to_fights(
    odds_events: list[dict], fights: list[tuple], fighters: dict[int, str]
) -> list[dict]:
    """
    Match odds API events to our DB fights by fighter name.
    Returns list of {fight_id, red_odds, blue_odds, red_prob, blue_prob, bookmaker}.
    """
    matched = []

    for odds_event in odds_events:
        home = odds_event.get("home_team", "")
        away = odds_event.get("away_team", "")

        for fight_id, red_fid, blue_fid in fights:
            red_name = fighters.get(red_fid, "")
            blue_name = fighters.get(blue_fid, "")

            # Try to match (home=red, away=blue) or (home=blue, away=red)
            if (_names_match(home, red_name) and _names_match(away, blue_name)):
                red_is_home = True
            elif (_names_match(home, blue_name) and _names_match(away, red_name)):
                red_is_home = False
            else:
                continue

            # Get consensus odds (average across bookmakers, or first available)
            best_bm = None
            for bm in odds_event.get("bookmakers", []):
                # Prefer DraftKings, FanDuel, or BetMGM
                if bm["key"] in ("draftkings", "fanduel", "betmgm"):
                    best_bm = bm
                    break
            if not best_bm and odds_event.get("bookmakers"):
                best_bm = odds_event["bookmakers"][0]

            if not best_bm:
                continue

            h2h = None
            for market in best_bm.get("markets", []):
                if market["key"] == "h2h":
                    h2h = market
                    break
            if not h2h:
                continue

            odds_map = {}
            for outcome in h2h.get("outcomes", []):
                odds_map[outcome["name"]] = outcome["price"]

            home_odds = odds_map.get(home)
            away_odds = odds_map.get(away)
            if home_odds is None or away_odds is None:
                continue

            if red_is_home:
                red_odds, blue_odds = home_odds, away_odds
            else:
                red_odds, blue_odds = away_odds, home_odds

            red_prob = _american_to_implied_prob(red_odds)
            blue_prob = _american_to_implied_prob(blue_odds)
            # Normalize to remove vig
            total = red_prob + blue_prob
            red_prob_clean = red_prob / total
            blue_prob_clean = blue_prob / total

            matched.append({
                "fight_id": fight_id,
                "red_odds": red_odds,
                "blue_odds": blue_odds,
                "red_implied_prob": round(red_prob_clean, 4),
                "blue_implied_prob": round(blue_prob_clean, 4),
                "bookmaker": best_bm["title"],
            })
            break  # Found match, move to next odds event

    return matched


def run_odds_scrape(since_year: int = 2022):
    """Fetch historical odds for all events and store on fights."""
    log.info(f"Starting odds scrape (events since {since_year})")
    db = SessionLocal()
    api_key = settings.ODDS_API_KEY

    if not api_key:
        log.error("ODDS_API_KEY not set in .env")
        return

    # Load all fighters for name lookup
    fighters = {f.id: f"{f.first_name} {f.last_name}" for f in db.query(UFCFighter).all()}

    # Get events with fights, ordered by date
    events = (
        db.query(UFCEvent)
        .filter(UFCEvent.date >= f"{since_year}-01-01")
        .order_by(UFCEvent.date)
        .all()
    )
    log.info(f"  {len(events)} events to process")

    total_matched = 0
    odds_by_fight = {}
    odds_cache = {}  # date_str -> odds_events (avoid duplicate API calls)

    for i, event in enumerate(events):
        if i % 10 == 0:
            log.info(f"  Processing event {i}/{len(events)}: {event.name} ({event.date})")

        fights = [
            (f.id, f.red_fighter_id, f.blue_fighter_id)
            for f in db.query(UFCFight).filter(UFCFight.event_id == event.id).all()
        ]
        if not fights:
            continue

        date_key = str(event.date)
        if date_key not in odds_cache:
            event_dt = datetime.combine(event.date, datetime.min.time())
            odds_cache[date_key] = fetch_historical_odds(event_dt, api_key)
            time.sleep(0.5)

        odds_events = odds_cache[date_key]
        if odds_events:
            matched = match_odds_to_fights(odds_events, fights, fighters)
            # Upsert matched odds to DB immediately (batch per event)
            for m in matched:
                odds_by_fight[m["fight_id"]] = m
                existing = (
                    db.query(UFCFightOdds)
                    .filter(UFCFightOdds.fight_id == m["fight_id"], UFCFightOdds.bookmaker == m["bookmaker"])
                    .first()
                )
                if existing:
                    existing.red_odds = m["red_odds"]
                    existing.blue_odds = m["blue_odds"]
                    existing.red_implied_prob = m["red_implied_prob"]
                    existing.blue_implied_prob = m["blue_implied_prob"]
                else:
                    db.add(UFCFightOdds(
                        fight_id=m["fight_id"],
                        bookmaker=m["bookmaker"],
                        red_odds=m["red_odds"],
                        blue_odds=m["blue_odds"],
                        red_implied_prob=m["red_implied_prob"],
                        blue_implied_prob=m["blue_implied_prob"],
                    ))
            if matched:
                db.commit()
            total_matched += len(matched)

    log.info(f"  Matched odds for {total_matched} fights")

    db.close()
    return odds_by_fight


def fetch_live_odds(api_key: str) -> list[dict]:
    """Fetch current odds for upcoming MMA fights from all US bookmakers."""
    url = (
        f"{API_BASE}/sports/mma_mixed_martial_arts/odds"
        f"?apiKey={api_key}&regions=us&markets=h2h&oddsFormat=american"
    )
    try:
        r = httpx.get(url, timeout=15)
        if r.status_code != 200:
            log.warning(f"Live odds API returned {r.status_code}")
            return []
        remaining = r.headers.get("x-requests-remaining", "?")
        log.info(f"  API calls remaining: {remaining}")
        return r.json()
    except httpx.HTTPError as e:
        log.warning(f"  HTTP error: {e}")
        return []


def run_live_odds_scrape():
    """Fetch live odds for all upcoming fights from every available bookmaker."""
    log.info("Starting live odds scrape for upcoming fights")
    db = SessionLocal()
    api_key = settings.ODDS_API_KEY

    if not api_key:
        log.error("ODDS_API_KEY not set in .env")
        return

    fighters = {f.id: f"{f.first_name} {f.last_name}" for f in db.query(UFCFighter).all()}

    # Get upcoming fights (no winner yet)
    upcoming_fights = (
        db.query(UFCFight)
        .filter(UFCFight.winner_id.is_(None))
        .all()
    )
    if not upcoming_fights:
        log.info("No upcoming fights in DB")
        db.close()
        return

    fights_tuples = [(f.id, f.red_fighter_id, f.blue_fighter_id) for f in upcoming_fights]
    log.info(f"  {len(fights_tuples)} upcoming fights in DB")

    odds_events = fetch_live_odds(api_key)
    if not odds_events:
        log.info("No odds data returned from API")
        db.close()
        return

    log.info(f"  {len(odds_events)} matchups returned from API")

    total_upserted = 0
    for odds_event in odds_events:
        home = odds_event.get("home_team", "")
        away = odds_event.get("away_team", "")

        for fight_id, red_fid, blue_fid in fights_tuples:
            red_name = fighters.get(red_fid, "")
            blue_name = fighters.get(blue_fid, "")

            if _names_match(home, red_name) and _names_match(away, blue_name):
                red_is_home = True
            elif _names_match(home, blue_name) and _names_match(away, red_name):
                red_is_home = False
            else:
                continue

            # Store odds from EVERY bookmaker
            for bm in odds_event.get("bookmakers", []):
                h2h = None
                for market in bm.get("markets", []):
                    if market["key"] == "h2h":
                        h2h = market
                        break
                if not h2h:
                    continue

                odds_map = {}
                for outcome in h2h.get("outcomes", []):
                    odds_map[outcome["name"]] = outcome["price"]

                home_odds = odds_map.get(home)
                away_odds = odds_map.get(away)
                if home_odds is None or away_odds is None:
                    continue

                if red_is_home:
                    red_odds, blue_odds = home_odds, away_odds
                else:
                    red_odds, blue_odds = away_odds, home_odds

                red_prob = _american_to_implied_prob(red_odds)
                blue_prob = _american_to_implied_prob(blue_odds)
                total = red_prob + blue_prob
                red_prob_clean = round(red_prob / total, 4)
                blue_prob_clean = round(blue_prob / total, 4)

                existing = (
                    db.query(UFCFightOdds)
                    .filter(UFCFightOdds.fight_id == fight_id, UFCFightOdds.bookmaker == bm["title"])
                    .first()
                )
                if existing:
                    existing.red_odds = red_odds
                    existing.blue_odds = blue_odds
                    existing.red_implied_prob = red_prob_clean
                    existing.blue_implied_prob = blue_prob_clean
                else:
                    db.add(UFCFightOdds(
                        fight_id=fight_id,
                        bookmaker=bm["title"],
                        red_odds=red_odds,
                        blue_odds=blue_odds,
                        red_implied_prob=red_prob_clean,
                        blue_implied_prob=blue_prob_clean,
                    ))
                total_upserted += 1

            break  # matched this odds_event, next

    db.commit()
    log.info(f"  Upserted {total_upserted} odds rows across all bookmakers")
    db.close()


from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UFC odds scraper")
    parser.add_argument("--since", type=int, default=2022, help="Start year for historical")
    parser.add_argument("--live", action="store_true", help="Fetch live odds for upcoming fights (all bookmakers)")
    args = parser.parse_args()

    if args.live:
        run_live_odds_scrape()
    else:
        run_odds_scrape(since_year=args.since)
