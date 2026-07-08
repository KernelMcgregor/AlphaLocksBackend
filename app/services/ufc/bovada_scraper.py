"""
Bovada UFC method odds scraper.

Fetches "How Will Fight End" and "Method of Victory" odds from Bovada's
public API and matches them to fights in our DB by fighter name.

Usage:
    python -m app.services.bovada_scraper          # scrape & store in DB
    python -m app.services.bovada_scraper --dry-run # just print odds, no DB writes
"""

from __future__ import annotations

import argparse
import logging
import re

import httpx

from app.database import SessionLocal
from app.models.ufc import UFCFight, UFCFighter, UFCMethodOdds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bovada_scraper")

BOVADA_API = "https://www.bovada.lv/services/sports/event/v2/events/A/description/ufc-mma"

# Bovada returns events grouped by competition. UFC competitions have
# paths like /ufc-mma/ufc/... while non-UFC (Bellator, PFL, etc.) use
# different sub-paths.
UFC_PATH_PREFIX = "/ufc-mma/ufc/"


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(".", "").replace("'", "").replace("-", " ")


def _names_match(a: str, b: str) -> bool:
    a_norm, b_norm = _normalize_name(a), _normalize_name(b)
    if a_norm == b_norm:
        return True
    a_parts, b_parts = a_norm.split(), b_norm.split()
    if a_parts and b_parts and a_parts[-1] == b_parts[-1]:
        if a_parts[0][0] == b_parts[0][0]:
            return True
    return False


def _parse_american_odds(value) -> int | None:
    """Parse Bovada's American odds value (can be int, str like '+265', or 'EVEN')."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s.upper() == "EVEN":
        return 100
    try:
        return int(s)
    except ValueError:
        return None


def _american_to_implied_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def fetch_bovada_events() -> list[dict]:
    """Fetch all UFC events with odds from Bovada's API."""
    try:
        r = httpx.get(
            BOVADA_API,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            timeout=15,
        )
        if r.status_code != 200:
            log.warning(f"Bovada API returned {r.status_code}")
            return []
        return r.json()
    except httpx.HTTPError as e:
        log.warning(f"HTTP error fetching Bovada: {e}")
        return []


def _extract_method_odds(event: dict) -> dict | None:
    """Extract method odds from a single Bovada event (fight).

    Returns dict with fighter names and odds, or None if no method markets found.
    """
    description = event.get("description", "")
    competitors = event.get("competitors", [])
    link = event.get("link", "")

    # Filter: only UFC events
    if not link.startswith(UFC_PATH_PREFIX):
        return None

    if len(competitors) < 2:
        return None

    # Identify fighters - Bovada uses home/away
    home_name = None
    away_name = None
    for c in competitors:
        if c.get("home"):
            home_name = c.get("name", "")
        else:
            away_name = c.get("name", "")

    if not home_name or not away_name:
        return None

    result = {
        "description": description,
        "home": home_name,
        "away": away_name,
        "link": link,
        # "How Will Fight End" market
        "ko_odds": None,
        "sub_odds": None,
        "dec_odds": None,
        # "Method of Victory" per-fighter
        "home_ko_odds": None,
        "home_sub_odds": None,
        "home_dec_odds": None,
        "away_ko_odds": None,
        "away_sub_odds": None,
        "away_dec_odds": None,
    }

    for dg in event.get("displayGroups", []):
        for market in dg.get("markets", []):
            desc = market.get("description", "")

            if desc == "How Will Fight End":
                for outcome in market.get("outcomes", []):
                    odesc = outcome.get("description", "")
                    price = outcome.get("price", {})
                    american = _parse_american_odds(price.get("american"))
                    if american is None:
                        continue

                    if "KO" in odesc or "TKO" in odesc:
                        result["ko_odds"] = american
                    elif "Submission" in odesc:
                        result["sub_odds"] = american
                    elif "Decision" in odesc:
                        result["dec_odds"] = american

            elif desc == "Method of Victory":
                for outcome in market.get("outcomes", []):
                    odesc = outcome.get("description", "")
                    price = outcome.get("price", {})
                    american = _parse_american_odds(price.get("american"))
                    if american is None:
                        continue

                    # Determine which fighter this outcome belongs to
                    fighter_key = None
                    if home_name and _outcome_matches_fighter(odesc, home_name):
                        fighter_key = "home"
                    elif away_name and _outcome_matches_fighter(odesc, away_name):
                        fighter_key = "away"
                    else:
                        continue

                    if "KO" in odesc or "TKO" in odesc or "DQ" in odesc:
                        result[f"{fighter_key}_ko_odds"] = american
                    elif "Submission" in odesc:
                        result[f"{fighter_key}_sub_odds"] = american
                    elif "Decision" in odesc:
                        result[f"{fighter_key}_dec_odds"] = american

    # Only return if we found at least one method market
    has_how = result["ko_odds"] is not None
    has_mov = result["home_ko_odds"] is not None
    if not has_how and not has_mov:
        return None

    return result


def _outcome_matches_fighter(outcome_desc: str, fighter_name: str) -> bool:
    """Check if an outcome description contains a fighter's last name."""
    last_name = fighter_name.strip().split()[-1].lower()
    return last_name.lower() in outcome_desc.lower()


def scrape_bovada_method_odds(dry_run: bool = False) -> list[dict]:
    """Scrape Bovada for UFC method odds and optionally store in DB.

    Returns list of extracted odds dicts for all UFC fights found.
    """
    log.info("Fetching Bovada UFC odds...")
    api_data = fetch_bovada_events()
    if not api_data:
        log.warning("No data returned from Bovada API")
        return []

    # The API returns a list of competition groups, each with events
    all_odds = []
    for group in api_data:
        events = group.get("events", [])
        for event in events:
            odds = _extract_method_odds(event)
            if odds:
                all_odds.append(odds)

    log.info(f"Found method odds for {len(all_odds)} UFC fights")

    if dry_run:
        for o in all_odds:
            print(f"\n{o['description']}")
            print(f"  How Will Fight End:  KO/TKO {o['ko_odds']}  |  Sub {o['sub_odds']}  |  Dec {o['dec_odds']}")
            print(f"  {o['home']}:  KO {o['home_ko_odds']}  |  Sub {o['home_sub_odds']}  |  Dec {o['home_dec_odds']}")
            print(f"  {o['away']}:  KO {o['away_ko_odds']}  |  Sub {o['away_sub_odds']}  |  Dec {o['away_dec_odds']}")
        return all_odds

    # Match to DB fights and store
    db = SessionLocal()
    try:
        fighters = {f.id: f"{f.first_name} {f.last_name}" for f in db.query(UFCFighter).all()}
        upcoming_fights = (
            db.query(UFCFight)
            .filter(UFCFight.winner_id.is_(None))
            .all()
        )
        if not upcoming_fights:
            log.info("No upcoming fights in DB to match against")
            return all_odds

        fights_info = [
            (f.id, f.red_fighter_id, f.blue_fighter_id)
            for f in upcoming_fights
        ]

        matched = 0
        for odds in all_odds:
            home_name = odds["home"]
            away_name = odds["away"]

            for fight_id, red_fid, blue_fid in fights_info:
                red_name = fighters.get(red_fid, "")
                blue_name = fighters.get(blue_fid, "")

                # Match: home=red & away=blue, or home=blue & away=red
                if _names_match(home_name, red_name) and _names_match(away_name, blue_name):
                    red_is_home = True
                elif _names_match(home_name, blue_name) and _names_match(away_name, red_name):
                    red_is_home = False
                else:
                    continue

                # Compute vig-removed probabilities for "How Will Fight End"
                ko_prob = sub_prob = dec_prob = None
                if odds["ko_odds"] is not None and odds["sub_odds"] is not None and odds["dec_odds"] is not None:
                    ko_imp = _american_to_implied_prob(odds["ko_odds"])
                    sub_imp = _american_to_implied_prob(odds["sub_odds"])
                    dec_imp = _american_to_implied_prob(odds["dec_odds"])
                    total = ko_imp + sub_imp + dec_imp
                    ko_prob = round(ko_imp / total, 4)
                    sub_prob = round(sub_imp / total, 4)
                    dec_prob = round(dec_imp / total, 4)

                # Map home/away to red/blue
                if red_is_home:
                    red_ko = odds["home_ko_odds"]
                    red_sub = odds["home_sub_odds"]
                    red_dec = odds["home_dec_odds"]
                    blue_ko = odds["away_ko_odds"]
                    blue_sub = odds["away_sub_odds"]
                    blue_dec = odds["away_dec_odds"]
                else:
                    red_ko = odds["away_ko_odds"]
                    red_sub = odds["away_sub_odds"]
                    red_dec = odds["away_dec_odds"]
                    blue_ko = odds["home_ko_odds"]
                    blue_sub = odds["home_sub_odds"]
                    blue_dec = odds["home_dec_odds"]

                # Upsert
                existing = (
                    db.query(UFCMethodOdds)
                    .filter(UFCMethodOdds.fight_id == fight_id, UFCMethodOdds.bookmaker == "Bovada")
                    .first()
                )
                if existing:
                    existing.ko_odds = odds["ko_odds"]
                    existing.sub_odds = odds["sub_odds"]
                    existing.dec_odds = odds["dec_odds"]
                    existing.ko_prob = ko_prob
                    existing.sub_prob = sub_prob
                    existing.dec_prob = dec_prob
                    existing.red_ko_odds = red_ko
                    existing.red_sub_odds = red_sub
                    existing.red_dec_odds = red_dec
                    existing.blue_ko_odds = blue_ko
                    existing.blue_sub_odds = blue_sub
                    existing.blue_dec_odds = blue_dec
                else:
                    db.add(UFCMethodOdds(
                        fight_id=fight_id,
                        bookmaker="Bovada",
                        ko_odds=odds["ko_odds"],
                        sub_odds=odds["sub_odds"],
                        dec_odds=odds["dec_odds"],
                        ko_prob=ko_prob,
                        sub_prob=sub_prob,
                        dec_prob=dec_prob,
                        red_ko_odds=red_ko,
                        red_sub_odds=red_sub,
                        red_dec_odds=red_dec,
                        blue_ko_odds=blue_ko,
                        blue_sub_odds=blue_sub,
                        blue_dec_odds=blue_dec,
                    ))
                matched += 1
                log.info(f"  Matched: {odds['description']} -> fight_id={fight_id}")
                break

        db.commit()
        log.info(f"Stored method odds for {matched}/{len(all_odds)} fights")

    finally:
        db.close()

    return all_odds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bovada UFC method odds scraper")
    parser.add_argument("--dry-run", action="store_true", help="Print odds without saving to DB")
    args = parser.parse_args()
    scrape_bovada_method_odds(dry_run=args.dry_run)
