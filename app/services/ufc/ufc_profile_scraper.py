"""
UFC profile scraper — fetches fighter nationality from roster.watch
and profile images from UFC.com.

Strategy:
  1. Scrape roster.watch (active + former) — has country emoji flags for all fighters
  2. Convert emoji flags to ISO country codes
  3. Match to DB fighters by name
  4. Optionally scrape UFC.com for profile images

Usage:
    python -m app.services.ufc_profile_scraper                # fetch countries from roster.watch
    python -m app.services.ufc_profile_scraper --images        # also scrape UFC.com for images
    python -m app.services.ufc_profile_scraper --images --limit 50
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import unicodedata

import httpx
from bs4 import BeautifulSoup

from datetime import date, timedelta

from sqlalchemy import or_

from app.database import SessionLocal
from app.models.ufc import UFCFight, UFCFighter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REQUEST_DELAY = 0.3


# ---------------------------------------------------------------------------
# Emoji flag → ISO 3166-1 alpha-2
# ---------------------------------------------------------------------------

def emoji_flag_to_iso(flag_emoji: str) -> str | None:
    """Convert a flag emoji (e.g. 🇺🇸) to ISO country code (e.g. 'US')."""
    if not flag_emoji:
        return None
    # Flag emojis are regional indicator symbols: each letter = 0x1F1E6 + (ord - ord('A'))
    codepoints = [ord(c) for c in flag_emoji if 0x1F1E6 <= ord(c) <= 0x1F1FF]
    if len(codepoints) < 2:
        return None
    # Take first flag only (some fighters have dual nationality like 🇬🇪 🇪🇸)
    a = chr(codepoints[0] - 0x1F1E6 + ord('A'))
    b = chr(codepoints[1] - 0x1F1E6 + ord('A'))
    return f"{a}{b}"


def _normalize(s: str) -> str:
    """Remove accents, lowercase, strip."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


# ---------------------------------------------------------------------------
# Phase 1: Fetch country data from roster.watch (2 pages, instant)
# ---------------------------------------------------------------------------

def fetch_roster_watch() -> dict[str, str]:
    """Fetch fighter names + country codes from roster.watch. Returns {normalized_name: country_code}."""
    log.info("Phase 1: Fetching fighter countries from roster.watch...")
    client = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        follow_redirects=True, timeout=15.0,
    )

    fighters = {}  # normalized_name -> country_code

    for url_label, url in [("active", "https://www.roster.watch"),
                            ("former", "https://www.roster.watch/former.html")]:
        try:
            resp = client.get(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            count = 0
            for row in soup.find_all("tr", attrs={"data-fighter": True}):
                name = row.get("data-fighter", "").strip()
                emoji = row.get("data-country", "").strip()
                if name and emoji:
                    code = emoji_flag_to_iso(emoji)
                    if code:
                        norm = _normalize(name)
                        fighters[norm] = code
                        count += 1
            log.info(f"  {url_label}: {count} fighters with country data")
        except Exception as e:
            log.warning(f"  Error fetching {url_label}: {e}")

    client.close()
    log.info(f"  Total: {len(fighters)} fighters with country codes")
    return fighters


# ---------------------------------------------------------------------------
# Phase 2: Match to DB and update
# ---------------------------------------------------------------------------

def update_countries(roster_data: dict[str, str]):
    """Match roster.watch fighters to DB and update country_code."""
    log.info("Phase 2: Matching to database fighters...")
    db = SessionLocal()

    all_fighters = db.query(UFCFighter).all()
    need_country = [f for f in all_fighters if f.country_code is None]
    log.info(f"  {len(need_country)} fighters need country data")

    matched = 0
    for fighter in need_country:
        norm = _normalize(f"{fighter.first_name} {fighter.last_name}")
        code = roster_data.get(norm)
        if code:
            fighter.country_code = code
            matched += 1

    log.info(f"  Matched {matched}/{len(need_country)} fighters")

    try:
        db.commit()
        log.info(f"  Committed {matched} updates")
    except Exception as e:
        log.warning(f"  Batch commit failed ({e}), trying individual commits...")
        db.rollback()
        committed = 0
        for fighter in need_country:
            if fighter.country_code is not None:
                try:
                    db.add(fighter)
                    db.commit()
                    committed += 1
                except Exception:
                    db.rollback()
        log.info(f"  Individually committed {committed} updates")

    db.close()


# ---------------------------------------------------------------------------
# Phase 3 (optional): Scrape UFC.com for profile images
# ---------------------------------------------------------------------------

def _fighter_slug(first_name: str, last_name: str) -> str:
    name = _normalize(f"{first_name} {last_name}")
    name = re.sub(r"['\".]+", "", name)
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return re.sub(r"-+", "-", name).strip("-")


def scrape_images(limit: int = 0, recent_years: int = 3):
    """Scrape UFC.com for profile images — only fighters active in the last N years."""
    log.info(f"Phase 3: Scraping UFC.com for images (fighters active in last {recent_years} years)...")
    db = SessionLocal()

    cutoff = date.today() - timedelta(days=recent_years * 365)

    # Get IDs of fighters who fought since cutoff
    recent_ids = set()
    recent_fights = (
        db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id)
        .filter(UFCFight.date >= cutoff)
        .all()
    )
    for red_id, blue_id in recent_fights:
        recent_ids.add(red_id)
        recent_ids.add(blue_id)

    log.info(f"  {len(recent_ids)} fighters active since {cutoff}")

    fighters = (
        db.query(UFCFighter)
        .filter(
            UFCFighter.image_url.is_(None),
            UFCFighter.id.in_(recent_ids),
        )
        .all()
    )
    if limit > 0:
        fighters = fighters[:limit]
    log.info(f"  {len(fighters)} recent fighters need images")

    client = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        follow_redirects=False,
        timeout=15.0,
    )

    matched = 0
    for i, fighter in enumerate(fighters):
        slug = _fighter_slug(fighter.first_name, fighter.last_name)
        try:
            resp = client.get(f"https://www.ufc.com/athlete/{slug}")
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            # Try multiple selectors — UFC.com layout varies
            img = (
                soup.find("img", class_="hero-profile__image")
                or soup.select_one("#block-mainpagecontent img")
            )
            if img and img.get("src"):
                image_url = img["src"]
                if not image_url.startswith("http"):
                    image_url = f"https://www.ufc.com{image_url}"
                fighter.image_url = image_url
                matched += 1
                try:
                    db.commit()
                except Exception:
                    db.rollback()
            if (i + 1) % 50 == 0:
                log.info(f"  [{i+1}/{len(fighters)}] {matched} images found")
        except Exception:
            pass

        time.sleep(REQUEST_DELAY)

    client.close()
    db.close()
    log.info(f"  Found {matched} images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(images: bool = False, limit: int = 0):
    log.info("=" * 60)
    log.info("UFC PROFILE SCRAPER")
    log.info("=" * 60)

    # Phase 1+2: Countries from roster.watch (fast, 2 HTTP requests)
    roster_data = fetch_roster_watch()
    update_countries(roster_data)

    # Phase 3: Images from UFC.com (slow, optional)
    if images:
        scrape_images(limit=limit)

    log.info("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", action="store_true", help="Also scrape UFC.com for profile images")
    parser.add_argument("--limit", type=int, default=0, help="Max images to scrape (0=all)")
    args = parser.parse_args()
    run(images=args.images, limit=args.limit)
