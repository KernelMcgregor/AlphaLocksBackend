"""
UFC profile scraper — fetches fighter bio data and profile images from UFC.com.

One request per fighter to https://www.ufc.com/athlete/<slug> yields everything:
the hero image plus the bio table (place of birth, gym, fighting style, leg reach,
octagon debut, status). Nationality is derived from the birthplace country.

This previously sourced nationality from roster.watch, which now sits behind a
Cloudflare challenge and returns 403 to any plain client. UFC.com serves these pages
without a challenge, so it is both the richer and the only working source.

Usage:
    python -m app.services.ufc.ufc_profile_scraper --dry-run --limit 20   # rehearse, writes nothing
    python -m app.services.ufc.ufc_profile_scraper --limit 20             # small real run
    python -m app.services.ufc.ufc_profile_scraper                        # full backfill
    python -m app.services.ufc.ufc_profile_scraper --recent-years 3       # only recently active
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import time
import unicodedata
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from datetime import date, datetime, timedelta

from sqlalchemy import or_

from app.database import SessionLocal
from app.models.ufc import UFCFight, UFCFighter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REQUEST_DELAY = 0.3

#: UFC.com bio label -> UFCFighter attribute. Anything else on the page is ignored.
BIO_LABEL_TO_FIELD = {
    "Place of Birth": "birthplace",
    "Fighting style": "fighting_style",
    "Trains at": "trains_at",
    "Leg reach": "leg_reach",
    "Status": "status",
}

#: Fields used to decide whether a fighter still needs scraping. Deliberately excludes
#: fighting_style / trains_at / leg_reach: those are legitimately absent for much of the
#: roster (UFC.com lists a style for only a minority), so keying off them would re-scrape
#: thousands of fighters on every run forever. These three were present for every fighter
#: sampled, so a NULL here means "not scraped yet" rather than "UFC.com has no value".
PROFILE_SENTINEL_FIELDS = ("image_url", "birthplace", "status")

#: Substrings identifying UFC.com's "no portrait" stand-ins. See extract_profile.
PLACEHOLDER_IMAGE_MARKERS = ("no-profile-image", "SHADOW_Fighter")

#: Country name as UFC.com spells it -> ISO 3166-1 alpha-2. Hand-maintained rather than
#: pulling in pycountry for one lookup; the roster spans a bounded set of countries and
#: unmapped names are logged at the end of a run so this can be extended.
COUNTRY_TO_ISO = {
    "American Samoa": "AS", "Anguilla": "AI", "Bahamas": "BS", "Cabo Verde": "CV",
    "Cape Verde": "CV", "Cyprus": "CY", "Democratic Republic of the Congo": "CD",
    "Guinea": "GN", "Hong Kong": "HK", "Myanmar": "MM", "Solomon Islands": "SB",
    "Uganda": "UG",
    #: UFC.com spells Turkey with the Turkish glyph, and lists Canary Islands (Spain)
    #: and Bosnia with an ampersand rather than "and".
    "Türkiye": "TR", "Canary Islands": "ES", "Bosnia & Herzegovina": "BA",
    "Afghanistan": "AF", "Albania": "AL", "Algeria": "DZ", "Angola": "AO",
    "Argentina": "AR", "Armenia": "AM", "Australia": "AU", "Austria": "AT",
    "Azerbaijan": "AZ", "Bahrain": "BH", "Belarus": "BY", "Belgium": "BE",
    "Bolivia": "BO", "Bosnia and Herzegovina": "BA", "Brazil": "BR", "Bulgaria": "BG",
    "Cameroon": "CM", "Canada": "CA", "Chile": "CL", "China": "CN",
    "Colombia": "CO", "Congo": "CG", "Costa Rica": "CR", "Croatia": "HR",
    "Cuba": "CU", "Czech Republic": "CZ", "Czechia": "CZ", "Denmark": "DK",
    "Dominican Republic": "DO", "Ecuador": "EC", "Egypt": "EG", "El Salvador": "SV",
    #: ISO 3166-2 subdivisions, not alpha-2 — flag-icons renders these as the individual
    #: home-nation flags. "United Kingdom" stays plain GB (Union Jack) since UFC.com does
    #: not say which nation it means.
    "England": "GB-ENG", "Scotland": "GB-SCT", "Wales": "GB-WLS",
    "Northern Ireland": "GB-NIR",
    "Estonia": "EE", "Finland": "FI", "France": "FR",
    "Georgia": "GE", "Germany": "DE", "Ghana": "GH", "Greece": "GR",
    "Guam": "GU", "Guyana": "GY", "Hungary": "HU", "Iceland": "IS",
    "India": "IN", "Indonesia": "ID", "Iran": "IR", "Iraq": "IQ",
    "Ireland": "IE", "Israel": "IL", "Italy": "IT", "Jamaica": "JM",
    "Japan": "JP", "Jordan": "JO", "Kazakhstan": "KZ", "Kenya": "KE",
    "Kyrgyzstan": "KG", "Latvia": "LV", "Lebanon": "LB", "Lithuania": "LT",
    "Macedonia": "MK", "North Macedonia": "MK", "Malaysia": "MY", "Mexico": "MX",
    "Moldova": "MD", "Mongolia": "MN", "Montenegro": "ME", "Morocco": "MA",
    "Netherlands": "NL", "New Zealand": "NZ", "Nicaragua": "NI", "Nigeria": "NG",
    "Norway": "NO", "Pakistan": "PK", "Panama": "PA",
    "Paraguay": "PY", "Peru": "PE", "Philippines": "PH", "Poland": "PL",
    "Portugal": "PT", "Puerto Rico": "PR", "Romania": "RO", "Russia": "RU",
    "Saudi Arabia": "SA", "Senegal": "SN", "Serbia": "RS",
    "Singapore": "SG", "Slovakia": "SK", "Slovenia": "SI", "Somalia": "SO",
    "South Africa": "ZA", "South Korea": "KR", "Korea": "KR", "Spain": "ES",
    "Suriname": "SR", "Sweden": "SE", "Switzerland": "CH", "Syria": "SY",
    "Taiwan": "TW", "Tajikistan": "TJ", "Thailand": "TH", "Trinidad and Tobago": "TT",
    "Tunisia": "TN", "Turkey": "TR", "Turkmenistan": "TM", "Ukraine": "UA",
    "United Arab Emirates": "AE", "United Kingdom": "GB", "United States": "US",
    "Uruguay": "UY", "Uzbekistan": "UZ", "Venezuela": "VE", "Vietnam": "VN",
    "Zimbabwe": "ZW",
}


def _normalize(s: str) -> str:
    """Remove accents, lowercase, strip."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _fighter_slug(first_name: str, last_name: str) -> str:
    name = _normalize(f"{first_name} {last_name}")
    name = re.sub(r"['\".]+", "", name)
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return re.sub(r"-+", "-", name).strip("-")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_bio(soup: BeautifulSoup) -> dict[str, str]:
    """Extract the UFC.com bio table as {label: value}.

    Returns {} for pages with no bio block — some retired fighters have a stub page.
    """
    bio = {}
    for field in soup.select(".c-bio__field"):
        label_el = field.select_one(".c-bio__label")
        text_el = field.select_one(".c-bio__text")
        if not label_el or not text_el:
            continue
        label = label_el.get_text(strip=True)
        # unescape() because some pages double-encode entities, which survives get_text()
        # as a literal "&amp;" and broke the country lookup for "Bosnia &amp; Herzegovina".
        value = html.unescape(" ".join(text_el.get_text().split()))
        if label and value:
            bio[label] = value
    return bio


def parse_octagon_debut(value: str) -> date | None:
    """Parse UFC.com's debut date, e.g. "Oct. 15, 2016".

    Note the period after the month abbreviation — that is why this cannot reuse the
    "%b %d, %Y" format the ufcstats scraper uses for DOB. Both spellings are attempted
    since the punctuation is not guaranteed across pages.
    """
    for fmt in ("%b. %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def parse_birth_country(birthplace: str) -> str | None:
    """Last comma-segment of a birthplace.

    UFC.com is usually "City, Country" ("Wollongong, Australia") but sometimes gives the
    country alone ("Germany"), so the last segment is the country either way.
    """
    if not birthplace:
        return None
    return birthplace.split(",")[-1].strip() or None


def extract_profile(soup: BeautifulSoup) -> dict:
    """Build the set of UFCFighter field updates for one athlete page."""
    bio = parse_bio(soup)
    updates = {}

    for label, field in BIO_LABEL_TO_FIELD.items():
        if label in bio:
            updates[field] = bio[label]

    if "Octagon Debut" in bio:
        debut = parse_octagon_debut(bio["Octagon Debut"])
        if debut:
            updates["octagon_debut"] = debut

    birthplace = updates.get("birthplace")
    if birthplace:
        country = parse_birth_country(birthplace)
        if country:
            updates["birth_country"] = country
            iso = COUNTRY_TO_ISO.get(country)
            if iso:
                updates["country_code"] = iso

    # Try multiple selectors — UFC.com layout varies
    img = (
        soup.find("img", class_="hero-profile__image")
        or soup.select_one("#block-mainpagecontent img")
    )
    if img and img.get("src"):
        src = img["src"]
        # UFC.com has two stand-ins for an athlete with no portrait, and both were being
        # stored as though they were real headshots:
        #   * '../themes/custom/ufc/assets/img/no-profile-image.png' — concatenating the
        #     host onto that produced 'https://www.ufc.com../themes/...', a malformed URL
        #     on 173 fighters.
        #   * 'SHADOW_Fighter_fullLength_RED.png' — a valid URL for a black silhouette,
        #     on 10 more. It loads, which is worse: it renders as a featureless body.
        # Skipping both lets the UI fall back to initials, and join relative paths
        # properly so a leading '..' can never survive.
        if not any(p in src for p in PLACEHOLDER_IMAGE_MARKERS):
            updates["image_url"] = urljoin("https://www.ufc.com/", src)

    return updates


# ---------------------------------------------------------------------------
# Scrape
# ---------------------------------------------------------------------------

def _select_fighters(db, recent_years: int, limit: int) -> list[UFCFighter]:
    """Fighters still missing profile data, newest-relevant first."""
    query = db.query(UFCFighter).filter(
        or_(*[getattr(UFCFighter, f).is_(None) for f in PROFILE_SENTINEL_FIELDS])
    )

    if recent_years > 0:
        cutoff = date.today() - timedelta(days=recent_years * 365)
        recent_ids = set()
        for red_id, blue_id in (
            db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id)
            .filter(UFCFight.date >= cutoff)
            .all()
        ):
            recent_ids.add(red_id)
            recent_ids.add(blue_id)
        log.info(f"  {len(recent_ids)} fighters active since {cutoff}")
        query = query.filter(UFCFighter.id.in_(recent_ids))

    fighters = query.all()
    if limit > 0:
        fighters = fighters[:limit]
    return fighters


def scrape_profiles(limit: int = 0, recent_years: int = 0, dry_run: bool = False):
    """Fetch UFC.com athlete pages and fill bio fields + image_url."""
    mode = "DRY RUN — no writes" if dry_run else "writing to database"
    log.info(f"Scraping UFC.com athlete profiles ({mode})...")
    db = SessionLocal()

    fighters = _select_fighters(db, recent_years, limit)
    log.info(f"  {len(fighters)} fighters need profile data")

    client = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        follow_redirects=False,
        timeout=15.0,
    )

    updated = 0
    not_found: list[str] = []
    unmapped_countries: set[str] = set()
    country_changes: list[str] = []
    field_counts = {f: 0 for f in (
        "birthplace", "birth_country", "fighting_style", "trains_at",
        "leg_reach", "octagon_debut", "status", "country_code", "image_url",
    )}

    for i, fighter in enumerate(fighters):
        name = f"{fighter.first_name} {fighter.last_name}"
        slug = _fighter_slug(fighter.first_name, fighter.last_name)
        try:
            resp = client.get(f"https://www.ufc.com/athlete/{slug}")
            if resp.status_code != 200:
                not_found.append(f"{name} ({slug}) -> {resp.status_code}")
                continue

            updates = extract_profile(BeautifulSoup(resp.text, "html.parser"))
            if not updates:
                not_found.append(f"{name} ({slug}) -> empty page")
                continue

            # Surface nationality overwrites: this run is the authority on country_code
            # now that roster.watch is dead, so any disagreement is a real change of record.
            new_code = updates.get("country_code")
            if new_code and fighter.country_code and new_code != fighter.country_code:
                country_changes.append(f"{name}: {fighter.country_code} -> {new_code}")

            if updates.get("birth_country") and not new_code:
                unmapped_countries.add(updates["birth_country"])

            for field in updates:
                if field in field_counts:
                    field_counts[field] += 1
            updated += 1

            # Nothing is assigned to the ORM object under --dry-run, so there is no dirty
            # state an incidental autoflush could push to the live database.
            if not dry_run:
                for field, value in updates.items():
                    setattr(fighter, field, value)
                try:
                    db.commit()
                except Exception:
                    db.rollback()

            if (i + 1) % 50 == 0:
                log.info(f"  [{i+1}/{len(fighters)}] {updated} profiles parsed")
        except Exception as e:
            log.warning(f"  {name} ({slug}): {e}")

        time.sleep(REQUEST_DELAY)

    if dry_run:
        db.rollback()
    client.close()
    db.close()

    # ---- Report ----
    log.info("-" * 60)
    log.info(f"  {updated}/{len(fighters)} profiles {'parsed' if dry_run else 'updated'}")
    for field, count in field_counts.items():
        log.info(f"    {field:<16} {count}")

    if country_changes:
        log.info(f"  country_code changed for {len(country_changes)} fighters:")
        for line in country_changes[:50]:
            log.info(f"    {line}")
        if len(country_changes) > 50:
            log.info(f"    ... and {len(country_changes) - 50} more")

    if unmapped_countries:
        log.warning(f"  {len(unmapped_countries)} countries missing from COUNTRY_TO_ISO: "
                    f"{sorted(unmapped_countries)}")

    if not_found:
        log.warning(f"  {len(not_found)} fighters had no usable UFC.com page:")
        for line in not_found[:50]:
            log.warning(f"    {line}")
        if len(not_found) > 50:
            log.warning(f"    ... and {len(not_found) - 50} more")


def remap_countries(dry_run: bool = False):
    """Re-derive country_code from the stored birth_country, with no HTTP at all.

    birth_country is captured verbatim, so extending COUNTRY_TO_ISO does not require
    re-scraping — this fills in fighters whose country name was unmapped at scrape time.
    Only touches rows where country_code is currently NULL, so it cannot overwrite a
    nationality that is already set.
    """
    log.info(f"Remapping country_code from birth_country ({'DRY RUN' if dry_run else 'writing'})...")
    db = SessionLocal()

    fighters = (
        db.query(UFCFighter)
        .filter(UFCFighter.birth_country.isnot(None), UFCFighter.country_code.is_(None))
        .all()
    )
    log.info(f"  {len(fighters)} fighters have a birth_country but no country_code")

    filled = 0
    still_unmapped: dict[str, int] = {}
    for fighter in fighters:
        iso = COUNTRY_TO_ISO.get(fighter.birth_country)
        if iso:
            if not dry_run:
                fighter.country_code = iso
            filled += 1
        else:
            still_unmapped[fighter.birth_country] = still_unmapped.get(fighter.birth_country, 0) + 1

    if dry_run:
        db.rollback()
    else:
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            log.warning(f"  commit failed: {e}")

    db.close()
    log.info(f"  {filled} country_code values {'would be' if dry_run else ''} filled")
    if still_unmapped:
        log.warning(f"  still unmapped: {sorted(still_unmapped.items(), key=lambda kv: -kv[1])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(limit: int = 0, recent_years: int = 0, dry_run: bool = False):
    log.info("=" * 60)
    log.info("UFC PROFILE SCRAPER")
    log.info("=" * 60)
    scrape_profiles(limit=limit, recent_years=recent_years, dry_run=dry_run)
    log.info("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max fighters to scrape (0=all)")
    parser.add_argument("--recent-years", type=int, default=0,
                        help="Only fighters active in the last N years (0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and parse but commit nothing — safe against production")
    parser.add_argument("--remap-countries", action="store_true",
                        help="Only re-derive country_code from stored birth_country (no HTTP)")
    args = parser.parse_args()
    if args.remap_countries:
        remap_countries(dry_run=args.dry_run)
    else:
        run(limit=args.limit, recent_years=args.recent_years, dry_run=args.dry_run)
