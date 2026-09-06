"""Re-scrape specific events that are already in the database.

WHY NOT run_recent_update()
---------------------------
`scraper.run_recent_update()` finds the most recent *completed* event in the DB and only
scrapes events newer than that. The gaps we found are in events that are already present
and already considered complete — UFC 329 (2026-07-11) holds 16 bouts with only 7 results,
including McGregor vs Holloway. Nothing in the existing pipeline will ever revisit them.

This script targets events explicitly and re-runs `scrape_fight_details` + `upsert_fight`
for every bout on the card, which fills in winners, methods and stats for bouts that were
scraped while still upcoming.

New fighters are handled: if a bout references a fighter not in the DB, `upsert_fight`
skips it with a warning, so `--fix-fighters` first refreshes the fighter listing.

SAFETY
------
Writes to whatever DATABASE_URL points at. Run it against the local copy first:

    DATABASE_URL=postgresql://localhost/alocks_local \\
      python -m scripts.rescrape_events --since 2026-05-01

Then verify with scripts.verify_events before pointing it at production.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time

from sqlalchemy import make_url

from app.config import settings
from app.database import SessionLocal
from app.models.ufc import UFCEvent, UFCFight
from app.services.ufc.scraper import (
    REQUEST_DELAY,
    Scraper,
    _scrape_fighter_listings_only,
    scrape_event_fights,
    scrape_fight_details,
    upsert_fight,
    upsert_fighters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("rescrape")

EVENT_URL = "http://ufcstats.com/event-details/{}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="", help="YYYY-MM-DD; re-scrape events on/after")
    ap.add_argument("--event", default="", help="single event ufcstats_id")
    ap.add_argument("--limit", type=int, default=0, help="most recent N past events")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--fix-fighters", action="store_true",
                    help="refresh the fighter roster first, so debutants resolve")
    args = ap.parse_args()

    url = make_url(settings.DATABASE_URL)
    log.info(f"target database: {url.host or 'local'}/{url.database}")
    if not args.dry_run and url.host and "rlwy" in url.host:
        log.warning("!! This is PRODUCTION. Ctrl-C within 5s to abort.")
        time.sleep(5)

    db = SessionLocal()
    q = db.query(UFCEvent).filter(UFCEvent.date.isnot(None))
    if args.event:
        events = q.filter(UFCEvent.ufcstats_id == args.event).all()
    else:
        q = q.filter(UFCEvent.date <= dt.date.today())
        if args.since:
            events = q.filter(UFCEvent.date >= dt.date.fromisoformat(args.since)) \
                      .order_by(UFCEvent.date.desc()).all()
        else:
            events = q.order_by(UFCEvent.date.desc()).limit(args.limit or 10).all()

    if not events:
        log.error("no matching events")
        return 1

    scraper = Scraper()

    if args.fix_fighters and not args.dry_run:
        # upsert_fight requires BOTH fighters to already exist and silently skips the
        # bout otherwise, so a card featuring a debutant loses that bout entirely.
        # Refresh the roster first. The a-z listing pages are ~26 requests and give
        # name + record without needing each fighter's detail page.
        log.info("Refreshing fighter roster from listing pages...")
        listed = _scrape_fighter_listings_only(scraper)
        if listed:
            upsert_fighters(db, listed)
            db.commit()
            log.info(f"  upserted {len(listed)} fighter records")
        else:
            log.warning("  no fighters returned; continuing without roster refresh")

    total_fights = total_new = total_updated = 0

    for ev in sorted(events, key=lambda e: e.date):
        before = db.query(UFCFight).filter(UFCFight.event_id == ev.id).count()
        before_decided = db.query(UFCFight).filter(
            UFCFight.event_id == ev.id, UFCFight.winner_id.isnot(None)
        ).count()

        links = scrape_event_fights(scraper, EVENT_URL.format(ev.ufcstats_id))
        log.info(f"{ev.date}  {str(ev.name)[:44]:<44} card={len(links):>2} "
                 f"db_before={before:>2} decided={before_decided:>2}")
        if args.dry_run:
            continue

        for fight_url in links:
            data = scrape_fight_details(scraper, fight_url)
            if not data:
                log.warning(f"    could not read {fight_url}")
                continue
            data["date"] = ev.date
            try:
                upsert_fight(db, data, ev.id)
                total_fights += 1
            except Exception:
                log.exception(f"    upsert failed for {fight_url}")
                db.rollback()
            time.sleep(REQUEST_DELAY)

        db.commit()
        after = db.query(UFCFight).filter(UFCFight.event_id == ev.id).count()
        after_decided = db.query(UFCFight).filter(
            UFCFight.event_id == ev.id, UFCFight.winner_id.isnot(None)
        ).count()
        total_new += after - before
        total_updated += after_decided - before_decided
        log.info(f"    -> db_after={after} decided={after_decided} "
                 f"(+{after - before} bouts, +{after_decided - before_decided} results)")

    db.close()
    scraper.close()
    log.info(f"Done. {total_fights} bouts processed, "
             f"+{total_new} new, +{total_updated} newly decided.")
    log.info("Now run: python -m scripts.verify_events --since <same date>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
