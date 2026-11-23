"""Backfill ufc_fight_bonuses (FOTN / POTN / KO / Submission of the Night) from ufcstats.

The nightly scrape reads bonuses off each new fight page. History, and bonuses announced
after that scrape ran, come from here: one event page per card (the bout rows carry the
icons), synced to exactly what the page shows.

Usage:
    python -m scripts.backfill_fight_bonuses --since 2000-01-01 [--dry-run]
    python -m scripts.backfill_fight_bonuses --days 14      # the post-event recheck
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import logging
import time

from app.database import SessionLocal, engine
from app.models.ufc import UFCEvent, UFCFight
from app.models.ufc_highlights import UFCFightBonus
from app.services.ufc.fight_bonuses import event_page_bonuses, sync_fight_bonuses
from app.services.ufc.scraper import BASE_URL, REQUEST_DELAY, Scraper

log = logging.getLogger("backfill_fight_bonuses")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=dt.date.fromisoformat)
    ap.add_argument("--days", type=int, help="only events in the last N days")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    since = a.since or (dt.date.today() - dt.timedelta(days=a.days) if a.days else dt.date(1993, 1, 1))

    if not a.dry_run:
        UFCFightBonus.__table__.create(bind=engine, checkfirst=True)
    db = SessionLocal()
    events = (db.query(UFCEvent).filter(UFCEvent.date >= since, UFCEvent.date < dt.date.today())
              .order_by(UFCEvent.date).all())
    log.info(f"{len(events)} events since {since}")
    scraper = Scraper()
    per_year: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    missing = 0
    for i, ev in enumerate(events, 1):
        soup = scraper.fetch(f"{BASE_URL}/event-details/{ev.ufcstats_id}")
        if soup is None:
            log.warning(f"  no page for {ev.name} ({ev.date})")
            continue
        page = event_page_bonuses(soup)
        fights = {f.ufcstats_id: f for f in db.query(UFCFight).filter(UFCFight.ufcstats_id.in_(list(page)))}
        for uid, kinds in page.items():
            f = fights.get(uid)
            if f is None:
                missing += bool(kinds)
                continue
            per_year[ev.date.year].update(kinds)
            if not a.dry_run:
                sync_fight_bonuses(db, f, kinds)
        if not a.dry_run:
            db.commit()
        if i % 25 == 0:
            log.info(f"  {i}/{len(events)} events ({ev.date})")
        time.sleep(REQUEST_DELAY)
    db.close()

    for y in sorted(per_year):
        c = per_year[y]
        log.info(f"{y}: " + "  ".join(f"{k}={c[k]}" for k in ("fotn", "potn", "kotn", "sotn")))
    if missing:
        log.warning(f"{missing} bonus bouts on ufcstats have no fight row here")


if __name__ == "__main__":
    main()
