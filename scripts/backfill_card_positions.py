"""Backfill ufc_fights.card_position (0 = main event) for past events.

The column was added in migration 011 and is filled for new events by the scraper, so
history is empty. This fetches each event's ufcstats page once, reads the bout order (the
page lists the main event first) and writes the positions to every database URL given.

Usage:
    python -m scripts.backfill_card_positions --since 2012-01-01 \
        --db postgresql://localhost/alocks_local --db "$PROD_URL"
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import re

from sqlalchemy import create_engine, text

from app.services.ufc.scraper import Scraper

log = logging.getLogger("backfill_card_positions")
FIGHT_ID_RE = re.compile(r"fight-details/([0-9a-f]{16})")


def page_order(scraper: Scraper, event_ufcstats_id: str) -> list[str]:
    soup = scraper.fetch(f"http://ufcstats.com/event-details/{event_ufcstats_id}")
    if soup is None:
        return []
    seen, order = set(), []
    for fid in FIGHT_ID_RE.findall(str(soup)):  # several links per row; keep first sighting
        if fid not in seen:
            seen.add(fid)
            order.append(fid)
    return order


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=dt.date.fromisoformat, default=dt.date(2012, 1, 1))
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--only-missing", action="store_true", default=True)
    a = ap.parse_args()

    engines = [create_engine(u) for u in a.db]
    with engines[0].connect() as c:
        events = c.execute(text("""
            select e.ufcstats_id, e.name, e.date from ufc.ufc_events e
            where e.date >= :s and e.date < current_date and exists (
              select 1 from ufc.ufc_fights f where f.event_id = e.id and f.card_position is null)
            order by e.date"""), {"s": a.since}).all()
    log.info(f"{len(events)} events to backfill since {a.since}")
    scraper = Scraper()
    done = 0
    for ev_id, name, date in events:
        order = page_order(scraper, ev_id)
        if not order:
            log.warning(f"  no bouts parsed for {name} ({date}); skipped")
            continue
        for eng in engines:
            with eng.begin() as c:
                for pos, fid in enumerate(order):
                    c.execute(text("""update ufc.ufc_fights set card_position = :p
                                      where ufcstats_id = :f and card_position is null"""),
                              {"p": pos, "f": fid})
        done += 1
        if done % 25 == 0:
            log.info(f"  {done}/{len(events)} events ({date})")
    log.info(f"backfilled {done} events")


if __name__ == "__main__":
    main()
