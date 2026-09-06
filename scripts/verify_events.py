"""Reconcile the database against ufcstats.com for recent events.

WHY
---
The scrape fails silently: a partial scrape looks exactly like a complete one. For
UFC Fight Night: Nurmagomedov vs Song (2026-08-29) the site listed 13 bouts and the
database held 10. Across the twelve most recent events, 17 bouts were absent and 13 had a
result on the site but none in the database — including headline fights.

THREE OUTCOMES, AND ONLY TWO ARE PROBLEMS
-----------------------------------------
- MISSING   : on the site, absent from the DB. A real scrape gap. Re-scrape.
- NO RESULT : both have the bout, the site has a winner and we do not. A real gap.
- CANCELLED : in the DB, absent from the card listing, and carries NO winner. This is
              almost always a replaced or scrapped booking. ufcstats removes the bout
              from the event page but keeps its fight-details page, which still names the
              event — so the two pages genuinely disagree and both records are
              defensible. Verified by hand: Liu Ce vs Junior Tafa (81efb50e7487a7ea) is
              exactly this; Tafa was replaced by Levi Rodrigues Jr. and the DB holds
              both bouts. All 24 such rows found had no winner. **Benign** — they are
              dropped from model training. Reported for visibility, but do not fail the
              check.
- PHANTOM   : absent from the card listing yet HAS a result. That would be genuine
              corruption. None were found.

Scale, for perspective: ~30 genuinely affected fights out of 11,545 (0.26%), all in 2026.
The practical cost is that recent fighter records, streaks and ratings are stale, which
degrades live picks more than the backtest.

Usage:
    python -m scripts.verify_events                  # last 10 past events
    python -m scripts.verify_events --limit 25
    python -m scripts.verify_events --since 2026-05-01
    python -m scripts.verify_events --event 9d61d8cb1c354867
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sys

from app.database import SessionLocal
from app.models.ufc import UFCEvent, UFCFight, UFCFighter
from app.services.ufc.scraper import Scraper

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
# httpx logs every request at INFO and drowns the report.
logging.getLogger("httpx").setLevel(logging.WARNING)

EVENT_URL = "http://ufcstats.com/event-details/{}"
FIGHT_ID_RE = re.compile(r"fight-details/([0-9a-f]+)")


def scrape_card(scraper: Scraper, ufcstats_id: str) -> list[dict] | None:
    """Return one dict per bout on the live page, or None if the page can't be read."""
    soup = scraper.fetch(EVENT_URL.format(ufcstats_id))
    if soup is None:
        return None

    bouts = []
    for row in soup.select("tr.b-fight-details__table-row"):
        link = row.get("data-link", "") or ""
        m = FIGHT_ID_RE.search(link)
        if not m:
            continue  # header row
        cells = row.select("td")

        def cell_lines(i):
            if i >= len(cells):
                return []
            ps = cells[i].select("p")
            src = ps or [cells[i]]
            return [re.sub(r"\s+", " ", p.get_text(strip=True)) for p in src if p.get_text(strip=True)]

        names = cell_lines(1)
        bouts.append({
            "fight_id": m.group(1),
            "fighters": names,
            "weight_class": " ".join(cell_lines(6)) or "?",
            "method": " ".join(cell_lines(7)) or "?",
            "round": " ".join(cell_lines(8)) or "?",
        })
    return bouts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10, help="how many recent past events")
    ap.add_argument("--since", default="", help="YYYY-MM-DD lower bound instead of --limit")
    ap.add_argument("--event", default="", help="check a single event ufcstats_id")
    args = ap.parse_args()

    db = SessionLocal()
    q = db.query(UFCEvent)
    if args.event:
        events = q.filter(UFCEvent.ufcstats_id == args.event).all()
        if not events:
            sys.exit(f"no event with ufcstats_id={args.event}")
    else:
        today = dt.date.today()
        q = q.filter(UFCEvent.date.isnot(None), UFCEvent.date <= today)
        if args.since:
            q = q.filter(UFCEvent.date >= dt.date.fromisoformat(args.since))
            events = q.order_by(UFCEvent.date.desc()).all()
        else:
            events = q.order_by(UFCEvent.date.desc()).limit(args.limit).all()

    names = {f.id: f"{f.first_name} {f.last_name}".strip()
             for f in db.query(UFCFighter).all()}
    scraper = Scraper()

    total_missing = total_phantom = total_noresult = total_dropped = 0
    unreadable = []

    print(f"Checking {len(events)} event(s) against ufcstats.com\n")
    for ev in sorted(events, key=lambda e: e.date, reverse=True):
        live = scrape_card(scraper, ev.ufcstats_id)
        if live is None:
            unreadable.append(ev)
            print(f"{ev.date}  {str(ev.name)[:48]:<48}  COULD NOT FETCH")
            continue

        db_fights = db.query(UFCFight).filter(UFCFight.event_id == ev.id).all()
        db_ids = {f.ufcstats_id: f for f in db_fights}
        live_ids = {b["fight_id"]: b for b in live}

        missing = [b for fid, b in live_ids.items() if fid not in db_ids]

        # A DB fight absent from the card listing is USUALLY a cancelled or replaced
        # booking, not corruption. ufcstats removes the bout from the event page but
        # keeps its fight-details page, which still names the event — so the two pages
        # genuinely disagree and both our records are defensible. Verified by hand:
        # Liu Ce vs Junior Tafa (81efb50e7487a7ea) is exactly this — Tafa was replaced
        # by Levi Rodrigues Jr., and the DB holds both bouts.
        #
        # These are harmless to the model (no winner -> dropped from training). Only a
        # dropped bout that somehow HAS a result is a real integrity problem.
        dropped = [f for fid, f in db_ids.items()
                   if fid not in live_ids and f.winner_id is None]
        phantom = [f for fid, f in db_ids.items()
                   if fid not in live_ids and f.winner_id is not None]
        # A bout the site shows as finished but we hold with no winner.
        noresult = [
            f for fid, f in db_ids.items()
            if fid in live_ids and f.winner_id is None
            and live_ids[fid]["method"] not in ("?", "")
        ]

        total_missing += len(missing)
        total_phantom += len(phantom)
        total_noresult += len(noresult)
        total_dropped += len(dropped)

        # Cancelled bookings are expected and benign, so they do not fail the check.
        real_problem = missing or phantom or noresult
        status = "OK" if not real_problem else "MISMATCH"
        extra = f"  (+{len(dropped)} cancelled)" if dropped else ""
        print(f"{ev.date}  {str(ev.name)[:48]:<48}  site={len(live_ids):>2} db={len(db_ids):>2}  {status}{extra}")

        for b in missing:
            print(f"     MISSING  {' vs '.join(b['fighters'][:2]):<44} {b['method']}  [{b['fight_id']}]")
        for f in phantom:
            r = names.get(f.red_fighter_id, "?"); bl = names.get(f.blue_fighter_id, "?")
            print(f"     PHANTOM  {r} vs {bl:<30} has a RESULT but is off-card  [{f.ufcstats_id}]")
        for f in noresult:
            r = names.get(f.red_fighter_id, "?"); bl = names.get(f.blue_fighter_id, "?")
            print(f"     NO RESULT {r} vs {bl:<32} site says {live_ids[f.ufcstats_id]['method']}")

    db.close()

    print(f"\n{'-' * 64}")
    print(f"missing bouts   : {total_missing}   (on site, absent from DB)")
    print(f"result gaps     : {total_noresult}   (site has a result, DB does not)")
    print(f"phantom bouts   : {total_phantom}   (has a result but off-card - real corruption)")
    print(f"cancelled bookings: {total_dropped} (benign: replaced/scrapped bouts, no winner)")
    if unreadable:
        print(f"unreadable pages: {len(unreadable)}")
    clean = not (total_missing or total_phantom or total_noresult or unreadable)
    print("All checked events reconcile." if clean else "Discrepancies found — re-scrape needed.")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
