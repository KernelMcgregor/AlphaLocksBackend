"""Every market a sportsbook lists for upcoming UFC bouts, read directly from the book.

Each book module (fanduel.py, bovada.py, betrivers.py, pinnacle.py) fetches and parses its own
API into bouts:

    {"external_event_id", "name", "fighter_a", "fighter_b", "date",
     "parse": callable(swapped) -> [row]}

and this module does the rest, the same way for every book: match bouts to upcoming
ufc_fights (bfo_scraper.match_matchups), and append each selection whose price or line moved
to ufc_book_market_history. Rows carry a market_key when they map onto a market this project
prices; app/services/ufc/picks_v2.py reads those.

Usage (runs every book; one failing does not stop the others):
    python -m app.services.ufc.book_markets [--books fanduel,bovada,betrivers,pinnacle] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re

log = logging.getLogger("book_markets")


# ---------------------------------------------------------------------------
# shared parsing helpers
# ---------------------------------------------------------------------------

def side_of(text: str, a: str, b: str) -> str | None:
    """'a' / 'b' when the text names exactly one of the two fighters (full name, else surname)."""
    t = text.lower()
    has_a, has_b = a.lower() in t, b.lower() in t
    if has_a == has_b:  # neither, or both (e.g. "A by Sub or B by Points")
        la, lb = a.split()[-1].lower(), b.split()[-1].lower()
        has_a, has_b = la in t, lb in t
        if has_a == has_b:
            return None
    return "a" if has_a else "b"


def ladder_line(name: str) -> float | None:
    """'40+' -> 39.5, so a ladder rung reads as an over/under line."""
    m = re.fullmatch(r"\s*(\d+)\+\s*", name)
    return int(m.group(1)) - 0.5 if m else None


def corners(swapped: bool) -> dict:
    """fighter_a / fighter_b -> our corner. swapped: fighter_a is our blue corner."""
    return {"a": "blue" if swapped else "red", "b": "red" if swapped else "blue"}


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------

def _db_fights(db, today):
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter
    fights = (db.query(UFCFight, UFCEvent).join(UFCEvent, UFCEvent.id == UFCFight.event_id)
              .filter(UFCEvent.date >= today - dt.timedelta(days=1), UFCFight.winner_id.is_(None)).all())
    names = {f.id: f for f in db.query(UFCFighter).filter(UFCFighter.id.in_(
        {i for fi, _ in fights for i in (fi.red_fighter_id, fi.blue_fighter_id)}))}
    full = lambda i: f"{names[i].first_name} {names[i].last_name}"
    return [{"id": f.id, "fight_date": e.date, "event_name": e.name,
             "red": full(f.red_fighter_id), "blue": full(f.blue_fighter_id)}
            for f, e in fights if f.red_fighter_id in names and f.blue_fighter_id in names]


def record(book: str, bouts: list[dict], dry_run: bool = False) -> dict:
    from app.database import SessionLocal, engine
    from app.models.ufc import UFCBookMarketHistory as H
    from app.services.ufc import bfo_scraper as bfo

    H.__table__.create(engine, checkfirst=True)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    today = now.date()
    db = SessionLocal()
    stats = {"book": book, "bouts_listed": len(bouts), "bouts_matched": 0, "rows": 0, "unmatched": []}
    try:
        db_fights = _db_fights(db, today)
        date_of = {f["id"]: f["fight_date"] for f in db_fights}
        last = {}
        for h in (db.query(H).filter(H.book == book, H.captured_at >= now - dt.timedelta(days=30))
                  .order_by(H.captured_at)):
            last[(h.external_market_id, h.external_selection_id)] = (h.american, h.line)
        matches = bfo.match_matchups(
            [{"bfo_matchup_id": b["external_event_id"], "fighter_a": b["fighter_a"],
              "fighter_b": b["fighter_b"], "event_date": b["date"]} for b in bouts], db_fights)
        new = []
        for b in bouts:
            hit = matches.get(b["external_event_id"])
            if hit:
                stats["bouts_matched"] += 1
            else:
                stats["unmatched"].append(b["name"])
            fid = hit["fight_id"] if hit else None
            for row in b["parse"](bool(hit and hit["swapped"])):
                k = (row["external_market_id"], row["external_selection_id"])
                if last.get(k) == (row["american"], row["line"]):
                    continue   # a row only when the price or line moves
                last[k] = (row["american"], row["line"])
                if not hit:
                    row = {**row, "side": None, "market_key": None}  # corners unknown
                new.append(H(fight_id=fid, book=book, external_event_id=b["external_event_id"],
                             captured_at=now,
                             days_to_fight=(date_of[fid] - today).days if fid in date_of else None, **row))
        stats["rows"] = len(new)
        if not dry_run:
            db.add_all(new)
            db.commit()
    finally:
        db.close()
    return stats


BOOKS = {
    "fanduel": "app.services.ufc.fanduel",
    "bovada": "app.services.ufc.bovada",
    "betrivers": "app.services.ufc.betrivers",
    "pinnacle": "app.services.ufc.pinnacle",
}


def main() -> None:
    import importlib
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--books", default=",".join(BOOKS))
    ap.add_argument("--dry-run", action="store_true", help="fetch and parse, write nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    failed = []
    for name in args.books.split(","):
        try:
            mod = importlib.import_module(BOOKS[name])
            bouts, info = mod.fetch_bouts()
            out = {**record(mod.BOOK, bouts, args.dry_run), **info}
        except Exception as err:  # one book's outage must not cost the others their snapshot
            log.exception("%s failed", name)
            out = {"book": name, "error": f"{type(err).__name__}: {err}"}
            failed.append(name)
        print(json.dumps(out, indent=1, default=str))
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
