"""Load and keep current the non-UFC data: Sherdog pro records and BFO historical lines.

Three entry points:

  --load-sherdog DIR   bulk-load a crawl's output (resolved.csv, fighters.jsonl, bouts.csv)
  --load-bfo CSV       load BestFightOdds open/close lines (bfo_scraper output)
  --sync-debutants     nightly: every fighter booked on an upcoming card who has no Sherdog
                       link is looked up (name + DOB / UFC bout dates), their full pro
                       record is stored, then their not-yet-known opponents are fetched so
                       their wins can be weighed by opponent quality. Request-budgeted.

Writes go to settings.DATABASE_URL (production in CI). The scraper itself never reads
app settings; this module is the only bridge between scraped files/pages and the DB.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import select

from app.database import SessionLocal, engine
from app.models.ufc import (
    SherdogBout, SherdogFighter, UFCEvent, UFCFight, UFCFighter, UFCFightOpenClose,
)

log = logging.getLogger("external_records")

LINKED_STATUSES = {"matched_dob", "matched_ufc_dates", "matched_name_only"}
BATCH = 1000


def _insert(table):
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert(table)


def _upsert(db, model, rows: list[dict], keys: list[str], update: bool = True) -> None:
    if not rows:
        return
    table = model.__table__
    for i in range(0, len(rows), BATCH):
        stmt = _insert(table).values(rows[i:i + BATCH])
        if update:
            cols = {c: stmt.excluded[c] for c in rows[0] if c not in keys}
            stmt = stmt.on_conflict_do_update(index_elements=keys, set_=cols)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=keys)
        db.execute(stmt)
    db.commit()


def _d(s) -> date | None:
    if not s:
        return None
    return s if isinstance(s, date) else date.fromisoformat(str(s)[:10])


def _i(s) -> int | None:
    try:
        return int(float(s)) if s not in (None, "") else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------- Sherdog


def fighter_row(p: dict, tier: str | None, link: dict | None) -> dict:
    return {
        "sherdog_id": int(p["sherdog_id"]), "url": p["url"], "name": p["name"][:200],
        "nickname": (p.get("nickname") or None), "birth_date": _d(p.get("birth_date")),
        "nationality": p.get("nationality"), "locality": (p.get("locality") or None),
        "height": p.get("height"), "weight": p.get("weight"),
        "association": (p.get("association") or None), "weight_class": p.get("weight_class"),
        "crawl_tier": tier or p.get("crawl_tier"),
        "ufc_fighter_id": link["ufc_fighter_id"] if link else None,
        "match_status": link["status"] if link else None,
        "fetched_at": datetime.fromisoformat(p["parsed_at"]) if p.get("parsed_at") else None,
    }


def bout_row(b: dict) -> dict:
    opp = _i(b.get("opponent_sherdog_id"))
    ev = _i(b.get("event_sherdog_id"))
    return {
        "fighter_sherdog_id": int(b["fighter_sherdog_id"]),
        "opponent_sherdog_id": opp, "opponent_name": (b.get("opponent_name") or "")[:200],
        "opponent_key": str(opp) if opp else (b.get("opponent_name") or "?").lower()[:220],
        "result": (b.get("result") or "?")[:4], "date": _d(b.get("date")),
        "event_name": (b.get("event_name") or "")[:300], "event_sherdog_id": ev,
        "event_key": str(ev) if ev else (b.get("event_name") or "?")[:320],
        "promotion": (b.get("promotion") or None), "method": (b.get("method") or None),
        "method_detail": (b.get("method_detail") or None),
        "method_class": b.get("method_class") or "OTHER",
        "referee": (b.get("referee") or None), "round": _i(b.get("round")),
        "time": (b.get("time") or None),
    }


def _store_profiles(db, profiles: list[dict], bouts: list[dict], links: dict[int, dict]) -> None:
    # A Sherdog id may only link to one UFC fighter (unique); first resolution wins.
    seen_ufc = set()
    rows = []
    for p in profiles:
        link = links.get(int(p["sherdog_id"]))
        if link and link["ufc_fighter_id"] in seen_ufc:
            link = None
        if link:
            seen_ufc.add(link["ufc_fighter_id"])
        rows.append(fighter_row(p, p.get("crawl_tier"), link))
    _upsert(db, SherdogFighter, rows, ["sherdog_id"])
    brows = [bout_row(b) for b in bouts if b.get("section", "pro") == "pro"]
    # A page can list the same opponent twice on one card (tournaments); keep one.
    uniq = {(r["fighter_sherdog_id"], r["opponent_key"], r["date"], r["event_key"]): r for r in brows}
    _upsert(db, SherdogBout, list(uniq.values()),
            ["fighter_sherdog_id", "opponent_key", "date", "event_key"])


def load_sherdog_dir(path: Path) -> None:
    links = {}
    with open(path / "resolved.csv") as fh:
        for r in csv.DictReader(fh):
            if r.get("sherdog_id") and r["status"] in LINKED_STATUSES:
                links[int(r["sherdog_id"])] = {"ufc_fighter_id": int(r["ufc_fighter_id"]),
                                               "status": r["status"]}
    profiles = [json.loads(line) for line in open(path / "fighters.jsonl") if line.strip()]
    with open(path / "bouts.csv") as fh:
        bouts = list(csv.DictReader(fh))
    db = SessionLocal()
    try:
        _store_profiles(db, profiles, bouts, links)
    finally:
        db.close()
    log.info(f"Loaded {len(profiles)} Sherdog profiles ({len(links)} linked to UFC fighters), "
             f"{len(bouts)} bout rows")


def sync_debutants(max_requests: int = 300, include_unbooked: bool = False) -> dict:
    """Look up every booked-but-unlinked fighter on Sherdog, then their opponents."""
    import dataclasses

    from app.services.ufc import sherdog_scraper as sd

    db = SessionLocal()
    try:
        linked = {r for (r,) in db.execute(
            select(SherdogFighter.ufc_fighter_id).where(SherdogFighter.ufc_fighter_id.isnot(None)))}
        known = {r for (r,) in db.execute(select(SherdogFighter.sherdog_id))}
        today = date.today()
        upcoming = (db.query(UFCFight).join(UFCEvent, UFCEvent.id == UFCFight.event_id)
                    .filter(UFCEvent.date >= today, UFCFight.winner_id.is_(None)).all())
        booked = {fid for f in upcoming for fid in (f.red_fighter_id, f.blue_fighter_id)}
        wanted = booked - linked
        if include_unbooked:
            wanted |= {f.id for f in db.query(UFCFighter.id)} - linked
        fighters = {f.id: f for f in db.query(UFCFighter).filter(UFCFighter.id.in_(wanted))}
        dates: dict[int, list[str]] = {}
        for f in db.query(UFCFight).filter((UFCFight.red_fighter_id.in_(wanted))
                                           | (UFCFight.blue_fighter_id.in_(wanted))):
            for fid in (f.red_fighter_id, f.blue_fighter_id):
                if fid in wanted and f.date:
                    dates.setdefault(fid, []).append(f.date.isoformat())

        # Booked fighters first (the next card needs them), soonest bout first.
        order = sorted(wanted, key=lambda i: (i not in booked, min(dates.get(i, ["9999"]))))
        fetcher = sd.PoliteFetcher(max_requests=max_requests)
        stats = {"wanted": len(wanted), "linked": 0, "unresolved": [], "opponents": 0}
        profiles, bouts, links = [], [], {}
        try:
            for fid in order:
                f = fighters.get(fid)
                if f is None:
                    continue
                ours = sd.OurFighter(id=f.id, ufcstats_id=f.ufcstats_id,
                                     first_name=f.first_name or "", last_name=f.last_name or "",
                                     nickname=f.nickname, dob=f.dob.isoformat() if f.dob else None,
                                     ufc_fight_dates=sorted(dates.get(fid, [])))
                res, parsed = sd.resolve_fighter(ours, fetcher)
                if res.status not in LINKED_STATUSES:
                    stats["unresolved"].append(f"{ours.full_name}: {res.status}")
                    continue
                for prof, pb in parsed:
                    if prof.sherdog_id == res.sherdog_id:
                        profiles.append({**dataclasses.asdict(prof), "crawl_tier": "ufc"})
                        bouts += [dataclasses.asdict(b) for b in pb]
                        links[prof.sherdog_id] = {"ufc_fighter_id": f.id, "status": res.status}
                        known.add(prof.sherdog_id)
                        stats["linked"] += 1
            # One hop: opponents of the fighters just added, most-shared first.
            opp_refs: dict[int, tuple[int, str]] = {}
            for b in bouts:
                oid = b.get("opponent_sherdog_id")
                if oid and oid not in known and b.get("opponent_url"):
                    n, url = opp_refs.get(oid, (0, b["opponent_url"]))
                    opp_refs[oid] = (n + 1, url)
            for oid, (_, url) in sorted(opp_refs.items(), key=lambda kv: -kv[1][0]):
                prof, pb = sd.parse_fighter_page(fetcher.get(url), url)
                if prof:
                    profiles.append({**dataclasses.asdict(prof), "crawl_tier": "opponent_1hop"})
                    bouts += [dataclasses.asdict(b) for b in pb]
                    known.add(oid)
                    stats["opponents"] += 1
        except sd.BudgetExhausted:
            log.info("Request budget used; the rest carries over to the next run.")
        except (sd.SherdogBlocked, sd.RobotsDisallowed) as e:
            log.error(f"Stopped: {e}")
        finally:
            _store_profiles(db, profiles, bouts, links)
        log.info(f"Debutant sync: {stats['linked']}/{stats['wanted']} linked, "
                 f"{stats['opponents']} opponents fetched, {fetcher.requests_made} requests")
        for u in stats["unresolved"]:
            log.warning(f"  needs manual review: {u}")
        return stats
    finally:
        db.close()


# ------------------------------------------------------------------------ BestFightOdds


def load_bfo_csv(path: Path, source: str = "bfo") -> int:
    """Load bfo_scraper output. Rows are in BFO's fighter A/B order; `db_swapped` says
    whether A is our blue corner."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if not r.get("db_fight_id") or r.get("is_exchange") == "True":
                continue
            swap = r.get("db_swapped") == "True"

            def side(a, b):
                return (b, a) if swap else (a, b)

            ro, bo = side(_i(r["open_a"]), _i(r["open_b"]))
            rc, bc = side(_i(r["close_a"]), _i(r["close_b"]))
            op = float(r["open_prob_a"]) if r.get("open_prob_a") else None
            cp = float(r["close_prob_a"]) if r.get("close_prob_a") else None
            if swap:
                op = 1 - op if op is not None else None
                cp = 1 - cp if cp is not None else None
            ts = lambda s: datetime.fromisoformat(s.replace("Z", "")) if s else None  # noqa: E731
            rows.append({
                "fight_id": int(r["db_fight_id"]), "source": source,
                "bookmaker": r["bookmaker"][:100],
                "red_open": ro, "blue_open": bo, "red_close": rc, "blue_close": bc,
                "red_open_prob": op, "red_close_prob": cp,
                "opened_at": ts(r.get("open_ts_a")), "closed_at": ts(r.get("close_ts_a")),
                "close_source": (r.get("close_source") or None),
                "flags": (r.get("flags") or None),
            })
    uniq = {(x["fight_id"], x["source"], x["bookmaker"]): x for x in rows}
    db = SessionLocal()
    try:
        # Matching may have run against a different copy of the fights table (the local
        # dev DB keeps some bouts production has since removed as cancelled). Only load
        # rows whose fight exists here.
        present = {i for (i,) in db.execute(select(UFCFight.id))}
        keep = [x for x in uniq.values() if x["fight_id"] in present]
        skipped = len(uniq) - len(keep)
        _upsert(db, UFCFightOpenClose, keep, ["fight_id", "source", "bookmaker"])
    finally:
        db.close()
    log.info(f"Loaded {len(keep)} open/close rows from {path}"
             + (f" ({skipped} skipped: fight not in this database)" if skipped else ""))
    return len(keep)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-sherdog", type=Path)
    ap.add_argument("--load-bfo", type=Path)
    ap.add_argument("--sync-debutants", action="store_true")
    ap.add_argument("--include-unbooked", action="store_true",
                    help="also backfill unlinked fighters with no upcoming bout")
    ap.add_argument("--max-requests", type=int, default=300)
    a = ap.parse_args()
    # The nightly job can run before the API has redeployed and run create_all, so make
    # sure these three (new) tables exist. checkfirst makes it a no-op afterwards.
    for model in (SherdogFighter, SherdogBout, UFCFightOpenClose):
        model.__table__.create(bind=engine, checkfirst=True)
    if a.load_sherdog:
        load_sherdog_dir(a.load_sherdog)
    if a.load_bfo:
        load_bfo_csv(a.load_bfo)
    if a.sync_debutants:
        sync_debutants(a.max_requests, a.include_unbooked)
