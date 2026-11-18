"""Division age curves: win rate (and Round Winning rating change) by age at fight time.

Every completed bout contributes one observation per fighter with a known date of
birth, filed under the BOUT's division, so a division's curve averages 50% overall and
its shape is purely the age effect. Draws count half a win; no-contests are skipped.

Precomputed into ufc_age_curves because it reads every bout and every Glicko snapshot.
Display only — never a model input.

    python -m app.services.ufc.fighter_extras_age --dry-run OUT.json   # prod reads, no writes
    python -m app.services.ufc.fighter_extras_age --write              # rewrites ufc_age_curves

Set FIGHTER_EXTRAS_DB_URL=sqlite:///<path> to point --write at a scratch database.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.ufc import UFCFight, UFCFighter, UFCGlickoSnapshot
from app.models.ufc_fighter_extras import UFCAgeCurve
from app.services.ufc.fighter_extras_common import (
    age_on, completed_filter, division_label, extras_engine, extras_session, fighter_bouts, outcome,
)
from app.services.ufc.fighter_registry import classify_weight_class, current_division

log = logging.getLogger("fighter_extras_age")

MIN_AGE, MAX_AGE = 19, 45
#: Ages with fewer observations than this are left off the served curve.
MIN_BIN_FIGHTS = 30
RESULT_SCORE = {"W": 1.0, "L": 0.0, "D": 0.5}


def build_curves(fights: list[tuple], dobs: dict[int, object], rating_deltas: dict[tuple[int, int], float]) -> list[dict]:
    """fights: (fight_id, date, weight_class, red_id, blue_id, winner_id, method).
    rating_deltas: (fight_id, fighter_id) -> next pre-fight pts minus this one.
    Returns one row per (division, age), "all" included."""
    from types import SimpleNamespace
    acc: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0])  # n, wins, d_sum, d_n
    for fid, day, wc, red, blue, winner, method in fights:
        div = classify_weight_class(wc)
        f = SimpleNamespace(winner_id=winner, method=method)
        for me in (red, blue):
            if me is None:
                continue
            a = age_on(dobs.get(int(me)), day)
            res = outcome(f, int(me))
            if a is None or res not in RESULT_SCORE:
                continue
            age = int(a)
            if not MIN_AGE <= age <= MAX_AGE:
                continue
            delta = rating_deltas.get((int(fid), int(me)))
            for key in ([div] if div != "unknown" else []) + ["all"]:
                cell = acc[(key, age)]
                cell[0] += 1
                cell[1] += RESULT_SCORE[res]
                if delta is not None:
                    cell[2] += delta
                    cell[3] += 1
    rows = []
    for (div, age), (n, w, ds, dn) in acc.items():
        num = den = 0.0
        for off, wt in ((-1, 1), (0, 2), (1, 1)):
            c = acc.get((div, age + off))
            if c:
                num += wt * c[1]
                den += wt * c[0]
        rows.append({
            "division": div, "age": age, "fights": n, "wins": w,
            "win_rate": round(w / n, 4),
            "smoothed": round(num / den, 4) if den else None,
            "rating_delta": round(ds / dn, 3) if dn else None,
        })
    return sorted(rows, key=lambda r: (r["division"], r["age"]))


def _rating_deltas(db: Session) -> dict[tuple[int, int], float]:
    rows = (
        db.query(UFCGlickoSnapshot.fight_id, UFCGlickoSnapshot.fighter_id, UFCGlickoSnapshot.pts, UFCFight.date)
        .join(UFCFight, UFCFight.id == UFCGlickoSnapshot.fight_id)
        .filter(UFCFight.date.isnot(None))
        .order_by(UFCGlickoSnapshot.fighter_id, UFCFight.date, UFCFight.id)
        .all()
    )
    by_fighter: dict[int, list] = defaultdict(list)
    for fid, me, pts, _ in rows:
        if pts:  # 0.0 is the placeholder-round artefact, not a rating
            by_fighter[int(me)].append((int(fid), float(pts)))
    out = {}
    for me, seq in by_fighter.items():
        for (fid, pts), (_, nxt) in zip(seq, seq[1:]):
            out[(fid, me)] = nxt - pts
    return out


def compute(db: Session) -> list[dict]:
    fights = (
        db.query(UFCFight.id, UFCFight.date, UFCFight.weight_class, UFCFight.red_fighter_id,
                 UFCFight.blue_fighter_id, UFCFight.winner_id, UFCFight.method)
        .filter(*completed_filter())
        .all()
    )
    dobs = {int(i): d for i, d in db.query(UFCFighter.id, UFCFighter.dob).filter(UFCFighter.dob.isnot(None)).all()}
    return build_curves([tuple(f) for f in fights], dobs, _rating_deltas(db))


def write(db: Session, rows: list[dict]) -> int:
    """Replace ufc_age_curves in one transaction."""
    from app.database import engine
    with extras_session(db) as xs:
        UFCAgeCurve.__table__.create(bind=extras_engine(engine), checkfirst=True)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        xs.query(UFCAgeCurve).delete()
        for i, r in enumerate(rows, 1):
            xs.add(UFCAgeCurve(id=i, computed_at=now, **r))
        xs.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

def _curve(xs: Session, division: str) -> list[dict]:
    rows = (xs.query(UFCAgeCurve).filter(UFCAgeCurve.division == division,
                                          UFCAgeCurve.fights >= MIN_BIN_FIGHTS)
            .order_by(UFCAgeCurve.age).all())
    return [{"age": r.age, "fights": r.fights, "win_rate": r.win_rate, "smoothed": r.smoothed,
             "rating_delta": r.rating_delta} for r in rows]


def fighter_age_curve(db: Session, fighter_id: int) -> dict:
    fighter = db.query(UFCFighter).filter(UFCFighter.id == fighter_id).first()
    if fighter is None:
        raise HTTPException(status_code=404, detail="Fighter not found")
    bouts = fighter_bouts(db, fighter_id)
    divs = [d for d in (classify_weight_class(f.weight_class) for f, _ in bouts) if d != "unknown"]
    division = current_division(divs) if divs else None
    with extras_session(db) as xs:
        curve = _curve(xs, division) if division else []
        overall = _curve(xs, "all")

    points, by_age = [], defaultdict(lambda: {"fights": 0, "wins": 0, "losses": 0, "draws": 0})
    for f, _ in bouts:
        a = age_on(fighter.dob, f.date)
        res = outcome(f, fighter_id)
        if a is None or res == "NC":
            continue
        opp = int(f.blue_fighter_id if int(f.red_fighter_id) == fighter_id else f.red_fighter_id)
        points.append({"fight_id": str(f.id), "date": f.date.isoformat(), "age": round(a, 2),
                       "result": res, "opponent_id": str(opp)})
        cell = by_age[int(a)]
        cell["fights"] += 1
        cell[{"W": "wins", "L": "losses", "D": "draws"}[res]] += 1
    ages = [{"age": k, **v, "win_rate": round((v["wins"] + 0.5 * v["draws"]) / v["fights"], 4)}
            for k, v in sorted(by_age.items())]
    from datetime import date
    return {
        "fighter_id": str(fighter_id),
        "division": division,
        "division_label": division_label(division),
        "curve": curve,
        "all_curve": overall,
        "current_age": round(age_on(fighter.dob, date.today()), 1) if fighter.dob else None,
        "ages": ages,
        "bouts": points,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", metavar="OUT.json", help="compute from prod reads, write JSON only")
    g.add_argument("--write", action="store_true", help="replace ufc_age_curves")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        rows = compute(db)
        log.info(f"{len(rows)} (division, age) rows over {len({r['division'] for r in rows})} divisions")
        if args.dry_run:
            with open(args.dry_run, "w") as fh:
                json.dump(rows, fh, indent=1)
            log.info(f"dry run: wrote {args.dry_run}, no database writes")
        else:
            log.info(f"wrote {write(db, rows)} rows to ufc_age_curves")
    finally:
        db.close()


if __name__ == "__main__":
    main()
