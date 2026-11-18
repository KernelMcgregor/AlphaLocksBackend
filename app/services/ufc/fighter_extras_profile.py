"""Per-request fighter-page extras: divisional percentiles, Glicko rating history and
situational splits. Display only (see fighter_extras_common).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.ufc import (
    SherdogBout, SherdogFighter, UFCCancelledBout, UFCEvent, UFCFight, UFCFightOpenClose,
    UFCFightStats, UFCFighter, UFCFighterCareerStats, UFCGlickoSnapshot,
)
from app.services.ufc.fight_context_service import GLICKO_DIMS, GLICKO_LABELS
from app.services.ufc.fighter_extras_common import (
    division_label, division_map, fighter_bouts, is_finish, outcome,
)

# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------

#: (column, label, group, higher is better, display format). Formats: num (2dp),
#: signed (2dp with sign), pct (0-1 shown as %), clock (seconds).
PERCENTILE_STATS = [
    ("slpm", "Sig. strikes landed / min", "striking", True, "num"),
    ("sapm", "Sig. strikes absorbed / min", "striking", False, "num"),
    ("sl_diff", "Striking differential / min", "striking", True, "signed"),
    ("sig_acc", "Sig. strike accuracy", "striking", True, "pct"),
    ("sig_def", "Sig. strike defense", "striking", True, "pct"),
    ("kd15", "Knockdowns / 15", "striking", True, "num"),
    ("kd_abs15", "Knocked down / 15", "striking", False, "num"),
    ("td15", "Takedowns / 15", "grappling", True, "num"),
    ("td_acc", "Takedown accuracy", "grappling", True, "pct"),
    ("td_def", "Takedown defense", "grappling", True, "pct"),
    ("ctrl15", "Control time / 15", "grappling", True, "clock"),
    ("sub_att15", "Submission attempts / 15", "grappling", True, "num"),
    ("finish_rate", "Finish rate", "outcomes", True, "pct"),
    ("win_pct", "Win rate", "outcomes", True, "pct"),
]
GROUP_LABELS = {"striking": "Striking", "grappling": "Grappling", "outcomes": "Outcomes"}
#: A fighter needs this many UFC bouts to be part of a division's reference pool.
MIN_POOL_FIGHTS = 3


def percentile_pools(db: Session) -> dict[str, dict[str, list[float]]]:
    """division -> stat -> sorted values over the division's qualifying fighters."""
    divisions = division_map(db)
    cols = [getattr(UFCFighterCareerStats, k) for k, *_ in PERCENTILE_STATS]
    rows = (
        db.query(UFCFighterCareerStats.fighter_id, *cols)
        .filter(UFCFighterCareerStats.fight_count >= MIN_POOL_FIGHTS)
        .all()
    )
    pools: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fid, *vals in rows:
        div = divisions.get(int(fid))
        if not div or div == "unknown":
            continue
        for (key, *_), v in zip(PERCENTILE_STATS, vals):
            if v is not None:
                pools[div][key].append(float(v))
    return {d: {k: sorted(v) for k, v in s.items()} for d, s in pools.items()}


def percentile_of(sorted_vals: list[float], value: float, higher_better: bool = True) -> int | None:
    """Mid-rank percentile, 0-100: share of the pool below `value`, ties counted half.
    Flipped for stats where less is better, so 100 always means best in division."""
    from bisect import bisect_left, bisect_right
    n = len(sorted_vals)
    if not n:
        return None
    lo, hi = bisect_left(sorted_vals, value), bisect_right(sorted_vals, value)
    p = 100.0 * (lo + 0.5 * (hi - lo)) / n
    return int(round(p if higher_better else 100.0 - p))


def fighter_percentiles(db: Session, fighter_id: int, pools: dict) -> dict:
    cs = db.query(UFCFighterCareerStats).filter(UFCFighterCareerStats.fighter_id == fighter_id).first()
    if cs is None:
        raise HTTPException(status_code=404, detail="No career stats for this fighter")
    div = division_map_one(db, fighter_id)
    pool = pools.get(div or "", {})
    groups: dict[str, list] = defaultdict(list)
    for key, label, group, hib, fmt in PERCENTILE_STATS:
        v = getattr(cs, key)
        vals = pool.get(key, [])
        groups[group].append({
            "key": key, "label": label, "format": fmt, "higher_better": hib,
            "value": None if v is None else round(float(v), 4),
            "percentile": None if v is None else percentile_of(vals, float(v), hib),
            "median": round(vals[len(vals) // 2], 4) if vals else None,
        })
    return {
        "fighter_id": str(fighter_id),
        "division": div,
        "division_label": division_label(div),
        "pool_size": max((len(v) for v in pool.values()), default=0),
        "min_fights": MIN_POOL_FIGHTS,
        "fight_count": cs.fight_count,
        "groups": [{"key": g, "label": GROUP_LABELS[g], "stats": groups[g]} for g in GROUP_LABELS],
    }


def division_map_one(db: Session, fighter_id: int) -> str | None:
    """Current division for one fighter, by the same rule as division_map."""
    from app.services.ufc.fighter_registry import classify_weight_class, current_division
    divs = [classify_weight_class(f.weight_class) for f, _ in fighter_bouts(db, fighter_id)]
    divs = [d for d in divs if d != "unknown"]
    return current_division(divs) if divs else None


# ---------------------------------------------------------------------------
# Rating history
# ---------------------------------------------------------------------------

def _is_placeholder(snap) -> bool:
    """Rows written from zero placeholder rounds carry 0.0 on every dimension."""
    vals = [getattr(snap, d) for d in GLICKO_DIMS]
    return all(v is None or v == 0 for v in vals)


def rating_history(db: Session, fighter_id: int) -> dict:
    """Pre-fight Glicko ratings per bout, plus the change each bout produced (the next
    bout's pre-fight rating minus this one's). The latest bout has no successor, so its
    change is null until the fighter fights again."""
    if db.query(UFCFighter.id).filter(UFCFighter.id == fighter_id).first() is None:
        raise HTTPException(status_code=404, detail="Fighter not found")
    rows = (
        db.query(UFCGlickoSnapshot, UFCFight, UFCEvent.name)
        .join(UFCFight, UFCFight.id == UFCGlickoSnapshot.fight_id)
        .outerjoin(UFCEvent, UFCEvent.id == UFCFight.event_id)
        .filter(UFCGlickoSnapshot.fighter_id == fighter_id, UFCFight.date.isnot(None))
        .order_by(UFCFight.date, UFCFight.id)
        .all()
    )
    rows = [r for r in rows if not _is_placeholder(r[0])]
    opp_ids = {int(f.blue_fighter_id if int(f.red_fighter_id) == fighter_id else f.red_fighter_id) for _, f, _ in rows}
    names = {int(i): f"{a} {b}" for i, a, b in db.query(UFCFighter.id, UFCFighter.first_name, UFCFighter.last_name)
             .filter(UFCFighter.id.in_(opp_ids)).all()} if opp_ids else {}
    opp_snaps = {}
    if rows:
        for s in (db.query(UFCGlickoSnapshot)
                  .filter(UFCGlickoSnapshot.fight_id.in_([f.id for _, f, _ in rows]),
                          UFCGlickoSnapshot.fighter_id != fighter_id).all()):
            if not _is_placeholder(s):
                opp_snaps[int(s.fight_id)] = s

    def dims(s) -> dict:
        return {d: None if getattr(s, d) is None else round(float(getattr(s, d)), 2) for d in GLICKO_DIMS}

    out = []
    for i, (snap, f, event_name) in enumerate(rows):
        opp = int(f.blue_fighter_id if int(f.red_fighter_id) == fighter_id else f.red_fighter_id)
        pre = dims(snap)
        nxt = dims(rows[i + 1][0]) if i + 1 < len(rows) else None
        o = opp_snaps.get(int(f.id))
        out.append({
            "fight_id": str(f.id),
            "date": f.date.isoformat(),
            "event": event_name,
            "opponent_id": str(opp),
            "opponent_name": names.get(opp),
            "result": outcome(f, fighter_id),
            "method": f.method,
            "pre": pre,
            "post": nxt,
            "delta": None if nxt is None else {
                d: None if pre[d] is None or nxt[d] is None else round(nxt[d] - pre[d], 2) for d in GLICKO_DIMS},
            "opp_pre": dims(o) if o is not None else None,
            "sigma": None if snap.meta_sigma is None else round(float(snap.meta_sigma), 1),
        })
    return {
        "fighter_id": str(fighter_id),
        "dims": [{"key": d, "label": GLICKO_LABELS[d]} for d in GLICKO_DIMS],
        "fights": out,
    }


# ---------------------------------------------------------------------------
# Situational splits
# ---------------------------------------------------------------------------

LAYOFF_DAYS = 365
SHORT_NOTICE_WINDOW = timedelta(days=10)  # same window short_notice.py uses
HIGH_ALTITUDE_M = 1000


def _rounds(f) -> int | None:
    tf = (f.time_format or "").strip()
    parts = tf.split("-")
    if parts and all(p.strip() == "5" for p in parts) and tf:
        return len(parts)
    if f.max_fight_time_seconds == 1500:
        return 5
    if f.max_fight_time_seconds == 900:
        return 3
    return None


def _altitude_m(location: str | None) -> int:
    from app.services.ufc.round_features import ALTITUDE
    loc = (location or "").lower()
    return next((m for k, m in ALTITUDE.items() if k in loc), 0)


def _replacement_flags(db: Session, fighter_id: int, bouts: list) -> dict[int, bool]:
    """fight id -> True when this fighter stepped in on short notice: their opponent had a
    different bout pulled from the same card and they themselves did not."""
    from app.services.ufc.short_notice import _had_pulled, _pulled_by_fighter
    opp_ids = {int(f.blue_fighter_id if int(f.red_fighter_id) == fighter_id else f.red_fighter_id) for f, _ in bouts}
    ids = list(opp_ids | {fighter_id})
    rows = (
        db.query(UFCCancelledBout.event_date, UFCCancelledBout.red_fighter_id, UFCCancelledBout.blue_fighter_id)
        .filter(or_(UFCCancelledBout.red_fighter_id.in_(ids), UFCCancelledBout.blue_fighter_id.in_(ids)))
        .all()
    )
    pulled = _pulled_by_fighter(rows)
    out = {}
    for f, _ in bouts:
        opp = int(f.blue_fighter_id if int(f.red_fighter_id) == fighter_id else f.red_fighter_id)
        me_changed = _had_pulled(pulled, fighter_id, f.date, opp)
        opp_changed = _had_pulled(pulled, opp, f.date, fighter_id)
        out[int(f.id)] = opp_changed and not me_changed
    return out


def _other_pro_dates(db: Session, fighter_id: int) -> list[date]:
    """Dates of the fighter's non-UFC pro bouts (Sherdog), so a spell in another
    promotion is not mistaken for a layoff."""
    sid = db.query(SherdogFighter.sherdog_id).filter(SherdogFighter.ufc_fighter_id == fighter_id).scalar()
    if sid is None:
        return []
    rows = (db.query(SherdogBout.date, SherdogBout.promotion)
            .filter(SherdogBout.fighter_sherdog_id == sid, SherdogBout.date.isnot(None)).all())
    return sorted(d for d, p in rows if (p or "").upper() != "UFC")


def _close_probs(db: Session, fight_ids: list[int]) -> dict[int, float]:
    """fight id -> consensus vig-free closing probability of the RED corner."""
    if not fight_ids:
        return {}
    rows = (db.query(UFCFightOpenClose.fight_id, UFCFightOpenClose.red_close_prob)
            .filter(UFCFightOpenClose.fight_id.in_(fight_ids), UFCFightOpenClose.bookmaker == "Consensus",
                    UFCFightOpenClose.red_close_prob.isnot(None)).all())
    return {int(f): float(p) for f, p in rows}


def _totals(db: Session, fight_ids: list[int]) -> dict[tuple[int, int], tuple]:
    """(fight, fighter) -> (sig landed, sig attempted, fight seconds) from totals rows."""
    if not fight_ids:
        return {}
    rows = (db.query(UFCFightStats.fight_id, UFCFightStats.fighter_id, UFCFightStats.sig_str_landed,
                     UFCFightStats.sig_str_attempted)
            .filter(UFCFightStats.fight_id.in_(fight_ids), UFCFightStats.round_number == 0).all())
    return {(int(f), int(p)): (l or 0, a or 0) for f, p, l, a in rows}


def bout_tags(db: Session, fighter_id: int) -> list[dict]:
    """One dict per completed bout: outcome, finish, the striking totals, and every
    situational flag the splits group on (None where the data cannot say)."""
    bouts = fighter_bouts(db, fighter_id)
    ids = [int(f.id) for f, _ in bouts]
    sn = _replacement_flags(db, fighter_id, bouts)
    probs = _close_probs(db, ids)
    totals = _totals(db, ids)
    other = _other_pro_dates(db, fighter_id)
    out, prev = [], None
    for f, location in bouts:
        red = int(f.red_fighter_id) == fighter_id
        opp = int(f.blue_fighter_id if red else f.red_fighter_id)
        # The most recent pro bout before this one, UFC or not.
        last = max([d for d in other if d < f.date] + ([prev] if prev else []), default=None)
        p = probs.get(int(f.id))
        mine = p if red or p is None else 1.0 - p
        rounds = _rounds(f)
        me, them = totals.get((int(f.id), fighter_id)), totals.get((int(f.id), opp))
        res = outcome(f, fighter_id)
        out.append({
            "fight_id": int(f.id), "date": f.date, "result": res,
            "finish_win": res == "W" and is_finish(f.method),
            "secs": f.fight_time_seconds,
            "me": me, "them": them,
            "short_notice": sn.get(int(f.id), False),
            "layoff": None if last is None else (f.date - last).days >= LAYOFF_DAYS,
            "favorite": None if mine is None or abs(mine - 0.5) < 1e-9 else mine > 0.5,
            "five_rounds": None if rounds is None else rounds >= 5,
            "altitude": _altitude_m(location) >= HIGH_ALTITUDE_M,
        })
        prev = f.date
    return out


def summarise(tags: list[dict]) -> dict:
    """Record and per-bout performance for one group of bouts."""
    n = {k: sum(1 for t in tags if t["result"] == k) for k in ("W", "L", "D", "NC")}
    decided = n["W"] + n["L"]
    sl = sa = ol = oa = secs = 0
    for t in tags:
        if t["me"] and t["them"] and t["secs"]:
            sl += t["me"][0]; sa += t["me"][1]; ol += t["them"][0]; oa += t["them"][1]
            secs += t["secs"]
    return {
        "fights": len(tags), "wins": n["W"], "losses": n["L"], "draws": n["D"], "nc": n["NC"],
        "win_pct": round(100 * n["W"] / decided, 1) if decided else None,
        "finish_pct": round(100 * sum(t["finish_win"] for t in tags) / decided, 1) if decided else None,
        "sig_acc": round(100 * sl / sa, 1) if sa else None,
        "sig_def": round(100 * (1 - ol / oa), 1) if oa else None,
        "diff_pm": round((sl - ol) / (secs / 60), 2) if secs else None,
    }


#: (key, label, flag, label when true, label when false). Each split's "true" side is
#: the situation; the other side is the fighter's baseline.
SPLITS = [
    ("short_notice", "Short notice", "short_notice", "Short notice", "Full camp"),
    ("layoff", "Layoff", "layoff", "After 1+ yr out", "Active"),
    ("odds", "Odds", "favorite", "Underdog", "Favorite"),
    ("rounds", "Scheduled rounds", "five_rounds", "5 rounds", "3 rounds"),
    ("altitude", "Altitude", "altitude", "Altitude", "Sea level"),
]


def fighter_splits(db: Session, fighter_id: int) -> dict:
    if db.query(UFCFighter.id).filter(UFCFighter.id == fighter_id).first() is None:
        raise HTTPException(status_code=404, detail="Fighter not found")
    tags = bout_tags(db, fighter_id)
    splits = []
    for key, label, flag, a_label, b_label in SPLITS:
        # Odds read the other way round: underdog is the situation, favorite the baseline.
        want = False if flag == "favorite" else True
        a = [t for t in tags if t[flag] is want]
        b = [t for t in tags if t[flag] is (not want)]
        splits.append({
            "key": key, "label": label,
            "a": {"label": a_label, **summarise(a)},
            "b": {"label": b_label, **summarise(b)},
        })
    return {"fighter_id": str(fighter_id), "overall": summarise(tags), "splits": splits}
