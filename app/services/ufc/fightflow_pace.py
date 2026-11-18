"""Pace profile: significant strikes landed per minute, round by round, against the
division's norm for that round.

The norm is per division AND per round because everyone slows down: a fighter whose
output drops from 5.0 in R1 to 4.4 in R3 is fading only if the division drops less.
So a fighter's rounds are indexed against what an average fighter in the same
division lands over the same minutes (expected = norm rate x minutes fought), and the
label compares the late-round index (R2 on) with the R1 index:

  fader   late index at least 15% below R1's
  closer  at least 15% above
  steady  in between

Both indexes are shrunk toward 1 with PSEUDO strikes, so a two-fight career does not
get a label from a single busy first round. Too few minutes either side -> no label.

Minutes come from the bout's time_format and fight_time_seconds (the same rule as the
frontend's roundDurations): every round before the finish runs its scheduled length,
the finishing round whatever is left. Rows past finish_round are UFCStats padding.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.ufc import UFCFight, UFCFightStats
from app.services.ufc.fighter_registry import classify_weight_class

#: Norms are pooled over the modern era (unified rules, five-minute rounds).
NORM_ERA = dt.date(2010, 1, 1)
MAX_ROUND = 5
#: Pseudo-strikes each index is shrunk with (toward the division norm).
PSEUDO = 30.0
#: Minutes needed on each side of the comparison before a label is given.
MIN_MINUTES = 15.0
THRESHOLD = math.log(1.15)
#: Fighter-minutes behind a division-round norm (a women's featherweight R5 has none).
NORM_MIN_MINUTES = 30.0


def round_seconds(time_format: str | None, finish_round: int | None,
                  fight_time_seconds: int | None) -> list[int]:
    """Seconds fought in each round, 1..finish_round. Rounds the format does not cover
    (or no parseable format) default to five minutes; a finishing round whose remainder
    is non-positive or longer than scheduled keeps its scheduled length."""
    n = finish_round or 0
    if n <= 0:
        return []
    lens = []
    for part in (time_format or "").split("-"):
        try:
            lens.append(int(part) * 60)
        except ValueError:
            break
    secs = [lens[i] if i < len(lens) else 300 for i in range(n)]
    total = fight_time_seconds or 0
    last = total - sum(secs[:-1])
    if total > 0 and 0 < last <= secs[-1] + 1:
        secs[-1] = last
    return secs


def _stat_rows(db: Session, *filters):
    """(fight, corner-row) pairs for rounds 1..finish_round."""
    q = (db.query(UFCFight.id, UFCFight.date, UFCFight.weight_class, UFCFight.time_format,
                  UFCFight.finish_round, UFCFight.fight_time_seconds,
                  UFCFightStats.fighter_id, UFCFightStats.round_number,
                  UFCFightStats.sig_str_landed)
         .join(UFCFightStats, UFCFightStats.fight_id == UFCFight.id)
         .filter(UFCFightStats.round_number >= 1, UFCFight.finish_round.isnot(None),
                 UFCFightStats.round_number <= UFCFight.finish_round, *filters))
    secs_by_fight: dict[int, list[int]] = {}
    for fid, date, wc, tf, fr, fts, fighter_id, rnd, landed in q:
        if fid not in secs_by_fight:
            secs_by_fight[fid] = round_seconds(tf, fr, fts)
        secs = secs_by_fight[fid]
        if rnd > len(secs) or secs[rnd - 1] <= 0:
            continue
        yield {"fight_id": fid, "date": date, "division": classify_weight_class(wc),
               "fighter_id": fighter_id, "round": rnd, "minutes": secs[rnd - 1] / 60,
               "landed": landed or 0}


def division_norms(db: Session) -> dict[str, dict[int, float]]:
    """division -> round -> sig. strikes landed per fighter-minute, since NORM_ERA.
    A round past what the division has data for falls back to the all-division rate."""
    landed = defaultdict(float)
    minutes = defaultdict(float)
    for r in _stat_rows(db, UFCFight.date >= NORM_ERA):
        rnd = min(r["round"], MAX_ROUND)
        for div in (r["division"], "all"):
            landed[(div, rnd)] += r["landed"]
            minutes[(div, rnd)] += r["minutes"]
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for (div, rnd), m in minutes.items():
        if m >= NORM_MIN_MINUTES:
            out[div][rnd] = landed[(div, rnd)] / m
    return dict(out)


def norm_rate(norms: dict, division: str, rnd: int) -> float | None:
    rnd = min(rnd, MAX_ROUND)
    return norms.get(division, {}).get(rnd) or norms.get("all", {}).get(rnd)


def label_for(early: dict, late: dict) -> tuple[str | None, float | None]:
    """(label, log late/early index ratio) from {landed, expected, minutes} sums."""
    if early["minutes"] < MIN_MINUTES or late["minutes"] < MIN_MINUTES:
        return None, None
    ie = (early["landed"] + PSEUDO) / (early["expected"] + PSEUDO)
    il = (late["landed"] + PSEUDO) / (late["expected"] + PSEUDO)
    d = math.log(il / ie)
    if d <= -THRESHOLD:
        return "fader", d
    if d >= THRESHOLD:
        return "closer", d
    return "steady", d


def profile(rows: list[dict], norms: dict) -> dict:
    """A fighter's per-round career rates indexed against the division norm, plus the
    fader/closer/steady label. `rows` are that fighter's _stat_rows."""
    by_round: dict[int, dict] = {}
    early = {"landed": 0.0, "expected": 0.0, "minutes": 0.0}
    late = {"landed": 0.0, "expected": 0.0, "minutes": 0.0}
    fights = set()
    for r in rows:
        rate = norm_rate(norms, r["division"], r["round"])
        if rate is None:
            continue
        fights.add(r["fight_id"])
        exp = rate * r["minutes"]
        rnd = min(r["round"], MAX_ROUND)
        b = by_round.setdefault(rnd, {"landed": 0.0, "expected": 0.0, "minutes": 0.0, "n": 0})
        b["landed"] += r["landed"]
        b["expected"] += exp
        b["minutes"] += r["minutes"]
        b["n"] += 1
        side = early if r["round"] == 1 else late
        side["landed"] += r["landed"]
        side["expected"] += exp
        side["minutes"] += r["minutes"]
    label, shift = label_for(early, late)
    return {
        "rounds": [{
            "round": rnd,
            "slpm": b["landed"] / b["minutes"] if b["minutes"] else None,
            # Rate an average fighter in the same divisions lands over the same minutes.
            "norm": b["expected"] / b["minutes"] if b["minutes"] else None,
            "index": b["landed"] / b["expected"] if b["expected"] else None,
            "minutes": round(b["minutes"], 1),
            "n": b["n"],
        } for rnd, b in sorted(by_round.items())],
        "label": label,
        "shift": shift,
        "fights": len(fights),
    }


def fight_pace_payload(db: Session, fight_id: int, norms: dict) -> dict | None:
    fight = db.get(UFCFight, fight_id)
    if fight is None:
        return None
    division = classify_weight_class(fight.weight_class)
    scheduled = len([p for p in (fight.time_format or "").split("-") if p.isdigit()]) or 3
    ids = {"red": fight.red_fighter_id, "blue": fight.blue_fighter_id}

    # Both corners' careers through this fight's date, in one query.
    career_rows = defaultdict(list)
    filters = [UFCFightStats.fighter_id.in_(list(ids.values()))]
    if fight.date is not None:
        filters.append(or_(UFCFight.date <= fight.date, UFCFight.id == fight.id))
    for r in _stat_rows(db, *filters):
        career_rows[r["fighter_id"]].append(r)

    out = {
        "fight_id": str(fight.id),
        "division": None if division == "unknown" else division,
        "scheduled_rounds": scheduled,
        "norm": [{"round": r, "slpm": norm_rate(norms, division, r)}
                 for r in range(1, scheduled + 1)],
    }
    for corner, fid in ids.items():
        rows = career_rows.get(fid, [])
        this = [r for r in rows if r["fight_id"] == fight.id]
        out[corner] = {
            "fight": [{
                "round": r["round"],
                "slpm": r["landed"] / r["minutes"],
                "norm": norm_rate(norms, division, r["round"]),
                "minutes": round(r["minutes"], 2),
            } for r in sorted(this, key=lambda r: r["round"])],
            "career": profile(rows, norms),
        }
    return out
