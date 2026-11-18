"""Divisional rankings over time, from ufc_ranking_history, for the time-lapse page.

The history holds a full table per division per event date (~1,000 dates a division).
The page steps through it, so the payload is thinned to one table per period (the last
publish in each month, quarter or year) and cut to the top N.
"""
from __future__ import annotations

from collections import defaultdict

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.ufc import UFCFighter, UFCRankingHistory
from app.services.ufc.fighter_extras_common import division_label

DIVISION_ORDER = [
    "flyweight", "bantamweight", "featherweight", "lightweight", "welterweight", "middleweight",
    "light_heavyweight", "heavyweight", "w_strawweight", "w_flyweight", "w_bantamweight",
]
STEPS = ("event", "month", "quarter", "year")


def _period(d, step: str):
    if step == "event":
        return d
    if step == "month":
        return (d.year, d.month)
    if step == "quarter":
        return (d.year, (d.month - 1) // 3)
    return d.year


def thin_dates(dates: list, step: str) -> list:
    """The last date in each period, ascending."""
    last = {}
    for d in sorted(dates):
        last[_period(d, step)] = d
    return sorted(last.values())


def timelapse(db: Session, division: str, step: str = "quarter", top: int = 15) -> dict:
    if division not in DIVISION_ORDER:
        raise HTTPException(status_code=404, detail="Unknown division")
    if step not in STEPS:
        raise HTTPException(status_code=422, detail=f"step must be one of {', '.join(STEPS)}")
    all_dates = [d for (d,) in db.query(UFCRankingHistory.as_of).filter(
        UFCRankingHistory.weight_class == division).distinct().all()]
    dates = thin_dates(all_dates, step)
    frames = []
    ids: set[int] = set()
    if dates:
        rows = (
            db.query(UFCRankingHistory.as_of, UFCRankingHistory.fighter_id, UFCRankingHistory.rank,
                     UFCRankingHistory.score, UFCRankingHistory.total_ranked)
            .filter(UFCRankingHistory.weight_class == division, UFCRankingHistory.rank <= top,
                    UFCRankingHistory.rank > 0, UFCRankingHistory.as_of.in_(dates))
            .order_by(UFCRankingHistory.as_of, UFCRankingHistory.rank)
            .all()
        )
        by_date = defaultdict(list)
        total = {}
        for as_of, fid, rank, score, n in rows:
            by_date[as_of].append({"id": str(fid), "rank": rank, "score": round(float(score), 1)})
            total[as_of] = n
            ids.add(int(fid))
        frames = [{"as_of": d.isoformat(), "total_ranked": total.get(d), "ranks": by_date[d]}
                  for d in dates if by_date[d]]
    fighters = {}
    if ids:
        for fid, first, last, cc in db.query(UFCFighter.id, UFCFighter.first_name, UFCFighter.last_name,
                                             UFCFighter.country_code).filter(UFCFighter.id.in_(ids)).all():
            fighters[str(fid)] = {"name": f"{first} {last}".strip(), "last_name": last, "country_code": cc}
    return {
        "division": division,
        "division_label": division_label(division),
        "step": step,
        "top": top,
        "divisions": [{"key": k, "label": division_label(k)} for k in DIVISION_ORDER],
        "fighters": fighters,
        "frames": frames,
    }

