"""/ufc endpoints for the best-of track: most exciting fights and cards, and the bonuses
(included into app.routers.ufc.router). Robberies live with the officials endpoints."""
from __future__ import annotations

import json
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.ufc_highlights import UFCFightExcitement
from app.services import response_cache
from app.services.response_cache import with_session
from app.services.ufc.fight_excitement import MODEL_VERSION, card_score
from app.services.ufc.time_window import parse_window

router = APIRouter()
TTL = 30 * 60
#: Bouts this far down the card (0 = main event) or higher are the main card.
MAIN_CARD = 4

_SQL = """
select x.fight_id, x.event_id, x.date, x.score, x.pct, x.components, x.p_fotn, x.p_perf,
       x.fotn, x.perf, ev.name event_name, f.card_position, f.weight_class, f.method,
       f.finish_round, f.finish_time, f.winner_id,
       r.id red_id, r.first_name || ' ' || r.last_name red_name, r.image_url is not null red_img,
       b.id blue_id, b.first_name || ' ' || b.last_name blue_name, b.image_url is not null blue_img
from ufc.ufc_fight_excitement x
join ufc.ufc_fights f on f.id = x.fight_id
join ufc.ufc_events ev on ev.id = x.event_id
join ufc.ufc_fighters r on r.id = f.red_fighter_id
join ufc.ufc_fighters b on b.id = f.blue_fighter_id
where x.model_version = :v {where}
"""


def _sql(where: str) -> str:
    s = _SQL.format(where=where)
    t = UFCFightExcitement.__table__
    # SQLite (tests) has no schemas.
    return s if t.schema else s.replace("ufc.", "")


def _ready(db: Session) -> bool:
    t = UFCFightExcitement.__table__
    return sa_inspect(db.get_bind()).has_table(t.name, schema=t.schema)


def _window(window: str):
    try:
        return parse_window(window)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _date_filter(start, end) -> tuple[str, dict]:
    where, args = "", {}
    if start:
        where += " and x.date >= :start"
        args["start"] = start
    if end:
        where += " and x.date <= :end"
        args["end"] = end
    return where, args


def _fight(r) -> dict:
    c = json.loads(r.components)
    side = "red" if r.winner_id == r.red_id else "blue" if r.winner_id == r.blue_id else None
    return {
        "fight_id": str(r.fight_id), "event_id": str(r.event_id), "event_name": r.event_name,
        "date": r.date.isoformat() if hasattr(r.date, "isoformat") else r.date,
        "card_position": r.card_position, "weight_class": r.weight_class,
        "method": r.method, "round": r.finish_round, "time": r.finish_time,
        "red": {"id": str(r.red_id), "name": r.red_name, "has_image": bool(r.red_img)},
        "blue": {"id": str(r.blue_id), "name": r.blue_name, "has_image": bool(r.blue_img)},
        "winner": side,
        "score": r.score, "pct": r.pct, "groups": c["groups"], "raw": c["raw"],
        "p_fotn": r.p_fotn, "p_perf": r.p_perf, "fotn": bool(r.fotn), "perf": bool(r.perf),
    }


def _fights_build(db: Session, start, end, limit: int, q: str) -> list[dict]:
    """The window's bouts best first, each with its rank in the window; `q` keeps only
    bouts with a fighter whose name contains it (ranks stay the window's)."""
    if not _ready(db):
        return []
    where, args = _date_filter(start, end)
    ranked = _sql(where).replace("select x.fight_id,", "select row_number() over (order by x.score desc) rnk, x.fight_id,", 1)
    outer = f"select * from ({ranked}) t"
    if q:
        outer += " where lower(t.red_name) like :q or lower(t.blue_name) like :q"
        args["q"] = f"%{q.lower()}%"
    rows = db.execute(text(outer + " order by t.rnk limit :n"), {"v": MODEL_VERSION, "n": limit, **args}).all()
    return [{**_fight(r), "rank": r.rnk} for r in rows]


@router.get("/exciting/fights")
def exciting_fights(window: str = Query("all"), limit: int = Query(100, ge=1, le=500),
                    q: str = Query("", max_length=60), db: Session = Depends(get_db)):
    start, end = _window(window)
    q = q.strip()
    return response_cache.cached(f"highlights:fights:{start}:{end}:{limit}:{q.lower()}",
                                 with_session(_fights_build, start, end, limit, q), TTL)


def _cards_build(db: Session, start, end, scope: str, limit: int) -> list[dict]:
    if not _ready(db):
        return []
    where, args = _date_filter(start, end)
    by_event: dict[int, list] = defaultdict(list)
    for r in db.execute(text(_sql(where)), {"v": MODEL_VERSION, **args}):
        by_event[r.event_id].append(r)
    cards = []
    for rows in by_event.values():
        # Old cards have no bout order; their whole card stands in for the main card.
        main = [r for r in rows if r.card_position is not None and r.card_position <= MAIN_CARD]
        pool = main if scope == "main" and main else rows
        pool.sort(key=lambda r: -r.score)
        pcts = [r.pct for r in pool]
        cards.append({
            "event_id": str(pool[0].event_id), "event_name": pool[0].event_name,
            "date": pool[0].date.isoformat() if hasattr(pool[0].date, "isoformat") else pool[0].date,
            "score": card_score(pcts), "n_fights": len(pool),
            "n_great": sum(p >= 90 for p in pcts),
            "fights": [_fight(r) for r in pool],
        })
    cards.sort(key=lambda c: -c["score"])
    return cards[:limit]


@router.get("/exciting/cards")
def exciting_cards(window: str = Query("all"), scope: str = Query("main", pattern="^(main|full)$"),
                   limit: int = Query(50, ge=1, le=1000), db: Session = Depends(get_db)):
    start, end = _window(window)
    return response_cache.cached(f"highlights:cards:{start}:{end}:{scope}:{limit}",
                                 with_session(_cards_build, start, end, scope, limit), TTL)


def _bonuses_build(db: Session, event_id: int) -> dict | None:
    if not _ready(db):
        return None
    rows = db.execute(text(_sql(" and x.event_id = :e") + " order by f.card_position"),
                      {"v": MODEL_VERSION, "e": event_id}).all()
    if not rows:
        return None
    return {"event_id": str(event_id), "event_name": rows[0].event_name,
            "date": rows[0].date.isoformat() if hasattr(rows[0].date, "isoformat") else rows[0].date,
            "fights": [_fight(r) for r in rows]}


@router.get("/events/{event_id}/bonuses")
def event_bonuses(event_id: int, db: Session = Depends(get_db)):
    out = response_cache.cached(f"highlights:bonuses:{event_id}", with_session(_bonuses_build, event_id), TTL)
    if out is None:
        raise HTTPException(status_code=404, detail="No scored fights for this event")
    return out


def excitement_by_fight(db: Session, fight_ids: list[int]) -> dict[int, dict]:
    """Compact excitement per fight for the event detail and fight pages ({} if the
    scorer hasn't run here)."""
    if not fight_ids or not _ready(db):
        return {}
    rows = (db.query(UFCFightExcitement)
            .filter(UFCFightExcitement.model_version == MODEL_VERSION,
                    UFCFightExcitement.fight_id.in_(fight_ids)).all())
    return {r.fight_id: {"pct": r.pct, "p_fotn": r.p_fotn, "p_perf": r.p_perf,
                         "fotn": r.fotn, "perf": r.perf, "groups": json.loads(r.components)["groups"]}
            for r in rows}
