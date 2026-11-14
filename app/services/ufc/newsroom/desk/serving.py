"""What the public pages read from the news desk: the story list and one story.

Every read tolerates the news tables not existing yet (an empty list, a 404).
"""

from __future__ import annotations

import datetime as dt
import json
import re

from sqlalchemy import func, or_
from sqlalchemy.orm import Session


def _iso(t):
    return t.isoformat() if t else None


def _dek(content: str) -> str | None:
    for para in re.split(r"\n\s*\n", content or ""):
        p = para.strip()
        if p and not p.startswith(("#", "```", "|", ">", "-")):
            return p
    return None


def _image(t) -> dict | None:
    if t.image is None:
        return None
    return {"focal_x": t.image_focal_x, "focal_y": t.image_focal_y,
            "v": int((t.updated_at or t.created_at).timestamp()), "source": t.image_source, "page": t.image_page}


def _fighter_brief(db: Session, ids: list[int]) -> list[dict]:
    from app.models.ufc import UFCFighter
    rows = {f.id: f for f in db.query(UFCFighter).filter(UFCFighter.id.in_(ids))} if ids else {}
    return [{"id": str(i), "first_name": rows[i].first_name, "last_name": rows[i].last_name,
             "image_url": rows[i].image_url, "country_code": rows[i].country_code} for i in ids if i in rows]


def news_list(db: Session, limit: int = 30) -> list[dict]:
    from app.models.ufc import UFCNewsTopic
    try:
        rows = (db.query(UFCNewsTopic).filter(UFCNewsTopic.status == "published")
                .order_by(UFCNewsTopic.published_at.desc()).limit(limit).all())
    except Exception:
        db.rollback()
        return []
    star = _star_power(db, {f for t in rows for f in _focus(t)})
    return [{"slug": t.slug, "title": t.title, "kind": t.kind, "layout": t.layout, "dek": _dek(t.content),
             "published_at": _iso(t.published_at), "updated_at": _iso(t.last_seen), "image": _image(t),
             "outlets": t.outlet_count or 1, "star": max((star.get(f, 0) for f in _focus(t)), default=0),
             "fighters": _fighter_brief(db, _focus(t))} for t in rows]


def _star_power(db: Session, ids: set[int]) -> dict[int, float]:
    """How big a name is, for ranking stories on the front page: 1 a champion, 0.6 top
    five, 0.3 ranked."""
    from app.models.ufc import UFCFighterRanking
    from app.services.ufc.champions import current_champions
    if not ids:
        return {}
    out: dict[int, float] = {}
    for f, r in (db.query(UFCFighterRanking.fighter_id, func.min(UFCFighterRanking.rank))
                 .filter(UFCFighterRanking.fighter_id.in_(ids)).group_by(UFCFighterRanking.fighter_id)):
        out[f] = 0.6 if r is not None and r <= 5 else 0.3
    for f in current_champions(db).values():
        if f in ids:
            out[f] = 1.0
    return out


def _focus(t) -> list[int]:
    return json.loads(t.focus_ids or "null") or json.loads(t.fighter_ids or "[]")


def _age(dob, today: dt.date) -> int | None:
    if not dob:
        return None
    d = dob if isinstance(dob, dt.date) else dt.date.fromisoformat(str(dob))
    return today.year - d.year - ((today.month, today.day) < (d.month, d.day))


def _card(db: Session, f, champs: dict, today: dt.date) -> dict:
    """A fighter's sidebar profile: the bio from their profile page, belt or rank, last
    three results, next fight."""
    from app.models.ufc import UFCEvent, UFCFight, UFCFighterRanking
    rank = (db.query(UFCFighterRanking).filter(UFCFighterRanking.fighter_id == f.id)
            .order_by(UFCFighterRanking.rank).first())
    fights = (db.query(UFCFight, UFCEvent).join(UFCEvent, UFCEvent.id == UFCFight.event_id)
              .filter(or_(UFCFight.red_fighter_id == f.id, UFCFight.blue_fighter_id == f.id))
              .order_by(UFCEvent.date.desc()).limit(14).all())
    last, nxt = [], None
    for fight, ev in fights:
        opp = fight.blue_fighter if fight.red_fighter_id == f.id else fight.red_fighter
        oname = f"{opp.first_name} {opp.last_name}".strip() if opp else "TBA"
        if fight.method is None and ev.date >= today:
            nxt = {"fight_id": str(fight.id), "event": ev.name, "date": str(ev.date), "opponent": oname}
        elif fight.method and len(last) < 10:
            res = "W" if fight.winner_id == f.id else "L" if fight.winner_id else "NC"
            last.append({"fight_id": str(fight.id), "date": str(ev.date), "opponent": oname,
                         "opponent_fighter": None if opp is None else {
                             "id": str(opp.id), "first_name": opp.first_name, "last_name": opp.last_name,
                             "image_url": opp.image_url, "country_code": opp.country_code},
                         "event": ev.name, "result": res, "method": fight.method,
                         "round": fight.finish_round, "time": fight.finish_time})
    # Skills: the fighter's Glicko percentiles going into their latest bout (the radar),
    # and a handful of career rates.
    skills = {}
    try:
        from app.services.ufc.fight_context_service import _get_glicko_data
        latest = fights[0][0] if fights else None
        g = _get_glicko_data(db, 0, f.id, latest.weight_class if latest else None) or {}
        skills = {k: round(v["percentile"]) for k, v in (g.get("dimensions") or {}).items()
                  if isinstance(v, dict) and v.get("percentile") is not None}
    except Exception:
        db.rollback()
    stats = None
    try:
        from app.models.ufc import UFCFighterCareerStats
        cs = db.query(UFCFighterCareerStats).filter(UFCFighterCareerStats.fighter_id == f.id).first()
        if cs and cs.fight_count:
            stats = {"slpm": cs.slpm, "sapm": cs.sapm, "sig_acc": cs.sig_acc, "sig_def": cs.sig_def,
                     "td15": cs.td15, "td_def": cs.td_def, "kd15": cs.kd15}
    except Exception:
        db.rollback()
    # Style: UFC.com's fighting style when listed, else the fighter's style family.
    style = f.fighting_style
    if not style:
        try:
            from app.models.ufc import UFCFighterStyle, UFCStyleFamily
            style = (db.query(UFCStyleFamily.name).join(UFCFighterStyle, UFCFighterStyle.family == UFCStyleFamily.id)
                     .filter(UFCFighterStyle.fighter_id == f.id).scalar())
        except Exception:
            db.rollback()
    return {"id": str(f.id), "first_name": f.first_name, "last_name": f.last_name,
            "image_url": f.image_url, "country_code": f.country_code,
            "bio": {"age": _age(f.dob, today), "height": f.height, "weight": f.weight, "reach": f.reach,
                    "stance": f.stance, "style": style, "hometown": f.birthplace,
                    "debut": str(f.octagon_debut) if f.octagon_debut else None},
            "record": f"{f.wins}-{f.losses}-{f.draws}", "champion": f.id in champs,
            "division": (champs.get(f.id) or (rank.weight_class if rank else None) or "").replace("_", " ") or None,
            "rank": rank.rank if rank else None, "last": last, "next": nxt, "skills": skills, "stats": stats}


def news_story(db: Session, slug: str) -> dict | None:
    from app.models.ufc import UFCFighter, UFCNewsTopic
    from app.services.ufc.champions import current_champions
    try:
        t = db.query(UFCNewsTopic).filter(UFCNewsTopic.slug == slug, UFCNewsTopic.status == "published").first()
    except Exception:
        db.rollback()
        return None
    if t is None:
        return None
    ids = _focus(t)
    champs = {fid: div for div, fid in current_champions(db).items()}
    fighters = {f.id: f for f in db.query(UFCFighter).filter(UFCFighter.id.in_(ids))} if ids else {}
    today = dt.date.today()
    cards = [_card(db, fighters[i], champs, today) for i in ids if i in fighters]
    layout = t.layout or ("profile" if len(cards) <= 1 else "pair" if len(cards) == 2 else "group")
    return {"slug": t.slug, "title": t.title, "kind": t.kind, "layout": layout, "content": t.content,
            "published_at": _iso(t.published_at), "updated_at": _iso(t.last_seen),
            "image": _image(t), "fighters": cards,
            "matchup": ({"red": cards[0]["id"], "blue": cards[1]["id"]} if layout == "matchup" and len(cards) >= 2 else None)}


def news_image(db: Session, slug: str):
    from app.models.ufc import UFCNewsTopic
    t = db.query(UFCNewsTopic).filter(UFCNewsTopic.slug == slug).first()
    return None if t is None else t.image
