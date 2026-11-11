"""The home dashboard's payload (/ufc/home), built once and cached.

The page used to assemble itself from six requests, two of them chained (events list ->
event detail) and one fanned out (career stats for ~930 ranked fighters in four chunks,
~2 MB). Cold, that was seconds. Here the server does the joins and ships only what the
page draws: ~20 KB, warmed at startup and after every scheduled job.

Sections: pound-for-pound top 10 (men), the last completed card's results with the
model's pick on each, rotating stat leaders among ranked fighters (per 15 minutes of
cage time), the model's primary pick per upcoming fight, its season record, and the
next card's newest articles.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

#: Leaderboard floor: enough cage time that a rate means something.
MIN_FIGHTS = 5
MIN_MINUTES = 45
LEADERS = 5

#: (key, label, unit, value from a career-stats row). slpm and sl_diff are per minute and
#: are scaled to the per-15 the other rates are stored in.
LEADER_STATS = [
    ("slpm", "Sig. strikes landed", "per 15 min", lambda c: (c.slpm or 0) * 15, "int"),
    ("sl_diff", "Striking differential", "per 15 min", lambda c: (c.sl_diff or 0) * 15, "signed"),
    ("kd15", "Knockdowns", "per 15 min", lambda c: c.kd15 or 0, "dec2"),
    ("td15", "Takedowns landed", "per 15 min", lambda c: c.td15 or 0, "dec1"),
    ("sub_att15", "Submission attempts", "per 15 min", lambda c: c.sub_att15 or 0, "dec1"),
    ("ctrl15", "Control time", "per 15 min", lambda c: c.ctrl15 or 0, "clock"),
    ("sig_def", "Strike defense", "career", lambda c: (c.sig_def or 0) * 100, "pct"),
]

GRADE_RANK = {"A+": 0, "A": 1, "A-": 2, "B+": 3, "B": 4, "B-": 5, "C+": 6, "C": 7}
MAX_EDGES = 8
MAX_PER_FAMILY = 2


def _fighter(f) -> dict | None:
    if f is None:
        return None
    return {"id": str(f.id), "first_name": f.first_name, "last_name": f.last_name,
            "image_url": f.image_url, "country_code": f.country_code}


def _alt(kind: str) -> list[dict]:
    """Top ten of the men's P4P or BMF list."""
    from app.services.ufc.alt_rankings_publisher import get_alt_rankings

    men = sorted(get_alt_rankings(kind)["pools"].get("men", []), key=lambda r: r["default_rank"])[:10]
    return [{
        "id": r["id"], "first_name": r["first_name"], "last_name": r["last_name"],
        "image_url": r["image_url"], "country_code": r["country_code"],
        "rank": r["default_rank"], "is_champion": r["is_champion"],
        "division_label": r["division_label"], "wins": r["wins"], "losses": r["losses"], "draws": r["draws"],
    } for r in men]


def _last_event(db: Session) -> dict | None:
    from sqlalchemy.orm import joinedload

    from app.models.ufc import UFCEvent, UFCFight

    ev = (db.query(UFCEvent)
          .filter(UFCEvent.date <= dt.date.today(),
                  UFCEvent.fights.any(UFCFight.winner_id.isnot(None)))
          .order_by(UFCEvent.date.desc()).first())
    if ev is None:
        return None
    fights = (db.query(UFCFight)
              .options(joinedload(UFCFight.red_fighter), joinedload(UFCFight.blue_fighter))
              .filter(UFCFight.event_id == ev.id, UFCFight.winner_id.isnot(None))
              .order_by(UFCFight.card_position.is_(None), UFCFight.card_position, UFCFight.id)
              .all())
    picks = _picks_for(db, [f.id for f in fights])
    out = []
    for f in fights:
        red_won = f.winner_id == f.red_fighter_id
        pick = picks.get(f.id)
        out.append({
            "id": str(f.id), "method": f.method, "finish_round": f.finish_round,
            "winner": _fighter(f.red_fighter if red_won else f.blue_fighter),
            "loser": _fighter(f.blue_fighter if red_won else f.red_fighter),
            # Was the model's pick the winner? None when there was no prediction.
            "model_correct": None if pick is None else pick == ("red" if red_won else "blue"),
        })
    graded = [x["model_correct"] for x in out if x["model_correct"] is not None]
    return {"id": str(ev.id), "name": ev.name, "date": ev.date.isoformat(), "fights": out,
            "model_record": {"correct": sum(graded), "graded": len(graded)}}


def _picks_for(db: Session, fight_ids: list[int]) -> dict[int, str]:
    """The model's predicted corner per fight (newest prediction row wins)."""
    from app.models.ufc import UFCFightPrediction

    out: dict[int, str] = {}
    for fid, side in (db.query(UFCFightPrediction.fight_id, UFCFightPrediction.predicted_winner)
                      .filter(UFCFightPrediction.fight_id.in_(fight_ids))
                      .order_by(UFCFightPrediction.created_at)):
        out[fid] = side
    return out


def _season(db: Session) -> dict:
    """The model's winner-pick accuracy this calendar year, over decided fights."""
    from app.models.ufc import UFCFight

    year = dt.date.today().year
    fights = (db.query(UFCFight.id, UFCFight.winner_id, UFCFight.red_fighter_id, UFCFight.blue_fighter_id)
              .filter(UFCFight.date >= dt.date(year, 1, 1), UFCFight.date <= dt.date.today(),
                      UFCFight.winner_id.isnot(None)).all())
    picks = _picks_for(db, [f.id for f in fights])
    graded = [picks[f.id] == ("red" if f.winner_id == f.red_fighter_id else "blue")
              for f in fights if f.id in picks]
    return {"year": year, "correct": sum(graded), "graded": len(graded)}


def _articles(db: Session, limit: int = 4) -> list[dict]:
    """The newest previews for the next card, main event first, each with its two
    fighters' portraits as the image: official UFC photos the site already caches, so
    every article has artwork without anyone sourcing it."""
    from app.models.ufc import UFCFight
    from app.routers.ufc import _build_previews

    rows = [r for r in _build_previews(db, 40, 0) if r.get("is_upcoming")]
    if not rows:
        rows = _build_previews(db, limit, 0)
    next_date = min((r["fight_date"] for r in rows if r.get("fight_date")), default=None)
    rows = [r for r in rows if r.get("fight_date") == next_date] or rows
    from app.models.ufc import UFCArticleImage

    ids = [int(r["fight_id"]) for r in rows]
    fights = {f.id: f for f in db.query(UFCFight).filter(UFCFight.id.in_(ids)).all()}
    images = {fid: (fx, fy, at) for fid, fx, fy, at in db.query(
        UFCArticleImage.fight_id, UFCArticleImage.focal_x, UFCArticleImage.focal_y, UFCArticleImage.fetched_at)
        .filter(UFCArticleImage.fight_id.in_(ids))}
    rows.sort(key=lambda r: (getattr(fights.get(int(r["fight_id"])), "card_position", None) is None,
                             getattr(fights.get(int(r["fight_id"])), "card_position", None) or 0))
    out = []
    for r in rows[:limit]:
        f = fights.get(int(r["fight_id"]))
        img = images.get(int(r["fight_id"]))
        out.append({
            # Picked artwork when there is one (focal point = where the faces are).
            "image": None if img is None else {
                "focal_x": img[0], "focal_y": img[1],
                "v": int(img[2].timestamp()) if img[2] else 0,
            },
            "fight_id": r["fight_id"], "headline": r["headline"], "lede": r["lede"],
            "event_name": r["event_name"], "fight_date": r["fight_date"],
            "red": _fighter(f.red_fighter) if f else None, "blue": _fighter(f.blue_fighter) if f else None,
        })
    return out


def _stat_leaders(db: Session) -> list[dict]:
    from app.models.ufc import UFCFighter, UFCFighterCareerStats, UFCFighterRanking
    from app.services.ufc.ranking_publisher import WEIGHT_CLASS_LABELS

    rows = (db.query(UFCFighterCareerStats, UFCFighter, UFCFighterRanking.weight_class)
            .join(UFCFighter, UFCFighter.id == UFCFighterCareerStats.fighter_id)
            .join(UFCFighterRanking, UFCFighterRanking.fighter_id == UFCFighter.id)
            .filter(UFCFighterRanking.rank > 0,
                    UFCFighterCareerStats.fight_count >= MIN_FIGHTS,
                    UFCFighterCareerStats.total_fight_min >= MIN_MINUTES)
            .all())
    seen, pool = set(), []
    for c, f, wc in rows:                       # a fighter ranked in two divisions counts once
        if f.id not in seen:
            seen.add(f.id)
            pool.append((c, f, WEIGHT_CLASS_LABELS.get(wc, wc)))
    boards = []
    for key, label, unit, val, fmt in LEADER_STATS:
        top = sorted(pool, key=lambda t: val(t[0]), reverse=True)[:LEADERS]
        boards.append({"key": key, "label": label, "unit": unit, "fmt": fmt, "rows": [
            {**_fighter(f), "division": div, "value": round(val(c), 3)} for c, f, div in top]})
    return boards


def _model_edges(db: Session) -> list[dict]:
    from app.services import response_cache
    from app.services.ufc.picks_v2 import build_all

    # Shares the picks page's cache entry (same key, same session-owning thunk), so a
    # warm one costs nothing here.
    cards = response_cache.cached("picks-v2:all", response_cache.with_session(build_all), 120)
    picks = []
    for card in cards or []:
        for f in card.get("fights", []):
            m = next((x for x in f.get("markets", [])
                      if x.get("primary") and x.get("pick") and x.get("grade") in GRADE_RANK), None)
            if m is None:
                continue
            picks.append({
                "fight_id": f["fight_id"], "family": m["family"], "grade": m["grade"],
                "pick_label": m["pick_label"], "market": m["label"],
                "price": m.get("median_american"), "edge_pts": m.get("edge_pts"),
                "red": f["red"]["name"], "blue": f["blue"]["name"],
            })
    picks.sort(key=lambda p: (GRADE_RANK[p["grade"]], -(p["edge_pts"] or 0)))
    per_family, out = {}, []
    for p in picks:
        per_family[p["family"]] = per_family.get(p["family"], 0) + 1
        if per_family[p["family"]] <= MAX_PER_FAMILY:
            out.append(p)
    return out[:MAX_EDGES]


#: Short labels for the rankings switcher, in the order the buttons appear.
#: Divisions without the "-weight", the way fans and broadcasts say them ("bantam",
#: "light heavy"). Lightweight keeps its name: "light" alone reads as nothing.
DIVISION_SHORT = {
    "flyweight": "Fly", "bantamweight": "Bantam", "featherweight": "Feather", "lightweight": "Lightweight",
    "welterweight": "Welter", "middleweight": "Middle", "light_heavyweight": "Light Heavy",
    "heavyweight": "Heavy", "w_strawweight": "W. Straw", "w_flyweight": "W. Fly",
    "w_bantamweight": "W. Bantam",
}


def _rankings(p4p: list[dict], bmf: list[dict]) -> list[dict]:
    """P4P plus each division's top ten (champion first), from the cached rankings
    payload the Rankings page serves, so this costs nothing when that is warm."""
    from app.routers.ufc import RANKINGS_TTL, _rankings_build
    from app.services import response_cache

    boards = [{"key": "p4p", "short": "P4P", "label": "Pound for pound", "rows": p4p},
              {"key": "bmf", "short": "BMF", "label": "Baddest Mother F***er", "rows": bmf}]
    data = response_cache.cached("rankings", _rankings_build, RANKINGS_TTL)
    for wc in data.get("weight_classes", []):
        rows = sorted(wc["fighters"], key=lambda f: f["rank"])[:10]
        boards.append({
            "key": wc["key"], "short": DIVISION_SHORT.get(wc["key"], wc["label"]),
            "label": ("Women's " if wc["key"].startswith("w_") else "") + wc["label"],
            "rows": [{
                "id": f["id"], "first_name": f["first_name"], "last_name": f["last_name"],
                "image_url": f["image_url"], "country_code": f["country_code"],
                "rank": f["display_rank"], "is_champion": f["is_champion"],
                "wins": f["wins"], "losses": f["losses"], "draws": f["draws"],
            } for f in rows],
        })
    return boards


def build_home(db: Session) -> dict:
    edges = _model_edges(db)
    p4p = _alt("p4p")
    return {
        "p4p": p4p,
        "rankings": _rankings(p4p, _alt("bmf")),
        "last_event": _last_event(db),
        "stat_leaders": _stat_leaders(db),
        "model_edges": edges,
        "strong_edges": sum(1 for e in edges if GRADE_RANK[e["grade"]] <= GRADE_RANK["B+"]),
        "season": _season(db),
        "articles": _articles(db),
    }
