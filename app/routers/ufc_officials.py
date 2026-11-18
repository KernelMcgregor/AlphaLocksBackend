"""/ufc endpoints for the officials feature track (included into app.routers.ufc.router)."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.services import response_cache
from app.services.response_cache import with_session

router = APIRouter()
import json

from sqlalchemy import func

from app.models.ufc import UFCFight, UFCFightStats, UFCJudge
from app.models.ufc_officials import (
    UFCDecisionReview, UFCFightOfficial, UFCJudgeCard, UFCJudgePair, UFCJudgeProfile,
    UFCJudgeRound, UFCOfficialsMeta, UFCReferee, UFCRefereeFight,
)
from app.services.ufc.officials_service import MODEL_VERSION, norm_name

OFFICIALS_TTL = 30 * 60
V = MODEL_VERSION


def _dissents(db: Session, judge_ids: list[int] | None = None) -> dict[int, int]:
    """Decisions where the judge's card went against the official result (the lone
    dissent of a split, or the draw card of a majority decision)."""
    c = UFCJudgeCard
    q = (db.query(c.judge_id, func.count())
         .filter(c.version == V, c.result != 0, c.side != c.result))
    if judge_ids is not None:
        q = q.filter(c.judge_id.in_(judge_ids))
    return dict(q.group_by(c.judge_id).all())


def _profile(p: UFCJudgeProfile, full: bool = False, dissents: int | None = None) -> dict:
    d = json.loads(p.detail)
    out = {
        "judge_id": p.judge_id, "name": p.name, "eligible": p.eligible,
        "n_fights": p.n_fights, "n_rounds": p.n_rounds, "accuracy": p.accuracy,
        "expected_accuracy": p.expected_accuracy, "acc_above_exp": p.acc_above_exp,
        "agree_above_exp": p.agree_above_exp, "deciding_cards": p.deciding_cards,
        "dissents": dissents,
        "robbery_cards": p.robbery_cards, "first_date": p.first_date.isoformat() if p.first_date else None,
        "last_date": p.last_date.isoformat() if p.last_date else None,
        "map_x": p.map_x, "map_y": p.map_y,
        "leans": {k: {"effect": d.get(k), "se": d.get(f"{k}_se"), "pct": d.get(f"pct_{k}")}
                  for k in ("red", "fav", "home", "carry", "sig", "ctrl", "td", "kd")},
        "pct_acc_above_exp": d.get("pct_acc_above_exp"),
        "pct_agree_above_exp": d.get("pct_agree_above_exp"),
        "n_decided": d.get("n_decided"), "agree": d.get("agree"),
    }
    if full:
        out.update({k: d.get(k) for k in (
            "ll_above_exp", "agree_exp", "even_rate", "against_model", "model_strong",
            "fights_by_year", "ten8_pre2017", "ten8_pre2017_exp", "ten8_pre2017_n",
            "ten8_post2017", "ten8_post2017_exp", "ten8_post2017_n")})
    return out


def _pool(db: Session) -> dict | None:
    m = db.query(UFCOfficialsMeta).filter_by(kind="judges", version=V).first()
    return json.loads(m.payload)["pool"] if m else None


def _review(r: UFCDecisionReview) -> dict:
    return {
        "fight_id": str(r.fight_id), "date": r.date.isoformat(), "event_name": r.event_name,
        "weight_class": r.weight_class,
        "red": {"id": str(r.red_id), "name": r.red_name},
        "blue": {"id": str(r.blue_id), "name": r.blue_name},
        "result": r.result, "kind": r.kind, "p_red": r.p_red, "p_blue": r.p_blue,
        "robbery_score": r.robbery_score,
        "media": {"red": r.media_red, "blue": r.media_blue, "draw": r.media_draw},
        "fans": None if r.fan_red is None else {"red": r.fan_red, "blue": r.fan_blue, "draw": r.fan_draw},
        "cards": json.loads(r.cards),
    }


# --- Judges ---

def _judges_build(db: Session) -> dict:
    rows = db.query(UFCJudgeProfile).filter_by(version=V).order_by(UFCJudgeProfile.n_rounds.desc()).all()
    dis = _dissents(db)
    return {"pool": _pool(db), "judges": [_profile(p, dissents=dis.get(p.judge_id, 0)) for p in rows]}


@router.get("/judges")
def list_judges(db: Session = Depends(get_db)):
    return response_cache.cached("officials:judges", with_session(_judges_build), OFFICIALS_TTL)


def _agreement_build(db: Session) -> dict:
    """Every judge (the page filters to 150+ rounds or not) and every pair that shared
    enough rounds to compare."""
    profs = {p.judge_id: p for p in db.query(UFCJudgeProfile).filter_by(version=V)}
    pairs = (db.query(UFCJudgePair).filter_by(version=V)
             .filter(UFCJudgePair.judge_a.in_(profs), UFCJudgePair.judge_b.in_(profs)).all())
    return {
        "nodes": [{"judge_id": j, "name": p.name, "n_rounds": p.n_rounds, "x": p.map_x, "y": p.map_y,
                   "acc_above_exp": p.acc_above_exp, "eligible": p.eligible} for j, p in profs.items()],
        "edges": [{"a": e.judge_a, "b": e.judge_b, "n_rounds": e.n_rounds, "agree": e.agree,
                   "expected": e.expected, "excess": e.agree - e.expected} for e in pairs],
    }


@router.get("/judges/agreement")
def judge_agreement_map(db: Session = Depends(get_db)):
    return response_cache.cached("officials:agreement", with_session(_agreement_build), OFFICIALS_TTL)


def _judge_build(db: Session, judge_id: int) -> dict | None:
    p = db.query(UFCJudgeProfile).filter_by(version=V, judge_id=judge_id).first()
    if p is None:
        return None
    cards = (db.query(UFCJudgeCard, UFCDecisionReview)
             .join(UFCDecisionReview, (UFCDecisionReview.fight_id == UFCJudgeCard.fight_id)
                   & (UFCDecisionReview.version == V))
             .filter(UFCJudgeCard.version == V, UFCJudgeCard.judge_id == judge_id)
             .order_by(UFCJudgeCard.date.desc()).all())
    fights = [{
        **{k: v for k, v in _review(r).items() if k not in ("cards", "media", "fans")},
        "card": {"red": c.red_total, "blue": c.blue_total, "side": c.side},
        "pivotal": c.pivotal, "robbery_card": c.robbery_card, "against_model": c.against_model,
    } for c, r in cards]
    dis = _dissents(db, [judge_id]).get(judge_id, 0)
    return {"judge": _profile(p, full=True, dissents=dis), "pool": _pool(db), "fights": fights,
            "partners": _partners(db, judge_id)}


def _partners(db: Session, judge_id: int, limit: int = 8) -> list[dict]:
    """The judges they have shared the most rounds with, and how often the two agree vs
    what the round model expects."""
    pr = UFCJudgePair
    rows = (db.query(pr).filter(pr.version == V, (pr.judge_a == judge_id) | (pr.judge_b == judge_id))
            .order_by(pr.n_rounds.desc()).limit(limit).all())
    other = [r.judge_b if r.judge_a == judge_id else r.judge_a for r in rows]
    names = dict(db.query(UFCJudgeProfile.judge_id, UFCJudgeProfile.name)
                 .filter(UFCJudgeProfile.version == V, UFCJudgeProfile.judge_id.in_(other))) if other else {}
    return [{"judge_id": o, "name": names.get(o, ""), "n_rounds": r.n_rounds, "agree": r.agree,
             "expected": r.expected, "excess": r.agree - r.expected} for o, r in zip(other, rows)]


@router.get("/judges/{judge_id}")
def get_judge(judge_id: int, db: Session = Depends(get_db)):
    out = response_cache.cached(f"officials:judge:{judge_id}", with_session(_judge_build, judge_id),
                                OFFICIALS_TTL)
    if out is None:
        raise HTTPException(status_code=404, detail="Judge not found")
    return out


@router.get("/judges/{judge_id}/rounds")
def judge_misses(judge_id: int, limit: int = Query(25, ge=1, le=200), db: Session = Depends(get_db)):
    """The judge's rounds furthest from the model: given to the fighter the model had as
    least likely to win them (only rounds given against the model's favorite)."""
    r = UFCJudgeRound
    miss = func.abs(r.p_red - (r.verdict + 1) / 2.0)
    rows = (db.query(r).filter(r.version == V, r.judge_id == judge_id, r.verdict != 0, miss > 0.5)
            .order_by(miss.desc()).limit(limit).all())
    return [{"fight_id": str(x.fight_id), "round": x.round, "date": x.date.isoformat(),
             "event_name": x.event_name, "red_name": x.red_name, "blue_name": x.blue_name,
             "verdict": x.verdict, "score": x.score, "p_red": x.p_red} for x in rows]


@router.get("/decisions/{fight_id}/rounds")
def decision_rounds(fight_id: int, db: Session = Depends(get_db)):
    """Round-by-round cards of the whole panel for one decision, next to the model's
    P(red wins the round), plus each judge's official total (deductions included)."""
    def build(db: Session) -> dict:
        cards = (db.query(UFCJudgeCard).filter_by(version=V, fight_id=fight_id)
                 .order_by(UFCJudgeCard.judge_seq).all())
        jids = [c.judge_id for c in cards]
        names = dict(db.query(UFCJudgeProfile.judge_id, UFCJudgeProfile.name)
                     .filter(UFCJudgeProfile.version == V, UFCJudgeProfile.judge_id.in_(jids))) if jids else {}
        rr = db.query(UFCJudgeRound).filter_by(version=V, fight_id=fight_id).all()
        by_round: dict[int, dict] = {}
        for x in rr:
            r = by_round.setdefault(x.round, {"round": x.round, "p_red": x.p_red, "scores": {}})
            r["scores"][x.judge_id] = {"score": x.score, "verdict": x.verdict}
        return {
            "fight_id": str(fight_id),
            "judges": [{"judge_id": c.judge_id, "name": names.get(c.judge_id, ""), "red": c.red_total,
                        "blue": c.blue_total, "side": c.side} for c in cards],
            "rounds": [{"round": r["round"], "p_red": r["p_red"],
                        "scores": [r["scores"].get(j) for j in jids]} for r in sorted(by_round.values(), key=lambda r: r["round"])],
        }
    return response_cache.cached(f"officials:decision-rounds:{fight_id}", with_session(build), OFFICIALS_TTL)


@router.get("/fights/{fight_id}/round-stats")
def fight_round_stats(fight_id: int, db: Session = Depends(get_db)):
    """Per-round significant strikes, knockdowns, takedowns and control for both corners
    (the referee profile's expanded fight rows)."""
    def build(db: Session) -> dict | None:
        f = db.get(UFCFight, fight_id)
        if f is None:
            return None
        rows = (db.query(UFCFightStats).filter(UFCFightStats.fight_id == fight_id, UFCFightStats.round_number >= 1)
                .order_by(UFCFightStats.round_number).all())
        by_round: dict[int, dict] = {}
        for x in rows:
            side = "red" if x.fighter_id == f.red_fighter_id else "blue" if x.fighter_id == f.blue_fighter_id else None
            if side is None:
                continue
            by_round.setdefault(x.round_number, {"round": x.round_number})[side] = {
                "sig": x.sig_str_landed, "kd": x.kd, "td": x.td_landed, "ctrl": x.ctrl_seconds}
        # UFCStats pads some bouts with all-zero rounds past the finish.
        last = f.finish_round or max(by_round, default=0)
        return {"fight_id": str(fight_id), "finish_round": f.finish_round,
                "rounds": [r for k, r in sorted(by_round.items()) if k <= last]}
    out = response_cache.cached(f"officials:round-stats:{fight_id}", with_session(build), OFFICIALS_TTL)
    if out is None:
        raise HTTPException(status_code=404, detail="Fight not found")
    return out


# --- Robberies ---

def _robberies_build(db: Session, limit: int, min_score: float) -> list[dict]:
    rows = (db.query(UFCDecisionReview).filter(UFCDecisionReview.version == V,
                                               UFCDecisionReview.robbery_score >= min_score)
            .order_by(UFCDecisionReview.robbery_score.desc()).limit(limit).all())
    return [_review(r) for r in rows]


@router.get("/robberies")
def robberies(limit: int = Query(100, ge=1, le=500), min_score: float = Query(0.5, ge=0, le=1),
              db: Session = Depends(get_db)):
    return response_cache.cached(f"officials:robberies:{limit}:{min_score}",
                                 with_session(_robberies_build, limit, min_score), OFFICIALS_TTL)


# --- Referees ---

def _referee(r: UFCReferee, full: bool = False) -> dict:
    out = {"referee_id": r.id, "name": r.name, "n_fights": r.n_fights, "n_regional": r.n_regional,
           "finish_rate": r.finish_rate, "stoppage_index": r.stoppage_index,
           "deductions_per_100": r.deductions_per_100, "dq_rate": r.dq_rate,
           "first_date": r.first_date.isoformat() if r.first_date else None,
           "last_date": r.last_date.isoformat() if r.last_date else None}
    if full and r.detail:
        out.update(json.loads(r.detail))
    return out


def _referees_build(db: Session) -> list[dict]:
    return [_referee(r) for r in db.query(UFCReferee).order_by(UFCReferee.n_fights.desc())]


@router.get("/referees")
def list_referees(db: Session = Depends(get_db)):
    return response_cache.cached("officials:referees", with_session(_referees_build), OFFICIALS_TTL)


@router.get("/referees/by-name")
def referee_by_name(name: str, db: Session = Depends(get_db)):
    r = db.query(UFCReferee).filter_by(name_norm=norm_name(name)).first()
    if r is None:
        raise HTTPException(status_code=404, detail="Referee not found")
    return {"referee_id": r.id, "name": r.name}


def _referee_build(db: Session, referee_id: int) -> dict | None:
    r = db.get(UFCReferee, referee_id)
    if r is None:
        return None
    fights = (db.query(UFCRefereeFight).filter_by(referee_id=referee_id)
              .order_by(UFCRefereeFight.date.desc()).all())
    return {"referee": _referee(r, full=True), "fights": [{
        "fight_id": str(f.fight_id), "date": f.date.isoformat(), "event_name": f.event_name,
        "weight_class": f.weight_class, "red_name": f.red_name, "blue_name": f.blue_name,
        "winner": f.winner, "method": f.method, "finish_round": f.finish_round,
        "finish_time": f.finish_time, "absorbed": f.absorbed, "absorbed_z": f.absorbed_z,
        "absorbed_expected": f.absorbed_expected,
        "deductions": f.deductions} for f in fights]}


@router.get("/referees/{referee_id}")
def get_referee(referee_id: int, db: Session = Depends(get_db)):
    out = response_cache.cached(f"officials:referee:{referee_id}",
                                with_session(_referee_build, referee_id), OFFICIALS_TTL)
    if out is None:
        raise HTTPException(status_code=404, detail="Referee not found")
    return out


# --- Assigned officials (upcoming fights) ---

def _officials_build(db: Session, fight_id: int) -> dict:
    rows = (db.query(UFCFightOfficial).filter_by(fight_id=fight_id)
            .order_by(UFCFightOfficial.role, UFCFightOfficial.seq).all())
    jids = [o.judge_id for o in rows if o.judge_id]
    profs = {p.judge_id: p for p in db.query(UFCJudgeProfile)
             .filter(UFCJudgeProfile.version == V, UFCJudgeProfile.judge_id.in_(jids))} if jids else {}
    ref = next((o for o in rows if o.role == "referee"), None)
    referee = db.get(UFCReferee, ref.referee_id) if ref and ref.referee_id else None
    return {
        "judges": [{"seq": o.seq, "name": o.name, "judge_id": o.judge_id,
                    "profile": _profile(profs[o.judge_id]) if o.judge_id in profs else None}
                   for o in rows if o.role == "judge"],
        "referee": None if ref is None else {"name": ref.name, "referee_id": ref.referee_id,
                                             "profile": _referee(referee) if referee else None},
    }


@router.get("/fights/{fight_id}/officials")
def fight_officials(fight_id: int, db: Session = Depends(get_db)):
    return response_cache.cached(f"officials:fight:{fight_id}",
                                 with_session(_officials_build, fight_id), 5 * 60)


def resolve_official(db: Session, role: str, name: str) -> tuple[int | None, int | None]:
    """(judge_id, referee_id) for a typed name: exact normalised match, else alias."""
    from app.models.ufc import UFCJudgeAlias
    from app.services.ufc.mmad_scraper import _norm_judge
    if role == "judge":
        nn = _norm_judge(name)
        j = db.query(UFCJudge).filter(UFCJudge.name_norm == nn).first()
        if j is None:
            a = db.query(UFCJudgeAlias).filter(UFCJudgeAlias.alias_norm == nn).first()
            return (a.judge_id if a else None), None
        return j.id, None
    r = db.query(UFCReferee).filter_by(name_norm=norm_name(name)).first()
    return None, (r.id if r else None)
