"""Post-fight review blocks for the completed-fight page.

The fight payload already carries the prediction, method prediction, expected stats,
props and exchange quotes. This adds what only makes sense once the bout is over:

  grade          the pre-fight call scored against the result, beside the closing
                 de-vigged consensus (log loss per fight, the accuracy-first measure,
                 not "did we hit")
  deserve_to_win ufc_deserve_to_win (deserve_to_win.py): share of 10,000 replays each
                 corner wins, the round model's P(red takes each round), likeliest cards
  scorecards     the judges' round cards (mmadecisions, else UFCStats totals), each
                 judge's career dissent rate, and the mmadecisions fan vote
  line           BestFightOdds consensus moneyline open -> close

Every block is None when its source has nothing for the fight, so a 1990s bout or a
fight the scrapers missed renders the sections it can.
"""
from __future__ import annotations

import json
import math

from sqlalchemy import case, func, inspect
from sqlalchemy.orm import Session

from app.models.ufc import (
    MMADDecision, UFCDeserveToWin, UFCFight, UFCFightOpenClose, UFCJudge, UFCJudgeScorecard,
    UFCRankingHistory,
)

#: Probabilities are clipped before the log so a 0/1 never produces an infinite loss.
EPS = 1e-6


def method_class(method: str | None) -> str | None:
    """ko | sub | dec, on the method model's three classes (doctor stoppages are KO/TKO)."""
    m = (method or "").lower()
    if "decision" in m:
        return "dec"
    if "sub" in m:
        return "sub"
    if "ko" in m or "doctor" in m:
        return "ko"
    return None


def _log_loss(p_red: float | None, red_won: bool) -> float | None:
    if p_red is None:
        return None
    p = min(1 - EPS, max(EPS, p_red if red_won else 1 - p_red))
    return -math.log(p)


def _consensus_line(db: Session, fight_id: int) -> UFCFightOpenClose | None:
    return (db.query(UFCFightOpenClose)
            .filter(UFCFightOpenClose.fight_id == fight_id,
                    UFCFightOpenClose.bookmaker == "Consensus")
            .first())


def _implied(american: int) -> float:
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def _plausible(red: int | None, blue: int | None) -> bool:
    """Both prices present and a believable hold. BFO's backfill has a few openings like
    -850 / -850 (one side's price copied to both), which de-vig to a fake 50%."""
    if red is None or blue is None:
        return False
    return _implied(red) + _implied(blue) < 1.2


def line_payload(line: UFCFightOpenClose | None) -> dict | None:
    if line is None:
        return None
    open_ok = _plausible(line.red_open, line.blue_open)
    close_ok = _plausible(line.red_close, line.blue_close)
    return {
        "red_open": line.red_open if open_ok else None,
        "blue_open": line.blue_open if open_ok else None,
        "red_close": line.red_close if close_ok else None,
        "blue_close": line.blue_close if close_ok else None,
        "red_open_prob": line.red_open_prob if open_ok else None,
        "red_close_prob": line.red_close_prob if close_ok else None,
        "opened_at": line.opened_at.isoformat() if line.opened_at else None,
        "closed_at": line.closed_at.isoformat() if line.closed_at else None,
    }


def grade_payload(fight: UFCFight, pred, method_pred, line: UFCFightOpenClose | None) -> dict | None:
    """The pre-fight numbers scored against what happened. None for draws / no contests
    (nothing to score a two-way probability against) and fights with no prediction."""
    if pred is None or fight.winner_id is None:
        return None
    red_won = fight.winner_id == fight.red_fighter_id
    market = line.red_close_prob if line and _plausible(line.red_close, line.blue_close) else None
    out = {
        "winner_side": "red" if red_won else "blue",
        "picked_side": pred.predicted_winner,
        "correct": pred.predicted_winner == ("red" if red_won else "blue"),
        "red_prob": pred.red_prob,
        "model_prob": pred.model_prob,
        "market_red_prob": market,
        "log_loss": _log_loss(pred.red_prob, red_won),
        "model_log_loss": _log_loss(pred.model_prob, red_won),
        "market_log_loss": _log_loss(market, red_won),
        "method": None,
    }

    cls = method_class(fight.method)
    if method_pred is not None and cls is not None:
        side = out["winner_side"]
        marg = {"ko": method_pred.ko_prob, "sub": method_pred.sub_prob, "dec": method_pred.dec_prob}
        # The winner x method cell ("red by KO") when the joint grid exists (method_v2).
        joint = getattr(method_pred, f"{side}_{cls}_prob", None)
        p = marg.get(cls)
        out["method"] = {
            "actual": cls,
            "predicted": method_class(method_pred.predicted_method),
            "prob": p,
            "joint_prob": joint,
            # 1 = the model's likeliest method, 3 = its least likely
            "rank": None if p is None else 1 + sum(1 for v in marg.values() if v is not None and v > p),
        }
    return out


def deserve_to_win_payload(db: Session, fight_id: int) -> dict | None:
    # Only the databases the deserve-to-win script has written to have the table.
    t = UFCDeserveToWin.__table__
    if not inspect(db.get_bind()).has_table(t.name, schema=t.schema):
        return None
    row = (db.query(UFCDeserveToWin)
           .filter(UFCDeserveToWin.fight_id == fight_id)
           .order_by(UFCDeserveToWin.created_at.desc())
           .first())
    if row is None:
        return None
    return {
        "model_version": row.model_version,
        "p_red": row.p_red, "p_draw": row.p_draw, "p_blue": row.p_blue,
        "panel_p_red": row.panel_p_red, "panel_p_draw": row.panel_p_draw,
        "panel_p_blue": row.panel_p_blue,
        "p_ud": row.p_ud, "p_sd": row.p_sd, "p_md": row.p_md,
        "rounds_observed": row.rounds_observed,
        "rounds_scheduled": row.rounds_scheduled,
        "extrapolated": row.extrapolated,
        "partial_round_seconds": row.partial_round_seconds,
        "round_p_red": json.loads(row.round_p_red) if row.round_p_red else [],
        "top_cards": json.loads(row.top_cards) if row.top_cards else [],
        "official_outcome": row.official_outcome,
        "robbery_score": row.robbery_score,
        "n_sims": row.n_sims,
    }


def _judge_dissent(db: Session, judge_ids: list[int]) -> dict[int, dict]:
    """Per judge: decisions scored and how often their total picked a different side
    from the panel majority (mmadecisions totals, rounds already summed to round 0)."""
    if not judge_ids:
        return {}
    sc = UFCJudgeScorecard
    side = case((sc.red_pts > sc.blue_pts, 1), (sc.red_pts < sc.blue_pts, -1), else_=0)
    totals = (db.query(sc.mmad_decision_id.label("d"), sc.judge_id.label("j"), side.label("s"))
              .filter(sc.source == "mmad", sc.round == 0, sc.red_pts.isnot(None))
              .subquery())
    panel = (db.query(totals.c.d, func.sum(totals.c.s).label("net"))
             .group_by(totals.c.d).subquery())
    majority = case((panel.c.net > 0, 1), (panel.c.net < 0, -1), else_=0)
    dissent = case((totals.c.s != majority, 1), else_=0)
    rows = (db.query(totals.c.j, func.count(), func.sum(dissent))
            .join(panel, panel.c.d == totals.c.d)
            .filter(totals.c.j.in_(judge_ids))
            .group_by(totals.c.j)
            .all())
    return {j: {"n": n, "dissents": int(d or 0), "dissent_rate": (d or 0) / n if n else None}
            for j, n, d in rows}


def scorecards_payload(db: Session, fight: UFCFight) -> dict | None:
    rows = db.query(UFCJudgeScorecard).filter(UFCJudgeScorecard.fight_id == fight.id).all()
    if not rows:
        return None
    # Round-by-round cards when mmadecisions has them; UFCStats only has the totals.
    source = "mmad" if any(r.source == "mmad" for r in rows) else "ufcstats"
    rows = [r for r in rows if r.source == source]

    judge_ids = sorted({r.judge_id for r in rows if r.judge_id is not None})
    names = dict(db.query(UFCJudge.id, UFCJudge.name).filter(UFCJudge.id.in_(judge_ids)).all()) if judge_ids else {}
    dissent = _judge_dissent(db, judge_ids)

    judges: dict[int, dict] = {}
    for r in sorted(rows, key=lambda r: (r.judge_seq, r.round)):
        j = judges.setdefault(r.judge_seq, {
            "seq": r.judge_seq,
            "judge_id": r.judge_id,
            "name": names.get(r.judge_id) or "Unknown judge",
            "career": dissent.get(r.judge_id),
            "total": None,
            "rounds": [],
        })
        cell = {"red": r.red_pts, "blue": r.blue_pts, "red_ded": r.red_ded, "blue_ded": r.blue_ded}
        if r.round == 0:
            j["total"] = cell
        else:
            j["rounds"].append({"round": r.round, **cell})

    out = {"source": source, "judges": list(judges.values()), "fans": None}

    # Fan vote, put on DB corners: mmad's fighter_a is the DB blue corner when swapped.
    dec = db.query(MMADDecision).filter(MMADDecision.fight_id == fight.id).first()
    if dec is not None and dec.fan_n:
        a, b = (dec.fan_b, dec.fan_a) if dec.swapped else (dec.fan_a, dec.fan_b)
        out["fans"] = {"n": dec.fan_n, "red": a, "blue": b, "draw": dec.fan_draw}
    return out


def _rank_rows(db: Session, as_of, fighter_ids) -> dict[int, dict]:
    """fighter -> {"division": row, "p4p": row} in the ranking stamped `as_of`."""
    out: dict[int, dict] = {}
    for r in (db.query(UFCRankingHistory)
              .filter(UFCRankingHistory.as_of == as_of,
                      UFCRankingHistory.fighter_id.in_(fighter_ids))):
        slot = "p4p" if r.weight_class.startswith("p4p") else "division"
        out.setdefault(r.fighter_id, {})[slot] = {
            "weight_class": r.weight_class, "rank": r.rank,
            "total_ranked": r.total_ranked, "score": r.score,
        }
    return out


def rank_movement_payload(db: Session, fight: UFCFight) -> dict | None:
    """Each corner's divisional rank entering the night and after it.

    Read from ufc_ranking_history, which the ranker writes at every event date with the
    division rules already applied — so this never re-derives a division. The rule that
    matters here (fighter_registry.current_division, Tapology's): a fighter is ranked in
    the class of their last TWO bouts. A first bout in a new class leaves them ranked in
    the old one ("pending" below); the second establishes the move ("moved"), so the
    before and after rows sit in different divisions. Catchweights count toward neither.

    Before = the latest ranking stamped before the fight date (the previous event);
    after = the one stamped on the fight date, which includes this card's results.
    None when the history does not cover both dates.
    """
    from app.services.ufc.fighter_registry import classify_weight_class

    if fight.date is None:
        return None
    before_date = (db.query(func.max(UFCRankingHistory.as_of))
                   .filter(UFCRankingHistory.as_of < fight.date).scalar())
    after_date = (db.query(func.min(UFCRankingHistory.as_of))
                  .filter(UFCRankingHistory.as_of >= fight.date).scalar())
    if before_date is None or after_date is None:
        return None
    ids = [fight.red_fighter_id, fight.blue_fighter_id]
    before = _rank_rows(db, before_date, ids)
    after = _rank_rows(db, after_date, ids)
    fought_in = classify_weight_class(fight.weight_class)

    out = {"before_date": before_date.isoformat(), "after_date": after_date.isoformat(),
           "fought_in": None if fought_in == "unknown" else fought_in}
    for corner, fid in (("red", fight.red_fighter_id), ("blue", fight.blue_fighter_id)):
        b = before.get(fid, {}).get("division")
        a = after.get(fid, {}).get("division")
        if b and a:
            if b["weight_class"] != a["weight_class"]:
                status = "moved"            # second bout in the new class: move established
            elif fought_in != "unknown" and fought_in != b["weight_class"]:
                status = "pending"          # first bout away: still ranked at home
            else:
                status = "same"
        elif a:
            # A debut and a return to the rankings read very differently.
            prior = (db.query(UFCFight.id)
                     .filter(UFCFight.date < fight.date,
                             (UFCFight.red_fighter_id == fid) | (UFCFight.blue_fighter_id == fid))
                     .first())
            status = "entered" if prior else "debut"
        elif b:
            status = "dropped"
        else:
            status = "unranked"
        out[corner] = {
            "status": status,
            "before": b, "after": a,
            "p4p_before": before.get(fid, {}).get("p4p"),
            "p4p_after": after.get(fid, {}).get("p4p"),
        }
    return out


def review_payload(db: Session, fight: UFCFight, pred, method_pred) -> dict:
    line = _consensus_line(db, fight.id)
    return {
        "grade": grade_payload(fight, pred, method_pred, line),
        "deserve_to_win": deserve_to_win_payload(db, fight.id),
        "scorecards": scorecards_payload(db, fight),
        "line": line_payload(line),
        "rankings": rank_movement_payload(db, fight),
    }
