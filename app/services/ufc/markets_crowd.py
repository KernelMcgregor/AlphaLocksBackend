"""Crowd, model and market on one completed fight: who the fans said won, beside the two prices.

Fan votes are mmadecisions.com scorecards (mmad_decisions.fan_a / fan_b / fan_draw), so they exist
only for fights that went to the judges, and they are cast *after* the fight -- a verdict on what
happened, not a forecast. That is the point of the comparison: how the people who watched it scored
it, against what the model and the closing line expected going in.

fighter_a is mmadecisions' first-listed fighter; `swapped` True means fighter_a is the DB blue corner.
"""
from __future__ import annotations

import statistics

from sqlalchemy.orm import Session

from app.models.ufc import MMADDecision, UFCFight, UFCFightOpenClose, UFCFightPrediction


def fan_split(fan_a: int | None, fan_b: int | None, fan_draw: int | None, swapped: bool | None) -> dict | None:
    """Fan scorecards as red / blue / draw shares, in DB corners."""
    a, b, d = fan_a or 0, fan_b or 0, fan_draw or 0
    n = a + b + d
    if n <= 0:
        return None
    red, blue = (b, a) if swapped else (a, b)
    return {"n": n, "red": round(red / n, 4), "blue": round(blue / n, 4), "draw": round(d / n, 4)}


def close_prob(db: Session, fight_id: int) -> tuple[float | None, str | None]:
    """Vig-free red close: BestFightOdds consensus where it exists, else the median book's last
    pre-cutoff price from the line history."""
    q = (db.query(UFCFightOpenClose.red_close_prob)
         .filter(UFCFightOpenClose.fight_id == fight_id, UFCFightOpenClose.bookmaker == "Consensus",
                 UFCFightOpenClose.red_close_prob.isnot(None))
         .scalar())
    if q is not None:
        return float(q), "consensus"
    from app.services.ufc.markets_book_history import book_history
    hist = book_history(db, fight_id)
    last = [b["points"][-1]["red_prob"] for b in (hist or {}).get("books", []) if b["points"]]
    if not last:
        return None, None
    return float(statistics.median(last)), "books"


def crowd(db: Session, fight_id: int) -> dict | None:
    fight = db.query(UFCFight).filter(UFCFight.id == fight_id).first()
    if fight is None:
        return None
    m = db.query(MMADDecision).filter(MMADDecision.fight_id == fight_id).first()
    fans = fan_split(m.fan_a, m.fan_b, m.fan_draw, m.swapped) if m else None
    pred = db.query(UFCFightPrediction).filter(UFCFightPrediction.fight_id == fight_id).first()
    q, source = close_prob(db, fight_id)
    winner = ("red" if fight.winner_id == fight.red_fighter_id
              else "blue" if fight.winner_id == fight.blue_fighter_id else None)
    return {
        "fight_id": str(fight_id),
        "winner": winner,
        "method": fight.method,
        "fans": fans,
        "model": {"red_prob": pred.red_prob, "model_prob": pred.model_prob} if pred else None,
        "market": {"red_close_prob": round(q, 4), "source": source} if q is not None else None,
    }
