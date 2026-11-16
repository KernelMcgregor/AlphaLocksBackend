"""A UFC fighter's bouts outside the UFC, from their linked Sherdog record (sherdog_bouts).

For the profile page's fight history: regional, Bellator, PFL, ONE... bouts listed beside
the UFC ones. UFC bouts are left out (the UFCStats record is the source for those), both by
promotion and by date, so a Sherdog row for a UFC bout can never show twice.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from app.models.ufc import SherdogBout, SherdogFighter, UFCFight
from app.services.ufc.promotions import clean

METHODS = {"KO/TKO": "KO/TKO", "SUB": "Submission", "DEC": "Decision", "DQ": "DQ", "DRAW": "Draw", "NC": "No contest"}


def promotion_label(event_name: str | None, promotion: str | None) -> str | None:
    """'Fury FC', 'Bellator', 'ONE', 'TUF'; an ambiguous abbreviation keeps its letters."""
    label = clean(event_name, promotion).split(":")[0]
    if label and label != "unknown":
        return label
    head = (event_name or "").split(" - ")[0].strip()
    return head or None


def external_fights(db: Session, fighter_id: int) -> list[dict]:
    sf = db.query(SherdogFighter).filter(SherdogFighter.ufc_fighter_id == fighter_id).first()
    if sf is None:
        return []
    ufc_dates = {d for (d,) in db.query(UFCFight.date).filter(
        (UFCFight.red_fighter_id == fighter_id) | (UFCFight.blue_fighter_id == fighter_id)) if d}
    near_ufc = {d + dt.timedelta(days=k) for d in ufc_dates for k in (-1, 0, 1)}
    bouts = (db.query(SherdogBout)
             .filter(SherdogBout.fighter_sherdog_id == sf.sherdog_id)
             .order_by(SherdogBout.date.desc())
             .all())
    keep = [b for b in bouts
            if clean(b.event_name, b.promotion) != "UFC" and (b.date is None or b.date not in near_ufc)]
    # Opponents who also fought in the UFC get a profile link.
    opp_ids = {b.opponent_sherdog_id for b in keep if b.opponent_sherdog_id}
    linked = dict(db.query(SherdogFighter.sherdog_id, SherdogFighter.ufc_fighter_id)
                  .filter(SherdogFighter.sherdog_id.in_(opp_ids), SherdogFighter.ufc_fighter_id.isnot(None))
                  ) if opp_ids else {}
    return [{
        "date": b.date.isoformat() if b.date else None,
        "result": b.result,
        "opponent_name": b.opponent_name,
        "opponent_id": str(linked[b.opponent_sherdog_id]) if b.opponent_sherdog_id in linked else None,
        "promotion": promotion_label(b.event_name, b.promotion),
        "event_name": b.event_name,
        "method": METHODS.get(b.method_class, b.method or None),
        "round": b.round,
        "time": b.time,
    } for b in keep]
