"""A fighter's luck on the cards, from deserve to win (app/services/ufc/deserve_to_win.py).

For each of the fighter's decisions, deserve to win gives the share of 10,000 replays of
the fight's own round stats, scored by average judges, that the fighter wins. Summed over
their decisions that is how many they deserved to win; the cards gave them `wins`. The
difference is their luck. Finishes are left out: a comeback finish is not luck.

A robbery is a decision lost while winning most replays; a gift is one won while losing
most of them.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.ufc import UFCDeserveToWin, UFCFight, UFCFighter

DECISION_KINDS = {"ud", "sd", "md"}
#: Replay share at which a lost decision counts as a robbery (and 1 - this, a won one as a gift).
ROBBERY_P = 0.6


def decision_luck(db: Session, fighter_id: int) -> dict:
    rows = (
        db.query(UFCFight, UFCDeserveToWin)
        .join(UFCDeserveToWin, UFCDeserveToWin.fight_id == UFCFight.id)
        .filter((UFCFight.red_fighter_id == fighter_id) | (UFCFight.blue_fighter_id == fighter_id))
        .order_by(UFCDeserveToWin.created_at.desc())
        .all()
    )
    seen: set[int] = set()
    fights = []
    for fight, dtw in rows:
        if fight.id in seen:  # newest model version only
            continue
        seen.add(fight.id)
        kind = (dtw.official_outcome or "").split(":")[-1]
        if kind not in DECISION_KINDS:
            continue
        red = fight.red_fighter_id == fighter_id
        p = dtw.p_red if red else dtw.p_blue
        fights.append({
            "fight_id": str(fight.id),
            "date": fight.date.isoformat() if fight.date else None,
            "opponent_id": str(fight.blue_fighter_id if red else fight.red_fighter_id),
            "kind": kind,
            "won": fight.winner_id == fighter_id,
            "draw": fight.winner_id is None,
            "p": round(p, 4),
            "p_draw": round(dtw.p_draw, 4),
        })
    names = {}
    if fights:
        ids = {int(f["opponent_id"]) for f in fights}
        names = {str(i): f"{fn} {ln}".strip() for i, fn, ln in
                 db.query(UFCFighter.id, UFCFighter.first_name, UFCFighter.last_name).filter(UFCFighter.id.in_(ids))}
    for f in fights:
        f["opponent_name"] = names.get(f["opponent_id"])
    fights.sort(key=lambda f: f["date"] or "", reverse=True)
    # A draw on the cards counts half, as does the replays' draw share.
    actual = sum(1.0 if f["won"] else 0.5 if f["draw"] else 0.0 for f in fights)
    expected = sum(f["p"] + f["p_draw"] / 2 for f in fights)
    return {
        "decisions": len(fights),
        "won": actual,
        "deserved": round(expected, 2),
        "luck": round(actual - expected, 2),
        "robberies": sum(1 for f in fights if not f["won"] and not f["draw"] and f["p"] >= ROBBERY_P),
        "gifts": sum(1 for f in fights if f["won"] and f["p"] <= 1 - ROBBERY_P),
        "fights": fights,
    }
