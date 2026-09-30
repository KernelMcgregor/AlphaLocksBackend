"""Classify how a bout ended, for rating systems that weight results by type.

A win is not a win: a split decision is weak evidence, and a round-one leg injury
(McGregor vs Poirier 3, Pantoja's arm) says little about who was the better fighter.
Rating code used to treat every KO/TKO the same, and the only split was a hand-set
decision weight. The outcome types here let each type's score be FITTED from data
(scripts/fit_elo_outcomes.py) instead of assumed.

ufcstats records injury stoppages as plain KO/TKO; the injury is only in `details`
("to McGregor knee injury", "to Arm Injury"). Doctor's stoppages are their own method.
"""
from __future__ import annotations

from app.services.ufc.fighter_registry import is_decided, is_draw

KO = "ko"
SUB = "sub"
UD = "ud"
MD = "md"
SD = "sd"
DOCTOR = "doctor"
INJURY = "injury"
DRAW = "draw"
VOID = "void"

#: Types that carry a winner, in a stable order (used as parameter names).
WIN_TYPES = (KO, SUB, UD, MD, SD, DOCTOR, INJURY)


def classify_outcome(method: str | None, details: str | None, winner_id) -> str:
    m = method if isinstance(method, str) else ""
    d = details.lower() if isinstance(details, str) else ""
    winner = winner_id if winner_id == winner_id else None  # NaN -> None
    if not is_decided(m, winner):
        return DRAW if is_draw(m, winner) else VOID
    if "Doctor" in m:
        return DOCTOR
    if "KO" in m:
        # An injury the loser suffered, not damage the winner dealt. "Eye injury" is
        # usually strike damage and is ruled a doctor's stoppage, handled above.
        return INJURY if "injur" in d else KO
    if "Sub" in m:
        return SUB
    if "Split" in m:
        return SD
    if "Majority" in m:
        return MD
    return UD  # "Decision - Unanimous" and the legacy bare "Decision"
