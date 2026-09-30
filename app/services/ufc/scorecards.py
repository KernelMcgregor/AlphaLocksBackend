"""Judges' scorecards from the UFCStats fight details.

For decisions, ufc_fights.details holds each judge's total, e.g.
    "Mike Bell 28 - 29. Chris Lee 29 - 28. Sal D'amato 28 - 29."
UFCStats lists the LOSER's points first (checked on all 4,400+ scored decisions: the
majority never has the first number higher), so "28 - 29" is a judge scoring it for the
winner. Draws have no winner and the order does not matter.

Used two ways:
  - Elo: a decision's score for the winner comes from how the judges scored it
    (mean point margin) instead of the three fixed buckets split/majority/unanimous.
  - Features: a fighter's "decision luck" (judges' verdicts minus how the stats say the
    fight went) and their average decision margin, both over previous fights only.
"""
from __future__ import annotations

import math
import re

_JUDGE = re.compile(r"([A-Za-z][^.\d]*?)\s+(\d+)\s*-\s*(\d+)")


def parse(details: str | None) -> list[tuple[str, int, int]]:
    """-> [(judge, winner_pts, loser_pts)] (for draws: just the two numbers as listed)."""
    if not isinstance(details, str):  # None / NaN from a DataFrame
        return []
    return [(j.strip(), int(b), int(a)) for j, a, b in _JUDGE.findall(details)]


def winner_margin(details: str | None) -> float | None:
    """Mean over judges of (winner points - loser points). None if no scores parsed.
    30-27 x3 -> 3.0; 29-28 x3 -> 1.0; a 29-28, 29-28, 28-29 split -> 0.33."""
    cards = parse(details)
    if not cards:
        return None
    return sum(w - l for _, w, l in cards) / len(cards)


def margin_score(margin: float, alpha: float) -> float:
    """Elo score for a decision winner from the judges' mean margin, in [0.5, 1)."""
    return 0.5 + 0.5 * math.tanh(alpha * max(margin, 0.0))
