"""One source of truth for a fighter's division, activity, and rankability.

Before this module, `ranking_service` (Glicko) and `points_ranking_service` (Points+Elo)
each derived division, last-fight-date, and eligibility independently, from *differently
filtered* fight streams. Points skipped any bout with no `winner_id`; Glicko did not. The
two therefore disagreed about who was active and even about which division someone
fought in, and since Glicko wrote placeholder rows with `rank=0` while Points wrote the
real ranks, every disagreement left a `rank=0` row on the site — sorted above rank 1.

The distinction that fixes it:

    activity and division come from EVERY bout, including no-contests and draws
    rating updates come only from DECIDED bouts

A no-contest is not a year of inactivity. Tom Aspinall's 2025-10-25 title fight was waved
off with no winner; Points concluded his last fight was 2024-07-27, put him 771 days past
the 548-day cutoff, and dropped him — while Glicko, which counted the bout, kept him. A
no-contest also still tells you which division someone competes in: Ode Osbourne's last
*decided* bout was at Bantamweight but his last bout was at Flyweight, so the two services
filed him under different divisions and both rows survived.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from app.models.ufc import UFCFight, UFCFightStats

log = logging.getLogger("fighter_registry")

#: Methods that mean "this bout produced no rating-bearing result".
UNDECIDED_METHOD_MARKERS = ("No Contest", "DQ", "Draw", "Overturned")


def classify_weight_class(wc: str | None) -> str:
    """Map a raw ufcstats bout name ("Lightweight Bout") to a division key.

    Women's featherweight is pooled into `w_bantamweight` deliberately — it is a defunct
    two-fighter division and both the Glicko and Points classifiers have always pooled it
    identically, so this is not a divergence.
    """
    if not isinstance(wc, str):
        return "unknown"
    wc_lower = wc.lower()
    is_womens = "women" in wc_lower
    if "strawweight" in wc_lower:
        return "w_strawweight" if is_womens else "strawweight"
    if "flyweight" in wc_lower:
        return "w_flyweight" if is_womens else "flyweight"
    if "bantamweight" in wc_lower:
        return "w_bantamweight" if is_womens else "bantamweight"
    if "featherweight" in wc_lower:
        return "w_bantamweight" if is_womens else "featherweight"
    if "lightweight" in wc_lower:
        return "lightweight"
    if "welterweight" in wc_lower:
        return "welterweight"
    if "middleweight" in wc_lower:
        return "middleweight"
    if "light heavyweight" in wc_lower or "light_heavyweight" in wc_lower:
        return "light_heavyweight"
    if "heavyweight" in wc_lower:
        return "heavyweight"
    return "unknown"


def is_decided(method: str | None, winner_id: int | None) -> bool:
    """Whether a bout carries a rating-bearing result.

    Both conditions matter. `winner_id` is NULL for waved-off bouts that still have a
    method string, and the method markers catch overturned results where a winner was
    recorded and later vacated.
    """
    if not winner_id:
        return False
    m = method or ""
    return not any(marker in m for marker in UNDECIDED_METHOD_MARKERS)


@dataclass(frozen=True)
class FighterState:
    """What every ranking consumer needs to agree on."""

    division: str          #: last division fought in — ALL bouts, incl. NC/draw
    last_activity: date    #: last bout of any kind — NOT last decided bout
    decided_fights: int    #: bouts with a rating-bearing result
    rounds: int            #: scored rounds in decided bouts
    last_decided: date | None = None  #: for diagnostics; never used for eligibility


@dataclass(frozen=True)
class Eligibility:
    """Thresholds a fighter must clear to appear in a published ranking.

    ANDed, not ORed. Both services previously applied only one of these, which is the
    other half of why their eligible sets differed: 10 rounds admits a fighter with two
    five-round bouts that Points would reject, and 3 fights admits a fighter with four
    first-round finishes that Glicko would reject for having too little scored material.
    """

    min_decided_fights: int = 3
    min_rounds: int = 10
    max_days_inactive: int = 548


def is_rankable(st: FighterState, today: date, crit: Eligibility = Eligibility()) -> bool:
    return (
        st.decided_fights >= crit.min_decided_fights
        and st.rounds >= crit.min_rounds
        and st.division != "unknown"
        and (today - st.last_activity).days <= crit.max_days_inactive
    )


def build_fighter_registry(db) -> dict[int, FighterState]:
    """Derive every fighter's state in two passes over the fight table."""
    fights = db.query(UFCFight).order_by(UFCFight.date, UFCFight.id).all()

    last_activity: dict[int, date] = {}
    last_decided: dict[int, date] = {}
    division: dict[int, str] = {}
    decided_fights: dict[int, int] = defaultdict(int)

    decided_fight_ids: set[int] = set()

    for f in fights:
        if not f.date:
            continue
        wc = classify_weight_class(f.weight_class)
        decided = is_decided(f.method, f.winner_id)
        if decided:
            decided_fight_ids.add(f.id)

        for fid in (f.red_fighter_id, f.blue_fighter_id):
            if fid is None:
                continue
            # Activity counts every bout. Walking out to the cage is not inactivity,
            # whatever the result was.
            if f.date >= last_activity.get(fid, date.min):
                last_activity[fid] = f.date
                # Division follows the latest bout in a *recognised* division, so a
                # catchweight or unparsed bout does not erase a known division.
                if wc != "unknown":
                    division[fid] = wc
            if decided:
                decided_fights[fid] += 1
                if f.date >= last_decided.get(fid, date.min):
                    last_decided[fid] = f.date

    # Rounds: only those actually scored in decided bouts, matching what the rating
    # engines consume. round_number 0 is the per-fight totals row, not a round.
    rounds: dict[int, int] = defaultdict(int)
    if decided_fight_ids:
        rows = (
            db.query(UFCFightStats.fighter_id, UFCFightStats.fight_id)
            .filter(UFCFightStats.round_number > 0)
            .all()
        )
        for fighter_id, fight_id in rows:
            if fight_id in decided_fight_ids:
                rounds[fighter_id] += 1

    registry = {
        fid: FighterState(
            division=division.get(fid, "unknown"),
            last_activity=act,
            decided_fights=decided_fights.get(fid, 0),
            rounds=rounds.get(fid, 0),
            last_decided=last_decided.get(fid),
        )
        for fid, act in last_activity.items()
    }
    log.info(f"  Registry: {len(registry)} fighters, "
             f"{len(decided_fight_ids)} decided bouts of {len(fights)}")
    return registry
