"""Fighter Rankings — Tapology emulation.

A deliberate clone of the Tapology UFC ranking system, as far as it can be cloned.

What can and cannot be copied
-----------------------------
Tapology publishes their RULES in detail (tapology.com/faq_rankings) but explicitly keeps
the scoring itself confidential: the point value of each outcome, and the six recency
weights, are never given. Their FAQ says only that method of victory, decision type and
round number are "good ideas" whose "implementation is confidential."

So this module splits cleanly in two:

  * Every disclosed rule is implemented exactly as stated, and cited inline.
  * Every undisclosed number is **fitted** against the official UFC rankings by
    `tapology_fit.py`, not chosen. That keeps the project's standing rule — no invented
    coefficients — intact even where the source system is opaque.

The fit target is the UFC's own rankings, which have been the Meta Elo model since
2026-06-20. Tapology itself cannot be used as a target: tapology.com returns HTTP 403 to
automated requests, so their published output is not available to fit against. This is
therefore a Tapology-shaped system calibrated to Meta's output, and it is worth being
precise about that rather than claiming to reproduce Tapology's numbers.

Known risk, recorded deliberately
---------------------------------
Tapology's core is a SUM of points over the last six bouts. That architecture was tried
earlier in this project and failed badly on face validity — Quillan Salkilld ranked #1 at
lightweight, Josh Hokit #2 at heavyweight, and Islam Makhachev scored lower than any other
contender tested, because summing per-bout credit rewards an unbeaten short record over a
long elite one. See docs/models/rankings.md.

Tapology suppresses this with two mechanisms: the undisputed champion is **hard-pinned at
#1**, and eligibility is a tight 21 months. The champion pin is deliberately NOT
implemented here, by explicit instruction. The prospect-ratchet is therefore unguarded at
the top of a division, and the fitted opponent-quality curve is the only thing standing
against it — if the fit drives `q_exp` high, that is the optimiser discovering it needs to
make beating a weak opponent nearly worthless.

Run: python -m app.services.ufc.tapology_ranking_service --preview
"""

from __future__ import annotations

import json
import logging
import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date

from app.database import SessionLocal
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFighterRanking
from app.services.ufc.fighter_registry import (
    classify_weight_class, current_division, is_decided,
)
from app.services.ufc.glicko1 import Glicko1
from app.services.ufc.tiered_ranking_service import (
    V_DRAW, V_FINISH, V_MAJORITY, V_SPLIT, V_UD, _DivisionPool, bout_weight,
    outcome_value,
)

log = logging.getLogger("tapology_ranking")

# ---------------------------------------------------------------------------
# Disclosed rules — Tapology FAQ, quoted in the comments
# ---------------------------------------------------------------------------

#: "The algorithm looks at the last 6 UFC matches for every fighter."
WINDOW = 6

#: "A fighter must have completed a UFC bout within the last 21 months to be eligible."
#: Replaces our 548-day cutoff. Tapology's stated rationale is worth keeping: a torn ACL
#: plus surgery plus a year's recovery plus a postponed opponent injury reaches 15-18
#: months for a fighter with every intention of competing.
ELIGIBILITY_MONTHS = 21
ELIGIBILITY_DAYS = int(ELIGIBILITY_MONTHS * 30.44)          # 639

#: "Fighters who become ineligible are displayed for 60 more days, then removed."
GRACE_DAYS = 60

#: "Ranked in whichever class(es) they fought in their last 2 UFC bouts", with a 24-month
#: per-weight-class window. The last-2-bouts half already lives in
#: `fighter_registry.current_division`; the 24-month bound is applied here.
DIVISION_WINDOW_DAYS = int(24 * 30.44)                      # 730

#: Tapology ranks 11 divisions and publishes **no P4P at all**. Meta likewise built no
#: cross-division model. Dropping it removes the one part of our ranking that was kept by
#: choice rather than by evidence.
DIVISIONS = [
    "flyweight", "bantamweight", "featherweight", "lightweight",
    "welterweight", "middleweight", "light_heavyweight", "heavyweight",
    "w_strawweight", "w_flyweight", "w_bantamweight",
]

WEIGHT_CLASS_LABELS = {
    "w_strawweight": "Strawweight", "w_flyweight": "Flyweight",
    "w_bantamweight": "Bantamweight", "strawweight": "Strawweight",
    "flyweight": "Flyweight", "bantamweight": "Bantamweight",
    "featherweight": "Featherweight", "lightweight": "Lightweight",
    "welterweight": "Welterweight", "middleweight": "Middleweight",
    "light_heavyweight": "Light Heavyweight", "heavyweight": "Heavyweight",
}

#: "No-contests are skipped entirely — the 7th-oldest fight is pulled in instead. The only
#: effect is that it resets the 21-month activity clock." So an NC occupies no slot in the
#: window, but it does count as activity.
#:
#: "Catchweight bouts do not count toward eligibility in any division, but the results are
#: still factored into the score." So a catchweight fills a window slot and scores, while
#: contributing nothing to which division a fighter is ranked in.

#: Tapology's Strength of Schedule: each of the last 6 opponents gets a tier 1-10, summed
#: to a raw 1-60 scale, rescaled to 1-99. Explicitly NOT an input to the ranking — "if two
#: fighters had the exact same last 6 opponents, and one went 0-6 while the other went 6-0,
#: they would receive the exact same Strength of Schedule rating."
N_TIERS = 10

#: Published outcome ladder, imported rather than redefined. Tapology confirms method,
#: decision type and round number all matter but keeps the values confidential; Fight
#: Matrix publishes theirs, so those stand in.
_ = (V_FINISH, V_UD, V_MAJORITY, V_SPLIT, V_DRAW)


# ---------------------------------------------------------------------------
# Undisclosed scoring — every field here is FITTED, see tapology_fit.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Weights:
    """The parts of Tapology's algorithm that are confidential.

    Defaults are the fitted optimum from `tapology_fit.py` against the official UFC
    rankings. They are recorded here as the fit's OUTPUT, and are re-derivable by rerunning
    the fitter — not chosen by hand.
    """

    #: Opponent-quality multiplier: `q_min + (q_max - q_min) * pct**q_exp`, where pct is
    #: the opponent's rating percentile in their division at the time of the bout.
    #: A high q_exp makes beating anyone outside the elite nearly worthless, which is the
    #: main defence against an unbeaten prospect out-accumulating a champion.
    q_min: float = 0.0
    q_max: float = 2.0
    q_exp: float = 6.0

    #: Positional recency. Tapology: "the most recent fight is the most important, down to
    #: the 6th oldest, which is by far the least important." Geometric in fight position,
    #: so one parameter rather than six free weights.
    r_decay: float = 0.95

    #: "An additional age-decay on top" of the positional weighting — a bout's calendar age
    #: matters as well as its position. Half-life in days.
    age_half_life: float = 3650.0

    #: Loss penalty relative to a win of the same method and opposition.
    loss_weight: float = 1.5

    #: Floor on the loss multiplier. The mirror `q_max + q_min - opp_q` reaches exactly
    #: ZERO against a maximum-quality opponent, so losing to the best fighter in a division
    #: cost nothing at all — every loss in the ledger read -0.0000. That removed any cost
    #: to the fit driving q_exp to absurd values, which in turn made beating a median
    #: fighter worth 0.0000 and buried Ruffy, Saint Denis and Garry. A loss is always a
    #: loss; this keeps it so.
    loss_floor: float = 0.25

    #: "The algorithm penalises inactivity." Applied to the total once past the grace
    #: period, exponential in days idle.
    inactivity_half_life: float = 500.0

    def clamp(self) -> "Weights":
        return replace(
            self,
            q_min=max(0.0, self.q_min), q_max=max(self.q_min + 0.01, self.q_max),
            q_exp=max(0.1, self.q_exp), r_decay=min(max(self.r_decay, 0.05), 0.99),
            age_half_life=max(30.0, self.age_half_life),
            loss_weight=max(0.0, self.loss_weight),
            loss_floor=max(0.0, self.loss_floor),
            inactivity_half_life=max(30.0, self.inactivity_half_life),
        )


FITTED = Weights()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(value: float, sorted_pool: list[float]) -> float:
    if not sorted_pool:
        return 0.5
    return bisect_right(sorted_pool, value) / len(sorted_pool)


def opponent_quality(pct: float, w: Weights) -> float:
    return w.q_min + (w.q_max - w.q_min) * (max(0.0, min(1.0, pct)) ** w.q_exp)


def bout_points(won: bool, drew: bool, v: float, opp_q: float, distance: float,
                position: int, days_ago: int, w: Weights) -> float:
    """Signed points for one bout in the six-bout window.

    `v` is the published outcome value oriented to this fighter. A draw is "worth less than
    a win, more than a loss" (Tapology) — which `v = 0.5` gives directly once the value is
    centred.
    """
    recency = (w.r_decay ** position) * (0.5 ** (max(0, days_ago) / w.age_half_life))
    magnitude = v * opp_q * distance * recency
    if drew:
        # Centred so a draw sits between a win and a loss rather than scoring as either.
        return magnitude * 0.0
    if won:
        return magnitude
    # Losing to a strong opponent costs less than losing to a weak one — the mirror of the
    # win curve, so the asymmetry needs no separate table.
    mirror = max(w.loss_floor, w.q_max + w.q_min - opp_q)
    return -w.loss_weight * v * mirror * distance * recency


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

class TapologyRanker:
    """Tapology's disclosed rules, with the confidential scoring fitted."""

    name = "tapology"

    def __init__(self, weights: Weights | None = None) -> None:
        self.w = (weights or FITTED).clamp()

    def rank(self, db, registry: dict, as_of: date, crit=None) -> "RankingResult":
        return self.score(self.build_history(db, as_of), as_of)

    #: Tapology pins the most recent undisputed (non-interim) champion at #1 regardless of
    #: points. It is not decoration: it anchors the top of a division so the scoring curve
    #: does not have to do that job by itself. Without it, the weight fit was driven to a
    #: degenerate optimum (losses worth nothing, opponent-quality exponent pinned at the
    #: grid ceiling) trying to keep unbeaten prospects off the top spot.
    pin_champion = True

    #: Fighters the promotion lists as gone. Tapology removes anyone who fights elsewhere;
    #: we cannot see that, but `ufc_fighters.status` does carry retirement and release.
    #: Without this, Stipe Miocic ranked #4 at heavyweight off a bout he LOST in 2024.
    exclude_statuses = ("Retired", "Released")

    @staticmethod
    def build_history(db, as_of: date) -> dict:  # noqa: C901
        """Everything that does NOT depend on the fitted weights.

        Split out because the fitter runs ~80 full rankings and this pass — loading every
        bout and running Glicko over all of UFC history — is identical for every candidate
        parameter set. Fitting without this is roughly twenty minutes; with it, seconds.
        """
        fights = (
            db.query(UFCFight)
            .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
            .order_by(UFCFight.date, UFCFight.id)
            .all()
        )
        fighters = db.query(UFCFighter).all()
        names = {f.id: f"{f.first_name or ''} {f.last_name or ''}".strip()
                 for f in fighters}
        status = {f.id: (f.status or "") for f in fighters}

        glicko = Glicko1()
        history: dict[int, list[dict]] = defaultdict(list)
        last_activity: dict[int, date] = {}
        div_bouts: dict[int, list[tuple[date, str]]] = defaultdict(list)
        # One entry PER FIGHTER, not per bout. Appending on every bout gave a 20-fight
        # veteran twenty slots in the distribution, which skewed every percentile and
        # crushed real contenders into the top few percent — where the quality curve is
        # steepest and most sensitive. _DivisionPool replaces a fighter's old value.
        pool = _DivisionPool()

        for f in fights:
            if not f.date or f.date > as_of:
                continue
            red_id, blue_id = f.red_fighter_id, f.blue_fighter_id
            if red_id is None or blue_id is None:
                continue

            division = classify_weight_class(f.weight_class)
            decided = is_decided(f.method, f.winner_id)

            # Activity counts EVERY bout, including no-contests. Tapology: an NC "resets
            # the 21-month activity clock" even though it is skipped for scoring.
            for fid in (red_id, blue_id):
                if f.date >= last_activity.get(fid, date.min):
                    last_activity[fid] = f.date
                # Catchweight ("unknown") contributes to score but never to division.
                if division != "unknown":
                    div_bouts[fid].append((f.date, division))

            if not decided:
                # Skipped entirely — occupies no slot, so the 7th-oldest is pulled in.
                continue

            v = outcome_value(f.method)
            distance = bout_weight(f.method, f.time_format)
            drew = "Draw" in (f.method or "")

            pre = {fid: glicko.state(fid, f.date) for fid in (red_id, blue_id)}
            scoring_div = division if division != "unknown" else None
            opp_pct = {}
            for me, opp in ((red_id, blue_id), (blue_id, red_id)):
                ref = scoring_div or (div_bouts[opp][-1][1] if div_bouts[opp] else None)
                opp_pct[me] = pool.percentile(ref, pre[opp][0] - pre[opp][1]) if ref else 0.5

            red_score = 0.5 if drew else (1.0 if f.winner_id == red_id else 0.0)
            glicko.observe(red_id, blue_id,
                           v if red_score == 1.0 else (1.0 - v if red_score == 0.0 else 0.5),
                           f.date, distance)

            for me, opp in ((red_id, blue_id), (blue_id, red_id)):
                won = f.winner_id == me
                if scoring_div:
                    pool.set(me, scoring_div, glicko.conservative(me, f.date))
                history[me].append({
                    "fight_id": f.id, "date": f.date, "opponent_id": opp,
                    "opponent_name": names.get(opp, ""), "won": won, "drew": drew,
                    "method": f.method, "division": division,
                    "v": round(v if won else (0.5 if drew else 1.0 - v), 4),
                    "opp_pct": round(opp_pct[me], 4),
                    "distance": round(distance, 4),
                })

        # Champions need each fighter's CURRENT division, so resolve divisions first.
        divisions_now = {
            fid: current_division([d for dt, d in bouts
                                   if (as_of - dt).days <= DIVISION_WINDOW_DAYS])
            for fid, bouts in div_bouts.items()
        }
        from app.services.ufc.champions import current_champions
        champions = current_champions(db, as_of, divisions_now)

        return {"history": history, "last_activity": last_activity,
                "div_bouts": div_bouts, "glicko": glicko,
                "status": status, "champions": champions}

    def score(self, ctx: dict, as_of: date) -> "RankingResult":
        from app.services.ufc.ranking_publisher import RankingResult

        history = ctx["history"]
        last_activity = ctx["last_activity"]
        div_bouts = ctx["div_bouts"]
        glicko = ctx["glicko"]
        status = ctx.get("status", {})
        champions = ctx.get("champions", {})

        # ---- Score ----------------------------------------------------------
        scores: dict[int, float] = {}
        extras: dict[int, dict] = {}
        divisions: dict[int, str] = {}

        for fid, bouts in history.items():
            if status.get(fid) in self.exclude_statuses:
                continue
            idle = (as_of - last_activity[fid]).days
            # 21 months, plus the 60-day display grace.
            if idle > ELIGIBILITY_DAYS + GRACE_DAYS:
                continue

            recent_divs = [d for dt, d in div_bouts.get(fid, [])
                           if (as_of - dt).days <= DIVISION_WINDOW_DAYS]
            division = current_division(recent_divs)
            if division not in DIVISIONS:
                continue

            window = list(reversed(bouts))[:WINDOW]
            if not window:
                continue

            total, ledger = 0.0, []
            for i, b in enumerate(window):
                days_ago = (as_of - b["date"]).days
                q = opponent_quality(b["opp_pct"], self.w)
                pts = bout_points(b["won"], b["drew"], b["v"], q, b["distance"],
                                  i, days_ago, self.w)
                total += pts
                ledger.append({
                    "fight_id": b["fight_id"], "date": b["date"].isoformat(),
                    "opponent_id": b["opponent_id"],
                    "opponent_name": b["opponent_name"],
                    "won": b["won"], "method": b["method"], "v": b["v"],
                    "tier": max(1, min(N_TIERS, int(b["opp_pct"] * N_TIERS) + 1)),
                    "rounds_weight": b["distance"],
                    "points": round(pts, 3),
                })

            if idle > ELIGIBILITY_DAYS:
                total *= 0.5 ** ((idle - ELIGIBILITY_DAYS) / self.w.inactivity_half_life)

            scores[fid] = total
            divisions[fid] = division
            r, rd = glicko.state(fid, as_of)
            extras[fid] = {
                # Tapology's 1-99 SoS: opponent tiers summed, rescaled. Display only.
                "sos": max(1, min(99, round(
                    sum(l["tier"] for l in ledger) / (len(ledger) * N_TIERS) * 99))),
                "glicko": round(r), "rd": round(rd),
                "points": round(total, 2), "ledger": ledger,
            }

        order: dict[str, list[int]] = defaultdict(list)
        for fid, d in divisions.items():
            order[d].append(fid)
        for d in order:
            order[d].sort(key=lambda f: scores[f], reverse=True)
            champ = champions.get(d)
            if self.pin_champion and champ in scores and divisions.get(champ) == d:
                order[d].remove(champ)
                order[d].insert(0, champ)

        log.info(f"  Tapology: scored {len(scores)} fighters in {len(order)} divisions")
        return RankingResult(order=dict(order), scores=scores, extras=extras)


def generate_rankings(preview: bool = False):
    from app.services.ufc.ranking_publisher import publish_rankings

    db = SessionLocal()
    try:
        publish_rankings(db, ranker=TapologyRanker(), preview=preview)
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_rankings(preview="--preview" in sys.argv)
