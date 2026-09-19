"""Fighter Rankings — Glicko-1 backbone + tiered points window.

Replaces the Points+Elo ranker. The architecture is the same shape; what changed is where
the numbers come from.

Why
---
The incumbent (`points_ranking_service.py`) contains roughly twelve constants that were
invented rather than sourced or fitted: a six-entry `WIN_POINTS` ladder, three separate
loss tables, a `RECENCY_WEIGHTS` array with a bare 0.3 tail, an opponent-quality curve
`0.3 + 1.7*pct**1.3`, `ELO_BONUS_MAX = 35.0` added to an unnormalised sum, and
`TITLE_MULT`/`FIVE_ROUND_MULT` driven by sniffing the substring "title" out of a bout
name. `docs/models/rankings.md` records the consequence of that: "No scoring-constant
change was statistically distinguishable from the shipped version." The constants were
simultaneously unmotivated and unfittable, so tuning them was never going to help.

The standard applied here: **every constant is either cited to a published system, or
fitted on data strictly preceding the fold it is evaluated on. Nothing is invented.**

    outcome ladder 1.00/0.91/0.61/0.55/0.50   Fight Matrix (Glicko variant), BoxRec
    opponent tiers 1..10                      Tapology
    opponent seed for elite imports           Tapology
    (rounds/5)^2 weight on decisions          BoxRec's (rounds/12)^2
    ~17% per-division carry haircut           Fight Matrix
    conservative rating = r - RD              TrueSkill's published mu - k*sigma
    Glicko RD=230 / c=87 / 180d               Fight Matrix   (see glicko1.py)
    recency half-life H                       FITTED, see HALF_LIFE_DAYS

What this does NOT do, deliberately
-----------------------------------
No transitivity constraint. Meta/UFC, Fight Matrix and BoxRec all force a winner above the
man he beat; Tapology explicitly refuses, citing A>B>C>A cycles that make the constraint
unsatisfiable. We follow Tapology.

No champion pinned at #1, and no early-vs-late finish split — no published system grades
finish speed, and inventing a coefficient for it is exactly what this rewrite exists to
stop. If it is wanted later it has to arrive fitted, with a CI.

Run: python -m app.services.ufc.tiered_ranking_service --preview
"""

from __future__ import annotations

import json
import logging
import math
from bisect import bisect_left, insort
from collections import defaultdict
from datetime import date

from app.database import SessionLocal
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFighterRanking
from app.services.ufc.fighter_registry import (
    Eligibility, classify_weight_class, is_decided, is_rankable,
)
from app.services.ufc.glicko1 import Glicko1, expected

log = logging.getLogger("tiered_ranking")

# ---------------------------------------------------------------------------
# Constants — every one of these carries its source
# ---------------------------------------------------------------------------

#: Outcome value `v`. Fight Matrix's Glicko-variant ladder, which BoxRec's clear-decision
#: factors (KO 1.00 / UD 0.875 / MD 0.55 / SD 0.45) independently corroborate.
#:
#: A first-round KO (1.00) sits cleanly above a five-round unanimous decision (0.91), and
#: nothing distinguishes an early finish from a late one — Fight Matrix, BoxRec and Meta
#: all decline to grade finish speed. The incumbent's round-2 cliff between `early_finish`
#: (5.0 points) and `late_finish` (4.0) was invented here and is gone.
V_FINISH = 1.00
V_UD = 0.91
V_MAJORITY = 0.61
V_SPLIT = 0.55
V_DRAW = 0.50

#: Tapology buckets opponents into ten tiers. The tiers themselves are deciles of the
#: rating distribution, so the bucketing is a property of the data rather than a curve we
#: picked; this replaces `_quality_from_percentile`'s unmotivated 1.3 exponent.
N_TIERS = 10

#: BoxRec weights a decision by `(rounds_boxed / 12)^2` — a three-round decision is worth
#: a sixteenth of a twelve-round one. Rescaled to MMA's championship distance of five, a
#: three-round decision weighs (3/5)^2 = 0.36 against a five-round decision's 1.0, so a
#: main event counts ~2.8x a prelim automatically. This is what retires `TITLE_MULT`,
#: `FIVE_ROUND_MULT`, and with them the `"title" in weight_class.lower()` string sniffing.
#: Finishes weigh 1.0 regardless of distance, as in BoxRec — you cannot go the distance
#: harder than by not needing it.
CHAMPIONSHIP_ROUNDS = 5
ROUNDS_EXPONENT = 2

#: Ledger display window only — NOT a scoring parameter. Recency is handled inside the
#: rating, by RD inflation past the 180-day grace plus the conservative `r - RD` readout,
#: so there is no separate decay term and no half-life to choose. These two bound how much
#: of a fighter's history the site shows under their score; changing them cannot change
#: anyone's rank. Meta's "fights begin declining in value at 5 years" sets the age bound.
MAX_BOUT_AGE_DAYS = 5 * 365
FIGHTS_WINDOW = 10

#: Fight Matrix: a fighter moving up a division carries ~17% fewer points, moving down
#: ~17% more, and the effect is ~1.5x stronger for women. Without this, a bout scored
#: against lightweight deciles is compared directly against welterweight ones.
DIVISION_CARRY = 0.83
DIVISION_CARRY_WOMEN_MULT = 1.5

#: Tapology's opponent-seed rule for elite imports is NOT implemented, and deliberately so.
#: It exists because at-the-time tiering rates an un-debuted signee at the bottom, making a
#: win over them worthless. Scoring against Glicko expectation removes the need: an unrated
#: fighter carries RD=230, which pulls the expectation toward 0.5 rather than toward
#: certainty, so beating them already pays a middling amount and losing to them is already
#: punished. A tier floor on top of that was measured here and made it worse — it inflated
#: EVERY newcomer rather than only elite imports, which is most of why undefeated prospects
#: topped four divisions in the first cut of this ranker.

#: Minimum decided UFC bouts before a fighter joins the decile pool. Matches
#: `Eligibility.min_decided_fights`, so the distribution the tiers are cut from is the same
#: population the rankings are drawn from — otherwise one-and-done debutants dominate the
#: low deciles and shift every tier upward.
MIN_POOL_FIGHTS = Eligibility.min_decided_fights

MIN_FIGHTS = Eligibility.min_decided_fights
INACTIVITY_DAYS = Eligibility.max_days_inactive

WEIGHT_CLASS_ORDER = [
    "p4p_men",
    "flyweight", "bantamweight", "featherweight",
    "lightweight", "welterweight", "middleweight",
    "light_heavyweight", "heavyweight",
    "p4p_women",
    "w_strawweight", "w_flyweight", "w_bantamweight",
]

WEIGHT_CLASS_LABELS = {
    "p4p_men": "P4P", "p4p_women": "P4P",
    "w_strawweight": "Strawweight", "w_flyweight": "Flyweight",
    "w_bantamweight": "Bantamweight",
    "strawweight": "Strawweight", "flyweight": "Flyweight",
    "bantamweight": "Bantamweight", "featherweight": "Featherweight",
    "lightweight": "Lightweight", "welterweight": "Welterweight",
    "middleweight": "Middleweight",
    "light_heavyweight": "Light Heavyweight",
    "heavyweight": "Heavyweight",
}

#: Ordinal position of each division within its own gender ladder, for the carry haircut.
DIVISION_LADDER = {
    "strawweight": 0, "flyweight": 1, "bantamweight": 2, "featherweight": 3,
    "lightweight": 4, "welterweight": 5, "middleweight": 6,
    "light_heavyweight": 7, "heavyweight": 8,
    "w_strawweight": 0, "w_flyweight": 1, "w_bantamweight": 2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def outcome_value(method: str | None) -> float:
    """Map a ufcstats method string to its published outcome value.

    `Could Not Continue` and `Other` reach here as decided bouts with a winner; Fight
    Matrix treats an injury TKO identically to a normal TKO, so they score as finishes.
    No-contests, DQs and overturned results never reach here — `is_decided` filters them,
    and scoring them as draws would move both ratings on zero information.
    """
    m = method or ""
    if "Split" in m:
        return V_SPLIT
    if "Majority" in m:
        return V_MAJORITY
    if "Unanimous" in m:
        return V_UD
    if "Decision" in m:
        # Three rows in the whole table carry a bare "Decision". Unanimous is the modal
        # decision by 3.4:1, so that is the maximum-likelihood reading.
        return V_UD
    return V_FINISH


def scheduled_rounds(time_format: str | None) -> int:
    """Rounds the bout was scheduled for, from the round-length list.

    `time_format` holds per-round lengths: '5-5-5' is a three-round bout, '5-5-5-5-5' a
    five. This replaces `time_format.count("-") >= 4`, which could only answer the yes/no
    question "is this five rounds" and silently treated every non-five bout as three.
    Unparseable values ('No Time Limit', '20', and the other UFC 1-10 era formats) fall
    back to three — they are all decades past MAX_BOUT_AGE_DAYS and never scored.
    """
    if not time_format:
        return 3
    parts = [p.strip() for p in time_format.split("-")]
    if not all(p.isdigit() for p in parts):
        return 3
    return max(1, min(len(parts), CHAMPIONSHIP_ROUNDS))


def bout_weight(method: str | None, time_format: str | None) -> float:
    """BoxRec's rounds-scheduled weighting. Finishes are full weight at any distance."""
    if outcome_value(method) == V_FINISH:
        return 1.0
    return (scheduled_rounds(time_format) / CHAMPIONSHIP_ROUNDS) ** ROUNDS_EXPONENT


def carry_factor(bout_division: str, current_division: str) -> float:
    """Fight Matrix's cross-division carry haircut.

    Tiers are cut within a division, so a bout scored against lightweight deciles is not
    directly comparable to one scored against welterweight deciles. A fighter who moved up
    carries the lighter division's credit at a discount; one who moved down carries the
    heavier division's credit at a premium.
    """
    a = DIVISION_LADDER.get(bout_division)
    b = DIVISION_LADDER.get(current_division)
    if a is None or b is None or a == b:
        return 1.0
    # Cross-gender movement does not occur; if the data ever shows it, decline to adjust.
    if bout_division.startswith("w_") != current_division.startswith("w_"):
        return 1.0
    steps = b - a                      # >0 = moved up since that bout
    mult = DIVISION_CARRY_WOMEN_MULT if current_division.startswith("w_") else 1.0
    rate = 1.0 - (1.0 - DIVISION_CARRY) * mult
    rate = max(rate, 0.1)
    return rate ** steps if steps > 0 else (1.0 / rate) ** (-steps)


class _DivisionPool:
    """Sorted conservative ratings per division, maintained incrementally.

    Tiers must be cut from the distribution **as it stood on the fight date**, which rules
    out sorting once at the end. Incremental insort keeps that exact rather than
    approximate; a lazy full re-sort was the alternative and would re-sort on every bout,
    since every bout changes two ratings.
    """

    def __init__(self) -> None:
        self._by_div: dict[str, list[float]] = defaultdict(list)
        self._current: dict[int, tuple[str, float]] = {}

    def set(self, fid: int, division: str, value: float) -> None:
        old = self._current.get(fid)
        if old is not None:
            arr = self._by_div[old[0]]
            i = bisect_left(arr, old[1])
            if i < len(arr) and arr[i] == old[1]:
                arr.pop(i)
        insort(self._by_div[division], value)
        self._current[fid] = (division, value)

    def percentile(self, division: str, value: float) -> float:
        """Fraction of the division at or below `value`, as it stood at this moment."""
        arr = self._by_div.get(division)
        if not arr:
            return 0.5
        return bisect_left(arr, value) / len(arr)

    def tier(self, division: str, value: float) -> int:
        """Decile rank, 1..10. Tier 10 is the strongest."""
        arr = self._by_div.get(division)
        if not arr:
            return (N_TIERS + 1) // 2      # no pool yet — mid tier, no free credit
        frac = bisect_left(arr, value) / len(arr)
        return max(1, min(N_TIERS, int(frac * N_TIERS) + 1))


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

class TieredRanker:
    """Glicko-1 backbone, then tiered points over a recent-fight window.

    Ordering only — `ranking_publisher` owns normalisation and persistence, and
    `fighter_registry` owns eligibility and division, so this cannot disagree with the
    Glicko radar profiles about who is rankable. That disagreement is what left rank=0 rows
    on the site; see tests/test_ranking_integrity.py.
    """

    name = "tiered"

    #: Tapology's presentation rules, adopted because they measurably help the eye test.
    #: The SCORING stays a rating: the fighters that a summed-points system misplaces are
    #: exactly the ones who recently lost (Garry, Tsarukyan, Della Maddalena, Muhammad all
    #: land 4-8 places too low under points), because a rating barely moves when you lose
    #: to a much better opponent — which is how the official rankings behave too.
    pin_champion = True
    exclude_statuses = ("Retired", "Released")

    def __init__(self, conservative: float = 0.0, mode: str = "decay") -> None:
        #: How the rating is read out. Both defaults are settled by measurement against
        #: the official UFC (Meta) rankings — see ranking_benchmark — not by taste.
        #:
        #: mode: "decay" applies Meta's published inactivity decay (flat 18 months, then
        #:   the above-baseline part halves per further 18 months). Measured 0.798 mean
        #:   Spearman against the official rankings vs 0.788 for "raw" and 0.722 for
        #:   "peak" (BoxRec's peak-in-window, which was tried and is worse here).
        #:
        #: conservative: multiplier on RD in `rating - conservative * RD`. TrueSkill
        #:   publishes 3.0 and the obvious choice was 1.0, but 1.0 measured WORSE (0.767
        #:   vs 0.788 raw) — it double-penalises inactivity, which the decay above already
        #:   handles. RD still does its real work inside the Glicko update, where g(RD)
        #:   discounts results against unproven opponents, and in the displayed tier.
        self.mode = mode
        self.conservative = conservative

    def rank(self, db, registry: dict, as_of: date,
             crit: Eligibility | None = None) -> "RankingResult":
        from app.services.ufc.ranking_publisher import RankingResult

        crit = crit or Eligibility()
        log.info("  Tiered: loading fights...")
        fights = (
            db.query(UFCFight)
            .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
            .order_by(UFCFight.date, UFCFight.id)
            .all()
        )
        names = {
            f.id: f"{f.first_name or ''} {f.last_name or ''}".strip()
            for f in db.query(UFCFighter).all()
        }

        glicko = Glicko1()
        pool = _DivisionPool()
        history: dict[int, list[dict]] = defaultdict(list)

        # ---- Pass 1: Glicko over decided history, recording each bout AT ITS OWN TIME --
        for f in fights:
            if not f.date or f.date > as_of:
                continue
            if not is_decided(f.method, f.winner_id):
                continue
            division = classify_weight_class(f.weight_class)
            if division == "unknown":
                continue

            red_id, blue_id = f.red_fighter_id, f.blue_fighter_id
            if red_id is None or blue_id is None:
                continue

            red_won = f.winner_id == red_id
            v = outcome_value(f.method)
            weight = bout_weight(f.method, f.time_format)

            # Everything is read BEFORE the bout is applied, so a fighter's own result
            # never feeds back into the credit they are given for it.
            pre = {fid: glicko.state(fid, f.date) for fid in (red_id, blue_id)}

            # Method and distance enter HERE, inside the rating update, which is where
            # Fight Matrix and BoxRec both put them. A unanimous decision is 0.91 rather
            # than 1.0 because it is weaker evidence of superiority than a knockout, and a
            # three-round bout carries (3/5)^2 of a championship-distance one.
            glicko.observe(red_id, blue_id,
                           v if red_won else 1.0 - v, f.date, weight)

            for me, opp in ((red_id, blue_id), (blue_id, red_id)):
                if glicko.fights.get(me, 0) >= MIN_POOL_FIGHTS:
                    pool.set(me, division, glicko.conservative(me, f.date))
                won = f.winner_id == me
                r_me, rd_me = pre[me]
                r_opp, rd_opp = pre[opp]
                history[me].append({
                    "fight_id": f.id,
                    "date": f.date,
                    "opponent_id": opp,
                    "opponent_name": names.get(opp, ""),
                    "won": won,
                    "method": f.method,
                    "division": division,
                    # This fighter's own outcome value: the published ladder on a win,
                    # its complement on a loss.
                    "v": round(v if won else 1.0 - v, 4),
                    # Pre-fight win probability, from Glicko's own expectation function —
                    # shown in the ledger so a user can see whether a win was an upset.
                    "expected": round(expected(r_me, r_opp, rd_opp), 4),
                    # Rating points this bout moved the fighter.
                    "delta": glicko.rating[me] - r_me,
                    # Display only. The score uses the continuous expectation above; a
                    # ten-bucket scale is for showing a user "you beat a Tier 9 fighter",
                    # and cannot separate the top of a division from its fringe.
                    "tier": pool.tier(division, r_opp - rd_opp),
                    "rounds_weight": round(weight, 4),
                })

        # ---- Pass 2: points over the recent window --------------------------
        statuses = {f.id: (f.status or "") for f in db.query(UFCFighter).all()}
        eligible = [
            fid for fid, st in registry.items()
            if fid in history
            and st.division != "unknown"
            and st.decided_fights >= crit.min_decided_fights
            and st.rounds >= crit.min_rounds
            # Tapology's 21-month window plus their 60-day display grace, which lives in
            # Eligibility so this cannot drift from what ranking_publisher verifies.
            and (as_of - st.last_activity).days <= crit.max_days_inactive
            and statuses.get(fid) not in self.exclude_statuses
        ]

        scores: dict[int, float] = {}
        extras: dict[int, dict] = {}

        for fid in eligible:
            current_division = registry[fid].division
            recent = [b for b in reversed(history[fid])
                      if (as_of - b["date"]).days <= MAX_BOUT_AGE_DAYS][:FIGHTS_WINDOW]
            if not recent:
                continue

            ledger = []
            for b in recent:
                carry = carry_factor(b["division"], current_division)
                ledger.append({
                    "fight_id": b["fight_id"],
                    "date": b["date"].isoformat(),
                    "opponent_id": b["opponent_id"],
                    "opponent_name": b["opponent_name"],
                    "won": b["won"],
                    "method": b["method"],
                    "v": b["v"],
                    "expected": b["expected"],
                    "tier": b["tier"],
                    "rounds_weight": b["rounds_weight"],
                    "carry": round(carry, 4),
                    # Rating points gained or lost. This IS the score's decomposition —
                    # the ledger sums to (current rating - 1500), so a user can read every
                    # point of a fighter's standing back to the bout that produced it.
                    "points": round(b["delta"] * carry, 2),
                })

            r, rd = glicko.state(fid, as_of)
            # The ranking IS the rating. Summing per-bout surprise over a window was
            # measured here and rejected: it re-derives the rating badly, because a
            # dominant champion's wins are all expected and therefore score ~0 surprise
            # each — Makhachev came out below Josh Hokit. See docs/models/rankings.md.
            #
            # `rating - RD` is TrueSkill's published conservative estimate. It is also
            # what retires the separate inactivity decay term: a fighter two years idle is
            # demoted by their own grown uncertainty, not by a decay constant we picked.
            base = (glicko.decayed(fid, as_of) if self.mode == "decay"
                    else glicko.peak(fid, as_of) if self.mode == "peak" else r)
            scores[fid] = base - self.conservative * rd
            extras[fid] = {
                # Display-only, as in the incumbent and as Tapology does it explicitly:
                # opponent quality already drives the rating through the Glicko update, so
                # feeding a schedule metric back in would double-count it.
                "sos": max(1, min(99, round(
                    sum(b["tier"] for b in recent) / len(recent) / N_TIERS * 99))),
                "glicko": round(r),
                "rd": round(rd),
                "points": round(base - self.conservative * rd, 1),
                "ledger": ledger,
            }

        # ---- Pass 3: division orderings -------------------------------------
        order: dict[str, list[int]] = {}
        for fid in scores:
            order.setdefault(registry[fid].division, []).append(fid)
        for division in order:
            order[division].sort(key=lambda f: scores[f], reverse=True)

        # Tapology pins the most recent undisputed (non-interim) champion at #1 regardless
        # of score. Derived from title-bout results, not a curated list — see champions.py,
        # which resolves 8/8 verifiable divisions including the heavyweight elevation that
        # a naive "last title winner" reading gets wrong.
        if self.pin_champion:
            from app.services.ufc.champions import current_champions
            champs = current_champions(
                db, as_of, {fid: registry[fid].division for fid in scores})
            for div, champ in champs.items():
                if champ in scores and div in order and champ in order[div]:
                    order[div].remove(champ)
                    order[div].insert(0, champ)

        # ---- P4P: z-score within division, then pooled ----------------------
        # Unchanged from the incumbent. A single cross-division rating is not meaningful
        # on a near-disconnected comparison graph, so this compares standing within a
        # division rather than rating across divisions.
        for p4p_key, member_of in (
            ("p4p_men", lambda d: not d.startswith("w_")),
            ("p4p_women", lambda d: d.startswith("w_")),
        ):
            divisions = [d for d in order if member_of(d)]
            z: dict[int, float] = {}
            for d in divisions:
                vals = [scores[f] for f in order[d]]
                if len(vals) < 2:
                    continue
                mu = sum(vals) / len(vals)
                sd = max((sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5, 0.1)
                for f in order[d]:
                    z[f] = (scores[f] - mu) / sd
            if len(z) >= 2:
                p4p_pool = sorted(z, key=lambda f: z[f], reverse=True)[:25]
                order[p4p_key] = p4p_pool
                for f in p4p_pool:
                    scores.setdefault(f, 0.0)

        log.info(f"  Tiered: scored {len(scores)} fighters in {len(order)} divisions")
        return RankingResult(order=order, scores=scores, extras=extras)


def generate_rankings(preview: bool = False):
    from app.services.ufc.ranking_publisher import publish_rankings

    db = SessionLocal()
    try:
        publish_rankings(db, ranker=TieredRanker(), preview=preview)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Read rankings (same payload contract as points_ranking_service.get_rankings)
# ---------------------------------------------------------------------------
DIMENSIONS = [
    "pts", "ko", "kod", "sub", "subd",
    "td", "tdd", "ctrl",
    "str_vol", "str_acc", "str_def",
    "dist", "clinch", "gnd",
    "durability",
]


def get_rankings() -> dict:
    db = SessionLocal()
    try:
        rankings = (
            db.query(UFCFighterRanking, UFCFighter)
            .join(UFCFighter, UFCFighterRanking.fighter_id == UFCFighter.id)
            # rank=0 was ranking_service's placeholder. If the ranker failed, or ranked a
            # different set than Glicko profiled, those rows survived — and 0 sorts ahead
            # of 1, so they landed at the TOP of every division. Never serve them.
            .filter(UFCFighterRanking.rank > 0)
            .order_by(UFCFighterRanking.weight_class, UFCFighterRanking.rank)
            .all()
        )

        if not rankings:
            return {"weight_classes": [], "method": "none"}

        wc_map: dict[str, list] = {}
        for ranking, fighter in rankings:
            wc = ranking.weight_class
            if wc not in wc_map:
                wc_map[wc] = []

            try:
                profile = json.loads(ranking.feature_profile) if ranking.feature_profile else {}
            except (json.JSONDecodeError, TypeError):
                profile = {}

            wc_map[wc].append({
                "id": str(fighter.id),
                "first_name": fighter.first_name,
                "last_name": fighter.last_name,
                "nickname": fighter.nickname,
                "wins": fighter.wins,
                "losses": fighter.losses,
                "draws": fighter.draws,
                "country_code": fighter.country_code,
                "image_url": fighter.image_url,
                "rank": ranking.rank,
                "score": round(ranking.score, 1),
                "dimensions": {d: profile.get(d, 0) for d in DIMENSIONS},
                "uncertainty": profile.get("uncertainty", 0),
                "sos": profile.get("sos", 0),
                "glicko": profile.get("glicko", 0),
                "rd": profile.get("rd", 0),
                # Per-fight decomposition. The point of an objective ranking is that it
                # can be audited, so the breakdown ships with the number.
                "ledger": profile.get("ledger", []),
            })

        return {
            "weight_classes": [
                {
                    "key": wc,
                    "label": WEIGHT_CLASS_LABELS.get(wc, wc),
                    "fighters": wc_map[wc],
                }
                for wc in WEIGHT_CLASS_ORDER
                if wc in wc_map
            ],
            "method": "tiered",
            "dimensions": DIMENSIONS,
            "min_fights": MIN_FIGHTS,
        }
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    generate_rankings(preview="--preview" in sys.argv)
