"""
Fighter Rankings — Points + Elo System

Simple, interpretable ranking system inspired by Fight Matrix and Tapology.

Two phases:
1. Elo backbone: Run Elo over all UFC history with method-of-victory adjustments.
   This gives every fighter a "true strength" estimate that implicitly encodes SOS.
2. Point scoring: Score each fighter's last 6 UFC fights based on:
   - Method of victory (finish >> decision)
   - Opponent Elo quality (beating good fighters = more points)
   - Recency (recent fights weighted more)
   - Loss penalty (scaled by method and opponent quality)

SOS (Strength of Schedule) is computed from opponent Elo percentiles and displayed
alongside rankings but does NOT feed back into the score (avoids double-counting).

Run: python -m app.services.points_ranking_service
     python -m app.services.points_ranking_service --preview
"""

from __future__ import annotations

import json
import logging
import math
from bisect import bisect_right
from collections import defaultdict
from datetime import date

from app.database import SessionLocal
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter, UFCFighterRanking,
)
from app.services.ufc.fighter_registry import (
    Eligibility, classify_weight_class, is_decided, is_rankable,
)

log = logging.getLogger("points_ranking")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Bouts scored for points. Was 6, which compressed a 36-fight career into the same
#: window as a 6-fight hot streak: Charles Oliveira's window held two losses while his
#: defining wins fell outside it, so he scored 17.9 fight-points against Quillan
#: Salkilld's 27.0 for a 6-0 run. Positions beyond RECENCY_WEIGHTS get the 0.3 tail
#: weight, so the extra bouts contribute without dominating.
FIGHTS_WINDOW = 10

#: Eligibility lives in fighter_registry so Glicko and Points cannot disagree about who
#: is rankable — that disagreement is what left rank=0 rows on the site. Re-exported here
#: only for the `/ufc/rankings` payload.
MIN_FIGHTS = Eligibility.min_decided_fights
INACTIVITY_DAYS = Eligibility.max_days_inactive

# Elo parameters
ELO_START = 1500
ELO_K_BASE = 40
ELO_K_NEWCOMER = 60         # higher K for first few fights (faster calibration)
ELO_NEWCOMER_FIGHTS = 5

# Method-of-victory K multipliers for Elo (finishes reveal more information)
ELO_METHOD_K = {
    "early_finish": 1.4,
    "late_finish": 1.2,
    "ud": 1.0,
    "majority": 0.9,
    "split": 0.75,
    "decision": 0.95,
}

# Win points by method category (base, before opponent quality multiplier)
WIN_POINTS = {
    "early_finish": 5.0,    # KO/Sub in rounds 1-2
    "late_finish": 4.0,     # KO/Sub in rounds 3+
    "ud": 3.0,              # Unanimous decision
    "majority": 2.5,        # Majority decision
    "split": 2.0,           # Split decision
    "decision": 2.5,        # Generic decision fallback
}

# Loss penalty: base * method_mult * opponent_factor * recency
LOSS_PENALTY_BASE = -3.0
LOSS_METHOD_MULT = {
    "early_finish": 1.5,    # getting stopped early hurts most
    "late_finish": 1.2,
    "ud": 0.9,
    "majority": 0.8,
    "split": 0.7,
    "decision": 0.85,
}

#: Bounds on how much the opponent's quality can soften a loss.
#: The old factor was `max(2.0 - opp_mult, 0.3)`, which INVERTED with quality: the better
#: the opponent, the smaller the penalty, bottoming out at 0.3. Combined with a -1.5 base
#: and a 0.4 split multiplier, a split-decision loss to an elite cost 0.18 points while a
#: finish over one paid 9.50 — a 50:1 asymmetry that made fighting nearly risk-free and
#: turned activity into a one-way ratchet. Losing to a great fighter should still hurt
#: less, just not 50x less; test_points_scoring pins the ratio at <= 3.
LOSS_OPP_FACTOR_MAX = 1.15   # vs the weakest opposition
LOSS_OPP_FACTOR_MIN = 0.70   # vs the strongest

# Recency multipliers by fight position (index 0 = most recent fight)
RECENCY_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.55, 0.4]

# Context multipliers
TITLE_MULT = 1.3
FIVE_ROUND_MULT = 1.1

#: Career-strength bonus, added to the summed fight points.
ELO_BONUS_MAX = 35.0

#: Elo above ELO_START that earns the full bonus. This term used to be scaled by the
#: fighter's Elo PERCENTILE among all eligible fighters, which saturates: with 563
#: eligible, every contender sits between the 94th and 100th percentile, so the bonus was
#: a near-constant 33-35 across an entire top 15 and could not separate anyone in it.
#: Oliveira's 97-Elo lead over Salkilld converted to 1.93 points of score against a +9.1
#: fight-points gap — so career quality was effectively absent and the ranking was decided
#: by recent form alone. 400 is the Elo scale's 10:1 odds interval, so ~1900 earns the cap.
ELO_LINEAR_SPREAD = 400.0

#: Retained only so ranking_baselines can reconstruct the tried-and-reverted variants
#: for comparison. Not used by the shipping scorer — see the note in PointsEloRanker.
MIN_DIVISOR = sum(RECENCY_WEIGHTS[:5])   # 3.95
PRIOR_STRENGTH = 3.0
ELO_PRIOR_SCALE = 6.0

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Kept as an alias so existing callers (preview_service) keep working. The registry is
#: now the authority on divisions; this must not diverge from it again.
_classify_weight_class = classify_weight_class


def _classify_method(method: str, finish_round: int | None) -> str:
    """Classify fight outcome into a scoring category."""
    if not method:
        return "decision"
    if "KO" in method or "TKO" in method or "Submission" in method or "Sub" in method:
        if finish_round and finish_round <= 2:
            return "early_finish"
        return "late_finish"
    if "Split" in method:
        return "split"
    if "Majority" in method:
        return "majority"
    if "Unanimous" in method:
        return "ud"
    if "Decision" in method:
        return "decision"
    return "decision"


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _elo_percentile(target_elo: float, sorted_active_elos: list[float]) -> float:
    """Fraction of active fighters at or below `target_elo`.

    `sorted_active_elos` must be sorted; bisect replaces a linear scan that ran inside
    the per-fight loop (~13M comparisons per full run).
    """
    if not sorted_active_elos:
        return 0.5
    return bisect_right(sorted_active_elos, target_elo) / len(sorted_active_elos)


def _quality_from_percentile(pct: float) -> float:
    """Opponent-quality multiplier (0.3 – 2.0), top-heavy."""
    return 0.3 + 1.7 * (pct ** 1.3)


def _loss_opponent_factor(opp_mult: float) -> float:
    """How much the opponent's quality softens a loss.

    Monotone decreasing in quality — losing to a better fighter still hurts less — but
    bounded, unlike the old `max(2.0 - opp_mult, 0.3)` which drove the penalty toward
    zero exactly where the competition was hardest.
    """
    span = LOSS_OPP_FACTOR_MAX - LOSS_OPP_FACTOR_MIN
    frac = (opp_mult - 0.3) / 1.7          # opp_mult is 0.3..2.0 -> 0..1
    return LOSS_OPP_FACTOR_MAX - span * max(0.0, min(1.0, frac))


def _inactivity_factor(days_since_last: int) -> float:
    """Score multiplier based on inactivity. Starts decaying after 180 days."""
    if days_since_last <= 180:
        return 1.0
    excess = days_since_last - 180
    return math.exp(-0.0005 * excess)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class PointsEloRanker:
    """Elo backbone + points over a recent-fight window.

    Ordering only. It no longer loads its own fighter metadata or writes rows: division
    and activity come from the registry so it cannot disagree with the Glicko profiles,
    and ranking_publisher owns persistence.
    """

    name = "points_elo"

    def rank(self, db, registry: dict, as_of: date,
             crit: Eligibility | None = None) -> "RankingResult":
        from app.services.ufc.ranking_publisher import RankingResult

        crit = crit or Eligibility()
        log.info("  Points+Elo: loading fights...")
        fights = (
            db.query(UFCFight)
            .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
            .order_by(UFCFight.date, UFCFight.id)
            .all()
        )

        # ---- Phase 1: Elo over all decided UFC history ----------------------
        elo: dict[int, float] = defaultdict(lambda: ELO_START)
        fight_count: dict[int, int] = defaultdict(int)
        fighter_fights: dict[int, list[dict]] = defaultdict(list)

        for f in fights:
            if not f.date or f.date > as_of:
                continue
            if not is_decided(f.method, f.winner_id):
                continue
            if classify_weight_class(f.weight_class) == "unknown":
                continue

            red_id, blue_id = f.red_fighter_id, f.blue_fighter_id
            is_title = "title" in (f.weight_class or "").lower()
            is_5rd = bool(f.time_format and f.time_format.count("-") >= 4)
            method_cat = _classify_method(f.method or "", f.finish_round)

            red_elo_pre, blue_elo_pre = elo[red_id], elo[blue_id]
            exp_red = _elo_expected(red_elo_pre, blue_elo_pre)

            # K per fighter. Using min() over BOTH corners meant a 30-fight veteran
            # facing a debutant also updated at the newcomer rate, so the prospect rose
            # fast AND the veteran's rating was destabilised by the same bout.
            method_k = ELO_METHOD_K.get(method_cat, 1.0)
            k_red = (ELO_K_NEWCOMER if fight_count[red_id] < ELO_NEWCOMER_FIGHTS
                     else ELO_K_BASE) * method_k
            k_blue = (ELO_K_NEWCOMER if fight_count[blue_id] < ELO_NEWCOMER_FIGHTS
                      else ELO_K_BASE) * method_k

            red_won = f.winner_id == red_id
            elo[red_id] += k_red * ((1.0 if red_won else 0.0) - exp_red)
            elo[blue_id] += k_blue * ((0.0 if red_won else 1.0) - (1.0 - exp_red))

            fight_count[red_id] += 1
            fight_count[blue_id] += 1

            for me, opp, opp_elo_pre in ((red_id, blue_id, blue_elo_pre),
                                         (blue_id, red_id, red_elo_pre)):
                fighter_fights[me].append({
                    "date": f.date, "opponent_id": opp,
                    "won": f.winner_id == me, "method_cat": method_cat,
                    "opp_elo": opp_elo_pre, "is_title": is_title, "is_5rd": is_5rd,
                })

        # ---- Phase 2: point scoring over the recent window ------------------
        eligible = [fid for fid, st in registry.items()
                    if is_rankable(st, as_of, crit) and fid in fighter_fights]
        active_elos = sorted(elo[f] for f in eligible)

        scores: dict[int, float] = {}
        extras: dict[int, dict] = {}

        for fid in eligible:
            recent = sorted(fighter_fights[fid], key=lambda x: x["date"],
                            reverse=True)[:FIGHTS_WINDOW]
            if not recent:
                continue

            total, weight_sum, opp_pcts = 0.0, 0.0, []
            for i, fight in enumerate(recent):
                recency = RECENCY_WEIGHTS[i] if i < len(RECENCY_WEIGHTS) else 0.3
                opp_pct = _elo_percentile(elo[fight["opponent_id"]], active_elos)
                opp_mult = _quality_from_percentile(opp_pct)
                opp_pcts.append(opp_pct)

                ctx = TITLE_MULT if fight["is_title"] else (
                    FIVE_ROUND_MULT if fight["is_5rd"] else 1.0)

                if fight["won"]:
                    value = WIN_POINTS.get(fight["method_cat"], 2.5) * opp_mult * ctx
                else:
                    value = (LOSS_PENALTY_BASE
                             * LOSS_METHOD_MULT.get(fight["method_cat"], 1.0)
                             * _loss_opponent_factor(opp_mult) * ctx)
                total += value * recency
                weight_sum += recency

            # The OWGR-style divisor floor and the Bayesian shrinkage prior were both
            # tried here and BOTH are reverted. ranking_eval measured each change
            # against the shipped scoring on 1440 walk-forward bouts: paired accuracy
            # deltas of -0.0069 (divisor) and -0.0056 (prior), CIs straddling zero, and
            # `prior` had the worst Brier and log-loss of any variant. Neither earned
            # its place, and the job they were meant to do — stop unproven fighters
            # topping a division — is done properly by WHR's posterior uncertainty
            # rather than by a constant tuned here. Sum, as it shipped.
            form = total
            elo_strength = max(0.0, min(1.0, (elo[fid] - ELO_START) / ELO_LINEAR_SPREAD))
            blended = form + elo_strength * ELO_BONUS_MAX

            # Smooth decay toward the hard cutoff, so ring rust is a gradient rather than
            # a cliff at 548 days. Measured from last ACTIVITY, so a no-contest counts.
            days_idle = (as_of - registry[fid].last_activity).days
            scores[fid] = blended * _inactivity_factor(days_idle)

            extras[fid] = {
                "sos": max(1, min(99, round((sum(opp_pcts) / len(opp_pcts)) * 99))),
                "elo": round(elo[fid]),
                "points": round(scores[fid], 2),
            }

        # ---- Phase 3: division orderings ------------------------------------
        order: dict[str, list[int]] = {}
        for fid in scores:
            order.setdefault(registry[fid].division, []).append(fid)
        for division in order:
            order[division].sort(key=lambda f: scores[f], reverse=True)

        # ---- P4P: z-score within division, then pooled ----------------------
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
                pool = sorted(z, key=lambda f: z[f], reverse=True)[:25]
                order[p4p_key] = pool
                for f in pool:
                    scores.setdefault(f, 0.0)

        log.info(f"  Points+Elo: scored {len(scores)} fighters in {len(order)} divisions")
        return RankingResult(order=order, scores=scores, extras=extras)


def generate_rankings(preview: bool = False):
    """Backwards-compatible entry point; publishing lives in ranking_publisher."""
    from app.services.ufc.ranking_publisher import publish_rankings

    db = SessionLocal()
    try:
        publish_rankings(db, ranker=PointsEloRanker(), preview=preview)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Read rankings (same interface as ranking_service.get_rankings)
# ---------------------------------------------------------------------------
DIMENSIONS = [
    "pts", "ko", "kod", "sub", "subd",
    "td", "tdd", "ctrl",
    "str_vol", "str_acc", "str_def",
    "dist", "clinch", "gnd",
    "durability",
]


def get_rankings() -> dict:
    """Read rankings from DB (compatible with existing frontend)."""
    db = SessionLocal()
    try:
        rankings = (
            db.query(UFCFighterRanking, UFCFighter)
            .join(UFCFighter, UFCFighterRanking.fighter_id == UFCFighter.id)
            # rank=0 is ranking_service's placeholder, written before this service
            # assigns real ranks. If it fails, or ranks a different set of fighters
            # than Glicko profiled, those rows survive — and 0 sorts ahead of 1, so
            # they land at the TOP of every division. Never serve them.
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
                "elo": profile.get("elo", 0),
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
            "method": "points_elo",
            "dimensions": DIMENSIONS,
            "min_fights": MIN_FIGHTS,
        }
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    preview = "--preview" in sys.argv
    generate_rankings(preview=preview)
