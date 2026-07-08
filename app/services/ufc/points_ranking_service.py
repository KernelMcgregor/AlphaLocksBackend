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
from collections import defaultdict
from datetime import date, timedelta

from app.database import SessionLocal
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter, UFCFighterRanking,
)

log = logging.getLogger("points_ranking")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGHTS_WINDOW = 6           # last N UFC fights scored for points
MIN_FIGHTS = 3              # minimum UFC fights to appear in rankings
INACTIVITY_DAYS = 548       # 18 months — max days since last fight

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
# Kept moderate — losing to elite opponents by decision barely hurts
LOSS_PENALTY_BASE = -1.5
LOSS_METHOD_MULT = {
    "early_finish": 1.5,    # getting stopped early hurts most
    "late_finish": 1.2,
    "ud": 0.8,
    "majority": 0.6,
    "split": 0.4,           # split-decision loss barely hurts
    "decision": 0.7,
}

# Recency multipliers by fight position (index 0 = most recent fight)
RECENCY_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.55, 0.4]

# Context multipliers
TITLE_MULT = 1.3
FIVE_ROUND_MULT = 1.1

# Elo strength bonus: established fighters get credit for career Elo
# This prevents rising prospects from leap-frogging proven veterans
# who happen to have a few decision wins recently
ELO_BONUS_MAX = 35.0        # max additional points from Elo standing

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

def _classify_weight_class(wc: str | None) -> str:
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


def _opponent_quality_mult(opp_elo: float, all_active_elos: list[float]) -> float:
    """
    Map opponent Elo to a quality multiplier (0.3 – 2.0).
    Top-heavy: beating elite opponents is worth much more than beating average ones.
    50th-pct opponent -> ~0.9x.  95th-pct -> ~1.9x.  10th-pct -> ~0.35x.
    """
    if not all_active_elos:
        return 1.0
    below = sum(1 for e in all_active_elos if e <= opp_elo)
    pct = below / len(all_active_elos)
    return 0.3 + 1.7 * (pct ** 1.3)


def _inactivity_factor(days_since_last: int) -> float:
    """Score multiplier based on inactivity. Starts decaying after 180 days."""
    if days_since_last <= 180:
        return 1.0
    excess = days_since_last - 180
    return math.exp(-0.0005 * excess)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_rankings(preview: bool = False):
    """Run the Points + Elo ranking system."""
    log.info("=" * 60)
    log.info("GENERATING FIGHTER RANKINGS (Points + Elo)")
    log.info("=" * 60)

    db = SessionLocal()
    try:
        # --- Load data -----------------------------------------------------------
        log.info("  Loading fight data...")
        fights = (
            db.query(UFCFight)
            .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
            .order_by(UFCFight.date, UFCFight.id)
            .all()
        )
        fighters_q = db.query(UFCFighter).all()
        fighter_info = {f.id: f for f in fighters_q}
        log.info(f"  Loaded {len(fights)} fights, {len(fighter_info)} fighters")

        today = date.today()
        cutoff = today - timedelta(days=INACTIVITY_DAYS)

        # =====================================================================
        # PHASE 1: Elo over all UFC history
        # =====================================================================
        log.info("  Phase 1: Computing Elo ratings...")
        elo: dict[int, float] = defaultdict(lambda: ELO_START)
        fight_count: dict[int, int] = defaultdict(int)
        fighter_weight_class: dict[int, str] = {}
        fighter_last_fight: dict[int, date] = {}
        fighter_fights: dict[int, list[dict]] = defaultdict(list)

        for f in fights:
            if not f.date or not f.winner_id:
                continue
            method = f.method or ""
            if "No Contest" in method or "DQ" in method or "Draw" in method:
                continue
            wc = _classify_weight_class(f.weight_class)
            if wc == "unknown":
                continue

            red_id, blue_id = f.red_fighter_id, f.blue_fighter_id
            is_title = "title" in (f.weight_class or "").lower()
            is_5rd = bool(f.time_format and f.time_format.count("-") >= 4)
            method_cat = _classify_method(method, f.finish_round)

            # Snapshot pre-fight Elo
            red_elo_pre = elo[red_id]
            blue_elo_pre = elo[blue_id]

            # Elo update
            exp_red = _elo_expected(red_elo_pre, blue_elo_pre)
            K = (ELO_K_NEWCOMER if min(fight_count[red_id], fight_count[blue_id]) < ELO_NEWCOMER_FIGHTS
                 else ELO_K_BASE)
            K *= ELO_METHOD_K.get(method_cat, 1.0)

            if f.winner_id == red_id:
                elo[red_id] += K * (1.0 - exp_red)
                elo[blue_id] += K * (exp_red - 1.0)
            else:
                elo[red_id] += K * (0.0 - exp_red)
                elo[blue_id] += K * (1.0 - (1.0 - exp_red))

            # Track metadata
            fighter_weight_class[red_id] = wc
            fighter_weight_class[blue_id] = wc
            fighter_last_fight[red_id] = f.date
            fighter_last_fight[blue_id] = f.date
            fight_count[red_id] += 1
            fight_count[blue_id] += 1

            # Record fight for red
            fighter_fights[red_id].append({
                "date": f.date,
                "opponent_id": blue_id,
                "won": f.winner_id == red_id,
                "method_cat": method_cat,
                "opp_elo": blue_elo_pre,
                "is_title": is_title,
                "is_5rd": is_5rd,
            })
            # Record fight for blue
            fighter_fights[blue_id].append({
                "date": f.date,
                "opponent_id": red_id,
                "won": f.winner_id == blue_id,
                "method_cat": method_cat,
                "opp_elo": red_elo_pre,
                "is_title": is_title,
                "is_5rd": is_5rd,
            })

        log.info(f"  Elo computed for {len(elo)} fighters")

        # Build global list of active-fighter Elos (for opponent quality percentiles)
        all_active_elos = sorted([
            elo[fid] for fid in fighter_weight_class
            if fight_count[fid] >= MIN_FIGHTS
            and fighter_last_fight.get(fid, date.min) >= cutoff
        ])

        # =====================================================================
        # PHASE 2: Point scoring on last N fights
        # =====================================================================
        log.info("  Phase 2: Computing point scores...")
        fighter_scores: dict[int, float] = {}
        fighter_sos: dict[int, int] = {}

        for fid in fighter_weight_class:
            if fight_count[fid] < MIN_FIGHTS:
                continue
            last = fighter_last_fight.get(fid)
            if not last or last < cutoff:
                continue

            recent = sorted(fighter_fights[fid], key=lambda x: x["date"], reverse=True)[:FIGHTS_WINDOW]
            if not recent:
                continue

            total_points = 0.0
            opp_elos = []

            for i, fight in enumerate(recent):
                recency = RECENCY_WEIGHTS[i] if i < len(RECENCY_WEIGHTS) else 0.3
                opp_mult = _opponent_quality_mult(fight["opp_elo"], all_active_elos)
                opp_elos.append(fight["opp_elo"])

                ctx = 1.0
                if fight["is_title"]:
                    ctx = TITLE_MULT
                elif fight["is_5rd"]:
                    ctx = FIVE_ROUND_MULT

                if fight["won"]:
                    base = WIN_POINTS.get(fight["method_cat"], 2.5)
                    total_points += base * opp_mult * recency * ctx
                else:
                    base = LOSS_PENALTY_BASE
                    loss_m = LOSS_METHOD_MULT.get(fight["method_cat"], 1.0)
                    # Losing to a strong opponent hurts less (inverted quality)
                    opp_loss = max(2.0 - opp_mult, 0.3)
                    total_points += base * loss_m * opp_loss * recency * ctx

            # Elo strength bonus: reward career strength
            if all_active_elos:
                below = sum(1 for e in all_active_elos if e <= elo[fid])
                elo_pct = below / len(all_active_elos)
                total_points += elo_pct * ELO_BONUS_MAX

            # Inactivity decay
            days_inactive = (today - fighter_last_fight[fid]).days
            total_points *= _inactivity_factor(days_inactive)

            fighter_scores[fid] = total_points

            # SOS: average opponent Elo percentile -> 1-99 scale
            if opp_elos and all_active_elos:
                avg_pct = sum(
                    sum(1 for e in all_active_elos if e <= oe) / len(all_active_elos)
                    for oe in opp_elos
                ) / len(opp_elos)
                fighter_sos[fid] = max(1, min(99, round(avg_pct * 99)))
            else:
                fighter_sos[fid] = 50

        log.info(f"  Scored {len(fighter_scores)} eligible fighters")

        # =====================================================================
        # PHASE 3: Division rankings
        # =====================================================================
        log.info("  Phase 3: Generating division rankings...")

        # Load existing Glicko profiles for radar charts (if available)
        existing = db.query(UFCFighterRanking).all()
        glicko_profiles = {(r.fighter_id, r.weight_class): r.feature_profile for r in existing}

        db.query(UFCFighterRanking).delete()
        db.commit()

        total_ranked = 0
        wc_ranked_scores: dict[str, dict[int, float]] = {}

        for wc in WEIGHT_CLASS_ORDER:
            if wc.startswith("p4p"):
                continue

            eligible = [
                fid for fid in fighter_scores
                if fighter_weight_class.get(fid) == wc and fid in fighter_info
            ]
            if len(eligible) < 2:
                continue

            ranked = sorted(eligible, key=lambda f: fighter_scores[f], reverse=True)
            wc_ranked_scores[wc] = {fid: fighter_scores[fid] for fid in ranked}

            # Normalize to 0-1000
            s_max = fighter_scores[ranked[0]]
            s_min = fighter_scores[ranked[-1]]
            s_range = s_max - s_min if s_max > s_min else 1.0

            if not preview:
                for rank, fid in enumerate(ranked, 1):
                    norm = round((fighter_scores[fid] - s_min) / s_range * 1000, 1)

                    try:
                        profile = json.loads(glicko_profiles.get((fid, wc), "{}") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        profile = {}
                    profile["sos"] = fighter_sos.get(fid, 50)
                    profile["elo"] = round(elo.get(fid, ELO_START))
                    profile["points"] = round(fighter_scores[fid], 1)

                    db.add(UFCFighterRanking(
                        fighter_id=int(fid),
                        weight_class=wc,
                        rank=rank,
                        score=norm,
                        expected_wins=norm,
                        total_opponents=len(ranked) - 1,
                        feature_profile=json.dumps(profile),
                    ))
                    total_ranked += 1
                db.commit()

            fi = fighter_info.get(ranked[0])
            if fi:
                log.info(
                    f"    {WEIGHT_CLASS_LABELS.get(wc, wc)}: "
                    f"#1 {fi.first_name} {fi.last_name} "
                    f"(pts={fighter_scores[ranked[0]]:.1f}, "
                    f"elo={elo[ranked[0]]:.0f}, "
                    f"sos={fighter_sos.get(ranked[0], 0)})"
                )

            if preview:
                print(f"\n=== {WEIGHT_CLASS_LABELS.get(wc, wc)} ===")
                for rank, fid in enumerate(ranked[:30], 1):
                    fi = fighter_info.get(fid)
                    if fi:
                        ufc_record = f"{fi.wins}-{fi.losses}"
                        if fi.draws:
                            ufc_record += f"-{fi.draws}"
                        print(
                            f"  #{rank:>2} {fi.first_name} {fi.last_name} "
                            f"({ufc_record}) "
                            f"pts={fighter_scores[fid]:>6.1f}  "
                            f"elo={elo[fid]:>6.0f}  "
                            f"sos={fighter_sos.get(fid, 0):>2}"
                        )

        # =====================================================================
        # P4P rankings (z-score normalized across weight classes)
        # =====================================================================
        log.info("  Computing P4P rankings...")
        mens_wcs = [w for w in WEIGHT_CLASS_ORDER
                    if not w.startswith("w_") and not w.startswith("p4p")]
        womens_wcs = [w for w in WEIGHT_CLASS_ORDER if w.startswith("w_")]

        for p4p_key, wc_list, label in [
            ("p4p_men", mens_wcs, "Men's P4P"),
            ("p4p_women", womens_wcs, "Women's P4P"),
        ]:
            wc_stats: dict[str, tuple[float, float]] = {}
            for wc in wc_list:
                vals = list(wc_ranked_scores.get(wc, {}).values())
                if len(vals) < 2:
                    continue
                mu = sum(vals) / len(vals)
                std = max((sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5, 0.1)
                wc_stats[wc] = (mu, std)

            p4p_fids = [
                fid for fid in fighter_scores
                if fighter_weight_class.get(fid) in wc_list and fid in fighter_info
            ]
            if len(p4p_fids) < 2:
                continue

            p4p_z: dict[int, float] = {}
            for fid in p4p_fids:
                wc = fighter_weight_class[fid]
                if wc in wc_stats:
                    mu, std = wc_stats[wc]
                    p4p_z[fid] = (fighter_scores[fid] - mu) / std
                else:
                    p4p_z[fid] = 0.0

            p4p_ranked = sorted(p4p_fids, key=lambda f: p4p_z[f], reverse=True)[:25]

            if p4p_ranked:
                z_max = max(p4p_z[f] for f in p4p_ranked)
                z_min = min(p4p_z[f] for f in p4p_ranked)
                z_range = z_max - z_min if z_max > z_min else 1.0
                p4p_norm = {f: round((p4p_z[f] - z_min) / z_range * 1000, 1)
                            for f in p4p_ranked}
            else:
                p4p_norm = {}

            if not preview:
                for rank, fid in enumerate(p4p_ranked, 1):
                    wc = fighter_weight_class.get(fid, "unknown")
                    try:
                        profile = json.loads(
                            glicko_profiles.get((fid, wc), "{}") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        profile = {}
                    profile["sos"] = fighter_sos.get(fid, 50)
                    profile["elo"] = round(elo.get(fid, ELO_START))

                    db.add(UFCFighterRanking(
                        fighter_id=int(fid),
                        weight_class=p4p_key,
                        rank=rank,
                        score=p4p_norm.get(fid, 0),
                        expected_wins=p4p_norm.get(fid, 0),
                        total_opponents=len(p4p_ranked) - 1,
                        feature_profile=json.dumps(profile),
                    ))
                    total_ranked += 1
                db.commit()

            fi = fighter_info.get(p4p_ranked[0]) if p4p_ranked else None
            if fi:
                log.info(f"    {label}: #1 {fi.first_name} {fi.last_name} "
                         f"(z={p4p_z[p4p_ranked[0]]:.2f})")

            if preview and p4p_ranked:
                print(f"\n=== {label} ===")
                for rank, fid in enumerate(p4p_ranked[:15], 1):
                    fi = fighter_info.get(fid)
                    if fi:
                        wc = fighter_weight_class.get(fid, "")
                        print(
                            f"  #{rank:>2} {fi.first_name} {fi.last_name} "
                            f"({fi.wins}-{fi.losses}) "
                            f"z={p4p_z[fid]:.2f} "
                            f"[{WEIGHT_CLASS_LABELS.get(wc, wc)}]"
                        )

        log.info(f"  Ranked {total_ranked} fighters across all divisions")
        log.info("  Points + Elo rankings complete")

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
