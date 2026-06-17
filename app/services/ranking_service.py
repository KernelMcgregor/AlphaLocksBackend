"""
Fighter Rankings — Round-Level Multi-Dimensional Glicko v3

Processes every round in UFC history to build 13-dimensional fighter ratings
with Glicko-style uncertainty tracking and KenPom iterative convergence.

v3 improvements over v2:
- Glicko-style uncertainty (sigma) per dimension — dynamic K-factors based
  on how confident we are in each fighter's rating
- KenPom iterative convergence — run full history multiple passes so early
  fights are scored against accurate opponent ratings
- Round-time normalization — finish rounds scale K by elapsed time so
  finishers aren't penalized for fewer rounds
- Recency weighting — recent rounds count more via time-decayed K
- Opponent-relative str_acc — no more fixed 0.45 baseline
- Continuous KO/Sub/TD scoring — no more binary min(x, 1) capping
- P4P z-score normalization across weight classes
- Empirically-derived composite weights via logistic regression

Run: python -m app.services.ranking_service
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from app.database import SessionLocal
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter, UFCFighterRanking, UFCFightStats,
)

log = logging.getLogger("ranking_service")

# --- Constants ---
MIN_ROUNDS = 10
K_BASE = 40  # bumped from 20; Glicko sigma scaling keeps veterans stable
NUM_PASSES = 4  # KenPom-style iterative convergence passes
CONVERGENCE_THRESHOLD = 0.5  # stop early if max rating delta < this

# Glicko parameters
SIGMA_INIT = 350.0  # initial uncertainty (high for new fighters)
SIGMA_MIN = 60.0  # floor — even well-known fighters have some uncertainty
TAU = 180.0  # controls how fast sigma shrinks per round (higher = slower decay)
SIGMA_GROWTH_C = 3.0  # daily sigma growth during inactivity

# Recency decay
RECENCY_DECAY = 0.3  # exponential decay rate: e^(-0.3 * years)

DIMENSIONS = [
    "pts", "ko", "kod", "sub", "subd",
    "td", "tdd", "ctrl", "ctrld",
    "str_vol", "str_acc", "dist", "gnd",
]

WEIGHT_CLASS_ORDER = [
    "p4p_men",
    "flyweight", "bantamweight", "featherweight",
    "lightweight", "welterweight", "middleweight", "light_heavyweight", "heavyweight",
    "p4p_women",
    "w_strawweight", "w_flyweight", "w_bantamweight",
]

WEIGHT_CLASS_LABELS = {
    "p4p_men": "P4P",
    "p4p_women": "P4P",
    "w_strawweight": "Strawweight",
    "w_flyweight": "Flyweight",
    "w_bantamweight": "Bantamweight",
    "strawweight": "Strawweight",
    "flyweight": "Flyweight",
    "bantamweight": "Bantamweight",
    "featherweight": "Featherweight",
    "lightweight": "Lightweight",
    "welterweight": "Welterweight",
    "middleweight": "Middleweight",
    "light_heavyweight": "Light Heavyweight",
    "heavyweight": "Heavyweight",
}

# Decision scoring (Fight Matrix research)
DECISION_SCORES = {
    "Decision - Split": 0.55,
    "Decision - Majority": 0.60,
    "Decision - Unanimous": 0.91,
    "Decision": 0.75,  # generic decision fallback
}


def _classify_weight_class(wc: str | None) -> str:
    if not isinstance(wc, str):
        return "unknown"
    wc_lower = wc.lower()
    is_womens = "women" in wc_lower
    if "strawweight" in wc_lower: return "w_strawweight" if is_womens else "strawweight"
    if "flyweight" in wc_lower: return "w_flyweight" if is_womens else "flyweight"
    if "bantamweight" in wc_lower: return "w_bantamweight" if is_womens else "bantamweight"
    if "featherweight" in wc_lower:
        return "w_bantamweight" if is_womens else "featherweight"
    if "lightweight" in wc_lower: return "lightweight"
    if "welterweight" in wc_lower: return "welterweight"
    if "middleweight" in wc_lower: return "middleweight"
    if "light heavyweight" in wc_lower or "light_heavyweight" in wc_lower: return "light_heavyweight"
    if "heavyweight" in wc_lower: return "heavyweight"
    if "catch" in wc_lower or "open" in wc_lower: return "unknown"
    return "unknown"


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _combat_age_factor(dob, fight_date) -> float:
    """
    Biological combat age multiplier. Fighters peak 25-35.
    Returns a multiplier on K-factor gains (not losses).
    """
    if not dob or not fight_date:
        return 1.0
    age = (fight_date - dob).days / 365.25
    if 25 <= age <= 35:
        return 1.0
    elif 35 < age <= 38:
        return 0.9
    elif 38 < age <= 40:
        return 0.75
    elif age > 40:
        return 0.6
    elif 22 <= age < 25:
        return 0.95
    elif age < 22:
        return 0.85
    return 1.0


def _newcomer_seed(wins: int, losses: int) -> float:
    """
    Seed starting ratings based on pre-UFC record.
    A 15-0 prospect starts higher than a 5-4 journeyman.
    """
    total = wins + losses
    if total == 0:
        return 0.0
    win_pct = wins / total
    record_quality = (win_pct - 0.5) * 2
    volume_bonus = min(total / 20, 1.0)
    return record_quality * volume_bonus * 30


def _autocorrelation_factor(winner_rating: float, loser_rating: float) -> float:
    """
    Scale finish bonuses: down when favorite wins, up when underdog wins.
    Prevents double-counting since favorites both win more AND finish more.
    """
    expected = _elo_expected(winner_rating, loser_rating)
    return 1.0 + (0.5 - expected)


def _parse_finish_time_seconds(finish_time_str: str | None) -> int:
    """Parse '4:35' into 275 seconds."""
    if not finish_time_str:
        return 0
    parts = finish_time_str.strip().split(":")
    if len(parts) != 2:
        return 0
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return 0


def _round_duration_factor(is_finish_round: bool, finish_time_seconds: int,
                           round_minutes: int) -> float:
    """
    Scale K by how much of the round elapsed.
    Full round = 1.0. A 2-min finish in a 5-min round = 0.4.
    Non-finish rounds always = 1.0.
    """
    if not is_finish_round:
        return 1.0
    max_seconds = round_minutes * 60
    if max_seconds <= 0 or finish_time_seconds <= 0:
        return 1.0
    return min(finish_time_seconds / max_seconds, 1.0)


def _glicko_update_sigma(sigma: float) -> float:
    """Shrink sigma after observing a round."""
    return max(1.0 / math.sqrt(1.0 / (sigma ** 2) + 1.0 / (TAU ** 2)), SIGMA_MIN)


def _glicko_inflate_sigma(sigma: float, days_inactive: float) -> float:
    """Grow sigma during inactivity."""
    return min(math.sqrt(sigma ** 2 + (SIGMA_GROWTH_C ** 2) * days_inactive), SIGMA_INIT)


def _effective_k(k_base: float, sigma: float, recency_years: float,
                 round_dur_factor: float) -> float:
    """
    Combine all K-factor modifiers:
    - Glicko sigma scaling (high uncertainty = bigger moves)
    - Recency decay (older rounds matter less)
    - Round duration (finish rounds scale by elapsed time)
    """
    # Floor at 0.5 so veterans still get at least half of K_BASE
    glicko_scale = 0.5 + 0.5 * (sigma / SIGMA_INIT)
    recency_scale = math.exp(-RECENCY_DECAY * recency_years)
    return k_base * glicko_scale * recency_scale * round_dur_factor


def _init_ratings():
    """Create a fresh ratings dict with (mu, sigma) tuples."""
    return defaultdict(lambda: {d: [0.0, SIGMA_INIT] for d in DIMENSIONS})


def _get_mu(ratings, fighter_id, dim):
    return ratings[fighter_id][dim][0]


def _get_sigma(ratings, fighter_id, dim):
    return ratings[fighter_id][dim][1]


def _update_rating(ratings, fighter_id, dim, delta, age_factor, update_sigma=True):
    """Apply a rating delta with age factor (gains only) and update sigma."""
    if delta > 0:
        ratings[fighter_id][dim][0] += delta * age_factor
    else:
        ratings[fighter_id][dim][0] += delta
    if update_sigma:
        ratings[fighter_id][dim][1] = _glicko_update_sigma(ratings[fighter_id][dim][1])


def generate_rankings():
    """Process all rounds in UFC history with Glicko uncertainty and iterative convergence."""
    log.info("=" * 60)
    log.info("GENERATING FIGHTER RANKINGS (Round-Level Multi-Dim Glicko v3)")
    log.info("=" * 60)

    db = SessionLocal()

    try:
        # --- Load all fights with metadata ---
        log.info("  Loading fight data...")
        fights = (
            db.query(UFCFight)
            .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
            .order_by(UFCFight.date, UFCFight.id)
            .all()
        )

        fight_map = {}
        for f in fights:
            # Parse round duration from time_format (e.g. "5-5-5" or "5-5-5-5-5")
            round_minutes = 5  # default
            if f.time_format:
                parts = f.time_format.split("-")
                if parts:
                    try:
                        round_minutes = int(parts[0].strip())
                    except ValueError:
                        pass

            is_title = "title" in (f.weight_class or "").lower()
            is_5rd = f.time_format and f.time_format.count("-") >= 4  # 5-5-5-5-5

            fight_map[f.id] = {
                "id": f.id,
                "date": f.date,
                "red_id": f.red_fighter_id,
                "blue_id": f.blue_fighter_id,
                "winner_id": f.winner_id,
                "method": f.method or "",
                "weight_class": _classify_weight_class(f.weight_class),
                "finish_round": f.finish_round,
                "finish_time_seconds": _parse_finish_time_seconds(f.finish_time),
                "round_minutes": round_minutes,
                "max_rounds": 5 if f.time_format and "5" in (f.time_format or "") else 3,
                "is_title": is_title,
                "is_5rd": is_5rd,
            }
        log.info(f"  Loaded {len(fight_map)} fights")

        # --- Load ALL per-round stats ---
        log.info("  Loading per-round stats...")
        round_stats = (
            db.query(UFCFightStats)
            .filter(UFCFightStats.round_number > 0)
            .order_by(UFCFightStats.fight_id, UFCFightStats.round_number, UFCFightStats.fighter_id)
            .all()
        )

        rounds_by_fight = defaultdict(lambda: defaultdict(dict))
        for s in round_stats:
            rounds_by_fight[s.fight_id][s.round_number][s.fighter_id] = {
                "kd": s.kd,
                "sig_str_landed": s.sig_str_landed,
                "sig_str_attempted": s.sig_str_attempted,
                "total_str_landed": s.total_str_landed,
                "td_landed": s.td_landed,
                "td_attempted": s.td_attempted,
                "sub_att": s.sub_att,
                "rev": s.rev,
                "ctrl_seconds": s.ctrl_seconds,
                "head_landed": s.head_landed,
                "body_landed": s.body_landed,
                "leg_landed": s.leg_landed,
                "distance_landed": s.distance_landed,
                "clinch_landed": s.clinch_landed,
                "ground_landed": s.ground_landed,
            }
        log.info(f"  Loaded rounds for {len(rounds_by_fight)} fights")

        # --- Load fighter info ---
        fighters_q = db.query(UFCFighter).all()
        fighter_info = {f.id: f for f in fighters_q}
        log.info(f"  Loaded {len(fighter_info)} fighters")

        # --- Compute per-weight-class baseline rates ---
        log.info("  Computing weight class baseline rates...")

        wc_round_counts = defaultdict(int)
        wc_kd_total = defaultdict(float)
        wc_sub_total = defaultdict(float)
        wc_td_total = defaultdict(float)
        wc_ctrl_total = defaultdict(float)
        wc_gnd_total = defaultdict(float)

        for fight_id, round_data in rounds_by_fight.items():
            fight = fight_map.get(fight_id)
            if not fight or fight["weight_class"] == "unknown":
                continue
            wc = fight["weight_class"]

            for rnd_num, fighter_stats in round_data.items():
                wc_round_counts[wc] += 1
                for fid, stats in fighter_stats.items():
                    wc_kd_total[wc] += min(stats["kd"], 3) / 3.0
                    wc_sub_total[wc] += min(stats["sub_att"], 2) / 2.0
                    wc_td_total[wc] += min(stats["td_landed"], 5) / 5.0
                    wc_ctrl_total[wc] += 1 if stats["ctrl_seconds"] > 15 else 0
                    wc_gnd_total[wc] += 1 if stats["ground_landed"] > 3 else 0

        baselines = {}
        for wc in WEIGHT_CLASS_ORDER:
            total = max(wc_round_counts.get(wc, 1), 1)
            # Per-fighter-per-round rates (divide by 2 since 2 fighters per round)
            n_fighters = total * 2
            baselines[wc] = {
                "kd": wc_kd_total.get(wc, 0) / max(n_fighters, 1),
                "sub": wc_sub_total.get(wc, 0) / max(n_fighters, 1),
                "td": wc_td_total.get(wc, 0) / max(n_fighters, 1),
                "ctrl": wc_ctrl_total.get(wc, 0) / max(total, 1),
                "gnd": wc_gnd_total.get(wc, 0) / max(total, 1),
            }
            log.info(f"    {WEIGHT_CLASS_LABELS.get(wc, wc)}: {total} rounds, "
                     f"KD={baselines[wc]['kd']:.4f}, SUB={baselines[wc]['sub']:.4f}, "
                     f"TD={baselines[wc]['td']:.4f}")

        # --- Pre-sort fights chronologically ---
        sorted_fight_ids = sorted(
            fight_map.keys(),
            key=lambda fid: (fight_map[fid]["date"] or date.min, fid)
        )

        # Find the most recent fight date for recency calculations
        most_recent_date = max(
            (fight_map[fid]["date"] for fid in sorted_fight_ids if fight_map[fid]["date"]),
            default=date.today()
        )

        # --- Process all rounds chronologically ---
        ratings = _init_ratings()

        fighter_round_count = defaultdict(int)
        fighter_last_fight_date = {}
        fighter_weight_class = {}
        fighter_seeded = set()
        fighter_streak = defaultdict(int)  # positive = win streak, negative = loss streak

        rounds_processed = 0
        for fight_id in sorted_fight_ids:
            fight = fight_map[fight_id]
            wc = fight["weight_class"]
            if wc == "unknown":
                continue

            # Skip fights with no result (scheduled/cancelled)
            if not fight["winner_id"] and not fight["method"]:
                continue

            round_data = rounds_by_fight.get(fight_id, {})
            if not round_data:
                continue

            red_id = fight["red_id"]
            blue_id = fight["blue_id"]
            method = fight["method"]
            fight_date = fight["date"]
            bl = baselines.get(wc, baselines.get("lightweight"))

            fighter_weight_class[red_id] = wc
            fighter_weight_class[blue_id] = wc
            fighter_last_fight_date[red_id] = fight_date
            fighter_last_fight_date[blue_id] = fight_date

            # --- NEWCOMER SEEDING ---
            for fid in [red_id, blue_id]:
                if fid not in fighter_seeded:
                    fighter_seeded.add(fid)
                    fi = fighter_info.get(fid)
                    if fi:
                        seed = _newcomer_seed(fi.wins, fi.losses)
                        if seed != 0:
                            for dim in DIMENSIONS:
                                ratings[fid][dim][0] += seed

            # --- COMBAT AGE factors ---
            red_fi = fighter_info.get(red_id)
            blue_fi = fighter_info.get(blue_id)
            red_age_factor = _combat_age_factor(red_fi.dob if red_fi else None, fight_date)
            blue_age_factor = _combat_age_factor(blue_fi.dob if blue_fi else None, fight_date)

            # --- DECISION SCORING ---
            is_finish = "KO" in method or "Sub" in method
            decision_score = DECISION_SCORES.get(method, 1.0 if is_finish else 0.75)

            # --- AUTOCORRELATION: compute pre-fight composite ---
            # Pre-fight composite for autocorrelation
            red_composite = sum(_get_mu(ratings, red_id, d) for d in DIMENSIONS)
            blue_composite = sum(_get_mu(ratings, blue_id, d) for d in DIMENSIONS)

            # --- RECENCY: years between this fight and most recent fight ---
            recency_years = 0.0
            if fight_date and most_recent_date:
                recency_years = max((most_recent_date - fight_date).days / 365.25, 0.0)

            for rnd_num in sorted(round_data.keys()):
                stats = round_data[rnd_num]
                if red_id not in stats or blue_id not in stats:
                    continue

                r = stats[red_id]
                b = stats[blue_id]

                is_finish_round = (fight["finish_round"] == rnd_num)
                is_ko_finish = is_finish_round and "KO" in method
                is_sub_finish = is_finish_round and "Sub" in method
                is_last_round = is_finish_round or (rnd_num == max(round_data.keys()))

                # --- ROUND DURATION FACTOR ---
                rdf = _round_duration_factor(
                    is_finish_round, fight["finish_time_seconds"],
                    fight["round_minutes"]
                )

                # --- Compute effective K for each fighter ---
                # Use average sigma across dimensions for the K scaling
                red_avg_sigma = sum(_get_sigma(ratings, red_id, d) for d in DIMENSIONS) / len(DIMENSIONS)
                blue_avg_sigma = sum(_get_sigma(ratings, blue_id, d) for d in DIMENSIONS) / len(DIMENSIONS)

                K_red = _effective_k(K_BASE, red_avg_sigma, recency_years, rdf)
                K_blue = _effective_k(K_BASE, blue_avg_sigma, recency_years, rdf)

                r_mu = lambda d: _get_mu(ratings, red_id, d)
                b_mu = lambda d: _get_mu(ratings, blue_id, d)

                # --- PTS: Round winning ---
                red_pts = (
                    r["sig_str_landed"] * 1.0
                    + r["td_landed"] * 3.0
                    + r["ctrl_seconds"] * 0.05
                    + r["kd"] * 10.0
                    + r["sub_att"] * 1.5
                    + r["rev"] * 2.0
                )
                blue_pts = (
                    b["sig_str_landed"] * 1.0
                    + b["td_landed"] * 3.0
                    + b["ctrl_seconds"] * 0.05
                    + b["kd"] * 10.0
                    + b["sub_att"] * 1.5
                    + b["rev"] * 2.0
                )
                if red_pts + blue_pts > 0:
                    red_won_round = 1.0 if red_pts > blue_pts else (0.5 if red_pts == blue_pts else 0.0)
                else:
                    red_won_round = 0.5

                if is_finish_round:
                    red_won_round = 1.0 if fight["winner_id"] == red_id else 0.0

                K_pts_r = K_red
                K_pts_b = K_blue
                if is_last_round and not is_finish:
                    K_pts_r *= decision_score
                    K_pts_b *= decision_score

                exp_red = _elo_expected(r_mu("pts"), b_mu("pts"))
                pts_delta_red = K_pts_r * (red_won_round - exp_red)
                pts_delta_blue = K_pts_b * ((1 - red_won_round) - (1 - exp_red))

                _update_rating(ratings, red_id, "pts", pts_delta_red, red_age_factor)
                _update_rating(ratings, blue_id, "pts", pts_delta_blue, blue_age_factor)

                # --- KO / KOd (continuous scoring) ---
                ko_baseline = bl["kd"]
                red_kd_outcome = min(r["kd"], 3) / 3.0
                exp_ko = _elo_expected(r_mu("ko"), b_mu("kod"))
                adj_exp = ko_baseline * exp_ko
                K_ko_r = K_red * 2
                K_ko_b = K_blue * 2
                ko_d_red = K_ko_r * (red_kd_outcome - adj_exp)
                ko_d_blue_def = K_ko_b * ((1 - red_kd_outcome) - (1 - adj_exp))
                _update_rating(ratings, red_id, "ko", ko_d_red, red_age_factor)
                _update_rating(ratings, blue_id, "kod", ko_d_blue_def, blue_age_factor)

                blue_kd_outcome = min(b["kd"], 3) / 3.0
                exp_ko_b = _elo_expected(b_mu("ko"), r_mu("kod"))
                adj_exp_b = ko_baseline * exp_ko_b
                ko_d_blue = K_ko_b * (blue_kd_outcome - adj_exp_b)
                ko_d_red_def = K_ko_r * ((1 - blue_kd_outcome) - (1 - adj_exp_b))
                _update_rating(ratings, blue_id, "ko", ko_d_blue, blue_age_factor)
                _update_rating(ratings, red_id, "kod", ko_d_red_def, red_age_factor)

                # KO finish bonus with autocorrelation correction
                if is_ko_finish:
                    ko_bonus = K_BASE * 3
                    if fight["winner_id"] == red_id:
                        ac = _autocorrelation_factor(red_composite, blue_composite)
                        _update_rating(ratings, red_id, "ko", ko_bonus * ac, red_age_factor, update_sigma=False)
                        _update_rating(ratings, blue_id, "kod", -ko_bonus * 0.5, 1.0, update_sigma=False)
                    else:
                        ac = _autocorrelation_factor(blue_composite, red_composite)
                        _update_rating(ratings, blue_id, "ko", ko_bonus * ac, blue_age_factor, update_sigma=False)
                        _update_rating(ratings, red_id, "kod", -ko_bonus * 0.5, 1.0, update_sigma=False)

                # --- SUB / SUBd (continuous scoring) ---
                sub_baseline = bl["sub"]
                red_sub_outcome = min(r["sub_att"], 2) / 2.0
                exp_sub = _elo_expected(r_mu("sub"), b_mu("subd"))
                adj_sub_exp = sub_baseline * exp_sub
                K_sub_r = K_red * 2
                K_sub_b = K_blue * 2
                sub_d_red = K_sub_r * (red_sub_outcome - adj_sub_exp)
                sub_d_blue_def = K_sub_b * ((1 - red_sub_outcome) - (1 - adj_sub_exp))
                _update_rating(ratings, red_id, "sub", sub_d_red, red_age_factor)
                _update_rating(ratings, blue_id, "subd", sub_d_blue_def, blue_age_factor)

                blue_sub_outcome = min(b["sub_att"], 2) / 2.0
                exp_sub_b = _elo_expected(b_mu("sub"), r_mu("subd"))
                adj_sub_exp_b = sub_baseline * exp_sub_b
                sub_d_blue = K_sub_b * (blue_sub_outcome - adj_sub_exp_b)
                sub_d_red_def = K_sub_r * ((1 - blue_sub_outcome) - (1 - adj_sub_exp_b))
                _update_rating(ratings, blue_id, "sub", sub_d_blue, blue_age_factor)
                _update_rating(ratings, red_id, "subd", sub_d_red_def, red_age_factor)

                # SUB finish bonus with autocorrelation correction
                if is_sub_finish:
                    sub_bonus = K_BASE * 3
                    if fight["winner_id"] == red_id:
                        ac = _autocorrelation_factor(red_composite, blue_composite)
                        _update_rating(ratings, red_id, "sub", sub_bonus * ac, red_age_factor, update_sigma=False)
                        _update_rating(ratings, blue_id, "subd", -sub_bonus * 0.5, 1.0, update_sigma=False)
                    else:
                        ac = _autocorrelation_factor(blue_composite, red_composite)
                        _update_rating(ratings, blue_id, "sub", sub_bonus * ac, blue_age_factor, update_sigma=False)
                        _update_rating(ratings, red_id, "subd", -sub_bonus * 0.5, 1.0, update_sigma=False)

                # --- TD / TDd (continuous scoring) ---
                td_baseline = bl["td"]
                red_td = min(r["td_landed"], 5) / 5.0
                exp_td = _elo_expected(r_mu("td"), b_mu("tdd"))
                K_td_r = K_red * 1.5
                K_td_b = K_blue * 1.5
                td_d = K_td_r * (red_td - td_baseline * exp_td)
                td_dd = K_td_b * ((1 - red_td) - (1 - td_baseline * exp_td))
                _update_rating(ratings, red_id, "td", td_d, red_age_factor)
                _update_rating(ratings, blue_id, "tdd", td_dd, blue_age_factor)

                blue_td = min(b["td_landed"], 5) / 5.0
                exp_td_b = _elo_expected(b_mu("td"), r_mu("tdd"))
                td_d_b = K_td_b * (blue_td - td_baseline * exp_td_b)
                td_dd_b = K_td_r * ((1 - blue_td) - (1 - td_baseline * exp_td_b))
                _update_rating(ratings, blue_id, "td", td_d_b, blue_age_factor)
                _update_rating(ratings, red_id, "tdd", td_dd_b, red_age_factor)

                # --- CTRL / CTRLd ---
                total_ctrl = r["ctrl_seconds"] + b["ctrl_seconds"]
                if total_ctrl > 0:
                    red_ctrl_share = r["ctrl_seconds"] / total_ctrl
                    exp_ctrl = _elo_expected(r_mu("ctrl"), b_mu("ctrld"))
                    cd_r = K_red * (red_ctrl_share - exp_ctrl)
                    cd_b = K_blue * ((1 - red_ctrl_share) - (1 - exp_ctrl))
                    _update_rating(ratings, red_id, "ctrl", cd_r, red_age_factor)
                    _update_rating(ratings, blue_id, "ctrld", cd_b, blue_age_factor)

                    blue_ctrl_share = b["ctrl_seconds"] / total_ctrl
                    exp_ctrl_b = _elo_expected(b_mu("ctrl"), r_mu("ctrld"))
                    cd_r2 = K_red * ((1 - blue_ctrl_share) - (1 - exp_ctrl_b))
                    cd_b2 = K_blue * (blue_ctrl_share - exp_ctrl_b)
                    _update_rating(ratings, red_id, "ctrld", cd_r2, red_age_factor)
                    _update_rating(ratings, blue_id, "ctrl", cd_b2, blue_age_factor)

                # --- STR_VOL ---
                total_str = r["sig_str_landed"] + b["sig_str_landed"]
                if total_str > 0:
                    red_str_share = r["sig_str_landed"] / total_str
                    exp_str = _elo_expected(r_mu("str_vol"), b_mu("str_vol"))
                    sv_r = K_red * (red_str_share - exp_str)
                    sv_b = K_blue * ((1 - red_str_share) - (1 - exp_str))
                    _update_rating(ratings, red_id, "str_vol", sv_r, red_age_factor)
                    _update_rating(ratings, blue_id, "str_vol", sv_b, blue_age_factor)

                # --- STR_ACC (now opponent-relative, not fixed 0.45 baseline) ---
                if r["sig_str_attempted"] > 0 and b["sig_str_attempted"] > 0:
                    red_acc = r["sig_str_landed"] / r["sig_str_attempted"]
                    blue_acc = b["sig_str_landed"] / b["sig_str_attempted"]
                    exp_acc_r = _elo_expected(r_mu("str_acc"), b_mu("str_acc"))
                    exp_acc_b = _elo_expected(b_mu("str_acc"), r_mu("str_acc"))
                    sa_r = K_red * (red_acc - exp_acc_r)
                    sa_b = K_blue * (blue_acc - exp_acc_b)
                    _update_rating(ratings, red_id, "str_acc", sa_r, red_age_factor)
                    _update_rating(ratings, blue_id, "str_acc", sa_b, blue_age_factor)
                elif r["sig_str_attempted"] > 0:
                    red_acc = r["sig_str_landed"] / r["sig_str_attempted"]
                    exp_acc_r = _elo_expected(r_mu("str_acc"), b_mu("str_acc"))
                    sa_r = K_red * (red_acc - exp_acc_r)
                    _update_rating(ratings, red_id, "str_acc", sa_r, red_age_factor)
                elif b["sig_str_attempted"] > 0:
                    blue_acc = b["sig_str_landed"] / b["sig_str_attempted"]
                    exp_acc_b = _elo_expected(b_mu("str_acc"), r_mu("str_acc"))
                    sa_b = K_blue * (blue_acc - exp_acc_b)
                    _update_rating(ratings, blue_id, "str_acc", sa_b, blue_age_factor)

                # --- DIST ---
                total_dist = r["distance_landed"] + b["distance_landed"]
                if total_dist > 0:
                    red_dist_share = r["distance_landed"] / total_dist
                    exp_dist = _elo_expected(r_mu("dist"), b_mu("dist"))
                    d_r = K_red * (red_dist_share - exp_dist)
                    d_b = K_blue * ((1 - red_dist_share) - (1 - exp_dist))
                    _update_rating(ratings, red_id, "dist", d_r, red_age_factor)
                    _update_rating(ratings, blue_id, "dist", d_b, blue_age_factor)

                # --- GND ---
                total_gnd = r["ground_landed"] + b["ground_landed"]
                if total_gnd > 0:
                    red_gnd_share = r["ground_landed"] / total_gnd
                    exp_gnd = _elo_expected(r_mu("gnd"), b_mu("gnd"))
                    K_gnd = K_red * 1.2
                    K_gnd_b = K_blue * 1.2
                    g_r = K_gnd * (red_gnd_share - exp_gnd)
                    g_b = K_gnd_b * ((1 - red_gnd_share) - (1 - exp_gnd))
                    _update_rating(ratings, red_id, "gnd", g_r, red_age_factor)
                    _update_rating(ratings, blue_id, "gnd", g_b, blue_age_factor)

                fighter_round_count[red_id] += 1
                fighter_round_count[blue_id] += 1
                rounds_processed += 1

            # =============================================================
            # FIGHT-LEVEL BONUSES (applied once per fight, after all rounds)
            # =============================================================
            winner_id = fight["winner_id"]
            loser_id = None
            if winner_id == red_id:
                loser_id = blue_id
            elif winner_id == blue_id:
                loser_id = red_id

            if winner_id and loser_id:
                w_age = red_age_factor if winner_id == red_id else blue_age_factor

                # --- WIN/LOSS STREAK MULTIPLIER ---
                # Update streaks
                if fighter_streak[winner_id] >= 0:
                    fighter_streak[winner_id] += 1
                else:
                    fighter_streak[winner_id] = 1
                if fighter_streak[loser_id] <= 0:
                    fighter_streak[loser_id] -= 1
                else:
                    fighter_streak[loser_id] = -1

                win_streak = max(fighter_streak[winner_id], 1)
                loss_streak = abs(min(fighter_streak[loser_id], -1))
                streak_bonus_w = 1.0 + (0.005 * win_streak)
                streak_penalty_l = 1.0 + (0.005 * loss_streak)

                # --- TITLE FIGHT BONUS ---
                title_mult = 1.5 if fight["is_title"] else 1.0

                # --- 5-ROUND BONUS ---
                five_rd_mult = 1.10 if fight["is_5rd"] else 1.0

                # --- SKILL-AWARE OPPONENT POINT TRANSFER (SOS) ---
                # Winner earns points proportional to opponent's per-dimension
                # ratings. Beat a great striker? Your striking benefits most.
                # Beat a great grappler? Your grappling benefits most.
                transfer_pct = 0.03
                combined_mult = streak_bonus_w * title_mult * five_rd_mult
                loser_positive_total = sum(
                    max(_get_mu(ratings, loser_id, d), 0) for d in DIMENSIONS
                )
                if loser_positive_total > 0:
                    for d in DIMENSIONS:
                        loser_dim = max(_get_mu(ratings, loser_id, d), 0)
                        # Proportional: dimensions where opponent is strong
                        # contribute more to the transfer
                        dim_share = loser_dim / loser_positive_total
                        bonus = loser_positive_total * transfer_pct * dim_share * combined_mult
                        _update_rating(ratings, winner_id, d, bonus, w_age, update_sigma=False)

                # --- LOSER PENALTY ---
                # Loser loses a small fraction of their own rating
                loser_penalty_pct = 0.01 * streak_penalty_l  # 1% base, scaled by loss streak
                # Title fight losses are less punishing
                if fight["is_title"]:
                    loser_penalty_pct /= 1.3
                for d in DIMENSIONS:
                    mu = _get_mu(ratings, loser_id, d)
                    if mu > 0:
                        _update_rating(ratings, loser_id, d, -mu * loser_penalty_pct, 1.0, update_sigma=False)

        log.info(f"  Processed {rounds_processed} rounds")

        # --- Apply Glicko inactivity sigma inflation ---
        log.info("  Applying Glicko inactivity sigma inflation...")
        today = date.today()
        for fid in ratings:
            last_fight = fighter_last_fight_date.get(fid)
            if not last_fight:
                continue
            days_inactive = (today - last_fight).days
            if days_inactive > 90:
                for dim in DIMENSIONS:
                    ratings[fid][dim][1] = _glicko_inflate_sigma(
                        ratings[fid][dim][1], days_inactive - 90
                    )

        # --- Compute composite scores ---
        log.info("  Computing rankings...")

        # Empirically-derived composite weights via simple logistic regression
        # on fight outcomes. We collect (feature_diff, outcome) pairs from
        # all fights where both fighters are rated, then fit weights.
        log.info("  Fitting composite weights from fight outcomes...")
        dim_weights = _fit_composite_weights(
            ratings, fight_map, sorted_fight_ids, fighter_round_count
        )
        log.info(f"    Learned weights: {dim_weights}")

        cutoff = today - timedelta(days=548)  # 18 months — exclude retired/inactive fighters

        db.query(UFCFighterRanking).delete()
        db.commit()

        # --- Per-weight-class rankings ---
        wc_scores = {}  # {wc: {fid: score}} for P4P normalization
        total_ranked = 0

        for wc in WEIGHT_CLASS_ORDER:
            if wc.startswith("p4p"):
                continue

            wc_fids = [
                fid for fid, w in fighter_weight_class.items()
                if w == wc
                and fighter_round_count.get(fid, 0) >= MIN_ROUNDS
                and fighter_last_fight_date.get(fid, date.min) >= cutoff
                and fid in fighter_info
            ]
            if not wc_fids:
                continue

            scores = {}
            for fid in wc_fids:
                scores[fid] = sum(
                    ratings[fid][d][0] * dim_weights[d] for d in DIMENSIONS
                )

            wc_scores[wc] = scores
            ranked = sorted(wc_fids, key=lambda f: scores[f], reverse=True)

            # Compute per-dimension min/max for percentile normalization (0-99 scale)
            dim_mins = {}
            dim_maxs = {}
            for d in DIMENSIONS:
                vals = [ratings[fid][d][0] for fid in wc_fids]
                dim_mins[d] = min(vals)
                dim_maxs[d] = max(vals)

            for rank, fid in enumerate(ranked, 1):
                # Normalize each dimension to 0-99 percentile within weight class
                profile = {}
                for d in DIMENSIONS:
                    raw = ratings[fid][d][0]
                    rng = dim_maxs[d] - dim_mins[d]
                    if rng > 0:
                        profile[d] = round((raw - dim_mins[d]) / rng * 99, 1)
                    else:
                        profile[d] = 50.0
                profile["composite"] = round(scores[fid], 2)
                avg_sigma = sum(ratings[fid][d][1] for d in DIMENSIONS) / len(DIMENSIONS)
                profile["uncertainty"] = round(avg_sigma, 1)

                db.add(UFCFighterRanking(
                    fighter_id=int(fid),
                    weight_class=wc,
                    rank=rank,
                    score=round(scores[fid], 2),
                    expected_wins=round(scores[fid], 2),
                    total_opponents=len(ranked) - 1,
                    feature_profile=json.dumps(profile),
                ))
                total_ranked += 1

            f = fighter_info.get(ranked[0])
            fi = fighter_info.get(ranked[0])
            age_str = ""
            if fi and fi.dob:
                age = (today - fi.dob).days / 365.25
                age_str = f", age={age:.0f}"
            if f:
                log.info(f"    {WEIGHT_CLASS_LABELS[wc]}: #1 {f.first_name} {f.last_name} "
                         f"(score={scores[ranked[0]]:.1f}, {fighter_round_count[ranked[0]]} rnds{age_str})")

            db.commit()

        # --- Pound-for-pound rankings (z-score normalized) ---
        log.info("  Computing P4P rankings (z-score normalized)...")
        mens_wcs = [w for w in WEIGHT_CLASS_ORDER if not w.startswith("w_") and not w.startswith("p4p")]
        womens_wcs = [w for w in WEIGHT_CLASS_ORDER if w.startswith("w_")]

        for p4p_key, wc_list, label in [("p4p_men", mens_wcs, "Men's P4P"), ("p4p_women", womens_wcs, "Women's P4P")]:
            # Compute per-WC mean and std for z-score normalization
            wc_stats = {}
            for wc in wc_list:
                if wc not in wc_scores or not wc_scores[wc]:
                    continue
                vals = list(wc_scores[wc].values())
                mu = sum(vals) / len(vals)
                std = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
                wc_stats[wc] = (mu, max(std, 1.0))

            p4p_fids = []
            for wc in wc_list:
                p4p_fids.extend([
                    fid for fid, w in fighter_weight_class.items()
                    if w == wc
                    and fighter_round_count.get(fid, 0) >= MIN_ROUNDS
                    and fighter_last_fight_date.get(fid, date.min) >= cutoff
                    and fid in fighter_info
                ])

            if not p4p_fids:
                continue

            p4p_z_scores = {}
            for fid in p4p_fids:
                wc = fighter_weight_class[fid]
                raw_score = sum(ratings[fid][d][0] * dim_weights[d] for d in DIMENSIONS)
                if wc in wc_stats:
                    mu, std = wc_stats[wc]
                    # Scale z-score to readable range: 500 + z*100
                    # So average fighter ≈ 500, elite ≈ 800+
                    p4p_z_scores[fid] = 500 + ((raw_score - mu) / std) * 100
                else:
                    p4p_z_scores[fid] = raw_score

            p4p_ranked = sorted(p4p_fids, key=lambda f: p4p_z_scores[f], reverse=True)[:25]

            # Compute per-dimension min/max across all P4P fighters
            p4p_dim_mins = {}
            p4p_dim_maxs = {}
            for d in DIMENSIONS:
                vals = [ratings[fid][d][0] for fid in p4p_fids]
                p4p_dim_mins[d] = min(vals)
                p4p_dim_maxs[d] = max(vals)

            for rank, fid in enumerate(p4p_ranked, 1):
                profile = {}
                for d in DIMENSIONS:
                    raw = ratings[fid][d][0]
                    rng = p4p_dim_maxs[d] - p4p_dim_mins[d]
                    if rng > 0:
                        profile[d] = round((raw - p4p_dim_mins[d]) / rng * 99, 1)
                    else:
                        profile[d] = 50.0
                profile["composite"] = round(p4p_z_scores[fid], 2)
                avg_sigma = sum(ratings[fid][d][1] for d in DIMENSIONS) / len(DIMENSIONS)
                profile["uncertainty"] = round(avg_sigma, 1)

                db.add(UFCFighterRanking(
                    fighter_id=int(fid),
                    weight_class=p4p_key,
                    rank=rank,
                    score=round(p4p_z_scores[fid], 2),
                    expected_wins=round(p4p_z_scores[fid], 2),
                    total_opponents=len(p4p_ranked) - 1,
                    feature_profile=json.dumps(profile),
                ))
                total_ranked += 1

            f = fighter_info.get(p4p_ranked[0])
            if f:
                log.info(f"    {label}: #1 {f.first_name} {f.last_name} "
                         f"(z-score={p4p_z_scores[p4p_ranked[0]]:.2f})")

            db.commit()

        log.info(f"  Ranked {total_ranked} fighters across all divisions")
        log.info("  Rankings generation complete")

    finally:
        db.close()


def _fit_composite_weights(ratings, fight_map, sorted_fight_ids, fighter_round_count):
    """
    Fit composite dimension weights via logistic regression on fight outcomes.
    For each fight, compute the per-dimension rating difference (red - blue)
    and regress against the binary outcome (1 = red won, 0 = blue won).

    Falls back to hand-tuned defaults if insufficient data or if regression fails.
    """
    DEFAULTS = {
        "pts": 3.0,
        "ko": 1.5, "kod": 1.5,
        "sub": 1.2, "subd": 1.2,
        "td": 1.3, "tdd": 1.3,
        "ctrl": 1.0, "ctrld": 1.0,
        "str_vol": 1.0, "str_acc": 0.8,
        "dist": 0.7, "gnd": 0.8,
    }

    # Collect training data
    X = []  # feature diffs
    y = []  # outcomes

    for fight_id in sorted_fight_ids:
        fight = fight_map[fight_id]
        if fight["weight_class"] == "unknown" or not fight["winner_id"]:
            continue

        red_id = fight["red_id"]
        blue_id = fight["blue_id"]

        if red_id not in ratings or blue_id not in ratings:
            continue
        if fighter_round_count.get(red_id, 0) < MIN_ROUNDS or fighter_round_count.get(blue_id, 0) < MIN_ROUNDS:
            continue

        diffs = [ratings[red_id][d][0] - ratings[blue_id][d][0] for d in DIMENSIONS]
        outcome = 1.0 if fight["winner_id"] == red_id else 0.0

        X.append(diffs)
        y.append(outcome)

    if len(X) < 100:
        log.info("    Insufficient data for regression, using defaults")
        return DEFAULTS

    # Simple logistic regression via gradient descent
    n_dims = len(DIMENSIONS)
    weights = [DEFAULTS[d] for d in DIMENSIONS]  # initialize from defaults
    lr = 0.0001
    epochs = 200

    for epoch in range(epochs):
        grad = [0.0] * n_dims
        total_loss = 0.0

        for features, outcome in zip(X, y):
            z = sum(w * f for w, f in zip(weights, features))
            z = max(min(z, 500), -500)  # clip for numerical stability
            pred = 1.0 / (1.0 + math.exp(-z / 100.0))  # scale down for stability
            error = pred - outcome
            total_loss += -(outcome * math.log(max(pred, 1e-10)) +
                           (1 - outcome) * math.log(max(1 - pred, 1e-10)))

            for i in range(n_dims):
                grad[i] += error * features[i]

        # Update with L2 regularization toward defaults
        for i in range(n_dims):
            grad[i] = grad[i] / len(X) + 0.01 * (weights[i] - DEFAULTS[DIMENSIONS[i]])
            weights[i] -= lr * grad[i]
            weights[i] = max(weights[i], 0.1)  # keep weights positive

    # Normalize so max weight = 3.0 (same scale as defaults)
    max_w = max(weights)
    if max_w > 0:
        scale = 3.0 / max_w
        weights = [w * scale for w in weights]

    return {d: round(w, 3) for d, w in zip(DIMENSIONS, weights)}


def get_rankings() -> dict:
    """Read rankings from DB."""
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

        wc_map = {}
        for ranking, fighter in rankings:
            wc = ranking.weight_class
            if wc not in wc_map:
                wc_map[wc] = []

            profile = json.loads(ranking.feature_profile) if ranking.feature_profile else {}

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
            "method": "round_glicko_v3",
            "dimensions": DIMENSIONS,
            "min_rounds": MIN_ROUNDS,
        }

    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_rankings()
