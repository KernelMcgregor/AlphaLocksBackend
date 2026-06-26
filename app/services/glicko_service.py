"""
Glicko Dimension Ratings — Round-Level Multi-Dimensional Glicko

Processes every round in UFC history to build 15-dimensional fighter ratings
with Glicko-style uncertainty tracking. These ratings serve two purposes:

1. **ML prediction features** — GlickoSnapshot records are the most predictive
   features in the winner prediction model (model.py).
2. **Radar chart dimensions** — Dimension profiles are saved to UFCFighterRanking
   by points_ranking_service.py for frontend display.

This module handles Glicko computation only. Rankings are in ranking_service.py.

Run standalone: python -m app.services.glicko_service
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date

from app.database import SessionLocal
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter, UFCFightStats,
    UFCGlickoSnapshot,
)

__all__ = ["DIMENSIONS", "GlickoParams", "compute_and_save_snapshots",
           "run_glicko_inmemory"]

log = logging.getLogger("glicko_service")

# ---------------------------------------------------------------------------
# DIMENSIONS
# ---------------------------------------------------------------------------
DIMENSIONS = [
    "pts", "ko", "kod", "sub", "subd",
    "td", "tdd", "ctrl",
    "str_vol", "str_acc", "str_def",
    "dist", "clinch", "gnd",
    "durability",
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

MIN_ROUNDS = 10


# ---------------------------------------------------------------------------
# TUNABLE PARAMETERS
# ---------------------------------------------------------------------------
@dataclass
class GlickoParams:
    """All tunable Glicko parameters. Tuned via Optuna (150 trials, 2026-06-26)."""
    k_base: float = 20.36
    sigma_init: float = 434.29
    sigma_min: float = 60.0
    tau: float = 116.36
    sigma_growth_c: float = 3.0
    recency_decay: float = 0.461
    inactivity_decay_rate: float = 0.001
    num_passes: int = 4
    convergence_threshold: float = 0.5
    sos_transfer_pct: float = 0.022
    loser_penalty_pct: float = 0.020
    title_mult: float = 1.5
    five_rd_mult: float = 1.10
    k_mult_ko: float = 2.0
    k_mult_sub: float = 2.0
    k_mult_td: float = 1.5
    k_mult_gnd: float = 1.2
    k_mult_dur: float = 1.5
    decision_ud: float = 0.91
    decision_split: float = 0.55
    decision_maj: float = 0.60


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

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
    Finish rounds get full credit (1.0) — finishing IS the complete round.
    Non-finish rounds always = 1.0.
    """
    return 1.0


def _glicko_update_sigma(sigma: float, params: GlickoParams) -> float:
    """Shrink sigma after observing a round."""
    return max(1.0 / math.sqrt(1.0 / (sigma ** 2) + 1.0 / (params.tau ** 2)), params.sigma_min)


def _glicko_inflate_sigma(sigma: float, days_inactive: float, params: GlickoParams) -> float:
    """Grow sigma during inactivity."""
    return min(math.sqrt(sigma ** 2 + (params.sigma_growth_c ** 2) * days_inactive), params.sigma_init)


def _effective_k(k_base: float, sigma: float, recency_years: float,
                 round_dur_factor: float, params: GlickoParams) -> float:
    """
    Combine all K-factor modifiers:
    - Glicko sigma scaling (high uncertainty = bigger moves)
    - Recency decay (older rounds matter less)
    - Round duration (finish rounds scale by elapsed time)
    """
    # Floor at 0.5 so veterans still get at least half of K_BASE
    glicko_scale = 0.5 + 0.5 * (sigma / params.sigma_init)
    recency_scale = math.exp(-params.recency_decay * recency_years)
    return k_base * glicko_scale * recency_scale * round_dur_factor


def _init_ratings(params: GlickoParams):
    """Create a fresh ratings dict with (mu, sigma) tuples."""
    return defaultdict(lambda: {d: [0.0, params.sigma_init] for d in DIMENSIONS})


def _get_mu(ratings, fighter_id, dim):
    return ratings[fighter_id][dim][0]


def _get_sigma(ratings, fighter_id, dim):
    return ratings[fighter_id][dim][1]


def _update_rating(ratings, fighter_id, dim, delta, age_factor, params: GlickoParams, update_sigma=True):
    """Apply a rating delta with age factor (gains only) and update sigma."""
    if delta > 0:
        ratings[fighter_id][dim][0] += delta * age_factor
    else:
        ratings[fighter_id][dim][0] += delta
    if update_sigma:
        ratings[fighter_id][dim][1] = _glicko_update_sigma(ratings[fighter_id][dim][1], params)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def _load_data(db):
    """Load all fight data, per-round stats, derived totals, and fighter info from DB."""
    log.info("  Loading fight data...")
    fights = (
        db.query(UFCFight)
        .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
        .order_by(UFCFight.date, UFCFight.id)
        .all()
    )

    fight_map = {}
    for f in fights:
        round_minutes = 5
        if f.time_format:
            parts = f.time_format.split("-")
            if parts:
                try:
                    round_minutes = int(parts[0].strip())
                except ValueError:
                    pass

        is_title = "title" in (f.weight_class or "").lower()
        is_5rd = f.time_format and f.time_format.count("-") >= 4

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

    log.info("  Loading per-fight derived totals...")
    totals_stats = (
        db.query(UFCFightStats)
        .filter(UFCFightStats.round_number == 0)
        .all()
    )
    derived_totals = {}
    DERIVED_FEATURES = [
        "slpm", "sig_def", "td_def", "td_acc", "ctrl15g",
        "kd15s", "sub_att15g", "head_pm", "dist_def", "sig_acc",
    ]
    for s in totals_stats:
        row = {}
        for feat in DERIVED_FEATURES:
            row[feat] = getattr(s, feat, None) or 0.0
        derived_totals[(s.fight_id, s.fighter_id)] = row
    log.info(f"  Loaded {len(derived_totals)} derived totals rows")

    fighters_q = db.query(UFCFighter).all()
    fighter_info = {f.id: f for f in fighters_q}
    log.info(f"  Loaded {len(fighter_info)} fighters")

    return fight_map, rounds_by_fight, derived_totals, fighter_info


# ---------------------------------------------------------------------------
# BASELINES
# ---------------------------------------------------------------------------

def _compute_baselines(fight_map, rounds_by_fight):
    """Compute per-weight-class baseline rates and rate-based percentile caps."""
    log.info("  Computing weight class baseline rates...")

    wc_round_counts = defaultdict(int)
    wc_kd_total = defaultdict(float)
    wc_sub_total = defaultdict(float)
    wc_td_total = defaultdict(float)
    wc_ctrl_total = defaultdict(float)
    wc_gnd_total = defaultdict(float)

    wc_td15s_vals = defaultdict(list)
    wc_ctrl15g_vals = defaultdict(list)
    wc_gnp15g_vals = defaultdict(list)

    for fight_id, round_data in rounds_by_fight.items():
        fight = fight_map.get(fight_id)
        if not fight or fight["weight_class"] == "unknown":
            continue
        wc = fight["weight_class"]
        round_minutes = fight.get("round_minutes", 5)

        for rnd_num, fighter_stats in round_data.items():
            wc_round_counts[wc] += 1

            fids = list(fighter_stats.keys())
            if len(fids) == 2:
                s0, s1 = fighter_stats[fids[0]], fighter_stats[fids[1]]
                est_ground_min = (s0["ctrl_seconds"] + s1["ctrl_seconds"]) / 60.0
                est_standing_min = max(round_minutes - est_ground_min, 0.0)

                for stats in [s0, s1]:
                    wc_kd_total[wc] += min(stats["kd"], 3) / 3.0
                    wc_sub_total[wc] += min(stats["sub_att"], 2) / 2.0
                    wc_td_total[wc] += min(stats["td_landed"], 5) / 5.0
                    wc_ctrl_total[wc] += 1 if stats["ctrl_seconds"] > 15 else 0
                    wc_gnd_total[wc] += 1 if stats["ground_landed"] > 3 else 0

                    if est_standing_min > 0.5 and stats["td_landed"] > 0:
                        wc_td15s_vals[wc].append(stats["td_landed"] * 15 / est_standing_min)
                    if est_ground_min > 0.5:
                        if stats["ctrl_seconds"] > 0:
                            wc_ctrl15g_vals[wc].append(stats["ctrl_seconds"] / 60.0 * 15 / est_ground_min)
                        if stats["ground_landed"] > 0:
                            wc_gnp15g_vals[wc].append(stats["ground_landed"] * 15 / est_ground_min)
            else:
                for fid, stats in fighter_stats.items():
                    wc_kd_total[wc] += min(stats["kd"], 3) / 3.0
                    wc_sub_total[wc] += min(stats["sub_att"], 2) / 2.0
                    wc_td_total[wc] += min(stats["td_landed"], 5) / 5.0
                    wc_ctrl_total[wc] += 1 if stats["ctrl_seconds"] > 15 else 0
                    wc_gnd_total[wc] += 1 if stats["ground_landed"] > 3 else 0

    def _percentile_90(vals):
        if not vals:
            return 10.0
        s = sorted(vals)
        idx = int(len(s) * 0.9)
        return s[min(idx, len(s) - 1)]

    baselines = {}
    for wc in WEIGHT_CLASS_ORDER:
        total = max(wc_round_counts.get(wc, 1), 1)
        n_fighters = total * 2
        baselines[wc] = {
            "kd": wc_kd_total.get(wc, 0) / max(n_fighters, 1),
            "sub": wc_sub_total.get(wc, 0) / max(n_fighters, 1),
            "td": wc_td_total.get(wc, 0) / max(n_fighters, 1),
            "ctrl": wc_ctrl_total.get(wc, 0) / max(total, 1),
            "gnd": wc_gnd_total.get(wc, 0) / max(total, 1),
            "td15s_cap": _percentile_90(wc_td15s_vals.get(wc, [])),
            "ctrl15g_cap": _percentile_90(wc_ctrl15g_vals.get(wc, [])),
            "gnp15g_cap": _percentile_90(wc_gnp15g_vals.get(wc, [])),
        }
        log.info(f"    {WEIGHT_CLASS_LABELS.get(wc, wc)}: {total} rounds, "
                 f"KD={baselines[wc]['kd']:.4f}, SUB={baselines[wc]['sub']:.4f}, "
                 f"TD={baselines[wc]['td']:.4f}, "
                 f"td15s_cap={baselines[wc]['td15s_cap']:.1f}, "
                 f"ctrl15g_cap={baselines[wc]['ctrl15g_cap']:.1f}, "
                 f"gnp15g_cap={baselines[wc]['gnp15g_cap']:.1f}")

    return baselines


# ---------------------------------------------------------------------------
# CORE GLICKO COMPUTATION
# ---------------------------------------------------------------------------

def _run_glicko(fight_map, rounds_by_fight, fighter_info, baselines,
                params: GlickoParams | None = None):
    """Run the full Glicko rating computation over all fights. Returns ratings and metadata."""
    if params is None:
        params = GlickoParams()

    sorted_fight_ids = sorted(
        fight_map.keys(),
        key=lambda fid: (fight_map[fid]["date"] or date.min, fid)
    )

    most_recent_date = max(
        (fight_map[fid]["date"] for fid in sorted_fight_ids if fight_map[fid]["date"]),
        default=date.today()
    )

    ratings = _init_ratings(params)

    fighter_round_count = defaultdict(int)
    fighter_fight_count = defaultdict(int)
    fighter_last_fight_date = {}
    fighter_weight_class = {}
    fighter_seeded = set()
    fighter_streak = defaultdict(int)
    fight_rating_snapshots = {}  # {(fight_id, fighter_id): {dim: mu}} pre-fight ratings

    # Build decision scores from params
    decision_scores = {
        "Decision - Split": params.decision_split,
        "Decision - Majority": params.decision_maj,
        "Decision - Unanimous": params.decision_ud,
        "Decision": 0.75,
    }

    rounds_processed = 0
    for fight_id in sorted_fight_ids:
        fight = fight_map[fight_id]
        wc = fight["weight_class"]
        if wc == "unknown":
            continue

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

        # --- Snapshot pre-fight ratings for ML features ---
        for fid in [red_id, blue_id]:
            fight_rating_snapshots[(fight_id, fid)] = {
                d: ratings[fid][d][0] for d in DIMENSIONS
            }

        # --- COMBAT AGE factors ---
        red_fi = fighter_info.get(red_id)
        blue_fi = fighter_info.get(blue_id)
        red_age_factor = _combat_age_factor(red_fi.dob if red_fi else None, fight_date)
        blue_age_factor = _combat_age_factor(blue_fi.dob if blue_fi else None, fight_date)

        # --- DECISION SCORING ---
        is_finish = "KO" in method or "Sub" in method
        decision_score = decision_scores.get(method, 1.0 if is_finish else 0.75)

        # --- AUTOCORRELATION: compute pre-fight composite ---
        red_composite = sum(_get_mu(ratings, red_id, d) for d in DIMENSIONS)
        blue_composite = sum(_get_mu(ratings, blue_id, d) for d in DIMENSIONS)

        # --- RECENCY ---
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

            rdf = _round_duration_factor(
                is_finish_round, fight["finish_time_seconds"],
                fight["round_minutes"]
            )

            red_avg_sigma = sum(_get_sigma(ratings, red_id, d) for d in DIMENSIONS) / len(DIMENSIONS)
            blue_avg_sigma = sum(_get_sigma(ratings, blue_id, d) for d in DIMENSIONS) / len(DIMENSIONS)

            K_red = _effective_k(params.k_base, red_avg_sigma, recency_years, rdf, params)
            K_blue = _effective_k(params.k_base, blue_avg_sigma, recency_years, rdf, params)

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

            _update_rating(ratings, red_id, "pts", pts_delta_red, red_age_factor, params)
            _update_rating(ratings, blue_id, "pts", pts_delta_blue, blue_age_factor, params)

            # --- KO / KOd (continuous scoring) ---
            ko_baseline = bl["kd"]
            red_kd_outcome = min(r["kd"], 3) / 3.0
            exp_ko = _elo_expected(r_mu("ko"), b_mu("kod"))
            adj_exp = ko_baseline * exp_ko
            K_ko_r = K_red * params.k_mult_ko
            K_ko_b = K_blue * params.k_mult_ko
            ko_d_red = K_ko_r * (red_kd_outcome - adj_exp)
            ko_d_blue_def = K_ko_b * ((1 - red_kd_outcome) - (1 - adj_exp))
            _update_rating(ratings, red_id, "ko", ko_d_red, red_age_factor, params)
            _update_rating(ratings, blue_id, "kod", ko_d_blue_def, blue_age_factor, params)

            blue_kd_outcome = min(b["kd"], 3) / 3.0
            exp_ko_b = _elo_expected(b_mu("ko"), r_mu("kod"))
            adj_exp_b = ko_baseline * exp_ko_b
            ko_d_blue = K_ko_b * (blue_kd_outcome - adj_exp_b)
            ko_d_red_def = K_ko_r * ((1 - blue_kd_outcome) - (1 - adj_exp_b))
            _update_rating(ratings, blue_id, "ko", ko_d_blue, blue_age_factor, params)
            _update_rating(ratings, red_id, "kod", ko_d_red_def, red_age_factor, params)

            # KO finish bonus with autocorrelation correction
            if is_ko_finish:
                ko_bonus = params.k_base * 5
                if fight["winner_id"] == red_id:
                    ac = _autocorrelation_factor(red_composite, blue_composite)
                    _update_rating(ratings, red_id, "ko", ko_bonus * ac, red_age_factor, params, update_sigma=False)
                    _update_rating(ratings, blue_id, "kod", -ko_bonus * 0.5, 1.0, params, update_sigma=False)
                else:
                    ac = _autocorrelation_factor(blue_composite, red_composite)
                    _update_rating(ratings, blue_id, "ko", ko_bonus * ac, blue_age_factor, params, update_sigma=False)
                    _update_rating(ratings, red_id, "kod", -ko_bonus * 0.5, 1.0, params, update_sigma=False)

            # --- SUB / SUBd (continuous scoring) ---
            sub_baseline = bl["sub"]
            red_sub_outcome = min(r["sub_att"], 2) / 2.0
            exp_sub = _elo_expected(r_mu("sub"), b_mu("subd"))
            adj_sub_exp = sub_baseline * exp_sub
            K_sub_r = K_red * params.k_mult_sub
            K_sub_b = K_blue * params.k_mult_sub
            sub_d_red = K_sub_r * (red_sub_outcome - adj_sub_exp)
            sub_d_blue_def = K_sub_b * ((1 - red_sub_outcome) - (1 - adj_sub_exp))
            _update_rating(ratings, red_id, "sub", sub_d_red, red_age_factor, params)
            _update_rating(ratings, blue_id, "subd", sub_d_blue_def, blue_age_factor, params)

            blue_sub_outcome = min(b["sub_att"], 2) / 2.0
            exp_sub_b = _elo_expected(b_mu("sub"), r_mu("subd"))
            adj_sub_exp_b = sub_baseline * exp_sub_b
            sub_d_blue = K_sub_b * (blue_sub_outcome - adj_sub_exp_b)
            sub_d_red_def = K_sub_r * ((1 - blue_sub_outcome) - (1 - adj_sub_exp_b))
            _update_rating(ratings, blue_id, "sub", sub_d_blue, blue_age_factor, params)
            _update_rating(ratings, red_id, "subd", sub_d_red_def, red_age_factor, params)

            # SUB finish bonus with autocorrelation correction
            if is_sub_finish:
                sub_bonus = params.k_base * 5
                if fight["winner_id"] == red_id:
                    ac = _autocorrelation_factor(red_composite, blue_composite)
                    _update_rating(ratings, red_id, "sub", sub_bonus * ac, red_age_factor, params, update_sigma=False)
                    _update_rating(ratings, blue_id, "subd", -sub_bonus * 0.5, 1.0, params, update_sigma=False)
                else:
                    ac = _autocorrelation_factor(blue_composite, red_composite)
                    _update_rating(ratings, blue_id, "sub", sub_bonus * ac, blue_age_factor, params, update_sigma=False)
                    _update_rating(ratings, red_id, "subd", -sub_bonus * 0.5, 1.0, params, update_sigma=False)

            # --- TD / TDd (rate-based: td15s normalized by standing time) ---
            est_ground_min = (r["ctrl_seconds"] + b["ctrl_seconds"]) / 60.0
            round_min = fight["round_minutes"]
            est_standing_min = max(round_min - est_ground_min, 0.0)

            td_baseline = bl["td"]
            K_td_r = K_red * params.k_mult_td
            K_td_b = K_blue * params.k_mult_td

            if est_standing_min > 0.5:
                td15s_cap = bl["td15s_cap"]
                red_td = min(r["td_landed"] * 15 / est_standing_min / td15s_cap, 1.0)
                exp_td = _elo_expected(r_mu("td"), b_mu("tdd"))
                adj_exp = td_baseline * exp_td
                _update_rating(ratings, red_id, "td", K_td_r * (red_td - adj_exp), red_age_factor, params)
                _update_rating(ratings, blue_id, "tdd", K_td_b * ((1 - red_td) - (1 - adj_exp)), blue_age_factor, params)

                blue_td = min(b["td_landed"] * 15 / est_standing_min / td15s_cap, 1.0)
                exp_td_b = _elo_expected(b_mu("td"), r_mu("tdd"))
                adj_exp_b = td_baseline * exp_td_b
                _update_rating(ratings, blue_id, "td", K_td_b * (blue_td - adj_exp_b), blue_age_factor, params)
                _update_rating(ratings, red_id, "tdd", K_td_r * ((1 - blue_td) - (1 - adj_exp_b)), red_age_factor, params)
            else:
                red_td = min(r["td_landed"], 5) / 5.0
                exp_td = _elo_expected(r_mu("td"), b_mu("tdd"))
                _update_rating(ratings, red_id, "td", K_td_r * (red_td - td_baseline * exp_td), red_age_factor, params)
                _update_rating(ratings, blue_id, "tdd", K_td_b * ((1 - red_td) - (1 - td_baseline * exp_td)), blue_age_factor, params)

                blue_td = min(b["td_landed"], 5) / 5.0
                exp_td_b = _elo_expected(b_mu("td"), r_mu("tdd"))
                _update_rating(ratings, blue_id, "td", K_td_b * (blue_td - td_baseline * exp_td_b), blue_age_factor, params)
                _update_rating(ratings, red_id, "tdd", K_td_r * ((1 - blue_td) - (1 - td_baseline * exp_td_b)), red_age_factor, params)

            # --- CTRL (rate-based: ctrl15g normalized by ground time) ---
            if est_ground_min > 0.5:
                ctrl15g_cap = bl["ctrl15g_cap"]
                red_ctrl_rate = min(r["ctrl_seconds"] / 60.0 * 15 / est_ground_min / ctrl15g_cap, 1.0)
                exp_ctrl = _elo_expected(r_mu("ctrl"), b_mu("ctrl"))
                _update_rating(ratings, red_id, "ctrl", K_red * (red_ctrl_rate - exp_ctrl), red_age_factor, params)
                _update_rating(ratings, blue_id, "ctrl", K_blue * ((1 - red_ctrl_rate) - (1 - exp_ctrl)), blue_age_factor, params)
            elif r["ctrl_seconds"] + b["ctrl_seconds"] > 0:
                total_ctrl = r["ctrl_seconds"] + b["ctrl_seconds"]
                red_ctrl_share = r["ctrl_seconds"] / total_ctrl
                exp_ctrl = _elo_expected(r_mu("ctrl"), b_mu("ctrl"))
                _update_rating(ratings, red_id, "ctrl", K_red * (red_ctrl_share - exp_ctrl), red_age_factor, params)
                _update_rating(ratings, blue_id, "ctrl", K_blue * ((1 - red_ctrl_share) - (1 - exp_ctrl)), blue_age_factor, params)

            # --- STR_VOL ---
            total_str = r["sig_str_landed"] + b["sig_str_landed"]
            if total_str > 0:
                red_str_share = r["sig_str_landed"] / total_str
                exp_str = _elo_expected(r_mu("str_vol"), b_mu("str_vol"))
                sv_r = K_red * (red_str_share - exp_str)
                sv_b = K_blue * ((1 - red_str_share) - (1 - exp_str))
                _update_rating(ratings, red_id, "str_vol", sv_r, red_age_factor, params)
                _update_rating(ratings, blue_id, "str_vol", sv_b, blue_age_factor, params)

            # --- STR_ACC (opponent-relative) ---
            if r["sig_str_attempted"] > 0 and b["sig_str_attempted"] > 0:
                red_acc = r["sig_str_landed"] / r["sig_str_attempted"]
                blue_acc = b["sig_str_landed"] / b["sig_str_attempted"]
                exp_acc_r = _elo_expected(r_mu("str_acc"), b_mu("str_acc"))
                exp_acc_b = _elo_expected(b_mu("str_acc"), r_mu("str_acc"))
                sa_r = K_red * (red_acc - exp_acc_r)
                sa_b = K_blue * (blue_acc - exp_acc_b)
                _update_rating(ratings, red_id, "str_acc", sa_r, red_age_factor, params)
                _update_rating(ratings, blue_id, "str_acc", sa_b, blue_age_factor, params)
            elif r["sig_str_attempted"] > 0:
                red_acc = r["sig_str_landed"] / r["sig_str_attempted"]
                exp_acc_r = _elo_expected(r_mu("str_acc"), b_mu("str_acc"))
                sa_r = K_red * (red_acc - exp_acc_r)
                _update_rating(ratings, red_id, "str_acc", sa_r, red_age_factor, params)
            elif b["sig_str_attempted"] > 0:
                blue_acc = b["sig_str_landed"] / b["sig_str_attempted"]
                exp_acc_b = _elo_expected(b_mu("str_acc"), r_mu("str_acc"))
                sa_b = K_blue * (blue_acc - exp_acc_b)
                _update_rating(ratings, blue_id, "str_acc", sa_b, blue_age_factor, params)

            # --- STR_DEF (striking defense — opponent-relative, mirrors str_acc) ---
            if b["sig_str_attempted"] > 0 and r["sig_str_attempted"] > 0:
                red_def = 1.0 - (b["sig_str_landed"] / b["sig_str_attempted"])
                blue_def = 1.0 - (r["sig_str_landed"] / r["sig_str_attempted"])
                exp_def_r = _elo_expected(r_mu("str_def"), b_mu("str_def"))
                exp_def_b = _elo_expected(b_mu("str_def"), r_mu("str_def"))
                _update_rating(ratings, red_id, "str_def", K_red * (red_def - exp_def_r), red_age_factor, params)
                _update_rating(ratings, blue_id, "str_def", K_blue * (blue_def - exp_def_b), blue_age_factor, params)
            elif b["sig_str_attempted"] > 0:
                red_def = 1.0 - (b["sig_str_landed"] / b["sig_str_attempted"])
                exp_def_r = _elo_expected(r_mu("str_def"), b_mu("str_def"))
                _update_rating(ratings, red_id, "str_def", K_red * (red_def - exp_def_r), red_age_factor, params)
            elif r["sig_str_attempted"] > 0:
                blue_def = 1.0 - (r["sig_str_landed"] / r["sig_str_attempted"])
                exp_def_b = _elo_expected(b_mu("str_def"), r_mu("str_def"))
                _update_rating(ratings, blue_id, "str_def", K_blue * (blue_def - exp_def_b), blue_age_factor, params)

            # --- DIST ---
            total_dist = r["distance_landed"] + b["distance_landed"]
            if total_dist > 0:
                red_dist_share = r["distance_landed"] / total_dist
                exp_dist = _elo_expected(r_mu("dist"), b_mu("dist"))
                d_r = K_red * (red_dist_share - exp_dist)
                d_b = K_blue * ((1 - red_dist_share) - (1 - exp_dist))
                _update_rating(ratings, red_id, "dist", d_r, red_age_factor, params)
                _update_rating(ratings, blue_id, "dist", d_b, blue_age_factor, params)

            # --- CLINCH (share-based, mirrors dist/gnd pattern) ---
            total_clinch = r["clinch_landed"] + b["clinch_landed"]
            if total_clinch > 0:
                red_clinch_share = r["clinch_landed"] / total_clinch
                exp_clinch = _elo_expected(r_mu("clinch"), b_mu("clinch"))
                cl_r = K_red * (red_clinch_share - exp_clinch)
                cl_b = K_blue * ((1 - red_clinch_share) - (1 - exp_clinch))
                _update_rating(ratings, red_id, "clinch", cl_r, red_age_factor, params)
                _update_rating(ratings, blue_id, "clinch", cl_b, blue_age_factor, params)

            # --- GND (rate-based: gnp15g normalized by ground time) ---
            K_gnd = K_red * params.k_mult_gnd
            K_gnd_b = K_blue * params.k_mult_gnd
            if est_ground_min > 0.5 and (r["ground_landed"] + b["ground_landed"]) > 0:
                gnp15g_cap = bl["gnp15g_cap"]
                red_gnp = min(r["ground_landed"] * 15 / est_ground_min / gnp15g_cap, 1.0)
                exp_gnd = _elo_expected(r_mu("gnd"), b_mu("gnd"))
                _update_rating(ratings, red_id, "gnd", K_gnd * (red_gnp - exp_gnd), red_age_factor, params)
                _update_rating(ratings, blue_id, "gnd", K_gnd_b * ((1 - red_gnp) - (1 - exp_gnd)), blue_age_factor, params)
            elif r["ground_landed"] + b["ground_landed"] > 0:
                total_gnd = r["ground_landed"] + b["ground_landed"]
                red_gnd_share = r["ground_landed"] / total_gnd
                exp_gnd = _elo_expected(r_mu("gnd"), b_mu("gnd"))
                _update_rating(ratings, red_id, "gnd", K_gnd * (red_gnd_share - exp_gnd), red_age_factor, params)
                _update_rating(ratings, blue_id, "gnd", K_gnd_b * ((1 - red_gnd_share) - (1 - exp_gnd)), blue_age_factor, params)

            # --- DURABILITY (absorption resilience) ---
            K_dur = K_red * params.k_mult_dur
            K_dur_b = K_blue * params.k_mult_dur
            if b["sig_str_landed"] > 0:
                red_durability = 1.0 - min(b["kd"], 3) / max(b["sig_str_landed"], 1)
                exp_dur_r = _elo_expected(r_mu("durability"), b_mu("durability"))
                _update_rating(ratings, red_id, "durability", K_dur * (red_durability - exp_dur_r), red_age_factor, params)
            if r["sig_str_landed"] > 0:
                blue_durability = 1.0 - min(r["kd"], 3) / max(r["sig_str_landed"], 1)
                exp_dur_b = _elo_expected(b_mu("durability"), r_mu("durability"))
                _update_rating(ratings, blue_id, "durability", K_dur_b * (blue_durability - exp_dur_b), blue_age_factor, params)

            fighter_round_count[red_id] += 1
            fighter_round_count[blue_id] += 1
            rounds_processed += 1

        # Track fight count (once per fight, not per round)
        fighter_fight_count[red_id] += 1
        fighter_fight_count[blue_id] += 1

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
            streak_bonus_w = 1.0 + (0.02 * win_streak)
            streak_penalty_l = 1.0 + (0.01 * loss_streak)

            # --- TITLE FIGHT BONUS ---
            title_mult = params.title_mult if fight["is_title"] else 1.0

            # --- 5-ROUND BONUS ---
            five_rd_mult = params.five_rd_mult if fight["is_5rd"] else 1.0

            # --- SKILL-AWARE OPPONENT POINT TRANSFER (SOS) ---
            combined_mult = streak_bonus_w * title_mult * five_rd_mult
            loser_positive_total = sum(
                max(_get_mu(ratings, loser_id, d), 0) for d in DIMENSIONS
            )
            if loser_positive_total > 0:
                for d in DIMENSIONS:
                    loser_dim = max(_get_mu(ratings, loser_id, d), 0)
                    dim_share = loser_dim / loser_positive_total
                    bonus = loser_positive_total * params.sos_transfer_pct * dim_share * combined_mult
                    _update_rating(ratings, winner_id, d, bonus, w_age, params, update_sigma=False)

            # --- LOSER PENALTY ---
            loser_pen = params.loser_penalty_pct * streak_penalty_l
            if fight["is_title"]:
                loser_pen /= 1.3
            for d in DIMENSIONS:
                mu = _get_mu(ratings, loser_id, d)
                if mu > 0:
                    _update_rating(ratings, loser_id, d, -mu * loser_pen, 1.0, params, update_sigma=False)

    log.info(f"  Processed {rounds_processed} rounds")

    return ratings, fighter_round_count, fighter_fight_count, fighter_last_fight_date, fighter_weight_class, sorted_fight_ids, fight_rating_snapshots


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def run_glicko_inmemory(params: GlickoParams | None = None) -> dict:
    """Load data from DB, run Glicko, return fight_rating_snapshots without persisting.

    Used by the tuner for fast parameter search without DB writes.

    Returns:
        dict mapping (fight_id, fighter_id) -> {dim: mu_value}
    """
    if params is None:
        params = GlickoParams()

    db = SessionLocal()
    try:
        fight_map, rounds_by_fight, derived_totals, fighter_info = _load_data(db)
    finally:
        db.close()

    baselines = _compute_baselines(fight_map, rounds_by_fight)
    results = _run_glicko(fight_map, rounds_by_fight, fighter_info, baselines, params)
    return results[6]  # fight_rating_snapshots


def run_glicko_inmemory_cached(fight_map, rounds_by_fight, fighter_info, baselines,
                               params: GlickoParams | None = None) -> dict:
    """Run Glicko with pre-loaded data. Used by tuner to avoid reloading data each trial.

    Returns:
        dict mapping (fight_id, fighter_id) -> {dim: mu_value}
    """
    if params is None:
        params = GlickoParams()

    results = _run_glicko(fight_map, rounds_by_fight, fighter_info, baselines, params)
    return results[6]  # fight_rating_snapshots


def load_glicko_data():
    """Load all data needed for Glicko computation. Cache-friendly for tuner.

    Returns:
        (fight_map, rounds_by_fight, derived_totals, fighter_info, baselines)
    """
    db = SessionLocal()
    try:
        fight_map, rounds_by_fight, derived_totals, fighter_info = _load_data(db)
    finally:
        db.close()

    baselines = _compute_baselines(fight_map, rounds_by_fight)
    return fight_map, rounds_by_fight, derived_totals, fighter_info, baselines


def compute_and_save_snapshots(db, params: GlickoParams | None = None):
    """Run Glicko computation and persist snapshots to DB.

    Called by ranking_service.generate_rankings().

    Returns:
        (ratings, fighter_round_count, fighter_fight_count,
         fighter_last_fight_date, fighter_weight_class, sorted_fight_ids)
    """
    if params is None:
        params = GlickoParams()

    fight_map, rounds_by_fight, derived_totals, fighter_info = _load_data(db)
    baselines = _compute_baselines(fight_map, rounds_by_fight)
    ratings, fighter_round_count, fighter_fight_count, fighter_last_fight_date, \
        fighter_weight_class, sorted_fight_ids, fight_rating_snapshots = \
        _run_glicko(fight_map, rounds_by_fight, fighter_info, baselines, params)

    # --- Persist pre-fight Glicko snapshots to DB (used by ML model) ---
    log.info(f"  Saving {len(fight_rating_snapshots)} Glicko snapshots to DB...")
    db.query(UFCGlickoSnapshot).delete()
    db.commit()
    snapshot_rows = []
    for (fight_id, fighter_id), dims in fight_rating_snapshots.items():
        row = UFCGlickoSnapshot(
            fight_id=int(fight_id),
            fighter_id=int(fighter_id),
            **{d: round(dims.get(d, 0.0), 4) for d in DIMENSIONS},
        )
        snapshot_rows.append(row)
    db.bulk_save_objects(snapshot_rows)
    db.commit()
    log.info(f"  Saved {len(snapshot_rows)} snapshots")

    # --- Apply Glicko inactivity: sigma inflation + rating decay ---
    log.info("  Applying inactivity sigma inflation and rating decay...")
    today = date.today()
    for fid in ratings:
        last_fight = fighter_last_fight_date.get(fid)
        if not last_fight:
            continue
        days_inactive = (today - last_fight).days
        if days_inactive > 90:
            decay_days = days_inactive - 90
            decay_factor = math.exp(-params.inactivity_decay_rate * decay_days)
            for dim in DIMENSIONS:
                ratings[fid][dim][1] = _glicko_inflate_sigma(
                    ratings[fid][dim][1], decay_days, params
                )
                ratings[fid][dim][0] *= decay_factor

    return (ratings, fighter_round_count, fighter_fight_count,
            fighter_last_fight_date, fighter_weight_class, sorted_fight_ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Running standalone Glicko computation...")
    snapshots = run_glicko_inmemory()
    log.info(f"Computed {len(snapshots)} snapshots")
