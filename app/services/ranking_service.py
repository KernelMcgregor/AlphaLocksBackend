"""
Fighter Rankings — Round-Level Multi-Dimensional Elo v2

Processes every round in UFC history to build 13-dimensional fighter ratings.

v2 improvements:
- Autocorrelation correction: finish bonuses scaled down when favorite wins
  (prevents double-counting since favorites both win more AND finish more)
- Decision scoring granularity: Split=0.55, Majority=0.6, Unanimous=0.91, Finish=1.0
- Combat age modeling: biological peak 25-40, gradual decline outside window
- Intelligent newcomer seeding: pre-UFC record seeds starting ratings

Run: python -m app.services.ranking_service
"""

from __future__ import annotations

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

MIN_ROUNDS = 10
DECAY_RATE = 0.97
K_BASE = 20

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
        # Women's featherweight is essentially defunct — roll into W. Bantamweight
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
    Biological combat age multiplier. Fighters peak 25-40.
    Returns a multiplier on K-factor gains (not losses).
    Outside peak window, gains are reduced — fighter is declining.
    """
    if not dob or not fight_date:
        return 1.0
    age = (fight_date - dob).days / 365.25
    if 25 <= age <= 35:
        return 1.0  # prime
    elif 35 < age <= 38:
        return 0.9  # early decline
    elif 38 < age <= 40:
        return 0.75  # significant decline
    elif age > 40:
        return 0.6  # steep decline
    elif 22 <= age < 25:
        return 0.95  # still developing
    elif age < 22:
        return 0.85  # very raw
    return 1.0


def _newcomer_seed(wins: int, losses: int) -> float:
    """
    Seed starting ratings based on pre-UFC record.
    A 15-0 prospect starts higher than a 5-4 journeyman.
    Returns a delta applied to all dimensions on first fight.
    """
    total = wins + losses
    if total == 0:
        return 0.0
    win_pct = wins / total
    # Scale: undefeated with many wins = +30, .500 record = 0, bad record = -15
    record_quality = (win_pct - 0.5) * 2  # -1 to +1
    volume_bonus = min(total / 20, 1.0)  # more fights = more confident in seed
    return record_quality * volume_bonus * 30


def _autocorrelation_factor(winner_rating: float, loser_rating: float) -> float:
    """
    Autocorrelation correction (FiveThirtyEight/Cage Calculus).
    Favorites both win more AND finish more, so finish bonuses
    double-count the favorite's advantage. Scale down the bonus
    when the favorite wins, scale up when the underdog wins.
    """
    expected = _elo_expected(winner_rating, loser_rating)
    # When expected > 0.5 (favorite won): reduce bonus
    # When expected < 0.5 (upset): increase bonus
    # Factor ranges from ~0.5 (big favorite) to ~1.5 (big upset)
    return 1.0 + (0.5 - expected)


def generate_rankings():
    """Process all rounds in UFC history and compute multi-dimensional Elo ratings."""
    log.info("=" * 60)
    log.info("GENERATING FIGHTER RANKINGS (Round-Level Multi-Dim Elo v2)")
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
            fight_map[f.id] = {
                "id": f.id,
                "date": f.date,
                "red_id": f.red_fighter_id,
                "blue_id": f.blue_fighter_id,
                "winner_id": f.winner_id,
                "method": f.method or "",
                "weight_class": _classify_weight_class(f.weight_class),
                "finish_round": f.finish_round,
                "max_rounds": 5 if f.time_format and "5" in (f.time_format or "") else 3,
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

        # --- Load fighter info (including DOB for combat age) ---
        fighters_q = db.query(UFCFighter).all()
        fighter_info = {f.id: f for f in fighters_q}
        log.info(f"  Loaded {len(fighter_info)} fighters")

        # --- Compute per-weight-class baseline rates ---
        log.info("  Computing weight class baseline rates...")

        wc_round_counts = defaultdict(int)
        wc_ko_rounds = defaultdict(int)
        wc_sub_rounds = defaultdict(int)
        wc_td_rounds = defaultdict(int)
        wc_ctrl_rounds = defaultdict(int)
        wc_kd_rounds = defaultdict(int)
        wc_gnd_rounds = defaultdict(int)

        for fight_id, round_data in rounds_by_fight.items():
            fight = fight_map.get(fight_id)
            if not fight or fight["weight_class"] == "unknown":
                continue
            wc = fight["weight_class"]

            for rnd_num, fighter_stats in round_data.items():
                wc_round_counts[wc] += 1
                is_finish_round = (fight["finish_round"] == rnd_num)
                if is_finish_round and "KO" in fight["method"]:
                    wc_ko_rounds[wc] += 1
                if is_finish_round and "Sub" in fight["method"]:
                    wc_sub_rounds[wc] += 1
                for fid, stats in fighter_stats.items():
                    if stats["td_landed"] > 0:
                        wc_td_rounds[wc] += 1; break
                for fid, stats in fighter_stats.items():
                    if stats["ctrl_seconds"] > 15:
                        wc_ctrl_rounds[wc] += 1; break
                for fid, stats in fighter_stats.items():
                    if stats["kd"] > 0:
                        wc_kd_rounds[wc] += 1; break
                for fid, stats in fighter_stats.items():
                    if stats["ground_landed"] > 3:
                        wc_gnd_rounds[wc] += 1; break

        baselines = {}
        for wc in WEIGHT_CLASS_ORDER:
            total = max(wc_round_counts.get(wc, 1), 1)
            baselines[wc] = {
                "ko": wc_ko_rounds.get(wc, 0) / total,
                "sub": wc_sub_rounds.get(wc, 0) / total,
                "td": wc_td_rounds.get(wc, 0) / total,
                "ctrl": wc_ctrl_rounds.get(wc, 0) / total,
                "kd": wc_kd_rounds.get(wc, 0) / total,
                "gnd": wc_gnd_rounds.get(wc, 0) / total,
            }
            log.info(f"    {WEIGHT_CLASS_LABELS.get(wc, wc)}: {total} rounds, "
                     f"KO={baselines[wc]['ko']:.3f}, SUB={baselines[wc]['sub']:.3f}, "
                     f"TD={baselines[wc]['td']:.3f}")

        # --- Process all rounds chronologically ---
        log.info("  Processing rounds...")

        ratings = defaultdict(lambda: {d: 0.0 for d in DIMENSIONS})
        fighter_round_count = defaultdict(int)
        fighter_last_fight_date = {}
        fighter_weight_class = {}
        fighter_seeded = set()  # track who has been seeded

        sorted_fight_ids = sorted(
            fight_map.keys(),
            key=lambda fid: (fight_map[fid]["date"] or date.min, fid)
        )

        rounds_processed = 0
        for fight_id in sorted_fight_ids:
            fight = fight_map[fight_id]
            wc = fight["weight_class"]
            if wc == "unknown":
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

            # --- NEWCOMER SEEDING (IOPP) ---
            # On a fighter's first UFC round, seed their ratings from pre-UFC record
            for fid in [red_id, blue_id]:
                if fid not in fighter_seeded:
                    fighter_seeded.add(fid)
                    fi = fighter_info.get(fid)
                    if fi:
                        seed = _newcomer_seed(fi.wins, fi.losses)
                        if seed != 0:
                            for dim in DIMENSIONS:
                                ratings[fid][dim] += seed

            # --- COMBAT AGE factors ---
            red_fi = fighter_info.get(red_id)
            blue_fi = fighter_info.get(blue_id)
            red_age_factor = _combat_age_factor(red_fi.dob if red_fi else None, fight_date)
            blue_age_factor = _combat_age_factor(blue_fi.dob if blue_fi else None, fight_date)

            # --- DECISION SCORING ---
            # Determine fight-level outcome score for the PTS dimension
            is_finish = "KO" in method or "Sub" in method
            decision_score = DECISION_SCORES.get(method, 1.0 if is_finish else 0.75)

            # --- AUTOCORRELATION: compute pre-fight rating advantage ---
            red_composite = sum(ratings[red_id][d] for d in DIMENSIONS)
            blue_composite = sum(ratings[blue_id][d] for d in DIMENSIONS)

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

                r_rating = ratings[red_id]
                b_rating = ratings[blue_id]

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

                # Finish round: winner gets the round
                if is_finish_round:
                    red_won_round = 1.0 if fight["winner_id"] == red_id else 0.0

                # DECISION SCORING: on the last round, scale the PTS update
                # by the decision quality. Split decisions barely move ratings.
                K_pts = K_BASE
                if is_last_round and not is_finish:
                    K_pts = K_BASE * decision_score

                exp_red = _elo_expected(r_rating["pts"], b_rating["pts"])
                pts_delta_red = K_pts * (red_won_round - exp_red)
                pts_delta_blue = K_pts * ((1 - red_won_round) - (1 - exp_red))

                # Apply combat age: gains are multiplied, losses are not
                r_rating["pts"] += pts_delta_red * (red_age_factor if pts_delta_red > 0 else 1.0)
                b_rating["pts"] += pts_delta_blue * (blue_age_factor if pts_delta_blue > 0 else 1.0)

                # --- KO / KOd ---
                ko_expected = bl["kd"]
                red_kd_outcome = min(r["kd"], 1)
                exp_ko = _elo_expected(r_rating["ko"], b_rating["kod"])
                adj_exp = ko_expected * exp_ko
                K_ko = K_BASE * 2
                ko_d_red = K_ko * (red_kd_outcome - adj_exp)
                ko_d_blue_def = K_ko * ((1 - red_kd_outcome) - (1 - adj_exp))
                r_rating["ko"] += ko_d_red * (red_age_factor if ko_d_red > 0 else 1.0)
                b_rating["kod"] += ko_d_blue_def * (blue_age_factor if ko_d_blue_def > 0 else 1.0)

                blue_kd_outcome = min(b["kd"], 1)
                exp_ko_b = _elo_expected(b_rating["ko"], r_rating["kod"])
                adj_exp_b = ko_expected * exp_ko_b
                ko_d_blue = K_ko * (blue_kd_outcome - adj_exp_b)
                ko_d_red_def = K_ko * ((1 - blue_kd_outcome) - (1 - adj_exp_b))
                b_rating["ko"] += ko_d_blue * (blue_age_factor if ko_d_blue > 0 else 1.0)
                r_rating["kod"] += ko_d_red_def * (red_age_factor if ko_d_red_def > 0 else 1.0)

                # KO finish bonus with AUTOCORRELATION CORRECTION
                if is_ko_finish:
                    ko_bonus = K_BASE * 3
                    if fight["winner_id"] == red_id:
                        ac = _autocorrelation_factor(red_composite, blue_composite)
                        r_rating["ko"] += ko_bonus * ac * red_age_factor
                        b_rating["kod"] -= ko_bonus * 0.5
                    else:
                        ac = _autocorrelation_factor(blue_composite, red_composite)
                        b_rating["ko"] += ko_bonus * ac * blue_age_factor
                        r_rating["kod"] -= ko_bonus * 0.5

                # --- SUB / SUBd ---
                sub_expected = bl["sub"]
                red_sub_outcome = min(r["sub_att"], 1)
                exp_sub = _elo_expected(r_rating["sub"], b_rating["subd"])
                adj_sub_exp = sub_expected * exp_sub
                K_sub = K_BASE * 2
                sub_d_red = K_sub * (red_sub_outcome - adj_sub_exp)
                sub_d_blue_def = K_sub * ((1 - red_sub_outcome) - (1 - adj_sub_exp))
                r_rating["sub"] += sub_d_red * (red_age_factor if sub_d_red > 0 else 1.0)
                b_rating["subd"] += sub_d_blue_def * (blue_age_factor if sub_d_blue_def > 0 else 1.0)

                blue_sub_outcome = min(b["sub_att"], 1)
                exp_sub_b = _elo_expected(b_rating["sub"], r_rating["subd"])
                adj_sub_exp_b = sub_expected * exp_sub_b
                sub_d_blue = K_sub * (blue_sub_outcome - adj_sub_exp_b)
                sub_d_red_def = K_sub * ((1 - blue_sub_outcome) - (1 - adj_sub_exp_b))
                b_rating["sub"] += sub_d_blue * (blue_age_factor if sub_d_blue > 0 else 1.0)
                r_rating["subd"] += sub_d_red_def * (red_age_factor if sub_d_red_def > 0 else 1.0)

                # SUB finish bonus with AUTOCORRELATION CORRECTION
                if is_sub_finish:
                    sub_bonus = K_BASE * 3
                    if fight["winner_id"] == red_id:
                        ac = _autocorrelation_factor(red_composite, blue_composite)
                        r_rating["sub"] += sub_bonus * ac * red_age_factor
                        b_rating["subd"] -= sub_bonus * 0.5
                    else:
                        ac = _autocorrelation_factor(blue_composite, red_composite)
                        b_rating["sub"] += sub_bonus * ac * blue_age_factor
                        r_rating["subd"] -= sub_bonus * 0.5

                # --- TD / TDd ---
                td_expected = bl["td"]
                red_td = min(r["td_landed"], 1)
                exp_td = _elo_expected(r_rating["td"], b_rating["tdd"])
                K_td = K_BASE * 1.5
                td_d = K_td * (red_td - td_expected * exp_td)
                td_dd = K_td * ((1 - red_td) - (1 - td_expected * exp_td))
                r_rating["td"] += td_d * (red_age_factor if td_d > 0 else 1.0)
                b_rating["tdd"] += td_dd * (blue_age_factor if td_dd > 0 else 1.0)

                blue_td = min(b["td_landed"], 1)
                exp_td_b = _elo_expected(b_rating["td"], r_rating["tdd"])
                td_d_b = K_td * (blue_td - td_expected * exp_td_b)
                td_dd_b = K_td * ((1 - blue_td) - (1 - td_expected * exp_td_b))
                b_rating["td"] += td_d_b * (blue_age_factor if td_d_b > 0 else 1.0)
                r_rating["tdd"] += td_dd_b * (red_age_factor if td_dd_b > 0 else 1.0)

                # --- CTRL / CTRLd ---
                total_ctrl = r["ctrl_seconds"] + b["ctrl_seconds"]
                if total_ctrl > 0:
                    red_ctrl_share = r["ctrl_seconds"] / total_ctrl
                    exp_ctrl = _elo_expected(r_rating["ctrl"], b_rating["ctrld"])
                    K_ctrl = K_BASE
                    cd_r = K_ctrl * (red_ctrl_share - exp_ctrl)
                    cd_b = K_ctrl * ((1 - red_ctrl_share) - (1 - exp_ctrl))
                    r_rating["ctrl"] += cd_r * (red_age_factor if cd_r > 0 else 1.0)
                    b_rating["ctrld"] += cd_b * (blue_age_factor if cd_b > 0 else 1.0)

                    blue_ctrl_share = b["ctrl_seconds"] / total_ctrl
                    exp_ctrl_b = _elo_expected(b_rating["ctrl"], r_rating["ctrld"])
                    cd_r2 = K_ctrl * ((1 - blue_ctrl_share) - (1 - exp_ctrl_b))
                    cd_b2 = K_ctrl * (blue_ctrl_share - exp_ctrl_b)
                    r_rating["ctrld"] += cd_r2 * (red_age_factor if cd_r2 > 0 else 1.0)
                    b_rating["ctrl"] += cd_b2 * (blue_age_factor if cd_b2 > 0 else 1.0)

                # --- STR_VOL ---
                total_str = r["sig_str_landed"] + b["sig_str_landed"]
                if total_str > 0:
                    red_str_share = r["sig_str_landed"] / total_str
                    exp_str = _elo_expected(r_rating["str_vol"], b_rating["str_vol"])
                    sv_r = K_BASE * (red_str_share - exp_str)
                    sv_b = K_BASE * ((1 - red_str_share) - (1 - exp_str))
                    r_rating["str_vol"] += sv_r * (red_age_factor if sv_r > 0 else 1.0)
                    b_rating["str_vol"] += sv_b * (blue_age_factor if sv_b > 0 else 1.0)

                # --- STR_ACC ---
                if r["sig_str_attempted"] > 0:
                    red_acc = r["sig_str_landed"] / r["sig_str_attempted"]
                    sa_r = K_BASE * 0.5 * (red_acc - 0.45)
                    r_rating["str_acc"] += sa_r * (red_age_factor if sa_r > 0 else 1.0)
                if b["sig_str_attempted"] > 0:
                    blue_acc = b["sig_str_landed"] / b["sig_str_attempted"]
                    sa_b = K_BASE * 0.5 * (blue_acc - 0.45)
                    b_rating["str_acc"] += sa_b * (blue_age_factor if sa_b > 0 else 1.0)

                # --- DIST ---
                total_dist = r["distance_landed"] + b["distance_landed"]
                if total_dist > 0:
                    red_dist_share = r["distance_landed"] / total_dist
                    exp_dist = _elo_expected(r_rating["dist"], b_rating["dist"])
                    d_r = K_BASE * (red_dist_share - exp_dist)
                    d_b = K_BASE * ((1 - red_dist_share) - (1 - exp_dist))
                    r_rating["dist"] += d_r * (red_age_factor if d_r > 0 else 1.0)
                    b_rating["dist"] += d_b * (blue_age_factor if d_b > 0 else 1.0)

                # --- GND ---
                total_gnd = r["ground_landed"] + b["ground_landed"]
                if total_gnd > 0:
                    red_gnd_share = r["ground_landed"] / total_gnd
                    exp_gnd = _elo_expected(r_rating["gnd"], b_rating["gnd"])
                    K_gnd = K_BASE * 1.2
                    g_r = K_gnd * (red_gnd_share - exp_gnd)
                    g_b = K_gnd * ((1 - red_gnd_share) - (1 - exp_gnd))
                    r_rating["gnd"] += g_r * (red_age_factor if g_r > 0 else 1.0)
                    b_rating["gnd"] += g_b * (blue_age_factor if g_b > 0 else 1.0)

                fighter_round_count[red_id] += 1
                fighter_round_count[blue_id] += 1
                rounds_processed += 1

        log.info(f"  Processed {rounds_processed} rounds")

        # --- Apply inactivity decay ---
        log.info("  Applying inactivity decay...")
        today = date.today()
        for fid in ratings:
            last_fight = fighter_last_fight_date.get(fid)
            if not last_fight:
                continue
            days_inactive = (today - last_fight).days
            if days_inactive > 180:
                decay_months = (days_inactive - 180) / 30
                decay_factor = DECAY_RATE ** decay_months
                for dim in DIMENSIONS:
                    if ratings[fid][dim] > 0:
                        ratings[fid][dim] *= decay_factor

        # --- Compute composite scores and store rankings ---
        log.info("  Computing rankings...")

        DIM_WEIGHTS = {
            "pts": 3.0,
            "ko": 1.5, "kod": 1.5,
            "sub": 1.2, "subd": 1.2,
            "td": 1.3, "tdd": 1.3,
            "ctrl": 1.0, "ctrld": 1.0,
            "str_vol": 1.0, "str_acc": 0.8,
            "dist": 0.7, "gnd": 0.8,
        }

        cutoff = today - timedelta(days=1095)

        db.query(UFCFighterRanking).delete()
        db.commit()

        total_ranked = 0
        for wc in WEIGHT_CLASS_ORDER:
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
                r = ratings[fid]
                scores[fid] = sum(r[d] * DIM_WEIGHTS[d] for d in DIMENSIONS)

            ranked = sorted(wc_fids, key=lambda f: scores[f], reverse=True)

            for rank, fid in enumerate(ranked, 1):
                r = ratings[fid]
                profile = {d: round(r[d], 2) for d in DIMENSIONS}
                profile["composite"] = round(scores[fid], 2)

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

        # --- Pound-for-pound rankings ---
        log.info("  Computing P4P rankings...")
        mens_wcs = [w for w in WEIGHT_CLASS_ORDER if not w.startswith("w_")]
        womens_wcs = [w for w in WEIGHT_CLASS_ORDER if w.startswith("w_")]

        for p4p_key, wc_list, label in [("p4p_men", mens_wcs, "Men's P4P"), ("p4p_women", womens_wcs, "Women's P4P")]:
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

            p4p_scores = {}
            for fid in p4p_fids:
                r = ratings[fid]
                p4p_scores[fid] = sum(r[d] * DIM_WEIGHTS[d] for d in DIMENSIONS)

            p4p_ranked = sorted(p4p_fids, key=lambda f: p4p_scores[f], reverse=True)[:25]

            for rank, fid in enumerate(p4p_ranked, 1):
                r = ratings[fid]
                profile = {d: round(r[d], 2) for d in DIMENSIONS}
                profile["composite"] = round(p4p_scores[fid], 2)

                db.add(UFCFighterRanking(
                    fighter_id=int(fid),
                    weight_class=p4p_key,
                    rank=rank,
                    score=round(p4p_scores[fid], 2),
                    expected_wins=round(p4p_scores[fid], 2),
                    total_opponents=len(p4p_ranked) - 1,
                    feature_profile=json.dumps(profile),
                ))
                total_ranked += 1

            f = fighter_info.get(p4p_ranked[0])
            if f:
                log.info(f"    {label}: #1 {f.first_name} {f.last_name} (score={p4p_scores[p4p_ranked[0]]:.1f})")

            db.commit()

        log.info(f"  Ranked {total_ranked} fighters across all divisions")
        log.info("  Rankings generation complete")

    finally:
        db.close()


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
            "method": "round_elo_v2",
            "dimensions": DIMENSIONS,
            "min_rounds": MIN_ROUNDS,
        }

    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_rankings()
