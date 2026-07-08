"""
Fighter Career Stats — Compute all 76 derived career statistics.

Aggregates raw per-fight stats (own + opponent) into rate, distribution,
accuracy, and position-aware metrics. Results are stored in
ufc_fighter_career_stats for use as model features and API responses.

Run: python -m app.services.career_stats_service
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import text

from app.database import SessionLocal
from app.models.ufc import UFCFighterCareerStats

log = logging.getLogger("career_stats_service")

# Fields to sum from fight stats rows
STAT_FIELDS = [
    "kd", "sig_str_landed", "sig_str_attempted",
    "total_str_landed", "total_str_attempted",
    "td_landed", "td_attempted", "sub_att", "rev", "ctrl_seconds",
    "head_landed", "head_attempted", "body_landed", "body_attempted",
    "leg_landed", "leg_attempted", "distance_landed", "distance_attempted",
    "clinch_landed", "clinch_attempted", "ground_landed", "ground_attempted",
]


def _safe_div(num: float, den: float) -> float | None:
    """Divide, returning None if denominator is zero."""
    return num / den if den else None


def _rate15(count: float, minutes: float) -> float | None:
    """Per-15-minute rate."""
    return _safe_div(count * 15, minutes)


def _rate_pm(count: float, minutes: float) -> float | None:
    """Per-minute rate."""
    return _safe_div(count, minutes)


def _pct(landed: float, attempted: float) -> float | None:
    """Accuracy / percentage."""
    return _safe_div(landed, attempted)


def _dist(part: float, total: float) -> float | None:
    """Distribution percentage."""
    return _safe_div(part, total)


def _def(opp_landed: float, opp_attempted: float) -> float | None:
    """Defense rate = 1 - opponent accuracy."""
    if not opp_attempted:
        return None
    return 1.0 - (opp_landed / opp_attempted)


def compute_all_career_stats() -> int:
    """Compute career stats for every fighter with fight data. Returns count of fighters updated."""
    db = SessionLocal()
    try:
        return _compute(db)
    finally:
        db.close()


def _compute(db) -> int:
    # Pull all fights with time data
    fights_q = db.execute(text("""
        SELECT id, red_fighter_id, blue_fighter_id, winner_id,
               method, fight_time_seconds
        FROM ufc.ufc_fights
        WHERE fight_time_seconds IS NOT NULL AND fight_time_seconds > 0
    """)).fetchall()

    fight_map = {}  # fight_id -> fight row
    fighter_fight_ids = defaultdict(set)  # fighter_id -> set of fight_ids
    for f in fights_q:
        fight_map[f.id] = f
        fighter_fight_ids[f.red_fighter_id].add(f.id)
        fighter_fight_ids[f.blue_fighter_id].add(f.id)

    # Pull all total-row stats (round_number=0)
    stats_q = db.execute(text("""
        SELECT fight_id, fighter_id, round_number,
               kd, sig_str_landed, sig_str_attempted,
               total_str_landed, total_str_attempted,
               td_landed, td_attempted, sub_att, rev, ctrl_seconds,
               head_landed, head_attempted, body_landed, body_attempted,
               leg_landed, leg_attempted, distance_landed, distance_attempted,
               clinch_landed, clinch_attempted, ground_landed, ground_attempted
        FROM ufc.ufc_fight_stats
        WHERE round_number = 0
    """)).fetchall()

    # Index stats by (fight_id, fighter_id)
    stats_by_fight_fighter = {}
    for s in stats_q:
        stats_by_fight_fighter[(s.fight_id, s.fighter_id)] = s

    count = 0
    now = datetime.now(timezone.utc)

    for fighter_id, fight_ids in fighter_fight_ids.items():
        own_totals = defaultdict(float)
        opp_totals = defaultdict(float)
        total_secs = 0.0
        ko_wins = 0
        sub_wins = 0
        dec_wins = 0
        wins = 0
        losses = 0
        fight_count = 0
        fight_secs_list = []

        for fid in fight_ids:
            fight = fight_map[fid]

            # Determine opponent
            if fight.red_fighter_id == fighter_id:
                opp_id = fight.blue_fighter_id
            else:
                opp_id = fight.red_fighter_id

            own_stats = stats_by_fight_fighter.get((fid, fighter_id))
            opp_stats = stats_by_fight_fighter.get((fid, opp_id))
            if not own_stats:
                continue

            fight_count += 1
            total_secs += fight.fight_time_seconds
            fight_secs_list.append(fight.fight_time_seconds)

            # Sum own stats
            for field in STAT_FIELDS:
                own_totals[field] += getattr(own_stats, field) or 0

            # Sum opponent stats (for absorbed/defense)
            if opp_stats:
                for field in STAT_FIELDS:
                    opp_totals[field] += getattr(opp_stats, field) or 0

            # Outcomes
            if fight.winner_id == fighter_id:
                wins += 1
                m = (fight.method or "").lower()
                if "ko" in m or "tko" in m:
                    ko_wins += 1
                elif "sub" in m:
                    sub_wins += 1
                else:
                    dec_wins += 1
            elif fight.winner_id is not None:
                losses += 1

        if fight_count == 0:
            continue

        total_min = total_secs / 60.0

        # Position time estimates
        own_ctrl = own_totals["ctrl_seconds"]
        opp_ctrl = opp_totals["ctrl_seconds"]
        est_ground_min = (own_ctrl + opp_ctrl) / 60.0
        est_standing_min = max(total_min - est_ground_min, 0.0)

        # Shorthand accessors
        o = own_totals
        p = opp_totals  # opponent

        stats = UFCFighterCareerStats(
            fighter_id=fighter_id,
            fight_count=fight_count,
            total_fight_min=round(total_min, 2),
            est_standing_min=round(est_standing_min, 2),
            est_ground_min=round(est_ground_min, 2),

            # Striking: Overall
            slpm=_rate_pm(o["sig_str_landed"], total_min),
            sapm=_rate_pm(p["sig_str_landed"], total_min),
            sl_diff=_rate_pm(o["sig_str_landed"] - p["sig_str_landed"], total_min),
            sig_acc=_pct(o["sig_str_landed"], o["sig_str_attempted"]),
            sig_def=_def(p["sig_str_landed"], p["sig_str_attempted"]),
            tslpm=_rate_pm(o["total_str_landed"], total_min),

            # Head offense
            head_pct=_dist(o["head_landed"], o["sig_str_landed"]),
            head_pm=_rate_pm(o["head_landed"], total_min),
            head_acc=_pct(o["head_landed"], o["head_attempted"]),
            # Head defense
            head_abs_pct=_dist(p["head_landed"], p["sig_str_landed"]),
            head_abs_pm=_rate_pm(p["head_landed"], total_min),
            head_def=_def(p["head_landed"], p["head_attempted"]),

            # Body offense
            body_pct=_dist(o["body_landed"], o["sig_str_landed"]),
            body_pm=_rate_pm(o["body_landed"], total_min),
            body_acc=_pct(o["body_landed"], o["body_attempted"]),
            # Body defense
            body_abs_pct=_dist(p["body_landed"], p["sig_str_landed"]),
            body_abs_pm=_rate_pm(p["body_landed"], total_min),
            body_def=_def(p["body_landed"], p["body_attempted"]),

            # Leg offense
            leg_pct=_dist(o["leg_landed"], o["sig_str_landed"]),
            leg_pm=_rate_pm(o["leg_landed"], total_min),
            leg_acc=_pct(o["leg_landed"], o["leg_attempted"]),
            # Leg defense
            leg_abs_pct=_dist(p["leg_landed"], p["sig_str_landed"]),
            leg_abs_pm=_rate_pm(p["leg_landed"], total_min),
            leg_def=_def(p["leg_landed"], p["leg_attempted"]),

            # Distance offense
            dist_pct=_dist(o["distance_landed"], o["sig_str_landed"]),
            dist_pm=_rate_pm(o["distance_landed"], total_min),
            dist_acc=_pct(o["distance_landed"], o["distance_attempted"]),
            # Distance defense
            dist_abs_pct=_dist(p["distance_landed"], p["sig_str_landed"]),
            dist_abs_pm=_rate_pm(p["distance_landed"], total_min),
            dist_def=_def(p["distance_landed"], p["distance_attempted"]),

            # Clinch offense
            clinch_pct=_dist(o["clinch_landed"], o["sig_str_landed"]),
            clinch_pm=_rate_pm(o["clinch_landed"], total_min),
            clinch_acc=_pct(o["clinch_landed"], o["clinch_attempted"]),
            # Clinch defense
            clinch_abs_pct=_dist(p["clinch_landed"], p["sig_str_landed"]),
            clinch_abs_pm=_rate_pm(p["clinch_landed"], total_min),
            clinch_def=_def(p["clinch_landed"], p["clinch_attempted"]),

            # Ground offense
            ground_pct=_dist(o["ground_landed"], o["sig_str_landed"]),
            ground_pm=_rate_pm(o["ground_landed"], total_min),
            ground_acc=_pct(o["ground_landed"], o["ground_attempted"]),
            # Ground defense
            ground_abs_pct=_dist(p["ground_landed"], p["sig_str_landed"]),
            ground_abs_pm=_rate_pm(p["ground_landed"], total_min),
            ground_def=_def(p["ground_landed"], p["ground_attempted"]),
            # Ground position-aware
            gnp15g=_rate15(o["ground_landed"], est_ground_min),
            gnp_abs15g=_rate15(p["ground_landed"], est_ground_min),

            # Knockdowns
            kd15=_rate15(o["kd"], total_min),
            kd15s=_rate15(o["kd"], est_standing_min),
            kd_abs15=_rate15(p["kd"], total_min),
            kd_abs15s=_rate15(p["kd"], est_standing_min),

            # Takedowns
            td15=_rate15(o["td_landed"], total_min),
            td15s=_rate15(o["td_landed"], est_standing_min),
            td_acc=_pct(o["td_landed"], o["td_attempted"]),
            td_abs15=_rate15(p["td_landed"], total_min),
            td_abs15s=_rate15(p["td_landed"], est_standing_min),
            td_def=_def(p["td_landed"], p["td_attempted"]),

            # Control time
            ctrl15=_rate15(own_ctrl, total_min),
            ctrl15g=_rate15(own_ctrl, est_ground_min),
            ctrl_abs15=_rate15(opp_ctrl, total_min),
            ctrl_abs15g=_rate15(opp_ctrl, est_ground_min),

            # Submissions
            sub_att15=_rate15(o["sub_att"], total_min),
            sub_att15g=_rate15(o["sub_att"], est_ground_min),
            sub_abs15=_rate15(p["sub_att"], total_min),
            sub_abs15g=_rate15(p["sub_att"], est_ground_min),

            # Reversals
            rev15=_rate15(o["rev"], total_min),
            rev_abs15=_rate15(p["rev"], total_min),

            # Outcomes
            ko_wins=ko_wins,
            sub_wins=sub_wins,
            dec_wins=dec_wins,
            finish_rate=_pct(ko_wins + sub_wins, ko_wins + sub_wins + dec_wins),
            win_pct=_pct(wins, wins + losses),
            avg_fight_sec=sum(fight_secs_list) / len(fight_secs_list) if fight_secs_list else None,

            computed_at=now,
        )

        # Upsert: delete existing then insert
        db.query(UFCFighterCareerStats).filter(
            UFCFighterCareerStats.fighter_id == fighter_id
        ).delete()
        db.add(stats)
        count += 1

    db.commit()
    log.info("Computed career stats for %d fighters", count)
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = compute_all_career_stats()
    print(f"Done — computed career stats for {n} fighters")
