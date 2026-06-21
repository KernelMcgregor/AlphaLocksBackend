"""
Fight Stats Derived — Populate derived columns on ufc_fight_stats.

For each fight's totals rows (round_number=0), computes per-minute rates,
accuracy percentages, distribution splits, defense rates, and position-aware
metrics from the raw counts + opponent's raw counts + fight time.

Run: python -m app.services.fight_stats_derived_service
"""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlalchemy import text

from app.database import SessionLocal

log = logging.getLogger("fight_stats_derived")


def _safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def _rate15(count: float, minutes: float) -> float | None:
    return _safe_div(count * 15, minutes)


def _rate_pm(count: float, minutes: float) -> float | None:
    return _safe_div(count, minutes)


def _pct(landed: float, attempted: float) -> float | None:
    return _safe_div(landed, attempted)


def _dist(part: float, total: float) -> float | None:
    return _safe_div(part, total)


def _def_rate(opp_landed: float, opp_attempted: float) -> float | None:
    if not opp_attempted:
        return None
    return 1.0 - (opp_landed / opp_attempted)


def compute_all_derived_stats() -> int:
    """Populate derived columns on all totals rows. Returns rows updated."""
    db = SessionLocal()
    try:
        return _compute(db)
    finally:
        db.close()


def _compute(db) -> int:
    # Load fight times
    fights = db.execute(text("""
        SELECT id, fight_time_seconds FROM ufc_fights
        WHERE fight_time_seconds IS NOT NULL AND fight_time_seconds > 0
    """)).fetchall()
    fight_time = {f.id: f.fight_time_seconds for f in fights}

    # Load all totals rows
    stats = db.execute(text("""
        SELECT id, fight_id, fighter_id,
               kd, sig_str_landed, sig_str_attempted,
               total_str_landed, total_str_attempted,
               td_landed, td_attempted, sub_att, rev, ctrl_seconds,
               head_landed, head_attempted, body_landed, body_attempted,
               leg_landed, leg_attempted, distance_landed, distance_attempted,
               clinch_landed, clinch_attempted, ground_landed, ground_attempted
        FROM ufc_fight_stats
        WHERE round_number = 0
    """)).fetchall()

    # Index by (fight_id, fighter_id) and group by fight_id
    by_ff = {}
    by_fight = defaultdict(list)
    for s in stats:
        by_ff[(s.fight_id, s.fighter_id)] = s
        by_fight[s.fight_id].append(s)

    updated = 0

    for fight_id, rows in by_fight.items():
        secs = fight_time.get(fight_id)
        if not secs:
            continue
        mins = secs / 60.0

        # Need exactly 2 rows (one per fighter) to compute opponent stats
        if len(rows) != 2:
            continue

        for own in rows:
            # Find opponent
            opp = [r for r in rows if r.fighter_id != own.fighter_id]
            if not opp:
                continue
            opp = opp[0]

            # Position time estimates
            own_ctrl = own.ctrl_seconds or 0
            opp_ctrl = opp.ctrl_seconds or 0
            est_ground_min = (own_ctrl + opp_ctrl) / 60.0
            est_standing_min = max(mins - est_ground_min, 0.0)

            o = own  # own stats
            p = opp  # opponent stats

            vals = {
                "fight_time_min": round(mins, 4),
                "est_standing_min": round(est_standing_min, 4),
                "est_ground_min": round(est_ground_min, 4),

                # Striking overall
                "slpm": _rate_pm(o.sig_str_landed, mins),
                "sapm": _rate_pm(p.sig_str_landed, mins),
                "sl_diff": _rate_pm((o.sig_str_landed or 0) - (p.sig_str_landed or 0), mins),
                "sig_acc": _pct(o.sig_str_landed, o.sig_str_attempted),
                "sig_def": _def_rate(p.sig_str_landed, p.sig_str_attempted),
                "tslpm": _rate_pm(o.total_str_landed, mins),

                # Head
                "head_pct": _dist(o.head_landed, o.sig_str_landed),
                "head_pm": _rate_pm(o.head_landed, mins),
                "head_acc": _pct(o.head_landed, o.head_attempted),
                "head_abs_pct": _dist(p.head_landed, p.sig_str_landed),
                "head_abs_pm": _rate_pm(p.head_landed, mins),
                "head_def": _def_rate(p.head_landed, p.head_attempted),

                # Body
                "body_pct": _dist(o.body_landed, o.sig_str_landed),
                "body_pm": _rate_pm(o.body_landed, mins),
                "body_acc": _pct(o.body_landed, o.body_attempted),
                "body_abs_pct": _dist(p.body_landed, p.sig_str_landed),
                "body_abs_pm": _rate_pm(p.body_landed, mins),
                "body_def": _def_rate(p.body_landed, p.body_attempted),

                # Legs
                "leg_pct": _dist(o.leg_landed, o.sig_str_landed),
                "leg_pm": _rate_pm(o.leg_landed, mins),
                "leg_acc": _pct(o.leg_landed, o.leg_attempted),
                "leg_abs_pct": _dist(p.leg_landed, p.sig_str_landed),
                "leg_abs_pm": _rate_pm(p.leg_landed, mins),
                "leg_def": _def_rate(p.leg_landed, p.leg_attempted),

                # Distance
                "dist_pct": _dist(o.distance_landed, o.sig_str_landed),
                "dist_pm": _rate_pm(o.distance_landed, mins),
                "dist_acc": _pct(o.distance_landed, o.distance_attempted),
                "dist_abs_pct": _dist(p.distance_landed, p.sig_str_landed),
                "dist_abs_pm": _rate_pm(p.distance_landed, mins),
                "dist_def": _def_rate(p.distance_landed, p.distance_attempted),

                # Clinch
                "clinch_pct": _dist(o.clinch_landed, o.sig_str_landed),
                "clinch_pm": _rate_pm(o.clinch_landed, mins),
                "clinch_acc": _pct(o.clinch_landed, o.clinch_attempted),
                "clinch_abs_pct": _dist(p.clinch_landed, p.sig_str_landed),
                "clinch_abs_pm": _rate_pm(p.clinch_landed, mins),
                "clinch_def": _def_rate(p.clinch_landed, p.clinch_attempted),

                # Ground
                "ground_pct": _dist(o.ground_landed, o.sig_str_landed),
                "ground_pm": _rate_pm(o.ground_landed, mins),
                "ground_acc": _pct(o.ground_landed, o.ground_attempted),
                "ground_abs_pct": _dist(p.ground_landed, p.sig_str_landed),
                "ground_abs_pm": _rate_pm(p.ground_landed, mins),
                "ground_def": _def_rate(p.ground_landed, p.ground_attempted),
                "gnp15g": _rate15(o.ground_landed, est_ground_min),
                "gnp_abs15g": _rate15(p.ground_landed, est_ground_min),

                # Knockdowns
                "kd15": _rate15(o.kd, mins),
                "kd15s": _rate15(o.kd, est_standing_min),
                "kd_abs15": _rate15(p.kd, mins),
                "kd_abs15s": _rate15(p.kd, est_standing_min),

                # Takedowns
                "td15": _rate15(o.td_landed, mins),
                "td15s": _rate15(o.td_landed, est_standing_min),
                "td_acc": _pct(o.td_landed, o.td_attempted),
                "td_abs15": _rate15(p.td_landed, mins),
                "td_abs15s": _rate15(p.td_landed, est_standing_min),
                "td_def": _def_rate(p.td_landed, p.td_attempted),

                # Control
                "ctrl15": _rate15(own_ctrl, mins),
                "ctrl15g": _rate15(own_ctrl, est_ground_min),
                "ctrl_abs15": _rate15(opp_ctrl, mins),
                "ctrl_abs15g": _rate15(opp_ctrl, est_ground_min),

                # Submissions
                "sub_att15": _rate15(o.sub_att, mins),
                "sub_att15g": _rate15(o.sub_att, est_ground_min),
                "sub_abs15": _rate15(p.sub_att, mins),
                "sub_abs15g": _rate15(p.sub_att, est_ground_min),

                # Reversals
                "rev15": _rate15(o.rev, mins),
                "rev_abs15": _rate15(p.rev, mins),
            }

            # Build UPDATE statement
            set_clauses = ", ".join(f"{k} = :{k}" for k in vals)
            db.execute(
                text(f"UPDATE ufc_fight_stats SET {set_clauses} WHERE id = :row_id"),
                {**vals, "row_id": own.id},
            )
            updated += 1

    db.commit()
    log.info("Updated derived stats on %d fight stat rows", updated)
    return updated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = compute_all_derived_stats()
    print(f"Done — updated {n} fight stat rows with derived columns")
