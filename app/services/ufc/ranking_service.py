"""
UFC Fighter Rankings — Glicko dimension profiles + radar chart data.

Delegates Glicko computation to glicko_service.py, then builds per-weight-class
dimension profiles and saves them to UFCFighterRanking for frontend display.

Rank ordering is handled by points_ranking_service.py which runs after this.

Run: python -m app.services.ranking_service
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta

from app.database import SessionLocal
from app.models.ufc import UFCFighterRanking
from app.services.ufc.glicko_service import (
    DIMENSIONS,
    WEIGHT_CLASS_ORDER,
    WEIGHT_CLASS_LABELS,
    MIN_ROUNDS,
    GlickoParams,
    compute_and_save_snapshots,
)

# Re-exported for model.py and points_ranking_service.py
__all__ = ["DIMENSIONS", "generate_rankings"]

log = logging.getLogger("ranking_service")


def generate_rankings():
    """Compute Glicko dimension ratings, save snapshots (for ML) and
    dimension profiles to UFCFighterRanking (for radar chart display).

    Rank ordering is handled by points_ranking_service.py which runs after this.
    """
    log.info("=" * 60)
    log.info("COMPUTING GLICKO DIMENSION RATINGS")
    log.info("=" * 60)

    db = SessionLocal()

    try:
        ratings, fighter_round_count, fighter_fight_count, \
            fighter_last_fight_date, fighter_weight_class, sorted_fight_ids = \
            compute_and_save_snapshots(db)

        # --- Save dimension profiles to UFCFighterRanking ---
        # These are temporary rows; points_ranking_service.py will read the
        # feature_profile JSON, delete these rows, and re-create them with
        # the correct Points+Elo rank ordering.
        log.info("  Saving Glicko dimension profiles...")
        today = date.today()
        cutoff = today - timedelta(days=548)

        db.query(UFCFighterRanking).delete()
        db.commit()

        total_saved = 0
        for wc in WEIGHT_CLASS_ORDER:
            if wc.startswith("p4p"):
                continue

            wc_fids = [
                fid for fid, w in fighter_weight_class.items()
                if w == wc
                and fighter_round_count.get(fid, 0) >= MIN_ROUNDS
                and fighter_last_fight_date.get(fid, date.min) >= cutoff
                and fid in ratings
            ]
            if not wc_fids:
                continue

            # Compute per-dimension min/max for percentile normalization (0-99)
            dim_mins = {}
            dim_maxs = {}
            for d in DIMENSIONS:
                vals = [ratings[fid][d][0] for fid in wc_fids]
                dim_mins[d] = min(vals)
                dim_maxs[d] = max(vals)

            for fid in wc_fids:
                profile = {}
                for d in DIMENSIONS:
                    raw = ratings[fid][d][0]
                    rng = dim_maxs[d] - dim_mins[d]
                    if rng > 0:
                        profile[d] = round((raw - dim_mins[d]) / rng * 99, 1)
                    else:
                        profile[d] = 50.0
                avg_sigma = sum(ratings[fid][d][1] for d in DIMENSIONS) / len(DIMENSIONS)
                profile["uncertainty"] = round(avg_sigma, 1)

                db.add(UFCFighterRanking(
                    fighter_id=int(fid),
                    weight_class=wc,
                    rank=0,  # placeholder — points_ranking_service sets real ranks
                    score=0.0,
                    expected_wins=0.0,
                    total_opponents=len(wc_fids) - 1,
                    feature_profile=json.dumps(profile),
                ))
                total_saved += 1

            db.commit()

        log.info(f"  Saved {total_saved} dimension profiles")
        log.info("  Glicko computation complete")

    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_rankings()
