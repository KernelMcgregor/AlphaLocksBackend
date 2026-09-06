"""Glicko dimension profiles for the radar chart.

Owns `ufc_glicko_snapshots` (the ML feature table) and nothing else. It used to also
write `ufc_fighter_rankings` with `rank=0` placeholders on the assumption that
points_ranking_service would immediately overwrite every row — see ranking_publisher for
why that assumption failed on the live site. Ranks now have exactly one writer, and this
module returns profiles in memory instead of persisting them.

Run: python -m app.services.ufc.ranking_publisher
"""

from __future__ import annotations

import logging
from datetime import date

from app.services.ufc.fighter_registry import Eligibility, is_rankable
from app.services.ufc.glicko_service import (
    DIMENSIONS,
    GlickoParams,
    compute_and_save_snapshots,
)

__all__ = ["DIMENSIONS", "compute_dimension_profiles"]

log = logging.getLogger("ranking_service")


def _percentile_profile(values: dict[int, float]) -> dict[int, float]:
    """Ordinal percentile rank within the division, 0-99.

    Was min-max normalisation, which the UI has always labelled "percentile" and which is
    outlier-dominated: one Charles Oliveira at the top of SUB compresses the entire rest
    of the division toward 0, so the radar chart showed a division of non-grapplers.
    """
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    n = len(ordered)
    if n == 1:
        return {ordered[0][0]: 50.0}
    out: dict[int, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        # Ties share the midpoint of the positions they span.
        pct = ((i + j) / 2) / (n - 1) * 99
        for k in range(i, j + 1):
            out[ordered[k][0]] = round(pct, 1)
        i = j + 1
    return out


def compute_dimension_profiles(
    db,
    registry: dict,
    as_of: date | None = None,
    crit: Eligibility | None = None,
    params: GlickoParams | None = None,
) -> dict[tuple[int, str], dict]:
    """Compute Glicko ratings, persist ML snapshots, and return radar profiles.

    Returns {(fighter_id, division): {dim: percentile, ..., "uncertainty": sigma}}.
    """
    today = as_of or date.today()
    crit = crit or Eligibility()

    log.info("  Computing Glicko dimension ratings...")
    ratings, _round_count, _fight_count, _last_date, _wc, _fids = \
        compute_and_save_snapshots(db, params)

    # Division and eligibility come from the registry, never re-derived here. That
    # divergence is exactly what produced rank=0 rows.
    by_division: dict[str, list[int]] = {}
    for fid, st in registry.items():
        if fid in ratings and is_rankable(st, today, crit):
            by_division.setdefault(st.division, []).append(fid)

    profiles: dict[tuple[int, str], dict] = {}
    for division, fids in by_division.items():
        for dim in DIMENSIONS:
            pct = _percentile_profile({f: ratings[f][dim][0] for f in fids})
            for f, v in pct.items():
                profiles.setdefault((f, division), {})[dim] = v
        for f in fids:
            sigmas = [ratings[f][d][1] for d in DIMENSIONS]
            profiles[(f, division)]["uncertainty"] = round(sum(sigmas) / len(sigmas), 1)

    log.info(f"  Built {len(profiles)} dimension profiles across {len(by_division)} divisions")
    return profiles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from app.database import SessionLocal
    from app.services.ufc.ranking_publisher import publish_rankings

    _db = SessionLocal()
    try:
        publish_rankings(_db)
    finally:
        _db.close()
