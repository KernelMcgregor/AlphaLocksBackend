"""The only module that writes `ufc_fighter_rankings`.

Rank used to have two owners. `ranking_service` deleted the table and wrote every
fighter at `rank=0` as a placeholder; `points_ranking_service` then deleted it again and
wrote the real ranks. Correctness depended entirely on the second one running to
completion, in the same process, over the same set of fighters. It did not:
`main.py` wrapped each in its own `try/except`, so a Points failure logged an error and
left a table full of zeros — and because 0 sorts ahead of 1, those rows appeared at the
TOP of every division.

There were also two delete+commit windows in which `/ufc/rankings` could observe an empty
or half-written table. Both are gone: one transaction, and a set of invariants checked
before it commits rather than discovered on the site.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

from app.models.ufc import UFCFighterRanking
from app.services.ufc.fighter_registry import (
    Eligibility, build_fighter_registry, is_rankable,
)

log = logging.getLogger("ranking_publisher")

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


@dataclass
class RankingResult:
    """What a ranker returns. Ordering only — normalisation and persistence are the
    publisher's job, so every ranker is comparable on the same scale."""

    order: dict[str, list[int]]              #: division -> fighter_ids, best first
    scores: dict[int, float] = field(default_factory=dict)   #: raw, pre-normalisation
    extras: dict[int, dict] = field(default_factory=dict)    #: merged into feature_profile


class RankingIntegrityError(RuntimeError):
    """Raised instead of committing a ranking that would repeat a known failure."""


def _check(result: RankingResult, profiles: dict, registry: dict,
           today: date, crit: Eligibility) -> None:
    """Everything that has previously reached production and should not again."""
    for division, fids in result.order.items():
        if not fids:
            continue
        if len(set(fids)) != len(fids):
            raise RankingIntegrityError(f"{division}: duplicate fighter in ordering")
        for fid in fids:
            if division.startswith("p4p"):
                continue
            st = registry.get(fid)
            if st is None:
                raise RankingIntegrityError(
                    f"{division}: fighter {fid} is not in the registry")
            if not is_rankable(st, today, crit):
                raise RankingIntegrityError(
                    f"{division}: fighter {fid} was ranked but is not rankable "
                    f"(fights={st.decided_fights} rounds={st.rounds} "
                    f"last_activity={st.last_activity})")
            if st.division != division:
                raise RankingIntegrityError(
                    f"fighter {fid} ranked in {division} but registry says "
                    f"{st.division} — the division disagreement that produced rank=0 rows")
            if (fid, division) not in profiles:
                raise RankingIntegrityError(
                    f"{division}: fighter {fid} has no dimension profile; the radar "
                    "chart would render all 15 dimensions as 0")


def publish_rankings(db, ranker=None, as_of: date | None = None,
                     crit: Eligibility | None = None, preview: bool = False) -> dict:
    """Compute, verify and atomically publish rankings for every division."""
    from app.services.ufc.points_ranking_service import PointsEloRanker
    from app.services.ufc.ranking_service import compute_dimension_profiles

    ranker = ranker or PointsEloRanker()
    today = as_of or date.today()
    crit = crit or Eligibility()

    log.info("=" * 60)
    log.info(f"PUBLISHING RANKINGS  ranker={getattr(ranker, 'name', type(ranker).__name__)}"
             f"  as_of={today}")
    log.info("=" * 60)

    registry = build_fighter_registry(db)
    profiles = compute_dimension_profiles(db, registry, today, crit)
    result = ranker.rank(db, registry, today, crit)

    _check(result, profiles, registry, today, crit)

    rows, ranked_total = [], 0
    for division in WEIGHT_CLASS_ORDER:
        fids = result.order.get(division) or []
        if len(fids) < 2:
            continue

        # Normalise to 0-1000 within the division so every ranker lands on one scale.
        vals = [result.scores.get(f, 0.0) for f in fids]
        s_max, s_min = max(vals), min(vals)
        span = (s_max - s_min) or 1.0

        for rank, fid in enumerate(fids, 1):
            profile = dict(profiles.get((fid, division))
                           or profiles.get((fid, registry[fid].division), {}))
            profile.update(result.extras.get(fid, {}))
            rows.append(UFCFighterRanking(
                fighter_id=int(fid),
                weight_class=division,
                rank=rank,
                score=round((result.scores.get(fid, 0.0) - s_min) / span * 1000, 1),
                expected_wins=round((result.scores.get(fid, 0.0) - s_min) / span * 1000, 1),
                total_opponents=len(fids) - 1,
                feature_profile=json.dumps(profile),
            ))
            ranked_total += 1

        log.info(f"    {WEIGHT_CLASS_LABELS.get(division, division):<20} {len(fids):>4} ranked")

    if any(r.rank <= 0 for r in rows):
        raise RankingIntegrityError("a row was built with rank<=0")

    if preview:
        log.info(f"  PREVIEW — {ranked_total} rows built, nothing written")
        return {"rows": rows, "result": result, "registry": registry}

    # One transaction. The previous delete/commit + insert/commit pair left a window in
    # which the API could read an empty table.
    db.query(UFCFighterRanking).delete()
    db.bulk_save_objects(rows)
    db.commit()
    log.info(f"  Published {ranked_total} rows across "
             f"{len({r.weight_class for r in rows})} divisions")
    return {"rows": rows, "result": result, "registry": registry}
