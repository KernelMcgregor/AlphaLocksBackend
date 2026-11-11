"""The only module that writes `ufc_alt_rankings` (BMF and P4P), and the read side the
API serves.

Same discipline as ranking_publisher: compute everything, check invariants, then replace
the published rows in ONE transaction, so /ufc/alt-rankings never sees a half-written
table.

Run:
    python -m app.services.ufc.alt_rankings_publisher --preview   # read-only, prints top 25
    python -m app.services.ufc.alt_rankings_publisher             # publishes (writes prod)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date

from app.database import SessionLocal
from app.services.ufc import bmf_rankings, p4p_rankings
from app.services.ufc.alt_rankings_common import (
    POOLS, eligible_fighters, is_eligible, load_context,
)

log = logging.getLogger("alt_rankings")

KINDS = {
    "bmf": {"label": "BMF", "module": bmf_rankings, "compute": bmf_rankings.compute_bmf},
    "p4p": {"label": "Pound for pound", "module": p4p_rankings,
            "compute": p4p_rankings.compute_p4p},
}
MIN_POOL = 10


class AltRankingIntegrityError(RuntimeError):
    pass


def check(kind: str, rows: list[dict], registry: dict, today: date,
          champions: set[int] = frozenset()) -> None:
    """Refuse to publish anything that would look wrong on the site."""
    seen = set()
    per_pool = {p: 0 for p in POOLS}
    for r in rows:
        key = (r["fighter_id"], r["pool"])
        if key in seen:
            raise AltRankingIntegrityError(f"{kind}: duplicate fighter {key}")
        seen.add(key)
        if r["pool"] not in POOLS:
            raise AltRankingIntegrityError(f"{kind}: unknown pool {r['pool']}")
        per_pool[r["pool"]] += 1
        for k, v in r["components"].items():
            if not (isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 100):
                raise AltRankingIntegrityError(
                    f"{kind}: fighter {r['fighter_id']} component {k}={v} outside [0, 100]")
        if not math.isfinite(r["default_score"]):
            raise AltRankingIntegrityError(f"{kind}: fighter {r['fighter_id']} score not finite")
        st = registry.get(r["fighter_id"])
        if st is None or not is_eligible(st, today, r["fighter_id"] in champions):
            raise AltRankingIntegrityError(
                f"{kind}: fighter {r['fighter_id']} ranked but not eligible")
    for p, n in per_pool.items():
        if n < MIN_POOL:
            raise AltRankingIntegrityError(f"{kind}: {p} pool has only {n} fighters")


def assign_ranks(rows: list[dict]) -> None:
    for pool in POOLS:
        members = sorted((r for r in rows if r["pool"] == pool),
                         key=lambda r: (-r["default_score"], r.get("division_rank") or 999,
                                        -r.get("raw", {}).get("score_raw", r["default_score"]),
                                        r["fighter_id"]))
        for i, r in enumerate(members, 1):
            r["default_rank"] = i


def compute_all(db, kinds=tuple(KINDS), as_of: date | None = None) -> tuple[dict, dict]:
    ctx = load_context(db, as_of)
    fighters = eligible_fighters(ctx)
    log.info(f"  {len(fighters)} eligible fighters")
    out = {}
    for kind in kinds:
        rows = KINDS[kind]["compute"](ctx, fighters)
        assign_ranks(rows)
        check(kind, rows, ctx["registry"], ctx["today"], set(ctx["champions"].values()))
        out[kind] = rows
    return out, ctx


def publish_alt_rankings(db, kinds=tuple(KINDS), as_of: date | None = None,
                         preview: bool = False) -> dict:
    from app.models.ufc import UFCAltRanking

    results, ctx = compute_all(db, kinds, as_of)
    if preview:
        _print_preview(results, ctx)
        return results

    objs = [
        UFCAltRanking(
            kind=kind, fighter_id=int(r["fighter_id"]), pool=r["pool"],
            division=r["division"], as_of=ctx["today"],
            components=json.dumps({**r["components"],
                                   "raw": {**r["raw"], "is_champion": r["is_champion"]}}),
            default_score=r["default_score"], default_rank=r["default_rank"],
            n_bouts=r["n_bouts"], last_fight_date=r["last_fight_date"],
            ledger=json.dumps(r["ledger"]),
        )
        for kind, rows in results.items() for r in rows
    ]
    db.query(UFCAltRanking).filter(UFCAltRanking.kind.in_(list(results))).delete(
        synchronize_session=False)
    db.add_all(objs)
    db.commit()
    log.info(f"  Published {len(objs)} alt-ranking rows ({', '.join(results)})")
    return results


def _print_preview(results: dict, ctx: dict, n: int = 25) -> None:
    names = ctx["hist"]["names"]
    champs = set(ctx["champions"].values())
    for kind, rows in results.items():
        keys = [c["key"] for c in KINDS[kind]["module"].COMPONENTS]
        for pool in POOLS:
            top = sorted((r for r in rows if r["pool"] == pool),
                         key=lambda r: r["default_rank"])[:n]
            print(f"\n=== {kind.upper()} — {pool} ({sum(r['pool'] == pool for r in rows)} "
                  "eligible) ===")
            print(f"{'#':>3}  {'fighter':<26}{'division':<19}{'score':>6}  "
                  + "  ".join(f"{k[:5]:>5}" for k in keys))
            for r in top:
                belt = "C" if r["fighter_id"] in champs else " "
                print(f"{r['default_rank']:>3}{belt} {names.get(r['fighter_id'], '?')[:25]:<26}"
                      f"{r['division']:<19}{r['default_score']:>6.1f}  "
                      + "  ".join(f"{r['components'][k]:>5.0f}" for k in keys))


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------

#: Alt-ranking ranks going into the last event, by baseline date:
#: {date: {(kind, pool, fighter_id): rank}}. The published table is replaced on every
#: publish and keeps no history, so the prior board is recomputed as of the division
#: history's previous snapshot (~4s). Keyed by date, it is computed once per event, not
#: once per cache refresh.
_PRIOR_RANKS: dict[date, dict[tuple[str, str, int], int]] = {}


def _prior_ranks(db, since: date) -> dict[tuple[str, str, int], int]:
    if since not in _PRIOR_RANKS:
        try:
            results, _ = compute_all(db, as_of=since)
        except Exception:  # an old board failing today's invariants must not take the page down
            log.exception("alt rankings: baseline as of %s failed; serving without movement", since)
            return {}
        _PRIOR_RANKS.clear()
        _PRIOR_RANKS[since] = {
            (kind, r["pool"], int(r["fighter_id"])): r["default_rank"]
            for kind, rows in results.items() for r in rows
        }
    return _PRIOR_RANKS[since]


def get_alt_rankings(kind: str) -> dict:
    """The /ufc/alt-rankings/{kind} payload. Opens its own session (cache thunk)."""
    from app.models.ufc import UFCAltRanking, UFCFighter
    from app.services.ufc.ranking_publisher import WEIGHT_CLASS_LABELS
    from app.services.ufc.tapology_rankings import previous_snapshot_date

    spec = KINDS[kind]
    mod = spec["module"]
    db = SessionLocal()
    try:
        rows = (
            db.query(UFCAltRanking, UFCFighter)
            .join(UFCFighter, UFCAltRanking.fighter_id == UFCFighter.id)
            .filter(UFCAltRanking.kind == kind)
            .order_by(UFCAltRanking.pool, UFCAltRanking.default_rank)
            .all()
        )
        # Movement, as on the division boards: rank now vs going into the last event.
        since = previous_snapshot_date(db)
        prior = _prior_ranks(db, since) if since else {}
        since_iso = since.isoformat() if since and prior else None

        pools: dict[str, list] = {p: [] for p in POOLS}
        as_of = None
        for r, f in rows:
            as_of = r.as_of
            comps = json.loads(r.components or "{}")
            raw = comps.pop("raw", {})
            # Stored at publish time, judged with the ranker's division rules, rather
            # than re-derived here from a single stored division.
            is_champion = bool(raw.pop("is_champion", False))
            pools.setdefault(r.pool, []).append({
                "id": str(f.id),
                "first_name": f.first_name, "last_name": f.last_name,
                "nickname": f.nickname,
                "wins": f.wins, "losses": f.losses, "draws": f.draws,
                "country_code": f.country_code, "image_url": f.image_url,
                "division": r.division,
                "division_label": WEIGHT_CLASS_LABELS.get(r.division, r.division),
                "is_champion": is_champion,
                "components": comps, "raw": raw,
                "division_rank": raw.get("division_rank"),
                "default_score": r.default_score, "default_rank": r.default_rank,
                "n_bouts": r.n_bouts,
                "last_fight_date": r.last_fight_date.isoformat() if r.last_fight_date else None,
                "ledger": json.loads(r.ledger or "[]"),
                # `delta` is POSITIVE when the fighter climbed; prior_rank None = not on
                # the board then (NEW).
                "movement": _movement(prior.get((kind, r.pool, f.id)), r.default_rank, since_iso),
            })
        return {
            "kind": kind, "label": spec["label"],
            "as_of": as_of.isoformat() if as_of else None,
            "defaults": mod.DEFAULT_WEIGHTS,
            "components": mod.COMPONENTS,
            "pools": pools,
            "movement_since": since_iso,
        }
    finally:
        db.close()


def _movement(prior_rank: int | None, rank: int, since_iso: str | None) -> dict:
    return {
        "prior_rank": prior_rank,
        "delta": (prior_rank - rank) if prior_rank is not None else None,
        "as_of": since_iso,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db = SessionLocal()
    try:
        publish_alt_rankings(db, preview="--preview" in sys.argv)
    finally:
        db.close()
