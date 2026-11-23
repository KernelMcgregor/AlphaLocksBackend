"""Check BMF weights against Fight of the Night awards. Read-only.

    PYTHONPATH=. venv/bin/python -m scripts.fit_bmf [--as-of 2024-07-01]

Builds the BMF board as of `--as-of` from fights before it, then scores each weight
setting by Spearman against each fighter's FOTN rate in their UFC fights AFTER it
(shrunk toward the overall rate; fighters with fewer than MIN_FUTURE fights left out).
FOTN rather than Performance of the Night: POTN pays for any quick finish, which is what
a BMF list should not chase. Prints component correlations, the old and current
defaults, and the best grid settings.

The excitement model's weights were fit on all cards, future ones included; they are a
handful of global weights, so the leak into this check is small.
"""
from __future__ import annotations

import argparse
import itertools
from datetime import date

import numpy as np
from scipy.stats import spearmanr

from app.database import SessionLocal
from app.services.ufc import bmf_rankings as bmf
from app.services.ufc.alt_rankings_common import eligible_fighters, load_context

MIN_FUTURE = 3
SHRINK = 3.0
OLD_WEIGHTS = {"finishing": 0.45, "toughness": 0.30, "exchange": 0.0, "excitement": 0.0,
               "recency": 0.25, "gamma": 0.5}
GRID = (0.0, 0.1, 0.2, 0.3, 0.4)
GAMMAS = (0.0, 0.5, 1.0)


def future_fotn(db, as_of: date) -> dict[int, tuple[int, int]]:
    """fighter_id -> (UFC fights after `as_of`, FOTN awards in them)."""
    from sqlalchemy import text

    rows = db.execute(text("""
        select fid, count(*) n, sum(fotn) k from (
            select f.id, unnest(array[f.red_fighter_id, f.blue_fighter_id]) fid,
                   (exists (select 1 from ufc.ufc_fight_bonuses b
                            where b.fight_id = f.id and b.bonus_type = 'fotn'))::int fotn
            from ufc.ufc_fights f
            where f.date > :d and f.date <= current_date and f.method is not null) t
        group by fid"""), {"d": as_of}).fetchall()
    return {int(fid): (int(n), int(k)) for fid, n, k in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=date.fromisoformat, default=date(2024, 7, 1))
    ap.add_argument("--try", dest="tries", action="append", default=[],
                    help="extra setting to score, e.g. finishing=.25,excitement=.3")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        ctx = load_context(db, args.as_of)
        rows = bmf.compute_bmf(ctx, eligible_fighters(ctx))
        fut = future_fotn(db, args.as_of)
    finally:
        db.close()

    n_all = sum(n for n, _ in fut.values())
    p0 = sum(k for _, k in fut.values()) / n_all
    keep = [r for r in rows if fut.get(r["fighter_id"], (0, 0))[0] >= MIN_FUTURE]
    y = np.array([(fut[r["fighter_id"]][1] + SHRINK * p0) / (fut[r["fighter_id"]][0] + SHRINK)
                  for r in keep])
    C = {k: np.array([r["components"][k] for r in keep])
         for k in (*bmf.BASE_KEYS, "opp_quality")}
    print(f"as of {args.as_of}: {len(keep)} fighters with >= {MIN_FUTURE} later fights, "
          f"base FOTN rate {p0:.3f}")

    print("\ncomponent alone vs later FOTN rate (Spearman)")
    for k, v in C.items():
        print(f"  {k:12s} {spearmanr(v, y).statistic:+.3f}")

    def scores(w: dict) -> np.ndarray:
        tot = sum(w[k] for k in bmf.BASE_KEYS)
        base = sum(w[k] * C[k] for k in bmf.BASE_KEYS) / tot
        return base * (bmf.GATE_FLOOR + (1 - bmf.GATE_FLOOR)
                       * (C["opp_quality"] / 100) ** w["gamma"])

    def rho(w: dict) -> float:
        return spearmanr(scores(w), y).statistic

    print(f"\nold weights      rho {rho(OLD_WEIGHTS):+.3f}")
    print(f"current defaults rho {rho(bmf.DEFAULT_WEIGHTS):+.3f}  {bmf.DEFAULT_WEIGHTS}")

    for t in args.tries:
        w = {**bmf.DEFAULT_WEIGHTS, **{k: float(v) for k, v in (p.split("=") for p in t.split(","))}}
        print(f"try              rho {rho(w):+.3f}  {t}")

    results, seen = [], set()
    for combo in itertools.product(GRID, repeat=len(bmf.BASE_KEYS)):
        if sum(combo) == 0:
            continue
        norm = tuple(round(c / sum(combo), 3) for c in combo)
        if norm in seen:
            continue
        seen.add(norm)
        for g in GAMMAS:
            w = dict(zip(bmf.BASE_KEYS, combo), gamma=g)
            results.append((rho(w), w))
    results.sort(key=lambda t: -t[0])
    print("\nbest grid settings")
    for r, w in results[:10]:
        tot = sum(w[k] for k in bmf.BASE_KEYS)
        print(f"  rho {r:+.3f}  " + "  ".join(f"{k[:5]} {w[k] / tot:.2f}" for k in bmf.BASE_KEYS)
              + f"  gamma {w['gamma']}")


if __name__ == "__main__":
    main()
