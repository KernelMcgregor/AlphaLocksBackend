"""Does blending deserve-to-win into decision credit improve the division rankings?
Read-only.

    PYTHONPATH=. venv/bin/python -m scripts.dtw_rankings_test [--since 2022-01-01]

For each blend in BLENDS (Weights.dtw_blend; 0 = shipped):

  1. Tapology match: the fit objective against data/tapology_rankings.txt, as of the
     date it was pasted, with the shipped weights (the blend is not refit).
  2. Next fight: for every decided UFC bout since `--since`, both fighters' division
     scores the day before. Accuracy of "higher score wins", and log loss of a one-
     parameter logistic on the score gap. Fights where either fighter has no score in
     that division are skipped. A paired count of the fights the blend flips is printed,
     since accuracy differences of a point are inside the noise.
  3. Movers: today's top 15 per division, shipped vs the best blend.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
from scipy.optimize import minimize_scalar

from app.database import SessionLocal
from app.models.ufc import UFCEvent, UFCFight
from app.services.ufc.fighter_registry import classify_weight_class, is_decided
from app.services.ufc.tapology_fit import evaluate
from app.services.ufc.tapology_rankings import DIVISIONS, TapologyRanker, Weights, build_history

BLENDS = (0.0, 0.1, 0.2, 0.3, 0.5)
TAPOLOGY_AS_OF = date(2026, 9, 18)


def tapology_check(ctx: dict) -> None:
    from app.services.ufc.tapology_target import load_full, resolve

    full = load_full()
    target, _ = resolve({d: [n for n, _ in rows] for d, rows in full.items()}, ctx["names"])
    print(f"1. Tapology match ({TAPOLOGY_AS_OF}, {len(target)} divisions)")
    for lam in BLENDS:
        w = Weights(dtw_blend=lam)
        per = {d: evaluate(w, ctx, target, TAPOLOGY_AS_OF, [d]) for d in sorted(target)}
        print(f"   blend {lam:.1f}  mean {np.mean(list(per.values())):.4f}   "
              + " ".join(f"{d[:5]} {v:.3f}" for d, v in per.items()))


def logloss(gap: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Best single-scale logistic on the score gap: (log loss, scale)."""
    def f(k):
        p = 1 / (1 + np.exp(-k * gap))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    r = minimize_scalar(f, bounds=(0, 10), method="bounded")
    return float(r.fun), float(r.x)


def next_fight_check(db, ctx: dict, since: date) -> None:
    rows = (db.query(UFCFight.id, UFCFight.date, UFCFight.red_fighter_id,
                     UFCFight.blue_fighter_id, UFCFight.winner_id, UFCFight.method,
                     UFCFight.weight_class)
            .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
            .filter(UFCFight.date >= since, UFCFight.date <= date.today()).all())
    by_date = defaultdict(list)
    for fid, d, red, blue, winner, method, wc in rows:
        div = classify_weight_class(wc)
        if div in DIVISIONS and is_decided(method, winner):
            by_date[d].append((fid, red, blue, winner == red, div))

    rankers = {lam: TapologyRanker(Weights(dtw_blend=lam)) for lam in BLENDS}
    gaps = {lam: [] for lam in BLENDS}
    y, keep_ids = [], []
    for d in sorted(by_date):
        before = d - timedelta(days=1)
        res = {lam: r.score(ctx, before).division_scores for lam, r in rankers.items()}
        for fid, red, blue, red_won, div in by_date[d]:
            if (red, div) not in res[0.0] or (blue, div) not in res[0.0]:
                continue
            for lam in BLENDS:
                gaps[lam].append(res[lam][(red, div)] - res[lam][(blue, div)])
            y.append(float(red_won))
            keep_ids.append(fid)
    y = np.array(y)
    print(f"\n2. Next fight: {len(y)} decided bouts since {since} with both fighters scored")
    base_right = (np.array(gaps[0.0]) > 0) == (y == 1)
    for lam in BLENDS:
        g = np.array(gaps[lam])
        right = (g > 0) == (y == 1)
        ll, k = logloss(g, y)
        gained = int((right & ~base_right).sum())
        lost = int((~right & base_right).sum())
        flips = gained + lost
        # Two-sided sign test on the flipped fights.
        p = (min(1.0, 2 * sum(math.comb(flips, i) for i in range(min(gained, lost) + 1))
                 / 2 ** flips) if flips else 1.0)
        print(f"   blend {lam:.1f}  accuracy {right.mean():.4f}  log loss {ll:.4f}  "
              f"flipped {flips:3d} (right {gained}, wrong {lost}, sign-test p {p:.2f})")


def movers(ctx: dict, lam: float, n: int = 15) -> None:
    today = date.today()
    base = TapologyRanker(Weights()).score(ctx, today).order
    alt = TapologyRanker(Weights(dtw_blend=lam)).score(ctx, today).order
    names = ctx["names"]
    print(f"\n3. Top {n} movers, blend {lam:.1f} vs shipped (today)")
    for d in DIVISIONS:
        b = {f: i + 1 for i, f in enumerate(base.get(d, []))}
        a = {f: i + 1 for i, f in enumerate(alt.get(d, []))}
        moved = sorted(((b[f] - a[f], f) for f in a if f in b and min(a[f], b[f]) <= n
                        and a[f] != b[f]), key=lambda t: -abs(t[0]))
        if moved:
            print(f"   {d:18s} " + ", ".join(
                f"{names.get(f, '?')} {b[f]}->{a[f]}" for _, f in moved[:6]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=date.fromisoformat, default=date(2022, 1, 1))
    ap.add_argument("--movers-blend", type=float, default=0.2)
    args = ap.parse_args()
    db = SessionLocal()
    try:
        ctx = build_history(db)
        print(f"DTW on {len(ctx['dtw']) // 2} decisions/draws\n")
        tapology_check(ctx)
        next_fight_check(db, ctx, args.since)
        movers(ctx, args.movers_blend)
    finally:
        db.close()


if __name__ == "__main__":
    main()
