"""Choose LOSS_CREDIT and the tier line (TIER_INTERCEPT, TIER_SLOPE) for tapology_rankings.

The tier line was originally fitted to Tapology's published Strength of Schedule. That is
the right target for SoS but not for the ranking: the line that best matches SoS ranks
worse than the shipped one. So here the line is chosen by the ranking objective, and the
held-out score re-chooses it inside every leave-one-division-out fold — otherwise two
free parameters picked on all eight divisions would leak into the "held-out" number.

LOSS_CREDIT=0 goes through the identical procedure, so it is the baseline.

    DATABASE_URL=postgresql://localhost/alocks_local PYTHONPATH=. \
        ./venv/bin/python scripts/fit_tier_line.py --as-of 2026-09-18
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

import numpy as np

from app.database import SessionLocal
from app.services.ufc.tapology_fit import evaluate, fit
from app.services.ufc.tapology_rankings import (
    TapologyRanker, Weights, _resume, build_history, opponent_tier,
)
from app.services.ufc.tapology_target import load_full, normalise, resolve

INTERCEPTS = np.round(np.arange(-2.1, 0.61, 0.3), 1)
SLOPES = np.round(np.arange(1.75, 3.26, 0.25), 2)


def with_line(ctx: dict, resumes: dict, a: float, b: float) -> dict:
    return {**ctx, "tiers": {k: opponent_tier(r, a, b) for k, r in resumes.items()}}


def choose(ctx, resumes, target, as_of, divisions, rounds, keep):
    """Screen every line with default weights, then refit weights on the best `keep`."""
    screened = sorted(
        ((evaluate(Weights(), with_line(ctx, resumes, a, b), target, as_of, divisions), a, b)
         for a in INTERCEPTS for b in SLOPES), reverse=True)[:keep]
    best = None
    for _, a, b in screened:
        c = with_line(ctx, resumes, a, b)
        w = fit(c, target, as_of, Weights(), rounds=rounds, divisions=divisions)
        s = evaluate(w, c, target, as_of, divisions)
        if best is None or s > best[0]:
            best = (s, a, b, w)
    return best


def sos_mae(ranked, sos_target):
    e = [abs(ranked.extras[f]["sos"] - t) for f, t in sos_target.items() if f in ranked.extras]
    return float(np.mean(e)) if e else float("nan")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=date.fromisoformat, required=True,
                    help="the date the pasted Tapology target was taken")
    ap.add_argument("--credits", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    args = ap.parse_args()

    db = SessionLocal()
    try:
        ctx = build_history(db)
    finally:
        db.close()
    names, bouts = ctx["names"], ctx["bouts"]
    full = load_full()
    target, _ = resolve({d: [n for n, _ in rows] for d, rows in full.items()}, names)
    by_key: dict[str, list[int]] = {}
    for f, n in names.items():
        by_key.setdefault(normalise(n), []).append(f)
    sos_target = {by_key[normalise(n)][0]: s for rows in full.values() for n, s in rows
                  if s is not None and len(by_key.get(normalise(n), [])) == 1}
    divs = sorted(target)

    for credit in args.credits:
        resumes = {(f, b.fight_id): _resume(bouts, b.opponent_id, b.date, loss_credit=credit)
                   for f, bl in bouts.items() for b in bl}
        s_in, a, b, w = choose(ctx, resumes, target, args.as_of, None, rounds=3, keep=3)
        held = []
        for d in divs:
            others = [x for x in divs if x != d]
            _, ha, hb, hw = choose(ctx, resumes, target, args.as_of, others, rounds=1, keep=2)
            held.append(evaluate(hw, with_line(ctx, resumes, ha, hb), target, args.as_of, [d]))
        ranked = TapologyRanker(w).score(with_line(ctx, resumes, a, b), args.as_of)
        lw = ranked.order["lightweight"]
        ruffy = next((i + 1 for i, f in enumerate(lw) if "Ruffy" in names[f]), None)
        print(f"credit={credit:<5} line={a:+.1f}{b:+.2f}r  in={s_in:.4f}  "
              f"held={np.mean(held):.4f}  sosMAE={sos_mae(ranked, sos_target):.2f}  "
              f"ruffy={ruffy}", flush=True)
        print("    held by division:", dict(zip(divs, np.round(held, 3))))
        print("    weights:", w)


if __name__ == "__main__":
    main()
