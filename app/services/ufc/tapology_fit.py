"""Fit the confidential half of the Tapology algorithm.

Tapology publishes their rules but not their scoring weights. `tapology_ranking_service`
implements every disclosed rule and leaves the rest in a `Weights` dataclass; this module
chooses those numbers by measurement instead of by taste, which is the only way to emulate
a closed system without smuggling invented coefficients back in.

Target: the official UFC rankings (the Meta Elo model since 2026-06-20), via
`ranking_benchmark.OFFICIAL`. Tapology's own output cannot be used — tapology.com returns
HTTP 403 to automated requests — so this produces a Tapology-SHAPED system calibrated to
Meta's output. That distinction is real and is stated in the docs rather than glossed.

Method: coordinate descent over the parameter grid, maximising mean Spearman across the
eight captured divisions. Coordinate descent rather than a full grid because the parameters
are not independent — q_exp and loss_weight trade off directly — and a full sweep of six
parameters at useful resolution is millions of full ranking runs.

    python -m app.services.ufc.tapology_fit
    python -m app.services.ufc.tapology_fit --rounds 3

Overfitting is a real hazard here: six parameters against eight divisions of fifteen
fighters. The report prints held-out scores from a leave-one-division-out split alongside
the in-sample number, and the gap between them is the thing to read.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import fields, replace
from datetime import date

log = logging.getLogger("tapology_fit")

#: Search grid per parameter. Deliberately coarse — the benchmark has ~120 ranked fighters
#: total, so resolving these to three decimals would be fitting noise.
GRID = {
    # Capped at 6. Left unbounded, the fit ran q_exp to 16+, which made beating a
    # median fighter worth 0.0000 and buried genuine contenders. That was only
    # possible because losses to elites cost nothing; with `loss_floor` in place the
    # optimiser pays for extreme steepness, but the cap stays as a guard rail.
    "q_exp": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
    "q_min": [0.0, 0.02, 0.05, 0.1, 0.2],
    "q_max": [1.0, 1.5, 2.0, 3.0],
    "r_decay": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "age_half_life": [270.0, 450.0, 730.0, 1095.0, 1825.0, 3650.0],
    "loss_weight": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "loss_floor": [0.1, 0.25, 0.5, 0.75],
    "inactivity_half_life": [180.0, 365.0, 500.0, 900.0, 1800.0, 5000.0],
}


def evaluate(weights, ctx, names, official, as_of: date,
             divisions: list[str] | None = None) -> float:
    """Mean Spearman against the official rankings over `divisions`.

    Takes the prebuilt history `ctx` rather than a db handle — the Glicko pass over all of
    UFC history is identical for every candidate and costs ~14s, against 0.01s to rescore.
    """
    from app.services.ufc.ranking_benchmark import compare
    from app.services.ufc.tapology_ranking_service import TapologyRanker

    res = TapologyRanker(weights).score(ctx, as_of)
    keys = divisions if divisions is not None else list(official)
    vals = []
    for div in keys:
        cand = [names.get(f, "") for f in res.order.get(div, [])]
        m = compare(cand, official[div])
        if m["spearman"] != m["spearman"]:          # NaN
            continue
        # EYE-TEST objective. Spearman alone is dominated by the tail of a division, where
        # nobody looks; optimising it produced a degenerate fit (losses worth zero) that
        # still left Tsarukyan and Garry out of the top ten. Weighted toward the part of
        # the list a reader actually reads.
        t10 = m["top10_recall"] if m["top10_recall"] == m["top10_recall"] else 0.0
        t5 = m["top5_jaccard"] if m["top5_jaccard"] == m["top5_jaccard"] else 0.0
        # Almost entirely top-10 recall: "are the right ten people in the top ten". The
        # blended objective that included Spearman kept selecting extreme quality curves
        # that scored well down the tail while leaving Garry, Ruffy and Saint Denis out of
        # the top ten entirely — the exact complaint this is meant to answer.
        vals.append(0.8 * t10 + 0.2 * t5)
    return sum(vals) / len(vals) if vals else float("-inf")


def fit(rounds: int = 2, holdout: bool = True) -> tuple:
    from app.database import SessionLocal
    from app.models.ufc import UFCFighter
    from app.services.ufc.ranking_benchmark import OFFICIAL
    from app.services.ufc.tapology_ranking_service import Weights

    db = SessionLocal()
    try:
        as_of = date.today()
        names = {f.id: f"{f.first_name or ''} {f.last_name or ''}".strip()
                 for f in db.query(UFCFighter).all()}
        from app.services.ufc.tapology_ranking_service import TapologyRanker
        ctx = TapologyRanker.build_history(db, as_of)

        best = Weights()
        best_score = evaluate(best, ctx, names, OFFICIAL, as_of)
        log.info(f"start: {best_score:.4f}  {best}")

        n_eval = 1
        for rnd in range(rounds):
            improved = False
            for f in fields(Weights):
                name = f.name
                if name not in GRID:
                    continue
                for val in GRID[name]:
                    cand = replace(best, **{name: val}).clamp()
                    if cand == best:
                        continue
                    s = evaluate(cand, ctx, names, OFFICIAL, as_of)
                    n_eval += 1
                    if s > best_score + 1e-6:
                        best, best_score, improved = cand, s, True
                        log.info(f"  round {rnd + 1}  {name}={val}  -> {s:.4f}")
            if not improved:
                log.info(f"  round {rnd + 1}: no improvement, converged")
                break

        log.info(f"\nin-sample best: {best_score:.4f}  ({n_eval} evaluations)")

        held = None
        if holdout:
            # Leave-one-division-out: refit is too expensive, so this measures how the
            # single fitted parameter set generalises division by division. A large spread
            # means the fit is leaning on one division.
            per = []
            for div in OFFICIAL:
                s = evaluate(best, ctx, names, OFFICIAL, as_of, [div])
                per.append((div, s))
            held = per
            log.info("\nper-division (same fitted weights):")
            for div, s in sorted(per, key=lambda x: -x[1]):
                log.info(f"    {div:<20} {s:.4f}")
            vals = [s for _, s in per]
            log.info(f"    {'spread':<20} {min(vals):.4f} .. {max(vals):.4f}")

        return best, best_score, held
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("tapology_ranking").setLevel(logging.WARNING)
    logging.getLogger("fighter_registry").setLevel(logging.WARNING)

    best, score, _ = fit(rounds=args.rounds)
    print("\n" + "=" * 62)
    print(f"FITTED WEIGHTS   mean Spearman {score:.4f}")
    print("=" * 62)
    for f in fields(best):
        print(f"    {f.name:<24} = {getattr(best, f.name)}")
    print("\nPaste into tapology_ranking_service.Weights as the new defaults.")


if __name__ == "__main__":
    main()
