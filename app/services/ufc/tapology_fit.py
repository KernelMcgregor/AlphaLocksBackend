"""Fit the confidential half of the Tapology algorithm against their published output.

`tapology_rankings` implements every disclosed rule and leaves the rest in `Weights`.
This module chooses those numbers by measurement against Tapology's own rankings, which
is the only honest way to emulate a closed system — the alternative is inventing
coefficients and calling the result a clone.

    python -m app.services.ufc.tapology_fit              # fit and report
    python -m app.services.ufc.tapology_fit --rounds 4   # more coordinate passes
    python -m app.services.ufc.tapology_fit --report     # score current weights only

Method: coordinate descent over a coarse grid. Not a full sweep — the parameters trade
off directly against each other (q_exp against loss_scale most of all), and a full grid
at useful resolution is millions of full rescorings.

Overfitting is the live hazard: ~13 free parameters against maybe 165 ranked slots. The
report prints leave-one-division-out held-out scores next to the in-sample number, and
the GAP between them is the number to read. A large gap means the weights have memorised
these divisions and will not survive the next event.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import fields, replace
from datetime import date

from app.services.ufc.tapology_rankings import TapologyRanker, Weights, build_history

log = logging.getLogger("tapology_fit")

#: Deliberately coarse. The target is ~165 ranked slots; resolving these to three decimals
#: would be fitting noise, and the report would not be able to tell.
GRID = {
    "v_finish": [1.0, 1.1, 1.2, 1.35, 1.5],
    "v_md": [0.6, 0.7, 0.8, 0.9],
    "v_sd": [0.5, 0.6, 0.7, 0.8],
    "v_draw": [0.2, 0.35, 0.5, 0.65],
    "round_bonus": [0.0, 0.03, 0.06, 0.1],
    "q_min": [0.0, 0.02, 0.05, 0.1, 0.2],
    "q_max": [1.0, 1.5, 2.0, 3.0],
    "q_exp": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
    "pos_decay": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "age_half_life": [270.0, 450.0, 730.0, 1095.0, 1825.0],
    "loss_scale": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "loss_tier_relief": [0.0, 0.25, 0.5, 0.75],
    "carry_other_division": [0.5, 0.7, 0.85, 1.0],
}


def spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    d2 = sum((x - y) ** 2 for x, y in zip(a, b))
    return 1 - (6 * d2) / (n * (n * n - 1))


def compare(ours: list[int], theirs: list[int]) -> dict:
    """Score our ordering against Tapology's for one division.

    Two metrics, because either one alone is gameable:

    * **spearman** — ordering of the fighters we both rank. Computed on the shared set
      only, since the paste is necessarily shallower than our full-roster output.
    * **precision@k** — how much of our top-k is in their top-k. This is the one that
      catches the failure that killed the previous ranker: an unbeaten prospect sitting at
      #2 is invisible to Spearman (he simply is not in their list, so he is skipped) but
      is the single most obvious thing wrong with the output. Anything of ours inside
      their depth that they do not rank there is an interloper, and it counts against us.
    """
    k = len(theirs)
    ours_pos = {f: i for i, f in enumerate(ours)}
    shared = [f for f in theirs if f in ours_pos]

    top_k = ours[:k]
    hits = len(set(top_k) & set(theirs))
    precision = hits / k if k else 0.0
    interlopers = [f for f in top_k if f not in set(theirs)]

    if len(shared) < 2:
        return {"n": len(shared), "spearman": 0.0, "top1": 0.0, "mae": 0.0,
                "precision": precision, "interlopers": interlopers,
                "missing": len(theirs) - len(shared)}

    target_pos = {f: i for i, f in enumerate(shared)}
    ours_rel = sorted(shared, key=lambda f: ours_pos[f])
    ours_rank = {f: i for i, f in enumerate(ours_rel)}

    xs = [ours_rank[f] for f in shared]
    ys = [target_pos[f] for f in shared]
    return {
        "n": len(shared),
        "spearman": spearman(xs, ys),
        "top1": 1.0 if shared and ours_rel[0] == shared[0] else 0.0,
        "mae": sum(abs(x - y) for x, y in zip(xs, ys)) / len(shared),
        "precision": precision,
        "interlopers": interlopers,
        "missing": len(theirs) - len(shared),
    }


def objective(m: dict) -> float:
    """What the fit maximises: getting the right fighters, then ordering them.

    Weighted evenly. Precision alone would not care about the order within the top 15;
    Spearman alone would not care that four of our top 15 are fighters Tapology does not
    rank at all. The previous ranker scored well on the second and catastrophically on
    the first, which is how it passed its benchmark and failed on sight.
    """
    return 0.5 * m["precision"] + 0.5 * max(0.0, m["spearman"])


def evaluate(weights: Weights, ctx: dict, target: dict, as_of: date,
             divisions: list[str] | None = None) -> float:
    """Mean objective against the target over `divisions`."""
    result = TapologyRanker(weights).score(ctx, as_of)
    keys = divisions if divisions is not None else list(target)
    scored = [objective(compare(result.order.get(d, []), target[d]))
              for d in keys if d in target]
    return sum(scored) / len(scored) if scored else 0.0


def fit(ctx: dict, target: dict, as_of: date, start: Weights,
        rounds: int = 3, divisions: list[str] | None = None) -> Weights:
    best = start.clamp()
    best_score = evaluate(best, ctx, target, as_of, divisions)
    names = [f.name for f in fields(Weights) if f.name in GRID]

    for r in range(rounds):
        improved = False
        for name in names:
            for value in GRID[name]:
                cand = replace(best, **{name: value}).clamp()
                score = evaluate(cand, ctx, target, as_of, divisions)
                if score > best_score + 1e-9:
                    best, best_score, improved = cand, score, True
        log.info(f"  round {r + 1}: mean spearman {best_score:.4f}")
        if not improved:
            break
    return best


def report(weights: Weights, ctx: dict, target: dict, as_of: date) -> None:
    result = TapologyRanker(weights).score(ctx, as_of)
    names = ctx["names"]

    print(f"\n{'division':<20} {'n':>4} {'prec':>6} {'spear':>7} {'mae':>6} "
          f"{'top1':>5} {'intrs':>6}")
    print("-" * 62)
    metrics = {d: compare(result.order.get(d, []), target[d]) for d in sorted(target)}
    for d, m in metrics.items():
        print(f"{d:<20} {m['n']:>4} {m['precision']:>6.2f} {m['spearman']:>7.3f} "
              f"{m['mae']:>6.2f} {m['top1']:>5.0f} {len(m['interlopers']):>6}")
    print("-" * 62)
    n = len(metrics)
    print(f"{'MEAN':<20} {'':>4} {sum(m['precision'] for m in metrics.values()) / n:>6.2f} "
          f"{sum(m['spearman'] for m in metrics.values()) / n:>7.3f}")

    # The worst division is where the remaining structural error lives — a uniformly
    # mediocre fit and one catastrophic division call for different fixes.
    worst = min(metrics, key=lambda d: objective(metrics[d]))
    theirs = target[worst]
    ours = result.order.get(worst, [])[:len(theirs)]
    in_target = set(theirs)
    print(f"\nWorst division: {worst}   (* = we rank them here, Tapology does not)")
    print(f"  {'#':>3}  {'ours':<28} {'tapology':<26}")
    for i in range(max(len(ours), len(theirs))):
        a = names.get(ours[i], "?") if i < len(ours) else ""
        b = names.get(theirs[i], "?") if i < len(theirs) else ""
        mark = " *" if i < len(ours) and ours[i] not in in_target else ""
        print(f"  {i + 1:>3}  {a + mark:<28} {b:<26}{'' if a == b else '  <-'}")


def sos_report(ctx: dict, sos_target: dict[int, int], as_of: date) -> None:
    """Compare our Strength of Schedule against Tapology's published numbers.

    This is the strongest check available. SoS is the one quantity where Tapology gives
    BOTH the formula and the output, so any gap is a defect in our opponent-tier curve
    and nothing else — no weights involved, no ordering to argue about.
    """
    if not sos_target:
        return
    result = TapologyRanker().score(ctx, as_of)
    names = ctx["names"]

    rows = [(names.get(f, "?"), result.extras[f]["sos"], t)
            for f, t in sos_target.items() if f in result.extras]
    if not rows:
        return
    rows.sort(key=lambda r: -r[2])

    err = [abs(o - t) for _, o, t in rows]
    bias = sum(o - t for _, o, t in rows) / len(rows)
    print(f"\nStrength of Schedule vs Tapology  ({len(rows)} fighters)")
    print(f"  {'fighter':<24} {'ours':>5} {'theirs':>7} {'diff':>6}")
    for n, o, t in rows:
        print(f"  {n:<24} {o:>5} {t:>7} {o - t:>+6}")
    print(f"  {'':<24} {'':>5} {'MAE':>7} {sum(err) / len(err):>6.1f}")
    print(f"  {'':<24} {'':>5} {'bias':>7} {bias:>+6.1f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--report", action="store_true",
                    help="score the current weights without refitting")
    ap.add_argument("--path", default=None, help="path to the pasted rankings")
    args = ap.parse_args()

    from app.database import SessionLocal
    from app.services.ufc.tapology_target import load_full, resolve

    db = SessionLocal()
    try:
        as_of = date.today()
        log.info("Building history (every UFC bout + the tier recursion)...")
        ctx = build_history(db, as_of)

        full = load_full(args.path)
        raw = {d: [n for n, _ in rows] for d, rows in full.items()}
        target, unmatched = resolve(raw, ctx["names"])

        # Published SoS, keyed by fighter id, for the tier-curve check.
        name_to_id = {}
        for division, ids in target.items():
            for name, fid in zip([n for n in raw[division]
                                  if n not in {u[1] for u in unmatched}], ids):
                name_to_id[(division, name)] = fid
        sos_target = {}
        for division, rows in full.items():
            for name, sos in rows:
                fid = name_to_id.get((division, name))
                if fid is not None and sos is not None:
                    sos_target[fid] = sos
        log.info(f"Target: {sum(len(v) for v in target.values())} fighters across "
                 f"{len(target)} divisions")
        if unmatched:
            log.warning(f"{len(unmatched)} target names did not match a fighter:")
            for division, name in unmatched:
                log.warning(f"    {division:<20} {name}")

        if args.report:
            report(Weights(), ctx, target, as_of)
            sos_report(ctx, sos_target, as_of)
            return

        best = fit(ctx, target, as_of, Weights(), rounds=args.rounds)

        # Held-out: refit without each division, score on it. The gap against the
        # in-sample number is the overfitting readout. Needs at least two divisions —
        # with one there is nothing to hold out, and reporting a number anyway would
        # dress up a pure in-sample fit as validated.
        held = []
        if len(target) >= 2:
            for d in sorted(target):
                others = [x for x in target if x != d]
                w = fit(ctx, target, as_of, Weights(), rounds=1, divisions=others)
                held.append(evaluate(w, ctx, target, as_of, [d]))

        report(best, ctx, target, as_of)
        sos_report(ctx, sos_target, as_of)
        print(f"\nIn-sample mean : {evaluate(best, ctx, target, as_of):.4f}")
        if held:
            print(f"Held-out mean  : {sum(held) / len(held):.4f}   "
                  f"(gap {evaluate(best, ctx, target, as_of) - sum(held) / len(held):+.4f})")
        else:
            print("Held-out mean  : n/a — only one division in the target, so these "
                  "weights are UNVALIDATED. Add divisions before trusting them.")

        print("\nFitted weights — paste into tapology_rankings.Weights:")
        for f in fields(Weights):
            print(f"    {f.name}: {getattr(best, f.name)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
