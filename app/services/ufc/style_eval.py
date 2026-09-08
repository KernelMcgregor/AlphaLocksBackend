"""Does the style space mean anything? — offline validation for style_service.

The temptation with a similarity feature is to eyeball a few lists, agree that they look
reasonable, and ship. That tests nothing: any distance metric over fighter stats produces
plausible-looking neighbours, because every pair of fighters shares *something*.

THE TEST: STYLE SHOULD PREDICT TRANSFER
---------------------------------------
If A and B genuinely fight alike, then they present the same problems to an opponent, so a
third fighter C who has faced both should tend to get the same RESULT against both. That
is a real, falsifiable, leakage-free claim, and 11,333 fights contain enough shared
opponents to measure it.

For every (A, B) pair with at least one common opponent C, we ask whether C's outcomes
against A and against B agree, then bucket pairs by style similarity and check that
agreement rises with it.

    PASS = agreement increases across similarity deciles, with the top decile materially
           above the all-pairs baseline.

A flat curve means the space is measuring something that is not style, and the feature
should be retuned (BLOCK_B_WEIGHT, the Block A column set) before it ships.

One caveat on reading the headline number: the raw lift is inflated by a fighter-QUALITY
confound, because similar-style pairs also tend to be similar-quality pairs. Quality alone
scores +0.059 on this metric — better than style does. The figure that isolates this
space's contribution is the lift computed WITHIN quality strata (+0.018). See the MEASURED
block in style_service for both numbers and why the stratified one is the honest one.
This is the gate, not a report.

Note the direction of the claim. We are NOT asserting that similar fighters beat each
other, or that similarity predicts a head-to-head winner — that would be a model feature,
and this table is display-only. We are asserting the weaker and checkable thing: shared
style implies shared results against shared opposition.

SECONDARY, SANITY ONLY
----------------------
A hand-labelled set of pairs most fans would call obvious comps. Reported as recall@K, and
worth glancing at, but it is a handful of subjective labels and must not be used to tune
anything — that is how you overfit a metric to your own priors.

Run: python -m app.services.ufc.style_eval
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from app.database import SessionLocal
from app.models.ufc import UFCFight, UFCFighter
from app.services.ufc.fighter_registry import is_decided
from app.services.ufc.style_service import (
    ALL_FEATURES,
    build_matrix,
    fit_space,
    load_space,
    transform,
)

log = logging.getLogger("style_eval")

#: Pairs are bucketed into this many equal-count similarity bins.
N_BUCKETS = 10

#: A pair needs at least this many common opponents to contribute. 1 is usable but noisy;
#: the trend is reported at 1 and re-reported at 2 as a robustness check.
MIN_COMMON = 1

#: Obvious-comp sanity set. Names are matched case-insensitively on "first last".
#: Deliberately short and deliberately not used for tuning.
SANITY_PAIRS = [
    ("khabib nurmagomedov", "islam makhachev"),
    ("stephen thompson", "lyoto machida"),
    ("charles oliveira", "brian ortega"),
    ("max holloway", "alexander volkanovski"),
    ("jose aldo", "cody garbrandt"),
]


def _common_opponent_agreement(db, fids: list[int]) -> dict[tuple[int, int], tuple[int, int]]:
    """{(a, b): (n_agree, n_total)} over common opponents, for a in fids, b in fids.

    Only decided bouts count — a no-contest tells you nothing about whether two fighters
    present the same problem. `is_decided` is reused so this agrees with what the rating
    services consider a result.
    """
    eligible = set(fids)

    # opponent -> {fighter: won?}  for every decided bout involving an eligible fighter
    results: dict[int, dict[int, bool]] = defaultdict(dict)
    for f in db.query(UFCFight).all():
        if not is_decided(f.method, f.winner_id):
            continue
        r, b = f.red_fighter_id, f.blue_fighter_id
        if r is None or b is None:
            continue
        r, b = int(r), int(b)
        w = int(f.winner_id)
        # Record from the perspective of each eligible fighter, keyed by their opponent.
        if r in eligible:
            results[b][r] = (w == r)
        if b in eligible:
            results[r][b] = (w == b)

    agree: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for _opp, seen in results.items():
        if len(seen) < 2:
            continue
        items = sorted(seen.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, a_won = items[i]
                b, b_won = items[j]
                rec = agree[(a, b)]
                rec[1] += 1
                if a_won == b_won:
                    rec[0] += 1

    return {k: (v[0], v[1]) for k, v in agree.items()}


def _bucket_report(sims: np.ndarray, agrees: np.ndarray, totals: np.ndarray, label: str):
    """Print agreement by similarity decile and return (deciles, rates, baseline)."""
    order = np.argsort(sims)
    baseline = agrees.sum() / totals.sum()

    edges = np.linspace(0, len(order), N_BUCKETS + 1).astype(int)
    rates, mids = [], []
    log.info(f"\n  {label} — {int(totals.sum())} common-opponent observations "
             f"across {len(sims)} pairs")
    log.info(f"  {'decile':<8}{'sim range':<22}{'pairs':>8}{'obs':>8}{'agree':>9}")
    for i in range(N_BUCKETS):
        sl = order[edges[i]:edges[i + 1]]
        if len(sl) == 0:
            continue
        tot = totals[sl].sum()
        if tot == 0:
            continue
        rate = agrees[sl].sum() / tot
        rates.append(rate)
        mids.append(float(np.mean(sims[sl])))
        log.info(
            f"  {i + 1:<8}{sims[sl].min():.3f}-{sims[sl].max():.3f}      "
            f"{len(sl):>8}{int(tot):>8}{rate:>9.3f}"
        )
    log.info(f"  {'baseline':<8}{'(all pairs)':<22}{len(sims):>8}"
             f"{int(totals.sum()):>8}{baseline:>9.3f}")
    return np.array(mids), np.array(rates), baseline


def evaluate(refit: bool = False) -> dict:
    db = SessionLocal()
    try:
        fids, X, divisions = build_matrix(db)
        if len(fids) < 50:
            log.error(f"Only {len(fids)} eligible fighters — run the stats pipeline first")
            return {}

        space = None if refit else load_space()
        if space is None:
            log.info("No frozen space on disk; fitting one for evaluation only "
                     "(not saved — style_service owns the artifact)")
            space = fit_space(X)
        E = transform(X, space)

        # Cosine similarity of every pair, in the same space retrieval uses.
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        En = E / norms
        S = En @ En.T

        pos = {f: i for i, f in enumerate(fids)}
        agreement = _common_opponent_agreement(db, fids)
        log.info(f"  {len(agreement)} fighter pairs share at least one opponent")

        result: dict = {"verdict": "INCONCLUSIVE"}
        for min_common in (MIN_COMMON, 2):
            sims, agrees, totals = [], [], []
            for (a, b), (n_agree, n_tot) in agreement.items():
                if n_tot < min_common or a not in pos or b not in pos:
                    continue
                sims.append(S[pos[a], pos[b]])
                agrees.append(n_agree)
                totals.append(n_tot)

            if len(sims) < N_BUCKETS * 5:
                log.warning(f"  Too few pairs at min_common={min_common}; skipping")
                continue

            mids, rates, baseline = _bucket_report(
                np.array(sims), np.array(agrees, dtype=float),
                np.array(totals, dtype=float),
                f"min_common={min_common}",
            )

            if min_common == MIN_COMMON:
                # Spearman-style monotonicity: correlation of decile index with rate.
                rho = float(np.corrcoef(np.arange(len(rates)), rates)[0, 1])
                lift = float(rates[-1] - baseline)
                log.info(f"\n  monotonicity (corr of decile vs agreement): {rho:+.3f}")
                log.info(f"  top-decile lift over baseline:              {lift:+.3f}")
                verdict = "PASS" if rho > 0.5 and lift > 0.02 else "FAIL"
                log.info(f"  VERDICT: {verdict}")
                result = {"rho": rho, "lift": lift, "baseline": baseline,
                          "verdict": verdict, "n_pairs": len(sims)}

        _sanity(db, fids, S, pos)
        return result
    finally:
        db.close()


def _sanity(db, fids, S, pos):
    """Recall@K on the hand-labelled comps. Glance at it; do not tune on it."""
    name_to_id = {}
    for f in db.query(UFCFighter).all():
        key = f"{f.first_name} {f.last_name}".strip().lower()
        name_to_id[key] = int(f.id)

    log.info("\n  Sanity comps (hand-labelled, NOT a tuning target):")
    for a_name, b_name in SANITY_PAIRS:
        a, b = name_to_id.get(a_name), name_to_id.get(b_name)
        if a is None or b is None or a not in pos or b not in pos:
            log.info(f"    {a_name} / {b_name}: not both in the eligible set")
            continue
        row = S[pos[a]].copy()
        row[pos[a]] = -np.inf
        rank = int((row > row[pos[b]]).sum()) + 1
        log.info(f"    {a_name} -> {b_name}: rank {rank} of {len(fids) - 1} "
                 f"(sim {S[pos[a], pos[b]]:.3f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate()
