"""Walk-forward evaluation of RANKINGS (not of the winner model).

Nothing in this repo measured whether `ufc_fighter_rankings.rank` predicts anything.
`tuner.py` and `model.walk_forward_eval` both evaluate the winner model, which consumes
Glicko *dimensions* and never touches the ranking table. So the Points+Elo system that
actually produces the rank numbers had no test, no baseline, and no way to compare a
proposed change against the incumbent — which is how a 50:1 win/loss asymmetry survived
in it, and why "the rankings don't pass the eye test" had no numerical form.

This module gives ranking changes the same standard of evidence the model changes get.

    python -m app.services.ufc.ranking_eval --rankers points,elo --folds 8

Design notes
------------
`walk_forward_eval` is deliberately NOT reused: it builds a matchup frame and fits a GBT,
none of which a ranking needs. Only the expanding-window bound convention is shared, so
fold definitions stay comparable between the two harnesses.

Leakage is prevented structurally rather than by care. An OnlineRanker is asked to rate
both corners BEFORE it is shown the result (`rate` then `observe`), so it cannot see the
fight it is being scored on. A BatchRanker is refit on `fights[:lo]` at each fold
boundary and only rates within `[lo:hi]`.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

log = logging.getLogger("ranking_eval")

#: Only these count as evidence — see fighter_registry.is_decided.
from app.services.ufc.fighter_registry import classify_weight_class, is_decided

#: Brackets, not candidates. Excluded from the common-fight intersection because they
#: have very different coverage from a real ranking.
REFERENCE_RANKERS = {"always_red", "market"}

#: Fights used to fit the probability scale for each fold, immediately before it.
CALIBRATION_WINDOW = 2000


class OnlineRanker(Protocol):
    """Sees fights one at a time, in order. Elo and Points are online."""

    name: str

    def rate(self, fid: int, as_of: date) -> tuple[float, float] | None: ...
    def observe(self, fight) -> None: ...


class BatchRanker(Protocol):
    """Refit from scratch at each fold boundary. WHR and Bradley-Terry are batch."""

    name: str

    def fit(self, fights: list) -> None: ...
    def rate(self, fid: int, as_of: date) -> tuple[float, float] | None: ...


# --------------------------------------------------------------------------- metrics
def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _fit_scale(diffs: list[float], labels: list[int]) -> tuple[float, float]:
    """Fit p = sigmoid(a + b*diff) by a few Newton steps on the training portion only.

    Fitting this in-sample would hand every candidate a free parameter tuned on the
    fights it is being scored on, which would make the comparison meaningless.
    """
    a, b = 0.0, 0.01
    if not diffs:
        return a, b
    for _ in range(25):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for d, y in zip(diffs, labels):
            p = _sigmoid(a + b * d)
            r = p - y
            w = max(p * (1 - p), 1e-9)
            g_a += r
            g_b += r * d
            h_aa += w
            h_ab += w * d
            h_bb += w * d * d
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (g_a * h_bb - g_b * h_ab) / det
        db = (g_b * h_aa - g_a * h_ab) / det
        a -= da
        b -= db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return a, b


def _auc(probs: list[float], labels: list[int]) -> float:
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and probs[order[j + 1]] == probs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(ranks[i] for i in range(len(probs)) if labels[i] == 1)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@dataclass
class Scored:
    """Per-fight records, accumulated across folds."""

    fight_ids: list[int] = field(default_factory=list)
    diffs: list[float] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    top15: list[bool] = field(default_factory=list)


def _metrics(s: Scored, subset: list[int] | None = None) -> dict:
    idx = subset if subset is not None else list(range(len(s.labels)))
    if not idx:
        return {"n": 0}
    lab = [s.labels[i] for i in idx]
    dif = [s.diffs[i] for i in idx]
    prb = [s.probs[i] for i in idx]
    # "Did the higher-rated fighter win", corner-agnostic. The tie check must come
    # FIRST: `(d > 0) == (y == 1)` scores a tie against a blue win as a correct call,
    # which handed the all-ties floor 0.72 instead of 0.50.
    acc = sum(0.5 if d == 0 else (1.0 if (d > 0) == (y == 1) else 0.0)
              for d, y in zip(dif, lab)) / len(idx)
    brier = sum((p - y) ** 2 for p, y in zip(prb, lab)) / len(idx)
    ll = -sum(math.log(max(p if y == 1 else 1 - p, 1e-15))
              for p, y in zip(prb, lab)) / len(idx)
    return {"n": len(idx), "acc": acc, "auc": _auc(prb, lab),
            "brier": brier, "logloss": ll}


def _top15(ranker, past_fights: list, as_of: date | None) -> set[int]:
    """Fighters in the top 15 of their division by this ranker's own rating.

    The top-15 slice is where the eye test actually lives — a ranking can look fine
    overall while ordering the contenders badly, which is exactly the reported symptom
    (a prospect above Charles Oliveira). Restricting the metrics to bouts between two
    ranked contenders is what makes that visible as a number.
    """
    if as_of is None:
        return set()
    division: dict[int, str] = {}
    active: dict[int, date] = {}
    for f in past_fights:
        wc = classify_weight_class(f.weight_class)
        for fid in (f.red_fighter_id, f.blue_fighter_id):
            if f.date >= active.get(fid, date.min):
                active[fid] = f.date
                if wc != "unknown":
                    division[fid] = wc
    by_div: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for fid, div in division.items():
        if (as_of - active[fid]).days > 548:
            continue
        r = ranker.rate(fid, as_of)
        if r is not None:
            by_div[div].append((r[0], fid))
    elite: set[int] = set()
    for div, rows in by_div.items():
        rows.sort(reverse=True)
        elite.update(fid for _, fid in rows[:15])
    return elite


# --------------------------------------------------------------------------- harness
def evaluate(rankers: list, fights: list, n_folds: int = 8,
             eval_frac: float = 0.25) -> dict:
    """Expanding-window walk-forward over decided bouts."""
    fights = [f for f in fights
              if f.date and is_decided(f.method, f.winner_id)
              and classify_weight_class(f.weight_class) != "unknown"]
    fights.sort(key=lambda f: (f.date, f.id))
    n = len(fights)
    eval_start = int(n * (1 - eval_frac))
    bounds = [eval_start + round(i * (n - eval_start) / n_folds) for i in range(n_folds + 1)]
    log.info(f"  {n} decided bouts | {n_folds} folds over the last {eval_frac:.0%} "
             f"({n - eval_start} eval bouts)")

    out: dict[str, Scored] = {}
    for ranker in rankers:
        s = Scored()
        is_batch = hasattr(ranker, "fit")
        bind = getattr(ranker, "bind", None)
        if not is_batch:
            # Warm up the instance the caller passed. Re-instantiating with
            # `ranker.__class__()` silently discarded MarketBaseline's loaded odds and
            # gave it 0 coverage.
            for f in fights[:eval_start]:
                if bind:
                    bind(f)
                ranker.observe(f)

        for fold in range(n_folds):
            lo, hi = bounds[fold], bounds[fold + 1]
            cal_lo = max(0, lo - CALIBRATION_WINDOW)

            # Scale is fit on the fights BEFORE this fold, never on the fold itself —
            # and for a BATCH ranker it must also be fit on ratings that did not see
            # those fights. Calibrating a retrodictive ranker on its own training slice
            # makes its rating gaps look far sharper than they are out of sample: WHR
            # scored Brier 0.317 / log-loss 1.39 (worse than predicting 0.5 flat) purely
            # from this, while its accuracy was the best in the field.
            if is_batch:
                ranker.fit(fights[:cal_lo])

            tr_d, tr_y = [], []
            for f in fights[cal_lo:lo]:
                if bind:
                    bind(f)
                r = ranker.rate(f.red_fighter_id, f.date)
                b = ranker.rate(f.blue_fighter_id, f.date)
                if r and b:
                    tr_d.append(r[0] - b[0])
                    tr_y.append(1 if f.winner_id == f.red_fighter_id else 0)
            a, bb = _fit_scale(tr_d, tr_y)

            if is_batch:
                ranker.fit(fights[:lo])

            # Divisional top-15 as of the fold start. Held fixed for the fold so this
            # costs one pass rather than one per fight; the set barely moves in ~200
            # bouts, and every ranker is measured the same way.
            elite = _top15(ranker, fights[:lo], fights[lo].date if lo < n else None)

            for f in fights[lo:hi]:
                if bind:
                    bind(f)
                r = ranker.rate(f.red_fighter_id, f.date)
                bl = ranker.rate(f.blue_fighter_id, f.date)
                if r and bl:
                    d = r[0] - bl[0]
                    s.fight_ids.append(f.id)
                    s.diffs.append(d)
                    s.labels.append(1 if f.winner_id == f.red_fighter_id else 0)
                    s.probs.append(_sigmoid(a + bb * d))
                    s.top15.append(f.red_fighter_id in elite and f.blue_fighter_id in elite)
                if not is_batch:
                    ranker.observe(f)
        out[ranker.name] = s
        log.info(f"    {ranker.name:<16} covered {len(s.labels)}/{n - eval_start}")

    # Intersection: a ranker that abstains on debutants is otherwise scoring an easier
    # fight set, which would make the comparison invalid. Computed over CANDIDATES only —
    # `market` covers just the ~25% of bouts with stored odds, so including it would
    # shrink the common set to the odds coverage and hide the candidates' real overlap.
    candidates = [n for n in out if n not in REFERENCE_RANKERS]
    common = (set.intersection(*(set(out[n].fight_ids) for n in candidates))
              if candidates else set())
    report = {}
    for name, s in out.items():
        pos = {fid: i for i, fid in enumerate(s.fight_ids)}
        inter = [pos[f] for f in s.fight_ids if f in common]
        top = [i for i, t in enumerate(s.top15) if t]
        report[name] = {
            "own": _metrics(s),
            "common": _metrics(s, inter),
            "top15": _metrics(s, top),
            "coverage": len(s.labels),
        }
    report["_common_n"] = len(common)
    report["_scored"] = out
    return report


def _kendall_tau(a: list[int], b: list[int]) -> float:
    """Tau-b over the fighters common to both orderings."""
    common = [f for f in a if f in set(b)]
    if len(common) < 2:
        return float("nan")
    ra = {f: i for i, f in enumerate(a)}
    rb = {f: i for i, f in enumerate(b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            s = (ra[x] - ra[y]) * (rb[x] - rb[y])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else float("nan")


def stability(make_ranker, fights: list, checkpoints: list[date],
              top_n: int = 15) -> dict:
    """How much a division's published order churns between two dates.

    A ranking that predicts well but reorders the top 15 every card is unusable as a
    display — the number people actually look at is the rank, and it is supposed to mean
    something durable. Measured on consecutive checkpoints ~90 days apart.

    Takes a FACTORY, not an instance. An online ranker holds no time-indexed state, so a
    single warmed instance returns the same rating whatever `as_of` says — which scored
    a meaningless tau of exactly 1.0000 for every online ranker. Each checkpoint must
    rebuild from only the fights that preceded it.
    """
    taus, moves = [], []
    prev: dict[str, list[int]] = {}
    for cp in checkpoints:
        past = [f for f in fights if f.date <= cp]
        ranker = make_ranker()
        if hasattr(ranker, "fit"):
            ranker.fit(past)
        else:
            for f in past:
                ranker.observe(f)
        elite = _divisional_order(ranker, past, cp, top_n)
        for div, order in elite.items():
            if div in prev:
                t = _kendall_tau(prev[div], order)
                if t == t:  # not NaN
                    taus.append(t)
                pa = {f: i for i, f in enumerate(prev[div])}
                for i, f in enumerate(order):
                    if f in pa:
                        moves.append(abs(pa[f] - i))
        prev = elite
    return {
        "kendall_tau": sum(taus) / len(taus) if taus else float("nan"),
        "mean_rank_move": sum(moves) / len(moves) if moves else float("nan"),
        "n_pairs": len(taus),
    }


def _divisional_order(ranker, past_fights: list, as_of: date,
                      top_n: int) -> dict[str, list[int]]:
    division: dict[int, str] = {}
    active: dict[int, date] = {}
    for f in past_fights:
        wc = classify_weight_class(f.weight_class)
        for fid in (f.red_fighter_id, f.blue_fighter_id):
            if f.date >= active.get(fid, date.min):
                active[fid] = f.date
                if wc != "unknown":
                    division[fid] = wc
    by_div: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for fid, div in division.items():
        if (as_of - active[fid]).days > 548:
            continue
        r = ranker.rate(fid, as_of)
        if r is not None:
            by_div[div].append((r[0], fid))
    out = {}
    for div, rows in by_div.items():
        rows.sort(reverse=True)
        out[div] = [fid for _, fid in rows[:top_n]]
    return out


def _per_fight_correct(s: Scored) -> dict[int, float]:
    return {fid: (0.5 if d == 0 else (1.0 if (d > 0) == (y == 1) else 0.0))
            for fid, d, y in zip(s.fight_ids, s.diffs, s.labels)}


def paired_bootstrap(a: Scored, b: Scored, subset: set[int] | None = None,
                     n_boot: int = 2000, seed: int = 12345) -> dict:
    """Paired accuracy difference (a - b) with a bootstrap CI.

    Every ranker is scored on the same fights, so an unpaired standard error badly
    understates the power here — most of the variance is "was this fight predictable at
    all", which is shared and cancels. Without this, differences of ~0.01 look like
    noise against an unpaired SE of ~0.013 when the paired CI is several times tighter.
    """
    ca, cb = _per_fight_correct(a), _per_fight_correct(b)
    ids = [f for f in ca if f in cb and (subset is None or f in subset)]
    if not ids:
        return {"n": 0}
    diffs = [ca[f] - cb[f] for f in ids]
    point = sum(diffs) / len(diffs)

    # Deterministic LCG so reruns are reproducible without touching global RNG state.
    state, n = seed, len(diffs)
    boots = []
    for _ in range(n_boot):
        tot = 0.0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            tot += diffs[state % n]
        boots.append(tot / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(int(0.975 * n_boot), n_boot - 1)]
    return {"n": len(ids), "delta": point, "lo": lo, "hi": hi,
            "significant": lo > 0 or hi < 0}


def print_pairwise(report_scored: dict, baseline: str) -> None:
    base = report_scored.get(baseline)
    if base is None:
        return
    print(f"\nPaired accuracy vs {baseline} (95% bootstrap CI, same fights):")
    print(f"{'ranker':<32}{'delta':>9}{'95% CI':>20}{'n':>7}   verdict")
    print("-" * 82)
    for name, s in report_scored.items():
        if name == baseline:
            continue
        r = paired_bootstrap(s, base)
        if not r.get("n"):
            continue
        ci = f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"
        verdict = "SIGNIFICANT" if r["significant"] else "not distinguishable"
        print(f"{name:<32}{r['delta']:>+9.4f}{ci:>20}{r['n']:>7}   {verdict}")
    print("-" * 82)


def print_report(report: dict) -> None:
    n_common = report.pop("_common_n", 0)
    report.pop("_scored", None)
    print(f"\n{'ranker':<18}{'n':>7}{'acc':>8}{'auc':>8}{'brier':>8}{'logloss':>9}"
          f"{'acc@int':>9}{'acc@t15':>9}{'n@t15':>7}")
    print("-" * 82)
    for name, r in report.items():
        o, c, t = r["own"], r["common"], r["top15"]
        print(f"{name:<18}{o['n']:>7}{o.get('acc', float('nan')):>8.4f}"
              f"{o.get('auc', float('nan')):>8.4f}{o.get('brier', float('nan')):>8.4f}"
              f"{o.get('logloss', float('nan')):>9.4f}"
              f"{c.get('acc', float('nan')):>9.4f}"
              f"{t.get('acc', float('nan')):>9.4f}{t.get('n', 0):>7}")
    print("-" * 82)
    print(f"common fight set: {n_common}")
    print("\nSanity gates: always-red ~0.50, devigged market ~0.62-0.66.")
    print("A candidate outside that bracket means a harness bug, not a great ranker.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankers", default="always_red,market,elo,points")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--baseline", default="", help="ranker to paired-test everything against")
    args = ap.parse_args()

    from app.database import SessionLocal
    from app.models.ufc import UFCFight
    from app.services.ufc.ranking_baselines import build_ranker

    db = SessionLocal()
    try:
        fights = db.query(UFCFight).order_by(UFCFight.date, UFCFight.id).all()
        rankers = [build_ranker(n.strip(), db) for n in args.rankers.split(",") if n.strip()]
        report = evaluate(rankers, fights, n_folds=args.folds, eval_frac=args.eval_frac)
    finally:
        db.close()
    scored = report.get("_scored", {})
    print_report(report)
    if args.baseline:
        print_pairwise(scored, args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
