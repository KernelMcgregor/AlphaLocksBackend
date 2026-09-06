"""Ablation: was it the decorrelation penalty, or just a different model?

THE QUESTION
------------
The `decorrelated` walk-forward arm was the first thing in this project not to lose money
(+2.3% ROI at the 15% edge rung, CI [-3.5, +8.5]). But it changed TWO things at once
versus the GBT arms:

  1. a different learner  -- an MLP minimising squared error, not a GBT minimising log loss
  2. the Hubacek & Sir Eq. 59 decorrelation penalty, weighted by gamma

and the per-fold gamma selection chose **gamma = 0 in five of eight folds**, i.e. no
penalty at all. So the result may be entirely attributable to (1). Attributing it to (2)
without testing would be a guess, and guessing about why a number improved is how the
original (retracted) 72.9% happened.

DESIGN
------
Identical folds, identical features, identical everything except gamma:

  gamma = 0     -> the new learner with NO penalty (the architecture-only control)
  gamma > 0     -> penalty always on, at several strengths
  "auto"        -> the per-fold inner-window selection used by the walk-forward arm

MULTIPLE SEEDS ARE THE POINT. One MLP fit has real run-to-run variance; comparing single
runs would let noise decide the answer. Each configuration is repeated over several seeds
and reported as mean +/- sd, so a difference has to clear the training noise to count.

READING THE RESULT
------------------
  gamma=0 matches the rest      -> the penalty does nothing; the win is the architecture
  some gamma>0 clearly better   -> the penalty is real and auto-selection was picking badly
  "auto" beats every fixed gamma -> the per-fold selection is doing something genuine

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local \\
      python -m scripts.ablate_decorrelation --seeds 5
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from app.services.ufc.decorrelated import DecorrelatedModel
from app.services.ufc.glicko_service import run_glicko_inmemory
from app.services.ufc.market_anchor import devig, logit
from app.services.ufc.model import (
    DECORR_GAMMAS,
    _american_to_decimal,
    _betting_ladder,
    _bootstrap_ci,
    _fillna_from_train,
    build_features,
    build_matchup_df,
    load_fight_data,
    select_winner_features,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
for noisy in ("model", "glicko_service", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _market(d):
    m = devig(d["odds_red_prob"].values, d["odds_blue_prob"].values)
    return np.where(d["odds_red_prob"].notna().values, m, np.nan)


def prepare_folds(matchup, arm_features, bounds, n_folds, top_n):
    """Imputation + feature selection per fold, ONCE.

    Neither depends on gamma or the seed, but mutual_info_classif over ~270 features is
    the most expensive step in the loop. Recomputing it for every config x seed made the
    ablation ~10x slower than the model fitting it was supposed to measure.
    """
    n = len(matchup)
    prepared = []
    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        if hi <= lo:
            continue
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:lo] = True
        fold_df, _ = _fillna_from_train(matchup, arm_features, train_mask)
        selected = select_winner_features(fold_df, arm_features, train_mask,
                                          top_n=top_n, include_odds=True, verbose=False)
        prepared.append((lo, hi, fold_df, selected))
    return prepared


def run_config(prepared, gamma, seed, calibrate=False):
    """One full walk-forward pass at a fixed gamma (or 'auto' for per-fold selection).

    `calibrate=True` fits isotonic regression on the inner validation window and applies
    it to the test fold. Fold-scoped on purpose: fitting calibration on the reported eval
    slice was one of the sources of optimism in the original (retracted) numbers.
    """
    proba, chosen = [], []

    for lo, hi, fold_df, selected in prepared:
        train_df, test_df = fold_df.iloc[:lo], fold_df.iloc[lo:hi]
        inner = int(len(train_df) * 0.8)
        itr, iva = train_df.iloc[:inner], train_df.iloc[inner:]

        g = gamma
        if gamma == "auto":
            best_g, best_roi = 0.0, -np.inf
            for cand in DECORR_GAMMAS:
                m = DecorrelatedModel(gamma=cand, seed=seed).fit(
                    itr[selected].values, itr["red_wins"].values, _market(itr),
                    iva[selected].values, iva["red_wins"].values, _market(iva))
                pv, mv = m.predict_proba(iva[selected].values), _market(iva)
                ok = np.isfinite(mv)
                if ok.sum() < 50:
                    continue
                ladder = _betting_ladder(
                    pv[ok], iva["red_wins"].values[ok],
                    np.array([_american_to_decimal(o)
                              for o in iva["odds_red_american"].values[ok]]),
                    np.array([_american_to_decimal(o)
                              for o in iva["odds_blue_american"].values[ok]]),
                    mv[ok], 1 - mv[ok])
                rois = [r["roi_pct"] for r in ladder
                        if r["bets"] >= 25 and np.isfinite(r["roi_pct"])]
                roi = float(np.mean(rois)) if rois else -np.inf
                if roi > best_roi:
                    best_roi, best_g = roi, cand
            g = best_g
        chosen.append(g)

        mdl = DecorrelatedModel(gamma=g, seed=seed).fit(
            train_df[selected].values, train_df["red_wins"].values, _market(train_df),
            iva[selected].values, iva["red_wins"].values, _market(iva))
        pt = mdl.predict_proba(test_df[selected].values)

        if calibrate:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(mdl.predict_proba(iva[selected].values), iva["red_wins"].values)
            pt = iso.predict(pt)

        proba.append(pt)

    return np.concatenate(proba), chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--eval-frac", type=float, default=0.4)
    ap.add_argument("--top-n", type=int, default=39)
    args = ap.parse_args()

    t0 = time.time()
    print("Building features (fresh Glicko)...", flush=True)
    snaps = run_glicko_inmemory()
    df, rd = load_fight_data()
    probe = build_matchup_df(build_features(df.copy(), rd.copy(),
                                            glicko_snapshots=snaps))[0]
    cutoff = probe.sort_values("date")["date"].iloc[int(len(probe) * (1 - args.eval_frac))]
    feats_df = build_features(df, rd, style_cutoff_date=cutoff, glicko_snapshots=snaps)
    matchup, features = build_matchup_df(feats_df)
    matchup = matchup.sort_values("date").reset_index()

    n = len(matchup)
    eval_start = int(n * (1 - args.eval_frac))
    bounds = np.linspace(eval_start, n, args.folds + 1).astype(int)
    y = matchup["red_wins"].to_numpy()[eval_start:]
    sub = matchup.iloc[eval_start:]
    has_odds = sub["odds_red_prob"].notna().values & sub["odds_blue_prob"].notna().values

    mkt = devig(sub["odds_red_prob"].values, sub["odds_blue_prob"].values)[has_odds]
    y_odds = y[has_odds]
    red_dec = np.array([_american_to_decimal(o)
                        for o in sub["odds_red_american"].values[has_odds]])
    blue_dec = np.array([_american_to_decimal(o)
                         for o in sub["odds_blue_american"].values[has_odds]])
    mkt_acc = float(((mkt >= 0.5).astype(int) == y_odds).mean())

    print(f"  {n} fights, {len(y)} eval, {has_odds.sum()} priced. "
          f"market accuracy {mkt_acc:.4f}   ({time.time() - t0:.0f}s)", flush=True)
    print("Preparing folds (imputation + feature selection, once)...", flush=True)
    prepared = prepare_folds(matchup, features, bounds, args.folds, args.top_n)
    print(f"  {len(prepared)} folds ready   ({time.time() - t0:.0f}s)\n", flush=True)

    # (gamma, calibrate). The ablation already showed gamma=0 best; the open question
    # now is whether isotonic calibration on top helps, since the MLP's calibration
    # error is 0.129 against 0.028 for the GBT and 0.020 for the market.
    configs = [(0.0, False), (0.0, True), (0.25, True), (1.0, True)]
    rows = []
    for g, cal in configs:
        accs, corrs, rois15, rois8, briers, gam = [], [], [], [], [], []
        for seed in range(1, args.seeds + 1):
            p, chosen = run_config(prepared, g, seed, calibrate=cal)
            po = p[has_odds]
            accs.append(float(((po >= 0.5).astype(int) == y_odds).mean()))
            briers.append(float(np.mean((po - y_odds) ** 2)))
            corrs.append(float(np.corrcoef(logit(po), logit(mkt))[0, 1]))
            ladder = _betting_ladder(po, y_odds, red_dec, blue_dec, mkt, 1 - mkt)
            by = {r["min_edge"]: r for r in ladder}
            rois15.append(by[0.15]["roi_pct"])
            rois8.append(by[0.08]["roi_pct"])
            gam.extend(chosen)
        label = ("auto" if g == "auto" else f"{g:g}") + ("+iso" if cal else "")
        rows.append((label, accs, briers, corrs, rois8, rois15, gam))
        print(f"  gamma={label:<5} acc={np.mean(accs):.4f}  brier={np.mean(briers):.4f}  "
              f"corr={np.mean(corrs):.3f}  ROI@8%={np.mean(rois8):+.2f}  "
              f"ROI@15%={np.mean(rois15):+.2f}   ({time.time() - t0:.0f}s)", flush=True)

    print("\n" + "=" * 92)
    print(f"ABLATION — {args.seeds} seeds per configuration, mean +/- sd")
    print("=" * 92)
    print(f"{'gamma':<7}{'accuracy':>16}{'brier':>15}{'corr(mkt)':>14}"
          f"{'ROI@8%':>16}{'ROI@15%':>16}")
    print("-" * 92)
    for label, accs, briers, corrs, r8, r15, gam in rows:
        print(f"{label:<7}"
              f"{np.mean(accs):>10.4f}+/-{np.std(accs):<4.4f}"
              f"{np.mean(briers):>9.4f}+/-{np.std(briers):<4.4f}"
              f"{np.mean(corrs):>8.3f}+/-{np.std(corrs):<4.3f}"
              f"{np.mean(r8):>+10.2f}+/-{np.std(r8):<4.2f}"
              f"{np.mean(r15):>+10.2f}+/-{np.std(r15):<4.2f}")
    print("-" * 92)
    print(f"market accuracy {mkt_acc:.4f}   |   ROI is flat-stake, thresholds frozen")

    base = [r for r in rows if r[0] == "0"][0]
    iso = [r for r in rows if r[0] == "0+iso"]
    if iso:
        iso = iso[0]
        d15 = np.mean(iso[5]) - np.mean(base[5])
        sd = np.hypot(np.std(base[5]), np.std(iso[5]))
        print(f"\nCalibration effect at gamma=0, 15% rung: {d15:+.2f} pp "
              f"(training-noise sd {sd:.2f} pp)")
        print(f"  Brier  {np.mean(base[2]):.4f} -> {np.mean(iso[2]):.4f}")
        print(f"  acc    {np.mean(base[1]):.4f} -> {np.mean(iso[1]):.4f}")
        print("Isotonic is monotone, so accuracy/AUC barely move; Brier and the betting")
        print("thresholds are what it changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
