"""Step 3D: train against the market. Walk-forward, read-only.

The served blend fits the winner model on its own and then stacks it with the market
(b_mkt 0.92, b_model 0.26). Here CatBoost instead learns only a CORRECTION to the market:

    logit P(red) = logit(devigged market) + f(features)

fitted on priced fights with the market's log-odds as a fixed baseline, so f starts at 0
("the market is right") and only moves where features repeatedly show the market wrong.
Corner-swap augmented (baseline negates), symmetric at predict time, and the correction is
shrunk on a held-out slice with MarketAnchor (logit p = logit q + b f + c, b in [0, 1]), so a
noisy f collapses back to the market instead of hurting.

Same 8 expanding folds over the last 40% of fights as fast_wf / feature_wf; priced fights only.
Reports log loss vs the market, and vs the baseline ensemble's anchored stack on the same fights
(from a feature_wf results CSV).

    PYTHONPATH=. venv/bin/python -m scripts.market_offset_wf [--extra newcomer:both mismatch]
"""
from __future__ import annotations

import argparse
import importlib
import logging

import numpy as np
import pandas as pd

from app.services.ufc.ensemble import CATBOOST_PARAMS, feature_set
from app.services.ufc.market_anchor import MarketAnchor, devig
from app.services.ufc.model import _corner_swap_augment, _fillna_from_train
from scripts.feature_wf import EVAL_FRAC, OUT_DIR, FeatureContext, _ci, _ll, build_matrix

log = logging.getLogger("market_offset_wf")


def _swap_idx(names):
    """Column permutation/sign for a corner swap (diff_ negate, red_/blue_ swap)."""
    pos = {n: i for i, n in enumerate(names)}
    perm, sign = np.arange(len(names)), np.ones(len(names))
    for i, n in enumerate(names):
        if n.startswith("diff_"):
            sign[i] = -1
        elif n.startswith("red_") and "blue_" + n[4:] in pos:
            perm[i] = pos["blue_" + n[4:]]
        elif n.startswith("blue_") and "red_" + n[5:] in pos:
            perm[i] = pos["red_" + n[5:]]
    return perm, sign


def fit_offset(X, y, base, names):
    from catboost import CatBoostClassifier, Pool
    params = {**CATBOOST_PARAMS, "loss_function": "Logloss", "random_seed": 42, "verbose": False,
              "allow_writing_files": False}
    cut = int(len(X) * 0.85)
    def aug(Xs, ys, bs):
        Xa, ya = _corner_swap_augment(Xs, ys, names)
        return Pool(Xa, ya, baseline=np.concatenate([bs, -bs]))
    probe = CatBoostClassifier(iterations=3000, od_type="Iter", od_wait=150, **params)
    probe.fit(aug(X[:cut], y[:cut], base[:cut]), eval_set=aug(X[cut:], y[cut:], base[cut:]), use_best_model=True)
    best = max(probe.get_best_iteration() or 50, 20)
    model = CatBoostClassifier(iterations=best, **params).fit(aug(X, y, base))
    perm, sign = _swap_idx(names)
    def correction(Xe):
        f = model.predict(Xe, prediction_type="RawFormulaVal")
        fs = model.predict(Xe[:, perm] * sign, prediction_type="RawFormulaVal")
        return (f - fs) / 2
    return correction, best


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="pit transforms to apply first, e.g. newcomer:both mismatch")
    ap.add_argument("--baseline-csv", default="newcomer_flags_wf.csv")
    ap.add_argument("--tag", default="market_offset")
    a = ap.parse_args(argv)

    matchup, features, slim = build_matrix(False)
    feats = feature_set("noodds", features)
    for spec in a.extra:
        name, _, variant = spec.partition(":")
        mod = importlib.import_module(f"app.services.ufc.pit.{name}_pit")
        if hasattr(mod, "transform"):
            new, _ = mod.transform(matchup, features, slim, FeatureContext.session,
                                   **({"variant": variant} if variant else {}))
        else:   # a compute() block, from feature_wf's cache
            from scripts.feature_wf import attach
            block = pd.read_pickle(OUT_DIR / f"{name}_block.pkl")
            new = attach(matchup, slim, block, getattr(mod, "FIGHT_LEVEL", ()), getattr(mod, "RAW", ()))
        feats += new
    n = len(matchup)
    q_all = devig(matchup["odds_red_prob"].values, matchup["odds_blue_prob"].values)
    priced = matchup["odds_red_prob"].notna().to_numpy() & np.isfinite(q_all)
    base_all = np.log(np.clip(q_all, 1e-4, 1 - 1e-4) / (1 - np.clip(q_all, 1e-4, 1 - 1e-4)))
    eval_start = int(n * (1 - EVAL_FRAC))
    bounds = np.linspace(eval_start, n, 9).astype(int)
    out = np.full(n, np.nan)
    for k in range(8):
        lo, hi = bounds[k], bounds[k + 1]
        tr = np.zeros(n, dtype=bool); tr[:lo] = True
        fold, _ = _fillna_from_train(matchup, feats, tr)
        tri = np.where(tr & priced)[0]
        cut = int(len(tri) * 0.85)
        fit_i, cal_i = tri[:cut], tri[cut:]
        X = fold[feats].to_numpy(float)
        y = matchup["red_wins"].to_numpy(float)
        corr, iters = fit_offset(X[fit_i], y[fit_i], base_all[fit_i], feats)
        f_cal = corr(X[cal_i])
        anchor = MarketAnchor().fit(1 / (1 + np.exp(-(base_all[cal_i] + f_cal))), q_all[cal_i], y[cal_i])
        te = np.arange(lo, hi)[priced[lo:hi]]
        p_raw = 1 / (1 + np.exp(-(base_all[te] + corr(X[te]))))
        out[te] = anchor.predict_proba(p_raw, q_all[te])
        log.info("  fold %d: %d priced train, iters %d, shrink b=%.3f", k + 1, len(fit_i), iters, anchor.b_)

    ev = np.arange(eval_start, n)
    ev = ev[priced[ev] & ~np.isnan(out[ev])]
    y = matchup["red_wins"].to_numpy(float)[ev]
    ll_m, ll_o = _ll(q_all[ev], y), _ll(out[ev], y)
    lo_, hi_ = _ci(ll_o - ll_m)
    print(f"\nMARKET OFFSET ({a.tag}; extra={a.extra})  priced eval fights {len(ev):,}")
    print(f"  market      {ll_m.mean():.4f}")
    print(f"  offset      {ll_o.mean():.4f}   vs market {ll_o.mean() - ll_m.mean():+.4f} [{lo_:+.4f}, {hi_:+.4f}]")
    try:
        b = pd.read_csv(OUT_DIR / a.baseline_csv)[["fight_id", "base_ensemble_anc"]]
        m = pd.DataFrame({"fight_id": matchup["fight_id"].to_numpy()[ev], "off": out[ev], "y": y,
                          "q": q_all[ev]}).merge(b, on="fight_id")
        lb, lo2 = _ll(m.base_ensemble_anc.to_numpy(), m.y.to_numpy()), _ll(m.off.to_numpy(), m.y.to_numpy())
        l3, h3 = _ci(lo2 - lb)
        print(f"  current stack (baseline ensemble + anchor) {lb.mean():.4f}; offset vs it "
              f"{lo2.mean() - lb.mean():+.4f} [{l3:+.4f}, {h3:+.4f}]  (n={len(m):,})")
    except FileNotFoundError:
        pass
    pd.DataFrame({"fight_id": matchup["fight_id"].to_numpy()[ev], "offset_p": out[ev]}).to_csv(
        OUT_DIR / f"{a.tag}_wf.csv", index=False)


if __name__ == "__main__":
    main()
