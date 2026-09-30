"""Walk-forward evaluation of method-of-victory models.

Same window and folds as scripts/fast_wf.py (last 40% of decided fights, 8 expanding
folds), so numbers line up with the winner model and ensemble_oof.csv.

Arms:
  base_rate   class shares by division x scheduled rounds, from training fights
  old_style   winner-agnostic 3-class CatBoost on the red/blue matrix + method ratings
              (stands in for the current production model, which is also winner-agnostic)
  v2_base     conditional model, winner-matrix features only
  v2_full     + method ratings, Sherdog method shares, alignment features
  v2_mkt      + the winner's devigged market probability
  v2_full_nocal / v2_full_cb / v2_full_mc   calibration / member / structure variants

Metrics (lower is better), with paired bootstrap CIs vs base_rate:
  cond_ll     log loss of P(method | actual winner)          (v2 arms, base_rate)
  marg_ll     log loss of P(method), mixing both winners by the ensemble's OOF P(red)
  six_ll      log loss of the 6-way cell (winner x method)

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.method_wf \\
        --out /tmp/method_wf.csv [--rebuild] [--arms base_rate,old_style,v2_base,v2_full,v2_mkt]
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from app.services.ufc.method_v2 import (
    CATBOOST_PARAMS, ConditionalMethodModel, build_training_matrix, feature_names, joint,
    marginal, orient,
)
from app.services.ufc.model import MODEL_DIR

log = logging.getLogger("method_wf")
OOF = MODEL_DIR / "ensemble_oof.csv"
ALL_ARMS = ("base_rate", "old_style", "v2_base", "v2_full", "v2_mkt")
V2_ARMS = {
    "v2_base": ("base", {}), "v2_full": ("full", {}), "v2_mkt": ("full_mkt", {}),
    "v2_full_nocal": ("full", {"calibrate": False}),          # members on all rows, raw
    "v2_full_cb": ("full", {"calibrate": False, "backends": ("catboost",)}),
    "v2_full_mc": ("full", {"structure": "multiclass"}),      # one 3-class CatBoost
}


def build_matrix(rebuild: bool) -> pd.DataFrame:
    return build_training_matrix(rebuild)


def _ll(p):
    return -np.log(np.clip(p, 1e-6, 1.0))


def _division(m: pd.DataFrame) -> np.ndarray:
    divs = [c for c in m.columns if c.startswith("fight_div_")]
    return np.array(divs)[m[divs].to_numpy(float).argmax(axis=1)]


def base_rates(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    y = train["outcome_method_class"].to_numpy(int)
    glob = (np.bincount(y, minlength=3) + 1) / (len(y) + 3)
    key_tr = pd.Series(_division(train)).str.cat(train["fight_is_five_round"].astype(str).values)
    key_te = pd.Series(_division(test)).str.cat(test["fight_is_five_round"].astype(str).values)
    table = {}
    for k, idx in key_tr.groupby(key_tr).groups.items():
        c = np.bincount(y[idx], minlength=3)
        table[k] = (c + 20 * glob) / (c.sum() + 20)  # shrink small groups to global
    return np.array([table.get(k, glob) for k in key_te])


def old_style(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from catboost import CatBoostClassifier
    feats = [c for c in train.columns if c.startswith(("diff_", "red_", "blue_", "fight_"))
             and c not in ("red_wins", "fight_id")]
    X, y = train[feats].to_numpy(float), train["outcome_method_class"].to_numpy(int)
    cut = int(len(X) * 0.85)
    p = {**CATBOOST_PARAMS, "loss_function": "MultiClass", "random_seed": 42,
         "verbose": False, "allow_writing_files": False}
    probe = CatBoostClassifier(iterations=3000, od_type="Iter", od_wait=150, **p)
    probe.fit(X[:cut], y[:cut], eval_set=(X[cut:], y[cut:]), use_best_model=True)
    m = CatBoostClassifier(iterations=max(probe.get_best_iteration() or 100, 50), **p).fit(X, y)
    return m.predict_proba(test[feats].to_numpy(float))


def run(matchup: pd.DataFrame, arms, n_folds: int = 8, eval_frac: float = 0.4) -> pd.DataFrame:
    n = len(matchup)
    eval_start = int(n * (1 - eval_frac))
    bounds = np.linspace(eval_start, n, n_folds + 1).astype(int)
    red_won = matchup["red_wins"].to_numpy(float) == 1
    or_actual = orient(matchup, red_won)
    or_red, or_blue = orient(matchup, np.ones(n, bool)), orient(matchup, np.zeros(n, bool))
    y = matchup["outcome_method_class"].to_numpy(int)
    res = matchup[["fight_id", "date", "red_wins", "outcome_method_class"]].iloc[eval_start:].copy()
    for arm in arms:
        cr = np.full((n, 3), np.nan); cb = np.full((n, 3), np.nan); mg = np.full((n, 3), np.nan)
        for k in range(n_folds):
            lo, hi = bounds[k], bounds[k + 1]
            tr, te = matchup.iloc[:lo], matchup.iloc[lo:hi]
            if arm == "base_rate":
                p = base_rates(tr, te); cr[lo:hi] = cb[lo:hi] = p
            elif arm == "old_style":
                mg[lo:hi] = old_style(tr, te)
            else:
                fs, kw = V2_ARMS[arm]
                feats = feature_names(or_actual, fs)
                model = ConditionalMethodModel(feats, **kw).fit(or_actual.iloc[:lo], y[:lo])
                cr[lo:hi] = model.predict(or_red.iloc[lo:hi])
                cb[lo:hi] = model.predict(or_blue.iloc[lo:hi])
            log.info(f"  {arm} fold {k + 1}/{n_folds}")
        for j, c in enumerate(("ko", "sub", "dec")):
            res[f"{arm}_red_{c}"] = cr[eval_start:, j]
            res[f"{arm}_blue_{c}"] = cb[eval_start:, j]
            res[f"{arm}_marg_{c}"] = mg[eval_start:, j]
    return res


def score(res: pd.DataFrame, arms, n_boot: int = 2000) -> pd.DataFrame:
    oof = pd.read_csv(OOF, dtype={"fight_id": str})[["fight_id", "model_prob"]]
    r = res.assign(fight_id=res["fight_id"].astype(str)).merge(oof, on="fight_id", how="left")
    y = r["outcome_method_class"].to_numpy(int)
    red = r["red_wins"].to_numpy(float) == 1
    p_red = r["model_prob"].to_numpy(float)
    ok = ~np.isnan(p_red)
    losses = {}
    for arm in arms:
        cr = r[[f"{arm}_red_{c}" for c in ("ko", "sub", "dec")]].to_numpy(float)
        cb = r[[f"{arm}_blue_{c}" for c in ("ko", "sub", "dec")]].to_numpy(float)
        mg = r[[f"{arm}_marg_{c}" for c in ("ko", "sub", "dec")]].to_numpy(float)
        rows = np.arange(len(r))
        if not np.isnan(cr).all():
            cond = np.where(red, cr[rows, y], cb[rows, y])
            six = joint(np.where(ok, p_red, 0.5), cr, cb)
            mg = marginal(six)
            losses[(arm, "cond_ll")] = _ll(cond)
            cell = np.where(red, six[rows, y], six[rows, 3 + y])
            losses[(arm, "six_ll")] = np.where(ok, _ll(cell), np.nan)
        losses[(arm, "marg_ll")] = np.where(ok, _ll(mg[rows, y]), np.nan)
    rng = np.random.default_rng(0)
    out = []
    for (arm, metric), l in losses.items():
        base = losses.get(("base_rate", metric))
        row = {"arm": arm, "metric": metric, "n": int(np.isfinite(l).sum()), "ll": np.nanmean(l)}
        if base is not None and arm != "base_rate":
            d = l - base
            d = d[np.isfinite(d)]
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
            row.update(vs_base=d.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5))
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    a = ap.parse_args()
    arms = a.arms.split(",")
    matchup = build_matrix(a.rebuild)
    log.info(f"{len(matchup)} decided fights; class shares "
             f"{np.bincount(matchup['outcome_method_class'].astype(int), minlength=3) / len(matchup)}")
    res = run(matchup, arms)
    res.to_csv(a.out, index=False)
    table = score(res, arms)
    pd.set_option("display.width", 160)
    print(table.round(4).to_string(index=False))
    table.to_csv(a.out.replace(".csv", "_scores.csv"), index=False)


if __name__ == "__main__":
    main()
