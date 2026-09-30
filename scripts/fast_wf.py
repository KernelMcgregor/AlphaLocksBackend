"""Fast walk-forward for model/feature experiments.

Same folds, imputation, feature selection, isotonic calibration and MarketAnchor
stacking as model.walk_forward_eval, but:
  - the feature matrix is built once and cached (--rebuild to refresh),
  - only the arms you name are run (feature set x backend x top_n),
  - output is a predictions CSV readable by scripts/compare_eval.py.

Arm spec: "<features>:<backend>:<top_n>", e.g.
    noodds:hgb:39        current no-odds arm
    noodds_noglicko:hgb:39
    noodds:hgb:0         top_n=0 -> no MI selection, all candidate features
    noodds:catboost:0
    noodds:tabpfn:60
    noodds:logit:60      standardised L2 logistic regression

Every arm also gets a "<arm>+anchor" column: the MarketAnchor stack on its calibrated
output, which is the number that says whether the model adds anything to the market.

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.fast_wf \\
        noodds:hgb:39 noodds:hgb:0 --out /tmp/wf.csv
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from app.services.ufc.market_anchor import MarketAnchor, devig
from app.services.ufc.ensemble import feature_set, fit_backend
from app.services.ufc.model import (
    MODEL_DIR, _fillna_from_train, build_features, build_matchup_df, load_fight_data,
    select_winner_features,
)

log = logging.getLogger("fast_wf")
from app.services.ufc.model import CAREER_FEATURES_ENABLED, SHORT_NOTICE_ENABLED

# Separate caches per feature configuration so runs never mix.
CACHE = MODEL_DIR / ("fast_wf_matchup" + ("_career" if CAREER_FEATURES_ENABLED else "")
                     + ("_sn" if SHORT_NOTICE_ENABLED else "") + ".pkl")


def build_matrix(rebuild: bool) -> tuple[pd.DataFrame, list[str]]:
    if CACHE.exists() and not rebuild:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    from app.services.ufc.glicko_service import run_glicko_inmemory
    snaps = run_glicko_inmemory()
    df, rd = load_fight_data()
    df = build_features(df, rd, glicko_snapshots=snaps)
    matchup, features = build_matchup_df(df)
    matchup = matchup.sort_values("date").reset_index()
    with open(CACHE, "wb") as f:
        pickle.dump((matchup, features), f)
    return matchup, features


def fit_predict(backend: str, X_fit, y_fit, X_eval_list, names):
    """Returns P(red) for each matrix in X_eval_list (backends live in ensemble.py)."""
    model = fit_backend(backend, X_fit, y_fit, names)
    return [model.predict(X) for X in X_eval_list]


def run_arm(spec: str, matchup: pd.DataFrame, features: list[str],
            n_folds: int = 8, eval_frac: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fname, backend, top_n = spec.split(":")
    top_n = int(top_n)
    cands = feature_set(fname, features)
    n = len(matchup)
    eval_start = int(n * (1 - eval_frac))
    bounds = np.linspace(eval_start, n, n_folds + 1).astype(int)
    out_raw = np.full(n, np.nan); out_cal = np.full(n, np.nan); out_anc = np.full(n, np.nan)
    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        train_mask = np.zeros(n, dtype=bool); train_mask[:lo] = True
        fold_df, _ = _fillna_from_train(matchup, cands, train_mask)
        if top_n > 0:
            sel = select_winner_features(fold_df, cands, train_mask, top_n=top_n,
                                         include_odds=(fname == "withodds"), verbose=False)
        else:
            sel = cands
        train_df, test_df = fold_df.iloc[:lo], fold_df.iloc[lo:hi]
        cut = int(len(train_df) * 0.85)
        fit_df, cal_df = train_df.iloc[:cut], train_df.iloc[cut:]
        p_cal_in, p_test = fit_predict(
            backend, fit_df[sel].to_numpy(float), fit_df["red_wins"].to_numpy(float),
            [cal_df[sel].to_numpy(float), test_df[sel].to_numpy(float)], sel)
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso.fit(p_cal_in, cal_df["red_wins"].to_numpy(float))
        cal_mkt = devig(cal_df["odds_red_prob"].values, cal_df["odds_blue_prob"].values)
        cal_mkt = np.where(cal_df["odds_red_prob"].notna().values, cal_mkt, np.nan)
        test_mkt = devig(test_df["odds_red_prob"].values, test_df["odds_blue_prob"].values)
        test_mkt = np.where(test_df["odds_red_prob"].notna().values, test_mkt, np.nan)
        anchor = MarketAnchor().fit(iso.predict(p_cal_in), cal_mkt,
                                    cal_df["red_wins"].to_numpy(float))
        out_raw[lo:hi] = p_test
        out_cal[lo:hi] = iso.predict(p_test)
        out_anc[lo:hi] = anchor.predict_proba(out_cal[lo:hi], test_mkt)
        log.info(f"  {spec} fold {k+1}/{n_folds} feats={len(sel)} anchor b={anchor.b_:.3f}")
    return out_raw, out_cal, out_anc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    matchup, features = build_matrix(a.rebuild)
    n = len(matchup)
    eval_start = int(n * 0.6)
    res = matchup[["fight_id", "date", "red_wins", "odds_red_prob"]].copy()
    for spec in a.arms:
        raw, cal, anc = run_arm(spec, matchup, features)
        tag = spec.replace(":", "_")
        res[f"{tag}_proba"] = raw
        res[f"{tag}_proba_cal"] = cal
        res[f"{tag}+anchor_proba"] = anc
    res = res.iloc[eval_start:]
    res.to_csv(a.out, index=False)
    log.info(f"wrote {a.out}")


if __name__ == "__main__":
    main()
