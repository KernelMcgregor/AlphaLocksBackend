"""Tune CatBoost and test feature-group pruning WITHOUT touching the evaluation window.

The walk-forward evaluation (scripts/fast_wf.py) scores the last 40% of fights. Settings
chosen by looking at those fights would flatter the model, so everything here uses only
fights BEFORE that window: fit on the first 80% of the pre-window fights, score on the
remaining 20% (roughly 2016-2020 -> 2020-mid 2022). The chosen settings are then evaluated
once on the real window with fast_wf.

  --tune N      Optuna search over CatBoost settings (N trials)
  --ablate      drop each feature group in turn, report the change in validation log loss

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.tune_ensemble --tune 40
    DATABASE_URL=... python -m scripts.tune_ensemble --ablate
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression

from app.services.ufc.ensemble import feature_set, fit_backend
from app.services.ufc.model import MODEL_DIR, _fillna_from_train

log = logging.getLogger("tune_ensemble")

#: Feature groups by name pattern (matched against the red/blue/diff/fight_ feature name).
GROUPS = {
    "raw_rolling": lambda f: any(p in f for p in ("_avg_", "_recent_", "_last3_"))
                             and "composite" not in f and "_r1_" not in f and "_late_" not in f,
    "composites": lambda f: "composite" in f,
    "round_profile": lambda f: any(p in f for p in ("_r1_", "_late_", "_output_", "ctrl_trend")),
    "elo_adj_stats": lambda f: "elo_adj" in f,
    "elo": lambda f: ("_elo" in f or f.endswith("elo_expected")) and "elo_adj" not in f
                     and "pro_elo" not in f,
    "glicko": lambda f: "glicko" in f,
    "expected_stats": lambda f: "_xs_" in f,
    "sherdog_career": lambda f: "_pro_" in f or "ufc_experience_share" in f,
    "style_clusters": lambda f: "style_" in f,
    "age_curve": lambda f: any(p in f for p in ("_age", "years_past", "years_to", "age_resid")),
    "layoff": lambda f: any(p in f for p in ("days_since", "layoff", "short_turnaround")),
    "damage": lambda f: any(p in f for p in ("head_strikes_absorbed", "knockdowns_absorbed",
                                             "ko_loss", "damage_per", "fight_minutes")),
    "division_moves": lambda f: any(p in f for p in ("division", "moved_", "catchweight")),
    "physical_bio": lambda f: any(p in f for p in ("height", "reach", "weight_lbs", "stance",
                                                   "bio_")),
    "record": lambda f: any(p in f for p in ("career_win_pct", "career_fights", "streak",
                                             "finish_rate", "been_finished")),
    "fight_context": lambda f: f.startswith("fight_"),
}


def load_split():
    with open(MODEL_DIR / "fast_wf_matchup_career.pkl", "rb") as f:
        matchup, features = pickle.load(f)
    n = len(matchup)
    eval_start = int(n * 0.6)                 # same boundary as fast_wf / walk_forward_eval
    pre = matchup.iloc[:eval_start].reset_index(drop=True)
    cut = int(len(pre) * 0.8)
    cands = feature_set("noodds", features)
    mask = np.zeros(len(pre), dtype=bool); mask[:cut] = True
    filled, _ = _fillna_from_train(pre, cands, mask)
    log.info(f"tuning split: fit {cut} fights ({pre['date'].iloc[0]} -> {pre['date'].iloc[cut-1]}), "
             f"validate {len(pre)-cut} ({pre['date'].iloc[cut]} -> {pre['date'].iloc[-1]}); "
             f"eval window starts {matchup['date'].iloc[eval_start]}")
    return filled.iloc[:cut], filled.iloc[cut:], cands


def rolling_blocks(n_blocks: int = 4):
    """Rolling-origin splits inside the pre-window data: for each of the last n_blocks
    equal blocks, fit on everything before it and score on it. Scores ~65% of the
    pre-window fights instead of one 20% block."""
    with open(MODEL_DIR / "fast_wf_matchup_career.pkl", "rb") as f:
        matchup, features = pickle.load(f)
    eval_start = int(len(matchup) * 0.6)
    pre = matchup.iloc[:eval_start].reset_index(drop=True)
    cands = feature_set("noodds", features)
    edges = np.linspace(int(len(pre) * 0.35), len(pre), n_blocks + 1).astype(int)
    splits = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = np.zeros(len(pre), dtype=bool); mask[:lo] = True
        filled, _ = _fillna_from_train(pre, cands, mask)
        splits.append((filled.iloc[:lo], filled.iloc[lo:hi]))
    log.info("rolling blocks: " + "; ".join(
        f"fit ..{v['date'].iloc[0]} score {v['date'].iloc[0]}..{v['date'].iloc[-1]} (n={len(v)})"
        for _, v in splits))
    return splits, cands


def cv_score(backend, splits, cols, params=None) -> float:
    """Fight-weighted mean validation log loss over the rolling blocks."""
    lls = [(score(backend, f, v, cols, params), len(v)) for f, v in splits]
    return float(sum(l * n for l, n in lls) / sum(n for _, n in lls))


def tune_cv(n_trials: int) -> dict:
    import optuna
    splits, cols = rolling_blocks()
    base = cv_score("catboost", splits, cols)
    prev = json.loads((MODEL_DIR / "catboost_tuning.json").read_text())["best_params"] \
        if (MODEL_DIR / "catboost_tuning.json").exists() else None
    log.info(f"rolling-CV log loss, default CatBoost: {base:.4f}")
    if prev:
        log.info(f"rolling-CV log loss, single-split winner: {cv_score('catboost', splits, cols, prev):.4f}")

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 50, log=True),
            "rsm": trial.suggest_float("rsm", 0.2, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        return cv_score("catboost", splits, cols, params)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
    if prev:
        study.enqueue_trial({k: v for k, v in prev.items() if k != "bootstrap_type"})
    study.optimize(objective, n_trials=n_trials)
    best = {**study.best_params, "bootstrap_type": "Bernoulli"}
    log.info(f"rolling-CV best {study.best_value:.4f} (default {base:.4f}): {best}")
    out = {"method": "rolling-cv", "default_cv_ll": base, "best_cv_ll": study.best_value,
           "best_params": best}
    (MODEL_DIR / "catboost_tuning.json").write_text(json.dumps(out, indent=2))
    return out


def ablate_cv(params: dict | None, groups=("raw_rolling", "age_curve", "sherdog_career",
                                          "expected_stats", "layoff")) -> None:
    """Re-check the near-zero groups from the single-split ablation with rolling CV."""
    splits, cols = rolling_blocks()
    full = np.mean([cv_score("catboost", splits, cols, params), cv_score("hgb", splits, cols)])
    log.info(f"rolling-CV all features (catboost+hgb mean): {full:.4f}")
    for g in groups:
        drop = [c for c in cols if GROUPS[g](c)]
        keep = [c for c in cols if c not in drop]
        ll = np.mean([cv_score("catboost", splits, keep, params), cv_score("hgb", splits, keep)])
        log.info(f"  rolling-CV drop {g:16s} ({len(drop):3d}): change {ll - full:+.4f}")


def score(backend, fit_df, val_df, cols, params=None) -> float:
    """Fit on fit_df (last 15% held out for isotonic calibration, as in production)."""
    c = int(len(fit_df) * 0.85)
    tr, cal = fit_df.iloc[:c], fit_df.iloc[c:]
    m = fit_backend(backend, tr[cols].to_numpy(float), tr["red_wins"].to_numpy(float), cols, params)
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(m.predict(cal[cols].to_numpy(float)), cal["red_wins"].to_numpy(float))
    p = np.clip(iso.predict(m.predict(val_df[cols].to_numpy(float))), 1e-6, 1 - 1e-6)
    y = val_df["red_wins"].to_numpy(float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def tune(n_trials: int) -> dict:
    import optuna
    fit_df, val_df, cols = load_split()
    base = score("catboost", fit_df, val_df, cols)
    log.info(f"default CatBoost validation log loss: {base:.4f}")

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 50, log=True),
            "rsm": trial.suggest_float("rsm", 0.2, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        return score("catboost", fit_df, val_df, cols, params)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    best = {**study.best_params, "bootstrap_type": "Bernoulli"}
    log.info(f"best validation log loss {study.best_value:.4f} (default {base:.4f}): {best}")
    out = {"default_val_ll": base, "best_val_ll": study.best_value, "best_params": best}
    (MODEL_DIR / "catboost_tuning.json").write_text(json.dumps(out, indent=2))
    return out


def ablate(params: dict | None) -> None:
    fit_df, val_df, cols = load_split()
    backends = ("catboost", "hgb")

    def ens_ll(use_cols):
        return np.mean([score(b, fit_df, val_df, use_cols, params if b == "catboost" else None)
                        for b in backends])

    full = ens_ll(cols)
    log.info(f"all {len(cols)} features: mean validation log loss (catboost, hgb) {full:.4f}")
    rows = []
    for g, pred in GROUPS.items():
        drop = [c for c in cols if pred(c)]
        if not drop:
            continue
        ll = ens_ll([c for c in cols if c not in drop])
        rows.append((g, len(drop), ll - full))
        log.info(f"  drop {g:16s} ({len(drop):3d} features): change {ll - full:+.4f}")
    (MODEL_DIR / "feature_ablation.json").write_text(json.dumps(
        {"full": full, "groups": [{"group": g, "n": n, "change": d} for g, n, d in rows]}, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", type=int, default=0)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--tune-cv", type=int, default=0)
    ap.add_argument("--ablate-cv", action="store_true")
    a = ap.parse_args()
    if a.tune_cv:
        cvp = tune_cv(a.tune_cv)["best_params"]
        if a.ablate_cv:
            ablate_cv(cvp)
        raise SystemExit(0)
    params = None
    if a.tune:
        params = tune(a.tune)["best_params"]
    if a.ablate:
        if params is None and (MODEL_DIR / "catboost_tuning.json").exists():
            params = json.loads((MODEL_DIR / "catboost_tuning.json").read_text())["best_params"]
        ablate(params)
