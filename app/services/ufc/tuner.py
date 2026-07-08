"""
Optuna-based hyperparameter tuner for Glicko + GBT joint optimization.

Tunes Glicko parameters (K-factor, decay rates, sigma, etc.) and GBT
hyperparameters jointly using Bayesian optimization (TPE).

Usage:
    python -m app.services.tuner                    # 150 trials
    python -m app.services.tuner --n-trials 30      # quick search
    python -m app.services.tuner --glicko-only      # tune only Glicko params
    python -m app.services.tuner --gbt-only         # tune only GBT params
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from app.services.ufc.glicko_service import (
    DIMENSIONS,
    GlickoParams,
    load_glicko_data,
    run_glicko_inmemory_cached,
)

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models" / "ufc" / "h2h"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("tuner")


# ---------------------------------------------------------------------------
# DATA CACHING
# ---------------------------------------------------------------------------

class CachedData:
    """Holds pre-loaded data that doesn't change across trials."""

    def __init__(self):
        log.info("Loading and caching data (this happens once)...")

        # Load Glicko input data
        t0 = time.time()
        self.fight_map, self.rounds_by_fight, self.derived_totals, \
            self.fighter_info, self.baselines = load_glicko_data()
        log.info(f"  Glicko data loaded in {time.time() - t0:.1f}s")

        # Load model features (non-Glicko)
        t0 = time.time()
        from app.services.ufc.model import load_fight_data, build_features
        self.df, self.round_data = load_fight_data()
        self.df = build_features(self.df, self.round_data)
        log.info(f"  Model features built in {time.time() - t0:.1f}s")


def _inject_glicko_snapshots(df: pd.DataFrame, snapshots: dict) -> pd.DataFrame:
    """Replace glicko_* columns in df with new snapshot values."""
    for dim in DIMENSIONS:
        col = f"glicko_{dim}"
        df[col] = df.apply(
            lambda r, d=dim: snapshots.get(
                (r["fight_id"], r["stats_fighter_id"]), {}
            ).get(d, 0.0),
            axis=1,
        )
    return df


def _build_matchup_and_train(df: pd.DataFrame, gbt_params: dict,
                              top_n: int) -> dict:
    """Build matchup matrix, select features, train GBT, return metrics.

    This is a slimmed-down version of model.py's pipeline for fast iteration.
    """
    from app.services.ufc.model import build_matchup_df

    matchup, features = build_matchup_df(df)

    # Override feature count
    matchup_sorted = matchup.sort_values("date").reset_index(drop=True)

    # Feature selection (mutual information)
    has_aug = "_augmented" in matchup_sorted.columns
    train_mask = matchup_sorted.index < int(len(matchup_sorted) * 0.8)

    # Separate odds features
    odds_features = [f for f in features if "odds" in f or "elo_vs_odds" in f]
    non_odds_features = [f for f in features if f not in odds_features]

    # MI on non-odds features
    X_mi = matchup_sorted.loc[train_mask, non_odds_features].values
    y_mi = matchup_sorted.loc[train_mask, "red_wins"].values
    mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42, n_neighbors=5)
    mi_ranked = sorted(zip(non_odds_features, mi_scores), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in mi_ranked[:top_n]]
    selected += odds_features  # always include odds
    selected = list(dict.fromkeys(selected))

    # Train/test split
    split_idx = int(len(matchup_sorted) * 0.8)
    train = matchup_sorted.iloc[:split_idx]
    test_all = matchup_sorted.iloc[split_idx:]
    test = test_all[~test_all["_augmented"]] if has_aug else test_all

    X_train = train[selected].values
    y_train = train["red_wins"].values
    X_test = test[selected].values
    y_test = test["red_wins"].values

    model = HistGradientBoostingClassifier(
        max_iter=1000,
        max_depth=gbt_params["max_depth"],
        learning_rate=gbt_params["learning_rate"],
        max_features=gbt_params["max_features"],
        min_samples_leaf=gbt_params["min_samples_leaf"],
        l2_regularization=gbt_params["l2_regularization"],
        max_bins=128,
        early_stopping=True,
        n_iter_no_change=75,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    acc = np.mean((y_proba >= 0.5).astype(int) == y_test)

    return {"auc": auc, "accuracy": acc, "n_features": len(selected),
            "n_test": len(test)}


# ---------------------------------------------------------------------------
# OPTUNA OBJECTIVE
# ---------------------------------------------------------------------------

DEFAULT_GBT = {
    "max_depth": 4,
    "learning_rate": 0.02,
    "max_features": 0.7,
    "min_samples_leaf": 30,
    "l2_regularization": 2.0,
}


def create_objective(cached: CachedData, tune_glicko: bool = True,
                     tune_gbt: bool = True, n_trials_total: int = 150):
    """Create an Optuna objective function with cached data."""

    trial_times = []

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()

        # --- Sample Glicko params ---
        if tune_glicko:
            glicko_kwargs = {
                "k_base": trial.suggest_float("k_base", 20, 60),
                "recency_decay": trial.suggest_float("recency_decay", 0.1, 0.5),
                "sigma_init": trial.suggest_float("sigma_init", 250, 450),
                "tau": trial.suggest_float("tau", 100, 250),
                "sos_transfer_pct": trial.suggest_float("sos_transfer_pct", 0.02, 0.15),
                "loser_penalty_pct": trial.suggest_float("loser_penalty_pct", 0.01, 0.10),
                "num_passes": 2,  # speed: 2 passes during tuning
            }
            # Tier 2 params after trial 80
            if trial.number >= 80:
                glicko_kwargs["k_mult_ko"] = trial.suggest_float("k_mult_ko", 1.0, 3.0)
                glicko_kwargs["k_mult_td"] = trial.suggest_float("k_mult_td", 1.0, 2.5)
                glicko_kwargs["k_mult_dur"] = trial.suggest_float("k_mult_dur", 1.0, 2.5)
                glicko_kwargs["title_mult"] = trial.suggest_float("title_mult", 1.0, 2.0)
                glicko_kwargs["decision_ud"] = trial.suggest_float("decision_ud", 0.75, 1.0)
            params = GlickoParams(**glicko_kwargs)
        else:
            params = GlickoParams(num_passes=2)

        # --- Sample GBT params ---
        if tune_gbt:
            gbt_params = {
                "max_depth": trial.suggest_int("gbt_max_depth", 3, 6),
                "learning_rate": trial.suggest_float("gbt_lr", 0.01, 0.1, log=True),
                "max_features": trial.suggest_float("gbt_max_features", 0.5, 0.9),
                "min_samples_leaf": trial.suggest_int("gbt_min_leaf", 10, 50),
                "l2_regularization": trial.suggest_float("gbt_l2", 0.5, 5.0),
            }
            top_n = trial.suggest_int("gbt_top_n", 25, 50)
        else:
            gbt_params = DEFAULT_GBT.copy()
            top_n = 35

        # --- Run Glicko with sampled params ---
        snapshots = run_glicko_inmemory_cached(
            cached.fight_map, cached.rounds_by_fight,
            cached.fighter_info, cached.baselines, params
        )

        # --- Inject snapshots and train GBT ---
        df_copy = cached.df.copy()
        df_copy = _inject_glicko_snapshots(df_copy, snapshots)

        result = _build_matchup_and_train(df_copy, gbt_params, top_n)

        elapsed = time.time() - t0
        trial_times.append(elapsed)
        avg_time = sum(trial_times) / len(trial_times)
        remaining = (n_trials_total - trial.number - 1) * avg_time

        # --- Progress output ---
        try:
            best_val = trial.study.best_value
            star = " ★" if trial.study.best_trial.number == trial.number else ""
        except ValueError:
            best_val = result["auc"]
            star = " ★"
        print(f"Trial {trial.number + 1:>4d}/{n_trials_total} | "
              f"AUC: {result['auc']:.4f} | "
              f"Acc: {result['accuracy']:.4f} | "
              f"Best: {best_val:.4f}{star} | "
              f"{elapsed:.0f}s | "
              f"ETA: {remaining / 60:.0f}m",
              flush=True)

        # Top 5 every 10 trials
        if (trial.number + 1) % 10 == 0:
            print(f"\n{'─' * 80}")
            print(f"  Top 5 after {trial.number + 1} trials:")
            top_trials = sorted(trial.study.trials, key=lambda t: t.value or 0, reverse=True)[:5]
            for i, t in enumerate(top_trials):
                p = t.params
                glicko_str = ""
                if tune_glicko:
                    glicko_str = (f"k={p.get('k_base', 40):.0f} "
                                  f"decay={p.get('recency_decay', 0.3):.2f} "
                                  f"sigma={p.get('sigma_init', 350):.0f} "
                                  f"tau={p.get('tau', 180):.0f} "
                                  f"sos={p.get('sos_transfer_pct', 0.08):.3f}")
                gbt_str = ""
                if tune_gbt:
                    gbt_str = (f"depth={p.get('gbt_max_depth', 4)} "
                               f"lr={p.get('gbt_lr', 0.02):.3f} "
                               f"feat={p.get('gbt_max_features', 0.7):.2f}")
                print(f"  #{i+1}  AUC={t.value:.4f}  {glicko_str} {gbt_str}")
            print(f"{'─' * 80}\n", flush=True)

        return result["auc"]

    return objective


# ---------------------------------------------------------------------------
# RE-EVALUATE TOP RESULTS WITH 4 PASSES
# ---------------------------------------------------------------------------

def reevaluate_top_trials(study: optuna.Study, cached: CachedData,
                          tune_glicko: bool, tune_gbt: bool, top_k: int = 3):
    """Re-evaluate the top trials with num_passes=4 for final accuracy."""
    print(f"\n{'=' * 80}")
    print(f"RE-EVALUATING TOP {top_k} WITH 4 GLICKO PASSES")
    print(f"{'=' * 80}\n")

    top_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:top_k]
    results = []

    for i, trial in enumerate(top_trials):
        p = trial.params
        t0 = time.time()

        # Reconstruct params with 4 passes
        if tune_glicko:
            glicko_kwargs = {k: v for k, v in p.items() if not k.startswith("gbt_")}
            glicko_kwargs["num_passes"] = 4
            params = GlickoParams(**glicko_kwargs)
        else:
            params = GlickoParams(num_passes=4)

        if tune_gbt:
            gbt_params = {
                "max_depth": p.get("gbt_max_depth", 4),
                "learning_rate": p.get("gbt_lr", 0.02),
                "max_features": p.get("gbt_max_features", 0.7),
                "min_samples_leaf": p.get("gbt_min_leaf", 30),
                "l2_regularization": p.get("gbt_l2", 2.0),
            }
            top_n = p.get("gbt_top_n", 35)
        else:
            gbt_params = DEFAULT_GBT.copy()
            top_n = 35

        snapshots = run_glicko_inmemory_cached(
            cached.fight_map, cached.rounds_by_fight,
            cached.fighter_info, cached.baselines, params
        )

        df_copy = cached.df.copy()
        df_copy = _inject_glicko_snapshots(df_copy, snapshots)
        result = _build_matchup_and_train(df_copy, gbt_params, top_n)

        elapsed = time.time() - t0
        print(f"  #{i+1} (trial {trial.number}) | "
              f"2-pass AUC: {trial.value:.4f} → 4-pass AUC: {result['auc']:.4f} | "
              f"Acc: {result['accuracy']:.4f} | {elapsed:.0f}s")

        results.append({
            "trial_number": trial.number,
            "params": p,
            "auc_2pass": trial.value,
            "auc_4pass": result["auc"],
            "accuracy_4pass": result["accuracy"],
        })

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Glicko + GBT hyperparameter tuner")
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--glicko-only", action="store_true",
                        help="Tune only Glicko params (fix GBT)")
    parser.add_argument("--gbt-only", action="store_true",
                        help="Tune only GBT params (fix Glicko)")
    args = parser.parse_args()

    tune_glicko = not args.gbt_only
    tune_gbt = not args.glicko_only

    mode = "joint"
    if args.glicko_only:
        mode = "glicko-only"
    elif args.gbt_only:
        mode = "gbt-only"

    print(f"\n{'=' * 80}")
    print(f"  GLICKO + GBT HYPERPARAMETER TUNER")
    print(f"  Mode: {mode} | Trials: {args.n_trials}")
    print(f"  Glicko passes: 2 (tuning) → 4 (final evaluation)")
    print(f"{'=' * 80}\n")

    # Suppress noisy warnings during tuning
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load and cache data
    cached = CachedData()

    # Create Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"glicko_gbt_{mode}",
    )

    objective = create_objective(cached, tune_glicko, tune_gbt, args.n_trials)

    print(f"\nStarting {args.n_trials} trials...\n")
    t_start = time.time()
    study.optimize(objective, n_trials=args.n_trials)
    total_time = time.time() - t_start

    # Re-evaluate top 3 with 4 passes
    final_results = reevaluate_top_trials(study, cached, tune_glicko, tune_gbt)

    # Summary
    best = study.best_trial
    print(f"\n{'=' * 80}")
    print(f"  TUNING COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Best trial: #{best.number + 1}")
    print(f"  Best AUC (2-pass): {best.value:.4f}")
    if final_results:
        best_4pass = max(final_results, key=lambda r: r["auc_4pass"])
        print(f"  Best AUC (4-pass): {best_4pass['auc_4pass']:.4f}")
        print(f"  Best Accuracy (4-pass): {best_4pass['accuracy_4pass']:.4f}")
    print(f"\n  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save results
    output = {
        "mode": mode,
        "n_trials": args.n_trials,
        "total_time_minutes": round(total_time / 60, 1),
        "best_trial": best.number,
        "best_auc_2pass": best.value,
        "best_params": best.params,
        "final_evaluations": final_results,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    output_path = MODEL_DIR / "tuning_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # Suppress verbose loggers during tuning
    logging.getLogger("model").setLevel(logging.WARNING)
    logging.getLogger("glicko_service").setLevel(logging.WARNING)
    logging.getLogger("ranking_service").setLevel(logging.WARNING)
    main()
