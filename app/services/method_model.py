"""
UFC Method-of-Victory Prediction — GBT Multiclass Model (v1)

Predicts HOW a fight ends: KO/TKO, Submission, or Decision.
Independent of who wins — can be combined with the h2h model.

Usage:
    python -m app.services.method_model              # train + evaluate
    python -m app.services.method_model --predict     # generate predictions for all fights
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from app.database import SessionLocal
from app.services.model import (
    build_features,
    load_fight_data,
    _safe_divide,
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

METHOD_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "UFC" / "method"
METHOD_MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(METHOD_MODEL_DIR / "training.log", mode="w"),
    ],
)
log = logging.getLogger("method_model")

CLASS_NAMES = ["KO/TKO", "Submission", "Decision"]

METHOD_MAP = {
    "KO/TKO": 0,
    "TKO - Doctor's Stoppage": 0,
    "DQ": 0,
    "Submission": 1,
    "Decision - Unanimous": 2,
    "Decision - Split": 2,
    "Decision - Majority": 2,
    "Decision": 2,
}

WEIGHT_CLASS_ORDINAL = {
    "strawweight": 1, "flyweight": 2, "bantamweight": 3,
    "featherweight": 4, "lightweight": 5, "welterweight": 6,
    "middleweight": 7, "light_heavyweight": 8, "heavyweight": 9,
    "catchweight": 5, "unknown": 5,  # default to middleish
}


# ---------------------------------------------------------------------------
# METHOD-SPECIFIC FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def build_method_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add method-specific features on top of the base build_features() output."""
    log.info("Building method-specific features...")

    # --- Per-method win/loss flags (some already exist from build_features) ---
    # ko_win, sub_win, dec_win already computed in build_features
    # Add loss-by-method flags
    df["ko_loss"] = ((df["lost"] == 1) & df["method"].str.contains("KO", na=False)).astype(int)
    df["sub_loss"] = ((df["lost"] == 1) & df["method"].str.contains("Sub", case=False, na=False)).astype(int)
    df["dec_loss"] = ((df["lost"] == 1) & df["method"].str.contains("Dec", case=False, na=False)).astype(int)

    df = df.sort_values(["stats_fighter_id", "date"]).reset_index(drop=True)

    # --- Per-method career rates ---
    for col in ["ko_win", "sub_win", "dec_win"]:
        career_col = f"career_{col}s"
        df[career_col] = (
            df.groupby("stats_fighter_id")[col]
            .apply(lambda x: x.expanding().sum().shift(1).fillna(0))
            .reset_index(level=0, drop=True)
        )

    for col in ["ko_loss", "sub_loss", "dec_loss"]:
        career_col = f"career_{col}es"
        df[career_col] = (
            df.groupby("stats_fighter_id")[col]
            .apply(lambda x: x.expanding().sum().shift(1).fillna(0))
            .reset_index(level=0, drop=True)
        )

    # Method rates as proportion of wins/losses
    df["ko_rate"] = _safe_divide(df["career_ko_wins"].values, df["career_wins"].clip(lower=1).values)
    df["sub_rate"] = _safe_divide(df["career_sub_wins"].values, df["career_wins"].clip(lower=1).values)
    df["dec_rate"] = _safe_divide(df["career_dec_wins"].values, df["career_wins"].clip(lower=1).values)
    df["been_ko_rate"] = _safe_divide(df["career_ko_losses"].values, df["career_losses"].clip(lower=1).values)
    df["been_subbed_rate"] = _safe_divide(df["career_sub_losses"].values, df["career_losses"].clip(lower=1).values)

    # Rolling method rates (5 and 3 fight windows)
    for col in ["ko_win", "sub_win", "dec_win", "ko_loss", "sub_loss"]:
        for window, prefix in [(5, "recent"), (3, "last3")]:
            df[f"{prefix}_{col}"] = (
                df.groupby("stats_fighter_id")[col]
                .apply(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
                .reset_index(level=0, drop=True)
            )

    # --- Per-method Elo ---
    log.info("  Computing per-method Elo ratings...")
    ko_elo, sub_elo, dec_elo = {}, {}, {}
    ko_elo_at_fight, sub_elo_at_fight, dec_elo_at_fight = {}, {}, {}

    fights_chrono = (
        df[["fight_id", "date", "red_fighter_id", "blue_fighter_id", "winner_id", "method"]]
        .drop_duplicates("fight_id").sort_values("date")
    )
    for _, row in fights_chrono.iterrows():
        r_id, b_id = row["red_fighter_id"], row["blue_fighter_id"]
        fid = row["fight_id"]
        method = str(row.get("method", ""))
        mapped = METHOD_MAP.get(method)

        # Store pre-fight Elo for all three method types
        for elo_dict, at_dict in [(ko_elo, ko_elo_at_fight),
                                   (sub_elo, sub_elo_at_fight),
                                   (dec_elo, dec_elo_at_fight)]:
            at_dict[(fid, r_id)] = elo_dict.get(r_id, 1500.0)
            at_dict[(fid, b_id)] = elo_dict.get(b_id, 1500.0)

        # Only update the relevant method Elo
        if mapped is not None and row["winner_id"] is not None:
            elo_dict = {0: ko_elo, 1: sub_elo, 2: dec_elo}.get(mapped)
            if elo_dict is not None:
                r_e, b_e = elo_dict.get(r_id, 1500.0), elo_dict.get(b_id, 1500.0)
                exp_r = 1 / (1 + 10 ** ((b_e - r_e) / 400))
                actual_r = 1.0 if row["winner_id"] == r_id else 0.0
                K = 32
                elo_dict[r_id] = r_e + K * (actual_r - exp_r)
                elo_dict[b_id] = b_e + K * ((1 - actual_r) - (1 - exp_r))

    df["ko_elo"] = df.apply(lambda r: ko_elo_at_fight.get((r["fight_id"], r["stats_fighter_id"]), 1500.0), axis=1)
    df["sub_elo"] = df.apply(lambda r: sub_elo_at_fight.get((r["fight_id"], r["stats_fighter_id"]), 1500.0), axis=1)
    df["dec_elo"] = df.apply(lambda r: dec_elo_at_fight.get((r["fight_id"], r["stats_fighter_id"]), 1500.0), axis=1)

    # --- Weight class ordinal ---
    df["weight_class_ordinal"] = df["division"].map(WEIGHT_CLASS_ORDINAL).fillna(5).astype(float)

    # --- 5-round flag ---
    df["is_5_round"] = (df["max_fight_time_seconds"] >= 1500).astype(float)

    log.info(f"  Method features added. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# MATCHUP CONSTRUCTION FOR METHOD PREDICTION
# ---------------------------------------------------------------------------

def build_method_matchup(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build matchup DataFrame with method_class as target (3-class)."""
    log.info("Building method matchup matrix...")

    # --- Feature columns (same base as h2h + method-specific) ---
    feature_cols = [c for c in df.columns if c.startswith(("avg_", "recent_", "last3_"))]
    feature_cols += [
        "elo", "elo_expected", "resume_score",
        "height_inches", "weight_lbs", "reach_inches", "age",
        "stance_orthodox", "stance_southpaw", "stance_switch",
        "career_win_pct", "career_fights", "finish_rate", "been_finished_rate",
        "streak", "days_since_last", "style_matchup_adv",
    ]
    # Method-specific features
    feature_cols += [
        "ko_rate", "sub_rate", "dec_rate", "been_ko_rate", "been_subbed_rate",
        "ko_elo", "sub_elo", "dec_elo",
    ]
    # Composites
    feature_cols += [c for c in df.columns if "composite" in c and c.startswith(("avg_", "recent_", "last3_"))]
    # Round profiles
    feature_cols += [c for c in df.columns if c.startswith(("avg_r1_", "avg_late_", "avg_output_", "avg_ctrl_trend",
                                                             "recent_r1_", "recent_late_", "recent_output_", "recent_ctrl_trend"))]
    # Style & division dummies
    feature_cols += [c for c in df.columns if c.startswith("style_") and c not in feature_cols]
    feature_cols += [c for c in df.columns if c.startswith("div_") and c not in feature_cols]
    feature_cols = list(dict.fromkeys(feature_cols))

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Filter to features that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    red = df[df["corner"] == "red"].set_index("fight_id")
    blue = df[df["corner"] == "blue"].set_index("fight_id")
    common = red.index.intersection(blue.index)
    red, blue = red.loc[common], blue.loc[common]

    matchup = pd.DataFrame(index=common)
    matchup.index.name = "fight_id"
    matchup["date"] = red["date"].values
    matchup["method"] = red["method"].values

    # Map method to class
    matchup["method_class"] = matchup["method"].map(METHOD_MAP)
    n_before = len(matchup)
    matchup = matchup.dropna(subset=["method_class"])
    matchup["method_class"] = matchup["method_class"].astype(int)
    log.info(f"  Mapped methods: {len(matchup)} usable, dropped {n_before - len(matchup)} unmappable")

    # --- Difference features (who's better at what) ---
    for col in feature_cols:
        matchup[f"diff_{col}"] = red.loc[matchup.index, col].values - blue.loc[matchup.index, col].values

    # --- Combined/average features (both fighters together → method signal) ---
    combined_cols = [
        "avg_kd_per5", "avg_sig_str_landed_per5", "avg_sub_att_per5",
        "avg_ctrl_per5", "avg_ground_landed_per5", "avg_td_landed_per5",
        "avg_sig_str_acc", "avg_td_acc", "avg_sig_str_def", "avg_td_def",
        "ko_rate", "sub_rate", "dec_rate", "been_ko_rate", "been_subbed_rate",
        "ko_elo", "sub_elo", "dec_elo",
        "finish_rate", "been_finished_rate",
    ]
    # Add composites to combined
    for prefix in ["avg_", "recent_"]:
        for comp in ["striking_composite", "grappling_composite", "defense_composite",
                     "pressure_composite", "finishing_composite"]:
            col = f"{prefix}{comp}"
            if col in feature_cols:
                combined_cols.append(col)
    combined_cols = [c for c in combined_cols if c in feature_cols]

    for col in combined_cols:
        matchup[f"combined_{col}"] = (red.loc[matchup.index, col].values + blue.loc[matchup.index, col].values) / 2

    # --- Cross-tendency features (attacker's strength vs defender's weakness) ---
    matchup["ko_tendency"] = (
        (red.loc[matchup.index, "ko_rate"].values + blue.loc[matchup.index, "been_ko_rate"].values +
         blue.loc[matchup.index, "ko_rate"].values + red.loc[matchup.index, "been_ko_rate"].values) / 4
    )
    matchup["sub_tendency"] = (
        (red.loc[matchup.index, "sub_rate"].values + blue.loc[matchup.index, "been_subbed_rate"].values +
         blue.loc[matchup.index, "sub_rate"].values + red.loc[matchup.index, "been_subbed_rate"].values) / 4
    )
    matchup["dec_tendency"] = (
        (red.loc[matchup.index, "dec_rate"].values + blue.loc[matchup.index, "dec_rate"].values) / 2
    )

    # --- Raw values for key features (both sides) ---
    for col in ["elo", "elo_expected", "resume_score", "career_fights", "career_win_pct",
                "streak", "finish_rate", "ko_rate", "sub_rate", "dec_rate",
                "been_ko_rate", "been_subbed_rate", "ko_elo", "sub_elo", "dec_elo",
                "style_matchup_adv"]:
        if col in feature_cols:
            matchup[f"red_{col}"] = red.loc[matchup.index, col].values
            matchup[f"blue_{col}"] = blue.loc[matchup.index, col].values

    # --- Contextual features ---
    matchup["weight_class_ordinal"] = red.loc[matchup.index, "weight_class_ordinal"].values
    matchup["is_5_round"] = red.loc[matchup.index, "is_5_round"].values

    matchup = matchup.fillna(0)

    # --- Filter to modern era (2015+) ---
    modern_cutoff = _date(2015, 1, 1)
    matchup["date"] = pd.to_datetime(matchup["date"]).dt.date
    pre_modern = len(matchup[matchup["date"] < modern_cutoff])
    matchup = matchup[matchup["date"] >= modern_cutoff]
    log.info(f"  Filtered to modern era (2015+): {len(matchup)} fights (dropped {pre_modern} pre-2015)")

    # Log class distribution
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        n = (matchup["method_class"] == cls_id).sum()
        log.info(f"  {cls_name}: {n} ({n/len(matchup)*100:.1f}%)")

    # --- Feature selection ---
    feature_names = [c for c in matchup.columns
                     if c not in ("date", "method", "method_class")
                     and c.startswith(("diff_", "combined_", "red_", "blue_", "ko_tendency", "sub_tendency", "dec_tendency", "weight_class_ordinal", "is_5_round"))]
    # Also pick up standalone features
    feature_names += [c for c in ["weight_class_ordinal", "is_5_round",
                                   "ko_tendency", "sub_tendency", "dec_tendency"]
                      if c in matchup.columns and c not in feature_names]
    feature_names = list(dict.fromkeys(feature_names))

    log.info(f"  Running feature selection (mutual information, multiclass)...")
    X_all = matchup[feature_names].values
    y_all = matchup["method_class"].values
    mi_scores = mutual_info_classif(X_all, y_all, random_state=42)
    mi_ranked = sorted(zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True)

    log.info("  Top 30 features by mutual information:")
    for name, score in mi_ranked[:30]:
        log.info(f"    {name:55s} {score:.4f}")

    TOP_N = 55
    selected = [name for name, _ in mi_ranked[:TOP_N]]
    # Force-include domain-critical method features that MI may underrank
    force_include = [
        "ko_tendency", "sub_tendency", "dec_tendency",
        "weight_class_ordinal", "is_5_round",
        # Combined features — method is about BOTH fighters' tendencies together
        "combined_avg_ctrl_per5", "combined_avg_ground_landed_per5",
        "combined_avg_kd_per5", "combined_avg_td_landed_per5",
        "combined_avg_td_acc", "combined_avg_sig_str_def", "combined_avg_td_def",
        "combined_been_ko_rate", "combined_been_subbed_rate",
        "combined_ko_elo", "combined_sub_rate",
        "combined_been_finished_rate",
    ]
    for f in force_include:
        if f in matchup.columns and f not in selected:
            selected.append(f)

    log.info(f"  Selected {len(selected)} features (from {len(feature_names)} total)")
    log.info(f"  Matchup matrix: {matchup.shape[0]} fights")
    log.info(f"  Date range: {matchup['date'].min()} to {matchup['date'].max()}")

    return matchup, selected


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_method_gbt(matchup: pd.DataFrame, feature_names: list[str]) -> dict:
    """Train multiclass GBT for method prediction."""
    log.info("=" * 60)
    log.info("METHOD MODEL: Gradient Boosting (3-class)")
    log.info("=" * 60)

    matchup = matchup.sort_values("date").reset_index(drop=True)

    # Temporal split: 80% train, 20% test
    split_idx = int(len(matchup) * 0.8)
    train = matchup.iloc[:split_idx]
    test = matchup.iloc[split_idx:]

    X_train, y_train = train[feature_names].values, train["method_class"].values
    X_test, y_test = test[feature_names].values, test["method_class"].values

    log.info(f"  Train: {len(train)} ({train['date'].min()} to {train['date'].max()})")
    log.info(f"  Test:  {len(test)} ({test['date'].min()} to {test['date'].max()})")

    # Class distribution
    for split_name, y in [("Train", y_train), ("Test", y_test)]:
        dist = ", ".join(f"{CLASS_NAMES[i]}={np.mean(y == i)*100:.1f}%" for i in range(3))
        log.info(f"  {split_name} distribution: {dist}")

    # Mild sample weights — sqrt of inverse frequency to gently boost minority classes
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = np.sqrt(len(y_train) / (3 * class_counts))
    sample_weights = class_weights[y_train]
    log.info(f"  Class weights: {', '.join(f'{CLASS_NAMES[i]}={class_weights[i]:.2f}' for i in range(3))}")

    model = HistGradientBoostingClassifier(
        max_iter=2000,
        max_depth=4,
        learning_rate=0.01,
        max_features=0.8,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    logloss = log_loss(y_test, y_proba)
    baseline_acc = np.mean(y_test == 2)  # always predict Decision

    log.info(f"\n  --- Method GBT v1 Results ---")
    log.info(f"  Accuracy:     {acc:.4f} (baseline={baseline_acc:.4f}, lift={acc - baseline_acc:+.4f})")
    log.info(f"  Macro F1:     {macro_f1:.4f}")
    log.info(f"  Weighted F1:  {weighted_f1:.4f}")
    log.info(f"  Log Loss:     {logloss:.4f}")

    # Per-class metrics
    log.info(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    for line in report.split("\n"):
        log.info(f"  {line}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    log.info(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    log.info(f"  {'':>12s}  {'KO/TKO':>8s}  {'Submit':>8s}  {'Decision':>8s}")
    for i, name in enumerate(CLASS_NAMES):
        log.info(f"  {name:>12s}  {cm[i][0]:>8d}  {cm[i][1]:>8d}  {cm[i][2]:>8d}")

    # Feature importance
    importances = model.feature_importances_ if hasattr(model, "feature_importances_") else np.zeros(len(feature_names))
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    log.info(f"\n  Top 20 features by importance:")
    for name, imp in ranked[:20]:
        log.info(f"    {name:55s} {imp:.4f}")

    # Multiclass Brier score (mean of per-class Brier scores)
    brier_scores = []
    for i in range(3):
        y_bin = (y_test == i).astype(float)
        brier = np.mean((y_proba[:, i] - y_bin) ** 2)
        brier_scores.append(brier)
        log.info(f"  Brier score ({CLASS_NAMES[i]}): {brier:.4f}")
    log.info(f"  Mean Brier score: {np.mean(brier_scores):.4f}")

    # Save model
    model_path = METHOD_MODEL_DIR / "method_gbt_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": feature_names,
            "class_names": CLASS_NAMES,
        }, f)
    log.info(f"\n  Saved to {model_path}")

    return {
        "model": model,
        "features": feature_names,
        "X_test": X_test,
        "y_test": y_test,
        "y_proba": y_proba,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------------------

def calibrate_method_model(matchup: pd.DataFrame, feature_names: list[str]) -> dict:
    """Train a calibrated method model using temperature scaling."""
    log.info("=" * 60)
    log.info("METHOD MODEL: Calibration")
    log.info("=" * 60)

    matchup = matchup.sort_values("date").reset_index(drop=True)

    # 60/20/20 split: train / calibration / test
    n = len(matchup)
    train = matchup.iloc[:int(n * 0.6)]
    cal = matchup.iloc[int(n * 0.6):int(n * 0.8)]
    test = matchup.iloc[int(n * 0.8):]

    X_train, y_train = train[feature_names].values, train["method_class"].values
    X_cal, y_cal = cal[feature_names].values, cal["method_class"].values
    X_test, y_test = test[feature_names].values, test["method_class"].values

    log.info(f"  Train: {len(train)}, Cal: {len(cal)}, Test: {len(test)}")

    # Class weights
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = np.sqrt(len(y_train) / (3 * class_counts))
    sample_weights = class_weights[y_train]

    # Train base model
    model = HistGradientBoostingClassifier(
        max_iter=2000, max_depth=4, learning_rate=0.01,
        max_features=0.8, min_samples_leaf=40, l2_regularization=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Temperature scaling on calibration set
    raw_proba_cal = model.predict_proba(X_cal)
    raw_proba_test = model.predict_proba(X_test)

    # Find optimal temperature
    best_temp, best_loss = 1.0, float("inf")
    for temp in np.arange(0.5, 3.01, 0.05):
        scaled = np.exp(np.log(raw_proba_cal + 1e-10) / temp)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        loss = log_loss(y_cal, scaled)
        if loss < best_loss:
            best_temp, best_loss = temp, loss

    log.info(f"  Optimal temperature: {best_temp:.2f} (cal log_loss={best_loss:.4f})")

    # Apply calibration to test set
    scaled_test = np.exp(np.log(raw_proba_test + 1e-10) / best_temp)
    scaled_test = scaled_test / scaled_test.sum(axis=1, keepdims=True)

    # Evaluate calibrated predictions
    y_pred = np.argmax(scaled_test, axis=1)
    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, scaled_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    baseline_acc = np.mean(y_test == 2)

    log.info(f"\n  --- Calibrated Method Model Results ---")
    log.info(f"  Accuracy:     {acc:.4f} (baseline={baseline_acc:.4f})")
    log.info(f"  Macro F1:     {macro_f1:.4f}")
    log.info(f"  Log Loss:     {logloss:.4f}")

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    for line in report.split("\n"):
        log.info(f"  {line}")

    cm = confusion_matrix(y_test, y_pred)
    log.info(f"\n  Confusion Matrix:")
    log.info(f"  {'':>12s}  {'KO/TKO':>8s}  {'Submit':>8s}  {'Decision':>8s}")
    for i, name in enumerate(CLASS_NAMES):
        log.info(f"  {name:>12s}  {cm[i][0]:>8d}  {cm[i][1]:>8d}  {cm[i][2]:>8d}")

    # Save calibrated model
    cal_path = METHOD_MODEL_DIR / "method_calibrated_v1.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump({
            "model": model,
            "temperature": best_temp,
            "features": feature_names,
            "class_names": CLASS_NAMES,
        }, f)
    log.info(f"\n  Saved calibrated model to {cal_path}")

    return {
        "model": model,
        "temperature": best_temp,
        "features": feature_names,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# PREDICTION GENERATION
# ---------------------------------------------------------------------------

def generate_method_predictions():
    """Run calibrated method model on all fights and store in DB."""
    log.info("=" * 60)
    log.info("GENERATING METHOD PREDICTIONS FOR ALL FIGHTS")
    log.info("=" * 60)

    cal_path = METHOD_MODEL_DIR / "method_calibrated_v1.pkl"
    if not cal_path.exists():
        log.error("No calibrated method model found. Run training first.")
        return

    with open(cal_path, "rb") as f:
        cal = pickle.load(f)

    model = cal["model"]
    temperature = cal["temperature"]
    features = cal["features"]

    # Build features
    df, round_data = load_fight_data()
    df = build_features(df, round_data)
    df = build_method_features(df)

    # Build matchup (without filtering/selection — predict on everything)
    feature_cols = [c for c in df.columns if c.startswith(("avg_", "recent_", "last3_"))]
    feature_cols += [
        "elo", "elo_expected", "resume_score",
        "height_inches", "weight_lbs", "reach_inches", "age",
        "stance_orthodox", "stance_southpaw", "stance_switch",
        "career_win_pct", "career_fights", "finish_rate", "been_finished_rate",
        "streak", "days_since_last", "style_matchup_adv",
        "ko_rate", "sub_rate", "dec_rate", "been_ko_rate", "been_subbed_rate",
        "ko_elo", "sub_elo", "dec_elo",
    ]
    feature_cols += [c for c in df.columns if "composite" in c and c.startswith(("avg_", "recent_", "last3_"))]
    feature_cols += [c for c in df.columns if c.startswith(("avg_r1_", "avg_late_", "avg_output_", "avg_ctrl_trend",
                                                             "recent_r1_", "recent_late_", "recent_output_", "recent_ctrl_trend"))]
    feature_cols += [c for c in df.columns if c.startswith("style_") and c not in feature_cols]
    feature_cols += [c for c in df.columns if c.startswith("div_") and c not in feature_cols]
    feature_cols = list(dict.fromkeys(c for c in feature_cols if c in df.columns))

    for col in feature_cols:
        df[col] = df[col].fillna(df[col].mean())

    red = df[df["corner"] == "red"].set_index("fight_id")
    blue = df[df["corner"] == "blue"].set_index("fight_id")
    common = red.index.intersection(blue.index)
    red, blue = red.loc[common], blue.loc[common]

    matchup = pd.DataFrame(index=common)
    matchup.index.name = "fight_id"

    # Difference features
    for col in feature_cols:
        matchup[f"diff_{col}"] = red[col].values - blue[col].values

    # Combined features
    combined_cols = [
        "avg_kd_per5", "avg_sig_str_landed_per5", "avg_sub_att_per5",
        "avg_ctrl_per5", "avg_ground_landed_per5", "avg_td_landed_per5",
        "avg_sig_str_acc", "avg_td_acc", "avg_sig_str_def", "avg_td_def",
        "ko_rate", "sub_rate", "dec_rate", "been_ko_rate", "been_subbed_rate",
        "ko_elo", "sub_elo", "dec_elo",
        "finish_rate", "been_finished_rate",
    ]
    for prefix in ["avg_", "recent_"]:
        for comp in ["striking_composite", "grappling_composite", "defense_composite",
                     "pressure_composite", "finishing_composite"]:
            col = f"{prefix}{comp}"
            if col in feature_cols:
                combined_cols.append(col)
    combined_cols = [c for c in combined_cols if c in feature_cols]

    for col in combined_cols:
        matchup[f"combined_{col}"] = (red[col].values + blue[col].values) / 2

    # Tendency features
    matchup["ko_tendency"] = (
        (red["ko_rate"].values + blue["been_ko_rate"].values +
         blue["ko_rate"].values + red["been_ko_rate"].values) / 4
    )
    matchup["sub_tendency"] = (
        (red["sub_rate"].values + blue["been_subbed_rate"].values +
         blue["sub_rate"].values + red["been_subbed_rate"].values) / 4
    )
    matchup["dec_tendency"] = (red["dec_rate"].values + blue["dec_rate"].values) / 2

    # Raw values
    for col in ["elo", "elo_expected", "resume_score", "career_fights", "career_win_pct",
                "streak", "finish_rate", "ko_rate", "sub_rate", "dec_rate",
                "been_ko_rate", "been_subbed_rate", "ko_elo", "sub_elo", "dec_elo",
                "style_matchup_adv"]:
        if col in feature_cols:
            matchup[f"red_{col}"] = red[col].values
            matchup[f"blue_{col}"] = blue[col].values

    matchup["weight_class_ordinal"] = red["weight_class_ordinal"].values
    matchup["is_5_round"] = red["is_5_round"].values
    matchup = matchup.fillna(0)

    # Ensure all features exist
    for feat in features:
        if feat not in matchup.columns:
            matchup[feat] = 0.0

    # Predict
    X = matchup[features].values
    raw_proba = model.predict_proba(X)

    # Temperature-scale
    scaled = np.exp(np.log(raw_proba + 1e-10) / temperature)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)

    # Store in DB
    from app.models.ufc import UFCMethodPrediction
    db = SessionLocal()
    try:
        db.query(UFCMethodPrediction).delete()
        count = 0
        for fight_id, proba in zip(matchup.index, scaled):
            predicted_class = int(np.argmax(proba))
            db.add(UFCMethodPrediction(
                fight_id=int(fight_id),
                predicted_method=CLASS_NAMES[predicted_class],
                confidence=round(float(proba[predicted_class]), 4),
                ko_prob=round(float(proba[0]), 4),
                sub_prob=round(float(proba[1]), 4),
                dec_prob=round(float(proba[2]), 4),
            ))
            count += 1
        db.commit()
        log.info(f"Stored {count} method predictions")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run():
    """Full training pipeline: load data, build features, train, calibrate."""
    log.info("=" * 60)
    log.info("UFC METHOD-OF-VICTORY PREDICTION MODEL v1")
    log.info("=" * 60)

    # Load and build base features (reuse from h2h model)
    df, round_data = load_fight_data()
    df = build_features(df, round_data)

    # Add method-specific features
    df = build_method_features(df)

    # Build matchup
    matchup, features = build_method_matchup(df)

    # Train + evaluate
    results = train_method_gbt(matchup, features)

    # Calibrate
    cal_results = calibrate_method_model(matchup, features)

    log.info("\n" + "=" * 60)
    log.info("Method model pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true", help="Generate predictions for all fights")
    args = parser.parse_args()
    if args.predict:
        generate_method_predictions()
    else:
        run()
