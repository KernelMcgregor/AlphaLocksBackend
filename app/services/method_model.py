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

def _fillna_from_train(matchup: pd.DataFrame, feature_names: list[str], train_idx: int) -> pd.DataFrame:
    """Fill NaN values in feature columns using training-set means only."""
    train_means = matchup.iloc[:train_idx][feature_names].mean()
    matchup[feature_names] = matchup[feature_names].fillna(train_means).fillna(0)
    return matchup


def select_method_features(
    matchup: pd.DataFrame, feature_names: list[str], train_idx: int,
) -> list[str]:
    """Select features using MI on training data only (no leakage)."""
    train = matchup.iloc[:train_idx]
    X_train = train[feature_names].values
    y_train = train["method_class"].values

    log.info(f"  Running feature selection (MI on {len(train)} training rows)...")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_ranked = sorted(zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True)

    log.info("  Top 30 features by mutual information:")
    for name, score in mi_ranked[:30]:
        log.info(f"    {name:55s} {score:.4f}")

    TOP_N = 55
    selected = [name for name, _ in mi_ranked[:TOP_N]]
    force_include = [
        "sub_tendency", "dec_tendency",
        "weight_class_ordinal", "is_5_round",
        "combined_avg_ground_landed_per5",
        "combined_avg_kd_per5", "combined_avg_td_landed_per5",
        "combined_avg_td_acc", "combined_avg_sig_str_def", "combined_avg_td_def",
        "combined_been_subbed_rate",
        "combined_ko_elo", "combined_sub_rate",
        "combined_been_finished_rate",
        # Winner prediction features (symmetric only)
        "favorite_prob", "fav_sub_rate",
    ]
    for f in force_include:
        if f in matchup.columns and f not in selected:
            selected.append(f)

    log.info(f"  Selected {len(selected)} features (from {len(feature_names)} total)")
    return selected


def build_method_matchup(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build matchup DataFrame with method_class as target (3-class).

    Returns matchup DataFrame and the list of ALL candidate feature names.
    Feature selection (MI) should be done after the train/test split via
    select_method_features() to avoid information leakage.
    """
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

    # NOTE: fillna deferred to after train/test split to avoid leakage.
    # Training functions will compute means from training data only.

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

    # --- Winner prediction features (from h2h model) ---
    log.info("  Loading winner predictions from DB...")
    from app.models.ufc import UFCFightPrediction
    db = SessionLocal()
    winner_preds = {wp.fight_id: wp.red_prob for wp in db.query(UFCFightPrediction).all()}
    db.close()

    matchup["winner_red_prob"] = matchup.index.map(lambda fid: winner_preds.get(fid, np.nan))
    n_with_pred = matchup["winner_red_prob"].notna().sum()
    log.info(f"  Winner predictions found for {n_with_pred}/{len(matchup)} fights")

    # Derived winner features
    matchup["winner_blue_prob"] = 1 - matchup["winner_red_prob"]
    matchup["favorite_prob"] = np.maximum(matchup["winner_red_prob"].values,
                                          1 - matchup["winner_red_prob"].values)
    matchup["upset_potential"] = 1 - matchup["favorite_prob"]

    # Probability-weighted method tendencies (whose tendencies matter more = whoever is more likely to win)
    rp = matchup["winner_red_prob"].values
    bp = matchup["winner_blue_prob"].values
    matchup["fav_ko_rate"] = (
        red.loc[matchup.index, "ko_rate"].values * rp +
        blue.loc[matchup.index, "ko_rate"].values * bp
    )
    matchup["fav_sub_rate"] = (
        red.loc[matchup.index, "sub_rate"].values * rp +
        blue.loc[matchup.index, "sub_rate"].values * bp
    )
    matchup["fav_finish_rate"] = (
        red.loc[matchup.index, "finish_rate"].values * rp +
        blue.loc[matchup.index, "finish_rate"].values * bp
    )

    # NOTE: NaN filling deferred to training functions (use train-set means only)

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

    # --- Candidate feature names (selection happens later, after train/test split) ---
    feature_names = [c for c in matchup.columns
                     if c not in ("date", "method", "method_class")
                     and c.startswith(("diff_", "combined_", "red_", "blue_",
                                       "ko_tendency", "sub_tendency", "dec_tendency",
                                       "weight_class_ordinal", "is_5_round",
                                       "winner_", "favorite_", "upset_", "fav_"))]
    feature_names += [c for c in ["weight_class_ordinal", "is_5_round",
                                   "ko_tendency", "sub_tendency", "dec_tendency",
                                   "winner_red_prob", "favorite_prob", "upset_potential",
                                   "fav_ko_rate", "fav_sub_rate", "fav_finish_rate"]
                      if c in matchup.columns and c not in feature_names]
    feature_names = list(dict.fromkeys(feature_names))

    log.info(f"  Candidate features: {len(feature_names)}")
    log.info(f"  Matchup matrix: {matchup.shape[0]} fights")
    log.info(f"  Date range: {matchup['date'].min()} to {matchup['date'].max()}")

    return matchup, feature_names


# ---------------------------------------------------------------------------
# CORNER-SWAP AUGMENTATION
# ---------------------------------------------------------------------------

def _corner_swap_augment(X: np.ndarray, y: np.ndarray, feature_names: list[str],
                         sample_weights: np.ndarray | None = None,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Double training data by swapping red/blue corners.

    Method of victory is corner-invariant (KO is KO regardless of who's red),
    so we can create a mirror of every fight by swapping perspectives.
    - diff_* features: negate
    - red_* / blue_* features: swap
    - winner_red_prob: flip to 1-p, winner_blue_prob: flip to 1-p
    - favorite_prob, upset_potential, combined_*, *_tendency, weight_class_ordinal,
      is_5_round, fav_*: unchanged (symmetric by construction)
    """
    X_swap = X.copy()

    for i, name in enumerate(feature_names):
        if name.startswith("diff_"):
            X_swap[:, i] = -X[:, i]
        elif name.startswith("red_"):
            # Find corresponding blue_ feature
            blue_name = "blue_" + name[4:]
            if blue_name in feature_names:
                j = feature_names.index(blue_name)
                X_swap[:, i] = X[:, j]
                X_swap[:, j] = X[:, i]
        elif name == "winner_red_prob":
            X_swap[:, i] = 1 - X[:, i]
        elif name == "winner_blue_prob":
            X_swap[:, i] = 1 - X[:, i]

    X_aug = np.vstack([X, X_swap])
    y_aug = np.concatenate([y, y])
    sw_aug = np.concatenate([sample_weights, sample_weights]) if sample_weights is not None else None

    return X_aug, y_aug, sw_aug


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_method_gbt(matchup: pd.DataFrame, candidate_features: list[str]) -> dict:
    """Train multiclass GBT for method prediction."""
    log.info("=" * 60)
    log.info("METHOD MODEL: Gradient Boosting (3-class)")
    log.info("=" * 60)

    matchup = matchup.sort_values("date").reset_index(drop=True)

    # Temporal split: 80% train, 20% test
    split_idx = int(len(matchup) * 0.8)

    # Fill NaNs using training-set means only (no leakage)
    matchup = _fillna_from_train(matchup, candidate_features, split_idx)

    # Feature selection on training data only (no leakage)
    feature_names = select_method_features(matchup, candidate_features, split_idx)

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

    # Corner-swap augmentation (doubles training data)
    X_train, y_train, sample_weights = _corner_swap_augment(
        X_train, y_train, feature_names, sample_weights)
    log.info(f"  After corner-swap augmentation: {len(y_train)} training rows")

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
# HIERARCHICAL BINARY DECOMPOSITION
# ---------------------------------------------------------------------------

def _select_binary_features(
    X: np.ndarray, y: np.ndarray, feature_names: list[str],
    top_n: int = 40, force: list[str] | None = None, label: str = "",
) -> list[str]:
    """MI-based feature selection for a binary sub-task."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranked = sorted(zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True)
    log.info(f"  {label} Top 15 features by MI:")
    for name, score in mi_ranked[:15]:
        log.info(f"    {name:55s} {score:.4f}")
    selected = [name for name, _ in mi_ranked[:top_n]]
    if force:
        for f in force:
            if f in feature_names and f not in selected:
                selected.append(f)
    log.info(f"  {label} Selected {len(selected)} features")
    return selected


def _log_hierarchical_results(
    y_test_3: np.ndarray, y_proba: np.ndarray, label: str,
) -> dict:
    """Log full 3-class results for a hierarchical model variant."""
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_test_3, y_pred)
    macro_f1 = f1_score(y_test_3, y_pred, average="macro")
    weighted_f1 = f1_score(y_test_3, y_pred, average="weighted")
    logloss = log_loss(y_test_3, y_proba)
    baseline_acc = np.mean(y_test_3 == 2)

    log.info(f"\n  --- {label} ---")
    log.info(f"  Accuracy:     {acc:.4f} (baseline={baseline_acc:.4f}, lift={acc - baseline_acc:+.4f})")
    log.info(f"  Macro F1:     {macro_f1:.4f}")
    log.info(f"  Weighted F1:  {weighted_f1:.4f}")
    log.info(f"  Log Loss:     {logloss:.4f}")

    report = classification_report(y_test_3, y_pred, target_names=CLASS_NAMES, digits=4)
    for line in report.split("\n"):
        log.info(f"  {line}")

    cm = confusion_matrix(y_test_3, y_pred)
    log.info(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    log.info(f"  {'':>12s}  {'KO/TKO':>8s}  {'Submit':>8s}  {'Decision':>8s}")
    for i, name in enumerate(CLASS_NAMES):
        log.info(f"  {name:>12s}  {cm[i][0]:>8d}  {cm[i][1]:>8d}  {cm[i][2]:>8d}")

    brier_scores = []
    for i in range(3):
        y_bin = (y_test_3 == i).astype(float)
        brier = np.mean((y_proba[:, i] - y_bin) ** 2)
        brier_scores.append(brier)
        log.info(f"  Brier score ({CLASS_NAMES[i]}): {brier:.4f}")
    log.info(f"  Mean Brier score: {np.mean(brier_scores):.4f}")

    return {"accuracy": acc, "macro_f1": macro_f1, "log_loss": logloss,
            "mean_brier": np.mean(brier_scores)}


def train_method_hierarchical(matchup: pd.DataFrame, candidate_features: list[str]) -> dict:
    """Train hierarchical binary models with per-stage feature selection.

    Trains two decomposition strategies and compares:
    A) Finish/Decision → KO/Sub (conditional)
    B) Sub/Not-Sub → KO/Decision (independent sub detection)

    Each stage gets its own MI-based feature selection optimized for that
    binary task, rather than sharing features selected for the 3-class problem.
    """
    log.info("=" * 60)
    log.info("METHOD MODEL: Hierarchical Binary Decomposition v2")
    log.info("=" * 60)

    matchup = matchup.sort_values("date").reset_index(drop=True)
    split_idx = int(len(matchup) * 0.8)

    # Fill NaNs using training-set means only
    matchup = _fillna_from_train(matchup, candidate_features, split_idx)

    train = matchup.iloc[:split_idx]
    test = matchup.iloc[split_idx:]
    y_train_3 = train["method_class"].values
    y_test_3 = test["method_class"].values

    log.info(f"  Train: {len(train)} ({train['date'].min()} to {train['date'].max()})")
    log.info(f"  Test:  {len(test)} ({test['date'].min()} to {test['date'].max()})")

    # =====================================================================
    # APPROACH A: Finish/Decision → KO/Sub (with per-stage feature selection)
    # =====================================================================
    log.info("\n" + "=" * 60)
    log.info("  APPROACH A: Finish/Decision → KO/Sub")
    log.info("=" * 60)

    # ---- Stage 1: Finish vs Decision (own feature selection) ----
    log.info("\n  --- Stage 1: Finish vs Decision ---")
    y_train_s1 = (y_train_3 < 2).astype(int)  # 1 = finish, 0 = decision
    y_test_s1 = (y_test_3 < 2).astype(int)

    log.info(f"  Train: Finish={y_train_s1.sum()} ({y_train_s1.mean()*100:.1f}%), "
             f"Decision={len(y_train_s1) - y_train_s1.sum()} ({(1-y_train_s1.mean())*100:.1f}%)")

    # Feature selection for Finish vs Decision
    s1_features = _select_binary_features(
        train[candidate_features].values, y_train_s1, candidate_features,
        top_n=50,
        force=["weight_class_ordinal", "is_5_round", "dec_tendency",
               "combined_avg_kd_per5", "combined_been_finished_rate",
               "favorite_prob"],
        label="S1 (Finish/Dec)",
    )

    X_train_s1 = train[s1_features].values
    X_test_s1 = test[s1_features].values

    # Class weights & augmentation
    n_finish = y_train_s1.sum()
    n_dec = len(y_train_s1) - n_finish
    sw_s1 = np.where(y_train_s1 == 1,
                     np.sqrt(len(y_train_s1) / (2 * n_finish)),
                     np.sqrt(len(y_train_s1) / (2 * n_dec)))
    X_train_s1_aug, y_train_s1_aug, sw_s1_aug = _corner_swap_augment(
        X_train_s1, y_train_s1, s1_features, sw_s1)

    model_s1 = HistGradientBoostingClassifier(
        max_iter=2000, max_depth=4, learning_rate=0.01,
        max_features=0.8, min_samples_leaf=40, l2_regularization=2.0,
        random_state=42,
    )
    model_s1.fit(X_train_s1_aug, y_train_s1_aug, sample_weight=sw_s1_aug)

    s1_proba_test = model_s1.predict_proba(X_test_s1)
    p_finish_A = s1_proba_test[:, 1]
    p_decision_A = s1_proba_test[:, 0]

    s1_acc = accuracy_score(y_test_s1, (p_finish_A >= 0.5).astype(int))
    s1_baseline = max(y_test_s1.mean(), 1 - y_test_s1.mean())
    log.info(f"  Stage 1 Accuracy: {s1_acc:.4f} (baseline={s1_baseline:.4f}, lift={s1_acc-s1_baseline:+.4f})")
    log.info(f"  Stage 1 Log Loss: {log_loss(y_test_s1, s1_proba_test):.4f}")

    # ---- Stage 2: KO vs Sub — own feature selection on finishes only ----
    log.info("\n  --- Stage 2: KO vs Submission (finishes only, own features) ---")
    finish_mask_train = y_train_3 < 2
    finish_mask_test = y_test_3 < 2

    y_train_s2 = y_train_3[finish_mask_train]  # 0=KO, 1=Sub
    y_test_s2 = y_test_3[finish_mask_test]

    log.info(f"  Train finishes: {len(y_train_s2)} (KO={np.sum(y_train_s2==0)}, Sub={np.sum(y_train_s2==1)})")
    log.info(f"  Test finishes:  {len(y_test_s2)} (KO={np.sum(y_test_s2==0)}, Sub={np.sum(y_test_s2==1)})")

    # Feature selection specifically for KO vs Sub
    s2_features = _select_binary_features(
        train.loc[finish_mask_train, candidate_features].values,
        y_train_s2, candidate_features,
        top_n=40,
        force=["sub_tendency", "combined_avg_sub_att_per5",
               "combined_avg_ground_landed_per5", "combined_avg_td_landed_per5",
               "combined_avg_td_acc", "combined_been_subbed_rate",
               "combined_sub_rate", "combined_ko_elo",
               "weight_class_ordinal", "fav_sub_rate"],
        label="S2 (KO/Sub)",
    )

    X_train_s2 = train.loc[finish_mask_train, s2_features].values
    X_test_s2 = test.loc[finish_mask_test, s2_features].values
    X_test_s2_all = test[s2_features].values  # all test rows for composition

    # Class weights & augmentation
    n_ko = np.sum(y_train_s2 == 0)
    n_sub = np.sum(y_train_s2 == 1)
    sw_s2 = np.where(y_train_s2 == 0,
                     np.sqrt(len(y_train_s2) / (2 * n_ko)),
                     np.sqrt(len(y_train_s2) / (2 * n_sub)))
    X_train_s2_aug, y_train_s2_aug, sw_s2_aug = _corner_swap_augment(
        X_train_s2, y_train_s2, s2_features, sw_s2)

    model_s2 = HistGradientBoostingClassifier(
        max_iter=1500, max_depth=4, learning_rate=0.01,
        max_features=0.8, min_samples_leaf=30, l2_regularization=2.0,
        random_state=42,
    )
    model_s2.fit(X_train_s2_aug, y_train_s2_aug, sample_weight=sw_s2_aug)

    s2_proba_test = model_s2.predict_proba(X_test_s2)
    s2_acc = accuracy_score(y_test_s2, model_s2.predict(X_test_s2))
    s2_baseline = max(np.mean(y_test_s2 == 0), np.mean(y_test_s2 == 1))
    log.info(f"  Stage 2 Accuracy: {s2_acc:.4f} (baseline={s2_baseline:.4f}, lift={s2_acc-s2_baseline:+.4f})")
    log.info(f"  Stage 2 Log Loss: {log_loss(y_test_s2, s2_proba_test):.4f}")

    # Compose 3-class probabilities for Approach A
    s2_proba_all = model_s2.predict_proba(X_test_s2_all)
    y_proba_A = np.column_stack([
        p_finish_A * s2_proba_all[:, 0],  # KO
        p_finish_A * s2_proba_all[:, 1],  # Sub
        p_decision_A,                      # Decision
    ])
    y_proba_A = y_proba_A / y_proba_A.sum(axis=1, keepdims=True)

    metrics_A = _log_hierarchical_results(y_test_3, y_proba_A, "Approach A: Finish/Dec → KO/Sub")

    # =====================================================================
    # APPROACH B: Sub/Not-Sub → KO/Decision (independent sub detection)
    # =====================================================================
    log.info("\n" + "=" * 60)
    log.info("  APPROACH B: Sub/Not-Sub → KO/Decision")
    log.info("=" * 60)

    # ---- Stage 1B: Submission vs Not-Submission ----
    log.info("\n  --- Stage 1B: Submission vs Not-Submission ---")
    y_train_sub = (y_train_3 == 1).astype(int)  # 1 = sub, 0 = not sub
    y_test_sub = (y_test_3 == 1).astype(int)

    log.info(f"  Train: Sub={y_train_sub.sum()} ({y_train_sub.mean()*100:.1f}%), "
             f"Not-Sub={len(y_train_sub) - y_train_sub.sum()} ({(1-y_train_sub.mean())*100:.1f}%)")

    # Feature selection for Sub vs Not-Sub
    s1b_features = _select_binary_features(
        train[candidate_features].values, y_train_sub, candidate_features,
        top_n=45,
        force=["sub_tendency", "combined_avg_sub_att_per5",
               "combined_avg_ground_landed_per5", "combined_avg_td_landed_per5",
               "combined_avg_td_acc", "combined_been_subbed_rate",
               "combined_sub_rate", "weight_class_ordinal",
               "fav_sub_rate", "is_5_round"],
        label="S1B (Sub/NotSub)",
    )

    X_train_s1b = train[s1b_features].values
    X_test_s1b = test[s1b_features].values

    # Stronger class weights for sub detection (sub is only 18%)
    n_sub_all = y_train_sub.sum()
    n_notsub = len(y_train_sub) - n_sub_all
    sw_s1b = np.where(y_train_sub == 1,
                      np.sqrt(len(y_train_sub) / (2 * n_sub_all)),
                      np.sqrt(len(y_train_sub) / (2 * n_notsub)))
    X_train_s1b_aug, y_train_s1b_aug, sw_s1b_aug = _corner_swap_augment(
        X_train_s1b, y_train_sub, s1b_features, sw_s1b)

    model_sub = HistGradientBoostingClassifier(
        max_iter=2000, max_depth=4, learning_rate=0.01,
        max_features=0.8, min_samples_leaf=30, l2_regularization=1.5,
        random_state=42,
    )
    model_sub.fit(X_train_s1b_aug, y_train_s1b_aug, sample_weight=sw_s1b_aug)

    sub_proba_test = model_sub.predict_proba(X_test_s1b)
    p_sub_B = sub_proba_test[:, 1]  # P(Submission)

    sub_acc = accuracy_score(y_test_sub, (p_sub_B >= 0.5).astype(int))
    sub_baseline = max(y_test_sub.mean(), 1 - y_test_sub.mean())
    log.info(f"  Sub Detector Accuracy: {sub_acc:.4f} (baseline={sub_baseline:.4f}, lift={sub_acc-sub_baseline:+.4f})")
    log.info(f"  Sub Detector Log Loss: {log_loss(y_test_sub, sub_proba_test):.4f}")

    # ---- Stage 2B: KO vs Decision (among non-submissions) ----
    log.info("\n  --- Stage 2B: KO vs Decision (non-submissions only) ---")
    notsub_mask_train = y_train_3 != 1
    notsub_mask_test = y_test_3 != 1

    y_train_kd = (y_train_3[notsub_mask_train] == 0).astype(int)  # 1=KO, 0=Dec
    y_test_kd = (y_test_3[notsub_mask_test] == 0).astype(int)

    log.info(f"  Train non-sub: {len(y_train_kd)} (KO={y_train_kd.sum()}, Dec={len(y_train_kd)-y_train_kd.sum()})")
    log.info(f"  Test non-sub:  {len(y_test_kd)} (KO={y_test_kd.sum()}, Dec={len(y_test_kd)-y_test_kd.sum()})")

    # Feature selection for KO vs Decision
    s2b_features = _select_binary_features(
        train.loc[notsub_mask_train, candidate_features].values,
        y_train_kd, candidate_features,
        top_n=45,
        force=["dec_tendency", "weight_class_ordinal", "is_5_round",
               "combined_avg_kd_per5", "combined_been_ko_rate",
               "combined_been_finished_rate", "combined_ko_elo",
               "favorite_prob"],
        label="S2B (KO/Dec)",
    )

    X_train_s2b = train.loc[notsub_mask_train, s2b_features].values
    X_test_s2b = test.loc[notsub_mask_test, s2b_features].values
    X_test_s2b_all = test[s2b_features].values

    n_ko_b = y_train_kd.sum()
    n_dec_b = len(y_train_kd) - n_ko_b
    sw_s2b = np.where(y_train_kd == 1,
                      np.sqrt(len(y_train_kd) / (2 * n_ko_b)),
                      np.sqrt(len(y_train_kd) / (2 * n_dec_b)))
    X_train_s2b_aug, y_train_s2b_aug, sw_s2b_aug = _corner_swap_augment(
        X_train_s2b, y_train_kd, s2b_features, sw_s2b)

    model_kodec = HistGradientBoostingClassifier(
        max_iter=2000, max_depth=4, learning_rate=0.01,
        max_features=0.8, min_samples_leaf=40, l2_regularization=2.0,
        random_state=42,
    )
    model_kodec.fit(X_train_s2b_aug, y_train_s2b_aug, sample_weight=sw_s2b_aug)

    kd_proba_test = model_kodec.predict_proba(X_test_s2b)
    kd_acc = accuracy_score(y_test_kd, model_kodec.predict(X_test_s2b))
    kd_baseline = max(y_test_kd.mean(), 1 - y_test_kd.mean())
    log.info(f"  KO/Dec Accuracy: {kd_acc:.4f} (baseline={kd_baseline:.4f}, lift={kd_acc-kd_baseline:+.4f})")
    log.info(f"  KO/Dec Log Loss: {log_loss(y_test_kd, kd_proba_test):.4f}")

    # Compose 3-class probabilities for Approach B
    # P(Sub) from sub detector; remaining probability split between KO and Dec
    kd_proba_all = model_kodec.predict_proba(X_test_s2b_all)
    p_ko_given_notsub = kd_proba_all[:, 1]  # P(KO | not sub)
    p_dec_given_notsub = kd_proba_all[:, 0]  # P(Dec | not sub)

    y_proba_B = np.column_stack([
        (1 - p_sub_B) * p_ko_given_notsub,   # KO
        p_sub_B,                               # Submission
        (1 - p_sub_B) * p_dec_given_notsub,   # Decision
    ])
    y_proba_B = y_proba_B / y_proba_B.sum(axis=1, keepdims=True)

    metrics_B = _log_hierarchical_results(y_test_3, y_proba_B, "Approach B: Sub/NotSub → KO/Dec")

    # =====================================================================
    # APPROACH C: Ensemble — blend A's structure with B's sub signal
    # =====================================================================
    # Use a calibration split to optimize blend weights per-class.
    # Key idea: A is better at Finish/Decision, B is better at detecting subs.
    # Blend: take P(Sub) mostly from B, P(KO) and P(Dec) mostly from A.
    log.info("\n" + "=" * 60)
    log.info("  APPROACH C: Ensemble (A + B blend)")
    log.info("=" * 60)

    # Use last 30% of test as calibration for blend weights,
    # evaluate on first 70% (avoids re-using train data)
    cal_split = int(len(y_test_3) * 0.3)
    cal_idx = slice(0, cal_split)
    eval_idx = slice(cal_split, None)

    # Search for best per-class blend weight:
    #   P_ensemble(cls) = w_cls * P_A(cls) + (1 - w_cls) * P_B(cls)
    # Optimize log loss on calibration portion
    best_weights = [0.5, 0.5, 0.5]  # [KO, Sub, Dec]
    best_blend_loss = float("inf")

    log.info("  Searching blend weights (grid search on cal split)...")
    for w_ko in np.arange(0.0, 1.05, 0.1):
        for w_sub in np.arange(0.0, 1.05, 0.1):
            for w_dec in np.arange(0.0, 1.05, 0.1):
                blended = np.column_stack([
                    w_ko * y_proba_A[cal_idx, 0] + (1 - w_ko) * y_proba_B[cal_idx, 0],
                    w_sub * y_proba_A[cal_idx, 1] + (1 - w_sub) * y_proba_B[cal_idx, 1],
                    w_dec * y_proba_A[cal_idx, 2] + (1 - w_dec) * y_proba_B[cal_idx, 2],
                ])
                blended = blended / blended.sum(axis=1, keepdims=True)
                try:
                    ll = log_loss(y_test_3[cal_idx], blended)
                except ValueError:
                    continue
                if ll < best_blend_loss:
                    best_blend_loss = ll
                    best_weights = [w_ko, w_sub, w_dec]

    w_ko, w_sub, w_dec = best_weights
    log.info(f"  Best blend weights: KO={w_ko:.1f}, Sub={w_sub:.1f}, Dec={w_dec:.1f}")
    log.info(f"  Cal split log loss: {best_blend_loss:.4f}")

    # Apply blend to full test set
    y_proba_C = np.column_stack([
        w_ko * y_proba_A[:, 0] + (1 - w_ko) * y_proba_B[:, 0],
        w_sub * y_proba_A[:, 1] + (1 - w_sub) * y_proba_B[:, 1],
        w_dec * y_proba_A[:, 2] + (1 - w_dec) * y_proba_B[:, 2],
    ])
    y_proba_C = y_proba_C / y_proba_C.sum(axis=1, keepdims=True)

    metrics_C = _log_hierarchical_results(y_test_3, y_proba_C, "Approach C: Ensemble")

    # Also evaluate on eval-only portion (unseen by blend optimizer)
    metrics_C_eval = _log_hierarchical_results(
        y_test_3[eval_idx], y_proba_C[eval_idx],
        "Approach C: Ensemble (eval-only, unseen by optimizer)")

    # =====================================================================
    # COMPARISON: A vs B vs C
    # =====================================================================
    log.info("\n" + "=" * 60)
    log.info("  COMPARISON: A vs B vs C")
    log.info("=" * 60)
    for metric in ["accuracy", "macro_f1", "log_loss", "mean_brier"]:
        a_val = metrics_A[metric]
        b_val = metrics_B[metric]
        c_val = metrics_C[metric]
        vals = {"A": a_val, "B": b_val, "C": c_val}
        if metric in ("accuracy", "macro_f1"):
            best_label = max(vals, key=vals.get)
        else:
            best_label = min(vals, key=vals.get)
        log.info(f"  {metric:>12s}: A={a_val:.4f}  B={b_val:.4f}  C={c_val:.4f}  → {best_label}")

    # Save ensemble model (always save all components)
    model_path = METHOD_MODEL_DIR / "method_hierarchical_v1.pkl"
    save_data = {
        # Approach A components
        "stage1_model": model_s1, "stage2_model": model_s2,
        "stage1_features": s1_features, "stage2_features": s2_features,
        # Approach B components
        "sub_model": model_sub, "kodec_model": model_kodec,
        "sub_features": s1b_features, "kodec_features": s2b_features,
        # Ensemble config
        "blend_weights": {"ko": w_ko, "sub": w_sub, "dec": w_dec},
        "class_names": CLASS_NAMES,
        "type": "ensemble",
    }
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    log.info(f"\n  Saved ensemble model to {model_path}")

    return {
        "metrics_A": metrics_A,
        "metrics_B": metrics_B,
        "metrics_C": metrics_C,
        "blend_weights": best_weights,
        "y_test": y_test_3,
        "y_proba": y_proba_C,
        "accuracy": metrics_C["accuracy"],
        "macro_f1": metrics_C["macro_f1"],
    }


# ---------------------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------------------

def calibrate_method_model(matchup: pd.DataFrame, candidate_features: list[str]) -> dict:
    """Train a calibrated method model using temperature scaling."""
    log.info("=" * 60)
    log.info("METHOD MODEL: Calibration")
    log.info("=" * 60)

    matchup = matchup.sort_values("date").reset_index(drop=True)

    # 60/20/20 split: train / calibration / test
    n = len(matchup)
    train_end = int(n * 0.6)

    # Fill NaNs using training-set means only (no leakage)
    matchup = _fillna_from_train(matchup, candidate_features, train_end)

    # Feature selection on training data only (no leakage)
    feature_names = select_method_features(matchup, candidate_features, train_end)

    train = matchup.iloc[:train_end]
    cal = matchup.iloc[train_end:int(n * 0.8)]
    test = matchup.iloc[int(n * 0.8):]

    X_train, y_train = train[feature_names].values, train["method_class"].values
    X_cal, y_cal = cal[feature_names].values, cal["method_class"].values
    X_test, y_test = test[feature_names].values, test["method_class"].values

    log.info(f"  Train: {len(train)}, Cal: {len(cal)}, Test: {len(test)}")

    # Class weights
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = np.sqrt(len(y_train) / (3 * class_counts))
    sample_weights = class_weights[y_train]

    # Corner-swap augmentation (doubles training data)
    X_train, y_train, sample_weights = _corner_swap_augment(
        X_train, y_train, feature_names, sample_weights)
    log.info(f"  After corner-swap augmentation: {len(y_train)} training rows")

    # Train base model
    model = HistGradientBoostingClassifier(
        max_iter=2000, max_depth=4, learning_rate=0.01,
        max_features=0.8, min_samples_leaf=40, l2_regularization=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Dirichlet calibration: learns W (3x3) and b (3,) such that
    # calibrated_logits = W @ log(raw_p) + b, then softmax
    # More expressive than temperature scaling (which is W = (1/T)*I, b=0)
    from scipy.optimize import minimize

    raw_proba_cal = model.predict_proba(X_cal)
    raw_proba_test = model.predict_proba(X_test)

    log_proba_cal = np.log(raw_proba_cal + 1e-10)
    log_proba_test = np.log(raw_proba_test + 1e-10)

    n_classes = 3

    def _softmax(logits):
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _dirichlet_loss(params):
        W = params[:n_classes * n_classes].reshape(n_classes, n_classes)
        b = params[n_classes * n_classes:]
        calibrated_logits = log_proba_cal @ W.T + b
        proba = _softmax(calibrated_logits)
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        # Cross-entropy loss + L2 regularization to stay near identity
        ce = -np.mean(np.log(proba[np.arange(len(y_cal)), y_cal]))
        reg = 0.01 * np.sum((W - np.eye(n_classes)) ** 2)
        return ce + reg

    # Initialize near identity (equivalent to no calibration)
    W0 = np.eye(n_classes).flatten()
    b0 = np.zeros(n_classes)
    x0 = np.concatenate([W0, b0])

    result = minimize(_dirichlet_loss, x0, method="L-BFGS-B", options={"maxiter": 500})

    W_opt = result.x[:n_classes * n_classes].reshape(n_classes, n_classes)
    b_opt = result.x[n_classes * n_classes:]

    log.info(f"  Dirichlet calibration converged: {result.success} (loss={result.fun:.4f})")
    log.info(f"  W diagonal: [{W_opt[0,0]:.3f}, {W_opt[1,1]:.3f}, {W_opt[2,2]:.3f}]")
    log.info(f"  b: [{b_opt[0]:.3f}, {b_opt[1]:.3f}, {b_opt[2]:.3f}]")

    # Also fit temperature scaling for comparison
    best_temp, best_temp_loss = 1.0, float("inf")
    for temp in np.arange(0.5, 3.01, 0.05):
        scaled = np.exp(log_proba_cal / temp)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        loss = log_loss(y_cal, scaled)
        if loss < best_temp_loss:
            best_temp, best_temp_loss = temp, loss

    log.info(f"  Temperature scaling: T={best_temp:.2f} (cal loss={best_temp_loss:.4f})")
    log.info(f"  Dirichlet calibration: cal loss={result.fun:.4f}")

    # Apply Dirichlet calibration to test set
    calibrated_logits_test = log_proba_test @ W_opt.T + b_opt
    scaled_test = _softmax(calibrated_logits_test)

    # Evaluate calibrated predictions
    y_pred = np.argmax(scaled_test, axis=1)
    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, scaled_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    baseline_acc = np.mean(y_test == 2)

    log.info(f"\n  --- Calibrated Method Model Results (Dirichlet) ---")
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

    # Save calibrated model (new format with Dirichlet params)
    cal_path = METHOD_MODEL_DIR / "method_calibrated_v1.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump({
            "model": model,
            "calibration": "dirichlet",
            "W": W_opt,
            "b": b_opt,
            "temperature": best_temp,  # kept as fallback
            "features": feature_names,
            "class_names": CLASS_NAMES,
        }, f)
    log.info(f"\n  Saved calibrated model to {cal_path}")

    return {
        "model": model,
        "W": W_opt,
        "b": b_opt,
        "temperature": best_temp,
        "features": feature_names,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# PREDICTION GENERATION
# ---------------------------------------------------------------------------

def _build_prediction_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full matchup matrix for prediction (all fights, no filtering)."""
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

    for col in feature_cols:
        matchup[f"diff_{col}"] = red[col].values - blue[col].values

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

    matchup["ko_tendency"] = (
        (red["ko_rate"].values + blue["been_ko_rate"].values +
         blue["ko_rate"].values + red["been_ko_rate"].values) / 4
    )
    matchup["sub_tendency"] = (
        (red["sub_rate"].values + blue["been_subbed_rate"].values +
         blue["sub_rate"].values + red["been_subbed_rate"].values) / 4
    )
    matchup["dec_tendency"] = (red["dec_rate"].values + blue["dec_rate"].values) / 2

    for col in ["elo", "elo_expected", "resume_score", "career_fights", "career_win_pct",
                "streak", "finish_rate", "ko_rate", "sub_rate", "dec_rate",
                "been_ko_rate", "been_subbed_rate", "ko_elo", "sub_elo", "dec_elo",
                "style_matchup_adv"]:
        if col in feature_cols:
            matchup[f"red_{col}"] = red[col].values
            matchup[f"blue_{col}"] = blue[col].values

    matchup["weight_class_ordinal"] = red["weight_class_ordinal"].values
    matchup["is_5_round"] = red["is_5_round"].values

    from app.models.ufc import UFCFightPrediction
    db_wp = SessionLocal()
    winner_preds = {wp.fight_id: wp.red_prob for wp in db_wp.query(UFCFightPrediction).all()}
    db_wp.close()
    matchup["winner_red_prob"] = matchup.index.map(lambda fid: winner_preds.get(fid, 0.5))
    matchup["winner_blue_prob"] = 1 - matchup["winner_red_prob"]
    matchup["favorite_prob"] = np.maximum(matchup["winner_red_prob"].values,
                                          1 - matchup["winner_red_prob"].values)
    matchup["upset_potential"] = 1 - matchup["favorite_prob"]
    rp = matchup["winner_red_prob"].values
    bp = matchup["winner_blue_prob"].values
    matchup["fav_ko_rate"] = red["ko_rate"].values * rp + blue["ko_rate"].values * bp
    matchup["fav_sub_rate"] = red["sub_rate"].values * rp + blue["sub_rate"].values * bp
    matchup["fav_finish_rate"] = red["finish_rate"].values * rp + blue["finish_rate"].values * bp

    matchup = matchup.fillna(0)
    return matchup


def generate_method_predictions():
    """Run ensemble method model on all fights and store in DB."""
    log.info("=" * 60)
    log.info("GENERATING METHOD PREDICTIONS FOR ALL FIGHTS")
    log.info("=" * 60)

    model_path = METHOD_MODEL_DIR / "method_hierarchical_v1.pkl"
    if not model_path.exists():
        log.error("No ensemble method model found. Run --hierarchical first.")
        return

    with open(model_path, "rb") as f:
        ens = pickle.load(f)

    if ens.get("type") != "ensemble":
        log.error(f"Expected ensemble model, got type={ens.get('type')}. Re-run --hierarchical.")
        return

    # Build features
    df, round_data = load_fight_data()
    df = build_features(df, round_data)
    df = build_method_features(df)
    matchup = _build_prediction_matchup(df)

    # Ensure all feature columns exist for each sub-model
    all_feature_sets = [
        ens["stage1_features"], ens["stage2_features"],
        ens["sub_features"], ens["kodec_features"],
    ]
    for feat_list in all_feature_sets:
        for feat in feat_list:
            if feat not in matchup.columns:
                matchup[feat] = 0.0

    # --- Approach A predictions ---
    X_s1 = matchup[ens["stage1_features"]].values
    X_s2 = matchup[ens["stage2_features"]].values

    s1_proba = ens["stage1_model"].predict_proba(X_s1)
    p_finish = s1_proba[:, 1]
    p_decision_A = s1_proba[:, 0]

    s2_proba = ens["stage2_model"].predict_proba(X_s2)
    proba_A = np.column_stack([
        p_finish * s2_proba[:, 0],  # KO
        p_finish * s2_proba[:, 1],  # Sub
        p_decision_A,                # Decision
    ])
    proba_A = proba_A / proba_A.sum(axis=1, keepdims=True)

    # --- Approach B predictions ---
    X_sub = matchup[ens["sub_features"]].values
    X_kodec = matchup[ens["kodec_features"]].values

    sub_proba = ens["sub_model"].predict_proba(X_sub)
    p_sub_B = sub_proba[:, 1]

    kodec_proba = ens["kodec_model"].predict_proba(X_kodec)
    p_ko_given_notsub = kodec_proba[:, 1]
    p_dec_given_notsub = kodec_proba[:, 0]

    proba_B = np.column_stack([
        (1 - p_sub_B) * p_ko_given_notsub,  # KO
        p_sub_B,                              # Sub
        (1 - p_sub_B) * p_dec_given_notsub,  # Decision
    ])
    proba_B = proba_B / proba_B.sum(axis=1, keepdims=True)

    # --- Ensemble blend ---
    w = ens["blend_weights"]
    w_ko, w_sub, w_dec = w["ko"], w["sub"], w["dec"]

    scaled = np.column_stack([
        w_ko * proba_A[:, 0] + (1 - w_ko) * proba_B[:, 0],
        w_sub * proba_A[:, 1] + (1 - w_sub) * proba_B[:, 1],
        w_dec * proba_A[:, 2] + (1 - w_dec) * proba_B[:, 2],
    ])
    scaled = scaled / scaled.sum(axis=1, keepdims=True)

    log.info(f"  Ensemble blend weights: KO={w_ko:.1f}, Sub={w_sub:.1f}, Dec={w_dec:.1f}")
    log.info(f"  Generated predictions for {len(matchup)} fights")

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
        log.info(f"  Stored {count} method predictions")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# DIAGNOSTICS & EVALUATION
# ---------------------------------------------------------------------------

def evaluate_method_model():
    """Run comprehensive diagnostics on the method model.

    Outputs detailed analysis to training.log:
    - Per-class calibration (predicted prob vs actual frequency in decile bins)
    - Confidence vs accuracy stratification
    - Per-weight-class performance
    - 5-round vs 3-round analysis
    - Confusion pattern analysis
    - Permutation importance (test set)
    - Force-included feature audit
    """
    from sklearn.inspection import permutation_importance

    log.info("=" * 60)
    log.info("METHOD MODEL: COMPREHENSIVE DIAGNOSTICS")
    log.info("=" * 60)

    # Load data and build features
    df, round_data = load_fight_data()
    df = build_features(df, round_data)
    df = build_method_features(df)
    matchup, candidate_features = build_method_matchup(df)

    matchup = matchup.sort_values("date").reset_index(drop=True)
    split_idx = int(len(matchup) * 0.8)

    matchup = _fillna_from_train(matchup, candidate_features, split_idx)
    feature_names = select_method_features(matchup, candidate_features, split_idx)

    train = matchup.iloc[:split_idx]
    test = matchup.iloc[split_idx:]

    X_train, y_train = train[feature_names].values, train["method_class"].values
    X_test, y_test = test[feature_names].values, test["method_class"].values

    # Train model (same as train_method_gbt but we keep the model for diagnostics)
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = np.sqrt(len(y_train) / (3 * class_counts))
    sample_weights = class_weights[y_train]

    model = HistGradientBoostingClassifier(
        max_iter=2000, max_depth=4, learning_rate=0.01,
        max_features=0.8, min_samples_leaf=40, l2_regularization=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    baseline_acc = np.mean(y_test == 2)
    logloss = log_loss(y_test, y_proba)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    log.info(f"\n  --- Baseline Metrics ---")
    log.info(f"  Test size: {len(test)} fights ({test['date'].min()} to {test['date'].max()})")
    log.info(f"  Accuracy:     {acc:.4f} (baseline={baseline_acc:.4f}, lift={acc - baseline_acc:+.4f})")
    log.info(f"  Macro F1:     {macro_f1:.4f}")
    log.info(f"  Weighted F1:  {weighted_f1:.4f}")
    log.info(f"  Log Loss:     {logloss:.4f}")

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    for line in report.split("\n"):
        log.info(f"  {line}")

    cm = confusion_matrix(y_test, y_pred)
    log.info(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    log.info(f"  {'':>12s}  {'KO/TKO':>8s}  {'Submit':>8s}  {'Decision':>8s}")
    for i, name in enumerate(CLASS_NAMES):
        log.info(f"  {name:>12s}  {cm[i][0]:>8d}  {cm[i][1]:>8d}  {cm[i][2]:>8d}")

    # ---- 1. Per-class calibration curves ----
    log.info(f"\n  --- Per-Class Calibration (10 bins) ---")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        probs = y_proba[:, cls_id]
        actual = (y_test == cls_id).astype(float)
        # Bin into deciles
        bin_edges = np.linspace(0, 1, 11)
        for b in range(10):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (probs >= lo) & (probs < hi) if b < 9 else (probs >= lo) & (probs <= hi)
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            mean_pred = probs[mask].mean()
            mean_actual = actual[mask].mean()
            log.info(f"    {cls_name} [{lo:.1f}-{hi:.1f}]: n={n_bin:4d}, "
                     f"pred={mean_pred:.3f}, actual={mean_actual:.3f}, "
                     f"gap={mean_pred - mean_actual:+.3f}")

    # ---- 2. Confidence vs accuracy ----
    log.info(f"\n  --- Confidence vs Accuracy ---")
    max_conf = y_proba.max(axis=1)
    conf_buckets = [(0.33, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.01)]
    for lo, hi in conf_buckets:
        mask = (max_conf >= lo) & (max_conf < hi)
        n_bucket = mask.sum()
        if n_bucket == 0:
            continue
        bucket_acc = accuracy_score(y_test[mask], y_pred[mask])
        bucket_logloss = log_loss(y_test[mask], y_proba[mask], labels=[0, 1, 2])
        log.info(f"    Conf [{lo:.2f}-{hi:.2f}): n={n_bucket:4d}, "
                 f"accuracy={bucket_acc:.4f}, log_loss={bucket_logloss:.4f}")

    # ---- 3. Per-weight-class performance ----
    log.info(f"\n  --- Per-Weight-Class Performance ---")
    wc_col = "weight_class_ordinal"
    if wc_col in test.columns:
        wc_names = {1: "Straw", 2: "Fly", 3: "Bantam", 4: "Feather",
                    5: "Light", 6: "Welter", 7: "Middle", 8: "LHW", 9: "HW"}
        for wc_val in sorted(test[wc_col].unique()):
            mask = test[wc_col].values == wc_val
            n_wc = mask.sum()
            if n_wc < 10:
                continue
            wc_acc = accuracy_score(y_test[mask], y_pred[mask])
            wc_dist = ", ".join(f"{CLASS_NAMES[i]}={np.mean(y_test[mask] == i)*100:.0f}%"
                                for i in range(3))
            wc_name = wc_names.get(int(wc_val), f"WC{int(wc_val)}")
            log.info(f"    {wc_name:>8s}: n={n_wc:3d}, acc={wc_acc:.3f} (dist: {wc_dist})")

    # ---- 4. 5-round vs 3-round ----
    log.info(f"\n  --- 5-Round vs 3-Round ---")
    if "is_5_round" in test.columns:
        for rnd_val, rnd_name in [(1.0, "5-round"), (0.0, "3-round")]:
            mask = test["is_5_round"].values == rnd_val
            n_rnd = mask.sum()
            if n_rnd < 5:
                continue
            rnd_acc = accuracy_score(y_test[mask], y_pred[mask])
            rnd_dist = ", ".join(f"{CLASS_NAMES[i]}={np.mean(y_test[mask] == i)*100:.0f}%"
                                 for i in range(3))
            log.info(f"    {rnd_name}: n={n_rnd:3d}, acc={rnd_acc:.3f} (dist: {rnd_dist})")

    # ---- 5. Confusion pattern analysis ----
    log.info(f"\n  --- Top Confusion Patterns ---")
    total_errors = (y_test != y_pred).sum()
    for actual_cls in range(3):
        for pred_cls in range(3):
            if actual_cls == pred_cls:
                continue
            n_err = cm[actual_cls][pred_cls]
            pct = n_err / total_errors * 100 if total_errors else 0
            log.info(f"    Actual={CLASS_NAMES[actual_cls]:>10s} -> Pred={CLASS_NAMES[pred_cls]:>10s}: "
                     f"{n_err:4d} ({pct:.1f}% of errors)")

    # ---- 6. Permutation importance (test set) ----
    log.info(f"\n  --- Permutation Importance (top 20, test set) ---")
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring="accuracy",
    )
    perm_ranked = sorted(
        zip(feature_names, perm_result.importances_mean, perm_result.importances_std),
        key=lambda x: x[1], reverse=True,
    )
    for name, imp_mean, imp_std in perm_ranked[:20]:
        log.info(f"    {name:55s} {imp_mean:.4f} +/- {imp_std:.4f}")

    # ---- 7. Force-include feature audit ----
    log.info(f"\n  --- Force-Include Feature Audit ---")
    force_include = [
        "sub_tendency", "dec_tendency",
        "weight_class_ordinal", "is_5_round",
        "combined_avg_ground_landed_per5",
        "combined_avg_kd_per5", "combined_avg_td_landed_per5",
        "combined_avg_td_acc", "combined_avg_sig_str_def", "combined_avg_td_def",
        "combined_been_subbed_rate",
        "combined_ko_elo", "combined_sub_rate",
        "combined_been_finished_rate",
        # Winner prediction features (symmetric only)
        "favorite_prob", "fav_sub_rate",
    ]
    perm_dict = {name: (m, s) for name, m, s in perm_ranked}
    split_imp = dict(zip(feature_names,
                         model.feature_importances_ if hasattr(model, "feature_importances_")
                         else np.zeros(len(feature_names))))
    for f in force_include:
        if f in feature_names:
            pi_m, pi_s = perm_dict.get(f, (0, 0))
            si = split_imp.get(f, 0)
            log.info(f"    {f:55s} split_imp={si:.4f}  perm_imp={pi_m:.4f}+/-{pi_s:.4f}")
        else:
            log.info(f"    {f:55s} NOT IN SELECTED FEATURES")

    # ---- 8. Per-class Brier scores ----
    log.info(f"\n  --- Per-Class Brier Scores ---")
    brier_scores = []
    for i in range(3):
        y_bin = (y_test == i).astype(float)
        brier = np.mean((y_proba[:, i] - y_bin) ** 2)
        brier_scores.append(brier)
        log.info(f"    {CLASS_NAMES[i]:>12s}: {brier:.4f}")
    log.info(f"    {'Mean':>12s}: {np.mean(brier_scores):.4f}")

    log.info(f"\n  Diagnostics complete.")
    return {
        "accuracy": acc, "baseline": baseline_acc, "macro_f1": macro_f1,
        "log_loss": logloss, "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# TEMPORAL CROSS-VALIDATION
# ---------------------------------------------------------------------------

def temporal_cv(matchup: pd.DataFrame | None = None,
                candidate_features: list[str] | None = None,
                params: dict | None = None) -> dict:
    """Expanding-window temporal CV for reliable performance estimates.

    Folds:
      1: Train 2015-2018, test 2019
      2: Train 2015-2019, test 2020
      3: Train 2015-2020, test 2021
      4: Train 2015-2021, test 2022
      5: Train 2015-2022, test 2023+

    Returns dict with per-fold and aggregated metrics.
    """
    log.info("=" * 60)
    log.info("METHOD MODEL: TEMPORAL CROSS-VALIDATION")
    log.info("=" * 60)

    if matchup is None or candidate_features is None:
        df, round_data = load_fight_data()
        df = build_features(df, round_data)
        df = build_method_features(df)
        matchup, candidate_features = build_method_matchup(df)

    matchup = matchup.sort_values("date").reset_index(drop=True)

    if params is None:
        params = dict(
            max_iter=2000, max_depth=4, learning_rate=0.01,
            max_features=0.8, min_samples_leaf=40, l2_regularization=2.0,
        )

    fold_boundaries = [
        (_date(2019, 1, 1), _date(2020, 1, 1)),
        (_date(2020, 1, 1), _date(2021, 1, 1)),
        (_date(2021, 1, 1), _date(2022, 1, 1)),
        (_date(2022, 1, 1), _date(2023, 1, 1)),
        (_date(2023, 1, 1), _date(2099, 1, 1)),
    ]

    fold_results = []
    for fold_i, (test_start, test_end) in enumerate(fold_boundaries, 1):
        train_mask = matchup["date"] < test_start
        test_mask = (matchup["date"] >= test_start) & (matchup["date"] < test_end)

        if test_mask.sum() < 10:
            log.info(f"  Fold {fold_i}: skipped (only {test_mask.sum()} test fights)")
            continue

        train_idx = train_mask.sum()

        # Fill NaNs from training data
        fold_matchup = matchup.copy()
        fold_matchup = _fillna_from_train(fold_matchup, candidate_features, train_idx)

        # Feature selection on this fold's training data
        feature_names = select_method_features(fold_matchup, candidate_features, train_idx)

        X_train = fold_matchup.loc[train_mask, feature_names].values
        y_train = fold_matchup.loc[train_mask, "method_class"].values
        X_test = fold_matchup.loc[test_mask, feature_names].values
        y_test = fold_matchup.loc[test_mask, "method_class"].values

        # Class weights
        class_counts = np.bincount(y_train, minlength=3)
        class_weights = np.sqrt(len(y_train) / (3 * class_counts))
        sample_weights = class_weights[y_train]

        # Corner-swap augmentation
        X_train, y_train, sample_weights = _corner_swap_augment(
            X_train, y_train, feature_names, sample_weights)

        model = HistGradientBoostingClassifier(random_state=42, **params)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_proba)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        baseline = np.mean(y_test == np.bincount(y_test).argmax())

        fold_results.append({
            "fold": fold_i,
            "test_start": test_start,
            "test_end": test_end,
            "n_train": train_mask.sum(),
            "n_test": test_mask.sum(),
            "accuracy": acc,
            "baseline": baseline,
            "log_loss": logloss,
            "macro_f1": macro_f1,
        })

        log.info(f"  Fold {fold_i} ({test_start} to {test_end}): "
                 f"n_train={train_mask.sum()}, n_test={test_mask.sum()}, "
                 f"acc={acc:.4f} (base={baseline:.4f}), "
                 f"F1={macro_f1:.4f}, logloss={logloss:.4f}")

    # Aggregate
    if fold_results:
        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["macro_f1"] for r in fold_results]
        lls = [r["log_loss"] for r in fold_results]
        log.info(f"\n  --- Temporal CV Summary ---")
        log.info(f"  Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        log.info(f"  Macro F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        log.info(f"  Log Loss:  {np.mean(lls):.4f} +/- {np.std(lls):.4f}")

    return {"folds": fold_results, "mean_accuracy": np.mean(accs) if fold_results else 0}


# ---------------------------------------------------------------------------
# HYPERPARAMETER TUNING (OPTUNA)
# ---------------------------------------------------------------------------

def tune_hyperparameters(n_trials: int = 50):
    """Tune GBT hyperparameters using Optuna with temporal CV as objective."""
    try:
        import optuna
    except ImportError:
        log.error("Optuna not installed. Run: pip install optuna")
        return

    log.info("=" * 60)
    log.info("METHOD MODEL: OPTUNA HYPERPARAMETER TUNING")
    log.info("=" * 60)

    # Load data once
    df, round_data = load_fight_data()
    df = build_features(df, round_data)
    df = build_method_features(df)
    matchup, candidate_features = build_method_matchup(df)

    def objective(trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 500, 3000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 10.0, log=True),
        }

        result = temporal_cv(matchup.copy(), candidate_features, params)
        if not result["folds"]:
            return float("inf")

        # Optimize log loss (lower is better)
        mean_logloss = np.mean([f["log_loss"] for f in result["folds"]])
        return mean_logloss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(f"\n  --- Best Trial ---")
    log.info(f"  Log Loss: {study.best_value:.4f}")
    log.info(f"  Params:")
    for key, value in study.best_params.items():
        log.info(f"    {key}: {value}")

    # Run final temporal CV with best params for full reporting
    log.info(f"\n  --- Final CV with best params ---")
    temporal_cv(matchup.copy(), candidate_features, study.best_params)

    return study


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

    # Train flat multiclass
    results = train_method_gbt(matchup, features)

    # Train hierarchical (Finish/Decision → KO/Sub)
    hier_results = train_method_hierarchical(matchup, features)

    # Calibrate
    cal_results = calibrate_method_model(matchup, features)

    log.info("\n" + "=" * 60)
    log.info("Method model pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true", help="Generate predictions for all fights")
    parser.add_argument("--evaluate", action="store_true", help="Run comprehensive diagnostics")
    parser.add_argument("--cv", action="store_true", help="Run temporal cross-validation")
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters with Optuna")
    parser.add_argument("--tune-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--hierarchical", action="store_true", help="Train hierarchical model only")
    args = parser.parse_args()
    if args.predict:
        generate_method_predictions()
    elif args.evaluate:
        evaluate_method_model()
    elif args.cv:
        temporal_cv()
    elif args.tune:
        tune_hyperparameters(n_trials=args.tune_trials)
    elif args.hierarchical:
        df, round_data = load_fight_data()
        df = build_features(df, round_data)
        df = build_method_features(df)
        matchup, features = build_method_matchup(df)
        train_method_hierarchical(matchup, features)
    else:
        run()
