"""
Ranking Validation — Walk-forward backtesting for the ranking system.

Measures prediction accuracy of composite ratings via chronological holdout
splits. Computes accuracy, log-loss, Brier score, and AUC-ROC.

Usage:
    python -m app.services.ranking_service --validate
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date

log = logging.getLogger("ranking_validation")


def _sigmoid(z: float) -> float:
    """Logistic sigmoid with stability clipping."""
    z = max(min(z, 500), -500)
    return 1.0 / (1.0 + math.exp(-z / 100.0))


def _log_loss_single(pred: float, outcome: float) -> float:
    p = max(min(pred, 1 - 1e-10), 1e-10)
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))


def _fit_weights(X, y, dimensions, defaults, lr=0.0001, epochs=200):
    """Fit logistic regression weights. Returns (weights_dict, final_loss)."""
    n_dims = len(dimensions)
    weights = [defaults.get(d, 1.0) for d in dimensions]

    for epoch in range(epochs):
        grad = [0.0] * n_dims
        total_loss = 0.0

        for features, outcome in zip(X, y):
            z = sum(w * f for w, f in zip(weights, features))
            pred = _sigmoid(z)
            error = pred - outcome
            total_loss += _log_loss_single(pred, outcome)

            for i in range(n_dims):
                grad[i] += error * features[i]

        for i in range(n_dims):
            grad[i] = grad[i] / len(X) + 0.01 * (weights[i] - defaults.get(dimensions[i], 1.0))
            weights[i] -= lr * grad[i]
            weights[i] = max(weights[i], 0.1)

    avg_loss = total_loss / len(X) if X else 0.0
    max_w = max(weights) if weights else 1.0
    if max_w > 0:
        scale = 3.0 / max_w
        weights = [w * scale for w in weights]

    return {d: round(w, 3) for d, w in zip(dimensions, weights)}, avg_loss


def _build_training_data(ratings, fight_map, fight_ids, fighter_round_count,
                         dimensions, min_rounds=10):
    """Build (X, y) feature matrix from fights."""
    X, y = [], []
    for fight_id in fight_ids:
        fight = fight_map[fight_id]
        if fight["weight_class"] == "unknown" or not fight["winner_id"]:
            continue

        red_id = fight["red_id"]
        blue_id = fight["blue_id"]

        if red_id not in ratings or blue_id not in ratings:
            continue
        if (fighter_round_count.get(red_id, 0) < min_rounds or
                fighter_round_count.get(blue_id, 0) < min_rounds):
            continue

        diffs = [ratings[red_id][d][0] - ratings[blue_id][d][0] for d in dimensions]
        outcome = 1.0 if fight["winner_id"] == red_id else 0.0
        X.append(diffs)
        y.append(outcome)

    return X, y


def _evaluate_predictions(weights, X, y, dimensions):
    """Compute accuracy, log-loss, Brier score, AUC-ROC on test data."""
    if not X:
        return {"n": 0, "accuracy": 0.0, "log_loss": 0.0, "brier": 0.0, "auc": 0.0}

    w_list = [weights.get(d, 1.0) for d in dimensions]
    correct = 0
    total_loss = 0.0
    total_brier = 0.0
    preds_and_labels = []

    for features, outcome in zip(X, y):
        z = sum(w * f for w, f in zip(w_list, features))
        pred = _sigmoid(z)

        predicted_winner = 1.0 if pred >= 0.5 else 0.0
        if predicted_winner == outcome:
            correct += 1

        total_loss += _log_loss_single(pred, outcome)
        total_brier += (pred - outcome) ** 2
        preds_and_labels.append((pred, outcome))

    n = len(X)
    accuracy = correct / n
    avg_loss = total_loss / n
    avg_brier = total_brier / n

    # AUC-ROC via Mann-Whitney U statistic
    positives = [p for p, l in preds_and_labels if l == 1.0]
    negatives = [p for p, l in preds_and_labels if l == 0.0]
    auc = 0.5
    if positives and negatives:
        concordant = sum(1 for pos in positives for neg in negatives if pos > neg)
        ties = sum(1 for pos in positives for neg in negatives if pos == neg)
        auc = (concordant + 0.5 * ties) / (len(positives) * len(negatives))

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "log_loss": round(avg_loss, 4),
        "brier": round(avg_brier, 4),
        "auc": round(auc, 4),
    }


def walk_forward_validate(ratings, fight_map, sorted_fight_ids, fighter_round_count,
                          dimensions, defaults,
                          year_boundaries=None):
    """
    Walk-forward validation: for each year boundary, train on all fights before
    that year and test on fights in that year. Returns per-fold and average metrics.
    """
    if year_boundaries is None:
        year_boundaries = [2021, 2022, 2023, 2024, 2025]

    results = []

    for year in year_boundaries:
        cutoff = date(year, 1, 1)

        train_ids = [fid for fid in sorted_fight_ids
                     if fight_map[fid].get("date") and fight_map[fid]["date"] < cutoff]
        test_ids = [fid for fid in sorted_fight_ids
                    if fight_map[fid].get("date") and cutoff <= fight_map[fid]["date"] < date(year + 1, 1, 1)]

        X_train, y_train = _build_training_data(
            ratings, fight_map, train_ids, fighter_round_count, dimensions)
        X_test, y_test = _build_training_data(
            ratings, fight_map, test_ids, fighter_round_count, dimensions)

        if len(X_train) < 100 or len(X_test) < 20:
            log.info(f"    {year}: skipped (train={len(X_train)}, test={len(X_test)})")
            continue

        weights, train_loss = _fit_weights(X_train, y_train, dimensions, defaults)
        metrics = _evaluate_predictions(weights, X_test, y_test, dimensions)
        metrics["year"] = year
        metrics["train_n"] = len(X_train)
        metrics["train_loss"] = round(train_loss, 4)
        results.append(metrics)

        log.info(f"    {year}: n={metrics['n']}, "
                 f"acc={metrics['accuracy']:.3f}, "
                 f"loss={metrics['log_loss']:.4f}, "
                 f"brier={metrics['brier']:.4f}, "
                 f"auc={metrics['auc']:.3f}")

    if not results:
        log.warning("  No valid folds for validation")
        return {"folds": [], "avg": {}}

    # Average across folds
    avg = {}
    for key in ["accuracy", "log_loss", "brier", "auc"]:
        avg[key] = round(sum(r[key] for r in results) / len(results), 4)
    avg["total_test_n"] = sum(r["n"] for r in results)
    avg["n_folds"] = len(results)

    log.info(f"  AVERAGE: acc={avg['accuracy']:.3f}, "
             f"loss={avg['log_loss']:.4f}, "
             f"brier={avg['brier']:.4f}, "
             f"auc={avg['auc']:.3f} "
             f"({avg['n_folds']} folds, {avg['total_test_n']} test fights)")

    return {"folds": results, "avg": avg}
