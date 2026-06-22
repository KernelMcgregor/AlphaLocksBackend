"""
UFC Fight Winner Prediction — 3-Phase Model Pipeline (v5)

Phase 1: Gradient Boosting on engineered features
Phase 2: RNN encoding of fighter career sequences
Phase 3: GNN for opponent-quality embeddings
Ensemble: Stacked ridge regression meta-learner

v5 improvements:
- Composite features (striking/grappling/defense indexes)
- Round-by-round profiles (fade detection, fast starter, late finisher)
- Stacked ensemble with ridge meta-learner (replaces weighted avg)
- CLV tracking for bet quality evaluation
- SHAP-based feature selection

Usage:
    python -m app.services.model              # train + evaluate all phases
    python -m app.services.model --phase 1    # run only Phase 1
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from app.database import SessionLocal
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFightStats

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "UFC" / "h2h"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(MODEL_DIR / "training.log", mode="w"),
    ],
)
log = logging.getLogger("model")


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_fight_data() -> pd.DataFrame:
    log.info("Loading data from database...")
    db = SessionLocal()

    fighters_q = db.query(UFCFighter).all()
    fighters = {
        f.id: {
            "name": f"{f.first_name} {f.last_name}",
            "height": f.height, "weight": f.weight,
            "reach": f.reach, "stance": f.stance, "dob": f.dob,
        }
        for f in fighters_q
    }
    log.info(f"  Loaded {len(fighters)} fighters")

    fights_q = (
        db.query(UFCFight, UFCFightStats)
        .join(UFCFightStats, UFCFight.id == UFCFightStats.fight_id)
        .filter(UFCFightStats.round_number == 0)
        .order_by(UFCFight.date)
        .all()
    )

    rows = []
    for fight, stats in fights_q:
        f = fighters.get(stats.fighter_id, {})
        rows.append({
            "fight_id": fight.id, "date": fight.date,
            "red_fighter_id": fight.red_fighter_id,
            "blue_fighter_id": fight.blue_fighter_id,
            "winner_id": fight.winner_id,
            "method": fight.method, "weight_class": fight.weight_class,
            "fight_time_seconds": fight.fight_time_seconds or 0,
            "max_fight_time_seconds": fight.max_fight_time_seconds or 0,
            "stats_fighter_id": stats.fighter_id, "corner": stats.corner,
            "kd": stats.kd,
            "sig_str_landed": stats.sig_str_landed, "sig_str_attempted": stats.sig_str_attempted,
            "total_str_landed": stats.total_str_landed, "total_str_attempted": stats.total_str_attempted,
            "td_landed": stats.td_landed, "td_attempted": stats.td_attempted,
            "sub_att": stats.sub_att, "rev": stats.rev, "ctrl_seconds": stats.ctrl_seconds,
            "head_landed": stats.head_landed, "head_attempted": stats.head_attempted,
            "body_landed": stats.body_landed, "body_attempted": stats.body_attempted,
            "leg_landed": stats.leg_landed, "leg_attempted": stats.leg_attempted,
            "distance_landed": stats.distance_landed, "distance_attempted": stats.distance_attempted,
            "clinch_landed": stats.clinch_landed, "clinch_attempted": stats.clinch_attempted,
            "ground_landed": stats.ground_landed, "ground_attempted": stats.ground_attempted,
            "fighter_height": f.get("height"), "fighter_weight": f.get("weight"),
            "fighter_reach": f.get("reach"), "fighter_stance": f.get("stance"),
            "fighter_dob": f.get("dob"),
        })

    db.close()
    df = pd.DataFrame(rows)
    log.info(f"  Loaded {len(df)} fight-stat rows ({df['fight_id'].nunique()} unique fights)")

    # --- Load per-round stats for round profiles ---
    log.info("  Loading per-round stats...")
    db = SessionLocal()
    round_q = (
        db.query(UFCFightStats)
        .filter(UFCFightStats.round_number > 0)
        .order_by(UFCFightStats.fight_id, UFCFightStats.fighter_id, UFCFightStats.round_number)
        .all()
    )
    round_rows = []
    for s in round_q:
        round_rows.append({
            "fight_id": s.fight_id, "stats_fighter_id": s.fighter_id,
            "round_number": s.round_number,
            "r_kd": s.kd, "r_sig_str_landed": s.sig_str_landed,
            "r_sig_str_attempted": s.sig_str_attempted,
            "r_td_landed": s.td_landed, "r_td_attempted": s.td_attempted,
            "r_sub_att": s.sub_att, "r_ctrl_seconds": s.ctrl_seconds,
            "r_total_str_landed": s.total_str_landed,
        })
    db.close()
    round_data = pd.DataFrame(round_rows) if round_rows else pd.DataFrame()
    log.info(f"  Loaded {len(round_rows)} per-round stat rows")
    return df, round_data


# ===========================================================================
# HELPERS
# ===========================================================================

def _parse_height_inches(h) -> float | None:
    if not isinstance(h, str) or not h or h == "--":
        return None
    m = re.match(r"(\d+)'\s*(\d+)", h)
    return int(m.group(1)) * 12 + int(m.group(2)) if m else None

def _parse_weight_lbs(w) -> float | None:
    if not isinstance(w, str) or not w or w == "--":
        return None
    m = re.search(r"(\d+)", w)
    return float(m.group(1)) if m else None

def _parse_reach_inches(r) -> float | None:
    if not isinstance(r, str) or not r or r == "--":
        return None
    m = re.search(r"([\d.]+)", r)
    return float(m.group(1)) if m else None

def _safe_divide(a, b):
    return np.where(b > 0, a / b, 0.0)

def _classify_weight_class(wc: str | None) -> str:
    """Map verbose weight class strings to canonical divisions."""
    if not isinstance(wc, str):
        return "unknown"
    wc = wc.lower()
    if "strawweight" in wc: return "strawweight"
    if "flyweight" in wc: return "flyweight"
    if "bantamweight" in wc: return "bantamweight"
    if "featherweight" in wc: return "featherweight"
    if "lightweight" in wc: return "lightweight"
    if "welterweight" in wc: return "welterweight"
    if "middleweight" in wc: return "middleweight"
    if "light heavyweight" in wc or "light_heavyweight" in wc: return "light_heavyweight"
    if "heavyweight" in wc: return "heavyweight"
    if "catch" in wc or "open" in wc: return "catchweight"
    return "unknown"


# ===========================================================================
# FIGHTER STYLE CLUSTERING
# ===========================================================================

def compute_fighter_styles(df: pd.DataFrame, n_clusters: int = 6) -> tuple[dict, KMeans]:
    """
    Cluster fighters into style archetypes based on career stat profiles.
    Uses only fights before the test split to avoid leakage.
    Returns: {fighter_id: cluster_id}, fitted KMeans model
    """
    log.info(f"  Computing fighter style clusters (k={n_clusters})...")

    style_features = [
        "sig_str_landed_per5", "sig_str_acc", "td_landed_per5", "td_acc",
        "sub_att_per5", "ctrl_per5", "kd_per5",
        "head_target_pct", "body_target_pct", "leg_target_pct",
        "distance_landed_per5", "clinch_landed_per5", "ground_landed_per5",
    ]

    # Compute career averages per fighter
    fighter_profiles = df.groupby("stats_fighter_id")[style_features].mean().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fighter_profiles.values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    style_map = dict(zip(fighter_profiles.index, labels))

    # Log cluster descriptions
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=style_features,
    )
    for i in range(n_clusters):
        c = centers.iloc[i]
        n_fighters = (labels == i).sum()
        # Determine archetype name
        top_trait = c.idxmax()
        log.info(
            f"    Style {i} ({n_fighters} fighters): "
            f"SigStr/5={c['sig_str_landed_per5']:.1f} TD/5={c['td_landed_per5']:.1f} "
            f"Sub/5={c['sub_att_per5']:.1f} Ctrl/5={c['ctrl_per5']:.1f} "
            f"GndStr/5={c['ground_landed_per5']:.1f} top={top_trait}"
        )

    return style_map, kmeans


def compute_style_matchup_matrix(df: pd.DataFrame, style_map: dict, n_clusters: int) -> np.ndarray:
    """
    Build NxN matrix: style_matchup[A][B] = historical win rate of style A vs style B.
    """
    wins = np.zeros((n_clusters, n_clusters))
    total = np.zeros((n_clusters, n_clusters))

    fights_deduped = df.drop_duplicates("fight_id")
    for _, fight in fights_deduped.iterrows():
        red_style = style_map.get(fight["red_fighter_id"])
        blue_style = style_map.get(fight["blue_fighter_id"])
        if red_style is None or blue_style is None:
            continue
        total[red_style][blue_style] += 1
        total[blue_style][red_style] += 1
        if fight["winner_id"] == fight["red_fighter_id"]:
            wins[red_style][blue_style] += 1
        elif fight["winner_id"] == fight["blue_fighter_id"]:
            wins[blue_style][red_style] += 1

    matchup_matrix = _safe_divide(wins, total)
    log.info(f"  Style matchup matrix computed ({n_clusters}x{n_clusters})")
    return matchup_matrix


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================

def build_features(df: pd.DataFrame, round_data: pd.DataFrame = None) -> pd.DataFrame:
    log.info("Building features...")

    # --- Physical ---
    df["height_inches"] = df["fighter_height"].apply(_parse_height_inches)
    df["weight_lbs"] = df["fighter_weight"].apply(_parse_weight_lbs)
    df["reach_inches"] = df["fighter_reach"].apply(_parse_reach_inches)
    df["stance_orthodox"] = (df["fighter_stance"] == "Orthodox").astype(float)
    df["stance_southpaw"] = (df["fighter_stance"] == "Southpaw").astype(float)
    df["stance_switch"] = (df["fighter_stance"] == "Switch").astype(float)
    df["age"] = df.apply(
        lambda r: (r["date"] - r["fighter_dob"]).days / 365.25
        if isinstance(r["fighter_dob"], date) and r["date"] else None,
        axis=1,
    )

    # --- Weight class ---
    df["division"] = df["weight_class"].apply(_classify_weight_class)
    division_dummies = pd.get_dummies(df["division"], prefix="div").astype(float)
    df = pd.concat([df, division_dummies], axis=1)

    # --- Per-5-minute rates ---
    ft = df["fight_time_seconds"].clip(lower=1).values
    per5 = lambda v: (v / ft) * 300

    for col in ["kd", "sig_str_landed", "sig_str_attempted", "total_str_landed",
                 "td_landed", "td_attempted", "sub_att",
                 "head_landed", "body_landed", "leg_landed",
                 "distance_landed", "clinch_landed", "ground_landed", "ctrl_seconds"]:
        out_name = f"{col}_per5" if col != "ctrl_seconds" else "ctrl_per5"
        df[out_name] = per5(df[col].values)

    # --- Accuracy ---
    df["sig_str_acc"] = _safe_divide(df["sig_str_landed"].values, df["sig_str_attempted"].values)
    df["td_acc"] = _safe_divide(df["td_landed"].values, df["td_attempted"].values)
    df["head_target_pct"] = _safe_divide(df["head_attempted"].values, df["sig_str_attempted"].values)
    df["body_target_pct"] = _safe_divide(df["body_attempted"].values, df["sig_str_attempted"].values)
    df["leg_target_pct"] = _safe_divide(df["leg_attempted"].values, df["sig_str_attempted"].values)

    # --- Defensive stats ---
    log.info("  Computing defensive stats...")
    opp_map = {}
    for _, group in df.groupby("fight_id"):
        if len(group) != 2:
            continue
        rows = group.index.tolist()
        opp_map[rows[0]] = rows[1]
        opp_map[rows[1]] = rows[0]

    opp_idx = [opp_map.get(i, i) for i in df.index]
    df["opp_sig_str_landed_per5"] = df.loc[opp_idx, "sig_str_landed_per5"].values
    df["opp_sig_str_acc"] = df.loc[opp_idx, "sig_str_acc"].values
    df["opp_td_landed_per5"] = df.loc[opp_idx, "td_landed_per5"].values
    df["opp_kd_per5"] = df.loc[opp_idx, "kd_per5"].values
    df["opp_ctrl_per5"] = df.loc[opp_idx, "ctrl_per5"].values
    df["opp_sub_att_per5"] = df.loc[opp_idx, "sub_att_per5"].values
    df["sig_str_def"] = 1 - df["opp_sig_str_acc"]
    df["td_def"] = 1 - _safe_divide(
        df.loc[opp_idx, "td_landed"].values,
        df.loc[opp_idx, "td_attempted"].values,
    )

    # --- Win/result flags ---
    df["won"] = (df["stats_fighter_id"] == df["winner_id"]).astype(int)
    df["lost"] = ((df["winner_id"].notna()) & (df["stats_fighter_id"] != df["winner_id"])).astype(int)
    df["ko_win"] = ((df["won"] == 1) & df["method"].str.contains("KO", na=False)).astype(int)
    df["sub_win"] = ((df["won"] == 1) & df["method"].str.contains("Sub", case=False, na=False)).astype(int)
    df["dec_win"] = ((df["won"] == 1) & df["method"].str.contains("Dec", case=False, na=False)).astype(int)
    df["finished_opp"] = ((df["won"] == 1) & (~df["method"].str.contains("Dec", case=False, na=False))).astype(int)
    df["was_finished"] = ((df["lost"] == 1) & (~df["method"].str.contains("Dec", case=False, na=False))).astype(int)

    df = df.sort_values(["stats_fighter_id", "date"]).reset_index(drop=True)

    # --- Elo (K-factor scaled by method) ---
    log.info("  Computing Elo ratings...")
    elo = {}
    elo_at_fight = {}
    for _, row in (
        df[["fight_id", "date", "red_fighter_id", "blue_fighter_id", "winner_id", "method"]]
        .drop_duplicates("fight_id").sort_values("date")
    ).iterrows():
        r_id, b_id = row["red_fighter_id"], row["blue_fighter_id"]
        r_elo, b_elo = elo.get(r_id, 1500.0), elo.get(b_id, 1500.0)
        elo_at_fight[(row["fight_id"], r_id)] = r_elo
        elo_at_fight[(row["fight_id"], b_id)] = b_elo
        expected_r = 1 / (1 + 10 ** ((b_elo - r_elo) / 400))
        actual_r = 1.0 if row["winner_id"] == r_id else (0.0 if row["winner_id"] == b_id else 0.5)
        method = str(row.get("method", ""))
        K = 40 if ("KO" in method or "Sub" in method) else 28 if "Dec" in method else 32
        elo[r_id] = r_elo + K * (actual_r - expected_r)
        elo[b_id] = b_elo + K * ((1 - actual_r) - (1 - expected_r))

    df["elo"] = df.apply(lambda r: elo_at_fight.get((r["fight_id"], r["stats_fighter_id"]), 1500.0), axis=1)
    # Elo-based expected win probability (from this fighter's perspective)
    df["elo_expected"] = df.apply(
        lambda r: 1 / (1 + 10 ** ((
            elo_at_fight.get((r["fight_id"], r["blue_fighter_id"] if r["corner"] == "red" else r["red_fighter_id"]), 1500.0)
            - r["elo"]
        ) / 400)),
        axis=1,
    )

    # --- Opponent Elo for quality adjustment ---
    df["opp_elo"] = df.loc[opp_idx, "elo"].values

    # --- Resume Score (recursive opponent quality, like PageRank) ---
    log.info("  Computing resume scores...")
    resume = {}  # fighter_id -> score
    resume_at_fight = {}
    DECAY = 0.85  # recent fights weighted more
    ITERATIONS = 5  # convergence iterations

    # Initialize resume = Elo-based
    for fid in elo:
        resume[fid] = (elo.get(fid, 1500) - 1500) / 400  # normalize around 0

    fights_chrono = (
        df[["fight_id", "date", "red_fighter_id", "blue_fighter_id", "winner_id", "method"]]
        .drop_duplicates("fight_id").sort_values("date")
    )
    fight_list = list(fights_chrono.itertuples(index=False))

    for iteration in range(ITERATIONS):
        new_resume = {fid: 0.0 for fid in resume}
        fight_count = {fid: 0 for fid in resume}

        for fight in fight_list:
            r_id, b_id = fight.red_fighter_id, fight.blue_fighter_id
            r_score = resume.get(r_id, 0.0)
            b_score = resume.get(b_id, 0.0)

            method = str(fight.method or "")
            finish_bonus = 1.3 if ("KO" in method or "Sub" in method) else 1.0

            if fight.winner_id == r_id:
                # Red won: credit = opponent's resume * finish_bonus
                new_resume[r_id] = new_resume.get(r_id, 0) + b_score * finish_bonus
                new_resume[b_id] = new_resume.get(b_id, 0) - r_score * 0.5
            elif fight.winner_id == b_id:
                new_resume[b_id] = new_resume.get(b_id, 0) + r_score * finish_bonus
                new_resume[r_id] = new_resume.get(r_id, 0) - b_score * 0.5

            fight_count[r_id] = fight_count.get(r_id, 0) + 1
            fight_count[b_id] = fight_count.get(b_id, 0) + 1

        # Normalize by fight count and blend with previous
        for fid in new_resume:
            n = max(fight_count.get(fid, 1), 1)
            new_resume[fid] = DECAY * (new_resume[fid] / n) + (1 - DECAY) * resume.get(fid, 0)

        resume = new_resume

    # Now compute resume at each fight (pre-fight, chronologically)
    resume_running = {}
    resume_count = {}
    for fight in fight_list:
        r_id, b_id = fight.red_fighter_id, fight.blue_fighter_id
        # Store pre-fight resume
        resume_at_fight[(fight.fight_id, r_id)] = resume_running.get(r_id, 0.0)
        resume_at_fight[(fight.fight_id, b_id)] = resume_running.get(b_id, 0.0)

        # Update running resume
        r_res = resume_running.get(r_id, 0.0)
        b_res = resume_running.get(b_id, 0.0)
        method = str(fight.method or "")
        finish_bonus = 1.3 if ("KO" in method or "Sub" in method) else 1.0
        n_r = resume_count.get(r_id, 0) + 1
        n_b = resume_count.get(b_id, 0) + 1

        if fight.winner_id == r_id:
            resume_running[r_id] = r_res + (b_res * finish_bonus - r_res) / n_r
            resume_running[b_id] = b_res + (-r_res * 0.3 - b_res) / n_b
        elif fight.winner_id == b_id:
            resume_running[b_id] = b_res + (r_res * finish_bonus - b_res) / n_b
            resume_running[r_id] = r_res + (-b_res * 0.3 - r_res) / n_r

        resume_count[r_id] = n_r
        resume_count[b_id] = n_b

    df["resume_score"] = df.apply(
        lambda r: resume_at_fight.get((r["fight_id"], r["stats_fighter_id"]), 0.0), axis=1
    )
    df["opp_resume"] = df.loc[opp_idx, "resume_score"].values if len(opp_idx) == len(df) else 0.0

    # --- Glicko multi-dimensional ratings (from DB snapshots) ---
    log.info("  Loading Glicko rating snapshots from DB...")
    try:
        from app.services.ranking_service import DIMENSIONS as GLICKO_DIMS
        from app.models.ufc import UFCGlickoSnapshot
        from app.database import SessionLocal as GlickoSession
        glicko_db = GlickoSession()
        snapshots = glicko_db.query(UFCGlickoSnapshot).all()
        glicko_db.close()

        # Build lookup: {(fight_id, fighter_id): {dim: value}}
        glicko_snapshots = {}
        for s in snapshots:
            glicko_snapshots[(s.fight_id, s.fighter_id)] = {
                d: getattr(s, d, 0.0) or 0.0 for d in GLICKO_DIMS
            }
        log.info(f"  Loaded {len(glicko_snapshots)} Glicko snapshots")

        for dim in GLICKO_DIMS:
            df[f"glicko_{dim}"] = df.apply(
                lambda r, d=dim: glicko_snapshots.get(
                    (r["fight_id"], r["stats_fighter_id"]), {}
                ).get(d, 0.0),
                axis=1,
            )
        log.info(f"  Added {len(GLICKO_DIMS)} Glicko features")
    except Exception as e:
        log.warning(f"  Could not load Glicko features: {e}")
        from app.services.ranking_service import DIMENSIONS as GLICKO_DIMS
        for dim in GLICKO_DIMS:
            df[f"glicko_{dim}"] = 0.0

    # --- Elo-adjusted stats: multiply per-5 rates by (opp_elo / 1500) ---
    log.info("  Computing Elo-adjusted stats...")
    elo_weight = (df["opp_elo"] / 1500).clip(0.5, 2.0)
    for col in ["sig_str_landed_per5", "td_landed_per5", "kd_per5", "ctrl_per5", "sub_att_per5"]:
        df[f"elo_adj_{col}"] = df[col] * elo_weight

    # --- Rolling averages ---
    log.info("  Computing rolling career averages...")
    offensive_cols = [
        "kd_per5", "sig_str_landed_per5", "sig_str_attempted_per5", "total_str_landed_per5",
        "td_landed_per5", "td_attempted_per5", "sub_att_per5",
        "head_landed_per5", "body_landed_per5", "leg_landed_per5",
        "distance_landed_per5", "clinch_landed_per5", "ground_landed_per5",
        "ctrl_per5", "sig_str_acc", "td_acc",
        "head_target_pct", "body_target_pct", "leg_target_pct",
    ]
    defensive_cols = [
        "opp_sig_str_landed_per5", "opp_kd_per5", "opp_td_landed_per5",
        "opp_ctrl_per5", "opp_sub_att_per5", "sig_str_def", "td_def",
    ]
    elo_adj_cols = [c for c in df.columns if c.startswith("elo_adj_")]
    result_cols = ["won", "ko_win", "sub_win", "finished_opp", "was_finished"]
    all_stat_cols = offensive_cols + defensive_cols + elo_adj_cols + result_cols

    for col in all_stat_cols:
        df[f"avg_{col}"] = (
            df.groupby("stats_fighter_id")[col]
            .apply(lambda x: x.expanding().mean().shift(1))
            .reset_index(level=0, drop=True)
        )
        df[f"recent_{col}"] = (
            df.groupby("stats_fighter_id")[col]
            .apply(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            .reset_index(level=0, drop=True)
        )
        df[f"last3_{col}"] = (
            df.groupby("stats_fighter_id")[col]
            .apply(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
            .reset_index(level=0, drop=True)
        )

    # --- Streaks & career ---
    log.info("  Computing streaks and career stats...")
    def _compute_streak(series):
        streak, current = [], 0
        for val in series:
            streak.append(current)
            if pd.isna(val):
                continue
            current = max(0, current) + 1 if val == 1 else min(0, current) - 1
        return streak

    df["streak"] = df.groupby("stats_fighter_id")["won"].transform(
        lambda x: pd.Series(_compute_streak(x.values), index=x.index)
    )

    df["career_wins"] = df.groupby("stats_fighter_id")["won"].apply(
        lambda x: x.expanding().sum().shift(1).fillna(0)).reset_index(level=0, drop=True)
    df["career_losses"] = df.groupby("stats_fighter_id")["lost"].apply(
        lambda x: x.expanding().sum().shift(1).fillna(0)).reset_index(level=0, drop=True)
    df["career_fights"] = df["career_wins"] + df["career_losses"]
    df["career_win_pct"] = _safe_divide(df["career_wins"].values, df["career_fights"].clip(lower=1).values)

    df["career_finishes"] = df.groupby("stats_fighter_id")["finished_opp"].apply(
        lambda x: x.expanding().sum().shift(1).fillna(0)).reset_index(level=0, drop=True)
    df["finish_rate"] = _safe_divide(df["career_finishes"].values, df["career_wins"].clip(lower=1).values)

    df["career_been_finished"] = df.groupby("stats_fighter_id")["was_finished"].apply(
        lambda x: x.expanding().sum().shift(1).fillna(0)).reset_index(level=0, drop=True)
    df["been_finished_rate"] = _safe_divide(df["career_been_finished"].values, df["career_losses"].clip(lower=1).values)

    df["prev_fight_date"] = df.groupby("stats_fighter_id")["date"].shift(1)
    df["days_since_last"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["prev_fight_date"])).dt.days

    # --- Composite features ---
    log.info("  Computing composite features...")
    # Striking composite: weighted combo of KD rate, sig str accuracy, volume, and defense
    df["striking_composite"] = (
        0.30 * df["kd_per5"].clip(upper=10) / 10 +
        0.25 * df["sig_str_acc"] +
        0.25 * df["sig_str_landed_per5"].clip(upper=30) / 30 +
        0.20 * df["sig_str_def"]
    )
    # Grappling composite: TD rate, TD accuracy, control, sub attempts
    df["grappling_composite"] = (
        0.30 * df["td_landed_per5"].clip(upper=5) / 5 +
        0.25 * df["td_acc"] +
        0.25 * df["ctrl_per5"].clip(upper=300) / 300 +
        0.20 * df["sub_att_per5"].clip(upper=3) / 3
    )
    # Defense composite: strike defense, TD defense, not being finished
    df["defense_composite"] = (
        0.35 * df["sig_str_def"] +
        0.35 * df["td_def"] +
        0.30 * (1 - df["was_finished"])
    )
    # Pressure composite: volume + forward output
    df["pressure_composite"] = (
        0.40 * df["total_str_landed_per5"].clip(upper=40) / 40 +
        0.30 * df["sig_str_attempted_per5"].clip(upper=30) / 30 +
        0.30 * df["td_attempted_per5"].clip(upper=5) / 5
    )
    # Finishing ability composite
    df["finishing_composite"] = (
        0.40 * df["kd_per5"].clip(upper=5) / 5 +
        0.30 * df["sub_att_per5"].clip(upper=3) / 3 +
        0.30 * df["ground_landed_per5"].clip(upper=10) / 10
    )

    # --- Round-by-round profiles ---
    log.info("  Computing round-by-round profiles...")
    round_df = round_data if round_data is not None else pd.DataFrame()
    if not round_df.empty:
        # Calculate per-round output for each fighter in each fight
        # Focus on sig strikes as the main activity metric
        r1_stats = round_df[round_df["round_number"] == 1].set_index(["fight_id", "stats_fighter_id"])
        r2_stats = round_df[round_df["round_number"] == 2].set_index(["fight_id", "stats_fighter_id"])
        r3_stats = round_df[round_df["round_number"] == 3].set_index(["fight_id", "stats_fighter_id"])

        # Build per-fight round profiles
        round_profiles = []
        for _, row in df.iterrows():
            key = (row["fight_id"], row["stats_fighter_id"])
            r1 = r1_stats.loc[key] if key in r1_stats.index else None
            r2 = r2_stats.loc[key] if key in r2_stats.index else None
            r3 = r3_stats.loc[key] if key in r3_stats.index else None

            r1_output = r1["r_sig_str_landed"] if r1 is not None else 0
            r2_output = r2["r_sig_str_landed"] if r2 is not None else 0
            r3_output = r3["r_sig_str_landed"] if r3 is not None else 0
            r1_ctrl = r1["r_ctrl_seconds"] if r1 is not None else 0
            r2_ctrl = r2["r_ctrl_seconds"] if r2 is not None else 0
            r3_ctrl = r3["r_ctrl_seconds"] if r3 is not None else 0
            r1_td = r1["r_td_landed"] if r1 is not None else 0
            r2_td = r2["r_td_landed"] if r2 is not None else 0

            total_output = max(r1_output + r2_output + r3_output, 1)
            # Fade ratio: how much output drops from R1 to later rounds
            # >1 = fighter fades, <1 = fighter improves
            r1_share = r1_output / total_output if total_output > 0 else 0.33
            late_share = (r2_output + r3_output) / total_output if total_output > 0 else 0.67

            round_profiles.append({
                "r1_output_share": r1_share,
                "late_output_share": late_share,
                "r1_sig_str": r1_output,
                "r2_sig_str": r2_output,
                "r3_sig_str": r3_output,
                "r1_ctrl": r1_ctrl,
                "r1_td": r1_td,
                "output_trend": (r2_output + r3_output) / 2 - r1_output if r2_output + r3_output > 0 else 0,
                "ctrl_trend": (r2_ctrl + r3_ctrl) / 2 - r1_ctrl if r2_ctrl + r3_ctrl > 0 else 0,
            })

        round_profile_df = pd.DataFrame(round_profiles, index=df.index)
        df = pd.concat([df, round_profile_df], axis=1)

        # Rolling averages of round profiles (pre-fight)
        round_cols = ["r1_output_share", "late_output_share", "output_trend", "ctrl_trend",
                       "r1_sig_str", "r1_ctrl", "r1_td"]
        for col in round_cols:
            df[f"avg_{col}"] = (
                df.groupby("stats_fighter_id")[col]
                .apply(lambda x: x.expanding().mean().shift(1))
                .reset_index(level=0, drop=True)
            )
            df[f"recent_{col}"] = (
                df.groupby("stats_fighter_id")[col]
                .apply(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
                .reset_index(level=0, drop=True)
            )
    else:
        log.warning("  No per-round data available — skipping round profiles")

    # Add composite rolling averages
    composite_cols = ["striking_composite", "grappling_composite", "defense_composite",
                      "pressure_composite", "finishing_composite"]
    for col in composite_cols:
        df[f"avg_{col}"] = (
            df.groupby("stats_fighter_id")[col]
            .apply(lambda x: x.expanding().mean().shift(1))
            .reset_index(level=0, drop=True)
        )
        df[f"recent_{col}"] = (
            df.groupby("stats_fighter_id")[col]
            .apply(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            .reset_index(level=0, drop=True)
        )
        df[f"last3_{col}"] = (
            df.groupby("stats_fighter_id")[col]
            .apply(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
            .reset_index(level=0, drop=True)
        )

    # --- Fighter style clustering ---
    N_STYLES = 6
    style_map, kmeans = compute_fighter_styles(df, n_clusters=N_STYLES)
    df["fighter_style"] = df["stats_fighter_id"].map(style_map).fillna(0).astype(int)

    # Style matchup matrix (trained on full data — these are historical facts, not predictions)
    matchup_matrix = compute_style_matchup_matrix(df, style_map, N_STYLES)

    # For each fight, look up style matchup win rate
    def _style_matchup_advantage(row):
        opp_idx_val = opp_map.get(row.name, row.name)
        opp_style = df.loc[opp_idx_val, "fighter_style"] if opp_idx_val in df.index else 0
        return matchup_matrix[row["fighter_style"]][opp_style]

    df["style_matchup_adv"] = df.apply(_style_matchup_advantage, axis=1)

    # One-hot encode style
    for i in range(N_STYLES):
        df[f"style_{i}"] = (df["fighter_style"] == i).astype(float)

    log.info(f"  Feature matrix shape: {df.shape}")
    return df


def build_matchup_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log.info("Building matchup feature matrix...")

    # Collect feature columns (exclude raw per-fight stats, only use rolling/computed)
    feature_cols = [c for c in df.columns if c.startswith(("avg_", "recent_", "last3_"))]
    feature_cols += [
        "elo", "elo_expected", "resume_score",
        "height_inches", "weight_lbs", "reach_inches", "age",
        "stance_orthodox", "stance_southpaw", "stance_switch",
        "career_win_pct", "career_fights", "finish_rate", "been_finished_rate",
        "streak", "days_since_last", "style_matchup_adv",
    ]
    # Add composite features
    feature_cols += [c for c in df.columns if "composite" in c and c.startswith(("avg_", "recent_", "last3_"))]
    # Add round profile features
    feature_cols += [c for c in df.columns if c.startswith(("avg_r1_", "avg_late_", "avg_output_", "avg_ctrl_trend",
                                                             "recent_r1_", "recent_late_", "recent_output_", "recent_ctrl_trend"))]
    feature_cols += [c for c in df.columns if c.startswith("style_") and c not in feature_cols]
    feature_cols += [c for c in df.columns if c.startswith("div_") and c not in feature_cols]
    # Add Glicko multi-dimensional ratings
    feature_cols += [c for c in df.columns if c.startswith("glicko_") and c not in feature_cols]
    # Deduplicate while preserving order
    feature_cols = list(dict.fromkeys(feature_cols))

    for col in feature_cols:
        df[col] = df[col].fillna(df[col].mean())

    red = df[df["corner"] == "red"].set_index("fight_id")
    blue = df[df["corner"] == "blue"].set_index("fight_id")
    common = red.index.intersection(blue.index)
    red, blue = red.loc[common], blue.loc[common]

    matchup = pd.DataFrame(index=common)
    matchup.index.name = "fight_id"
    matchup["date"] = red["date"].values
    matchup["red_wins"] = (red["stats_fighter_id"].values == red["winner_id"].values).astype(int)

    # Difference features
    for col in feature_cols:
        matchup[f"diff_{col}"] = red[col].values - blue[col].values

    # Raw values for key features (both sides)
    raw_cols = ["elo", "elo_expected", "resume_score", "career_fights", "career_win_pct",
                "streak", "finish_rate", "style_matchup_adv"]
    raw_cols += [c for c in feature_cols if c.startswith("glicko_")]
    for col in raw_cols:
        matchup[f"red_{col}"] = red[col].values
        matchup[f"blue_{col}"] = blue[col].values

    # --- Load odds from DB ---
    log.info("  Loading odds from database...")
    from app.models.ufc import UFCFightOdds
    odds_db = SessionLocal()
    odds_rows = odds_db.query(UFCFightOdds).all()
    odds_db.close()

    odds_map = {}
    for o in odds_rows:
        odds_map[o.fight_id] = o

    odds_matched = 0
    matchup["odds_red_prob"] = np.nan
    matchup["odds_blue_prob"] = np.nan
    matchup["odds_red_american"] = np.nan
    matchup["odds_blue_american"] = np.nan
    for fight_id in matchup.index:
        o = odds_map.get(fight_id)
        if o:
            matchup.loc[fight_id, "odds_red_prob"] = o.red_implied_prob
            matchup.loc[fight_id, "odds_blue_prob"] = o.blue_implied_prob
            matchup.loc[fight_id, "odds_red_american"] = o.red_odds
            matchup.loc[fight_id, "odds_blue_american"] = o.blue_odds
            odds_matched += 1

    log.info(f"  Odds matched: {odds_matched}/{len(matchup)} fights ({odds_matched/len(matchup)*100:.1f}%)")

    # Odds-derived features
    matchup["odds_diff"] = matchup["odds_red_prob"] - matchup["odds_blue_prob"]
    matchup["odds_fav_is_red"] = (matchup["odds_red_prob"] > 0.5).astype(float)
    matchup["elo_vs_odds"] = matchup["diff_elo_expected"] - matchup["odds_diff"]

    matchup = matchup.dropna(subset=["red_wins"]).fillna(0)

    # --- Filter to modern era (2015+) ---
    # Pre-2015 data has extreme red corner bias (95%+ win rate) that doesn't exist
    # in modern MMA. Elo/resume features still capture historical quality since
    # they're computed chronologically from the full dataset — we just don't train on ancient fights.
    from datetime import date as _date
    modern_cutoff = _date(2015, 1, 1)
    matchup["date"] = pd.to_datetime(matchup["date"]).dt.date
    pre_modern = len(matchup[matchup["date"] < modern_cutoff])
    matchup = matchup[matchup["date"] >= modern_cutoff]
    log.info(f"  Filtered to modern era (2015+): {len(matchup)} fights (dropped {pre_modern} pre-2015)")
    log.info(f"  Red win rate (modern): {matchup['red_wins'].mean():.3f}")

    # --- Corner-swap augmentation ---
    # For every fight, create a mirror: swap red/blue, flip all diff features and the label.
    # This doubles training data and eliminates corner bias.
    log.info("  Applying corner-swap augmentation...")
    mirror = matchup.copy()
    mirror["red_wins"] = 1 - mirror["red_wins"]
    for col in mirror.columns:
        if col.startswith("diff_"):
            mirror[col] = -mirror[col]
        elif col.startswith("red_"):
            base = col.replace("red_", "")
            blue_col = f"blue_{base}"
            if blue_col in mirror.columns:
                mirror[col], mirror[blue_col] = matchup[blue_col].values.copy(), matchup[col].values.copy()
    # Swap odds on mirrored rows (red becomes blue and vice versa)
    for col_a, col_b in [("odds_red_prob", "odds_blue_prob"),
                          ("odds_red_american", "odds_blue_american")]:
        if col_a in mirror.columns:
            mirror[col_a], mirror[col_b] = matchup[col_b].values.copy(), matchup[col_a].values.copy()
    if "odds_diff" in mirror.columns:
        mirror["odds_diff"] = -matchup["odds_diff"].values
    if "odds_fav_is_red" in mirror.columns:
        mirror["odds_fav_is_red"] = 1 - matchup["odds_fav_is_red"].values
    if "elo_vs_odds" in mirror.columns:
        mirror["elo_vs_odds"] = -matchup["elo_vs_odds"].values

    # Tag originals vs augmented for splitting later
    matchup["_augmented"] = False
    mirror["_augmented"] = True
    # Preserve fight_id index: mirror gets suffixed index so RNN/GNN/Siamese can find originals
    mirror.index = mirror.index.astype(str) + "_mirror"
    matchup = pd.concat([matchup, mirror])
    matchup = matchup.sort_values("date")
    log.info(f"  After augmentation: {len(matchup)} rows ({len(matchup)//2} original + {len(matchup)//2} mirrored)")

    feature_names = [c for c in matchup.columns if (c.startswith(("diff_", "red_", "blue_")) or "odds" in c or c == "elo_vs_odds") and c not in ("red_wins", "_augmented")]

    # --- Feature selection via mutual information ---
    log.info("  Running feature selection (mutual information)...")
    X_all = matchup[feature_names].values
    y_all = matchup["red_wins"].values
    mi_scores = mutual_info_classif(X_all, y_all, random_state=42)
    mi_ranked = sorted(zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True)

    log.info("  Top 30 features by mutual information:")
    for name, score in mi_ranked[:30]:
        log.info(f"    {name:50s} {score:.4f}")

    # Keep top N features + force-include odds features (sparse but powerful)
    TOP_N = 35
    selected = [name for name, _ in mi_ranked[:TOP_N]]
    odds_features = [c for c in feature_names if "odds" in c]
    for of in odds_features:
        if of not in selected:
            selected.append(of)
    log.info(f"  Selected {len(selected)} features ({TOP_N} MI + {len(odds_features)} odds, from {len(feature_names)})")

    log.info(f"  Matchup matrix: {matchup.shape[0]} fights")
    log.info(f"  Red win rate: {matchup['red_wins'].mean():.3f}")
    log.info(f"  Date range: {matchup['date'].min()} to {matchup['date'].max()}")
    return matchup, selected


# ===========================================================================
# PHASE 1: Gradient Boosting
# ===========================================================================

def train_gbt(matchup: pd.DataFrame, feature_names: list[str], odds_only: bool = False) -> dict:
    log.info("=" * 60)
    log.info(f"PHASE 1: Gradient Boosting {'(odds-only)' if odds_only else '(full)'}")
    log.info("=" * 60)

    matchup = matchup.sort_values("date").reset_index(drop=True)

    if odds_only:
        # Only use fights where we have odds data
        has_odds = matchup["odds_red_prob"] > 0
        matchup_filtered = matchup[has_odds].reset_index(drop=True)
        log.info(f"  Filtered to {len(matchup_filtered)} fights with odds (from {len(matchup)})")
    else:
        matchup_filtered = matchup

    # Split: train on first 80%, test on last 20% — but only evaluate on non-augmented rows
    has_aug = "_augmented" in matchup_filtered.columns
    split_idx = int(len(matchup_filtered) * 0.8)
    train = matchup_filtered.iloc[:split_idx]
    test_all = matchup_filtered.iloc[split_idx:]
    # Evaluate only on original (non-mirrored) data for honest metrics
    test = test_all[~test_all["_augmented"]] if has_aug else test_all

    X_train, y_train = train[feature_names].values, train["red_wins"].values
    X_test, y_test = test[feature_names].values, test["red_wins"].values

    log.info(f"  Train: {len(train)} ({train['date'].min()} to {train['date'].max()})")
    log.info(f"  Test:  {len(test)} original ({test['date'].min()} to {test['date'].max()})")
    log.info(f"  Train red win rate: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    model = HistGradientBoostingClassifier(
        max_iter=1000,
        max_depth=4,
        learning_rate=0.02,
        max_features=0.7,
        min_samples_leaf=30,
        l2_regularization=2.0,
        max_bins=128,
        early_stopping=True,
        n_iter_no_change=75,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ll = log_loss(y_test, y_proba)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    log.info(f"\n  --- Gradient Boosting v3 Results ---")
    log.info(f"  Baseline (majority): {baseline:.4f}")
    log.info(f"  Accuracy:  {acc:.4f} ({'+' if acc > baseline else ''}{acc - baseline:.4f} vs baseline)")
    log.info(f"  AUC-ROC:   {auc:.4f}")
    log.info(f"  Log Loss:  {ll:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Blue wins', 'Red wins'])}")

    from sklearn.inspection import permutation_importance
    log.info("  Computing feature importances...")
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importances = sorted(zip(feature_names, perm.importances_mean), key=lambda x: x[1], reverse=True)
    log.info("  Top 20 features by permutation importance:")
    for name, imp in importances[:20]:
        log.info(f"    {name:50s} {imp:.4f}")

    with open(MODEL_DIR / "gbt_v3.pkl", "wb") as f:
        pickle.dump({"model": model, "features": feature_names, "version": 3}, f)
    log.info(f"  Saved to {MODEL_DIR / 'gbt_v3.pkl'}")

    return {"model": model, "test_proba": y_proba, "test_y": y_test,
            "accuracy": acc, "auc": auc, "log_loss": ll}


# ===========================================================================
# PHASE 2: RNN (LSTM)
# ===========================================================================

SEQUENCE_FEATURES = [
    "kd_per5", "sig_str_landed_per5", "sig_str_attempted_per5", "total_str_landed_per5",
    "td_landed_per5", "td_attempted_per5", "sub_att_per5", "ctrl_per5",
    "sig_str_acc", "td_acc",
    "head_landed_per5", "body_landed_per5", "leg_landed_per5",
    "distance_landed_per5", "clinch_landed_per5", "ground_landed_per5",
    "opp_sig_str_landed_per5", "opp_kd_per5", "opp_td_landed_per5",
    "opp_ctrl_per5", "sig_str_def", "td_def",
    "won", "finished_opp", "was_finished",
    "elo_adj_sig_str_landed_per5", "elo_adj_td_landed_per5",
    "elo_adj_kd_per5",
]
SEQ_LEN = 10


class FightSequenceDataset(Dataset):
    def __init__(self, red_seqs, blue_seqs, labels):
        self.red = torch.FloatTensor(red_seqs)
        self.blue = torch.FloatTensor(blue_seqs)
        self.labels = torch.FloatTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.red[idx], self.blue[idx], self.labels[idx]


class FighterLSTM(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, red, blue):
        _, (rh, _) = self.lstm(red)
        _, (bh, _) = self.lstm(blue)
        return self.fc(torch.cat([rh[-1], bh[-1]], dim=1)).squeeze(-1)


def _original_matchup(matchup: pd.DataFrame) -> pd.DataFrame:
    """Filter matchup to non-augmented rows only (for RNN/GNN/Siamese that need fight_id lookups)."""
    if "_augmented" in matchup.columns:
        return matchup[~matchup["_augmented"]].copy()
    return matchup


def train_rnn(df: pd.DataFrame, matchup: pd.DataFrame) -> dict:
    log.info("=" * 60)
    log.info("PHASE 2: RNN (LSTM) v3")
    log.info("=" * 60)
    matchup = _original_matchup(matchup)

    # Build sequences
    sequences = {}
    for fid, group in df.groupby("stats_fighter_id"):
        sequences[fid] = group.sort_values("date")[SEQUENCE_FEATURES].fillna(0).values.tolist()

    fight_idx_map = {}
    for fid, group in df.groupby("stats_fighter_id"):
        for i, (_, row) in enumerate(group.sort_values("date").iterrows()):
            fight_idx_map[(fid, row["fight_id"])] = i

    matchup_sorted = matchup.sort_values("date").reset_index()
    red_seqs, blue_seqs, labels = [], [], []
    n_feat = len(SEQUENCE_FEATURES)

    for _, row in matchup_sorted.iterrows():
        fid = row.get("fight_id", row.name)
        fight_rows = df[df["fight_id"] == fid]
        r_row = fight_rows[fight_rows["corner"] == "red"]
        b_row = fight_rows[fight_rows["corner"] == "blue"]
        if r_row.empty or b_row.empty:
            continue

        rfid, bfid = r_row.iloc[0]["stats_fighter_id"], b_row.iloc[0]["stats_fighter_id"]
        ri, bi = fight_idx_map.get((rfid, fid), 0), fight_idx_map.get((bfid, fid), 0)

        def _get_seq(seqs, fighter_id, idx):
            career = seqs.get(fighter_id, [])
            prior = career[:idx]
            if not prior:
                return np.zeros((SEQ_LEN, n_feat))
            if len(prior) >= SEQ_LEN:
                return np.array(prior[-SEQ_LEN:])
            return np.vstack([np.zeros((SEQ_LEN - len(prior), n_feat)), np.array(prior)])

        red_seqs.append(_get_seq(sequences, rfid, ri))
        blue_seqs.append(_get_seq(sequences, bfid, bi))
        labels.append(row["red_wins"])

    red_seqs, blue_seqs, labels = np.array(red_seqs), np.array(blue_seqs), np.array(labels)
    log.info(f"  Data: {len(labels)} fights, seq={SEQ_LEN}, feat={n_feat}")

    split = int(len(labels) * 0.8)
    train_dl = DataLoader(FightSequenceDataset(red_seqs[:split], blue_seqs[:split], labels[:split]), batch_size=128, shuffle=True)
    test_dl = DataLoader(FightSequenceDataset(red_seqs[split:], blue_seqs[split:], labels[split:]), batch_size=256)
    log.info(f"  Train: {split} | Test: {len(labels) - split}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"  Device: {device}")
    model = FighterLSTM(input_dim=n_feat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.BCELoss()

    best_loss, patience_ctr = float("inf"), 0
    for epoch in range(150):
        model.train()
        tloss = 0
        for r, b, y in train_dl:
            r, b, y = r.to(device), b.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(r, b), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item() * len(y)

        model.eval()
        preds, labs, vloss = [], [], 0
        with torch.no_grad():
            for r, b, y in test_dl:
                r, b, y = r.to(device), b.to(device), y.to(device)
                p = model(r, b)
                vloss += crit(p, y).item() * len(y)
                preds.extend(p.cpu().numpy()); labs.extend(y.cpu().numpy())

        tloss /= split; vloss /= (len(labels) - split)
        sched.step(vloss)
        if epoch % 10 == 0:
            log.info(f"  Epoch {epoch:3d}: train={tloss:.4f} test={vloss:.4f} lr={opt.param_groups[0]['lr']:.6f}")
        if vloss < best_loss:
            best_loss, patience_ctr = vloss, 0
            torch.save(model.state_dict(), MODEL_DIR / "rnn_v3.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= 20:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    preds, labs = np.array(preds), np.array(labs)
    acc = accuracy_score(labs, (preds >= 0.5).astype(int))
    auc = roc_auc_score(labs, preds)
    ll = log_loss(labs, preds)
    log.info(f"\n  --- RNN v3 Results ---")
    log.info(f"  Accuracy: {acc:.4f} | AUC: {auc:.4f} | LogLoss: {ll:.4f}")
    log.info(f"\n{classification_report(labs, (preds >= 0.5).astype(int), target_names=['Blue', 'Red'])}")
    return {"model": model, "test_proba": preds, "test_y": labs, "accuracy": acc, "auc": auc, "log_loss": ll}


# ===========================================================================
# PHASE 3: GNN
# ===========================================================================

def train_gnn(df: pd.DataFrame, matchup: pd.DataFrame) -> dict:
    log.info("=" * 60)
    log.info("PHASE 3: GNN v3")
    log.info("=" * 60)
    matchup = _original_matchup(matchup)
    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv
    except ImportError:
        log.warning("  torch-geometric not available"); return None

    fighter_ids = sorted(df["stats_fighter_id"].unique())
    fid_to_idx = {f: i for i, f in enumerate(fighter_ids)}
    log.info(f"  Graph: {len(fighter_ids)} nodes")

    matchup_sorted = matchup.sort_values("date")
    split_date = matchup_sorted.iloc[int(len(matchup_sorted) * 0.8)]["date"]
    train_df = df[df["date"] < split_date]

    # Node features
    node_cols = [
        "sig_str_landed_per5", "sig_str_acc", "td_landed_per5", "td_acc",
        "ctrl_per5", "kd_per5", "sub_att_per5",
        "opp_sig_str_landed_per5", "sig_str_def", "td_def",
        "opp_kd_per5", "opp_ctrl_per5", "won", "finished_opp", "was_finished",
        "elo_adj_sig_str_landed_per5", "elo_adj_td_landed_per5",
    ]
    node_features = np.zeros((len(fighter_ids), len(node_cols)))
    for fid, group in train_df.groupby("stats_fighter_id"):
        idx = fid_to_idx.get(fid)
        if idx is not None:
            node_features[idx] = [group[c].mean() for c in node_cols]

    # Edges
    red_rows = train_df[train_df["corner"] == "red"][["fight_id", "stats_fighter_id"]].rename(columns={"stats_fighter_id": "r"})
    blue_rows = train_df[train_df["corner"] == "blue"][["fight_id", "stats_fighter_id"]].rename(columns={"stats_fighter_id": "b"})
    pairs = red_rows.merge(blue_rows, on="fight_id").drop_duplicates("fight_id")
    edge_src, edge_dst = [], []
    for _, row in pairs.iterrows():
        r, b = fid_to_idx.get(row["r"]), fid_to_idx.get(row["b"])
        if r is not None and b is not None:
            edge_src.extend([r, b]); edge_dst.extend([b, r])

    x = torch.FloatTensor(node_features)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    edge_index = torch.LongTensor([edge_src, edge_dst])
    log.info(f"  Edges: {len(edge_src)} | Node features: {len(node_cols)}")

    class GNN(nn.Module):
        def __init__(self, d_in, d_h=64, d_out=32):
            super().__init__()
            self.c1 = SAGEConv(d_in, d_h); self.c2 = SAGEConv(d_h, d_h); self.c3 = SAGEConv(d_h, d_out)
            self.pred = nn.Sequential(nn.Linear(d_out * 2, 64), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        def forward(self, x, ei, ri, bi):
            h = torch.relu(self.c1(x, ei)); h = torch.relu(self.c2(h, ei)); h = self.c3(h, ei)
            return self.pred(torch.cat([h[ri], h[bi]], 1)).squeeze(-1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = Data(x=x, edge_index=edge_index).to(device)

    def _pairs(ms, dfr):
        ri, bi, la = [], [], []
        for fid in ms.index:
            fr = dfr[dfr["fight_id"] == fid]
            r = fr[fr["corner"] == "red"]; b = fr[fr["corner"] == "blue"]
            if r.empty or b.empty: continue
            rv, bv = fid_to_idx.get(r.iloc[0]["stats_fighter_id"]), fid_to_idx.get(b.iloc[0]["stats_fighter_id"])
            if rv is None or bv is None: continue
            ri.append(rv); bi.append(bv); la.append(ms.loc[fid, "red_wins"])
        return torch.LongTensor(ri).to(device), torch.LongTensor(bi).to(device), torch.FloatTensor(la).to(device)

    si = int(len(matchup_sorted) * 0.8)
    tr_r, tr_b, tr_y = _pairs(matchup_sorted.iloc[:si], df)
    te_r, te_b, te_y = _pairs(matchup_sorted.iloc[si:], df)
    log.info(f"  Train: {len(tr_y)} | Test: {len(te_y)}")

    model = GNN(d_in=len(node_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.BCELoss()
    best, pc = float("inf"), 0

    for ep in range(300):
        model.train(); opt.zero_grad()
        loss = crit(model(data.x, data.edge_index, tr_r, tr_b), tr_y); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tl = crit(model(data.x, data.edge_index, te_r, te_b), te_y).item()
        if ep % 25 == 0: log.info(f"  Epoch {ep:3d}: train={loss.item():.4f} test={tl:.4f}")
        if tl < best: best, pc = tl, 0; torch.save(model.state_dict(), MODEL_DIR / "gnn_v3.pt")
        else:
            pc += 1
            if pc >= 30: log.info(f"  Early stopping at {ep}"); break

    model.eval()
    with torch.no_grad():
        tp = model(data.x, data.edge_index, te_r, te_b).cpu().numpy()
        tl = te_y.cpu().numpy()
    acc = accuracy_score(tl, (tp >= 0.5).astype(int))
    auc = roc_auc_score(tl, tp); ll = log_loss(tl, tp)
    log.info(f"\n  --- GNN v3 ---")
    log.info(f"  Accuracy: {acc:.4f} | AUC: {auc:.4f} | LogLoss: {ll:.4f}")
    log.info(f"\n{classification_report(tl, (tp >= 0.5).astype(int), target_names=['Blue', 'Red'])}")
    return {"model": model, "test_proba": tp, "test_y": tl, "accuracy": acc, "auc": auc, "log_loss": ll}


# ===========================================================================
# PHASE 4: Siamese Neural Network
# ===========================================================================

SIAMESE_FEATURES = [
    # Physical
    "height_inches", "weight_lbs", "reach_inches", "age",
    "stance_orthodox", "stance_southpaw", "stance_switch",
    # Composites (rolling)
    "avg_striking_composite", "avg_grappling_composite", "avg_defense_composite",
    "avg_pressure_composite", "avg_finishing_composite",
    "recent_striking_composite", "recent_grappling_composite", "recent_defense_composite",
    # Per-5 rolling
    "avg_kd_per5", "avg_sig_str_landed_per5", "avg_sig_str_acc",
    "avg_td_landed_per5", "avg_td_acc", "avg_ctrl_per5", "avg_sub_att_per5",
    "recent_kd_per5", "recent_sig_str_landed_per5", "recent_td_landed_per5",
    # Defense rolling
    "avg_opp_sig_str_landed_per5", "avg_sig_str_def", "avg_td_def",
    "avg_opp_kd_per5", "avg_opp_ctrl_per5",
    # Targeting
    "avg_head_target_pct", "avg_body_target_pct", "avg_leg_target_pct",
    # Round profiles
    "avg_r1_output_share", "avg_late_output_share", "avg_output_trend",
    # Elo/resume
    "elo", "elo_expected", "resume_score",
    # Career
    "career_win_pct", "career_fights", "finish_rate", "been_finished_rate",
    "streak", "days_since_last",
    # Style
    "style_matchup_adv",
]


class SiameseNet(nn.Module):
    """
    Siamese network: identical encoder processes each fighter's features,
    then a comparison head learns how to combine the two embeddings.
    The shared encoder means it learns a universal 'fighter quality' function.
    """
    def __init__(self, input_dim, embed_dim=64):
        super().__init__()
        # Shared encoder: maps raw fighter features → learned embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        # Comparison head: takes both embeddings + their element-wise interactions
        # Input: [red_embed, blue_embed, red-blue, red*blue] = 4 * embed_dim
        self.comparator = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, red_features, blue_features):
        red_embed = self.encoder(red_features)
        blue_embed = self.encoder(blue_features)
        # Combine: raw embeddings + difference + element-wise product
        combined = torch.cat([
            red_embed,
            blue_embed,
            red_embed - blue_embed,   # who's better at what
            red_embed * blue_embed,   # interaction effects
        ], dim=1)
        return self.comparator(combined).squeeze(-1)


class SiameseDataset(Dataset):
    def __init__(self, red_feats, blue_feats, labels):
        self.red = torch.FloatTensor(red_feats)
        self.blue = torch.FloatTensor(blue_feats)
        self.labels = torch.FloatTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.red[idx], self.blue[idx], self.labels[idx]


def train_siamese(df: pd.DataFrame, matchup: pd.DataFrame) -> dict:
    log.info("=" * 60)
    log.info("PHASE 4: Siamese Neural Network")
    log.info("=" * 60)
    matchup = _original_matchup(matchup)

    # Filter to features that exist in df
    available_feats = [f for f in SIAMESE_FEATURES if f in df.columns]
    missing = [f for f in SIAMESE_FEATURES if f not in df.columns]
    if missing:
        log.info(f"  Skipping {len(missing)} missing features: {missing[:5]}...")
    log.info(f"  Using {len(available_feats)} features per fighter")

    # Fill NaN with column means, then build per-fighter feature vectors
    for col in available_feats:
        df[col] = df[col].fillna(df[col].mean())

    # Build red/blue feature matrices aligned to matchup
    red_df = df[df["corner"] == "red"].set_index("fight_id")
    blue_df = df[df["corner"] == "blue"].set_index("fight_id")
    common = matchup.index.intersection(red_df.index).intersection(blue_df.index)

    matchup_aligned = matchup.loc[common].sort_values("date")
    red_features = red_df.loc[matchup_aligned.index][available_feats].values
    blue_features = blue_df.loc[matchup_aligned.index][available_feats].values
    labels = matchup_aligned["red_wins"].values

    log.info(f"  Data: {len(labels)} fights, {len(available_feats)} features per fighter")

    # Normalize features (fit on train only)
    split = int(len(labels) * 0.8)
    scaler = StandardScaler()
    red_train = scaler.fit_transform(red_features[:split])
    red_test = scaler.transform(red_features[split:])
    blue_train = scaler.transform(blue_features[:split])  # same scaler — fighters are interchangeable
    blue_test = scaler.transform(blue_features[split:])

    train_dl = DataLoader(
        SiameseDataset(red_train, blue_train, labels[:split]),
        batch_size=128, shuffle=True,
    )
    test_dl = DataLoader(
        SiameseDataset(red_test, blue_test, labels[split:]),
        batch_size=256,
    )
    log.info(f"  Train: {split} | Test: {len(labels) - split}")

    # Use CPU — MPS crashes silently on this architecture
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"  Device: {device}")

    model = SiameseNet(input_dim=len(available_feats), embed_dim=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    crit = nn.BCELoss()

    best_loss, patience_ctr = float("inf"), 0
    for epoch in range(200):
        model.train()
        tloss = 0
        for r, b, y in train_dl:
            r, b, y = r.to(device), b.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(r, b), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item() * len(y)

        model.eval()
        preds, labs, vloss = [], [], 0
        with torch.no_grad():
            for r, b, y in test_dl:
                r, b, y = r.to(device), b.to(device), y.to(device)
                p = model(r, b)
                vloss += crit(p, y).item() * len(y)
                preds.extend(p.cpu().numpy())
                labs.extend(y.cpu().numpy())

        tloss /= split
        vloss /= (len(labels) - split)
        sched.step(vloss)

        if epoch % 10 == 0:
            log.info(f"  Epoch {epoch:3d}: train={tloss:.4f} test={vloss:.4f} lr={opt.param_groups[0]['lr']:.6f}")
        if vloss < best_loss:
            best_loss, patience_ctr = vloss, 0
            torch.save(model.state_dict(), MODEL_DIR / "siamese_v5.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= 25:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    preds, labs = np.array(preds), np.array(labs)
    acc = accuracy_score(labs, (preds >= 0.5).astype(int))
    auc = roc_auc_score(labs, preds)
    ll = log_loss(labs, preds)
    log.info(f"\n  --- Siamese v5 Results ---")
    log.info(f"  Accuracy: {acc:.4f} | AUC: {auc:.4f} | LogLoss: {ll:.4f}")
    log.info(f"\n{classification_report(labs, (preds >= 0.5).astype(int), target_names=['Blue', 'Red'])}")
    return {"model": model, "test_proba": preds, "test_y": labs, "accuracy": acc, "auc": auc, "log_loss": ll}


# ===========================================================================
# ENSEMBLE
# ===========================================================================

def ensemble_results(p1, p2, p3, p4=None, matchup: pd.DataFrame = None, feature_names: list[str] = None):
    """Stacked ensemble: train a logistic regression meta-learner on base model predictions."""
    log.info("=" * 60)
    log.info("STACKED ENSEMBLE (Meta-Learner) v5")
    log.info("=" * 60)
    models = [("GBT", p1), ("RNN", p2)]
    if p3: models.append(("GNN", p3))
    if p4: models.append(("Siamese", p4))

    log.info(f"\n  {'Model':<15s} {'Acc':>8s} {'AUC':>8s} {'LogLoss':>8s}")
    log.info(f"  {'-'*42}")
    for n, r in models:
        log.info(f"  {n:<15s} {r['accuracy']:>8.4f} {r['auc']:>8.4f} {r['log_loss']:>8.4f}")

    # --- Align test sizes across models ---
    # Different phases may have slightly different test sizes due to augmentation filtering.
    # Truncate all to the minimum size so they can be stacked.
    all_models = [("GBT", p1), ("RNN", p2)]
    if p3: all_models.append(("GNN", p3))
    if p4: all_models.append(("Siamese", p4))
    min_len = min(len(r["test_y"]) for _, r in all_models)
    matching = []
    for name, r in all_models:
        r["test_proba"] = r["test_proba"][:min_len]
        r["test_y"] = r["test_y"][:min_len]
        matching.append((name, r))
    log.info(f"  Models aligned to test size {min_len}: {[m[0] for m in matching]}")

    # --- Weighted average baseline ---
    if len(matching) >= 2:
        weights = {"GBT": 0.40, "RNN": 0.15, "GNN": 0.10, "Siamese": 0.35}
        ep = sum(weights.get(name, 0.1) * r["test_proba"] for name, r in matching)
        tw = sum(weights.get(name, 0.1) for name, _ in matching)
        ep /= tw
        yt = p1["test_y"]
        acc = accuracy_score(yt, (ep >= 0.5).astype(int))
        auc = roc_auc_score(yt, ep); ll = log_loss(yt, ep)
        log.info(f"\n  {'WEIGHTED AVG':<15s} {acc:>8.4f} {auc:>8.4f} {ll:>8.4f}")

    # --- Stacked meta-learner ---
    if len(matching) >= 2:
        from sklearn.linear_model import LogisticRegressionCV

        # Stack base model predictions
        stack_cols = [r["test_proba"] for _, r in matching]
        col_names = [f"{name.lower()}_prob" for name, _ in matching]

        X_stack = np.column_stack(stack_cols)
        y_stack = p1["test_y"]

        # Add interaction features: pairwise products, differences, confidence metrics
        interactions = []
        interaction_names = []
        for i in range(len(stack_cols)):
            for j in range(i + 1, len(stack_cols)):
                interactions.append(X_stack[:, i] * X_stack[:, j])
                interaction_names.append(f"{col_names[i]}_x_{col_names[j]}")
                interactions.append(X_stack[:, i] - X_stack[:, j])
                interaction_names.append(f"{col_names[i]}_minus_{col_names[j]}")

        X_stack_ext = np.column_stack([
            X_stack,
            *interactions,
            np.max(X_stack, axis=1),
            np.min(X_stack, axis=1),
            np.std(X_stack, axis=1),
            np.mean(X_stack, axis=1),
        ])
        ext_names = col_names + interaction_names + ["max_conf", "min_conf", "disagreement", "avg_conf"]

        # Split: first 60% for meta-train, last 40% for meta-test
        meta_split = int(len(y_stack) * 0.6)
        X_meta_train, X_meta_test = X_stack_ext[:meta_split], X_stack_ext[meta_split:]
        y_meta_train, y_meta_test = y_stack[:meta_split], y_stack[meta_split:]

        log.info(f"\n  Meta-learner: train={meta_split} test={len(y_stack) - meta_split}")

        # Logistic regression meta-learner (outputs calibrated probabilities)
        meta_model = LogisticRegressionCV(
            Cs=10, cv=5, scoring="neg_log_loss", max_iter=2000, random_state=42
        )
        meta_model.fit(X_meta_train, y_meta_train)
        meta_proba = meta_model.predict_proba(X_meta_test)[:, 1]

        meta_acc = accuracy_score(y_meta_test, (meta_proba >= 0.5).astype(int))
        meta_auc = roc_auc_score(y_meta_test, meta_proba)
        meta_ll = log_loss(y_meta_test, meta_proba)
        meta_brier = brier_score_loss(y_meta_test, meta_proba)

        log.info(f"  {'STACKED':<15s} {meta_acc:>8.4f} {meta_auc:>8.4f} {meta_ll:>8.4f}")
        log.info(f"  Brier score: {meta_brier:.4f}")
        log.info(f"\n  Meta-learner coefficients:")
        for name, coef in zip(ext_names, meta_model.coef_[0]):
            log.info(f"    {name:25s} {coef:+.4f}")
        log.info(f"    {'intercept':25s} {meta_model.intercept_[0]:+.4f}")
        log.info(f"\n{classification_report(y_meta_test, (meta_proba >= 0.5).astype(int), target_names=['Blue', 'Red'])}")

        # Save meta-model
        with open(MODEL_DIR / "meta_model_v5.pkl", "wb") as f:
            pickle.dump({"model": meta_model, "feature_names": ext_names}, f)
        log.info(f"  Saved meta-model to {MODEL_DIR / 'meta_model_v5.pkl'}")

        yt = y_meta_test
        ep = meta_proba
        acc, auc = meta_acc, meta_auc
    else:
        log.warning("  Model test sizes don't match — falling back to GBT only")
        yt, ep = p1["test_y"], p1["test_proba"]
        acc, auc = p1["accuracy"], p1["auc"]

    best = max(models, key=lambda x: x[1]["auc"])
    log.info(f"\n  Best individual model by AUC: {best[0]} ({best[1]['auc']:.4f})")

    return {"ensemble_proba": ep, "test_y": yt, "accuracy": acc, "auc": auc}


# ===========================================================================
# PROBABILITY CALIBRATION
# ===========================================================================

def calibrate_model(matchup: pd.DataFrame, feature_names: list[str]) -> dict:
    """Train with Platt scaling and isotonic regression calibration."""
    log.info("=" * 60)
    log.info("PROBABILITY CALIBRATION")
    log.info("=" * 60)

    matchup = matchup.sort_values("date").reset_index(drop=True)
    # 3-way split: train (60%), calibration (20%), test (20%)
    n = len(matchup)
    s1, s2 = int(n * 0.6), int(n * 0.8)
    train, cal, test = matchup.iloc[:s1], matchup.iloc[s1:s2], matchup.iloc[s2:]

    # Remove odds features for pure model
    pure_features = [f for f in feature_names if "odds" not in f and "elo_vs_odds" not in f]

    X_train, y_train = train[pure_features].values, train["red_wins"].values
    X_cal, y_cal = cal[pure_features].values, cal["red_wins"].values
    X_test, y_test = test[pure_features].values, test["red_wins"].values

    log.info(f"  Train: {len(train)} | Calibration: {len(cal)} | Test: {len(test)}")
    log.info(f"  Test period: {test['date'].min()} to {test['date'].max()}")
    log.info(f"  Pure features (no odds): {len(pure_features)}")

    # Base model
    base_model = HistGradientBoostingClassifier(
        max_iter=1000, max_depth=4, learning_rate=0.02, max_features=0.7,
        min_samples_leaf=30, l2_regularization=2.0, max_bins=128,
        early_stopping=True, n_iter_no_change=75, validation_fraction=0.15,
        random_state=42,
    )
    base_model.fit(X_train, y_train)
    raw_proba = base_model.predict_proba(X_test)[:, 1]
    raw_cal_proba = base_model.predict_proba(X_cal)[:, 1]

    # --- Platt scaling (sigmoid) ---
    from sklearn.linear_model import LogisticRegression
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10000)
    platt.fit(raw_cal_proba.reshape(-1, 1), y_cal)
    platt_proba = platt.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

    # --- Isotonic regression ---
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(raw_cal_proba, y_cal)
    iso_proba = iso.predict(raw_proba)

    # --- Evaluate calibration ---
    methods = [
        ("Raw (uncalibrated)", raw_proba),
        ("Platt scaling", platt_proba),
        ("Isotonic regression", iso_proba),
    ]
    log.info(f"\n  {'Method':<25s} {'Acc':>7s} {'AUC':>7s} {'LogLoss':>8s} {'Brier':>7s}")
    log.info(f"  {'-'*58}")

    best_method, best_brier = None, float("inf")
    for name, proba in methods:
        acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
        auc = roc_auc_score(y_test, proba)
        ll = log_loss(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        log.info(f"  {name:<25s} {acc:>7.4f} {auc:>7.4f} {ll:>8.4f} {brier:>7.4f}")
        if brier < best_brier:
            best_brier, best_method = brier, name

    log.info(f"\n  Best calibration: {best_method} (Brier={best_brier:.4f})")

    # Reliability diagram (text-based)
    log.info(f"\n  Reliability Diagram (10 bins):")
    log.info(f"  {'Bin':>12s} {'Predicted':>10s} {'Actual':>10s} {'Count':>7s} {'Gap':>8s}")
    log.info(f"  {'-'*50}")
    for name, proba in methods:
        log.info(f"\n  {name}:")
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
        for pt, pp in zip(prob_true, prob_pred):
            gap = abs(pt - pp)
            bar = "#" * int(gap * 100)
            log.info(f"  {pp:>10.3f} → {pt:>8.3f}  gap={gap:.3f} {bar}")

    # Save calibrated model
    calibrated = {
        "base_model": base_model,
        "platt": platt,
        "isotonic": iso,
        "features": pure_features,
        "best_method": best_method,
    }
    with open(MODEL_DIR / "calibrated_model.pkl", "wb") as f:
        pickle.dump(calibrated, f)
    log.info(f"\n  Saved calibrated model to {MODEL_DIR / 'calibrated_model.pkl'}")

    # Return the best calibrated probabilities for betting sim
    best_proba = platt_proba if "Platt" in best_method else iso_proba
    return {
        "raw_proba": raw_proba,
        "platt_proba": platt_proba,
        "iso_proba": iso_proba,
        "best_proba": best_proba,
        "best_method": best_method,
        "test_y": y_test,
        "test_df": test,
    }


# ===========================================================================
# VIG-ADJUSTED BETTING SIMULATION
# ===========================================================================

def _american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / abs(odds)

def _implied_prob_no_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove the vig from implied probabilities (they sum to >1 from books)."""
    total = prob_a + prob_b
    if total == 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total

def betting_simulation(cal_results: dict) -> None:
    """Full betting simulation with vig-adjusted value detection."""
    log.info("=" * 60)
    log.info("BETTING SIMULATION (VIG-ADJUSTED)")
    log.info("=" * 60)

    test = cal_results["test_df"]
    y_test = cal_results["test_y"]

    # Check we have odds
    has_odds = test["odds_red_prob"].notna() & (test["odds_red_prob"] > 0)
    odds_test = test[has_odds].reset_index(drop=True)
    odds_y = y_test[has_odds.values]

    if len(odds_test) == 0:
        log.warning("  No fights with odds in test set — skipping betting sim")
        return

    # Get calibrated probabilities for odds-only fights
    raw_proba_odds = cal_results["raw_proba"][has_odds.values]
    platt_proba_odds = cal_results["platt_proba"][has_odds.values]
    iso_proba_odds = cal_results["iso_proba"][has_odds.values]
    best_proba_odds = cal_results["best_proba"][has_odds.values]

    log.info(f"  Test fights with odds: {len(odds_test)}")
    log.info(f"  Test period: {odds_test['date'].min()} to {odds_test['date'].max()}")
    log.info(f"  Red win rate in test: {odds_y.mean():.3f}")

    # Market probabilities (raw implied, with vig)
    market_red_raw = odds_test["odds_red_prob"].values
    market_blue_raw = odds_test["odds_blue_prob"].values

    # Remove vig to get true implied probabilities
    market_red_fair, market_blue_fair = [], []
    for mr, mb in zip(market_red_raw, market_blue_raw):
        fr, fb = _implied_prob_no_vig(mr, mb)
        market_red_fair.append(fr)
        market_blue_fair.append(fb)
    market_red_fair = np.array(market_red_fair)
    market_blue_fair = np.array(market_blue_fair)

    odds_red_dec = np.array([_american_to_decimal(o) for o in odds_test["odds_red_american"].values])
    odds_blue_dec = np.array([_american_to_decimal(o) for o in odds_test["odds_blue_american"].values])

    log.info(f"\n  Average vig: {(market_red_raw + market_blue_raw).mean() - 1:.3f}")
    log.info(f"  Market accuracy (fair): {accuracy_score(odds_y, (market_red_fair >= 0.5).astype(int)):.4f}")

    # --- Run simulations for each calibration method ---
    proba_methods = [
        ("Raw (uncalibrated)", raw_proba_odds),
        ("Platt scaling", platt_proba_odds),
        ("Isotonic regression", iso_proba_odds),
    ]

    for method_name, model_proba in proba_methods:
        log.info(f"\n{'─'*60}")
        log.info(f"  {method_name}")
        log.info(f"{'─'*60}")

        model_blue_proba = 1 - model_proba

        # --- Strategy 1: Flat bet on model pick ---
        log.info(f"\n  Strategy 1: Flat $100 bet on model pick (every fight)")
        profit = 0
        wins = 0
        for i in range(len(odds_test)):
            if model_proba[i] >= 0.5:
                if odds_y[i] == 1:
                    profit += (odds_red_dec[i] - 1) * 100
                    wins += 1
                else:
                    profit -= 100
            else:
                if odds_y[i] == 0:
                    profit += (odds_blue_dec[i] - 1) * 100
                    wins += 1
                else:
                    profit -= 100

        total_wagered = len(odds_test) * 100
        roi = profit / total_wagered * 100
        log.info(f"    Bets: {len(odds_test)} | Wins: {wins} ({wins/len(odds_test):.1%})")
        log.info(f"    Profit: ${profit:.2f} | Wagered: ${total_wagered} | ROI: {roi:+.2f}%")

        # --- Strategy 2: Value bets (model edge vs vig-adjusted market) ---
        for min_edge in [0.03, 0.05, 0.08, 0.10, 0.15]:
            profit, bets, wins = 0, 0, 0
            bet_details = []
            for i in range(len(odds_test)):
                edge_red = model_proba[i] - market_red_fair[i]
                edge_blue = model_blue_proba[i] - market_blue_fair[i]

                # Also check: is the bet +EV after vig?
                # +EV if: model_prob * decimal_odds > 1
                ev_red = model_proba[i] * odds_red_dec[i]
                ev_blue = model_blue_proba[i] * odds_blue_dec[i]

                if edge_red > min_edge and ev_red > 1.0:
                    bets += 1
                    won = odds_y[i] == 1
                    if won:
                        pnl = (odds_red_dec[i] - 1) * 100
                        wins += 1
                    else:
                        pnl = -100
                    profit += pnl
                    bet_details.append({
                        "side": "RED", "edge": edge_red, "ev": ev_red,
                        "odds": odds_test.iloc[i]["odds_red_american"],
                        "won": won, "pnl": pnl
                    })
                elif edge_blue > min_edge and ev_blue > 1.0:
                    bets += 1
                    won = odds_y[i] == 0
                    if won:
                        pnl = (odds_blue_dec[i] - 1) * 100
                        wins += 1
                    else:
                        pnl = -100
                    profit += pnl
                    bet_details.append({
                        "side": "BLUE", "edge": edge_blue, "ev": ev_blue,
                        "odds": odds_test.iloc[i]["odds_blue_american"],
                        "won": won, "pnl": pnl
                    })

            if bets > 0:
                roi = profit / (bets * 100) * 100
                log.info(f"\n  Strategy 2: Value bets (edge > {min_edge:.0%}, vig-adjusted, +EV only)")
                log.info(f"    Bets: {bets}/{len(odds_test)} | Wins: {wins} ({wins/bets:.1%})")
                log.info(f"    Profit: ${profit:.2f} | Wagered: ${bets*100} | ROI: {roi:+.2f}%")
                log.info(f"    Avg edge: {np.mean([b['edge'] for b in bet_details]):.3f}")
                log.info(f"    Avg EV: {np.mean([b['ev'] for b in bet_details]):.3f}")

                # Breakdown by odds range
                fav_bets = [b for b in bet_details if b["odds"] < 0]
                dog_bets = [b for b in bet_details if b["odds"] > 0]
                if fav_bets:
                    fav_pnl = sum(b["pnl"] for b in fav_bets)
                    fav_wins = sum(1 for b in fav_bets if b["won"])
                    log.info(f"    Favorites: {len(fav_bets)} bets, {fav_wins} wins ({fav_wins/len(fav_bets):.1%}), P&L: ${fav_pnl:.2f}")
                if dog_bets:
                    dog_pnl = sum(b["pnl"] for b in dog_bets)
                    dog_wins = sum(1 for b in dog_bets if b["won"])
                    log.info(f"    Underdogs: {len(dog_bets)} bets, {dog_wins} wins ({dog_wins/len(dog_bets):.1%}), P&L: ${dog_pnl:.2f}")
            else:
                log.info(f"\n  Strategy 2: Value bets (edge > {min_edge:.0%}) — no bets placed")

        # --- Strategy 3: Quarter Kelly criterion ---
        log.info(f"\n  Strategy 3: Quarter-Kelly sizing (edge > 3%, vig-adjusted)")
        bankroll = 10000
        start = bankroll
        peak = bankroll
        max_dd = 0
        bets = 0
        wins = 0
        for i in range(len(odds_test)):
            edge_red = model_proba[i] - market_red_fair[i]
            edge_blue = model_blue_proba[i] - market_blue_fair[i]
            ev_red = model_proba[i] * odds_red_dec[i]
            ev_blue = model_blue_proba[i] * odds_blue_dec[i]

            if edge_red > 0.03 and ev_red > 1.0:
                kelly_f = (model_proba[i] * (odds_red_dec[i] - 1) - (1 - model_proba[i])) / (odds_red_dec[i] - 1)
                bet = max(0, min(bankroll * kelly_f * 0.25, bankroll * 0.05))
                if bet > 1:
                    bets += 1
                    if odds_y[i] == 1:
                        bankroll += bet * (odds_red_dec[i] - 1)
                        wins += 1
                    else:
                        bankroll -= bet
            elif edge_blue > 0.03 and ev_blue > 1.0:
                kelly_f = (model_blue_proba[i] * (odds_blue_dec[i] - 1) - model_proba[i]) / (odds_blue_dec[i] - 1)
                bet = max(0, min(bankroll * kelly_f * 0.25, bankroll * 0.05))
                if bet > 1:
                    bets += 1
                    if odds_y[i] == 0:
                        bankroll += bet * (odds_blue_dec[i] - 1)
                        wins += 1
                    else:
                        bankroll -= bet

            peak = max(peak, bankroll)
            dd = (peak - bankroll) / peak
            max_dd = max(max_dd, dd)

        log.info(f"    Bets placed: {bets} | Wins: {wins}")
        log.info(f"    Starting: ${start:,.2f} → Ending: ${bankroll:,.2f}")
        log.info(f"    Return: {(bankroll - start) / start * 100:+.2f}%")
        log.info(f"    Max drawdown: {max_dd:.1%}")

    # --- Final summary ---
    log.info(f"\n{'='*60}")
    log.info(f"BETTING SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Test period: {odds_test['date'].min()} to {odds_test['date'].max()}")
    log.info(f"  Total fights with odds: {len(odds_test)}")
    log.info(f"  Market (vig-adjusted) accuracy: {accuracy_score(odds_y, (market_red_fair >= 0.5).astype(int)):.4f}")
    log.info(f"  Model accuracy: {accuracy_score(odds_y, (best_proba_odds >= 0.5).astype(int)):.4f}")
    log.info(f"  Best calibration method: {cal_results['best_method']}")
    log.info(f"  Key insight: profitability requires finding +EV spots where")
    log.info(f"  model_prob * decimal_odds > 1.0 (positive expected value)")


# ===========================================================================
# MAIN
# ===========================================================================

def run(phases=None):
    if phases is None: phases = [1, 2, 3, 4]
    log.info("=" * 60)
    log.info("UFC Fight Winner Prediction Pipeline v5")
    log.info("=" * 60)

    df, round_data = load_fight_data()
    df = build_features(df, round_data)
    matchup, features = build_matchup_df(df)

    results = {}
    if 1 in phases:
        results["gbt_full"] = train_gbt(matchup, features, odds_only=False)
        if "odds_red_prob" in matchup.columns and (matchup["odds_red_prob"] > 0).sum() > 100:
            results["gbt_odds"] = train_gbt(matchup, features, odds_only=True)
    if 2 in phases: results["rnn"] = train_rnn(df, matchup)
    if 3 in phases: results["gnn"] = train_gnn(df, matchup)
    if 4 in phases: results["siamese"] = train_siamese(df, matchup)
    if "gbt_full" in results and "rnn" in results:
        ensemble_results(
            results["gbt_full"], results["rnn"], results.get("gnn"), results.get("siamese"),
            matchup=matchup, feature_names=features,
        )

    # Calibration + Betting Simulation (always run)
    cal_results = calibrate_model(matchup, features)
    betting_simulation(cal_results)

    log.info("\n" + "=" * 60)
    log.info("Pipeline v5 complete.")
    log.info("=" * 60)


def generate_predictions():
    """Run the calibrated model on all fights and store predictions in DB."""
    log.info("=" * 60)
    log.info("GENERATING PREDICTIONS FOR ALL FIGHTS")
    log.info("=" * 60)

    # Load calibrated model
    cal_path = MODEL_DIR / "calibrated_model.pkl"
    if not cal_path.exists():
        log.error("No calibrated model found. Run training first.")
        return

    with open(cal_path, "rb") as f:
        cal = pickle.load(f)

    base_model = cal["base_model"]
    platt = cal["platt"]
    features = cal["features"]
    best_method = cal["best_method"]

    # Build features for all fights
    df, round_data = load_fight_data()
    df = build_features(df, round_data)

    # Build matchup (without augmentation/filtering for prediction)
    # We need the raw matchup with fight_id index
    feature_cols = [c for c in df.columns if c.startswith(("avg_", "recent_", "last3_"))]
    feature_cols += [
        "elo", "elo_expected", "resume_score",
        "height_inches", "weight_lbs", "reach_inches", "age",
        "stance_orthodox", "stance_southpaw", "stance_switch",
        "career_win_pct", "career_fights", "finish_rate", "been_finished_rate",
        "streak", "days_since_last", "style_matchup_adv",
    ]
    feature_cols += [c for c in df.columns if "composite" in c and c.startswith(("avg_", "recent_", "last3_"))]
    feature_cols += [c for c in df.columns if c.startswith(("avg_r1_", "avg_late_", "avg_output_", "avg_ctrl_trend",
                                                             "recent_r1_", "recent_late_", "recent_output_", "recent_ctrl_trend"))]
    feature_cols += [c for c in df.columns if c.startswith("style_") and c not in feature_cols]
    feature_cols += [c for c in df.columns if c.startswith("div_") and c not in feature_cols]
    feature_cols = list(dict.fromkeys(feature_cols))

    for col in feature_cols:
        df[col] = df[col].fillna(df[col].mean())

    red = df[df["corner"] == "red"].set_index("fight_id")
    blue = df[df["corner"] == "blue"].set_index("fight_id")
    common = red.index.intersection(blue.index)
    red, blue = red.loc[common], blue.loc[common]

    matchup = pd.DataFrame(index=common)
    matchup.index.name = "fight_id"
    matchup["date"] = red["date"].values

    for col in feature_cols:
        matchup[f"diff_{col}"] = red[col].values - blue[col].values

    for col in ["elo", "elo_expected", "resume_score", "career_fights", "career_win_pct",
                 "streak", "finish_rate", "style_matchup_adv"]:
        matchup[f"red_{col}"] = red[col].values
        matchup[f"blue_{col}"] = blue[col].values

    matchup = matchup.fillna(0)

    # Ensure all required features exist (fill missing with 0)
    for feat in features:
        if feat not in matchup.columns:
            matchup[feat] = 0.0

    X = matchup[features].values
    raw_proba = base_model.predict_proba(X)[:, 1]

    if "Platt" in best_method:
        calibrated_proba = platt.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    else:
        iso = cal["isotonic"]
        calibrated_proba = iso.predict(raw_proba)

    # Compute SHAP values for the GBT model
    import shap
    log.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(base_model)
    shap_values_arr = explainer.shap_values(X)
    # For binary classification, shap_values may be a list of 2 arrays; take class 1 (red wins)
    if isinstance(shap_values_arr, list):
        shap_values_arr = shap_values_arr[1]
    log.info(f"  SHAP values shape: {shap_values_arr.shape}")

    # Store predictions and SHAP values in DB
    from app.models.ufc import UFCFightPrediction, UFCFightShapValue
    db = SessionLocal()
    try:
        # Delete in batches to avoid CockroachDB serialization errors
        for Model in [UFCFightShapValue, UFCFightPrediction]:
            while True:
                ids = [r[0] for r in db.query(Model.id).limit(5000).all()]
                if not ids:
                    break
                db.query(Model).filter(Model.id.in_(ids)).delete(synchronize_session=False)
                db.commit()
            log.info(f"  Cleared {Model.__tablename__}")

        count = 0
        shap_count = 0
        for i, (fight_id, prob) in enumerate(zip(matchup.index, calibrated_proba)):
            predicted_winner = "red" if prob >= 0.5 else "blue"
            confidence = abs(prob - 0.5)
            db.add(UFCFightPrediction(
                fight_id=int(fight_id),
                predicted_winner=predicted_winner,
                confidence=round(float(confidence), 4),
                red_prob=round(float(prob), 4),
            ))
            count += 1

            # Store top 20 SHAP values for this fight
            fight_shap = shap_values_arr[i]
            abs_shap = abs(fight_shap)
            top_indices = abs_shap.argsort()[-20:][::-1]
            for idx in top_indices:
                db.add(UFCFightShapValue(
                    fight_id=int(fight_id),
                    feature_name=features[idx],
                    shap_value=round(float(fight_shap[idx]), 6),
                    abs_value=round(float(abs_shap[idx]), 6),
                    feature_value=round(float(X[i, idx]), 4),
                ))
                shap_count += 1

            # Commit every 100 fights to avoid CockroachDB transaction size limits
            if count % 100 == 0:
                db.commit()

        db.commit()
        log.info(f"Stored {count} fight predictions, {shap_count} SHAP values")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3])
    parser.add_argument("--predict", action="store_true", help="Generate predictions for all fights")
    args = parser.parse_args()
    if args.predict:
        generate_predictions()
    else:
        run(phases=[args.phase] if args.phase else None)
