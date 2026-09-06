"""
UFC Fight Winner Prediction — Phase 1 Model Pipeline (v5)

Phase 1: Gradient Boosting on engineered features

v5 improvements:
- Composite features (striking/grappling/defense indexes)
- Round-by-round profiles (fade detection, fast starter, late finisher)
- CLV tracking for bet quality evaluation
- SHAP-based feature selection

Usage:
    python -m app.services.model              # train + evaluate
    python -m app.services.model --phase 1    # run Phase 1 (GBT)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from venn_abers import VennAbers
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from app.database import SessionLocal
from app.services.ufc.market_anchor import MarketAnchor, devig, logit
from app.services.ufc.glicko_service import run_glicko_inmemory
from app.services.ufc.simulator import (
    HazardRateModel, attach_fit_targets, build_simulator_frame, simulate,
)


def _decorrelated_model():
    """Import DecorrelatedModel lazily, because it pulls in torch.

    Deliberately NOT a module-level import. torch is a heavy dependency that the
    deployment and CI environments may not carry, and importing it at module scope makes
    `import app.services.ufc.model` fail outright without it -- which would take down the
    nightly prediction job and every caller that only needs the GBT path.

    Callers that can degrade gracefully should catch ImportError; `generate_predictions`
    falls back to the GBT rather than failing the run.
    """
    from app.services.ufc.decorrelated import DecorrelatedModel
    return DecorrelatedModel
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFightStats

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models" / "ufc" / "h2h"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        # Append, don't truncate. mode="w" meant a later --predict run silently
        # destroyed the training log that produced the published metrics, leaving
        # every headline number unreproducible.
        logging.FileHandler(MODEL_DIR / "training.log", mode="a"),
    ],
)
log = logging.getLogger("model")


# ===========================================================================
# DATA LOADING
# ===========================================================================

ID_COLUMNS = (
    "fight_id", "red_fighter_id", "blue_fighter_id", "winner_id", "stats_fighter_id",
)


def _coerce_id_columns(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Force ID columns to exact Python ints (object dtype), rebuilt from source rows.

    These IDs are ~19-digit snowflakes, far beyond float64's 2**53 exact range. Any
    column containing NULLs — `winner_id` does, for draws and no-contests — is inferred
    by pandas as float64, which silently rounds every ID.

    That rounding is invisible to vectorized comparisons (`df.a == df.b` casts int64 to
    float64, so both sides round identically and appear equal) but fatal inside
    `iterrows()`/`itertuples()`, which yield Python scalars, and Python compares
    int to float EXACTLY:

        1180969949270474753 == float(1180969949270474753)   -> False   (Python)
        np.int64(...)       == np.float64(...)              -> True    (NumPy)

    Every per-fight loop in this module uses itertuples/iterrows, so the winner check
    never fired: Elo stayed 1500.0 for every fighter, elo_expected 0.5, resume_score
    0.0 — all dead constants feeding the model as if they were signal.

    Rebuilding from `rows` rather than casting the DataFrame column matters: by the
    time the column exists as float64 the precision is already gone, and int() would
    lock in the rounded value.
    """
    for col in ID_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = pd.Series(
            [(None if r.get(col) is None else int(r[col])) for r in rows],
            index=df.index, dtype=object,
        )
    return df


def _assert_ids_exact(df: pd.DataFrame) -> None:
    """Guard against a silent regression to float64 IDs."""
    for col in ID_COLUMNS:
        if col in df.columns and df[col].dtype != object:
            raise TypeError(
                f"{col} has dtype {df[col].dtype}, expected object. Large IDs stored as "
                "float lose precision and make every winner comparison fail silently. "
                "See _coerce_id_columns()."
            )


def load_fight_data() -> pd.DataFrame:
    log.info("Loading data from database...")
    db = SessionLocal()

    fighters_q = db.query(UFCFighter).all()
    fighters = {
        f.id: {
            "name": f"{f.first_name} {f.last_name}",
            "height": f.height, "weight": f.weight,
            "reach": f.reach, "stance": f.stance, "dob": f.dob,
            "lifetime_wins": f.wins or 0, "lifetime_losses": f.losses or 0,
        }
        for f in fighters_q
    }
    log.info(f"  Loaded {len(fighters)} fighters")

    fights_q = (
        db.query(UFCFight, UFCFightStats)
        .join(UFCFightStats, UFCFight.id == UFCFightStats.fight_id)
        .filter(UFCFightStats.round_number == 0)
        # Exclude upcoming/unplayed fights (placeholder rows with all-zero stats)
        .filter(
            (UFCFightStats.sig_str_landed > 0)
            | (UFCFightStats.sig_str_attempted > 0)
            | (UFCFightStats.td_attempted > 0)
        )
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
            # time_format is the only source of the 3-vs-5-round flag; it was previously
            # read by glicko_service but never reached the winner model.
            "time_format": fight.time_format,
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
            "lifetime_wins": f.get("lifetime_wins", 0),
            "lifetime_losses": f.get("lifetime_losses", 0),
        })

    db.close()
    df = pd.DataFrame(rows)
    df = _coerce_id_columns(df, rows)
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
    if round_rows:
        round_data = _coerce_id_columns(round_data, round_rows)
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
    """Map verbose weight class strings to canonical divisions.

    Splits men's from women's: previously "Women's Flyweight" and "Flyweight" mapped to
    the same bucket, merging two divisions with very different base rates and physical
    profiles.

    Intentionally differs from glicko_service._classify_weight_class(), which pools
    women's featherweight into w_bantamweight and maps catchweight to "unknown". Those
    are sensible for rating-pool sample size; here we want the true division, and
    catchweight must stay distinct because it feeds the weight-class movement features.
    """
    if not isinstance(wc, str):
        return "unknown"
    wc = wc.lower()
    w = "w_" if "women" in wc else ""
    # Order matters: check catchweight first (a catchweight bout often still names a
    # division), and "light heavyweight" before "heavyweight".
    if "catch" in wc or "open" in wc: return "catchweight"
    if "strawweight" in wc: return f"{w}strawweight"
    if "flyweight" in wc: return f"{w}flyweight"
    if "bantamweight" in wc: return f"{w}bantamweight"
    if "featherweight" in wc: return f"{w}featherweight"
    if "light heavyweight" in wc or "light_heavyweight" in wc: return "light_heavyweight"
    if "lightweight" in wc: return f"{w}lightweight"
    if "welterweight" in wc: return f"{w}welterweight"
    if "middleweight" in wc: return "middleweight"
    if "heavyweight" in wc: return "heavyweight"
    return "unknown"


def _is_title_bout(wc: str | None) -> bool:
    """Title fights are 5 rounds, higher stakes, and better-scouted matchups.
    Derived the same way glicko_service does, but never reached the model before."""
    return isinstance(wc, str) and "title" in wc.lower()


def _round_lengths(time_format: str | None) -> list[int]:
    """Per-round minutes from `time_format`.

    The stored value is the ROUND-LENGTH string, not a round count: '5-5-5' is a
    three-round fight of five minutes each, '5-5-5-5-5' is five rounds. So the number
    of dash-separated segments is the round count. scraper._compute_fight_time() parses
    it the same way, which is what makes fight_time_seconds correct.

    Legacy formats exist and are not parseable this way ('No Time Limit', bare '20').
    """
    if not isinstance(time_format, str):
        return []
    parts = [p.strip() for p in time_format.split("-")]
    if not parts or not all(p.isdigit() for p in parts):
        return []
    return [int(p) for p in parts]


def _scheduled_rounds(time_format: str | None) -> float:
    lengths = _round_lengths(time_format)
    return float(len(lengths)) if lengths else float("nan")


def _scheduled_minutes(time_format: str | None) -> float:
    """Total scheduled length — separates a 3x5 from a legacy single 20-minute round,
    which the round count alone cannot."""
    lengths = _round_lengths(time_format)
    return float(sum(lengths)) if lengths else float("nan")


def _is_five_round(time_format: str | None) -> bool:
    r = _scheduled_rounds(time_format)
    return bool(r >= 5) if r == r else False  # r != r detects NaN


# ===========================================================================
# FIGHTER STYLE CLUSTERING
# ===========================================================================

def compute_fighter_styles(
    df: pd.DataFrame, n_clusters: int = 6, cutoff_date=None
) -> tuple[np.ndarray, tuple]:
    """
    Assign each fighter-fight a style archetype from that fighter's PRE-FIGHT profile.

    Clusters the expanding-mean `avg_*` columns, which are already `.shift(1)`-ed, so a
    row's style reflects only the fighter's prior fights. Career means over the raw
    per-fight columns would fold in the outcome of the very fight being predicted, and
    would also backdate a fighter's late-career identity onto their early fights.

    `cutoff_date` restricts the scaler/KMeans fit to rows strictly before that date, so
    no post-cutoff distribution reaches the fit. None fits on everything (unsupervised,
    but callers doing strict walk-forward should pass the fold boundary).

    Returns: per-row cluster labels, (scaler, kmeans, medians) for reuse at serve time.
    """
    log.info(f"  Computing fighter style clusters (k={n_clusters})...")

    style_features = [
        "sig_str_landed_per5", "sig_str_acc", "td_landed_per5", "td_acc",
        "sub_att_per5", "ctrl_per5", "kd_per5",
        "head_target_pct", "body_target_pct", "leg_target_pct",
        "distance_landed_per5", "clinch_landed_per5", "ground_landed_per5",
    ]
    avg_cols = [f"avg_{c}" for c in style_features]

    missing = [c for c in avg_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"compute_fighter_styles needs pre-fight rolling columns; missing: {missing}. "
            "It must be called after the rolling-average block in build_features()."
        )

    X = df[avg_cols].astype(float)

    if cutoff_date is None:
        fit_mask = np.ones(len(df), dtype=bool)
    else:
        fit_mask = (pd.to_datetime(df["date"]) < pd.Timestamp(cutoff_date)).values
        if fit_mask.sum() < n_clusters * 10:
            log.warning(
                f"    Only {fit_mask.sum()} rows before {cutoff_date}; fitting styles on all rows"
            )
            fit_mask = np.ones(len(df), dtype=bool)

    # Debut fights have no prior history — impute from the fit slice only
    medians = X[fit_mask].median()
    X = X.fillna(medians).fillna(0.0)

    scaler = StandardScaler().fit(X[fit_mask].values)
    X_scaled = scaler.transform(X.values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    kmeans.fit(X_scaled[fit_mask])
    labels = kmeans.predict(X_scaled)

    log.info(f"    Fit on {int(fit_mask.sum())} rows, assigned {len(labels)} fighter-fights")

    # Log cluster descriptions
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=style_features,
    )
    for i in range(n_clusters):
        c = centers.iloc[i]
        n_rows = (labels == i).sum()
        top_trait = c.idxmax()
        log.info(
            f"    Style {i} ({n_rows} fighter-fights): "
            f"SigStr/5={c['sig_str_landed_per5']:.1f} TD/5={c['td_landed_per5']:.1f} "
            f"Sub/5={c['sub_att_per5']:.1f} Ctrl/5={c['ctrl_per5']:.1f} "
            f"GndStr/5={c['ground_landed_per5']:.1f} top={top_trait}"
        )

    return labels, (scaler, kmeans, medians)


def compute_style_matchup_features(
    df: pd.DataFrame, style_at_fight: dict, n_clusters: int, prior_weight: float = 5.0
) -> dict:
    """
    Style-vs-style win rate, accumulated expanding-in-time.

    For each fight we record the style matchup rate as it stood *before* that fight,
    then fold the result in — the same read-then-update ordering the Elo loop uses.
    Building the matrix over the full dataset and then letting each fight read its own
    cell puts the predicted outcome inside its own feature, which is label leakage.

    Rates are smoothed toward 0.5 by `prior_weight` pseudo-fights so that sparse early
    cells don't emit hard 0.0/1.0 values off a single observation.

    Returns: {(fight_id, fighter_id): win_rate_of_my_style_vs_theirs}
    """
    wins = np.zeros((n_clusters, n_clusters))
    total = np.zeros((n_clusters, n_clusters))
    adv_at_fight = {}

    def _rate(w, t):
        return (w + 0.5 * prior_weight) / (t + prior_weight)

    fights_chrono = (
        df[["fight_id", "date", "red_fighter_id", "blue_fighter_id", "winner_id"]]
        .drop_duplicates("fight_id").sort_values("date")
    )

    for fight in fights_chrono.itertuples(index=False):
        red_style = style_at_fight.get((fight.fight_id, fight.red_fighter_id))
        blue_style = style_at_fight.get((fight.fight_id, fight.blue_fighter_id))
        if red_style is None or blue_style is None:
            continue

        # Snapshot pre-fight rates
        adv_at_fight[(fight.fight_id, fight.red_fighter_id)] = _rate(
            wins[red_style][blue_style], total[red_style][blue_style]
        )
        adv_at_fight[(fight.fight_id, fight.blue_fighter_id)] = _rate(
            wins[blue_style][red_style], total[blue_style][red_style]
        )

        # Then fold in this fight's result
        total[red_style][blue_style] += 1
        total[blue_style][red_style] += 1
        if fight.winner_id == fight.red_fighter_id:
            wins[red_style][blue_style] += 1
        elif fight.winner_id == fight.blue_fighter_id:
            wins[blue_style][red_style] += 1

    log.info(
        f"  Style matchup features computed expanding-in-time "
        f"({n_clusters}x{n_clusters}, {len(adv_at_fight)} fighter-fights)"
    )
    return adv_at_fight


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================

def build_features(
    df: pd.DataFrame, round_data: pd.DataFrame = None, style_cutoff_date=None,
    glicko_snapshots: dict | None = None,
) -> pd.DataFrame:
    log.info("Building features...")
    _assert_ids_exact(df)

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
    # Fight-level context, previously computed only inside glicko_service and never
    # exposed to the model. Both are known before the fight.
    df["is_title_fight"] = df["weight_class"].apply(_is_title_bout).astype(float)
    df["is_five_round"] = df["time_format"].apply(_is_five_round).astype(float)
    df["scheduled_rounds"] = df["time_format"].apply(_scheduled_rounds)
    df["scheduled_minutes"] = df["time_format"].apply(_scheduled_minutes)

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

    # The sort above invalidated `opp_map`/`opp_idx`, which were built against the
    # pre-sort index. Rebuild them here so every opponent lookup below points at the
    # actual opponent rather than an arbitrary row.
    opp_map = {}
    for _, group in df.groupby("fight_id"):
        if len(group) != 2:
            continue
        rows = group.index.tolist()
        opp_map[rows[0]] = rows[1]
        opp_map[rows[1]] = rows[0]

    opp_idx = [opp_map.get(i, i) for i in df.index]

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

    # Initialize resume = Elo-based
    for fid in elo:
        resume[fid] = (elo.get(fid, 1500) - 1500) / 400  # normalize around 0

    fights_chrono = (
        df[["fight_id", "date", "red_fighter_id", "blue_fighter_id", "winner_id", "method"]]
        .drop_duplicates("fight_id").sort_values("date")
    )
    fight_list = list(fights_chrono.itertuples(index=False))

    # NOTE: a 5-iteration PageRank-style refinement of `resume` used to run here. Its
    # output was written to a local that nothing ever read — `resume_running` below
    # starts empty and never consults it — so it was ~55k wasted fight iterations per
    # build with no effect on any feature. Removed.
    #
    # KNOWN DEFECT (not fixed here — it is a rating-design change, not a bug fix):
    # `resume_running` is a homogeneous linear recursion seeded at 0.0, so every
    # update is a combination of zeros and `resume_score`/`opp_resume` are identically
    # 0.0 for every fighter-fight. They are inert inputs, not signal. Giving the
    # recursion a non-zero source term (e.g. crediting wins by the opponent's Elo
    # rather than their resume) would make it meaningful.

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

    # --- Glicko multi-dimensional ratings ---
    try:
        from app.services.ufc.glicko_service import DIMENSIONS as GLICKO_DIMS

        if glicko_snapshots is not None:
            # Caller supplied freshly computed snapshots (e.g. run_glicko_inmemory()
            # after the newcomer-seed fix). Lets evaluation use corrected ratings
            # without overwriting the stored UFCGlickoSnapshot table.
            log.info(f"  Using {len(glicko_snapshots)} caller-supplied Glicko snapshots")
        else:
            log.info("  Loading Glicko rating snapshots from DB...")
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

        # Rating CONFIDENCE, not just the rating. Without these the model treats a
        # rating built on 40 rounds identically to one built on 2, even though 17.8% of
        # fighter-fight rows are debuts and 53.6% of fighters have <=3 career fights.
        # Present only when snapshots come from run_glicko_inmemory(); stored DB rows
        # predate them, so default to a "no information" sentinel.
        meta_defaults = {"_meta_sigma": 0.0, "_meta_rounds_seen": 0.0,
                         "_meta_fights_seen": 0.0, "_meta_days_since": -1.0}
        any_meta = any("_meta_sigma" in v for v in glicko_snapshots.values())
        for key, default in meta_defaults.items():
            df[f"glicko{key}"] = df.apply(
                lambda r, k=key, dflt=default: glicko_snapshots.get(
                    (r["fight_id"], r["stats_fighter_id"]), {}
                ).get(k, dflt),
                axis=1,
            )
        log.info(f"  Added {len(GLICKO_DIMS)} Glicko features"
                 + ("  + 4 confidence features" if any_meta
                    else "  (no confidence metadata in these snapshots)"))
    except Exception as e:
        log.warning(f"  Could not load Glicko features: {e}")
        from app.services.ufc.glicko_service import DIMENSIONS as GLICKO_DIMS
        for dim in GLICKO_DIMS:
            df[f"glicko_{dim}"] = 0.0
        for k, dflt in (("_meta_sigma", 0.0), ("_meta_rounds_seen", 0.0),
                        ("_meta_fights_seen", 0.0), ("_meta_days_since", -1.0)):
            df[f"glicko{k}"] = dflt

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

    # --- Pre-UFC record (honest debut prior) ---
    # A UFC debutant has no UFC history, which is the hole the leaked Glicko newcomer
    # seed was illegitimately filling with the fighter's future results. The legitimate
    # signal is their record BEFORE the UFC: lifetime minus every UFC fight in the
    # dataset. That quantity is fixed at debut and never moves, so it is safe at any
    # point in a career — unlike the raw lifetime record, which is a running total.
    #
    # This is a weak proxy. It is a bare W/L with no opponent quality, so it cannot
    # distinguish 15-0 against regional cans from 15-0 against future contenders.
    # Real strength here needs multi-promotion results; see docs/models/winner.md.
    # --- Age shape ---
    # `age` was previously a difference only, so a 22-vs-26 fight and a 38-vs-42 fight
    # were identical to the model. Absolute age now reaches it via WINNER_RAW_COLS;
    # these terms give it the shape of a career arc rather than a straight line.
    log.info("  Computing age curve features...")
    PEAK_LO, PEAK_HI = 28.0, 31.0
    age = df["age"].astype(float)
    df["age_sq"] = age ** 2
    df["years_past_peak"] = (age - PEAK_HI).clip(lower=0)
    df["years_to_peak"] = (PEAK_LO - age).clip(lower=0)

    # --- Layoff shape ---
    # Raw days is a poor scale: 30 vs 120 days matters enormously, 700 vs 800 barely at
    # all. Absolute layoff also now reaches the model via WINNER_RAW_COLS.
    dsl = df["days_since_last"].astype(float)
    df["log_days_since_last"] = np.log1p(dsl.clip(lower=0))
    df["is_long_layoff"] = (dsl > 548).astype(float)        # 18 months
    df["is_short_turnaround"] = (dsl < 60).astype(float)
    # The one direction the literature is consistent about is that quick turnarounds
    # hurt older fighters more; the physiological evidence is stronger than the
    # win-rate samples behind the "ring rust" narrative.
    df["age_x_log_layoff"] = age * df["log_days_since_last"]

    # --- Damage accumulation ---
    # Every existing durability signal is a RATE, so 25 wars and 5 quiet fights look
    # identical. These are cumulative and strictly pre-fight (expanding().sum().shift(1),
    # the same pattern as career_wins).
    log.info("  Computing damage accumulation...")
    df["head_absorbed"] = df.loc[opp_idx, "head_landed"].values
    df["kd_absorbed"] = df.loc[opp_idx, "kd"].values
    df["fight_minutes"] = df["fight_time_seconds"].astype(float) / 60.0
    df["ko_loss"] = ((df["lost"] == 1) & df["method"].str.contains("KO", na=False)).astype(int)

    for src, dest in [
        ("head_absorbed", "career_head_strikes_absorbed"),
        ("kd_absorbed", "career_knockdowns_absorbed"),
        ("fight_minutes", "career_fight_minutes"),
        ("ko_loss", "career_ko_losses"),
    ]:
        df[dest] = (
            df.groupby("stats_fighter_id")[src]
            .apply(lambda x: x.expanding().sum().shift(1).fillna(0))
            .reset_index(level=0, drop=True)
        )

    # Recency of the last knockout loss. `avg_was_finished` averages over a career and
    # cannot express "was knocked out last time out" — which is the form the literature's
    # OR=1.13 consecutive-KO finding actually takes.
    def _fights_since_ko_loss(flags: np.ndarray) -> list[float]:
        out, since = [], np.nan
        for f in flags:
            out.append(since)
            since = 0.0 if f == 1 else (since + 1 if since == since else np.nan)
        return out

    df["fights_since_ko_loss"] = (
        df.groupby("stats_fighter_id")["ko_loss"]
        .transform(lambda x: pd.Series(_fights_since_ko_loss(x.values), index=x.index))
    )
    df["damage_per_minute"] = _safe_divide(
        df["career_head_strikes_absorbed"].values,
        df["career_fight_minutes"].clip(lower=1).values,
    )

    # --- Weight-class movement ---
    # Tested directly against the market on this dataset: fighters changing division
    # were priced at 49.1% and won 49.8% (n=792) — no detectable edge, consistent with
    # the literature, where the often-cited win-rate drop is about what regression to
    # the mean predicts on its own. Built at explicit request; expectations low.
    log.info("  Computing weight-class movement...")
    DIVISION_ORDER = {
        "w_strawweight": 0, "w_flyweight": 1, "w_bantamweight": 2, "w_featherweight": 3,
        "flyweight": 10, "bantamweight": 11, "featherweight": 12, "lightweight": 13,
        "welterweight": 14, "middleweight": 15, "light_heavyweight": 16, "heavyweight": 17,
    }
    df["division_rank"] = df["division"].map(DIVISION_ORDER).astype(float)
    prev_div = df.groupby("stats_fighter_id")["division"].shift(1)
    prev_rank = df.groupby("stats_fighter_id")["division_rank"].shift(1)
    rank_delta = df["division_rank"] - prev_rank

    df["moved_up"] = (rank_delta > 0).astype(float)
    df["moved_down"] = (rank_delta < 0).astype(float)
    df["division_rank_delta"] = rank_delta.fillna(0.0)
    df["is_catchweight"] = (df["division"] == "catchweight").astype(float)
    df["changed_division"] = ((prev_div.notna()) & (df["division"] != prev_div)).astype(float)
    # Consecutive fights already made in this division — a proxy for being settled in it
    same = (df["division"] == prev_div)
    grp = (~same).cumsum()
    df["fights_in_current_division"] = df.groupby(["stats_fighter_id", grp]).cumcount().astype(float)
    def _cum_distinct(vals) -> list[float]:
        """Distinct divisions fought BEFORE each fight (appends before adding)."""
        out, seen = [], set()
        for v in vals:
            out.append(float(len(seen)))
            seen.add(v)
        return out

    df["divisions_fought_count"] = df.groupby("stats_fighter_id")["division"].transform(
        lambda s: pd.Series(_cum_distinct(s.values), index=s.index)
    )

    log.info("  Computing pre-UFC records...")
    ufc_w, ufc_l = {}, {}
    for fight in df[["fight_id", "red_fighter_id", "blue_fighter_id", "winner_id"]] \
            .drop_duplicates("fight_id").itertuples(index=False):
        if not fight.winner_id or pd.isna(fight.winner_id):
            continue
        for fid in (fight.red_fighter_id, fight.blue_fighter_id):
            if fid == fight.winner_id:
                ufc_w[fid] = ufc_w.get(fid, 0) + 1
            else:
                ufc_l[fid] = ufc_l.get(fid, 0) + 1

    ids = df["stats_fighter_id"]
    df["pre_ufc_wins"] = (
        df["lifetime_wins"].fillna(0) - ids.map(lambda i: ufc_w.get(i, 0))
    ).clip(lower=0)
    df["pre_ufc_losses"] = (
        df["lifetime_losses"].fillna(0) - ids.map(lambda i: ufc_l.get(i, 0))
    ).clip(lower=0)
    df["pre_ufc_fights"] = df["pre_ufc_wins"] + df["pre_ufc_losses"]
    df["pre_ufc_win_pct"] = _safe_divide(
        df["pre_ufc_wins"].values, df["pre_ufc_fights"].clip(lower=1).values
    )
    # Experience-weighted quality: 12-0 should outrank 2-0. Mirrors the Glicko seed
    # shape so the two agree rather than fight each other.
    df["pre_ufc_quality"] = (
        (df["pre_ufc_win_pct"] - 0.5) * 2 * (df["pre_ufc_fights"] / 20).clip(upper=1.0)
    )
    # How much of what we know about this fighter is UFC evidence vs pre-UFC hearsay.
    df["ufc_experience_share"] = _safe_divide(
        df["career_fights"].values,
        (df["career_fights"] + df["pre_ufc_fights"]).clip(lower=1).values,
    )

    # --- Skill decay: performance relative to age-expected ---
    # Built on PERFORMANCE, not win rate. Win rates conflate ability with matchmaking —
    # a declining champion's record is protected by opponent selection, so an aging
    # curve fitted to outcomes measures the UFC's booking policy as much as decline.
    #
    # For each fight we compare the fighter's recent output to the league-wide average
    # at that age, where the league average is accumulated expanding-in-time (only
    # fights strictly before this one). Same read-then-update ordering as Elo and the
    # style-matchup matrix.
    log.info("  Computing age-relative performance residuals...")
    # Each needs a `recent_*` rolling form, which restricts this to columns that went
    # through the rolling-average loop above.
    DECAY_METRICS = [
        ("sig_str_landed_per5", "output"),      # can they still produce
        ("sig_str_def", "defense"),             # can they still avoid damage
        ("opp_sig_str_landed_per5", "absorbed"),  # how much they now take
    ]
    fights_chrono_idx = df.sort_values("date").index

    for src, label in DECAY_METRICS:
        recent_col = f"recent_{src}"
        if recent_col not in df.columns or src not in df.columns:
            continue
        bucket_sum: dict[int, float] = {}
        bucket_n: dict[int, int] = {}
        residuals = np.full(len(df), np.nan)

        ages = df["age"].to_numpy(dtype=float)
        recents = df[recent_col].to_numpy(dtype=float)
        actuals = df[src].to_numpy(dtype=float)
        pos = {ix: i for i, ix in enumerate(df.index)}

        for ix in fights_chrono_idx:
            i = pos[ix]
            a = ages[i]
            if a != a:
                continue
            b = int(a)
            n = bucket_n.get(b, 0)
            if n >= 20 and recents[i] == recents[i]:
                residuals[i] = recents[i] - (bucket_sum[b] / n)
            # Fold this fight's realised performance in AFTER reading
            if actuals[i] == actuals[i]:
                bucket_sum[b] = bucket_sum.get(b, 0.0) + actuals[i]
                bucket_n[b] = n + 1

        df[f"age_resid_{label}"] = residuals

    # --- Fighter style clustering (per-fight, from pre-fight rolling profiles) ---
    N_STYLES = 6
    style_labels, style_artifacts = compute_fighter_styles(
        df, n_clusters=N_STYLES, cutoff_date=style_cutoff_date
    )
    df["fighter_style"] = style_labels.astype(int)

    style_at_fight = {
        (fid, pid): int(s)
        for fid, pid, s in zip(df["fight_id"], df["stats_fighter_id"], df["fighter_style"])
    }

    # Style matchup rate as it stood before each fight (never includes that fight's result)
    style_adv_at_fight = compute_style_matchup_features(df, style_at_fight, N_STYLES)
    df["style_matchup_adv"] = df.apply(
        lambda r: style_adv_at_fight.get((r["fight_id"], r["stats_fighter_id"]), 0.5), axis=1
    )

    # One-hot encode style
    for i in range(N_STYLES):
        df[f"style_{i}"] = (df["fighter_style"] == i).astype(float)

    log.info(f"  Feature matrix shape: {df.shape}")
    return df


# Per-fighter columns that become BOTH diff_ and red_/blue_ features. Anything left out
# here is visible to the model only as a difference, which silently erases any quantity
# that is shared by both corners or whose absolute level matters:
#   - `age` as diff-only made a 22-vs-26 fight identical to a 38-vs-42 fight
#   - `div_*` are fight-level, so red minus blue is ALWAYS 0 and division was invisible
WINNER_RAW_COLS = (
    "elo", "elo_expected", "resume_score", "career_fights", "career_win_pct",
    "streak", "finish_rate", "style_matchup_adv",
    "age", "days_since_last",
)

# Columns that describe the FIGHT, not a fighter — both corners carry the same value.
# They must be emitted once as `fight_*`; a diff is identically zero and a red_/blue_
# pair is two copies of the same number.
FIGHT_LEVEL_PREFIXES = ("div_",)
FIGHT_LEVEL_COLS = ("is_title_fight", "is_five_round", "scheduled_rounds", "scheduled_minutes")


def _is_fight_level(col: str) -> bool:
    return col.startswith(FIGHT_LEVEL_PREFIXES) or col in FIGHT_LEVEL_COLS


def winner_feature_columns(df: pd.DataFrame) -> list[str]:
    """The single source of truth for winner-model per-fighter feature columns.

    build_matchup_df() and generate_predictions() previously each built their own copy
    of this list, and they drifted: the serve-time list omitted all six pre_ufc_*
    features, so every live prediction filled them from train means instead of their
    real values. Duplicated lists are why that class of bug recurs — there is now one.
    """
    cols = [c for c in df.columns if c.startswith(("avg_", "recent_", "last3_"))]
    cols += [
        "elo", "elo_expected", "resume_score",
        "height_inches", "weight_lbs", "reach_inches", "age",
        "stance_orthodox", "stance_southpaw", "stance_switch",
        "career_win_pct", "career_fights", "finish_rate", "been_finished_rate",
        "streak", "days_since_last", "style_matchup_adv",
        "pre_ufc_wins", "pre_ufc_losses", "pre_ufc_fights", "pre_ufc_win_pct",
        "pre_ufc_quality", "ufc_experience_share",
        # Age curve
        "age_sq", "years_past_peak", "years_to_peak",
        "age_resid_output", "age_resid_defense", "age_resid_absorbed",
        # Layoff shape
        "log_days_since_last", "is_long_layoff", "is_short_turnaround",
        "age_x_log_layoff",
        # Damage accumulation
        "career_head_strikes_absorbed", "career_knockdowns_absorbed",
        "career_fight_minutes", "career_ko_losses", "fights_since_ko_loss",
        "damage_per_minute",
        # Weight-class movement
        "division_rank", "division_rank_delta", "moved_up", "moved_down",
        "is_catchweight", "changed_division", "fights_in_current_division",
        "divisions_fought_count",
    ]
    # Composite features
    cols += [c for c in df.columns if "composite" in c and c.startswith(("avg_", "recent_", "last3_"))]
    # Round profile features
    cols += [c for c in df.columns if c.startswith(("avg_r1_", "avg_late_", "avg_output_", "avg_ctrl_trend",
                                                    "recent_r1_", "recent_late_", "recent_output_", "recent_ctrl_trend"))]
    cols += [c for c in df.columns if c.startswith("style_") and c not in cols]
    # Fight-level context
    cols += [c for c in df.columns if c.startswith("div_") and c not in cols]
    cols += [c for c in FIGHT_LEVEL_COLS if c not in cols]
    # Glicko multi-dimensional ratings
    cols += [c for c in df.columns if c.startswith("glicko_") and c not in cols]
    # Only keep what actually exists on this frame, deduplicated, order preserved
    present = set(df.columns)
    return [c for c in dict.fromkeys(cols) if c in present]


def build_matchup_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log.info("Building matchup feature matrix...")

    feature_cols = winner_feature_columns(df)

    # NOTE: no imputation here. Filling from full-frame means would leak test-period
    # statistics into training and would also make serve-time constants differ from
    # train-time ones. NaNs are carried through and imputed per-fold from train means
    # by _fillna_from_train(); HistGradientBoosting also handles NaN natively.

    red = df[df["corner"] == "red"].set_index("fight_id")
    blue = df[df["corner"] == "blue"].set_index("fight_id")
    common = red.index.intersection(blue.index)
    red, blue = red.loc[common], blue.loc[common]

    matchup = pd.DataFrame(index=common)
    matchup.index.name = "fight_id"
    matchup["date"] = red["date"].values
    matchup["red_wins"] = (red["stats_fighter_id"].values == red["winner_id"].values).astype(int)

    # Realised outcomes, carried for the simulator's hazard fitting. Prefixed
    # `outcome_` so they can never be mistaken for features — note that a name starting
    # with `fight_` WOULD be picked up by the feature filter, and fight duration is a
    # post-fight quantity.
    method_str = red["method"].astype(str).str.upper()
    is_ko = method_str.str.contains("KO", na=False).to_numpy()
    is_sub = method_str.str.contains("SUB", na=False).to_numpy()
    matchup["outcome_method_class"] = np.where(is_ko, 0, np.where(is_sub, 1, 2)).astype(int)
    matchup["outcome_fight_minutes"] = (
        red["fight_time_seconds"].astype(float).values / 60.0
    )

    # Fight-level columns emitted once; per-fighter columns differenced.
    fight_cols = [c for c in feature_cols if _is_fight_level(c)]
    fighter_cols = [c for c in feature_cols if not _is_fight_level(c)]

    for col in fight_cols:
        matchup[f"fight_{col}"] = red[col].values

    # Difference features
    for col in fighter_cols:
        matchup[f"diff_{col}"] = red[col].values - blue[col].values

    # Raw values for key features (both sides)
    raw_cols = [c for c in WINNER_RAW_COLS if c in fighter_cols]
    raw_cols += [c for c in fighter_cols if c.startswith("glicko_")]
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
    # Keep NaN where odds are missing — `> 0.5` on NaN would silently become 0.0
    # ("blue is favorite") rather than "unknown".
    matchup["odds_fav_is_red"] = np.where(
        matchup["odds_red_prob"].isna(), np.nan,
        (matchup["odds_red_prob"] > 0.5).astype(float),
    )
    matchup["elo_vs_odds"] = matchup["diff_elo_expected"] - matchup["odds_diff"]

    # Only 25% of fights have odds. Filling the gap with 0.0 hands the model a clean
    # "this fight has odds" indicator, and odds coverage tracks card prominence — so
    # that sentinel is a real signal leak. NaN is the honest missing marker and
    # HistGradientBoosting splits on it natively.
    matchup = matchup.dropna(subset=["red_wins"])

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

    # Corner-swap augmentation used to happen here, before the split. Because every
    # fight then had an exact mirror twin, sklearn's shuffled `validation_fraction`
    # early-stopping split was populated with mirrors of rows being fit — so the
    # dominant regularization knob was tuned against leaked data. Augmentation now
    # happens on the training slice only, in _corner_swap_augment().

    matchup = matchup.sort_values("date")

    feature_names = [
        c for c in matchup.columns
        if (c.startswith(("diff_", "red_", "blue_", "fight_")) or "odds" in c or c == "elo_vs_odds")
        and c not in ("red_wins",)
    ]

    # Feature selection used to run mutual information over the full frame, choosing
    # the top 39 with test-set labels visible. It is now fold-scoped in
    # select_winner_features(), which sees training rows only.

    log.info(f"  Matchup matrix: {matchup.shape[0]} fights, {len(feature_names)} candidate features")
    log.info(f"  Red win rate: {matchup['red_wins'].mean():.3f}")
    log.info(f"  Date range: {matchup['date'].min()} to {matchup['date'].max()}")
    return matchup, feature_names


# ===========================================================================
# PHASE 1: Gradient Boosting
# ===========================================================================

# ===========================================================================
# FOLD-SCOPED PREPARATION
# Everything here is fit on training rows only. Keeping these out of
# build_matchup_df() is what makes walk-forward evaluation honest.
# ===========================================================================

# Features that encode the market price. Held separately so the evaluation harness
# can run a no-odds arm — the only arm that answers "do I beat the closing line".
ODDS_FEATURE_MARKERS = ("odds", "elo_vs_odds")


def is_odds_feature(name: str) -> bool:
    return ("odds" in name) or (name == "elo_vs_odds")


def is_glicko_feature(name: str) -> bool:
    """Glicko features are the top-ranked ones by MI, and until the newcomer-seed fix
    in glicko_service._pre_ufc_records they were contaminated by each fighter's
    LIFETIME record (which includes their future UFC results). Dropping them gives a
    conservative floor that holds regardless of whether snapshots have been recomputed.
    """
    return "glicko" in name


def _fillna_from_train(
    matchup: pd.DataFrame, feature_names: list[str], train_mask: np.ndarray,
    skip_odds: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fill NaNs in feature columns from TRAINING-row means only.

    Odds columns are left as NaN by default. Only ~25% of fights are priced, so
    mean-filling would hand three quarters of the data a fabricated market
    probability near 0.5 — which is both wrong and a de-facto "has odds" indicator.
    HistGradientBoosting learns a split direction for NaN natively.

    Returns the filled frame and the fitted means, which callers should persist so
    serve-time imputation uses identical constants.
    """
    out = matchup.copy()
    target = [f for f in feature_names if not (skip_odds and is_odds_feature(f))]
    train_means = out.loc[train_mask, target].mean()
    out[target] = out[target].fillna(train_means).fillna(0.0)
    return out, train_means


def select_winner_features(
    matchup: pd.DataFrame,
    feature_names: list[str],
    train_mask: np.ndarray,
    top_n: int = 39,
    include_odds: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Rank features by mutual information on TRAINING rows only.

    Odds features are ranked separately and force-included when `include_odds`, so the
    top_n budget is spent entirely on non-market signal (matching tuner.py's approach).
    """
    non_odds = [f for f in feature_names if not is_odds_feature(f)]
    odds_features = [f for f in feature_names if is_odds_feature(f)]

    train = matchup.loc[train_mask]
    X_train = train[non_odds].values
    y_train = train["red_wins"].values

    if verbose:
        log.info(f"  Feature selection: MI on {len(train)} training rows only")

    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_ranked = sorted(zip(non_odds, mi_scores), key=lambda x: x[1], reverse=True)

    if verbose:
        log.info("  Top 30 features by mutual information:")
        for name, score in mi_ranked[:30]:
            log.info(f"    {name:50s} {score:.4f}")

    selected = [name for name, _ in mi_ranked[:top_n]]
    if include_odds:
        selected += [f for f in odds_features if f not in selected]

    if verbose:
        log.info(
            f"  Selected {len(selected)} features "
            f"({min(top_n, len(mi_ranked))} MI + "
            f"{len(odds_features) if include_odds else 0} odds, from {len(feature_names)})"
        )
    return list(dict.fromkeys(selected))


def _corner_swap_augment(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Double the TRAINING set by mirroring red/blue corners.

    Applied after the split, never to the full frame — mirrors of test fights must not
    reach the training set, and mirrors of training rows must not reach any validation
    split. Label flips; diff_* negate; red_*/blue_* swap; odds mirror.
    """
    X_swap = X.copy()
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    for i, name in enumerate(feature_names):
        if name.startswith("diff_"):
            X_swap[:, i] = -X[:, i]
        elif name.startswith("red_"):
            blue_name = "blue_" + name[4:]
            j = name_to_idx.get(blue_name)
            if j is not None:
                X_swap[:, i] = X[:, j]
                X_swap[:, j] = X[:, i]
        elif name == "odds_red_prob":
            j = name_to_idx.get("odds_blue_prob")
            if j is not None:
                X_swap[:, i], X_swap[:, j] = X[:, j], X[:, i]
        elif name == "odds_red_american":
            j = name_to_idx.get("odds_blue_american")
            if j is not None:
                X_swap[:, i], X_swap[:, j] = X[:, j], X[:, i]
        elif name == "odds_diff":
            X_swap[:, i] = -X[:, i]
        elif name == "odds_fav_is_red":
            X_swap[:, i] = 1 - X[:, i]
        elif name == "elo_vs_odds":
            X_swap[:, i] = -X[:, i]

    return np.vstack([X, X_swap]), np.concatenate([y, 1 - y])


# Optuna-tuned (trial #58). NOTE: that search maximized AUC on the same chronological
# 20% slice it then reported, so these values carry selection optimism. The walk-forward
# harness records this explicitly rather than pretending otherwise.
GBT_PARAMS = dict(
    max_depth=3,
    learning_rate=0.037,
    max_features=0.65,
    min_samples_leaf=32,
    l2_regularization=4.4,
    max_bins=128,
)
GBT_MAX_ITER = 1000


def fit_gbt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    params: dict | None = None,
    val_fraction: float = 0.15,
    augment: bool = True,
    seed: int = 42,
) -> tuple[HistGradientBoostingClassifier, int]:
    """Fit the GBT with chronological early stopping and train-only augmentation.

    sklearn's built-in `validation_fraction` carves its validation set with a SHUFFLED
    split. With corner-swap mirrors in the training frame that validation set fills up
    with mirrors of rows being fit, so early stopping — the dominant regularization knob
    at max_iter=1000 — is chosen against leaked data and the model ends up badly
    under-regularized.

    Instead: hold out the most recent `val_fraction` of training rows chronologically,
    augment the fit and validation halves independently, pick the iteration count by
    validation log loss, then refit on the whole training slice at that count.

    Rows must already be in chronological order.
    """
    params = {**GBT_PARAMS, **(params or {})}
    n = len(X_train)
    n_val = int(n * val_fraction)

    def _make(max_iter):
        return HistGradientBoostingClassifier(
            max_iter=max_iter, early_stopping=False, random_state=seed, **params
        )

    best_iter = GBT_MAX_ITER
    if n_val >= 200:
        X_fit, y_fit = X_train[:-n_val], y_train[:-n_val]
        X_val, y_val = X_train[-n_val:], y_train[-n_val:]

        if augment:
            X_fit, y_fit = _corner_swap_augment(X_fit, y_fit, feature_names)
            X_val, y_val = _corner_swap_augment(X_val, y_val, feature_names)

        probe = _make(GBT_MAX_ITER).fit(X_fit, y_fit)

        best_loss, best_iter, since_best = np.inf, 1, 0
        for i, proba in enumerate(probe.staged_predict_proba(X_val), start=1):
            loss = log_loss(y_val, proba[:, 1], labels=[0, 1])
            if loss < best_loss - 1e-6:
                best_loss, best_iter, since_best = loss, i, 0
            else:
                since_best += 1
                if since_best >= 75:
                    break
        log.info(
            f"  Early stopping (chronological val, n={n_val}): "
            f"best_iter={best_iter} val_logloss={best_loss:.4f}"
        )
    else:
        log.warning(f"  Only {n_val} validation rows — skipping early stopping")

    X_full, y_full = (
        _corner_swap_augment(X_train, y_train, feature_names) if augment else (X_train, y_train)
    )
    model = _make(best_iter).fit(X_full, y_full)
    return model, best_iter


def train_gbt(matchup: pd.DataFrame, feature_names: list[str], odds_only: bool = False) -> dict:
    log.info("=" * 60)
    log.info(f"PHASE 1: Gradient Boosting {'(odds-only)' if odds_only else '(full)'}")
    log.info("=" * 60)

    matchup = matchup.sort_values("date").reset_index(drop=True)

    if odds_only:
        # Row filter — fights that have a market price. This is NOT a no-odds ablation;
        # see walk_forward_eval() for the arm that drops the odds features themselves.
        has_odds = matchup["odds_red_prob"].notna()
        matchup_filtered = matchup[has_odds].reset_index(drop=True)
        log.info(f"  Filtered to {len(matchup_filtered)} fights with odds (from {len(matchup)})")
    else:
        matchup_filtered = matchup

    # Chronological 80/20. Mirrors are no longer present in the frame, so a fight can
    # only ever land on one side of this boundary.
    split_idx = int(len(matchup_filtered) * 0.8)
    train_mask = np.zeros(len(matchup_filtered), dtype=bool)
    train_mask[:split_idx] = True

    matchup_filtered, train_means = _fillna_from_train(
        matchup_filtered, feature_names, train_mask
    )
    selected = select_winner_features(matchup_filtered, feature_names, train_mask)

    train = matchup_filtered.iloc[:split_idx]
    test = matchup_filtered.iloc[split_idx:]

    X_train, y_train = train[selected].values, train["red_wins"].values
    X_test, y_test = test[selected].values, test["red_wins"].values

    log.info(f"  Train: {len(train)} ({train['date'].min()} to {train['date'].max()})")
    log.info(f"  Test:  {len(test)} ({test['date'].min()} to {test['date'].max()})")
    log.info(f"  Train red win rate: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    feature_names = selected
    model, best_iter = fit_gbt(X_train, y_train, selected)

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
        pickle.dump({
            "model": model,
            "features": feature_names,
            # Persist the train-fitted imputation constants so serving fills NaNs with
            # exactly what training used, instead of recomputing its own means.
            "train_means": train_means,
            "best_iter": best_iter,
            "version": 4,
        }, f)
    log.info(f"  Saved to {MODEL_DIR / 'gbt_v3.pkl'}")

    return {"model": model, "test_proba": y_proba, "test_y": y_test,
            "accuracy": acc, "auc": auc, "log_loss": ll,
            "features": feature_names, "train_means": train_means}


# ===========================================================================
# PROBABILITY CALIBRATION
# ===========================================================================

def calibrate_model(matchup: pd.DataFrame, feature_names: list[str],
                    gbt_model=None, train_means: pd.Series | None = None) -> dict:
    """Calibrate the Phase 1 GBT model using Platt, isotonic, and Venn-Abers.

    Reuses the already-trained GBT model (80% train). Splits the 20% test set
    into calibration (10%) and final test (10%) to fit and evaluate calibration.

    Venn-Abers uses k-fold CV on the training set to generate ~5,000 out-of-fold
    calibration samples, then applies leave-one-out isotonic per prediction.

    CAVEAT: the winning calibrator is chosen by Brier on the same eval slice whose
    metrics are then reported, so those metrics carry selection optimism. Treat
    walk_forward_eval() as the source of truth for headline numbers.
    """
    log.info("=" * 60)
    log.info("PROBABILITY CALIBRATION")
    log.info("=" * 60)

    if gbt_model is None:
        with open(MODEL_DIR / "gbt_v3.pkl", "rb") as f:
            saved = pickle.load(f)
        gbt_model = saved["model"]
        feature_names = saved["features"]
        log.info("  Loaded Phase 1 GBT from disk")

    matchup = matchup.sort_values("date").reset_index(drop=True)

    # Same 80/20 split as Phase 1. Mirrors no longer live in the frame, so the test
    # slice is already all-original — no _augmented filtering needed.
    split_idx = int(len(matchup) * 0.8)
    train_mask = np.zeros(len(matchup), dtype=bool)
    train_mask[:split_idx] = True

    if train_means is not None:
        matchup[feature_names] = matchup[feature_names].fillna(train_means).fillna(0.0)
    else:
        matchup, train_means = _fillna_from_train(matchup, feature_names, train_mask)

    train_all = matchup.iloc[:split_idx]
    test = matchup.iloc[split_idx:]

    # Split the test set: first half for Platt/isotonic calibration, second half for evaluation
    cal_split = int(len(test) * 0.5)
    cal_set = test.iloc[:cal_split]
    eval_set = test.iloc[cal_split:]

    X_cal, y_cal = cal_set[feature_names].values, cal_set["red_wins"].values
    X_eval, y_eval = eval_set[feature_names].values, eval_set["red_wins"].values

    log.info(f"  Using Phase 1 GBT (trained on 80% data)")
    log.info(f"  Platt/Isotonic calibration: {len(cal_set)} | Eval: {len(eval_set)}")
    log.info(f"  Eval period: {eval_set['date'].min()} to {eval_set['date'].max()}")
    log.info(f"  Features: {len(feature_names)}")

    # Get raw probabilities from the Phase 1 model
    raw_cal_proba = gbt_model.predict_proba(X_cal)[:, 1]
    raw_eval_proba = gbt_model.predict_proba(X_eval)[:, 1]

    # --- Platt scaling (sigmoid) ---
    from sklearn.linear_model import LogisticRegression
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10000)
    platt.fit(raw_cal_proba.reshape(-1, 1), y_cal)
    platt_proba = platt.predict_proba(raw_eval_proba.reshape(-1, 1))[:, 1]

    # --- Isotonic regression ---
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(raw_cal_proba, y_cal)
    iso_proba = iso.predict(raw_eval_proba)

    # --- Venn-Abers with k-fold CV calibration data ---
    log.info(f"\n  Generating Venn-Abers calibration data via 5-fold CV on training set...")
    train_orig = train_all
    X_train_orig = train_orig[feature_names].values
    y_train_orig = train_orig["red_wins"].values

    skf = StratifiedKFold(n_splits=5, shuffle=False)  # no shuffle to respect temporal ordering
    oof_proba = np.full(len(train_orig), np.nan)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_orig, y_train_orig)):
        # fit_gbt augments inside the fold's own training rows only, so no mirror of a
        # held-out row can reach the fit or the early-stopping validation set.
        fold_model, _ = fit_gbt(
            X_train_orig[train_idx], y_train_orig[train_idx], feature_names
        )
        oof_proba[val_idx] = fold_model.predict_proba(X_train_orig[val_idx])[:, 1]
        log.info(f"    Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")

    log.info(f"  Venn-Abers calibration samples: {np.sum(~np.isnan(oof_proba))}")

    # Fit Venn-Abers on out-of-fold predictions
    va = VennAbers()
    oof_proba_2d = np.column_stack([1 - oof_proba, oof_proba])
    va.fit(p_cal=oof_proba_2d, y_cal=y_train_orig)

    # Get VA predictions on eval set
    eval_proba_2d = np.column_stack([1 - raw_eval_proba, raw_eval_proba])
    va_prime, va_p0p1 = va.predict_proba(p_test=eval_proba_2d)
    va_proba = va_prime[:, 1]  # calibrated red_prob
    va_p0 = va_p0p1[:, 0]     # lower bound
    va_p1 = va_p0p1[:, 1]     # upper bound
    va_widths = va_p1 - va_p0

    log.info(f"  VA interval widths — mean: {va_widths.mean():.4f}, "
             f"median: {np.median(va_widths):.4f}, p90: {np.percentile(va_widths, 90):.4f}")

    # --- Isotonic regression with k-fold CV data ---
    from sklearn.isotonic import IsotonicRegression as IR
    iso_kfold = IR(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_kfold.fit(oof_proba, y_train_orig)
    iso_kfold_proba = iso_kfold.predict(raw_eval_proba)
    log.info(f"  Isotonic (k-fold) fitted on {len(oof_proba)} out-of-fold samples")

    # --- Evaluate all calibration methods ---
    methods = [
        ("Raw (uncalibrated)", raw_eval_proba),
        ("Platt scaling", platt_proba),
        ("Isotonic regression", iso_proba),
        ("Isotonic (k-fold)", iso_kfold_proba),
        ("Venn-Abers (k-fold)", va_proba),
    ]
    log.info(f"\n  {'Method':<25s} {'Acc':>7s} {'AUC':>7s} {'LogLoss':>8s} {'Brier':>7s}")
    log.info(f"  {'-'*58}")

    best_method, best_brier = None, float("inf")
    for name, proba in methods:
        acc = accuracy_score(y_eval, (proba >= 0.5).astype(int))
        auc = roc_auc_score(y_eval, proba)
        ll = log_loss(y_eval, proba)
        brier = brier_score_loss(y_eval, proba)
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
        prob_true, prob_pred = calibration_curve(y_eval, proba, n_bins=10, strategy="uniform")
        for pt, pp in zip(prob_true, prob_pred):
            gap = abs(pt - pp)
            bar = "#" * int(gap * 100)
            log.info(f"  {pp:>10.3f} → {pt:>8.3f}  gap={gap:.3f} {bar}")

    # Save calibrated model
    calibrated = {
        "base_model": gbt_model,
        "platt": platt,
        "isotonic": iso,
        "isotonic_kfold": iso_kfold,
        "venn_abers": va,
        "features": feature_names,
        "best_method": best_method,
        # Serving must impute with the same constants training fit — see
        # generate_predictions().
        "train_means": train_means,
    }
    with open(MODEL_DIR / "calibrated_model.pkl", "wb") as f:
        pickle.dump(calibrated, f)
    log.info(f"\n  Saved calibrated model to {MODEL_DIR / 'calibrated_model.pkl'}")

    # Return calibrated probabilities for betting sim — eval set only
    raw_eval = raw_eval_proba
    platt_eval = platt.predict_proba(raw_eval.reshape(-1, 1))[:, 1]
    iso_eval = iso.predict(raw_eval)

    return {
        "raw_proba": raw_eval,
        "platt_proba": platt_eval,
        "iso_proba": iso_eval,
        "iso_kfold_proba": iso_kfold_proba,
        "va_proba": va_proba,
        "va_p0": va_p0,
        "va_p1": va_p1,
        "best_method": best_method,
        "test_y": y_eval,
        "test_df": eval_set,
        "venn_abers": va,
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
    iso_kfold_proba_odds = cal_results["iso_kfold_proba"][has_odds.values]
    va_proba_odds = cal_results["va_proba"][has_odds.values]
    va_p0_odds = cal_results["va_p0"][has_odds.values]
    va_p1_odds = cal_results["va_p1"][has_odds.values]
    va_widths_odds = va_p1_odds - va_p0_odds

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
        ("Isotonic (k-fold)", iso_kfold_proba_odds),
        ("Venn-Abers (k-fold)", va_proba_odds),
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

    # --- VA interval-filtered strategies ---
    log.info(f"\n{'─'*60}")
    log.info(f"  Venn-Abers Interval-Filtered Strategies")
    log.info(f"{'─'*60}")
    log.info(f"  VA interval widths — mean: {va_widths_odds.mean():.4f}, "
             f"median: {np.median(va_widths_odds):.4f}, "
             f"p75: {np.percentile(va_widths_odds, 75):.4f}, "
             f"p90: {np.percentile(va_widths_odds, 90):.4f}, "
             f"max: {va_widths_odds.max():.4f}")

    # Strategy: VA point estimate edge > 3% + interval width filter
    log.info(f"\n  --- VA Point Estimate Edge (>3%) + Interval Filter ---")
    for max_width in [0.005, 0.01, 0.02, 0.05, 0.10, 1.0]:
        narrow = va_widths_odds < max_width
        profit, bets, wins = 0, 0, 0
        for i in range(len(odds_test)):
            if not narrow[i]:
                continue
            edge_red = va_proba_odds[i] - market_red_fair[i]
            edge_blue = (1 - va_proba_odds[i]) - market_blue_fair[i]
            ev_red = va_proba_odds[i] * odds_red_dec[i]
            ev_blue = (1 - va_proba_odds[i]) * odds_blue_dec[i]

            if edge_red > 0.03 and ev_red > 1.0:
                bets += 1
                if odds_y[i] == 1:
                    profit += (odds_red_dec[i] - 1) * 100
                    wins += 1
                else:
                    profit -= 100
            elif edge_blue > 0.03 and ev_blue > 1.0:
                bets += 1
                if odds_y[i] == 0:
                    profit += (odds_blue_dec[i] - 1) * 100
                    wins += 1
                else:
                    profit -= 100

        label = f"<{max_width:.1%}" if max_width < 1.0 else "no filter"
        if bets > 0:
            roi = profit / (bets * 100) * 100
            log.info(f"    interval {label:>12s}: {bets:>4d} bets, "
                     f"{wins} wins ({wins/bets:.1%}), ${profit:>+10.2f}, ROI: {roi:+.2f}%")
        else:
            log.info(f"    interval {label:>12s}: no bets (0/{int(narrow.sum())} fights pass filter)")

    # Strategy: Conservative edge using va_prob_low (p0) instead of point estimate
    log.info(f"\n  --- Conservative Edge (va_prob_low) vs Point Estimate ---")
    for min_edge in [0.03, 0.05, 0.08, 0.10, 0.15]:
        # Conservative: use p0 (lower bound) for edge calculation
        c_profit, c_bets, c_wins = 0, 0, 0
        # Normal: use point estimate (for comparison)
        n_profit, n_bets, n_wins = 0, 0, 0
        for i in range(len(odds_test)):
            va_red = va_proba_odds[i]
            va_blue = 1 - va_red
            p0_red = va_p0_odds[i]   # conservative red prob
            p0_blue = 1 - va_p1_odds[i]  # conservative blue prob (1 - upper bound of red)

            # Normal edge (point estimate)
            edge_red_n = va_red - market_red_fair[i]
            edge_blue_n = va_blue - market_blue_fair[i]
            ev_red_n = va_red * odds_red_dec[i]
            ev_blue_n = va_blue * odds_blue_dec[i]

            if edge_red_n > min_edge and ev_red_n > 1.0:
                n_bets += 1
                if odds_y[i] == 1:
                    n_profit += (odds_red_dec[i] - 1) * 100
                    n_wins += 1
                else:
                    n_profit -= 100
            elif edge_blue_n > min_edge and ev_blue_n > 1.0:
                n_bets += 1
                if odds_y[i] == 0:
                    n_profit += (odds_blue_dec[i] - 1) * 100
                    n_wins += 1
                else:
                    n_profit -= 100

            # Conservative edge (use p0/lower bound)
            edge_red_c = p0_red - market_red_fair[i]
            edge_blue_c = p0_blue - market_blue_fair[i]
            ev_red_c = p0_red * odds_red_dec[i]
            ev_blue_c = p0_blue * odds_blue_dec[i]

            if edge_red_c > min_edge and ev_red_c > 1.0:
                c_bets += 1
                if odds_y[i] == 1:
                    c_profit += (odds_red_dec[i] - 1) * 100
                    c_wins += 1
                else:
                    c_profit -= 100
            elif edge_blue_c > min_edge and ev_blue_c > 1.0:
                c_bets += 1
                if odds_y[i] == 0:
                    c_profit += (odds_blue_dec[i] - 1) * 100
                    c_wins += 1
                else:
                    c_profit -= 100

        log.info(f"\n    Edge > {min_edge:.0%}:")
        if n_bets > 0:
            log.info(f"      Point estimate: {n_bets:>4d} bets, {n_wins} wins ({n_wins/n_bets:.1%}), "
                     f"${n_profit:>+10.2f}, ROI: {n_profit/(n_bets*100)*100:+.2f}%")
        else:
            log.info(f"      Point estimate: no bets")
        if c_bets > 0:
            log.info(f"      Conservative:   {c_bets:>4d} bets, {c_wins} wins ({c_wins/c_bets:.1%}), "
                     f"${c_profit:>+10.2f}, ROI: {c_profit/(c_bets*100)*100:+.2f}%")
        else:
            log.info(f"      Conservative:   no bets")

    # --- Final summary ---
    log.info(f"\n{'='*60}")
    log.info(f"BETTING SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Test period: {odds_test['date'].min()} to {odds_test['date'].max()}")
    log.info(f"  Total fights with odds: {len(odds_test)}")
    log.info(f"  Market (vig-adjusted) accuracy: {accuracy_score(odds_y, (market_red_fair >= 0.5).astype(int)):.4f}")
    log.info(f"  VA model accuracy: {accuracy_score(odds_y, (va_proba_odds >= 0.5).astype(int)):.4f}")
    log.info(f"  Best calibration method: {cal_results['best_method']}")
    log.info(f"  Key insight: profitability requires finding +EV spots where")
    log.info(f"  model_prob * decimal_odds > 1.0 (positive expected value)")


# ===========================================================================
# WALK-FORWARD EVALUATION
#
# The source of truth for headline numbers. Every fit — imputation, feature
# selection, corner-swap augmentation, early stopping, calibration — happens
# strictly inside a fold's training window. Betting thresholds are frozen below
# rather than swept and cherry-picked after seeing results.
# ===========================================================================

# Frozen BEFORE scoring. Report every rung; never quote the best cell as the headline.
FROZEN_EDGE_THRESHOLDS = (0.03, 0.05, 0.08, 0.10, 0.15)
# Weakest feature set first, so the floor is read before the headline.
ARM_ORDER = ("no_odds_no_glicko", "no_odds", "simulator", "with_odds",
             "market_anchored", "decorrelated")

# Frozen BEFORE running. gamma=0 is the undecorrelated control, so the sweep contains
# its own baseline. Selected per fold on INNER-window ROI, never on the reported eval.
DECORR_GAMMAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
# The anchored arm corrects this arm's predictions toward/away from the line. It must
# be a NO-ODDS arm: anchoring a model that already contains the market price would
# double-count the line.
ANCHOR_BASE_ARM = "no_odds"
FLAT_STAKE = 100.0


def market_correlation(model_p: np.ndarray, market_p: np.ndarray,
                      y: np.ndarray | None = None) -> tuple[float, float]:
    """Correlation between a model's log-odds and the market's, raw and partialled on
    the realised outcome.

    Hubacek & Sir (Int. J. Forecasting 2022) prove that a bettor whose estimates coincide
    with the market's has EXACTLY ZERO profitability regardless of how accurate either is
    — their Example 3.3. Profit requires the market to err and the model to err in the
    opposite direction, which is impossible when the two agree. So this number, not
    accuracy, is what determines whether an edge can exist at all.

    The partial correlation conditions on the outcome, isolating agreement that is NOT
    explained by both simply being right.
    """
    lt, lm = logit(model_p), logit(market_p)
    raw = float(np.corrcoef(lt, lm)[0, 1])
    if y is None or len(np.unique(y)) < 2:
        return raw, float("nan")
    yy = np.asarray(y, dtype=float)
    rt = lt - np.polyval(np.polyfit(yy, lt, 1), yy)
    rm = lm - np.polyval(np.polyfit(yy, lm, 1), yy)
    return raw, float(np.corrcoef(rt, rm)[0, 1])


def _bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 2000,
                  alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI. A point estimate with no interval is what made the
    original 72.9% look solid at n=510."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = stat(values[idx], axis=1)
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def _betting_ladder(model_prob: np.ndarray, y: np.ndarray,
                    red_dec: np.ndarray, blue_dec: np.ndarray,
                    market_red_fair: np.ndarray, market_blue_fair: np.ndarray) -> list[dict]:
    """Flat-stake value betting at the frozen thresholds, vig removed."""
    rungs = []
    blue_prob = 1 - model_prob

    for min_edge in FROZEN_EDGE_THRESHOLDS:
        pnls = []
        for i in range(len(y)):
            edge_red = model_prob[i] - market_red_fair[i]
            edge_blue = blue_prob[i] - market_blue_fair[i]
            ev_red = model_prob[i] * red_dec[i]
            ev_blue = blue_prob[i] * blue_dec[i]

            if edge_red > min_edge and ev_red > 1.0:
                pnls.append((red_dec[i] - 1) * FLAT_STAKE if y[i] == 1 else -FLAT_STAKE)
            elif edge_blue > min_edge and ev_blue > 1.0:
                pnls.append((blue_dec[i] - 1) * FLAT_STAKE if y[i] == 0 else -FLAT_STAKE)

        pnls = np.array(pnls, dtype=float)
        n = len(pnls)
        roi = float(pnls.sum() / (n * FLAT_STAKE) * 100) if n else float("nan")
        roi_lo, roi_hi = (
            _bootstrap_ci(pnls / FLAT_STAKE * 100) if n >= 20 else (float("nan"), float("nan"))
        )
        rungs.append({
            "min_edge": min_edge,
            "bets": n,
            "bet_rate": float(n / len(y)) if len(y) else float("nan"),
            "win_rate": float((pnls > 0).mean()) if n else float("nan"),
            "profit": float(pnls.sum()),
            "roi_pct": roi,
            "roi_ci95": [roi_lo, roi_hi],
        })
    return rungs


def walk_forward_eval(
    matchup: pd.DataFrame,
    feature_names: list[str],
    n_folds: int = 8,
    eval_frac: float = 0.4,
    top_n: int = 39,
    sim_n: int = 5000,
) -> dict:
    """Expanding-window walk-forward evaluation, run once per arm.

    Arms:
      with_odds  — production feature set (includes the market price)
      no_odds    — market features removed; the ONLY arm that answers
                   "does my handicapping beat the closing line"

    Both are scored against a market-only reference on the identical fight set.
    """
    log.info("=" * 60)
    log.info("WALK-FORWARD EVALUATION")
    log.info("=" * 60)

    # reset_index() (not drop=True) keeps fight_id as a column so per-fight
    # predictions can be joined back and audited.
    matchup = matchup.sort_values("date").reset_index()
    n = len(matchup)
    eval_start = int(n * (1 - eval_frac))
    bounds = np.linspace(eval_start, n, n_folds + 1).astype(int)

    log.info(f"  {n} fights | {n_folds} folds over the last {eval_frac:.0%} "
             f"({n - eval_start} eval fights)")
    log.info(f"  Hyperparameters are FIXED (not tuned per fold): {GBT_PARAMS}")
    log.info("  Those values came from an Optuna search that scored on a chronological")
    log.info("  20% slice overlapping this eval window, so a residual selection optimism")
    log.info("  remains. It is bounded and disclosed rather than hidden.")

    arms = {
        # Production feature set. Contains the market price, so its accuracy is NOT
        # independent evidence of beating the market.
        "with_odds": list(feature_names),
        # Market features removed — the arm that answers "does my handicapping beat
        # the closing line".
        "no_odds": [f for f in feature_names if not is_odds_feature(f)],
        # Conservative floor: also drops Glicko, which carried the newcomer-seed leak
        # until snapshots are recomputed with the fixed pre-UFC seeding.
        "no_odds_no_glicko": [
            f for f in feature_names
            if not is_odds_feature(f) and not is_glicko_feature(f)
        ],
    }

    results = {}
    anchor_folds, all_anchored, all_anchor_mkt = [], [], []

    for arm_name, arm_features in arms.items():
        log.info("\n" + "-" * 60)
        log.info(f"ARM: {arm_name}  ({len(arm_features)} candidate features)")
        log.info("-" * 60)

        fold_rows, all_proba, all_cal_proba, all_y, all_idx = [], [], [], [], []

        for k in range(n_folds):
            lo, hi = bounds[k], bounds[k + 1]
            if hi <= lo:
                continue

            train_mask = np.zeros(n, dtype=bool)
            train_mask[:lo] = True

            fold_df, _ = _fillna_from_train(matchup, arm_features, train_mask)
            selected = select_winner_features(
                fold_df, arm_features, train_mask, top_n=top_n,
                include_odds=(arm_name == "with_odds"), verbose=False,
            )

            train_df = fold_df.iloc[:lo]
            test_df = fold_df.iloc[lo:hi]

            # Hold out the tail of training for calibration; the model never sees it.
            cal_cut = int(len(train_df) * 0.85)
            fit_df, cal_df = train_df.iloc[:cal_cut], train_df.iloc[cal_cut:]

            X_fit = fit_df[selected].values
            y_fit = fit_df["red_wins"].values
            X_test = test_df[selected].values
            y_test = test_df["red_wins"].values

            model, best_iter = fit_gbt(X_fit, y_fit, selected)
            proba = model.predict_proba(X_test)[:, 1]

            # Isotonic calibration fit on the held-out tail of train only
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(model.predict_proba(cal_df[selected].values)[:, 1],
                    cal_df["red_wins"].values)
            cal_proba = iso.predict(proba)

            # --- Market-anchored variant, derived from the no-odds base model ---
            # The anchor is fit on cal_df: rows the base model was NOT trained on, so
            # `b` reflects out-of-sample disagreement rather than training sharpness.
            if arm_name == ANCHOR_BASE_ARM:
                cal_mkt = devig(cal_df["odds_red_prob"].values,
                                cal_df["odds_blue_prob"].values)
                cal_mkt = np.where(cal_df["odds_red_prob"].notna().values, cal_mkt, np.nan)
                test_mkt = devig(test_df["odds_red_prob"].values,
                                 test_df["odds_blue_prob"].values)
                test_mkt = np.where(test_df["odds_red_prob"].notna().values, test_mkt, np.nan)

                anchor = MarketAnchor().fit(
                    iso.predict(model.predict_proba(cal_df[selected].values)[:, 1]),
                    cal_mkt,
                    cal_df["red_wins"].values,
                )
                anchored_proba = anchor.predict_proba(cal_proba, test_mkt)
                anchor_folds.append({"fold": k + 1, "b": anchor.b_, "c": anchor.c_,
                                     "n_fit": anchor.n_fit_})
                all_anchored.append(anchored_proba)
                all_anchor_mkt.append(test_mkt)

            acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
            auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
            fold_rows.append({
                "fold": k + 1,
                "train_n": int(lo),
                "test_n": int(hi - lo),
                "test_start": str(test_df["date"].min()),
                "test_end": str(test_df["date"].max()),
                "best_iter": int(best_iter),
                "n_features": len(selected),
                "accuracy": float(acc),
                "auc": float(auc),
                "log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
                "brier": float(brier_score_loss(y_test, proba)),
                "brier_calibrated": float(brier_score_loss(y_test, cal_proba)),
            })
            log.info(
                f"  Fold {k+1}/{n_folds}  train={lo:>5}  test={hi-lo:>4}  "
                f"({test_df['date'].min()} → {test_df['date'].max()})  "
                f"acc={acc:.4f}  auc={auc:.4f}  iter={best_iter}"
            )

            all_proba.append(proba)
            all_cal_proba.append(cal_proba)
            all_y.append(y_test)
            all_idx.append(test_df.index.values)

        proba = np.concatenate(all_proba)
        cal_proba = np.concatenate(all_cal_proba)
        y = np.concatenate(all_y)
        idx = np.concatenate(all_idx)

        correct = (proba >= 0.5).astype(int) == y
        acc_lo, acc_hi = _bootstrap_ci(correct.astype(float))

        results[arm_name] = {
            "folds": fold_rows,
            "n_eval": int(len(y)),
            "accuracy": float(correct.mean()),
            "accuracy_ci95": [acc_lo, acc_hi],
            "auc": float(roc_auc_score(y, proba)),
            "log_loss": float(log_loss(y, proba, labels=[0, 1])),
            "brier": float(brier_score_loss(y, proba)),
            "brier_calibrated": float(brier_score_loss(y, cal_proba)),
            "_proba": proba, "_cal_proba": cal_proba, "_y": y, "_idx": idx,
        }

        log.info(
            f"  POOLED  n={len(y)}  acc={correct.mean():.4f} "
            f"[{acc_lo:.4f}, {acc_hi:.4f}]  auc={results[arm_name]['auc']:.4f}  "
            f"brier={results[arm_name]['brier']:.4f} "
            f"(calibrated {results[arm_name]['brier_calibrated']:.4f})"
        )

    # --- Simulator arm: competing-risks Monte Carlo over the Glicko ratings ---
    # Fitted per fold on training rows only, exactly like every other arm. Produces
    # winner AND method from one simulation, so the two cannot contradict each other.
    log.info("\n" + "-" * 60)
    log.info("ARM: simulator (competing risks over Glicko)")
    log.info("-" * 60)

    sim_all = attach_fit_targets(build_simulator_frame(matchup.set_index("fight_id")),
                                 matchup.set_index("fight_id")).reset_index(drop=True)
    sim_proba, sim_method, sim_idx = [], [], []

    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        if hi <= lo:
            continue
        hazard = HazardRateModel().fit(sim_all.iloc[:lo])
        out = simulate(sim_all.iloc[lo:hi], hazard, n_sims=sim_n)
        sim_proba.append(out["red_prob"].to_numpy(float))
        sim_method.append(np.column_stack([out["p_ko"], out["p_sub"], out["p_dec"]]))
        sim_idx.append(np.arange(lo, hi))
        log.info(f"  Fold {k+1}/{n_folds}  train={lo:>5}  test={hi-lo:>4}  "
                 f"sims={sim_n}  b_dec={hazard.decision_coef_[0]:+.3f}")

    if sim_proba:
        sp = np.concatenate(sim_proba)
        sy = results[ANCHOR_BASE_ARM]["_y"]
        correct = (sp >= 0.5).astype(int) == sy
        s_lo, s_hi = _bootstrap_ci(correct.astype(float))
        results["simulator"] = {
            "n_eval": int(len(sy)),
            "accuracy": float(correct.mean()),
            "accuracy_ci95": [s_lo, s_hi],
            "auc": float(roc_auc_score(sy, sp)),
            "log_loss": float(log_loss(sy, np.clip(sp, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "brier": float(brier_score_loss(sy, sp)),
            "brier_calibrated": float(brier_score_loss(sy, sp)),
            "_proba": sp, "_cal_proba": sp,
            "_y": sy, "_idx": results[ANCHOR_BASE_ARM]["_idx"],
        }

        # --- Method output: scored on REALIZED outcomes only ---
        # There is no method market in this database (55 rows, one snapshot), so the
        # benchmark is the realised class and a training base-rate prior.
        P = np.clip(np.vstack(sim_method), 1e-6, 1.0)
        P /= P.sum(axis=1, keepdims=True)
        true_method = sim_all["method_class"].to_numpy(int)[np.concatenate(sim_idx)]
        train_end = bounds[0]
        base = np.bincount(
            sim_all["method_class"].to_numpy(int)[:train_end], minlength=3
        ) / max(train_end, 1)
        base_mat = np.tile(base, (len(true_method), 1))

        method_block = {
            "n": int(len(true_method)),
            "log_loss": float(log_loss(true_method, P, labels=[0, 1, 2])),
            "log_loss_base_rate": float(log_loss(true_method, base_mat, labels=[0, 1, 2])),
            "accuracy": float(accuracy_score(true_method, P.argmax(axis=1))),
            "per_class_brier": {
                name: {
                    "model": float(brier_score_loss((true_method == i).astype(int), P[:, i])),
                    "base_rate": float(brier_score_loss(
                        (true_method == i).astype(int), np.full(len(true_method), base[i]))),
                }
                for i, name in enumerate(("KO", "SUB", "DEC"))
            },
            "predicted_mix": P.mean(axis=0).tolist(),
            "actual_mix": (np.bincount(true_method, minlength=3) / len(true_method)).tolist(),
        }
        results["simulator"]["method"] = method_block

        log.info(f"\n  METHOD (scored on realised outcomes — no market exists for this)")
        log.info(f"    log loss   {method_block['log_loss']:.4f}  "
                 f"vs base rate {method_block['log_loss_base_rate']:.4f}")
        log.info(f"    accuracy   {method_block['accuracy']:.4f}")
        log.info(f"    {'class':<6}{'Brier':>9}{'base':>9}")
        for name, d in method_block["per_class_brier"].items():
            log.info(f"    {name:<6}{d['model']:>9.4f}{d['base_rate']:>9.4f}")
        log.info(f"    predicted mix {np.round(method_block['predicted_mix'], 3)}  "
                 f"actual {np.round(method_block['actual_mix'], 3)}")

    # --- Decorrelated arm (Hubacek & Sir Eq. 59) ---
    # Trained on the full feature set INCLUDING odds: the controlled trade needs a
    # high-accuracy starting point. gamma is chosen per fold by flat-stake ROI on an
    # inner validation window carved from training, so the reported eval never
    # participates in selection.
    log.info("\n" + "-" * 60)
    log.info("ARM: decorrelated (market-decorrelation penalty)")
    log.info("-" * 60)

    dec_proba, dec_gammas = [], []
    arm_features = arms["with_odds"]

    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        if hi <= lo:
            continue
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:lo] = True
        fold_df, _ = _fillna_from_train(matchup, arm_features, train_mask)
        selected = select_winner_features(fold_df, arm_features, train_mask,
                                          top_n=top_n, include_odds=True, verbose=False)

        train_df, test_df = fold_df.iloc[:lo], fold_df.iloc[lo:hi]
        inner = int(len(train_df) * 0.8)
        itr, iva = train_df.iloc[:inner], train_df.iloc[inner:]

        def _mkt(d):
            m = devig(d["odds_red_prob"].values, d["odds_blue_prob"].values)
            return np.where(d["odds_red_prob"].notna().values, m, np.nan)

        DecorrelatedModel = _decorrelated_model()
        best_g, best_roi = 0.0, -np.inf
        for g in DECORR_GAMMAS:
            mdl = DecorrelatedModel(gamma=g, seed=42).fit(
                itr[selected].values, itr["red_wins"].values, _mkt(itr),
                iva[selected].values, iva["red_wins"].values, _mkt(iva),
            )
            pv = mdl.predict_proba(iva[selected].values)
            mv = _mkt(iva)
            ok = np.isfinite(mv)
            if ok.sum() < 50:
                continue
            # Flat stakes: the decorrelation result exists only under uniform staking.
            ladder = _betting_ladder(
                pv[ok], iva["red_wins"].values[ok],
                np.array([_american_to_decimal(o) for o in iva["odds_red_american"].values[ok]]),
                np.array([_american_to_decimal(o) for o in iva["odds_blue_american"].values[ok]]),
                mv[ok], 1 - mv[ok],
            )
            rois = [r["roi_pct"] for r in ladder if r["bets"] >= 25 and np.isfinite(r["roi_pct"])]
            roi = float(np.mean(rois)) if rois else -np.inf
            if roi > best_roi:
                best_roi, best_g = roi, g

        mdl = DecorrelatedModel(gamma=best_g, seed=42).fit(
            train_df[selected].values, train_df["red_wins"].values, _mkt(train_df),
            iva[selected].values, iva["red_wins"].values, _mkt(iva),
        )
        dec_proba.append(mdl.predict_proba(test_df[selected].values))
        dec_gammas.append(best_g)
        log.info(f"  Fold {k+1}/{n_folds}  train={lo:>5}  test={hi-lo:>4}  "
                 f"gamma={best_g:<4} inner_roi={best_roi:+.2f}%")

    if dec_proba:
        dp = np.concatenate(dec_proba)
        dy = results[ANCHOR_BASE_ARM]["_y"]
        correct = (dp >= 0.5).astype(int) == dy
        d_lo, d_hi = _bootstrap_ci(correct.astype(float))
        results["decorrelated"] = {
            "gammas_by_fold": dec_gammas,
            "n_eval": int(len(dy)),
            "accuracy": float(correct.mean()),
            "accuracy_ci95": [d_lo, d_hi],
            "auc": float(roc_auc_score(dy, dp)),
            "log_loss": float(log_loss(dy, np.clip(dp, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "brier": float(brier_score_loss(dy, dp)),
            "brier_calibrated": float(brier_score_loss(dy, dp)),
            "_proba": dp, "_cal_proba": dp,
            "_y": dy, "_idx": results[ANCHOR_BASE_ARM]["_idx"],
        }
        log.info(f"  gamma per fold: {dec_gammas}")

    # --- Market-anchored arm, assembled from the per-fold anchors above ---
    if all_anchored:
        base = results[ANCHOR_BASE_ARM]
        anchored = np.concatenate(all_anchored)
        y = base["_y"]
        correct = (anchored >= 0.5).astype(int) == y
        a_lo, a_hi = _bootstrap_ci(correct.astype(float))
        results["market_anchored"] = {
            "base_arm": ANCHOR_BASE_ARM,
            "folds": anchor_folds,
            "mean_b": float(np.mean([f["b"] for f in anchor_folds])),
            "n_eval": int(len(y)),
            "accuracy": float(correct.mean()),
            "accuracy_ci95": [a_lo, a_hi],
            "auc": float(roc_auc_score(y, anchored)),
            "log_loss": float(log_loss(y, anchored, labels=[0, 1])),
            "brier": float(brier_score_loss(y, anchored)),
            "brier_calibrated": float(brier_score_loss(y, anchored)),
            "_proba": anchored, "_cal_proba": anchored,
            "_y": y, "_idx": base["_idx"],
        }
        log.info("\n" + "-" * 60)
        log.info(f"MARKET-ANCHORED ARM  (base = {ANCHOR_BASE_ARM})")
        log.info("-" * 60)
        log.info("  b = fraction of the model's disagreement with the line that is trusted")
        log.info(f"  {'Fold':>5} {'b':>8} {'c':>8} {'n_fit':>7}")
        log.info(f"  {'-'*32}")
        for f in anchor_folds:
            log.info(f"  {f['fold']:>5} {f['b']:>8.3f} {f['c']:>+8.3f} {f['n_fit']:>7}")
        log.info(f"  mean b = {results['market_anchored']['mean_b']:.3f}")

    # --- Market comparison on the odds subset (identical fights for every arm) ---
    log.info("\n" + "-" * 60)
    log.info("MARKET COMPARISON (fights with odds)")
    log.info("-" * 60)

    ref = results["with_odds"]
    sub = matchup.loc[ref["_idx"]]
    has_odds = sub["odds_red_prob"].notna().values & sub["odds_blue_prob"].notna().values

    market_block = {"n_with_odds": int(has_odds.sum())}
    if has_odds.sum() < 50:
        log.warning(f"  Only {has_odds.sum()} eval fights have odds — skipping")
        results["market"] = market_block
    else:
        odds_df = sub[has_odds]
        y_odds = ref["_y"][has_odds]

        mr = odds_df["odds_red_prob"].values.astype(float)
        mb = odds_df["odds_blue_prob"].values.astype(float)
        fair = np.array([_implied_prob_no_vig(a, b) for a, b in zip(mr, mb)])
        market_red_fair, market_blue_fair = fair[:, 0], fair[:, 1]

        red_dec = np.array([_american_to_decimal(o) for o in odds_df["odds_red_american"].values])
        blue_dec = np.array([_american_to_decimal(o) for o in odds_df["odds_blue_american"].values])

        market_correct = (market_red_fair >= 0.5).astype(int) == y_odds
        m_lo, m_hi = _bootstrap_ci(market_correct.astype(float))
        market_block.update({
            "accuracy": float(market_correct.mean()),
            "accuracy_ci95": [m_lo, m_hi],
            "log_loss": float(log_loss(y_odds, market_red_fair, labels=[0, 1])),
            "brier": float(brier_score_loss(y_odds, market_red_fair)),
            "avg_vig": float((mr + mb).mean() - 1),
            "date_start": str(odds_df["date"].min()),
            "date_end": str(odds_df["date"].max()),
        })
        results["market"] = market_block

        log.info(f"  Eval fights with odds: {has_odds.sum()} "
                 f"({odds_df['date'].min()} → {odds_df['date'].max()})")
        log.info(f"  Average vig: {market_block['avg_vig']:.3f}")
        log.info(f"\n  {'Arm':<20s} {'Acc':>8s} {'95% CI':>18s} {'AUC':>8s} "
                 f"{'Brier':>8s} {'corr(mkt)':>10s} {'partial':>9s}")
        log.info(f"  {'-'*88}")
        log.info(f"  {'market only':<20s} {market_block['accuracy']:>8.4f} "
                 f"{f'[{m_lo:.3f}, {m_hi:.3f}]':>18s} {'—':>8s} "
                 f"{market_block['brier']:>8.4f} {'1.000':>10s} {'1.000':>9s}")

        for arm_name in [a for a in ARM_ORDER if a in results]:
            arm = results[arm_name]
            p = arm["_proba"][has_odds]
            pc = arm["_cal_proba"][has_odds]
            c = (p >= 0.5).astype(int) == y_odds
            lo_, hi_ = _bootstrap_ci(c.astype(float))
            corr_raw, corr_par = market_correlation(pc, market_red_fair, y_odds)

            # Paired bootstrap on the accuracy DIFFERENCE vs the market, same fights.
            diff = c.astype(float) - market_correct.astype(float)
            d_lo, d_hi = _bootstrap_ci(diff)

            arm["odds_subset"] = {
                "n": int(has_odds.sum()),
                "accuracy": float(c.mean()),
                "accuracy_ci95": [lo_, hi_],
                "auc": float(roc_auc_score(y_odds, p)),
                "corr_market": corr_raw,
                "corr_market_partial": corr_par,
                "brier": float(brier_score_loss(y_odds, p)),
                "brier_calibrated": float(brier_score_loss(y_odds, pc)),
                "acc_minus_market": float(diff.mean()),
                "acc_minus_market_ci95": [d_lo, d_hi],
                "beats_market_significantly": bool(d_lo > 0),
                "betting_ladder": _betting_ladder(
                    pc, y_odds, red_dec, blue_dec, market_red_fair, market_blue_fair
                ),
            }
            log.info(f"  {arm_name:<20s} {c.mean():>8.4f} "
                     f"{f'[{lo_:.3f}, {hi_:.3f}]':>18s} "
                     f"{arm['odds_subset']['auc']:>8.4f} {arm['odds_subset']['brier']:>8.4f} "
                     f"{corr_raw:>10.3f} {corr_par:>9.3f}")

        log.info(f"\n  {'Arm':<20s} {'Acc − market':>14s} {'95% CI of diff':>22s} {'Beats mkt?':>12s}")
        log.info(f"  {'-'*72}")
        for arm_name in [a for a in ARM_ORDER if a in results]:
            o = results[arm_name]["odds_subset"]
            lo_, hi_ = o["acc_minus_market_ci95"]
            log.info(
                f"  {arm_name:<20s} {o['acc_minus_market']:>+14.4f} "
                f"{f'[{lo_:+.4f}, {hi_:+.4f}]':>22s} "
                f"{'YES' if o['beats_market_significantly'] else 'no':>12s}"
            )

        for arm_name in [a for a in ARM_ORDER if a in results]:
            log.info(f"\n  Betting ladder — {arm_name} (calibrated, frozen thresholds):")
            log.info(f"  {'Edge':>6s} {'Bets':>6s} {'Bet%':>7s} {'Win%':>7s} "
                     f"{'ROI%':>8s} {'ROI 95% CI':>22s}")
            log.info(f"  {'-'*62}")
            for r in results[arm_name]["odds_subset"]["betting_ladder"]:
                lo_, hi_ = r["roi_ci95"]
                ci = f"[{lo_:+.1f}, {hi_:+.1f}]" if r["bets"] >= 20 else "n/a"
                log.info(
                    f"  {r['min_edge']:>6.0%} {r['bets']:>6d} {r['bet_rate']:>7.1%} "
                    f"{r['win_rate']:>7.1%} {r['roi_pct']:>8.2f} {ci:>22s}"
                )

    # Per-fight predictions, so every aggregate above can be re-derived independently
    # instead of taken on faith.
    preds = pd.DataFrame({
        "fight_id": matchup.loc[ref["_idx"], "fight_id"].values,
        "date": matchup.loc[ref["_idx"], "date"].values,
        "red_wins": ref["_y"],
        "odds_red_prob": matchup.loc[ref["_idx"], "odds_red_prob"].values,
        "odds_red_american": matchup.loc[ref["_idx"], "odds_red_american"].values,
        "odds_blue_american": matchup.loc[ref["_idx"], "odds_blue_american"].values,
    })
    for arm_name in [a for a in ARM_ORDER if a in results]:
        preds[f"{arm_name}_proba"] = results[arm_name]["_proba"]
        preds[f"{arm_name}_proba_cal"] = results[arm_name]["_cal_proba"]
    preds_path = MODEL_DIR / "eval_predictions.csv"
    preds.to_csv(preds_path, index=False)
    log.info(f"\n  Saved {len(preds)} per-fight eval predictions to {preds_path}")

    return results


def save_eval_results(results: dict, path: Path = None) -> Path:
    """Persist walk-forward metrics so headline numbers stop being oral history."""
    path = path or (MODEL_DIR / "eval_results.json")
    clean = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        else:
            clean[k] = v
    clean["_meta"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gbt_params": GBT_PARAMS,
        "frozen_edge_thresholds": list(FROZEN_EDGE_THRESHOLDS),
        "caveats": [
            "Hyperparameters were Optuna-tuned on a chronological 20% slice that "
            "overlaps this eval window; residual selection optimism remains.",
            "with_odds takes the closing line as an input feature, so its accuracy "
            "cannot be read as independent evidence of beating the market. Use the "
            "no_odds arm for that comparison.",
            "Historical odds are a T-2h snapshot with a noon-of-event-day fallback, "
            "not true closing lines.",
        ],
    }
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    log.info(f"\n  Saved evaluation results to {path}")
    return path


# ===========================================================================
# MAIN
# ===========================================================================

def run(phases=None, walk_forward=False, n_folds=8, fresh_glicko=False):
    if phases is None: phases = [1]
    log.info("=" * 60)
    log.info(f"UFC Fight Winner Prediction Pipeline v6  ({datetime.now():%Y-%m-%d %H:%M})")
    log.info("=" * 60)

    df, round_data = load_fight_data()

    snaps = None
    if fresh_glicko:
        # Recompute Glicko in memory with the corrected pre-UFC newcomer seed, without
        # overwriting the stored UFCGlickoSnapshot table.
        from app.services.ufc.glicko_service import run_glicko_inmemory
        log.info("\n  Recomputing Glicko snapshots in memory (fixed newcomer seed)...")
        snaps = run_glicko_inmemory()
        log.info(f"  Computed {len(snaps)} snapshots")

    if walk_forward:
        # Two passes. The first only locates the date where the eval window starts;
        # the second rebuilds features with the style clusterer restricted to data
        # before that date, so no eval-period distribution reaches the KMeans fit.
        probe = build_matchup_df(
            build_features(df.copy(), round_data.copy(), glicko_snapshots=snaps)
        )[0]
        cutoff = probe.sort_values("date")["date"].iloc[int(len(probe) * 0.6)]
        log.info(f"\n  Style-cluster cutoff set to eval start: {cutoff}")

        df = build_features(df, round_data, style_cutoff_date=cutoff,
                            glicko_snapshots=snaps)
        matchup, features = build_matchup_df(df)

        wf = walk_forward_eval(matchup, features, n_folds=n_folds)
        save_eval_results(wf)
        log.info("\n" + "=" * 60)
        log.info("Walk-forward evaluation complete.")
        log.info("=" * 60)
        return wf

    if snaps is None:
        log.warning(
            "\n  !! Using STORED Glicko snapshots. If ranking_service.generate_rankings()\n"
            "     has not been re-run since the newcomer-seed fix, these still encode each\n"
            "     fighter's lifetime record (including future UFC results) and every metric\n"
            "     below is inflated by roughly 10 accuracy points. Pass --fresh-glicko for\n"
            "     corrected ratings, and see --walk-forward for out-of-sample numbers."
        )

    df = build_features(df, round_data, glicko_snapshots=snaps)
    matchup, features = build_matchup_df(df)

    log.warning(
        "\n  !! This is a single 80/20 split with hyperparameters that were tuned on an\n"
        "     overlapping window. Treat --walk-forward as the source of truth."
    )

    results = {}
    if 1 in phases:
        results["gbt_full"] = train_gbt(matchup, features, odds_only=False)
        if "odds_red_prob" in matchup.columns and matchup["odds_red_prob"].notna().sum() > 100:
            results["gbt_odds"] = train_gbt(matchup, features, odds_only=True)

    # Calibration + Betting Simulation (use Phase 1 GBT model).
    # Pass the selected feature list and train-fitted imputation constants so
    # calibration scores the same feature space the model was fit on.
    gbt = results.get("gbt_full")
    cal_results = calibrate_model(
        matchup,
        gbt["features"] if gbt else features,
        gbt_model=gbt["model"] if gbt else None,
        train_means=gbt["train_means"] if gbt else None,
    )
    betting_simulation(cal_results)

    log.info("\n" + "=" * 60)
    log.info("Pipeline v6 complete.")
    log.info("=" * 60)


def train_mlp(fresh_glicko: bool = True, gamma: float = 0.0,
              val_frac: float = 0.15) -> dict:
    """Fit the MLP on ALL available fights and persist it for serving.

    The ablation (scripts/ablate_decorrelation.py, 5 seeds) found gamma=0 best on every
    metric -- accuracy, Brier and ROI all degrade monotonically as gamma rises -- so the
    decorrelation penalty is off by default. What survived is the LEARNER: a small MLP
    minimising squared error rather than a GBT minimising log loss. It is slightly less
    accurate than the GBT (66.2% vs 67.9%) but agrees with the market less (0.76 vs 0.82),
    and it is the only configuration with a positive backtested ROI.

    Trained on everything up to today. The last `val_frac` of fights (chronologically) is
    held out purely for early stopping -- it is NOT a performance estimate. For that, use
    `--walk-forward`.
    """
    log.info("=" * 60)
    log.info(f"TRAINING MLP FOR SERVING (gamma={gamma})")
    log.info("=" * 60)

    snaps = run_glicko_inmemory() if fresh_glicko else None
    if fresh_glicko:
        log.info(f"  Recomputed {len(snaps)} Glicko snapshots (fixed newcomer seed)")

    df, round_data = load_fight_data()
    probe = build_matchup_df(build_features(df.copy(), round_data.copy(),
                                            glicko_snapshots=snaps))[0]
    cutoff = probe.sort_values("date")["date"].iloc[int(len(probe) * 0.6)]
    df = build_features(df, round_data, style_cutoff_date=cutoff, glicko_snapshots=snaps)
    matchup, features = build_matchup_df(df)
    matchup = matchup.sort_values("date").reset_index()

    # Only decided fights can train
    matchup = matchup[matchup["red_wins"].notna()].reset_index(drop=True)
    n = len(matchup)
    cut = int(n * (1 - val_frac))
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:cut] = True

    matchup, train_means = _fillna_from_train(matchup, features, train_mask)
    selected = select_winner_features(matchup, features, train_mask,
                                      top_n=39, include_odds=True)

    tr, va = matchup.iloc[:cut], matchup.iloc[cut:]

    def _mkt(d):
        m = devig(d["odds_red_prob"].values, d["odds_blue_prob"].values)
        return np.where(d["odds_red_prob"].notna().values, m, np.nan)

    log.info(f"  Train {len(tr)} ({tr['date'].min()} to {tr['date'].max()})")
    log.info(f"  Early-stopping holdout {len(va)} (NOT a performance estimate)")
    log.info(f"  Features: {len(selected)}")

    mdl = _decorrelated_model()(gamma=gamma, seed=42).fit(
        tr[selected].values, tr["red_wins"].values, _mkt(tr),
        va[selected].values, va["red_wins"].values, _mkt(va),
    )

    pv = mdl.predict_proba(va[selected].values)
    yv = va["red_wins"].values
    log.info(f"  Holdout acc={accuracy_score(yv, (pv >= 0.5).astype(int)):.4f} "
             f"brier={brier_score_loss(yv, pv):.4f} (early-stopping set; optimistic)")

    path = MODEL_DIR / "mlp_v1.pkl"
    mdl.save(path, selected, train_means)
    log.info(f"  Saved to {path}")
    return {"model": mdl, "features": selected, "path": path}


def build_serving_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """Build the matchup frame used at SERVING time, indexed by fight_id.

    Differs from `build_matchup_df()` in one essential way: that function is for
    TRAINING, so it keeps only decided fights from 2015 on. Serving has to cover fights
    that have not happened yet — which is the entire point of a prediction — so upcoming
    bouts are added here from each fighter's most recent feature snapshot.

    Extracted so `generate_predictions()` and the picks pipeline share one construction.
    Two frames built by two similar-looking code paths is exactly how train/serve skew
    gets in; the parity test in tests/test_leakage.py pins this against
    `build_matchup_df()`.

    Odds columns are attached but no imputation is applied — callers impute with the
    train-fitted constants persisted alongside whichever model they are serving.
    """
    feature_cols = winner_feature_columns(df)

    # Per-fighter columns keep their NaNs here. Imputation happens once, at the matchup
    # level, from the train-fitted means persisted with the model — recomputing means
    # over the serving frame would silently use different constants than training did.

    # --- Build latest feature snapshot per fighter (for upcoming fights) ---
    # Sort by date descending and take each fighter's most recent row
    df_sorted = df.sort_values("date", ascending=False)
    latest_by_fighter = df_sorted.groupby("stats_fighter_id").first()
    log.info(f"  Built latest feature snapshots for {len(latest_by_fighter)} fighters")

    red = df[df["corner"] == "red"].set_index("fight_id")
    blue = df[df["corner"] == "blue"].set_index("fight_id")
    common = red.index.intersection(blue.index)
    red, blue = red.loc[common], blue.loc[common]

    matchup = pd.DataFrame(index=common)
    matchup.index.name = "fight_id"
    matchup["date"] = red["date"].values

    # Must mirror build_matchup_df() exactly — see the parity test in tests/test_leakage.py
    fight_cols = [c for c in feature_cols if _is_fight_level(c)]
    fighter_cols = [c for c in feature_cols if not _is_fight_level(c)]

    for col in fight_cols:
        matchup[f"fight_{col}"] = red[col].values

    for col in fighter_cols:
        matchup[f"diff_{col}"] = red[col].values - blue[col].values

    raw_cols = [c for c in WINNER_RAW_COLS if c in fighter_cols]
    raw_cols += [c for c in fighter_cols if c.startswith("glicko_")]
    for col in raw_cols:
        matchup[f"red_{col}"] = red[col].values
        matchup[f"blue_{col}"] = blue[col].values

    # --- Add upcoming fights (no fight stats yet) using latest fighter features ---
    from app.models.ufc import UFCFight
    upcoming_db = SessionLocal()
    historical_fight_ids = set(common)
    upcoming_fights = (
        upcoming_db.query(UFCFight)
        .filter(UFCFight.id.notin_(historical_fight_ids))
        .all()
    )
    upcoming_db.close()

    # Fighter DOB and last-fight date, so age and layoff can be recomputed against the
    # UPCOMING fight rather than inherited from the fighter's previous bout.
    dob_db = SessionLocal()
    fighter_dob = {int(f.id): f.dob for f in dob_db.query(UFCFighter).all()}
    dob_db.close()
    last_fight_date = (
        df.groupby("stats_fighter_id")["date"].max().to_dict()
    )

    def _time_varying_overrides(fighter_id, fight_date: date | None) -> dict:
        """`age` and `days_since_last` are the only features whose value depends on WHEN
        the fight happens. Every other feature is a property of the fighter's history.

        Copying them off the fighter's last historical row — as this function used to —
        gives the fighter's age at their PREVIOUS bout and the layoff BEFORE it, which
        for a fighter returning after 18 months is wrong by 18 months in both features.
        """
        out = {}
        if not fight_date:
            return out
        dob = fighter_dob.get(int(fighter_id))
        if isinstance(dob, date):
            out["age"] = (fight_date - dob).days / 365.25
        prev = last_fight_date.get(fighter_id)
        if prev:
            out["days_since_last"] = (
                pd.Timestamp(fight_date) - pd.Timestamp(prev)
            ).days
        return out

    def _feat(feats: pd.Series, overrides: dict, col: str) -> float:
        if col in overrides:
            return float(overrides[col])
        return float(feats.get(col, 0.0) or 0.0)

    def _upcoming_fight_level(fight, col: str) -> float:
        """Fight-level features for a bout that has not happened yet."""
        if col.startswith("div_"):
            return float(_classify_weight_class(fight.weight_class) == col[len("div_"):])
        if col == "is_title_fight":
            return float(_is_title_bout(fight.weight_class))
        tf = getattr(fight, "time_format", None)
        if col == "is_five_round":
            return float(_is_five_round(tf))
        if col == "scheduled_rounds":
            return _scheduled_rounds(tf)
        if col == "scheduled_minutes":
            return _scheduled_minutes(tf)
        return 0.0

    upcoming_rows = []
    for fight in upcoming_fights:
        r_feats = latest_by_fighter.loc[fight.red_fighter_id] if fight.red_fighter_id in latest_by_fighter.index else None
        b_feats = latest_by_fighter.loc[fight.blue_fighter_id] if fight.blue_fighter_id in latest_by_fighter.index else None
        if r_feats is None or b_feats is None:
            continue  # skip if either fighter has no history

        r_over = _time_varying_overrides(fight.red_fighter_id, fight.date)
        b_over = _time_varying_overrides(fight.blue_fighter_id, fight.date)

        row = {"fight_id": fight.id, "date": fight.date}
        # Fight-level values come from the upcoming fight itself, never from either
        # fighter's history.
        for col in fight_cols:
            row[f"fight_{col}"] = _upcoming_fight_level(fight, col)
        for col in fighter_cols:
            row[f"diff_{col}"] = _feat(r_feats, r_over, col) - _feat(b_feats, b_over, col)
        for col in raw_cols:
            row[f"red_{col}"] = _feat(r_feats, r_over, col)
            row[f"blue_{col}"] = _feat(b_feats, b_over, col)
        upcoming_rows.append(row)

    if upcoming_rows:
        upcoming_df = pd.DataFrame(upcoming_rows).set_index("fight_id")
        upcoming_df.index.name = "fight_id"
        matchup = pd.concat([matchup, upcoming_df])
        log.info(f"  Added {len(upcoming_rows)} upcoming fights using latest fighter features")

    # --- Load odds from DB (must match build_matchup_df) ---
    log.info("  Loading odds for predictions...")
    from app.models.ufc import UFCFightOdds
    odds_db = SessionLocal()
    odds_rows = odds_db.query(UFCFightOdds).all()
    odds_db.close()

    odds_map = {o.fight_id: o for o in odds_rows}

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
    # Keep NaN where odds are missing — `> 0.5` on NaN would silently become 0.0
    # ("blue is favorite") rather than "unknown".
    matchup["odds_fav_is_red"] = np.where(
        matchup["odds_red_prob"].isna(), np.nan,
        (matchup["odds_red_prob"] > 0.5).astype(float),
    )
    matchup["elo_vs_odds"] = matchup["diff_elo_expected"] - matchup["odds_diff"]

    return matchup


def generate_predictions():
    """Run the served model on all fights (historical and upcoming) and store them."""
    log.info("=" * 60)
    log.info("GENERATING PREDICTIONS FOR ALL FIGHTS")
    log.info("=" * 60)

    # Prefer the MLP if it has been trained. The ablation (5 seeds, identical folds)
    # found it the only configuration with a positive backtested ROI (+2.5% at the 15%
    # edge rung vs -7% for the GBT), despite being slightly LESS accurate (66.2% vs
    # 67.9%). It agrees with the market less (corr 0.76 vs 0.82), and agreeing with the
    # market is worth exactly zero.
    mlp_path = MODEL_DIR / "mlp_v1.pkl"
    cal_path = MODEL_DIR / "calibrated_model.pkl"

    # torch may be absent in a deployment or CI image. Serving stale-but-real GBT
    # predictions beats failing the nightly job and serving nothing, so treat a missing
    # torch as "no MLP available" rather than letting ImportError escape.
    mlp_available = mlp_path.exists()
    if mlp_available:
        try:
            DecorrelatedModel = _decorrelated_model()
        except ImportError as e:
            mlp_available = False
            log.error(
                "  mlp_v1.pkl is present but torch is not installed (%s). Falling back "
                "to the GBT. Add torch to requirements.txt to serve the MLP.", e)

    if mlp_available:
        base_model, meta = DecorrelatedModel.load(mlp_path)
        features = meta["features"]
        train_means = meta["train_means"]
        model_kind = "mlp"
        log.info(f"  Loaded MLP from {mlp_path} ({len(features)} features, "
                 f"gamma={meta['gamma']})")
    elif cal_path.exists():
        with open(cal_path, "rb") as f:
            cal = pickle.load(f)
        base_model = cal["base_model"]
        features = cal["features"]
        train_means = cal.get("train_means")
        model_kind = "gbt"
        log.info(f"  Loaded GBT from {cal_path} ({len(features)} features)")
        log.warning("  No mlp_v1.pkl found — serving the GBT. Run --train-mlp for the "
                    "model the ablation selected.")
    else:
        raise FileNotFoundError(
            f"No model found. Run `--train-mlp` (preferred) or `--phase 1`."
        )

    df, round_data = load_fight_data()
    df = build_features(df, round_data)
    matchup = build_serving_matchup(df)

    # Ensure all required features exist before imputing
    for feat in features:
        if feat not in matchup.columns:
            matchup[feat] = np.nan

    # Impute with the SAME constants training used. `train_means` was read from
    # whichever artifact was loaded above (MLP or GBT) — do NOT re-read `cal` here, it
    # does not exist on the MLP path.
    if train_means is None:
        log.warning(
            "  Model artifact has no persisted train_means (pre-v4). Falling back to "
            "zero-fill; retrain to restore train/serve imputation parity."
        )
        matchup[features] = matchup[features].fillna(0.0)
    else:
        matchup[features] = matchup[features].fillna(train_means).fillna(0.0)

    X = matchup[features].values
    if model_kind == "mlp":
        raw_proba = base_model.predict_proba(X)
    else:
        raw_proba = base_model.predict_proba(X)[:, 1]

    # Use raw GBT probabilities for all-fights predictions (frontend display).
    # Calibrated model (VA) is saved separately for future picks/betting page.
    cal_proba = raw_proba
    va_low = None
    va_high = None
    log.info(f"  Using raw {model_kind.upper()} probabilities for {len(raw_proba)} "
             f"predictions (uncalibrated)")

    # Compute SHAP values for the GBT model
    import shap
    log.info("Computing SHAP values...")

    if model_kind == "mlp":
        # TreeExplainer only handles tree ensembles. For the network SHAP offers two
        # deep methods; DeepExplainer fails its own additivity check against this torch
        # build, so use GradientExplainer (expected gradients), which verifies clean.
        import torch
        import torch.nn as nn

        class _ShapWrap(nn.Module):
            """SHAP's deep explainers require a 2-D (batch, outputs) output, but _MLP
            squeezes to 1-D. Also apply the sigmoid here so attributions explain the
            PROBABILITY the site displays, not the raw logit."""

            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, t):
                return torch.sigmoid(self.net(t)).unsqueeze(-1)

        wrapped = _ShapWrap(base_model.model_).eval()
        Xs = base_model._prep(X)          # same scaling the model was fitted with
        # Background must be REAL rows, not noise: SHAP values are relative to the mean
        # prediction over this set, so it defines what "average fight" means.
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(Xs), size=min(200, len(Xs)), replace=False)
        explainer = shap.GradientExplainer(wrapped, Xs[bg_idx])
        sv = explainer.shap_values(Xs)
        shap_values_arr = np.asarray(sv[0] if isinstance(sv, list) else sv)
        if shap_values_arr.ndim == 3:      # (rows, features, 1) -> (rows, features)
            shap_values_arr = shap_values_arr[:, :, 0]
    else:
        explainer = shap.TreeExplainer(base_model)
        shap_values_arr = explainer.shap_values(X)
        # For binary classification shap_values may be a list of 2 arrays; take class 1
        if isinstance(shap_values_arr, list):
            shap_values_arr = shap_values_arr[1]

    log.info(f"  SHAP values shape: {shap_values_arr.shape}")

    # Store predictions and SHAP values in DB
    from app.models.ufc import UFCFightPrediction, UFCFightShapValue
    db = SessionLocal()
    try:
        for Model in [UFCFightShapValue, UFCFightPrediction]:
            db.query(Model).delete(synchronize_session=False)
            db.commit()
            log.info(f"  Cleared {Model.__tablename__}")

        count = 0
        shap_count = 0
        for i, (fight_id, prob) in enumerate(zip(matchup.index, cal_proba)):
            predicted_winner = "red" if prob >= 0.5 else "blue"
            confidence = abs(prob - 0.5)
            pred = UFCFightPrediction(
                fight_id=int(fight_id),
                predicted_winner=predicted_winner,
                confidence=round(float(confidence), 4),
                red_prob=round(float(prob), 4),
            )
            if va_low is not None:
                pred.va_prob_low = round(float(va_low[i]), 4)
                pred.va_prob_high = round(float(va_high[i]), 4)
            db.add(pred)
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

            # Commit every 100 fights to avoid large transactions
            if count % 100 == 0:
                db.commit()

        db.commit()
        log.info(f"Stored {count} fight predictions, {shap_count} SHAP values")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1])
    parser.add_argument("--predict", action="store_true", help="Generate predictions for all fights")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Leak-free walk-forward evaluation (with-odds vs no-odds vs market)")
    parser.add_argument("--folds", type=int, default=8, help="Walk-forward fold count")
    parser.add_argument("--train-mlp", action="store_true",
                        help="Train the MLP on all fights and save models/ufc/h2h/mlp_v1.pkl")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Decorrelation weight for --train-mlp (ablation says 0 is best)")
    parser.add_argument("--fresh-glicko", action="store_true",
                        help="Recompute Glicko in memory with the fixed pre-UFC newcomer "
                             "seed instead of using stored (leaked) snapshots")
    args = parser.parse_args()
    if args.train_mlp:
        train_mlp(fresh_glicko=True, gamma=args.gamma)
    elif args.predict:
        generate_predictions()
    elif args.walk_forward:
        run(walk_forward=True, n_folds=args.folds, fresh_glicko=args.fresh_glicko)
    else:
        run(phases=[args.phase] if args.phase else None)
