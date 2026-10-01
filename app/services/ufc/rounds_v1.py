"""Fight duration (rounds / over-under) — the served survival model.

Configuration = the winner of the walk-forward bake-off in scripts/rounds_wf.py (arm
hz_cat_v2_fine_w4): discrete-time competing-risks hazard {continue, KO, Sub}, CatBoost,
feature set v2 (method_v2 matrix + method ratings + fight-duration history + round-by-round
pace + short notice + altitude), 1.25-minute bins, 4-year recency half-life. The curve is
then ANCHORED to method_v2 cause by cause: the KO and Sub curves are each rescaled to
method_v2's KO and Sub totals (so P(still going at the final bell) = its P(goes the
distance)), keeping the hazard model's timing — so the site's survival chart, six-way grid
and O/U always agree. See _anchor_all for the walk-forward result.

Walk-forward (2,450 fights, 2022-06 -> 2026-09), anchored vs base rates (log loss):
    starts R2 -0.026, O/U 1.5 -0.029, starts R3 -0.034, O/U 2.5 -0.030, round of finish -0.037.
Against BFO closing prices it does NOT beat the market (a model+market stack is level).

Served per fight (ufc_round_predictions):
    curve       JSON [{t, s, ko, sub, red_ko, blue_ko, red_sub, blue_sub}] every 1.25 min:
                s = still going, ko/sub = cumulative finishes by cause. The red/blue split of
                each cause uses the six-way grid's ratio, held constant over time.
    p_end_r1..r5, p_decision, over_1_5 .. over_4_5, expected_minutes,
    median_finish_minute (given a finish), peak_bin_start (minute of the riskiest bin)
Past fights in the walk-forward window are served from out-of-sample curves.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.ufc.method_v2 import METHOD_DIR, attach_method_features, feature_names, orient
from app.services.ufc.round_hazard import HazardModel, cause_curves, n_bins

log = logging.getLogger("rounds_v1")
MODEL_PATH = METHOD_DIR / "rounds_v1.pkl"
OOF_PATH = METHOD_DIR / "rounds_oof.csv"
MATRIX_CACHE = METHOD_DIR / "rounds_wf_matrix.pkl"
BIN_SIZE = 1.25
HALF_LIFE_YEARS = 4.0
CATBOOST = dict(learning_rate=0.05, depth=5, l2_leaf_reg=10)
STANDARD_FORMATS = ("5-5-5", "5-5-5-5-5")


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def _per_fighter(df: pd.DataFrame, rd: pd.DataFrame) -> pd.DataFrame:
    from app.services.ufc.method_ratings import timing_features
    from app.services.ufc.round_features import pace_features, short_notice_features
    return pd.concat([timing_features(df), pace_features(df, rd), short_notice_features(df)], axis=1)


def build_training_matrix(rebuild: bool = False) -> pd.DataFrame:
    """Decided 2015+ fights with standard 5-minute-round formats: method_v2 matrix + method
    ratings + duration history + pace + short notice + altitude. Cached."""
    if MATRIX_CACHE.exists() and not rebuild:
        with open(MATRIX_CACHE, "rb") as f:
            return pickle.load(f)
    from app.services.ufc.method_v2 import build_training_matrix as method_matrix
    from app.services.ufc.model import load_fight_data
    from app.services.ufc.round_features import altitude_by_fight
    m = method_matrix(False)
    df, rd = load_fight_data()
    m = attach_method_features(m, df[["fight_id", "corner"]], _per_fighter(df, rd))
    m = m.merge(altitude_by_fight(), on="fight_id", how="left")
    tf_map = df.drop_duplicates("fight_id").set_index("fight_id")["time_format"]
    m = m[m["fight_id"].map(tf_map).isin(STANDARD_FORMATS)]
    sched = m["fight_scheduled_minutes"].to_numpy(float)
    t = m["outcome_fight_minutes"].to_numpy(float)
    cls = m["outcome_method_class"].to_numpy(int)
    keep = np.isin(sched, (15.0, 25.0)) & ~((cls == 2) & (t < sched - 0.05))  # no technical decisions
    m = m[keep].sort_values("date").reset_index(drop=True)
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    with open(MATRIX_CACHE, "wb") as f:
        pickle.dump(m, f)
    return m


def labels(m: pd.DataFrame):
    t = m["outcome_fight_minutes"].to_numpy(float)
    sched = m["fight_scheduled_minutes"].to_numpy(float)
    event = np.select([m["outcome_method_class"] == 0, m["outcome_method_class"] == 1], [1, 2], 0)
    return t, sched, event


def recency_weights(dates: pd.Series, half_life_years: float = HALF_LIFE_YEARS) -> np.ndarray:
    d = pd.to_datetime(dates)
    return 0.5 ** (((d.max() - d).dt.days.to_numpy() / 365.25) / half_life_years)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train_and_save(rebuild: bool = False, oof_csv: Path | None = None) -> HazardModel:
    """Fit on every training fight. oof_csv: the walk-forward curves for this arm (2.5-minute
    edges) from scripts/rounds_wf.py, copied for serving past fights out-of-sample."""
    m = build_training_matrix(rebuild)
    n = len(m)
    views = [orient(m, np.ones(n, bool)), orient(m, np.zeros(n, bool))]
    feats = feature_names(views[0], "full")
    t, sched, event = labels(m)
    hm = HazardModel(feats, "catboost", BIN_SIZE, params=CATBOOST)
    hm.fit(views, t, event, sched, weights=recency_weights(m["date"]))
    hm.meta = {"trained_at": datetime.utcnow().isoformat(timespec="seconds"), "n": n,
               "last_fight": str(m["date"].max()), "bin": BIN_SIZE, "half_life": HALF_LIFE_YEARS}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(hm, f)
    if oof_csv is not None and Path(oof_csv).exists():
        pd.read_csv(oof_csv, dtype={"fight_id": str}).to_csv(OOF_PATH, index=False)
    log.info(f"  saved {MODEL_PATH} ({n} fights, {len(feats)} features)")
    return hm


def load() -> HazardModel | None:
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def serving_frame() -> pd.DataFrame:
    from app.services.ufc import method_ratings
    from app.services.ufc.model import build_features, build_serving_matchup, load_fight_data
    from app.services.ufc.round_features import altitude_by_fight
    df, rd = load_fight_data(include_upcoming=True)
    per = pd.concat([method_ratings.compute(df), _per_fighter(df, rd)], axis=1)
    keys = df[["fight_id", "corner"]].copy()
    matchup = build_serving_matchup(build_features(df.copy(), rd)).reset_index()
    matchup = attach_method_features(matchup, keys, per)
    return matchup.merge(altitude_by_fight(), on="fight_id", how="left")


def _scheduled(matchup: pd.DataFrame) -> np.ndarray:
    s = matchup.get("fight_scheduled_minutes", pd.Series(np.nan, index=matchup.index)).to_numpy(float)
    five = matchup.get("fight_is_five_round", pd.Series(0.0, index=matchup.index)).to_numpy(float)
    s = np.where(np.isin(s, (15.0, 25.0)), s, np.where(five > 0.5, 25.0, 15.0))
    return s


def summarize(S, KO, SUB, H, sched, cells, bin_size=BIN_SIZE) -> list[dict]:
    """Per fight: anchored-curve summary + JSON curve. cells: (n, 6) six-way grid."""
    out = []
    step = round(5.0 / bin_size)
    for i in range(len(S)):
        K = n_bins(sched[i], bin_size)
        s, ko, sub = S[i, :K + 1], KO[i, :K + 1], SUB[i, :K + 1]
        rk = cells[i, 0] / max(cells[i, 0] + cells[i, 3], 1e-9)    # red share of KOs
        rs = cells[i, 1] / max(cells[i, 1] + cells[i, 4], 1e-9)    # red share of subs
        tgrid = np.arange(K + 1) * bin_size
        curve = [{"t": round(float(t), 3), "s": round(float(a), 4), "ko": round(float(b), 4),
                  "sub": round(float(c), 4), "red_ko": round(float(b * rk), 4),
                  "blue_ko": round(float(b * (1 - rk)), 4), "red_sub": round(float(c * rs), 4),
                  "blue_sub": round(float(c * (1 - rs)), 4)} for t, a, b, c in zip(tgrid, s, ko, sub)]
        rounds = int(round(sched[i] / 5))
        p_end = [float(s[r * step] - s[(r + 1) * step]) for r in range(rounds)]
        fin = 1 - s[-1]
        med = None
        if fin > 1e-6:
            frac = (1 - s) / fin
            med = float(np.interp(0.5, frac, tgrid))
        def over(line):
            k = round(line / bin_size)
            return float(s[k]) if k <= K else None
        out.append({
            "curve": curve, **{f"p_end_r{r + 1}": (p_end[r] if r < rounds else None) for r in range(5)},
            "p_decision": float(s[-1]), "over_1_5": over(7.5), "over_2_5": over(12.5),
            "over_3_5": over(17.5) if rounds == 5 else None, "over_4_5": over(22.5) if rounds == 5 else None,
            "expected_minutes": float(np.trapezoid(s, tgrid)), "median_finish_minute": med,
            "peak_bin_start": float(np.argmax(H[i, :K]) * bin_size) if H is not None else None,
        })
    return out


def _anchor_all(S, KO, SUB, p_ko, p_sub, p_dec, sched, bin_size=BIN_SIZE):
    """Anchor each cause to method_v2's total, keeping the hazard model's timing:
    KO(t) *= P_ko / KO(end), Sub(t) *= P_sub / Sub(end), S = 1 - KO - Sub. Then S(end) =
    P(decision) and the curve's KO / Sub totals match the six-way grid, so the survival
    chart and the winner x method grid tell the same story.

    Walk-forward (scripts/rounds_cause_anchor_wf.py, 2,390 fights): method log loss
    0.9618 -> 0.9549 and round x method 1.4963 -> 1.4894 (both CIs exclude zero) against
    anchoring the decision total only; every O/U line and round of finish unchanged
    (|diff| <= 0.0001).

    Fights with no method_v2 KO / Sub fall back to the decision-only anchor; a cause the
    hazard model gives (near) no mass borrows the total finish curve's timing."""
    rows = np.arange(len(S))
    end = np.array([n_bins(x, bin_size) for x in sched])
    s_end = S[rows, end]
    fin = np.clip(1 - s_end, 1e-4, None)
    # Decision-only anchor: total finish mass scaled to 1 - P(decision).
    p = np.where(np.isfinite(p_dec), p_dec, s_end)
    scale = ((1 - p) / fin)[:, None]
    S1, KO1, SUB1 = 1 - (1 - S) * scale, KO * scale, SUB * scale
    # Per-cause anchor where method_v2 has both totals.
    has = np.isfinite(p_ko) & np.isfinite(p_sub)
    def cause(C, target):
        c_end = C[rows, end]
        own = C * (np.where(has, target, 0) / np.clip(c_end, 1e-4, None))[:, None]
        borrowed = (1 - S) * (np.where(has, target, 0) / fin)[:, None]
        return np.where((c_end >= 1e-3)[:, None], own, borrowed)
    KO2, SUB2 = cause(KO, p_ko), cause(SUB, p_sub)
    S2 = np.minimum.accumulate(np.clip(1 - KO2 - SUB2, 0.0, 1.0), axis=1)
    pick = has[:, None]
    return np.where(pick, S2, S1), np.where(pick, KO2, KO1), np.where(pick, SUB2, SUB1)


def _oof_curves(ids: np.ndarray, sched: np.ndarray, cells: np.ndarray, p_dec: np.ndarray):
    """Walk-forward curves (2.5-minute edges) for past fights, mapped onto the 1.25 grid by
    linear interpolation; KO/Sub split from the six-way grid (no cause curves in the OOF)."""
    if not OOF_PATH.exists():
        return {}
    oof = pd.read_csv(OOF_PATH, dtype={"fight_id": str}).set_index("fight_id")
    edges = np.arange(0, 25.01, 2.5)
    fine = np.arange(0, 25.01, BIN_SIZE)
    out = {}
    for i, fid in enumerate(ids):
        if fid not in oof.index:
            continue
        s = np.interp(fine, edges, oof.loc[fid, [f"S_{e:g}" for e in edges]].to_numpy(float))
        ko_share = (cells[i, 0] + cells[i, 3]) / max(cells[i, 0] + cells[i, 3] + cells[i, 1] + cells[i, 4], 1e-9)
        out[fid] = (s, (1 - s) * ko_share, (1 - s) * (1 - ko_share))
    return out


def generate_predictions(model: HazardModel | None = None) -> int:
    """Score every fight and replace ufc_round_predictions. Run after method_v2 has written
    ufc_method_predictions (it anchors to those decision probabilities)."""
    from app.database import SessionLocal, engine
    from app.models.ufc import UFCMethodPrediction, UFCRoundPrediction

    model = model or load()
    if model is None:
        raise FileNotFoundError(f"No rounds model at {MODEL_PATH}")
    UFCRoundPrediction.__table__.create(engine, checkfirst=True)
    matchup = serving_frame()
    ids = matchup["fight_id"].astype(str).to_numpy()
    sched = _scheduled(matchup)
    db = SessionLocal()
    try:
        mp = {str(r.fight_id): r for r in db.query(UFCMethodPrediction)}
    finally:
        db.close()
    cells = np.array([[getattr(mp[f], c) if f in mp and getattr(mp[f], c) is not None else np.nan
                       for c in ("red_ko_prob", "red_sub_prob", "red_dec_prob",
                                 "blue_ko_prob", "blue_sub_prob", "blue_dec_prob")] for f in ids], float)
    p_dec = np.array([mp[f].dec_prob if f in mp else np.nan for f in ids], float)
    p_ko = np.array([mp[f].ko_prob if f in mp and mp[f].ko_prob is not None else np.nan for f in ids], float)
    p_sub = np.array([mp[f].sub_prob if f in mp and mp[f].sub_prob is not None else np.nan for f in ids], float)
    cells = np.where(np.isnan(cells), 1 / 6, cells)

    n = len(matchup)
    views = [orient(matchup, np.ones(n, bool)), orient(matchup, np.zeros(n, bool))]
    views = [v.reindex(columns=model.base_features) for v in views]
    cur = cause_curves(model, views, sched)
    S, KO, SUB = _anchor_all(cur["S"], cur["KO"], cur["SUB"], p_ko, p_sub, p_dec, sched)
    H = cur["H"]
    oof = _oof_curves(ids, sched, cells, p_dec)
    for i, fid in enumerate(ids):
        if fid in oof:
            s, ko, sub = oof[fid]
            S[i], KO[i], SUB[i] = s, ko, sub
    # The walk-forward curves are the raw arm output; anchor everything (idempotent for the
    # curves anchored above).
    S, KO, SUB = _anchor_all(S, KO, SUB, p_ko, p_sub, p_dec, sched)
    log.info(f"  {len(oof)} past fights use out-of-sample round curves")
    summaries = summarize(S, KO, SUB, H, sched, cells)

    db = SessionLocal()
    try:
        db.query(UFCRoundPrediction).delete()
        for fid, sm in zip(ids, summaries):
            db.add(UFCRoundPrediction(fight_id=int(fid), curve=json.dumps(sm.pop("curve")),
                                      **{k: (None if v is None else round(v, 4)) for k, v in sm.items()}))
        db.commit()
    finally:
        db.close()
    log.info(f"  stored {len(summaries)} round predictions")
    return len(summaries)


def round_payload(db, fight_id: int) -> dict | None:
    """API shape for one fight, or None (no prediction / table not created yet)."""
    from sqlalchemy import inspect

    from app.database import engine
    from app.models.ufc import UFCRoundPrediction
    t = UFCRoundPrediction.__table__
    if not inspect(engine).has_table(t.name, schema=t.schema):
        return None
    r = db.query(UFCRoundPrediction).filter(UFCRoundPrediction.fight_id == fight_id).first()
    if r is None:
        return None
    return {"curve": json.loads(r.curve),
            "p_end": [getattr(r, f"p_end_r{k}") for k in range(1, 6)],
            "p_decision": r.p_decision,
            "over": {"1.5": r.over_1_5, "2.5": r.over_2_5, "3.5": r.over_3_5, "4.5": r.over_4_5},
            "expected_minutes": r.expected_minutes, "median_finish_minute": r.median_finish_minute,
            "peak_bin_start": r.peak_bin_start}


def _cli() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--oof", default=str(METHOD_DIR / "rounds_wf" / "hz_cat_v2_fine_w4.csv"),
                    help="walk-forward curves (scripts/rounds_wf.py arm output) served for past "
                         "fights; anchored to method_v2 at serve time")
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    if a.train:
        train_and_save(a.rebuild, Path(a.oof))
    if a.predict:
        generate_predictions()


if __name__ == "__main__":
    from app.services.ufc import rounds_v1 as _m   # pickle-safe module path
    _m._cli()
