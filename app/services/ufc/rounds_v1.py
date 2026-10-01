"""Fight duration (rounds / over-under) — the served survival model.

Discrete-time competing-risks hazard with FOUR outcomes per 1.25-minute slice — red KO,
red submission, blue KO, blue submission (or the fight goes on) — so each fighter's
finishes get their own timing. CatBoost, feature set v2 (method_v2 matrix + method ratings
+ fight-duration history + round-by-round pace + short notice + altitude) plus the focal
fighter's and the favourite's win probability, 4-year recency half-life. Each of the four
cumulative curves is then ANCHORED to its own method_v2 six-way cell (red KO total = the
grid's red KO, etc.), keeping the hazard model's timing; still going = 1 - sum, so at the
final bell it equals P(goes the distance). Survival chart, grid and O/U always agree.

Walk-forward (scripts/rounds_cause4_wf.py, 2,390 fights) vs the previous 2-outcome model
with a fixed fighter split: better on every measure, none significant on its own — joint
who x how x round -0.0014, round of finish -0.0012, starts R2 -0.0013, O/U 1.5 -0.0008.
Against BFO closing prices it still trails the market, by about twice as much on lopsided
fights (favourite >= 70%), a gap this version narrows by ~15-20%.

Served per fight (ufc_round_predictions):
    curve       JSON [{t, s, ko, sub, red_ko, blue_ko, red_sub, blue_sub}] every 1.25 min,
                cumulative; each fighter's KO / Sub has its own timing.
    p_end_r1..r5, p_decision, over_1_5 .. over_4_5, expected_minutes,
    median_finish_minute (given a finish), peak_bin_start (minute of the riskiest bin)
Past fights in the walk-forward window are served from out-of-sample curves (OOF4_PATH).
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
from app.services.ufc.round_hazard import HazardModel, n_bins

log = logging.getLogger("rounds_v1")
MODEL_PATH = METHOD_DIR / "rounds_v1.pkl"
OOF_PATH = METHOD_DIR / "rounds_oof.csv"      # still-going at 2.5-min edges (grade table)
OOF4_PATH = METHOD_DIR / "rounds_oof4.npz"    # 4 cause curves, fine grid (serving past fights)
WP_FEATURES = ["wp_view", "wp_fav"]
CAUSES = ("red_ko", "red_sub", "blue_ko", "blue_sub")
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

def add_win_prob(views: list[pd.DataFrame], p_red: np.ndarray | None = None) -> None:
    """wp_view = the focal fighter's win probability (view 0 focal = red, view 1 = blue),
    wp_fav = the favourite's. p_red: served winner probability; else the devigged market
    (mkt_w_prob) where priced, else the Elo expectation (how the model was trained)."""
    for k, v in enumerate(views):
        if p_red is not None:
            p = pd.Series(p_red if k == 0 else 1 - np.asarray(p_red, float), index=v.index)
        else:
            p = v["mkt_w_prob"] if "mkt_w_prob" in v else pd.Series(np.nan, index=v.index)
        if "w_elo_expected" in v:
            p = p.fillna(v["w_elo_expected"])
        v["wp_view"] = p.to_numpy(float)
        v["wp_fav"] = np.maximum(v["wp_view"], 1 - v["wp_view"])


def cause_events(m: pd.DataFrame) -> list[np.ndarray]:
    """Per-view outcome codes: 1 focal KO, 2 focal Sub, 3 other KO, 4 other Sub, 0 none."""
    _, _, event = labels(m)
    fin_red = (m["red_wins"].to_numpy(float) == 1) & (event > 0)
    return [np.where(event == 0, 0, np.where(fin_red, event, event + 2)),
            np.where(event == 0, 0, np.where(~fin_red, event, event + 2))]


def train_and_save(rebuild: bool = False, oof_curves: Path | None = None) -> HazardModel:
    """Fit on every training fight. oof_curves: walk-forward 4-cause curves for the eval
    window (scripts/rounds_cause4_wf.py arm c4_wp, .npy), saved for serving past fights."""
    m = build_training_matrix(rebuild)
    n = len(m)
    views = [orient(m, np.ones(n, bool)), orient(m, np.zeros(n, bool))]
    add_win_prob(views)
    feats = feature_names(views[0], "full") + WP_FEATURES
    t, sched, event = labels(m)
    hm = HazardModel(feats, "catboost", BIN_SIZE, params=CATBOOST)
    hm.fit(views, t, event, sched, weights=recency_weights(m["date"]), events=cause_events(m))
    hm.meta = {"trained_at": datetime.utcnow().isoformat(timespec="seconds"), "n": n,
               "last_fight": str(m["date"].max()), "bin": BIN_SIZE, "half_life": HALF_LIFE_YEARS,
               "outcomes": list(CAUSES)}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(hm, f)
    if oof_curves is not None and Path(oof_curves).exists():
        start = int(n * 0.6)
        raw = np.load(oof_curves)
        ids = m["fight_id"].iloc[start:].astype(str).to_numpy()
        np.savez(OOF4_PATH, fight_id=ids.astype("U32"), curves=raw)
        # still-going at the 2.5-minute edges, for scripts/build_grade_table.py
        S = 1 - raw.sum(axis=1)
        step = int(round(2.5 / BIN_SIZE))
        out = pd.DataFrame(S[:, ::step], columns=[f"S_{e:g}" for e in np.arange(0, 25.01, 2.5)])
        out.insert(0, "fight_id", ids)
        out.to_csv(OOF_PATH, index=False)
    log.info(f"  saved {MODEL_PATH} ({n} fights, {len(feats)} features, 4 outcomes)")
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


def curves4(model: HazardModel, views: list[pd.DataFrame], sched: np.ndarray):
    """(n, 4, E) cumulative incidence (red_ko, red_sub, blue_ko, blue_sub), averaged over the
    two corner views (view 0 focal = red, view 1 focal = blue)."""
    from app.services.ufc.round_hazard import view_cause_curves
    _, C0, _ = view_cause_curves(model, views[0], sched)
    _, C1, _ = view_cause_curves(model, views[1], sched)
    return np.stack([(C0[:, 0] + C1[:, 2]) / 2, (C0[:, 1] + C1[:, 3]) / 2,
                     (C0[:, 2] + C1[:, 0]) / 2, (C0[:, 3] + C1[:, 1]) / 2], axis=1)


def anchor_cells(C: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Scale each cause curve to its six-way cell, keeping its timing. cells: (n, 6) grid
    (red_ko, red_sub, red_dec, blue_ko, blue_sub, blue_dec). A cause the hazard model gives
    ~no mass borrows the total finish curve's shape."""
    target = cells[:, [0, 1, 3, 4]]
    end = C.shape[2] - 1
    tot = C.sum(axis=1)
    out = np.empty_like(C)
    for j in range(4):
        c_end = C[:, j, end]
        own = C[:, j] * (target[:, j] / np.clip(c_end, 1e-4, None))[:, None]
        borrowed = tot * (target[:, j] / np.clip(tot[:, end], 1e-4, None))[:, None]
        out[:, j] = np.where((c_end >= 1e-3)[:, None], own, borrowed)
    return out


def summarize(C: np.ndarray, sched: np.ndarray, bin_size=BIN_SIZE) -> list[dict]:
    """Per fight: JSON curve + summary from anchored cause curves C (n, 4, E)."""
    out = []
    step = round(5.0 / bin_size)
    S_all = np.minimum.accumulate(np.clip(1 - C.sum(axis=1), 0.0, 1.0), axis=1)
    for i in range(len(C)):
        K = n_bins(sched[i], bin_size)
        s = S_all[i, :K + 1]
        rko, rsub, bko, bsub = (C[i, j, :K + 1] for j in range(4))
        tgrid = np.arange(K + 1) * bin_size
        r4 = lambda x: round(float(x), 4)
        curve = [{"t": round(float(t), 3), "s": r4(a), "ko": r4(b + d), "sub": r4(c + e),
                  "red_ko": r4(b), "blue_ko": r4(d), "red_sub": r4(c), "blue_sub": r4(e)}
                 for t, a, b, c, d, e in zip(tgrid, s, rko, rsub, bko, bsub)]
        rounds = int(round(sched[i] / 5))
        p_end = [float(s[r * step] - s[(r + 1) * step]) for r in range(rounds)]
        fin = 1 - s[-1]
        med = float(np.interp(0.5, (1 - s) / fin, tgrid)) if fin > 1e-6 else None
        haz = 1 - s[1:] / np.clip(s[:-1], 1e-9, None)

        def over(line):
            k = round(line / bin_size)
            return float(s[k]) if k <= K else None
        out.append({
            "curve": curve, **{f"p_end_r{r + 1}": (p_end[r] if r < rounds else None) for r in range(5)},
            "p_decision": float(s[-1]), "over_1_5": over(7.5), "over_2_5": over(12.5),
            "over_3_5": over(17.5) if rounds == 5 else None, "over_4_5": over(22.5) if rounds == 5 else None,
            "expected_minutes": float(np.trapezoid(s, tgrid)), "median_finish_minute": med,
            "peak_bin_start": float(np.argmax(haz) * bin_size) if len(haz) else None,
        })
    return out


def _oof4(ids: np.ndarray) -> dict:
    """Out-of-sample raw 4-cause curves for past fights in the walk-forward window."""
    if not OOF4_PATH.exists():
        return {}
    z = np.load(OOF4_PATH)
    pos = {f: k for k, f in enumerate(z["fight_id"].astype(str))}
    return {f: z["curves"][pos[f]] for f in ids if f in pos}


def generate_predictions(model: HazardModel | None = None) -> int:
    """Score every fight and replace ufc_round_predictions. Run after method_v2 has written
    ufc_method_predictions (each cause curve anchors to its six-way cell)."""
    from app.database import SessionLocal, engine
    from app.models.ufc import UFCFightPrediction, UFCMethodPrediction, UFCRoundPrediction

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
        red_prob = {str(f): p for f, p in db.query(UFCFightPrediction.fight_id, UFCFightPrediction.red_prob)}
    finally:
        db.close()
    cols = ("red_ko_prob", "red_sub_prob", "red_dec_prob", "blue_ko_prob", "blue_sub_prob", "blue_dec_prob")
    cells = np.array([[getattr(mp[f], c) if f in mp and getattr(mp[f], c) is not None else np.nan
                       for c in cols] for f in ids], float)
    cells = np.where(np.isnan(cells), 1 / 6, cells)
    p_red = np.array([red_prob.get(f, np.nan) for f in ids], float)

    n = len(matchup)
    views = [orient(matchup, np.ones(n, bool)), orient(matchup, np.zeros(n, bool))]
    add_win_prob(views, p_red)            # served winner probability; Elo where missing
    views = [v.reindex(columns=model.base_features) for v in views]
    C = curves4(model, views, sched)
    oof = _oof4(ids)
    for i, fid in enumerate(ids):
        if fid in oof:
            C[i] = oof[fid]
    C = anchor_cells(C, cells)
    log.info(f"  {len(oof)} past fights use out-of-sample round curves")
    summaries = summarize(C, sched)

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
    ap.add_argument("--oof", default=str(METHOD_DIR / "rounds_wf" / "cause4_c4_wp.npy"),
                    help="walk-forward 4-cause curves (scripts/rounds_cause4_wf.py, arm c4_wp) "
                         "served for past fights; anchored to method_v2 at serve time")
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    if a.train:
        train_and_save(a.rebuild, Path(a.oof))
    if a.predict:
        generate_predictions()


if __name__ == "__main__":
    from app.services.ufc import rounds_v1 as _m   # pickle-safe module path
    _m._cli()
