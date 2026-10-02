"""Method of victory v2: P(method | who wins), combined with the winner ensemble.

    P(Red by KO) = P(Red wins) x P(KO | Red wins)

The winner ensemble supplies P(Red wins); this module supplies the conditional part. So
the six outcomes (Red KO/Sub/Dec, Blue KO/Sub/Dec) always sum to 1, always agree with the
moneyline, and every other market is a sum of cells (method = Red + Blue per method,
goes the distance = Red Dec + Blue Dec). Log loss decomposes exactly:
    6-way LL = winner LL + conditional method LL
so this model is improved and scored without touching the winner model.

Rows are oriented to the WINNER: "given this fighter beat this opponent, how?". w_* are
the winner's features, l_* the loser's, diff_* = winner - loser. At serve time every
fight is scored twice, once as if Red won and once as if Blue won.

Two binary stages (handles the rarer submission class better than one 3-class model):
    P(finish | W won)          P(KO | finish, W won)
Each stage: CatBoost + L2 logistic averaged in logit space, fit on all training rows.
No extra calibration: walk-forward showed isotonic on a 15% holdout HURT (conditional
log loss 0.9549 -> 0.9328 without it) and the raw output is already calibrated within
~2 points per quintile. No class weighting.

Walk-forward (scripts/method_wf.py, 2,514 fights 2022-06 -> 2026-09):
    conditional LL 0.9328 vs 1.0097 base rates (-7.6%)
    method-only LL 0.9569 vs 0.9674 winner-agnostic CatBoost, CI [-0.0177, -0.0036]

Classes: 0 = KO/TKO (incl. doctor/injury stoppage), 1 = Submission, 2 = Decision.
Draws, DQ, NC and overturned bouts are excluded (the winner matrix already drops them).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from app.services.ufc.market_anchor import devig
from app.services.ufc.method_ratings import FEATURES as MR_FEATURES

KO, SUB, DEC = 0, 1, 2
CLASS_NAMES = ("KO/TKO", "Submission", "Decision")
CATBOOST_PARAMS = dict(learning_rate=0.03, depth=4, l2_leaf_reg=10)
_NOT_FEATURES = ("red_wins",)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def attach_method_features(matchup: pd.DataFrame, df: pd.DataFrame,
                           mr: pd.DataFrame) -> pd.DataFrame:
    """Adds red_<mr>/blue_<mr> columns to a matchup frame (one row per fight).
    df: the load_fight_data frame mr was computed on (fight_id, corner per row)."""
    keyed = pd.concat([df[["fight_id", "corner"]], mr], axis=1)
    out = matchup
    for corner in ("red", "blue"):
        part = keyed[keyed["corner"] == corner].drop_duplicates("fight_id")
        part = part.drop(columns="corner").set_index("fight_id")
        part.columns = [f"{corner}_{c}" for c in part.columns]
        out = out.join(part, on="fight_id")
    return out


def orient(frame: pd.DataFrame, w_is_red: np.ndarray) -> pd.DataFrame:
    """Winner-perspective feature frame. frame: matchup rows (red/blue/diff/fight cols)."""
    w_is_red = np.asarray(w_is_red, dtype=bool)
    sign = np.where(w_is_red, 1.0, -1.0)
    cols = {}
    for c in frame.columns:
        if c.startswith("diff_"):
            cols["diff_" + c[5:]] = frame[c].to_numpy(float) * sign
        elif c.startswith("fight_") and c != "fight_id":
            cols[c] = frame[c].to_numpy(float)
        elif c.startswith("red_") and c not in _NOT_FEATURES:
            base = c[4:]
            if "blue_" + base in frame.columns:
                r, b = frame[c].to_numpy(float), frame["blue_" + base].to_numpy(float)
                cols["w_" + base] = np.where(w_is_red, r, b)
                cols["l_" + base] = np.where(w_is_red, b, r)
    out = pd.DataFrame(cols, index=frame.index)
    # Mismatch: how big a favourite the winner was. Market where priced, else Elo.
    if "odds_red_prob" in frame.columns:
        mkt = devig(frame["odds_red_prob"].to_numpy(float), frame["odds_blue_prob"].to_numpy(float))
        mkt = np.where(frame["odds_red_prob"].notna().to_numpy(), mkt, np.nan)
        out["mkt_w_prob"] = np.where(w_is_red, mkt, 1 - mkt)
    # Alignment (log5-style): the winner's way of winning meets the loser's way of losing.
    def lg(c):
        return _logit(out[c].to_numpy(float)) if c in out.columns else None
    for name, a, b in (("al_ko", "w_m_ko_win_share", "l_m_ko_loss_share"),
                       ("al_sub", "w_m_sub_win_share", "l_m_sub_loss_share"),
                       ("al_dec", "w_m_dec_win_share", "l_m_dec_loss_share"),
                       ("al_pro_ko", "w_pro_ko_win_share", "l_pro_ko_loss_share"),
                       ("al_pro_sub", "w_pro_sub_win_share", "l_pro_sub_loss_share")):
        if a in out.columns and b in out.columns:
            out[name] = lg(a) + lg(b)
    for name, a, b in (("al_power_chin", "w_m_power", "l_m_chin"),
                       ("al_sub_threat", "w_m_sub_threat", "l_m_sub_vuln")):
        if a in out.columns and b in out.columns:
            out[name] = np.log(out[a].clip(lower=1e-6)) + np.log(out[b].clip(lower=1e-6))
    return out


def feature_names(oriented: pd.DataFrame, feature_set: str = "full") -> list[str]:
    """base = winner-matrix features only; full = + method ratings, Sherdog shares and
    alignment; full_mkt = full + the winner's market probability."""
    mr = tuple(MR_FEATURES)
    def is_method(c):
        return c.startswith("al_") or any(c in (f"w_{m}", f"l_{m}") for m in mr)
    cols = [c for c in oriented.columns if c != "mkt_w_prob"]
    if feature_set == "base":
        return [c for c in cols if not is_method(c)]
    if feature_set == "full":
        return cols
    if feature_set == "full_mkt":
        return cols + (["mkt_w_prob"] if "mkt_w_prob" in oriented.columns else [])
    raise ValueError(feature_set)


class _CatMember:
    def __init__(self, model, multiclass: bool = False):
        self.model, self.multiclass = model, multiclass

    def __call__(self, Z):
        p = self.model.predict_proba(Z)
        return p if self.multiclass else p[:, 1]


class _LogitMember:
    def __init__(self, model, med):
        self.model, self.med = model, med

    def __call__(self, Z):
        return self.model.predict_proba(np.where(np.isnan(Z), self.med, Z))[:, 1]


def _fit_catboost(X: np.ndarray, y: np.ndarray, loss: str):
    """Early-stop on the last 15% (chronological), then refit on everything."""
    from catboost import CatBoostClassifier
    cut = int(len(X) * 0.85)
    p = {**CATBOOST_PARAMS, "loss_function": loss, "random_seed": 42,
         "verbose": False, "allow_writing_files": False}
    probe = CatBoostClassifier(iterations=3000, od_type="Iter", od_wait=150, **p)
    probe.fit(X[:cut], y[:cut], eval_set=(X[cut:], y[cut:]), use_best_model=True)
    best = max(probe.get_best_iteration() or 100, 50)
    return CatBoostClassifier(iterations=best, **p).fit(X, y)


def _fit_binary(backend: str, X: np.ndarray, y: np.ndarray):
    if backend == "catboost":
        return _CatMember(_fit_catboost(X, y, "Logloss"))
    if backend == "logit":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        m.fit(np.where(np.isnan(X), med, X), y)
        return _LogitMember(m, med)
    raise ValueError(backend)


@dataclass
class _Stage:
    members: list
    iso: IsotonicRegression

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = np.mean([_logit(m(X)) for m in self.members], axis=0)
        return self.iso.predict(_sigmoid(z))


class _Identity:
    def predict(self, p):
        return p


def _fit_stage(X: np.ndarray, y: np.ndarray, backends, calibrate: bool = True) -> _Stage:
    if not calibrate:  # members on all rows; CatBoost early-stops on its own inner slice
        return _Stage([_fit_binary(b, X, y) for b in backends], _Identity())
    cut = int(len(X) * 0.85)
    members = [_fit_binary(b, X[:cut], y[:cut]) for b in backends]
    z = np.mean([_logit(m(X[cut:])) for m in members], axis=0)
    iso = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip")
    iso.fit(_sigmoid(z), y[cut:])
    return _Stage(members, iso)


def _fit_multiclass(X: np.ndarray, y: np.ndarray):
    return _CatMember(_fit_catboost(X, y, "MultiClass"), multiclass=True)


@dataclass
class ConditionalMethodModel:
    features: list[str]
    backends: tuple = ("catboost", "logit")
    calibrate: bool = False
    structure: str = "two_stage"  # or "multiclass": one 3-class CatBoost, no calibration
    finish: _Stage | None = None
    ko_given_finish: _Stage | None = None
    mc: object = None
    meta: dict = field(default_factory=dict)

    def fit(self, oriented: pd.DataFrame, y: np.ndarray) -> "ConditionalMethodModel":
        """oriented rows in chronological order; y in {KO, SUB, DEC} for the winner."""
        X = oriented[self.features].to_numpy(float)
        y = np.asarray(y, dtype=int)
        self.meta = {"n": int(len(y)), "shares": np.bincount(y, minlength=3) / len(y)}
        if self.structure == "multiclass":
            self.mc = _fit_multiclass(X, y)
            return self
        self.finish = _fit_stage(X, (y != DEC).astype(float), self.backends, self.calibrate)
        fin = y != DEC
        self.ko_given_finish = _fit_stage(X[fin], (y[fin] == KO).astype(float), self.backends,
                                          self.calibrate)
        return self

    def predict(self, oriented: pd.DataFrame) -> np.ndarray:
        """(n, 3) P(KO, SUB, DEC | this row's winner won)."""
        X = oriented[self.features].to_numpy(float)
        if self.structure == "multiclass":
            return self.mc(X)
        pf = self.finish.predict(X)
        pk = self.ko_given_finish.predict(X)
        return np.column_stack([pf * pk, pf * (1 - pk), 1 - pf])


def joint(p_red: np.ndarray, cond_red: np.ndarray, cond_blue: np.ndarray) -> np.ndarray:
    """(n, 6): Red KO, Red Sub, Red Dec, Blue KO, Blue Sub, Blue Dec."""
    p_red = np.asarray(p_red, dtype=float)[:, None]
    return np.hstack([p_red * cond_red, (1 - p_red) * cond_blue])


def marginal(six: np.ndarray) -> np.ndarray:
    """(n, 3) P(KO), P(SUB), P(DEC) from the 6-way grid."""
    return six[:, :3] + six[:, 3:]


# ---------------------------------------------------------------------------
# Training, walk-forward OOF and serving
# ---------------------------------------------------------------------------

import logging as _logging
import pickle as _pickle
from datetime import datetime as _dt
from pathlib import Path as _Path

log = _logging.getLogger("method_v2")
METHOD_DIR = _Path(__file__).resolve().parents[3] / "models" / "ufc" / "method"
MODEL_PATH = METHOD_DIR / "method_v2.pkl"
OOF_PATH = METHOD_DIR / "method_oof.csv"
MATRIX_CACHE = METHOD_DIR / "method_wf_matrix.pkl"
FEATURE_SET = "full"
COND_COLS = ("red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec")


def build_training_matrix(rebuild: bool = False) -> pd.DataFrame:
    """Decided fights (winner-model matrix) + red_/blue_ method ratings, date-sorted."""
    if MATRIX_CACHE.exists() and not rebuild:
        with open(MATRIX_CACHE, "rb") as f:
            return _pickle.load(f)
    from app.services.ufc import method_ratings
    from app.services.ufc.glicko_service import run_glicko_inmemory
    from app.services.ufc.model import build_features, build_matchup_df, load_fight_data
    snaps = run_glicko_inmemory()
    df, rd = load_fight_data()
    mr = method_ratings.compute(df)
    keys = df[["fight_id", "corner"]].copy()
    feat = build_features(df.copy(), rd, glicko_snapshots=snaps)
    matchup, _ = build_matchup_df(feat)
    # fight_id is the matchup index; keep it as a column (as scripts/fast_wf.py does).
    matchup = matchup.sort_values("date").reset_index()
    matchup = attach_method_features(matchup, keys, mr)
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    with open(MATRIX_CACHE, "wb") as f:
        _pickle.dump(matchup, f)
    return matchup


def _labels(matchup: pd.DataFrame) -> np.ndarray:
    return matchup["outcome_method_class"].to_numpy(int)


def walk_forward_oof(matchup: pd.DataFrame, n_folds: int = 8,
                     eval_frac: float = 0.4) -> pd.DataFrame:
    """Out-of-sample conditional probabilities for the last eval_frac of fights (same
    folds as fast_wf / ensemble_oof.csv). Columns: fight_id + COND_COLS."""
    n = len(matchup)
    eval_start = int(n * (1 - eval_frac))
    bounds = np.linspace(eval_start, n, n_folds + 1).astype(int)
    red_won = matchup["red_wins"].to_numpy(float) == 1
    or_actual = orient(matchup, red_won)
    or_red, or_blue = orient(matchup, np.ones(n, bool)), orient(matchup, np.zeros(n, bool))
    feats = feature_names(or_actual, FEATURE_SET)
    y = _labels(matchup)
    out = np.full((n, 6), np.nan)
    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        m = ConditionalMethodModel(feats).fit(or_actual.iloc[:lo], y[:lo])
        out[lo:hi, :3] = m.predict(or_red.iloc[lo:hi])
        out[lo:hi, 3:] = m.predict(or_blue.iloc[lo:hi])
        log.info(f"  OOF fold {k + 1}/{n_folds}")
    res = pd.DataFrame(out[eval_start:], columns=COND_COLS)
    res.insert(0, "fight_id", matchup["fight_id"].iloc[eval_start:].astype(str).values)
    return res


def train_and_save(rebuild: bool = True, with_oof: bool = True) -> ConditionalMethodModel:
    matchup = build_training_matrix(rebuild)
    if with_oof:
        walk_forward_oof(matchup).to_csv(OOF_PATH, index=False)
        log.info(f"  wrote {OOF_PATH}")
    red_won = matchup["red_wins"].to_numpy(float) == 1
    oriented = orient(matchup, red_won)
    model = ConditionalMethodModel(feature_names(oriented, FEATURE_SET)).fit(oriented, _labels(matchup))
    model.meta.update(trained_at=_dt.utcnow().isoformat(timespec="seconds"),
                      last_fight=str(matchup["date"].max()), feature_set=FEATURE_SET)
    with open(MODEL_PATH, "wb") as f:
        _pickle.dump(model, f)
    log.info(f"  saved {MODEL_PATH} ({model.meta['n']} fights, {len(model.features)} features)")
    return model


def load() -> ConditionalMethodModel | None:
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return _pickle.load(f)


def serving_frame() -> pd.DataFrame:
    """Every fight with both corners (past and upcoming), with method ratings attached."""
    from app.services.ufc import method_ratings
    from app.services.ufc.model import build_features, build_serving_matchup, load_fight_data
    df, rd = load_fight_data(include_upcoming=True)
    mr = method_ratings.compute(df)
    keys = df[["fight_id", "corner"]].copy()
    feat = build_features(df.copy(), rd)
    matchup = build_serving_matchup(feat).reset_index()
    return attach_method_features(matchup, keys, mr)


def predict_frame(model: ConditionalMethodModel, matchup: pd.DataFrame,
                  p_red: np.ndarray) -> pd.DataFrame:
    """Conditional + 6-way + marginal probabilities per fight. Past fights in the
    walk-forward window use their out-of-sample conditional probabilities."""
    n = len(matchup)
    # Any feature the serving frame lacks is NaN (CatBoost handles it; logit median-fills).
    or_red = orient(matchup, np.ones(n, bool)).reindex(columns=model.features)
    or_blue = orient(matchup, np.zeros(n, bool)).reindex(columns=model.features)
    cond = np.hstack([model.predict(or_red), model.predict(or_blue)])
    ids = matchup["fight_id"].astype(str).to_numpy()
    if OOF_PATH.exists():
        oof = pd.read_csv(OOF_PATH, dtype={"fight_id": str}).set_index("fight_id")
        hit = np.isin(ids, oof.index)
        cond[hit] = oof.loc[ids[hit], list(COND_COLS)].to_numpy(float)
        log.info(f"  {int(hit.sum())} past fights use out-of-sample method predictions")
    six = joint(p_red, cond[:, :3], cond[:, 3:])
    mg = marginal(six)
    out = pd.DataFrame({"fight_id": matchup["fight_id"].to_numpy()})
    for j, c in enumerate(("red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec")):
        out[f"{c}_prob"] = six[:, j]
    out["ko_prob"], out["sub_prob"], out["dec_prob"] = mg[:, 0], mg[:, 1], mg[:, 2]
    out["distance_prob"] = six[:, 2] + six[:, 5]
    return out


def generate_predictions(model: ConditionalMethodModel | None = None) -> int:
    """Score every fight and replace ufc_method_predictions. P(red wins) is the served
    winner probability (ufc_fight_predictions.red_prob), so the six cells always agree
    with the moneyline on the site; run after the winner predictions."""
    from sqlalchemy import inspect

    from app.database import SessionLocal, engine
    from app.models.ufc import UFCFightPrediction, UFCMethodPrediction

    model = model or load()
    if model is None:
        raise FileNotFoundError(f"No v2 method model at {MODEL_PATH}; run --train first")
    matchup = serving_frame()
    db = SessionLocal()
    try:
        red_prob = dict(db.query(UFCFightPrediction.fight_id, UFCFightPrediction.red_prob))
    finally:
        db.close()
    p_red = np.array([red_prob.get(int(f), np.nan) for f in matchup["fight_id"]], dtype=float)
    missing = np.isnan(p_red)
    if missing.any():
        log.warning(f"  {int(missing.sum())} fights have no winner prediction; using 0.5")
    preds = predict_frame(model, matchup, np.where(missing, 0.5, p_red))

    cols = {c["name"] for c in inspect(engine).get_columns(
        UFCMethodPrediction.__tablename__, schema=UFCMethodPrediction.__table__.schema)}
    joint_cols = [c for c in preds.columns if c.startswith(("red_", "blue_")) or c == "distance_prob"]
    has_joint = all(c in cols for c in joint_cols)
    db = SessionLocal()
    try:
        db.query(UFCMethodPrediction).delete()
        for r in preds.itertuples(index=False):
            probs = (r.ko_prob, r.sub_prob, r.dec_prob)
            k = int(np.argmax(probs))
            row = dict(fight_id=int(r.fight_id), predicted_method=CLASS_NAMES[k],
                       confidence=round(float(probs[k]), 4), ko_prob=round(float(r.ko_prob), 4),
                       sub_prob=round(float(r.sub_prob), 4), dec_prob=round(float(r.dec_prob), 4))
            if has_joint:
                row.update({c: round(float(getattr(r, c)), 4) for c in joint_cols})
            db.add(UFCMethodPrediction(**row))
        db.commit()
    finally:
        db.close()
    log.info(f"  stored {len(preds)} method predictions (joint columns: {has_joint})")
    # Fight-duration curves anchor to these decision probabilities, so they run right after.
    try:
        from app.services.ufc import rounds_v1
        if rounds_v1.load() is not None:
            rounds_v1.generate_predictions()
    except Exception as e:  # never let the rounds layer block method predictions
        log.error(f"  round predictions failed: {e}")
    # Expected stats for display turn rates into totals with those duration curves.
    try:
        from app.services.ufc import expected_stats_serving
        expected_stats_serving.generate_predictions()
    except Exception as e:  # display only: never block method / round predictions
        log.error(f"  expected-stat predictions failed: {e}")
    return len(preds)


def _cli() -> None:
    import argparse
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--no-rebuild", action="store_true", help="reuse the cached training matrix")
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    if a.train:
        train_and_save(rebuild=not a.no_rebuild)
    if a.predict:
        generate_predictions()


if __name__ == "__main__":
    # Run through the package module so pickled classes are importable later
    # (classes defined under __main__ cannot be unpickled by the app).
    from app.services.ufc import method_v2 as _m
    _m._cli()
