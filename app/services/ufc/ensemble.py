"""The served winner model: a four-model no-odds ensemble, blended with the market.

Two probabilities per fight:

  model_prob  P(red wins) from fight data alone — no odds anywhere in its inputs. The
              average (in log-odds) of four calibrated models on every non-odds feature:
              CatBoost, CatBoost without the raw rolling-average families, HistGBT and an
              L2 logistic regression.
  final_prob  logit(final) = a + b_mkt * logit(market) + b_model * logit(model_prob),
              fit on walk-forward OUT-OF-SAMPLE model predictions over priced fights, so
              b_model measures how much the model adds to the market on fights it had
              not seen. Falls back to model_prob when a fight has no odds.

Why this shape (walk-forward, 2022-26, same priced fights; see scripts/fast_wf.py):
  - using every feature beat the old mutual-information top-39 selection,
  - CatBoost beat HistGBT, and the four-model average beat each member,
  - the old per-fold anchor was fit on a few hundred calibration rows; fitting the blend
    on all earlier out-of-sample predictions is what took it level with the market.

Usage:
    DATABASE_URL=... python -m app.services.ufc.ensemble --train
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

log = logging.getLogger("ensemble")

#: (name, feature set, backend)
MEMBERS = (
    ("catboost", "noodds", "catboost"),
    ("catboost_noraw", "noodds_noraw", "catboost"),
    ("hgb", "noodds", "hgb"),
    # Binary short-notice flags hurt the linear member (walk-forward); trees use them.
    ("logit", "noodds_nosn", "logit"),
)
CAL_FRAC = 0.15
ARTIFACT = "ensemble_v1.pkl"

#: CatBoost settings for the catboost members. Overridable for tuning experiments.
CATBOOST_PARAMS = dict(learning_rate=0.03, depth=4, l2_leaf_reg=10)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def feature_set(name: str, features: list[str]) -> list[str]:
    from app.services.ufc.model import is_glicko_feature, is_odds_feature
    base = [f for f in features if not is_odds_feature(f)]
    if name == "noodds":
        return base
    if name == "noodds_nosn":
        return [f for f in base if not f.startswith(("diff_sn_", "red_sn_", "blue_sn_", "sn_"))]
    if name == "noodds_noglicko":
        return [f for f in base if not is_glicko_feature(f)]
    if name == "noodds_noraw":
        # Drop the raw rolling-average families. Raw career averages predicted next-fight
        # stats worse than the league mean; the opponent-adjusted xs_* replace them.
        return [f for f in base if not any(p in f for p in ("_avg_", "_recent_", "_last3_"))]
    if name == "withodds":
        return list(features)
    raise ValueError(name)


def _swapped(X: np.ndarray, names: list[str]) -> np.ndarray:
    """The corner-mirrored copy of X."""
    from app.services.ufc.model import _corner_swap_augment
    Xa, _ = _corner_swap_augment(X, np.zeros(len(X)), names)
    return Xa[len(X):]


class _Symmetric:
    """Wraps a fitted classifier so P(red) = (f(x) + 1 - f(swap x)) / 2."""

    def __init__(self, model, names):
        self.model, self.names = model, names

    def predict(self, X):
        a = self.model.predict_proba(X)[:, 1]
        b = self.model.predict_proba(_swapped(X, self.names))[:, 1]
        return (a + 1 - b) / 2


class _Plain:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


def fit_backend(backend: str, X: np.ndarray, y: np.ndarray, names: list[str],
                params: dict | None = None):
    """Fit one backend on chronologically ordered rows. Returns an object with
    .predict(X) -> P(red). Every backend trains on corner-swap-augmented data."""
    from app.services.ufc.model import _corner_swap_augment, fit_gbt

    if backend == "hgb":
        m, _ = fit_gbt(X, y, names)  # augments and early-stops internally
        return _Plain(m)
    if backend == "catboost":
        from catboost import CatBoostClassifier
        cut = int(len(X) * 0.85)  # chronological early-stopping slice
        Xf, yf = _corner_swap_augment(X[:cut], y[:cut], names)
        Xv, yv = _corner_swap_augment(X[cut:], y[cut:], names)
        params = {**CATBOOST_PARAMS, **(params or {}), "loss_function": "Logloss",
                  "random_seed": 42, "verbose": False, "allow_writing_files": False}
        probe = CatBoostClassifier(iterations=3000, od_type="Iter", od_wait=150, **params)
        probe.fit(Xf, yf, eval_set=(Xv, yv), use_best_model=True)
        best = max(probe.get_best_iteration() or 100, 50)
        Xa, ya = _corner_swap_augment(X, y, names)
        return _Symmetric(CatBoostClassifier(iterations=best, **params).fit(Xa, ya), names)
    if backend == "logit":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        Xa, ya = _corner_swap_augment(X, y, names)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        return _Symmetric(m.fit(Xa, ya), names)
    if backend == "xgboost":
        from xgboost import XGBClassifier
        cut = int(len(X) * 0.85)
        Xf, yf = _corner_swap_augment(X[:cut], y[:cut], names)
        Xv, yv = _corner_swap_augment(X[cut:], y[cut:], names)
        p = dict(n_estimators=3000, learning_rate=0.03, max_depth=3, subsample=0.8,
                 colsample_bytree=0.6, min_child_weight=5, reg_lambda=5.0,
                 eval_metric="logloss", random_state=42, n_jobs=4)
        p.update(params or {})
        probe = XGBClassifier(early_stopping_rounds=150, **p).fit(Xf, yf, eval_set=[(Xv, yv)],
                                                                  verbose=False)
        best = max(int(probe.best_iteration or 100), 50)
        Xa, ya = _corner_swap_augment(X, y, names)
        return _Symmetric(XGBClassifier(**{**p, "n_estimators": best}).fit(Xa, ya), names)
    if backend == "tabpfn":
        from tabpfn import TabPFNClassifier
        Xa, ya = _corner_swap_augment(X, y, names)
        m = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True)
        return _Symmetric(m.fit(Xa, ya), names)
    raise ValueError(backend)


@dataclass
class Member:
    name: str
    features: list[str]
    model: object
    calibrator: IsotonicRegression

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict(frame[self.features].to_numpy(float))
        return self.calibrator.predict(raw)


@dataclass
class Ensemble:
    members: list[Member]
    train_means: pd.Series
    stack: dict | None = None  # {"a", "b_mkt", "b_model", "n"}
    meta: dict = field(default_factory=dict)
    #: Blend weights fit against OPENING lines (BestFightOdds history). An opener is less
    #: informed than a day-before line, so the model earns more weight against it. Used
    #: when the market price being blended is an opening line (forward test, line watcher).
    stack_open: dict | None = None

    @property
    def features(self) -> list[str]:
        return list(dict.fromkeys(f for m in self.members for f in m.features))

    def _impute(self, frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.copy()
        for c in self.features:
            if c not in f.columns:
                f[c] = np.nan
        f[self.features] = f[self.features].fillna(self.train_means).fillna(0.0)
        return f

    def model_prob(self, frame: pd.DataFrame) -> np.ndarray:
        f = self._impute(frame)
        return _sigmoid(np.mean([_logit(m.predict(f)) for m in self.members], axis=0))

    def final_prob(self, model_prob: np.ndarray, market_prob: np.ndarray,
                   opening: bool = False) -> np.ndarray:
        market_prob = np.asarray(market_prob, dtype=float)
        s = (getattr(self, "stack_open", None) or self.stack) if opening else self.stack
        if not s:
            return model_prob
        blended = _sigmoid(s["a"] + s["b_mkt"] * _logit(market_prob)
                           + s["b_model"] * _logit(model_prob))
        return np.where(np.isnan(market_prob), model_prob, blended)


def fit_ensemble(train: pd.DataFrame, features: list[str]) -> Ensemble:
    """Fit every member on `train` (chronological). The last CAL_FRAC is held out for
    each member's isotonic calibration and never used to fit it."""
    from app.services.ufc.model import _fillna_from_train

    all_cands = feature_set("noodds", features)
    mask = np.ones(len(train), dtype=bool)
    filled, means = _fillna_from_train(train, all_cands, mask)
    cut = int(len(filled) * (1 - CAL_FRAC))
    fit_df, cal_df = filled.iloc[:cut], filled.iloc[cut:]
    y_fit = fit_df["red_wins"].to_numpy(float)
    y_cal = cal_df["red_wins"].to_numpy(float)
    members = []
    for name, fset, backend in MEMBERS:
        cols = feature_set(fset, features)
        model = fit_backend(backend, fit_df[cols].to_numpy(float), y_fit, cols)
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso.fit(model.predict(cal_df[cols].to_numpy(float)), y_cal)
        members.append(Member(name, cols, model, iso))
        log.info(f"    member {name}: {len(cols)} features")
    return Ensemble(members, pd.Series(means))


def walk_forward_oof(matchup: pd.DataFrame, features: list[str], n_folds: int = 8,
                     eval_frac: float = 0.4) -> pd.DataFrame:
    """Out-of-sample model_prob for the last `eval_frac` of fights, fold by fold."""
    n = len(matchup)
    bounds = np.linspace(int(n * (1 - eval_frac)), n, n_folds + 1).astype(int)
    out = np.full(n, np.nan)
    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        ens = fit_ensemble(matchup.iloc[:lo], features)
        out[lo:hi] = ens.model_prob(matchup.iloc[lo:hi])
        log.info(f"  OOF fold {k + 1}/{n_folds}: {lo}..{hi}")
    res = matchup[["fight_id", "date", "red_wins", "odds_red_prob"]].copy()
    res["model_prob"] = out
    return res.iloc[bounds[0]:].reset_index(drop=True)


def fit_stack(oof: pd.DataFrame) -> dict:
    priced = oof[oof["odds_red_prob"].notna() & oof["model_prob"].notna()]
    X = np.c_[_logit(priced["odds_red_prob"]), _logit(priced["model_prob"])]
    m = LogisticRegression(C=10).fit(X, priced["red_wins"].to_numpy(float))
    return {"a": float(m.intercept_[0]), "b_mkt": float(m.coef_[0][0]),
            "b_model": float(m.coef_[0][1]), "n": int(len(priced))}


def fit_open_stack(oof: pd.DataFrame) -> dict | None:
    """Blend weights against BFO consensus OPENING lines, on the same out-of-sample
    model predictions. None when no opening-line history is available."""
    from sqlalchemy import inspect

    from app.database import SessionLocal, engine
    from app.models.ufc import UFCFightOpenClose

    t = UFCFightOpenClose.__table__
    if not inspect(engine).has_table(t.name, schema=t.schema):
        return None
    db = SessionLocal()
    try:
        opens = {fid: p for fid, p in db.query(UFCFightOpenClose.fight_id,
                                               UFCFightOpenClose.red_open_prob)
                 .filter(UFCFightOpenClose.bookmaker == "Consensus",
                         UFCFightOpenClose.red_open_prob.isnot(None))}
    finally:
        db.close()
    o = oof.copy()
    o["odds_red_prob"] = [opens.get(int(f)) for f in o["fight_id"]]
    o = o[o["odds_red_prob"].notna()]
    return fit_stack(o) if len(o) >= 200 else None


def expanding_stack_eval(oof: pd.DataFrame, n_blocks: int = 8) -> dict:
    """Honest estimate of the blend: each block is scored by a stack fit only on the
    blocks before it. Returns log losses on the scored priced fights."""
    df = oof[oof["odds_red_prob"].notna()].reset_index(drop=True)
    blocks = np.array_split(np.arange(len(df)), n_blocks)
    pred = np.full(len(df), np.nan)
    for i in range(1, n_blocks):
        s = fit_stack(df.iloc[np.concatenate(blocks[:i])])
        te = blocks[i]
        pred[te] = _sigmoid(s["a"] + s["b_mkt"] * _logit(df["odds_red_prob"].iloc[te])
                            + s["b_model"] * _logit(df["model_prob"].iloc[te]))
    ok = ~np.isnan(pred)
    y = df["red_wins"].to_numpy(float)[ok]

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    return {"n": int(ok.sum()), "market": ll(df["odds_red_prob"].to_numpy()[ok]),
            "model": ll(df["model_prob"].to_numpy()[ok]), "final": ll(pred[ok])}


def train_and_save(fresh_glicko: bool = True) -> Ensemble:
    from app.services.ufc.glicko_service import run_glicko_inmemory
    from app.services.ufc.model import (
        MODEL_DIR, build_features, build_matchup_df, load_fight_data,
    )

    snaps = run_glicko_inmemory() if fresh_glicko else None
    df, rd = load_fight_data()
    df = build_features(df, rd, glicko_snapshots=snaps)
    matchup, features = build_matchup_df(df)
    matchup = matchup.sort_values("date").reset_index()
    matchup = matchup[matchup["red_wins"].notna()].reset_index(drop=True)

    log.info("Walk-forward out-of-sample predictions (for the market blend)...")
    oof = walk_forward_oof(matchup, features)
    report = expanding_stack_eval(oof)
    log.info(f"  Honest blend eval on {report['n']} priced fights: market "
             f"{report['market']:.4f}  model {report['model']:.4f}  final {report['final']:.4f}")

    log.info("Fitting final ensemble on all fights...")
    ens = fit_ensemble(matchup, features)
    ens.stack = fit_stack(oof)
    ens.stack_open = fit_open_stack(oof)
    ens.meta = {"trained_at": datetime.now().isoformat(timespec="seconds"),
                "n_train": int(len(matchup)), "last_fight": str(matchup["date"].max()),
                "eval": report, "members": [m[0] for m in MEMBERS]}
    log.info(f"  Stack: {ens.stack}")
    with open(MODEL_DIR / ARTIFACT, "wb") as f:
        pickle.dump(ens, f)
    oof.to_csv(MODEL_DIR / "ensemble_oof.csv", index=False)
    with open(MODEL_DIR / "ensemble_meta.json", "w") as f:
        json.dump({**ens.meta, "stack": ens.stack}, f, indent=2, default=str)
    log.info(f"Saved {MODEL_DIR / ARTIFACT}")
    return ens


def load() -> Ensemble | None:
    from app.services.ufc.model import MODEL_DIR
    path = MODEL_DIR / ARTIFACT
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    if a.train:
        # Go through the package path so pickled classes are importable as
        # app.services.ufc.ensemble.*, not __main__.*.
        from app.services.ufc import ensemble as _ensemble
        _ensemble.train_and_save()
