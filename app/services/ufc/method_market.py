"""Market-informed correction to P(decision | winner) for the served method grid.

The no-odds conditional model (method_v2.ConditionalMethodModel) gives P(KO / Sub / Dec |
this fighter won). It does not recognise mismatches the way the market does: when a big
market favourite wins, the fight ends early far more often than the model expects
(walk-forward 2022-06 -> 2026-09: favourites of 85%+ who won went the distance 30% of the
time, the model said 43%). A mismatch feature inside the model did not fix it (odds exist
only from 2020, too few rows; scripts/method_wf.py arms v2_mkt / v2_wp were neutral).

So the correction sits on top of the model's output, per winner side:

    logit P'(dec | W won) = a*logit P(dec | W won) + c + b1*logit(q_W) + b2*max(q_W - KNOT, 0)

q_W is the winner's de-vigged market probability (low for an underdog winner, who gets no
hinge, so one fit covers both corners). The KO:Sub split within finishes is the model's,
rescaled to 1 - P'(dec). Fights without odds keep the raw model output. The form, KNOT and
ridge C were fixed before any fit (pre-registered with the method-v2 log). Features are left
unscaled, as in the held-out test that chose this form, so the ridge acts hardest on the
small-range hinge: a deliberately conservative fit.

Fit on walk-forward conditionals (method_oof.csv), each fight's ACTUAL winner side, so the
label is "did it go to decision given this fighter won".

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m app.services.ufc.method_market --fit
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from app.services.ufc.market_anchor import devig
from app.services.ufc.method_v2 import COND_COLS, METHOD_DIR, OOF_PATH

log = logging.getLogger("method_market")
PARAMS_PATH = METHOD_DIR / "method_market.json"
KNOT = 0.70
RIDGE_C = 0.1
MIN_HINGE_ROWS = 300   # priced fights with a favourite >= KNOT needed to fit the hinge


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _features(p_dec: np.ndarray, q_w: np.ndarray, hinge: bool) -> np.ndarray:
    cols = [_logit(p_dec), _logit(q_w)]
    if hinge:
        cols.append(np.clip(np.asarray(q_w, dtype=float) - KNOT, 0, None))
    return np.column_stack(cols)


@dataclass
class DecisionCorrection:
    coef: list[float]
    intercept: float
    hinge: bool
    n_fit: int
    fitted_through: str | None = None

    def p_dec(self, p_dec: np.ndarray, q_w: np.ndarray) -> np.ndarray:
        z = _features(p_dec, q_w, self.hinge) @ np.asarray(self.coef) + self.intercept
        return 1 / (1 + np.exp(-z))

    def adjust(self, cond: np.ndarray, q_w: np.ndarray) -> np.ndarray:
        """cond (n, 3) P(KO, Sub, Dec | W won); q_w (n,) W's market prob (NaN = no odds)."""
        cond = np.asarray(cond, dtype=float)
        q_w = np.asarray(q_w, dtype=float)
        ok = np.isfinite(q_w) & np.isfinite(cond).all(axis=1)
        out = cond.copy()
        if not ok.any():
            return out
        new_dec = self.p_dec(cond[ok, 2], q_w[ok])
        fin = cond[ok, :2]
        tot = fin.sum(axis=1, keepdims=True)
        share = np.where(tot > 0, fin / np.where(tot > 0, tot, 1), 0.5)
        out[ok, :2] = share * (1 - new_dec)[:, None]
        out[ok, 2] = new_dec
        return out

    def apply(self, cond_red: np.ndarray, cond_blue: np.ndarray, q_red: np.ndarray):
        """Both corners' conditionals; q_red = red's de-vigged market prob (NaN = none)."""
        q_red = np.asarray(q_red, dtype=float)
        return self.adjust(cond_red, q_red), self.adjust(cond_blue, 1 - q_red)

    def save(self, path=PARAMS_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))


def load(path=PARAMS_PATH) -> DecisionCorrection | None:
    if not path.exists():
        return None
    return DecisionCorrection(**json.loads(path.read_text()))


def market_red_prob(frame: pd.DataFrame) -> np.ndarray:
    """De-vigged red market probability from odds_red_prob/odds_blue_prob (NaN if unpriced),
    the same conversion method_v2.orient uses."""
    if "odds_red_prob" not in frame or "odds_blue_prob" not in frame:
        return np.full(len(frame), np.nan)
    r = frame["odds_red_prob"].to_numpy(float)
    b = frame["odds_blue_prob"].to_numpy(float)
    return np.where(np.isfinite(r) & np.isfinite(b), devig(r, b), np.nan)


def training_rows(oof: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """One row per priced, decided OOF fight: the actual winner's conditional, the winner's
    market probability and whether it went to decision."""
    m = matrix.assign(fight_id=matrix["fight_id"].astype(str))
    d = oof.assign(fight_id=oof["fight_id"].astype(str)).merge(
        m[["fight_id", "date", "red_wins", "outcome_method_class", "odds_red_prob", "odds_blue_prob"]],
        on="fight_id")
    q_red = market_red_prob(d)
    red = d["red_wins"].to_numpy(float) == 1
    cr = d[list(COND_COLS[:3])].to_numpy(float)
    cb = d[list(COND_COLS[3:])].to_numpy(float)
    out = pd.DataFrame({
        "fight_id": d["fight_id"], "date": pd.to_datetime(d["date"]),
        "q_w": np.where(red, q_red, 1 - q_red),
        "y_dec": (d["outcome_method_class"].to_numpy(int) == 2).astype(int)})
    out[["w_ko", "w_sub", "w_dec"]] = np.where(red[:, None], cr, cb)
    return out[np.isfinite(out["q_w"])].reset_index(drop=True)


def fit(rows: pd.DataFrame) -> DecisionCorrection:
    from sklearn.linear_model import LogisticRegression
    fav = np.maximum(rows["q_w"], 1 - rows["q_w"])
    hinge = int((fav >= KNOT).sum()) >= MIN_HINGE_ROWS
    X = _features(rows["w_dec"].to_numpy(float), rows["q_w"].to_numpy(float), hinge)
    lr = LogisticRegression(C=RIDGE_C, max_iter=2000).fit(X, rows["y_dec"].to_numpy(int))
    return DecisionCorrection(coef=[float(c) for c in lr.coef_[0]], intercept=float(lr.intercept_[0]),
                              hinge=hinge, n_fit=len(rows),
                              fitted_through=str(rows["date"].max().date()) if len(rows) else None)


def fit_and_save() -> DecisionCorrection:
    import pickle

    from app.services.ufc.method_v2 import MATRIX_CACHE
    oof = pd.read_csv(OOF_PATH, dtype={"fight_id": str})
    with open(MATRIX_CACHE, "rb") as f:
        matrix = pickle.load(f)
    corr = fit(training_rows(oof, matrix))
    corr.save()
    log.info(f"  decision correction: coef={np.round(corr.coef, 3).tolist()} "
             f"c={corr.intercept:+.3f} hinge={corr.hinge} n={corr.n_fit} "
             f"through {corr.fitted_through} -> {PARAMS_PATH}")
    return corr


def _cli() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    if ap.parse_args().fit:
        fit_and_save()


if __name__ == "__main__":
    _cli()
