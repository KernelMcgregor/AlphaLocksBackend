"""Expected stats for DISPLAY: fills ufc_expected_stat_predictions.

Uses expected-stats v2 with Glicko covariates, the most accurate stat forecaster in
scripts/xs_eval.py. It is deliberately not the winner-model configuration: Glicko inside
xs makes the winner ensemble worse, but nothing here feeds a model.

Per fighter per bout:
  {stat}_rate         expected per-minute rate vs this opponent
  {stat}_expected     expected bout total = rate x expected fight length
  {stat}_p10 / _p90   10th / 90th percentile of the bout total, from a negative-binomial
                      mixture over the fight-length curve in ufc_round_predictions
                      (control: zero hurdle + NB, since ~1/3 of fighters get no control)
  {stat}_if_distance  expected total if the bout goes the scheduled distance
  sig_p_more/td_p_more  P(lands strictly more than the opponent), ties excluded
  ctrl_share          expected share of the bout spent in control

Past bouts carry their walk-forward values, so "expected vs actual" is out of sample.
Dispersion and the control hurdle are fitted on played bouts from 2015 on, using each
bout's actual length. Accuracy (report window 05/2022-09/2026): see alocks-docs
models/expected-stats.md. Sig-strike totals are wide by nature (average miss ~25 per
fighter, a third of it from not knowing the fight length): show ranges, not points.

Run after rounds_v1.generate_predictions (it reads the fight-length curves):
    python -m app.services.ufc.expected_stats_serving --predict
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("expected_stats_serving")

MODEL_VERSION = "xs_v2_glicko"
DIST_FROM = pd.Timestamp("2015-01-01")
#: stat key (xs_{stat}_for on compute_expected_stats output) -> actual count column
STAT_COLS = {"sig": "sig_str_landed", "td": "td_landed", "kd": "kd", "sub": "sub_att",
             "ctrl": "ctrl_seconds"}
SUPPORT = {"sig": np.arange(0, 451), "td": np.arange(0, 31), "kd": np.arange(0, 9),
           "sub": np.arange(0, 21), "ctrl": np.arange(0, 1501, 5)}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def _attach_glicko(df: pd.DataFrame) -> pd.DataFrame:
    """Stored pre-fight Glicko snapshots as glicko_{dim} + glicko_meta_rounds_seen, the
    same columns build_features adds (upcoming bouts have snapshots too)."""
    from app.database import SessionLocal
    from app.models.ufc import UFCGlickoSnapshot
    from app.services.ufc.glicko_service import DIMENSIONS
    db = SessionLocal()
    try:
        snaps = db.query(UFCGlickoSnapshot).all()
    finally:
        db.close()
    rec = pd.DataFrame([{"fight_id": s.fight_id, "stats_fighter_id": s.fighter_id,
                         **{f"glicko_{d}": getattr(s, d, None) for d in DIMENSIONS},
                         "glicko_meta_rounds_seen": getattr(s, "meta_rounds_seen", None)}
                        for s in snaps])
    if rec.empty:
        log.warning("  no Glicko snapshots: expected stats fall back to no covariates")
        return df
    log.info(f"  attached {len(rec)} Glicko snapshots")
    return df.merge(rec, on=["fight_id", "stats_fighter_id"], how="left")


def _length_curves(fight_ids) -> dict:
    """fight_id -> (t, s) survival curve from ufc_round_predictions."""
    from app.database import SessionLocal
    from app.models.ufc import UFCRoundPrediction
    db = SessionLocal()
    try:
        rows = db.query(UFCRoundPrediction.fight_id, UFCRoundPrediction.curve).all()
    finally:
        db.close()
    want = set(fight_ids)
    out = {}
    for fid, curve in rows:
        if fid in want and curve:
            c = json.loads(curve)
            out[fid] = (np.array([p["t"] for p in c], float), np.array([p["s"] for p in c], float))
    return out


def _length_dist(curve, end: float):
    """Discrete fight length: finishes at bin midpoints, decision at the scheduled end."""
    t, s = curve
    keep = t <= end + 1e-9
    t, s = t[keep], s[keep]
    T = np.r_[(t[:-1] + t[1:]) / 2, end]
    P = np.r_[np.clip(s[:-1] - s[1:], 0, None), max(s[-1], 0.0)]
    return T, P / P.sum()


# ---------------------------------------------------------------------------
# predictive distributions
# ---------------------------------------------------------------------------

def _nb_theta(y, mu) -> float:
    """Method-of-moments NB size: Var = mu + mu^2/theta."""
    k = ((y - mu) ** 2 - mu).sum() / (mu ** 2).sum()
    return 1.0 / max(k, 1e-4)


def fit_dispersion(rows: pd.DataFrame) -> dict:
    """NB size per stat and the control zero hurdle, P(0) = sigmoid(a + b log mu)."""
    from sklearn.linear_model import LogisticRegression
    h = rows[(rows["date"] >= DIST_FROM) & (rows["minutes"] > 0)]
    out = {}
    for s, col in STAT_COLS.items():
        ok = h[f"xs_{s}_for"].notna() & h[col].notna()
        y = h.loc[ok, col].to_numpy(float)
        mu = h.loc[ok, f"xs_{s}_for"].to_numpy(float) * h.loc[ok, "minutes"].to_numpy(float)
        if s != "ctrl":
            out[s] = {"theta": _nb_theta(y, mu)}
            continue
        x = np.log(np.clip(mu, 1e-3, None))[:, None]
        lr = LogisticRegression(C=1e3).fit(x, (y == 0).astype(int))
        p0 = lr.predict_proba(x)[:, 1]
        pos = y > 0
        out[s] = {"theta": _nb_theta(y[pos], mu[pos] / (1 - p0[pos])),
                  "a": float(lr.intercept_[0]), "b": float(lr.coef_[0, 0])}
    return out


def _cdf(stat: str, mu: np.ndarray, disp: dict, support: np.ndarray) -> np.ndarray:
    """CDF over support for NB means mu (n,), with the control hurdle."""
    th = disp["theta"]
    mu = np.clip(mu, 1e-6, None)
    if stat != "ctrl":
        return stats.nbinom.cdf(support[None, :], th, (th / (th + mu))[:, None])
    p0 = 1 / (1 + np.exp(-(disp["a"] + disp["b"] * np.log(np.clip(mu, 1e-3, None)))))
    m = mu / (1 - p0)
    q = th / (th + m)
    f0 = stats.nbinom.pmf(0, th, q)
    Fpos = (stats.nbinom.cdf(support[None, :], th, q[:, None]) - f0[:, None]) / (1 - f0[:, None])
    return p0[:, None] + (1 - p0[:, None]) * np.clip(Fpos, 0, 1)


def _quantile(cdf: np.ndarray, support: np.ndarray, q: float) -> float:
    return float(support[min(np.searchsorted(cdf, q), len(support) - 1)])


def _p_more(rate_a, rate_b, T, P, disp, support) -> float:
    """P(A > B | not tied), A and B independent given the fight length."""
    th = disp["theta"]
    qa = th / (th + np.maximum(rate_a * T, 1e-6))
    qb = th / (th + np.maximum(rate_b * T, 1e-6))
    fa = stats.nbinom.pmf(support[None, :], th, qa[:, None])      # (bins, support)
    fb = stats.nbinom.pmf(support[None, :], th, qb[:, None])
    pa = (P * (fa[:, 1:] * np.cumsum(fb, axis=1)[:, :-1]).sum(axis=1)).sum()
    pb = (P * (fb[:, 1:] * np.cumsum(fa, axis=1)[:, :-1]).sum(axis=1)).sum()
    return pa / (pa + pb) if pa + pb > 0 else 0.5


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def build_rows() -> list[dict]:
    from app.services.ufc.expected_stats import compute_expected_stats
    from app.services.ufc.model import load_fight_data

    df, _ = load_fight_data(include_upcoming=True)
    df = _attach_glicko(df)
    t0 = time.time()
    xs = compute_expected_stats(df, v2=True, glicko=True)
    log.info(f"  expected stats computed in {time.time() - t0:.0f}s")
    rows = df[["fight_id", "stats_fighter_id", "date", "fight_time_seconds", "time_format"]
              + list(STAT_COLS.values())].merge(xs, on=["fight_id", "stats_fighter_id"])
    rows["date"] = pd.to_datetime(rows["date"])
    rows["minutes"] = rows["fight_time_seconds"].astype(float) / 60.0
    rows = rows[rows["xs_sig_for"].notna()].reset_index(drop=True)
    disp = fit_dispersion(rows)
    log.info("  dispersion: " + ", ".join(f"{s} theta={d['theta']:.2f}" for s, d in disp.items()))

    curves = _length_curves(rows["fight_id"].unique())
    log.info(f"  fight-length curves for {len(curves)} of {rows['fight_id'].nunique()} bouts")
    end = np.where(rows["time_format"].fillna("") == "5-5-5-5-5", 25.0, 15.0)
    rate = {s: rows[f"xs_{s}_for"].to_numpy(float) for s in STAT_COLS}
    rate_against = {s: rows[f"xs_{s}_against"].to_numpy(float) for s in STAT_COLS}

    out = []
    t0 = time.time()
    for i, r in enumerate(rows.itertuples(index=False)):
        if i and i % 5000 == 0:
            log.info(f"  distributions: {i}/{len(rows)} rows ({time.time() - t0:.0f}s)")
        rec = {"fight_id": int(r.fight_id), "fighter_id": int(r.stats_fighter_id),
               "model_version": MODEL_VERSION, "ctrl_share": rate["ctrl"][i] / 60.0}
        dist = _length_dist(curves[r.fight_id], end[i]) if r.fight_id in curves else None
        for s in STAT_COLS:
            rec[f"{s}_rate"] = rate[s][i]
            rec[f"{s}_if_distance"] = rate[s][i] * end[i]
            if dist is None:
                continue
            T, P = dist
            rec[f"{s}_expected"] = rate[s][i] * float((T * P).sum())
            cdf = (P[:, None] * _cdf(s, rate[s][i] * T, disp[s], SUPPORT[s])).sum(axis=0)
            rec[f"{s}_p10"] = _quantile(cdf, SUPPORT[s], 0.10)
            rec[f"{s}_p90"] = _quantile(cdf, SUPPORT[s], 0.90)
        if dist is not None:
            # xs_{s}_against is the opponent's expected rate in this same bout
            for s in ("sig", "td"):
                rec[f"{s}_p_more"] = _p_more(rate[s][i], rate_against[s][i], *dist, disp[s], SUPPORT[s])
        out.append({k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else
                        (round(float(v), 4) if isinstance(v, (float, np.floating)) else v))
                    for k, v in rec.items()})
    return out


def generate_predictions() -> int:
    """Replace ufc_expected_stat_predictions with every bout's expected stats."""
    from app.database import engine
    from app.models.ufc import UFCExpectedStatPrediction
    table = UFCExpectedStatPrediction.__table__
    table.create(engine, checkfirst=True)
    rows = build_rows()
    with engine.begin() as conn:
        conn.execute(table.delete())
        for k in range(0, len(rows), 2000):
            conn.execute(table.insert(), rows[k:k + 2000])
    log.info(f"  stored {len(rows)} expected-stat predictions")
    return len(rows)


def expected_stats_payload(db, fight_id: int) -> dict | None:
    """API shape for one bout: {"red": {...}, "blue": {...}}, or None."""
    from sqlalchemy import inspect

    from app.database import engine
    from app.models.ufc import (
        XS_DISPLAY_FIELDS, XS_DISPLAY_STATS, UFCExpectedStatPrediction, UFCFight,
    )
    t = UFCExpectedStatPrediction.__table__
    if not inspect(engine).has_table(t.name, schema=t.schema):
        return None
    fight = db.query(UFCFight).filter(UFCFight.id == fight_id).first()
    rows = {r.fighter_id: r for r in db.query(UFCExpectedStatPrediction)
            .filter(UFCExpectedStatPrediction.fight_id == fight_id)}
    if fight is None or not rows:
        return None
    out = {"model_version": next(iter(rows.values())).model_version}
    for corner, fid in (("red", fight.red_fighter_id), ("blue", fight.blue_fighter_id)):
        r = rows.get(fid)
        if r is None:
            out[corner] = None
            continue
        out[corner] = {s: {f: getattr(r, f"{s}_{f}") for f in XS_DISPLAY_FIELDS}
                       for s in XS_DISPLAY_STATS}
        out[corner].update(sig_p_more=r.sig_p_more, td_p_more=r.td_p_more, ctrl_share=r.ctrl_share)
    return out


def _cli() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    if ap.parse_args().predict:
        generate_predictions()


if __name__ == "__main__":
    _cli()
