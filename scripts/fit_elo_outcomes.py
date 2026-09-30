"""Fit Elo outcome scores from data instead of hand-setting them.

Each bout's result is scored for the winner by its outcome type (KO, sub, UD, MD, SD,
doctor's stoppage, injury stoppage). The score for each type and the K factor are
chosen to minimise the log loss of Elo's PRE-FIGHT win probability on later fights, so a
weight only survives if it makes the next prediction better. Fight Matrix's hand-set
0.55/0.61/0.91 (split/majority/unanimous) is one point this search can find or reject.

Also fits a "deserved" variant: the winner's score blends the official result with a
stats-based probability that they won the fight (a logistic of stat differentials fit
on pre-2015 decisions), so a round-one injury stoppage counts for what the fight showed.

Also fits "judges" variants: a decision's winner score comes from the judges' mean point
margin (scorecards.margin_score, one fitted slope) instead of the UD/MD/SD buckets.

Windows: fit on decided fights 2015-2021, report on 2022+ (holdout). Ratings run from
the first fight in the data so early careers are warmed up.

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.fit_elo_outcomes
    ... python -m scripts.fit_elo_outcomes --judges   # only deserved_fm vs judge variants
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from app.database import SessionLocal
from app.models.ufc import UFCFight, UFCFightStats
from app.services.ufc.outcome_types import DRAW, VOID, WIN_TYPES, classify_outcome
from app.services.ufc.scorecards import margin_score, winner_margin

TUNE = (date(2015, 1, 1), date(2022, 1, 1))
HOLDOUT_FROM = date(2022, 1, 1)
OUT = Path(__file__).resolve().parents[1] / "models" / "ufc" / "elo_outcomes.json"
STAT_KEYS = ("sig", "kd", "td", "ctrl", "sub")
DECISIONS = ("ud", "md", "sd")


@dataclass
class Bout:
    date: date
    red: int
    blue: int
    red_won: float  # 1/0, 0.5 draw, nan void
    otype: str
    minutes: float
    stats: dict | None  # {"red": {...}, "blue": {...}}
    margin: float | None = None  # judges' mean winner margin (decisions only)


def load_bouts() -> list[Bout]:
    db = SessionLocal()
    try:
        fights = db.query(UFCFight).filter(UFCFight.date.isnot(None)).all()
        totals = {}
        for s in db.query(UFCFightStats).filter(UFCFightStats.round_number == 0).all():
            totals[(s.fight_id, s.fighter_id)] = {
                "sig": s.sig_str_landed or 0, "kd": s.kd or 0, "td": s.td_landed or 0,
                "ctrl": (s.ctrl_seconds or 0) / 60.0, "sub": s.sub_att or 0,
            }
    finally:
        db.close()
    bouts = []
    for f in sorted(fights, key=lambda f: (f.date, f.id)):
        otype = classify_outcome(f.method, f.details, f.winner_id)
        if otype == VOID and not f.method:
            continue  # unplayed
        red_won = (np.nan if otype == VOID else 0.5 if otype == DRAW
                   else float(f.winner_id == f.red_fighter_id))
        r, b = totals.get((f.id, f.red_fighter_id)), totals.get((f.id, f.blue_fighter_id))
        bouts.append(Bout(f.date, f.red_fighter_id, f.blue_fighter_id, red_won, otype,
                          max((f.fight_time_seconds or 0) / 60.0, 0.5),
                          {"red": r, "blue": b} if r and b else None,
                          winner_margin(f.details) if otype in DECISIONS else None))
    return bouts


def _diffs(b: Bout) -> np.ndarray:
    """Red-minus-blue stat differentials per minute."""
    r, bl = b.stats["red"], b.stats["blue"]
    return np.array([(r[k] - bl[k]) / b.minutes for k in STAT_KEYS])


def fit_deserved(bouts: list[Bout]) -> LogisticRegression:
    """P(red won | stats), fit on judged decisions before the tuning window."""
    X, y = [], []
    for b in bouts:
        if b.date < TUNE[0] and b.stats and b.otype in ("ud", "md", "sd"):
            X.append(_diffs(b)); y.append(b.red_won)
    return LogisticRegression(C=1.0, fit_intercept=False).fit(np.array(X), np.array(y))


def run_elo(bouts, k: float, scores: dict[str, float], deserved=None, w: float = 1.0,
            alpha: float | None = None):
    """Returns pre-fight P(red wins) per bout (nan where not scored).
    alpha: score decisions from the judges' margin (bucket score where none parsed)."""
    elo: dict[int, float] = {}
    pre = np.full(len(bouts), np.nan)
    for i, b in enumerate(bouts):
        ra, rb = elo.get(b.red, 1500.0), elo.get(b.blue, 1500.0)
        e = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        pre[i] = e
        if b.red_won != b.red_won:
            continue
        if b.otype == DRAW:
            s_red = 0.5
        else:
            s_win = scores[b.otype]
            if alpha is not None and b.margin is not None:
                s_win = margin_score(b.margin, alpha)
            s_red = s_win if b.red_won == 1 else 1 - s_win
            if deserved is not None and b.stats is not None:
                s_red = w * s_red + (1 - w) * deserved[i]
        elo[b.red] = ra + k * (s_red - e)
        elo[b.blue] = rb - k * (s_red - e)
    return pre


def _loss(pre, bouts, lo, hi) -> float:
    idx = [i for i, b in enumerate(bouts)
           if lo <= b.date < hi and b.otype not in (DRAW, VOID)]
    p = np.clip(pre[idx], 1e-6, 1 - 1e-6)
    y = np.array([bouts[i].red_won for i in idx])
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def evaluate(pre, bouts) -> dict:
    return {"tune": _loss(pre, bouts, *TUNE),
            "holdout": _loss(pre, bouts, HOLDOUT_FROM, date(2100, 1, 1))}


def fit_judges(bouts, deserved, fm) -> dict:
    """deserved_fm (the production config) refit, vs the same with judge-margin scores."""
    def unpack(x):
        return (10 + 290 / (1 + np.exp(-x[0])), 1 / (1 + np.exp(-x[1])), np.exp(x[2]))
    out = {}
    for name, use_alpha in (("deserved_fm", False), ("deserved_judges", True)):
        def obj(x):
            k, w, a = unpack(x)
            return _loss(run_elo(bouts, k, fm, deserved, w, a if use_alpha else None), bouts, *TUNE)
        r = minimize(obj, np.r_[0.0, 0.0, np.log(0.5)], method="Nelder-Mead",
                     options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-6})
        k, w, a = unpack(r.x)
        pre = run_elo(bouts, k, fm, deserved, w, a if use_alpha else None)
        out[name] = {"k": k, "w_result": w, **({"alpha": a} if use_alpha else {}),
                     **evaluate(pre, bouts)}
        # Holdout on decisions only: where the change applies.
        idx = [i for i, b in enumerate(bouts) if b.date >= HOLDOUT_FROM and b.otype not in (DRAW, VOID)]
        out[name]["holdout_n"] = len(idx)
        print(f"{name:16s} K={k:6.1f} w={w:.2f}" + (f" alpha={a:.3f}" if use_alpha else "")
              + f"  tune={out[name]['tune']:.4f}  holdout={out[name]['holdout']:.4f}")
    # What alpha implies for typical cards.
    a = out["deserved_judges"]["alpha"]
    out["deserved_judges"]["implied"] = {
        "29-28 split (0.33)": margin_score(1 / 3, a), "29-28 x3 (1.0)": margin_score(1, a),
        "30-27 x3 (3.0)": margin_score(3, a)}
    print("implied winner scores:", {k: round(v, 3) for k, v in out["deserved_judges"]["implied"].items()})
    return out


def _deserved_probs(bouts):
    model = fit_deserved(bouts)
    deserved = np.full(len(bouts), 0.5)
    for i, b in enumerate(bouts):
        if b.stats is not None:
            deserved[i] = model.predict_proba(_diffs(b).reshape(1, -1))[0, 1]
    return deserved


def main() -> None:
    bouts = load_bouts()
    print(f"{len(bouts)} bouts, {sum(b.margin is not None for b in bouts)} with judge scores")
    if "--judges" in sys.argv:
        fm = {"ko": 1.0, "sub": 1.0, "ud": 0.91, "md": 0.61, "sd": 0.55,
              "doctor": 1.0, "injury": 1.0}
        res = fit_judges(bouts, _deserved_probs(bouts), fm)
        results = json.loads(OUT.read_text()) if OUT.exists() else {}
        results["judges_comparison"] = res
        OUT.write_text(json.dumps(results, indent=2, default=float))
        print(f"saved {OUT}")
        return
    results = {}

    # 1. Plain Elo, every win scored 1.0, K fitted.
    def plain(k):
        return run_elo(bouts, k, {t: 1.0 for t in WIN_TYPES})
    best_k = min(np.arange(12, 81, 4), key=lambda k: _loss(plain(k), bouts, *TUNE))
    results["plain"] = {"k": float(best_k), **evaluate(plain(best_k), bouts)}

    # 2. Fight Matrix hand-set scores (finishes 1.0), K fitted.
    fm = {"ko": 1.0, "sub": 1.0, "ud": 0.91, "md": 0.61, "sd": 0.55,
          "doctor": 1.0, "injury": 1.0}
    best_k_fm = min(np.arange(12, 81, 4), key=lambda k: _loss(run_elo(bouts, k, fm), bouts, *TUNE))
    results["fight_matrix"] = {"k": float(best_k_fm), "scores": fm,
                               **evaluate(run_elo(bouts, best_k_fm, fm), bouts)}

    # 3. Fitted per-type scores + K. Scores live in [0.5, 1]: a win is never evidence
    #    the winner was worse, and 0.5 means "this result tells us nothing".
    def unpack(x):
        k = 10 + 290 / (1 + np.exp(-x[0]))
        return k, {t: 0.5 + 0.5 / (1 + np.exp(-np.clip(v, -30, 30))) for t, v in zip(WIN_TYPES, x[1:])}

    def obj(x):
        k, sc = unpack(x)
        return _loss(run_elo(bouts, k, sc), bouts, *TUNE)

    x0 = np.r_[0.0, np.full(len(WIN_TYPES), 2.0)]
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-6})
    k_fit, sc_fit = unpack(res.x)
    results["fitted_types"] = {"k": k_fit, "scores": sc_fit,
                               **evaluate(run_elo(bouts, k_fit, sc_fit), bouts)}

    # 4. Deserved blend on top of the fitted type scores: fit K and w.
    model = fit_deserved(bouts)
    deserved = np.full(len(bouts), 0.5)
    for i, b in enumerate(bouts):
        if b.stats is not None:
            deserved[i] = model.predict_proba(_diffs(b).reshape(1, -1))[0, 1]

    # Only K and the blend weight are fitted here; the type scores are held fixed, so
    # the tiny injury/doctor/majority samples cannot be overfit.
    def fit_blend(scores):
        def obj2(x):
            k = 10 + 290 / (1 + np.exp(-x[0])); w = 1 / (1 + np.exp(-x[1]))
            return _loss(run_elo(bouts, k, scores, deserved, w), bouts, *TUNE)
        r = minimize(obj2, np.r_[0.0, 0.0], method="Nelder-Mead",
                     options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-6})
        k = 10 + 290 / (1 + np.exp(-r.x[0])); w = 1 / (1 + np.exp(-r.x[1]))
        return {"k": k, "w_result": w, "scores": scores,
                "deserved_coef": dict(zip(STAT_KEYS, model.coef_[0].tolist())),
                **evaluate(run_elo(bouts, k, scores, deserved, w), bouts)}

    results["deserved_plain"] = fit_blend({t: 1.0 for t in WIN_TYPES})
    results["deserved_fm"] = fit_blend(fm)

    # K alone with a wider range for the two baselines, since K=80 was near the edge.
    for name, sc in (("plain", {t: 1.0 for t in WIN_TYPES}), ("fight_matrix", fm)):
        ks = np.arange(20, 301, 10)
        kb = min(ks, key=lambda k: _loss(run_elo(bouts, k, sc), bouts, *TUNE))
        results[name] = {"k": float(kb), "scores": sc, **evaluate(run_elo(bouts, kb, sc), bouts)}

    # Type counts in the tuning window, so small-sample scores can be read with care.
    counts = {t: sum(1 for b in bouts if TUNE[0] <= b.date < TUNE[1] and b.otype == t)
              for t in WIN_TYPES}
    results["tune_counts"] = counts

    OUT.write_text(json.dumps(results, indent=2, default=float))
    for name, r in results.items():
        if name == "tune_counts":
            continue
        extra = ""
        if "scores" in r:
            extra = "  " + " ".join(f"{t}={r['scores'][t]:.2f}" for t in WIN_TYPES)
        if "w_result" in r:
            extra += f"  w_result={r['w_result']:.2f}"
        print(f"{name:15s} K={r['k']:5.1f}  tune={r['tune']:.4f}  holdout={r['holdout']:.4f}{extra}")
    print("tune counts:", counts)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
