"""Pick grades = the expected ROI of an edge band in a market, estimated from its backtest.

scripts/build_grade_table.py rebuilds every bet a simple rule would have made in walk-forward
backtests (per fight and market, the side with the highest EV if EV > 0, flat 1 unit at the
TYPICAL book's price -- a price a reader can actually get), groups the bets by market family
and EV band, and records each band's realised ROI. Prop probabilities are the model blended
with the market (blend_prob), exactly as the picks API serves them.

A band's raw ROI is a noisy estimate of its true ROI: a 17-bet band in a market that loses
14% overall came out at +53%, and 120 longshot bets can show +58% on a handful of hits. So
the graded number is the band's ROI estimated with empirical-Bayes shrinkage (two levels):
    market ROI    shrunk toward 0 ("no edge until shown") by its standard error
    band ROI      shrunk toward its market's estimate by the band's standard error
The standard error is the sd of the band's per-bet profit / sqrt(n), so small samples and
long odds are discounted more. Large, consistent results keep their size; flukes do not.
The letter is that estimate on the ROI scale below. Raw ROI, n and a bootstrap 95% CI are
kept beside it for display. Badges never change a grade.

Families (side-specific where the sides behave differently, e.g. decision yes vs no):
  winner_open, winner_close, sixway_{ko,sub,dec}_{fav,dog} (winner x method, per method,
  for the favourite / underdog), decision_yes, decision_no, itd_yes, itd_no, ou_1_5_over, ou_1_5_under,
  ou_2_5_over, ou_2_5_under, starts_r2_yes, starts_r2_no, starts_r3_yes, starts_r3_no
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

GRADE_TABLE_PATH = Path(__file__).resolve().parents[3] / "models" / "ufc" / "grade_table.json"
#: EV bands. Wide enough that each holds hundreds of historical bets in the main markets
#: (finer bands were noise: in one market the 25-40% band lost money between two that won).
BUCKETS = [0.0, 0.03, 0.08, 0.15, float("inf")]
#: Weight on the model in the model+market blend used for prop probabilities (logit space).
#: A 50/50 blend beat both the model and the closing market in the method benchmark.
PROP_MODEL_WEIGHT = 0.5
THRESHOLDS = [("A+", 0.08), ("A", 0.05), ("A-", 0.035), ("B+", 0.025), ("B", 0.015),
              ("B-", 0.005), ("C+", 0.0), ("C", -0.015), ("C-", -0.03), ("D", -0.06)]
GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
SMALL_SAMPLE_N = 50
#: Prior sd of a market's true ROI (shrinks markets toward 0) and of a band's true ROI around
#: its market's (shrinks bands toward their market).
TAU_FAMILY = 0.05
TAU_BAND = 0.04


def decimal(american: float | None) -> float | None:
    if american is None or american != american or abs(american) < 100:
        return None   # American odds are never strictly between -100 and +100
    return 1 + (american / 100 if american > 0 else 100 / -american)


def ev(p: float, american: float | None) -> float | None:
    d = decimal(american)
    return None if d is None or p is None or p != p else p * d - 1


def blend_prob(p_model: float | None, q_market: float | None, w: float = PROP_MODEL_WEIGHT):
    """Model and de-vigged market probability averaged in log-odds. Symmetric: the blend of
    the No side is 1 - the blend of the Yes side. Falls back to the model without a market."""
    if p_model is None or p_model != p_model:
        return p_model
    if q_market is None or q_market != q_market:
        return p_model
    lg = lambda x: math.log(min(max(x, 1e-4), 1 - 1e-4) / (1 - min(max(x, 1e-4), 1 - 1e-4)))
    return 1 / (1 + math.exp(-(w * lg(p_model) + (1 - w) * lg(q_market))))


def letter(croi: float) -> str:
    for g, t in THRESHOLDS:
        if croi >= t:
            return g
    return "F"


def notch(grade: str, by: int) -> str:  # kept for callers; grades are not notched
    """Move a grade by `by` notches (negative = worse)."""
    if grade not in GRADE_ORDER:
        return grade
    i = min(max(GRADE_ORDER.index(grade) - by, 0), len(GRADE_ORDER) - 1)
    return GRADE_ORDER[i]


def cap(grade: str, at: str) -> str:
    return at if GRADE_ORDER.index(grade) < GRADE_ORDER.index(at) else grade


def bucket_of(e: float) -> int:
    for i in range(len(BUCKETS) - 1):
        if BUCKETS[i] <= e < BUCKETS[i + 1]:
            return i
    return len(BUCKETS) - 2


def _boot(profit: np.ndarray, n_boot: int, rng) -> np.ndarray:
    idx = rng.integers(0, len(profit), (n_boot, len(profit)))
    return profit[idx].mean(axis=1)


def _shrink(roi: float, se: float, toward: float, tau: float) -> float:
    w = tau ** 2 / (tau ** 2 + se ** 2)
    return toward + w * (roi - toward)


def family_table(ev_best: np.ndarray, profit_best: np.ndarray, profit_median: np.ndarray,
                 price: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Per EV band: raw ROI at the typical book (n, bootstrap CI, ROI at the best price) and
    the graded expected ROI (two-level shrinkage, see module docstring). Argument names are
    historical: ev_best/profit_best carry the GRADED basis (typical book), profit_median the
    best-price profit."""
    rng = np.random.default_rng(seed)
    ev_best, profit_best = np.asarray(ev_best, float), np.asarray(profit_best, float)
    profit_median, price = np.asarray(profit_median, float), np.asarray(price, float)
    n_all = len(profit_best)
    if n_all:
        roi_f = float(profit_best.mean())
        se_f = float(profit_best.std(ddof=1) / np.sqrt(n_all)) if n_all > 1 else 1.0
        fam = _shrink(roi_f, se_f, 0.0, TAU_FAMILY)
    else:
        roi_f, se_f, fam = None, None, 0.0
    rows = []
    for i in range(len(BUCKETS) - 1):
        m = (ev_best >= BUCKETS[i]) & (ev_best < BUCKETS[i + 1])
        n = int(m.sum())
        row = {"lo": BUCKETS[i], "hi": None if np.isinf(BUCKETS[i + 1]) else BUCKETS[i + 1], "n": n,
               "small_sample": n < SMALL_SAMPLE_N}
        if n:
            x = profit_best[m]
            bs = _boot(x, n_boot, rng)
            roi = float(x.mean())
            se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 1.0
            exp_roi = _shrink(roi, se, fam, TAU_BAND)
            row.update(roi=roi, roi_best=float(np.nanmean(profit_median[m])), se=se,
                       ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                       avg_decimal=float(np.nanmean(price[m])), expected_roi=exp_roi,
                       grade=letter(exp_roi))
        else:
            row.update(roi=None, expected_roi=fam, grade=letter(fam))   # no history: market estimate
        rows.append(row)
    return {"n": n_all, "roi": roi_f, "se": se_f, "expected_roi": fam, "buckets": rows}


def save_table(families: dict, forward: dict, sources: dict) -> dict:
    table = {"version": datetime.utcnow().strftime("%Y-%m-%d"),
             "built_at": datetime.utcnow().isoformat(timespec="seconds"),
             "buckets": [b if not np.isinf(b) else None for b in BUCKETS],
             "thresholds": THRESHOLDS, "small_sample_n": SMALL_SAMPLE_N,
             "tau_family": TAU_FAMILY, "tau_band": TAU_BAND,
             "families": families, "forward": forward, "sources": sources}
    GRADE_TABLE_PATH.write_text(json.dumps(table, indent=1))
    return table


_TABLE = None


def load_table() -> dict | None:
    global _TABLE
    if _TABLE is None and GRADE_TABLE_PATH.exists():
        _TABLE = json.loads(GRADE_TABLE_PATH.read_text())
    return _TABLE


def grade(family: str, ev_value: float | None, table: dict | None = None) -> tuple[str, dict | None]:
    """Letter for the ROI of the pick's EV band. '—' = no pick (EV <= 0); 'NR' = no history
    for this market / band."""
    if ev_value is None or ev_value <= 0:
        return "—", None
    table = table or load_table()
    fam = (table or {}).get("families", {}).get(family)
    if not fam:
        return "NR", None
    b = fam["buckets"][bucket_of(ev_value)]
    return b["grade"], {"family": family, "band": [b["lo"], b["hi"]], "n": b["n"], "roi": b.get("roi"),
                        "expected_roi": b.get("expected_roi"), "market_roi": fam.get("roi"),
                        "roi_best": b.get("roi_best"), "basis": "typical book", "ci": b.get("ci"),
                        "small_sample": b["small_sample"]}
