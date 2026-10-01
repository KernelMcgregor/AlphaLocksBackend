"""Pick grades = the realised ROI of an edge band in a market.

scripts/build_grade_table.py rebuilds every bet a simple rule would have made in walk-forward
backtests (per fight and market, the side with the highest EV if EV > 0, flat 1 unit at the
best price), groups the bets by market family and EV band, and records each band's ROI.
A pick's grade is the letter for its band's ROI. Nothing else: no smoothing across bands, no
caps, no adjustments for live conditions. Each band also carries n and a bootstrap 95% CI so
the page can show how much history stands behind the letter; bands with fewer than
SMALL_SAMPLE_N bets are flagged.

Families (side-specific where the sides behave differently, e.g. decision yes vs no):
  winner_open, winner_close, sixway_ko, sixway_sub, sixway_dec (winner x method, per
  method), decision_yes, decision_no, itd_yes, itd_no, ou_1_5_over, ou_1_5_under,
  ou_2_5_over, ou_2_5_under, starts_r2_yes, starts_r2_no, starts_r3_yes, starts_r3_no
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

GRADE_TABLE_PATH = Path(__file__).resolve().parents[3] / "models" / "ufc" / "grade_table.json"
BUCKETS = [0.0, 0.02, 0.05, 0.10, 0.20, float("inf")]
THRESHOLDS = [("A+", 0.08), ("A", 0.05), ("A-", 0.035), ("B+", 0.025), ("B", 0.015),
              ("B-", 0.005), ("C+", 0.0), ("C", -0.015), ("C-", -0.03), ("D", -0.06)]
GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
SMALL_SAMPLE_N = 50


def decimal(american: float | None) -> float | None:
    if american is None or american != american:
        return None
    return 1 + (american / 100 if american > 0 else 100 / -american)


def ev(p: float, american: float | None) -> float | None:
    d = decimal(american)
    return None if d is None or p is None or p != p else p * d - 1


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


def family_table(ev_best: np.ndarray, profit_best: np.ndarray, profit_median: np.ndarray,
                 price: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Per EV band: n, ROI (best price) -> grade, plus ROI at the median book, a bootstrap
    95% CI and the average price, for display."""
    rng = np.random.default_rng(seed)
    ev_best, profit_best = np.asarray(ev_best, float), np.asarray(profit_best, float)
    profit_median, price = np.asarray(profit_median, float), np.asarray(price, float)
    rows = []
    for i in range(len(BUCKETS) - 1):
        m = (ev_best >= BUCKETS[i]) & (ev_best < BUCKETS[i + 1])
        n = int(m.sum())
        row = {"lo": BUCKETS[i], "hi": None if np.isinf(BUCKETS[i + 1]) else BUCKETS[i + 1], "n": n,
               "small_sample": n < SMALL_SAMPLE_N}
        if n:
            bs = _boot(profit_best[m], n_boot, rng)
            roi = float(profit_best[m].mean())
            row.update(roi=roi, roi_median=float(np.nanmean(profit_median[m])),
                       ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                       avg_decimal=float(np.nanmean(price[m])), grade=letter(roi))
        else:
            row.update(roi=None, grade=None)
        rows.append(row)
    return {"n": int(len(profit_best)), "roi": float(profit_best.mean()) if len(profit_best) else None,
            "buckets": rows}


def save_table(families: dict, forward: dict, sources: dict) -> dict:
    table = {"version": datetime.utcnow().strftime("%Y-%m-%d"),
             "built_at": datetime.utcnow().isoformat(timespec="seconds"),
             "buckets": [b if not np.isinf(b) else None for b in BUCKETS],
             "thresholds": THRESHOLDS, "small_sample_n": SMALL_SAMPLE_N,
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
    if not b.get("grade"):
        return "NR", None
    return b["grade"], {"family": family, "band": [b["lo"], b["hi"]], "n": b["n"], "roi": b["roi"],
                        "roi_median": b.get("roi_median"), "ci": b.get("ci"),
                        "small_sample": b["small_sample"]}
