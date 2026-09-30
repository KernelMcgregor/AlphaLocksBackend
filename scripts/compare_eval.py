"""Score walk-forward arms against the market on the SAME priced fights.

`eval_results.json` reports each arm over every eval fight but the market only over the
priced ones, so the headline numbers are not comparable. This reads the per-fight
predictions CSV(s) and reports log loss / Brier on the priced intersection, with a paired
bootstrap CI for (arm - market). Negative delta = arm beats the market.

Usage:
    python -m scripts.compare_eval models/ufc/h2h/eval_predictions.csv
    python -m scripts.compare_eval new.csv --baseline models/ufc/h2h/eval_predictions_baseline.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

EPS = 1e-6


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def _boot_ci(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def arms(df: pd.DataFrame) -> list[str]:
    return sorted(c[: -len("_proba")] for c in df.columns if c.endswith("_proba"))


def score(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    priced = df[df["odds_red_prob"].notna() & df["red_wins"].notna()]
    y = priced["red_wins"].to_numpy(float)
    mkt = priced["odds_red_prob"].to_numpy(float)
    m_ll, m_br = _ll(mkt, y), _brier(mkt, y)
    out = [{"arm": "market", "n": len(y), "log_loss": m_ll.mean(), "brier": m_br.mean(),
            "d_ll": 0.0, "d_ll_ci": "", "corr_mkt": 1.0}]
    for arm in arms(df):
        for suffix in ("_proba", "_proba_cal"):
            col = arm + suffix
            if col not in priced:
                continue
            p = priced[col].to_numpy(float)
            ll = _ll(p, y)
            d = ll - m_ll
            lo, hi = _boot_ci(d)
            lp = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1))
            lm = np.log(mkt / (1 - mkt))
            out.append({"arm": col, "n": len(y), "log_loss": ll.mean(),
                        "brier": _brier(p, y).mean(), "d_ll": d.mean(),
                        "d_ll_ci": f"[{lo:+.4f}, {hi:+.4f}]",
                        "corr_mkt": float(np.corrcoef(lp, lm)[0, 1])})
    res = pd.DataFrame(out)
    if label:
        res.insert(0, "run", label)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--baseline")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    new = pd.read_csv(a.csv)
    print(score(new, "new").to_string(index=False))
    if a.baseline:
        base = pd.read_csv(a.baseline)
        print()
        print(score(base, "baseline").to_string(index=False))
        # Same-fight comparison of each arm, new vs baseline.
        common = new.merge(base, on="fight_id", suffixes=("", "_b"))
        common = common[common["red_wins"].notna() & common["red_wins_b"].notna()]
        print(f"\nnew vs baseline on {len(common)} common fights (all, priced or not):")
        y = common["red_wins"].to_numpy(float)
        for arm in arms(new):
            col = arm + "_proba"
            if col + "_b" not in common:
                continue
            d = _ll(common[col].to_numpy(float), y) - _ll(common[col + "_b"].to_numpy(float), y)
            lo, hi = _boot_ci(d)
            print(f"  {col:32s} d_ll={d.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
