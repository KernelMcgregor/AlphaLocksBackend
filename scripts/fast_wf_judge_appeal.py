"""Paired walk-forward test of diff_judge_appeal in the winner model.

Same cached matrix, folds, calibration and market anchor as scripts/fast_wf.py; the
4 ensemble members (ensemble.MEMBERS) are run twice: baseline features, and baseline +
diff_judge_appeal (red minus blue, from scripts.judge_appeal_feature, leakage-free).
Ensemble = equal-weight mean of the members' calibrated outputs, same for both.

Reports per member and for the ensemble: log loss on all eval fights, paired delta with
a bootstrap CI, and the same on market-priced fights for the +anchor stack.

    DATABASE_URL=postgresql://localhost/alocks_local PYTHONPATH=. \
        venv/bin/python -m scripts.fast_wf_judge_appeal --out data/mmad/wf_judge_appeal.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fast_wf import build_matrix, run_arm

log = logging.getLogger("fast_wf_ja")
ARMS = ["noodds:catboost:0", "noodds_noraw:catboost:0", "noodds:hgb:0", "noodds_nosn:logit:0"]
FEAT = "diff_judge_appeal"


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-csv", type=Path, default=Path("data/mmad/judge_appeal_wf.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/mmad/wf_judge_appeal.csv"))
    ap.add_argument("--arms", nargs="*", default=ARMS)
    a = ap.parse_args()

    matchup, features = build_matrix(False)
    ja = pd.read_csv(a.feature_csv).pivot_table(index="fight_id", columns="corner",
                                                values="judge_appeal", aggfunc="first")
    matchup[FEAT] = (matchup.fight_id.map(ja["red"]).fillna(0.0)
                     - matchup.fight_id.map(ja["blue"]).fillna(0.0))
    n = len(matchup)
    eval_start = int(n * 0.6)
    ev = matchup.iloc[eval_start:]
    log.info("eval fights %d; %s nonzero on %.0f%% of them (SD %.3f)", len(ev), FEAT,
             100 * (ev[FEAT] != 0).mean(), ev[FEAT].std())

    res = matchup[["fight_id", "date", "red_wins", "odds_red_prob", FEAT]].copy()
    for variant, feats in (("base", features), ("ja", features + [FEAT])):
        for spec in a.arms:
            raw, cal, anc = run_arm(spec, matchup, feats)
            tag = f"{variant}_{spec.replace(':', '_')}"
            res[f"{tag}_proba"], res[f"{tag}_proba_cal"], res[f"{tag}+anchor_proba"] = raw, cal, anc
        res[f"{variant}_ensemble_proba_cal"] = res[[f"{variant}_{s.replace(':', '_')}_proba_cal"
                                                    for s in a.arms]].mean(1)
    res = res.iloc[eval_start:]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)

    y = res.red_wins.to_numpy(float)
    ok = ~np.isnan(y)
    priced = ok & res.odds_red_prob.notna().to_numpy()
    print(f"\nPAIRED: + {FEAT} vs baseline  (negative = better)   eval fights {ok.sum():,}")
    print(f"  {'member':<28} {'base LL':>8} {'+ja LL':>8} {'delta':>8} {'95% CI':>20}")
    rows = [(s.replace(":", "_"), "_proba_cal") for s in a.arms] + [("ensemble", "_proba_cal")]
    for tag, suf in rows:
        b = _ll(res[f"base_{tag}{suf}"].to_numpy(float)[ok], y[ok])
        j = _ll(res[f"ja_{tag}{suf}"].to_numpy(float)[ok], y[ok])
        lo, hi = _ci(j - b)
        print(f"  {tag:<28} {b.mean():>8.4f} {j.mean():>8.4f} {(j - b).mean():>+8.4f}   [{lo:+.4f}, {hi:+.4f}]")
    print(f"\n  with market anchor, priced fights ({priced.sum():,}):")
    for s in a.arms:
        tag = s.replace(":", "_")
        b = _ll(res[f"base_{tag}+anchor_proba"].to_numpy(float)[priced], y[priced])
        j = _ll(res[f"ja_{tag}+anchor_proba"].to_numpy(float)[priced], y[priced])
        lo, hi = _ci(j - b)
        print(f"  {tag:<28} {b.mean():>8.4f} {j.mean():>8.4f} {(j - b).mean():>+8.4f}   [{lo:+.4f}, {hi:+.4f}]")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
