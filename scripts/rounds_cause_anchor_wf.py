"""Walk-forward check: should the served round curves take their KO / Sub totals from
method_v2, not just their decision total?

rounds_v1 anchors one number: the curve's finish mass is rescaled so P(goes the distance)
equals method_v2's. The KO-vs-Sub split inside that mass is the hazard model's own, so the
site's survival chart and its winner x method grid can disagree on how a fight ends (one
upcoming fight: hazard Sub 21% of all outcomes, method_v2 10%).

Arms, all from the served configuration (hz_cat_v2_fine_w4: CatBoost, feature set v2,
1.25-minute bins, 4-year recency weights), refit on the same 8 expanding folds as
scripts/rounds_wf.py, keeping the cause-specific curves the bake-off discarded:
  raw       hazard model, unanchored
  anch_dec  served today: total finish mass scaled to method_v2's P(decision)
  anch_cause each cause scaled to its method_v2 total: KO(t) *= P_ko / KO(end),
            Sub(t) *= P_sub / Sub(end), S = 1 - KO - Sub. The hazard model keeps the timing
            of each cause; method_v2 sets how much of each there is.

Scored on the evaluation window (log loss, paired bootstrap CI of anch_cause - anch_dec):
the over/under lines, round of finish, method (KO / Sub / Dec) and round x method jointly.

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.rounds_cause_anchor_wf
    ... --score      (re-score from the saved curves without refitting)
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from app.services.ufc.method_v2 import METHOD_DIR, feature_names, orient
from app.services.ufc.round_hazard import HazardModel, cause_curves, n_bins
from app.services.ufc.rounds_v1 import (
    BIN_SIZE, CATBOOST, build_training_matrix, labels, recency_weights,
)
from scripts.rounds_wf import _boot, alive, folds, method_marginals

log = logging.getLogger("rounds_cause_anchor_wf")
CURVES = METHOD_DIR / "rounds_wf" / "cause_curves_hz_cat_v2_fine_w4.npz"
LINES = [("starts R2 (5:00)", 5.0, False), ("O/U 1.5 (7:30)", 7.5, False),
         ("starts R3 (10:00)", 10.0, False), ("O/U 2.5 (12:30)", 12.5, False),
         ("O/U 3.5 (17:30, 5-rd)", 17.5, True), ("O/U 4.5 (22:30, 5-rd)", 22.5, True)]


def fit_curves() -> None:
    m = build_training_matrix()
    n = len(m)
    start, fl = folds(n)
    views = [orient(m, np.ones(n, bool)), orient(m, np.zeros(n, bool))]
    feats = feature_names(views[0], "full")
    nb = n_bins(25.0, BIN_SIZE)
    out = {k: np.full((n - start, nb + 1), np.nan) for k in ("S", "KO", "SUB")}
    for i, (lo, hi) in enumerate(fl):
        tr = m.iloc[:lo]
        t, sched, event = labels(tr)
        hm = HazardModel(feats, "catboost", BIN_SIZE, params=CATBOOST)
        hm.fit([v.iloc[:lo] for v in views], t, event, sched, weights=recency_weights(tr["date"]))
        cur = cause_curves(hm, [v.iloc[lo:hi] for v in views],
                           m["fight_scheduled_minutes"].iloc[lo:hi].to_numpy(float))
        for k in out:
            out[k][lo - start:hi - start] = cur[k]
        log.info(f"  fold {i + 1}/{len(fl)}  ({hi - lo} fights)")
    np.savez(CURVES, fight_id=m["fight_id"].iloc[start:].astype(str).to_numpy(), **out)
    log.info(f"  wrote {CURVES}")


def _end_idx(sched: np.ndarray) -> np.ndarray:
    return np.where(sched <= 15, n_bins(15.0, BIN_SIZE), n_bins(25.0, BIN_SIZE))


def variants(S, KO, SUB, sched, mm) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """name -> (S, KO, SUB) on the 1.25-minute grid."""
    rows = np.arange(len(S))
    end = _end_idx(sched)
    out = {"raw": (S, KO, SUB)}
    # Served today (rounds_v1._anchor_all).
    s_end = S[rows, end]
    scale = ((1 - mm["dec"]) / np.clip(1 - s_end, 1e-4, None))[:, None]
    out["anch_dec"] = (1 - (1 - S) * scale, KO * scale, SUB * scale)
    # Each cause to its own method_v2 total.
    ko_s = (mm["ko"] / np.clip(KO[rows, end], 1e-4, None))[:, None]
    sub_s = (mm["sub"] / np.clip(SUB[rows, end], 1e-4, None))[:, None]
    ko2, sub2 = KO * ko_s, SUB * sub_s
    out["anch_cause"] = (1 - ko2 - sub2, ko2, sub2)
    return out


def _ll_bin(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ll_cat(P, truth):
    return -np.log(np.clip(P[np.arange(len(P)), truth], 1e-4, 1))


def score() -> None:
    m = build_training_matrix()
    start, _ = folds(len(m))
    ev = m.iloc[start:].reset_index(drop=True)
    ev["fight_id"] = ev["fight_id"].astype(str)
    z = np.load(CURVES, allow_pickle=True)
    assert (z["fight_id"] == ev["fight_id"].to_numpy()).all(), "curves out of step with matrix"
    t, sched, event = labels(ev)
    mm = ev[["fight_id"]].merge(method_marginals(), on="fight_id", how="left")
    ok = mm[["ko", "sub", "dec"]].notna().all(axis=1).to_numpy() & np.isfinite(z["S"]).all(axis=1)
    t, sched, event = t[ok], sched[ok], event[ok]
    mm = {c: mm.loc[ok, c].to_numpy(float) for c in ("ko", "sub", "dec")}
    V = variants(z["S"][ok], z["KO"][ok], z["SUB"][ok], sched, mm)
    end = _end_idx(sched)
    rows = np.arange(len(t))
    n_r = np.where(sched <= 15, 3, 5)
    rnd = np.minimum(np.ceil(np.round(t / 5.0, 6)).astype(int), n_r)          # 1-based
    print(f"\n{ok.sum()} fights with method_v2 OOF ({(sched > 15).sum()} five-round). "
          "Log loss, lower is better. Diff = anch_cause - anch_dec [95% CI].")

    def report(title, losses):
        print(f"\n{title}")
        for name, l in losses.items():
            print(f"    {name:10s} {np.nanmean(l):.4f}")
        d, lo, hi = _boot(losses["anch_cause"] - losses["anch_dec"])
        verdict = "better" if hi < 0 else "worse" if lo > 0 else "no clear difference"
        print(f"    diff       {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  -> anch_cause {verdict}")

    for name, line, five_only in LINES:
        sel = (sched > 15) if five_only else np.ones(len(t), bool)
        k = int(round(line / BIN_SIZE))
        y = alive(t, event, line).astype(float)
        report(f"{name}  n={sel.sum()}",
               {a: np.where(sel, _ll_bin(S[:, k], y), np.nan)[sel] for a, (S, _, _) in V.items()})

    # Round of finish: R1..R5 / decision.
    truth_r = np.where(event == 0, 5, rnd - 1)
    def round_P(S):
        P = np.zeros((len(S), 6))
        for r in range(5):
            a, b = int(r * 5 / BIN_SIZE), int((r + 1) * 5 / BIN_SIZE)
            P[:, r] = np.where(r < n_r, S[:, a] - S[:, b], 0)
        P[:, 5] = S[rows, end]
        return P
    report("Round of finish (R1..R5 / decision)", {a: _ll_cat(round_P(S), truth_r) for a, (S, _, _) in V.items()})

    # Method: KO / Sub / Dec.
    truth_m = np.where(event == 1, 0, np.where(event == 2, 1, 2))
    report("Method (KO / Sub / Dec)", {
        a: _ll_cat(np.column_stack([KO[rows, end], SUB[rows, end], S[rows, end]]), truth_m)
        for a, (S, KO, SUB) in V.items()})

    # Round x method: R1 KO, R1 Sub, ... R5 Sub, decision (11 classes).
    truth_j = np.where(event == 0, 10, 2 * (rnd - 1) + (event == 2))
    def joint_P(S, KO, SUB):
        P = np.zeros((len(S), 11))
        for r in range(5):
            a, b = int(r * 5 / BIN_SIZE), int((r + 1) * 5 / BIN_SIZE)
            P[:, 2 * r] = np.where(r < n_r, KO[:, b] - KO[:, a], 0)
            P[:, 2 * r + 1] = np.where(r < n_r, SUB[:, b] - SUB[:, a], 0)
        P[:, 10] = S[rows, end]
        return P
    report("Round x method (R1 KO .. R5 Sub / decision)",
           {a: _ll_cat(joint_P(*c), truth_j) for a, c in V.items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="score saved curves only")
    a = ap.parse_args()
    if not a.score:
        fit_curves()
    score()
