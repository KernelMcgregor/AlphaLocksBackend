"""Walk-forward test: corner-specific finish timing (4 outcomes) and a win-probability input.

Served today (rounds_v1): a 2-outcome hazard {KO, Sub}, each cause anchored to method_v2's
KO / Sub totals, then split between the fighters by the six-way grid's fixed ratio, so
each fighter's band is a constant slice of the KO band at every minute.

Arms (served configuration otherwise: CatBoost, feature set v2, 1.25-minute bins, 4-year
recency weights; same 8 expanding folds as scripts/rounds_wf.py):
  c2      served today
  c2_wp   + the focal fighter's win probability and the favourite's win probability
          (devigged market where priced, else Elo expectation) as timing inputs
  c4_wp   4 outcomes (red KO, red Sub, blue KO, blue Sub) with their own timing, + win
          probability; each curve anchored to its own six-way cell

Scores on the evaluation window (log loss; paired bootstrap CI vs c2):
  joint     who x how x which round (finish cells per round + decision by corner)
  round     round of finish (R1..R5 / decision)
  lines     still going at 5:00, 7:30, 10:00, 12:30 -- model, and against BFO closing
            prices, overall and on lopsided fights (favourite >= 70%)

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.rounds_cause4_wf --arm c2
    ... --arm c2_wp ; --arm c4_wp     (separate processes)
    ... --score
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from app.services.ufc.method_v2 import METHOD_DIR, OOF_PATH, feature_names, orient
from app.services.ufc.round_hazard import HazardModel, view_cause_curves
from app.services.ufc.rounds_v1 import BIN_SIZE, CATBOOST, build_training_matrix, labels, recency_weights

log = logging.getLogger("rounds_cause4_wf")
OUT = METHOD_DIR / "rounds_wf"
STEP = int(round(5.0 / BIN_SIZE))          # bins per round
LINES = [("starts R2 (5:00)", 5.0, "mkt_sr_2"), ("O/U 1.5 (7:30)", 7.5, "mkt_ou_1.5_over"),
         ("starts R3 (10:00)", 10.0, "mkt_sr_3"), ("O/U 2.5 (12:30)", 12.5, "mkt_ou_2.5_over")]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def setup():
    m = build_training_matrix(False)
    n = len(m)
    views = [orient(m, np.ones(n, bool)), orient(m, np.zeros(n, bool))]
    for v in views:
        p = v["mkt_w_prob"] if "mkt_w_prob" in v else pd.Series(np.nan, index=v.index)
        p = p.fillna(v["w_elo_expected"]) if "w_elo_expected" in v else p
        v["wp_view"] = p.to_numpy(float)
        v["wp_fav"] = np.maximum(v["wp_view"], 1 - v["wp_view"])
    feats = feature_names(views[0], "full")
    start = int(n * 0.6)
    bounds = np.linspace(start, n, 9).astype(int)
    return m, views, feats, start, list(zip(bounds[:-1], bounds[1:]))


def run_arm(arm: str) -> None:
    m, views, feats, start, folds = setup()
    t, sched, event = labels(m)
    red_won = m["red_wins"].to_numpy(float) == 1
    fin_red = red_won & (event > 0)
    # per-view 4-outcome codes: 1 focal KO, 2 focal Sub, 3 other KO, 4 other Sub
    ev_red = np.where(event == 0, 0, np.where(fin_red, event, event + 2))
    ev_blue = np.where(event == 0, 0, np.where(~fin_red, event, event + 2))
    f = feats + (["wp_view", "wp_fav"] if arm != "c2" else [])
    n = len(m)
    out = np.full((n, 4, int(round(25 / BIN_SIZE)) + 1), np.nan)   # red_ko, red_sub, blue_ko, blue_sub
    for k, (lo, hi) in enumerate(folds):
        trv, tev = [v.iloc[:lo] for v in views], [v.iloc[lo:hi] for v in views]
        w = recency_weights(m["date"].iloc[:lo])
        hm = HazardModel(f, "catboost", BIN_SIZE, params=CATBOOST)
        sc = sched[lo:hi]
        if arm == "c4_wp":
            hm.fit(trv, t[:lo], event[:lo], sched[:lo], weights=w, events=[ev_red[:lo], ev_blue[:lo]])
            _, C0, _ = view_cause_curves(hm, tev[0], sc)     # focal = red
            _, C1, _ = view_cause_curves(hm, tev[1], sc)     # focal = blue
            out[lo:hi, 0] = (C0[:, 0] + C1[:, 2]) / 2
            out[lo:hi, 1] = (C0[:, 1] + C1[:, 3]) / 2
            out[lo:hi, 2] = (C0[:, 2] + C1[:, 0]) / 2
            out[lo:hi, 3] = (C0[:, 3] + C1[:, 1]) / 2
        else:
            hm.fit(trv, t[:lo], event[:lo], sched[:lo], weights=w)
            C = np.mean([view_cause_curves(hm, v, sc)[1] for v in tev], axis=0)   # (n, 2, E): KO, Sub
            out[lo:hi, 0], out[lo:hi, 1] = C[:, 0], C[:, 1]                      # split at scoring
        log.info(f"  {arm} fold {k + 1}/{len(folds)}")
    np.save(OUT / f"cause4_{arm}.npy", out[start:])
    log.info(f"  wrote {OUT / f'cause4_{arm}.npy'}")


def _six_way(m_eval: pd.DataFrame) -> np.ndarray:
    """OOF six-way grid (red_ko, red_sub, red_dec, blue_ko, blue_sub, blue_dec), served-style
    winner probability (ensemble market blend)."""
    from app.services.ufc import ensemble
    ens = ensemble.load()
    oof = pd.read_csv(OOF_PATH, dtype={"fight_id": str})
    e = pd.read_csv(METHOD_DIR.parent / "h2h" / "ensemble_oof.csv", dtype={"fight_id": str})
    d = m_eval[["fight_id", "odds_red_prob"]].assign(fight_id=lambda x: x["fight_id"].astype(str)) \
        .merge(oof, on="fight_id", how="left").merge(e[["fight_id", "model_prob"]], on="fight_id", how="left")
    p = d["model_prob"].to_numpy(float)
    if ens is not None:
        p = ens.final_prob(p, d["odds_red_prob"].to_numpy(float))
    return np.column_stack([p * d[f"red_{c}"] for c in ("ko", "sub", "dec")] +
                           [(1 - p) * d[f"blue_{c}"] for c in ("ko", "sub", "dec")])


def anchored(arm: str, raw: np.ndarray, six: np.ndarray) -> np.ndarray:
    """(n, 4, E) corner-cause curves scaled to the six-way cells."""
    cell = six[:, [0, 1, 3, 4]]                     # red_ko, red_sub, blue_ko, blue_sub
    end = raw.shape[2] - 1
    if arm == "c4_wp":
        tot = raw.sum(axis=1)                        # total finish curve (fallback shape)
        out = np.empty_like(raw)
        for j in range(4):
            c_end = raw[:, j, end]
            own = raw[:, j] * (cell[:, j] / np.clip(c_end, 1e-4, None))[:, None]
            borrowed = tot * (cell[:, j] / np.clip(tot[:, end], 1e-4, None))[:, None]
            out[:, j] = np.where((c_end >= 1e-3)[:, None], own, borrowed)
        return out
    ko, sub = raw[:, 0], raw[:, 1]
    ko_t, sub_t = cell[:, 0] + cell[:, 2], cell[:, 1] + cell[:, 3]
    KO = ko * (ko_t / np.clip(ko[:, end], 1e-4, None))[:, None]
    SUB = sub * (sub_t / np.clip(sub[:, end], 1e-4, None))[:, None]
    rk = cell[:, 0] / np.clip(ko_t, 1e-9, None)
    rs = cell[:, 1] / np.clip(sub_t, 1e-9, None)
    return np.stack([KO * rk[:, None], SUB * rs[:, None], KO * (1 - rk)[:, None], SUB * (1 - rs)[:, None]], axis=1)


def _boot(d, n=2000):
    rng = np.random.default_rng(0)
    d = d[np.isfinite(d)]
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def score() -> None:
    m, _, _, start, _ = setup()
    ev = m.iloc[start:].reset_index(drop=True)
    t, sched, event = labels(ev)
    red_won = ev["red_wins"].to_numpy(float) == 1
    six = _six_way(ev)
    ok = np.isfinite(six).all(axis=1)
    rounds = np.round(sched / 5).astype(int)
    rnd = np.minimum(np.ceil(np.round(t / 5.0, 6)).astype(int), rounds)
    corner = np.where(red_won, 0, 1)
    arms = {}
    for arm in ("c2", "c2_wp", "c4_wp"):
        p = OUT / f"cause4_{arm}.npy"
        if p.exists():
            arms[arm] = anchored(arm, np.load(p), six)
    props = pd.read_csv("data/bfo/props_close.csv", dtype={"fight_id": str})
    mk = ev[["fight_id"]].astype(str).merge(props, on="fight_id", how="left")
    q_red = ev["odds_red_prob"].to_numpy(float)
    lopsided = np.isfinite(q_red) & (np.maximum(q_red, 1 - q_red) >= 0.70)
    res = {}
    for arm, C in arms.items():
        S = 1 - C.sum(axis=1)
        n = len(ev)
        # joint: who x how x round (finishes), decision by corner
        j = np.full(n, np.nan)
        for i in range(n):
            if not ok[i]:
                continue
            if event[i] == 0:
                j[i] = six[i, 2] if red_won[i] else six[i, 5]
            else:
                idx = corner[i] * 2 + (event[i] - 1)            # red_ko, red_sub, blue_ko, blue_sub
                r = rnd[i]
                j[i] = C[i, idx, r * STEP] - C[i, idx, (r - 1) * STEP]
        res[(arm, "joint")] = -np.log(np.clip(j, 1e-6, 1))
        # round of finish
        rp = np.full(n, np.nan)
        for i in range(n):
            if not ok[i]:
                continue
            rp[i] = S[i, (rounds[i]) * STEP] if event[i] == 0 else S[i, (rnd[i] - 1) * STEP] - S[i, rnd[i] * STEP]
        res[(arm, "round")] = -np.log(np.clip(rp, 1e-6, 1))
        for name, line, col in LINES:
            k = int(round(line / BIN_SIZE))
            alive = (t > line + 1e-9) | ((event == 0) & (t >= line - 1e-9))
            p = np.clip(S[:, k], 1e-4, 1 - 1e-4)
            l = -(alive * np.log(p) + (~alive) * np.log(1 - p))
            res[(arm, name)] = np.where(ok, l, np.nan)
    base = "c2"
    print(f"\n{int(ok.sum())} evaluation fights. Log loss (lower is better); diff vs c2 (served) [95% CI]\n")
    for metric in ["joint", "round"] + [x[0] for x in LINES]:
        print(metric)
        for arm in arms:
            l = res[(arm, metric)]
            txt = f"    {arm:6s} {np.nanmean(l):.4f}"
            if arm != base and (base, metric) in res:
                d = _boot(l - res[(base, metric)])
                txt += f"   {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
            print(txt)
    print("\nAgainst BFO closing prices (model - market; negative = model better)")
    for name, line, col in LINES:
        q = mk[col].to_numpy(float) if col in mk else np.full(len(ev), np.nan)
        alive = (t > line + 1e-9) | ((event == 0) & (t >= line - 1e-9))
        lq = -(alive * np.log(np.clip(q, 1e-4, 1)) + (~alive) * np.log(np.clip(1 - q, 1e-4, 1)))
        for subset, mask in (("all", np.isfinite(q) & ok), ("lopsided", np.isfinite(q) & ok & lopsided)):
            parts = [f"{name:18s} {subset:8s} n={int(mask.sum()):4d} market {np.nanmean(lq[mask]):.4f}"]
            for arm in arms:
                d = _boot((res[(arm, name)] - lq)[mask])
                parts.append(f"{arm} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]")
            print("  " + " | ".join(parts))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.arm:
        run_arm(a.arm)
    if a.score:
        score()


if __name__ == "__main__":
    main()
