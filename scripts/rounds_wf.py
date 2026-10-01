"""Walk-forward bake-off for fight-duration (round / over-under) models.

Same window and folds as fast_wf / method_wf (last 40% of decided fights, 8 expanding
folds). Every arm writes a survival curve S at the 2.5-minute edges (0 .. 25 min), so all
lines and the round of finish are scored from the same object.

Arms
  base      share of training fights still going at each edge, by 3/5 rounds x division
  direct    one classifier per line (CatBoost + logit, both corner views): P(T > 5, 7.5,
            10, 12.5, 15); five-round fights also 17.5, 20, 22.5, 25 (logit, small n).
            Monotone by cumulative minimum.
  hz_logit  discrete-time competing-risks hazard {continue, KO, Sub}, multinomial logit
  hz_cat    the same hazard with CatBoost MultiClass
  decomp    method_v2 (OOF) finish probabilities x timing-given-finish base rates
            (KO / Sub finish-time distribution by 3/5 rounds, last 6 years of training)

Scores (log loss, paired bootstrap CI vs base): over at 5:00 (starts R2), 7:30 (O/U 1.5),
10:00 (starts R3), 12:30 (O/U 2.5), 17:30 / 22:30 (five-round fights), and the round of
finish (R1..R5 / decision). Then against BestFightOdds closing prices on the same fights,
and a 50/50 log-odds blend with the market.

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.rounds_wf --arms base,direct
    ... --arms hz_logit        (arms can run as separate processes)
    ... --score                (reads every arm's saved curves, prints the tables)
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.ufc.method_v2 import (
    METHOD_DIR, OOF_PATH, _fit_binary, attach_method_features, build_training_matrix,
    feature_names, orient,
)
from app.services.ufc.round_hazard import BIN, HazardModel, anchor, round_probs

log = logging.getLogger("rounds_wf")
CACHE = METHOD_DIR / "rounds_wf_matrix.pkl"
OUT_DIR = METHOD_DIR / "rounds_wf"
EDGES = np.arange(0, 25.01, BIN)            # 11 edges
LINES = [("starts R2 (5:00)", 5.0, "mkt_sr_2", False), ("O/U 1.5 (7:30)", 7.5, "mkt_ou_1.5_over", False),
         ("starts R3 (10:00)", 10.0, "mkt_sr_3", False), ("O/U 2.5 (12:30)", 12.5, "mkt_ou_2.5_over", False),
         ("O/U 3.5 (17:30, 5-rd)", 17.5, "mkt_ou_3.5_over", True),
         ("O/U 4.5 (22:30, 5-rd)", 22.5, "mkt_ou_4.5_over", True)]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build_matrix(rebuild: bool = False) -> pd.DataFrame:
    """The served model's training matrix (rounds_v1.build_training_matrix), same cache."""
    from app.services.ufc.rounds_v1 import build_training_matrix
    return build_training_matrix(rebuild)


def labels(m: pd.DataFrame):
    t = m["outcome_fight_minutes"].to_numpy(float)
    sched = m["fight_scheduled_minutes"].to_numpy(float)
    event = np.select([m["outcome_method_class"] == 0, m["outcome_method_class"] == 1], [1, 2], 0)
    return t, sched, event


def alive(t: np.ndarray, event: np.ndarray, e: float) -> np.ndarray:
    """Still going at minute e. A decision reaches its scheduled end (T == end), so it
    counts as alive at the end edge itself."""
    return (t > e + 1e-9) | ((event == 0) & (t >= e - 1e-9))


def folds(n: int, k: int = 8, frac: float = 0.4):
    start = int(n * (1 - frac))
    b = np.linspace(start, n, k + 1).astype(int)
    return start, [(b[i], b[i + 1]) for i in range(k)]


def _division(m: pd.DataFrame) -> np.ndarray:
    divs = [c for c in m.columns if c.startswith("fight_div_")]
    return np.array(divs)[m[divs].to_numpy(float).argmax(axis=1)]


def _monotone(S: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(np.clip(S, 1e-4, 1.0), axis=1)


def _flat_after_end(S: np.ndarray, sched: np.ndarray) -> np.ndarray:
    S = S.copy()
    three = sched <= 15
    S[three, 7:] = S[three, 6:7]
    return S


# ---------------------------------------------------------------------------
# arms (each returns S (n_test, 11))
# ---------------------------------------------------------------------------

def arm_base(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    t, sched, event = labels(tr)
    div_tr, div_te = _division(tr), _division(te)
    S = np.ones((len(te), len(EDGES)))
    for five in (False, True):
        grp = (sched > 15) == five
        glob = np.array([alive(t[grp], event[grp], e).mean() for e in EDGES])
        for d in np.unique(div_te):
            g = grp & (div_tr == d)
            n = g.sum()
            local = np.array([alive(t[g], event[g], e).mean() if n else 0 for e in EDGES])
            rows = (div_te == d) & ((te["fight_scheduled_minutes"].to_numpy() > 15) == five)
            S[rows] = (n * local + 40 * glob) / (n + 40)
    return _flat_after_end(_monotone(S), te["fight_scheduled_minutes"].to_numpy(float))


def arm_direct(tr_views, te_views, tr: pd.DataFrame, te: pd.DataFrame, feats) -> np.ndarray:
    t, sched, event = labels(tr)
    sched_te = te["fight_scheduled_minutes"].to_numpy(float)
    S = np.full((len(te), len(EDGES)), np.nan)
    S[:, 0] = 1.0
    Xtr = np.vstack([v[feats].to_numpy(float) for v in tr_views])
    Xte = [v[feats].to_numpy(float) for v in te_views]
    for e in (5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0):
        k = int(round(e / BIN))
        # 15:00 = a 3-round fight's end (alive = went to decision) and a 5-rounder's R3 end:
        # fit on all fights (the 5-round flag is a feature). Past 15:00 only five-round
        # fights are informative.
        rows = sched > 15 if e > 15 else np.ones(len(tr), bool)
        y = np.tile(alive(t, event, e).astype(float), 2)
        mask = np.tile(rows, 2)
        backends = ("logit",) if e > 15 else ("catboost", "logit")
        members = [_fit_binary(b, Xtr[mask], y[mask]) for b in backends]
        lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        z = np.mean([np.mean([lg(mb(X)) for mb in members], axis=0) for X in Xte], axis=0)
        S[:, k] = 1 / (1 + np.exp(-z))
    S[:, 1] = np.sqrt(S[:, 0] * S[:, 2])     # 2:30: not a betting line; geometric fill
    for k in (3, 5, 7, 9):
        if np.isnan(S[:, k]).any():
            S[:, k] = np.where(np.isnan(S[:, k]), S[:, k - 1], S[:, k])
    S = _monotone(np.nan_to_num(S, nan=1.0))
    return _flat_after_end(S, sched_te)


def arm_hazard(tr_views, te_views, tr, te, feats, backend) -> np.ndarray:
    t, sched, event = labels(tr)
    m = HazardModel(feats, backend).fit(tr_views, t, event, sched)
    return m.survival(te_views, te["fight_scheduled_minutes"].to_numpy(float))


def arm_decomp(tr: pd.DataFrame, te: pd.DataFrame, method_marg: pd.DataFrame) -> np.ndarray:
    """P(T > e) = P(dec) + P(KO)(1 - F_KO(e)) + P(Sub)(1 - F_Sub(e)), with F the finish-time
    distribution of KOs / subs in the last 6 years of training fights, by 3/5 rounds."""
    t, sched, event = labels(tr)
    dates = pd.to_datetime(tr["date"])
    recent = (dates >= dates.max() - pd.Timedelta(days=6 * 365)).to_numpy()
    sched_te = te["fight_scheduled_minutes"].to_numpy(float)
    mm = te[["fight_id"]].astype({"fight_id": str}).merge(method_marg, on="fight_id", how="left")
    S = np.full((len(te), len(EDGES)), np.nan)
    for five in (False, True):
        rows = (sched_te > 15) == five
        acc = mm.loc[rows, "dec"].to_numpy(float)[:, None] * np.ones(len(EDGES))
        for ev_code, col in ((1, "ko"), (2, "sub")):
            ft = t[recent & (event == ev_code) & ((sched > 15) == five)]
            F = np.array([(ft <= e).mean() if len(ft) else 0.0 for e in EDGES])
            acc = acc + mm.loc[rows, col].to_numpy(float)[:, None] * (1 - F)[None, :]
        S[rows] = acc
    return _flat_after_end(_monotone(S), sched_te)


def method_marginals() -> pd.DataFrame:
    """fight_id, ko, sub, dec: method_v2 OOF conditional x ensemble OOF P(red)."""
    oof = pd.read_csv(OOF_PATH, dtype={"fight_id": str})
    ens = pd.read_csv(METHOD_DIR.parent / "h2h" / "ensemble_oof.csv", dtype={"fight_id": str})[["fight_id", "model_prob"]]
    d = oof.merge(ens, on="fight_id")
    p = d["model_prob"].to_numpy(float)
    out = pd.DataFrame({"fight_id": d["fight_id"]})
    for c in ("ko", "sub", "dec"):
        out[c] = p * d[f"red_{c}"] + (1 - p) * d[f"blue_{c}"]
    return out


# ---------------------------------------------------------------------------
# run + score
# ---------------------------------------------------------------------------

NEW_PREFIXES = ("w_pc_", "l_pc_", "diff_pc_", "w_sn_", "l_sn_", "diff_sn_",
                "fight_altitude_m", "fight_high_altitude")
# Effects allowed to change through the fight (logit interactions with round / position).
INTERACT = ["w_m_power", "l_m_power", "w_m_chin", "l_m_chin", "al_power_chin",
            "w_pc_fade", "l_pc_fade", "w_pc_att_r1", "l_pc_att_r1", "w_pc_kd_r1", "l_pc_kd_r1",
            "w_pc_kd_late", "l_pc_kd_late", "w_pc_kdabs_late", "l_pc_kdabs_late",
            "w_m_sub_threat", "l_m_sub_threat", "w_t_early_rate", "l_t_early_rate",
            "w_sn_replacement", "l_sn_replacement", "fight_high_altitude", "w_age", "l_age"]
TUNED_PATH = OUT_DIR / "hz_tuned_params.json"


def feature_sets(view: pd.DataFrame) -> dict[str, list[str]]:
    full = feature_names(view, "full")
    base = [c for c in full if not c.startswith(NEW_PREFIXES)]
    return {"v1": base, "v2": full}


ARM_CFG = {   # arm -> (backend, feature set, bin minutes, recency half-life years, extras)
    "hz_logit": ("logit", "v1", 2.5, None, {}),
    "hz_cat": ("catboost", "v1", 2.5, None, {}),
    "hz_cat_v2": ("catboost", "v2", 2.5, None, {}),
    "hz_cat_v2_fine": ("catboost", "v2", 1.25, None, {}),
    "hz_logit_v2_int": ("logit", "v2", 1.25, None, {"interactions": INTERACT}),
    "hz_cat_v2_fine_w4": ("catboost", "v2", 1.25, 4.0, {}),
    "hz_cat_v2_fine_tuned": ("catboost", "v2", 1.25, None, {"tuned": True}),
}


def recency_weights(dates: pd.Series, half_life_years: float | None) -> np.ndarray | None:
    if not half_life_years:
        return None
    d = pd.to_datetime(dates)
    age = (d.max() - d).dt.days.to_numpy() / 365.25
    return 0.5 ** (age / half_life_years)


def tune_hazard(m: pd.DataFrame, views, feats, eval_start: int) -> dict:
    """Small CatBoost grid on PRE-window fights only (train first 85%, validate the rest on
    O/U 1.5 + 2.5 + round-of-finish log loss). Saved for the tuned arm."""
    import itertools
    import json
    pre = m.iloc[:eval_start]
    cut = int(len(pre) * 0.85)
    tr, va = pre.iloc[:cut], pre.iloc[cut:]
    trv = [v.iloc[:cut] for v in views]; vav = [v.iloc[cut:eval_start] for v in views]
    t, sched, event = labels(tr)
    tv, sv, ev = labels(va)
    best = None
    for depth, lr, l2 in itertools.product((4, 6), (0.03, 0.08), (3, 20)):
        params = dict(depth=depth, learning_rate=lr, l2_leaf_reg=l2)
        hm = HazardModel(feats, "catboost", 1.25, params=params).fit(trv, t, event, sched)
        S = hm.survival(vav, sv)
        loss = sum(_ll(S[:, int(x / BIN)], alive(tv, ev, x).astype(float)).mean() for x in (7.5, 12.5))
        log.info(f"  tune depth={depth} lr={lr} l2={l2}: {loss:.4f}")
        if best is None or loss < best[0]:
            best = (loss, params)
    TUNED_PATH.write_text(json.dumps({"params": best[1], "val_loss": best[0]}))
    log.info(f"  tuned: {best}")
    return best[1]


def run(arms: list[str]) -> None:
    m = build_matrix()
    n = len(m)
    start, fl = folds(n)
    views = [orient(m, np.ones(n, bool)), orient(m, np.zeros(n, bool))]
    fsets = feature_sets(views[0])
    feats = fsets["v1"]
    log.info(f"{n} fights, features v1={len(fsets['v1'])} v2={len(fsets['v2'])}, "
             f"eval from {m['date'].iloc[start]}")
    mm = method_marginals() if "decomp" in arms else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        S_all = np.full((n, len(EDGES)), np.nan)
        for i, (lo, hi) in enumerate(fl):
            tr, te = m.iloc[:lo], m.iloc[lo:hi]
            trv, tev = [v.iloc[:lo] for v in views], [v.iloc[lo:hi] for v in views]
            if arm == "base":
                S = arm_base(tr, te)
            elif arm == "direct":
                S = arm_direct(trv, tev, tr, te, fsets["v1"])
            elif arm in ARM_CFG:
                backend, fs, bin_size, half_life, extra = ARM_CFG[arm]
                params = None
                if extra.get("tuned"):
                    import json
                    if not TUNED_PATH.exists():
                        tune_hazard(m, views, fsets[fs], start)
                    params = json.loads(TUNED_PATH.read_text())["params"]
                t_, sched_, event_ = labels(tr)
                hm = HazardModel(fsets[fs], backend, bin_size,
                                 interactions=extra.get("interactions"), params=params)
                hm.fit(trv, t_, event_, sched_, weights=recency_weights(tr["date"], half_life))
                S = hm.survival(tev, te["fight_scheduled_minutes"].to_numpy(float))
            elif arm == "decomp":
                S = arm_decomp(tr, te, mm)
            else:
                raise ValueError(arm)
            S_all[lo:hi] = S
            log.info(f"  {arm} fold {i + 1}/{len(fl)}")
        out = pd.DataFrame(S_all[start:], columns=[f"S_{e:g}" for e in EDGES])
        out.insert(0, "fight_id", m["fight_id"].iloc[start:].astype(str).values)
        out.to_csv(OUT_DIR / f"{arm}.csv", index=False)
        log.info(f"  wrote {OUT_DIR / f'{arm}.csv'}")


def _ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _boot(d, n=2000):
    rng = np.random.default_rng(0)
    d = d[np.isfinite(d)]
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def score() -> None:
    m = build_matrix()
    start, _ = folds(len(m))
    ev = m.iloc[start:].reset_index(drop=True)
    ev["fight_id"] = ev["fight_id"].astype(str)
    t, sched, event = labels(ev)
    arms = {p.stem: pd.read_csv(p, dtype={"fight_id": str}) for p in sorted(OUT_DIR.glob("*.csv"))}
    curves = {a: ev[["fight_id"]].merge(df, on="fight_id", how="left")[[f"S_{e:g}" for e in EDGES]].to_numpy(float)
              for a, df in arms.items()}
    # Anchored versions: each hazard curve rescaled so S(end) = method_v2's P(decision).
    p_dec = ev[["fight_id"]].merge(method_marginals(), on="fight_id", how="left")["dec"].to_numpy(float)
    for a in [a for a in curves if a.startswith("hz_")]:
        curves[f"{a}+anch"] = anchor(curves[a], p_dec, sched)
    base = curves.get("base")
    print(f"\n{len(ev)} fights in the evaluation window "
          f"({(sched > 15).sum()} five-round). Log loss, lower is better; diff vs base [95% CI].\n")
    for name, line, _, five_only in LINES:
        k = int(round(line / BIN))
        rows = (sched > 15) if five_only else np.ones(len(ev), bool)
        y = (t > line).astype(float)
        print(f"{name}   n={rows.sum()}  over rate {y[rows].mean():.3f}")
        for a, S in curves.items():
            ok = rows & np.isfinite(S[:, k])
            l = _ll(S[ok, k], y[ok])
            txt = f"    {a:9s} {l.mean():.4f}"
            if base is not None and a != "base":
                lb = _ll(base[ok, k], y[ok])
                d, lo, hi = _boot(l - lb)
                txt += f"   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  (n={ok.sum()})"
            print(txt)
    # round of finish
    print("\nRound of finish (R1..R5 / decision), multiclass log loss")
    truth = np.where(event == 0, 5, np.minimum((np.ceil(np.round(t / 5.0, 6)) - 1).astype(int), 4))
    for a, S in curves.items():
        ok = np.isfinite(S).all(axis=1)
        rp = round_probs(S[ok], sched[ok])
        l = -np.log(np.clip(rp[np.arange(ok.sum()), truth[ok]], 1e-4, 1))
        txt = f"    {a:9s} {l.mean():.4f}"
        if base is not None and a != "base":
            rb = round_probs(base[ok], sched[ok])
            lb = -np.log(np.clip(rb[np.arange(ok.sum()), truth[ok]], 1e-4, 1))
            d, lo, hi = _boot(l - lb)
            txt += f"   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  (n={ok.sum()})"
        print(txt)
    # vs market
    props = pd.read_csv("data/bfo/props_close.csv", dtype={"fight_id": str})
    mk = ev[["fight_id"]].merge(props, on="fight_id", how="left")
    print("\nAgainst BestFightOdds closing prices (same fights). model / market / 50-50 blend;"
          " diff = model - market and blend - market [95% CI]")
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    for name, line, col, five_only in LINES:
        if col not in mk:
            continue
        k = int(round(line / BIN))
        q = mk[col].to_numpy(float)
        y = (t > line).astype(float)
        rows = np.isfinite(q) & ((sched > 15) if five_only else True)
        if rows.sum() < 30:
            continue
        print(f"{name}   n={rows.sum()}  market {_ll(q[rows], y[rows]).mean():.4f}")
        for a, S in curves.items():
            ok = rows & np.isfinite(S[:, k])
            p = S[ok, k]
            b = 1 / (1 + np.exp(-(lg(p) + lg(q[ok])) / 2))
            lm, lk, lb = _ll(p, y[ok]), _ll(q[ok], y[ok]), _ll(b, y[ok])
            d1 = _boot(lm - lk); d2 = _boot(lb - lk)
            print(f"    {a:9s} model {lm.mean():.4f}  blend {lb.mean():.4f}   "
                  f"model-mkt {d1[0]:+.4f} [{d1[1]:+.4f},{d1[2]:+.4f}]  blend-mkt {d2[0]:+.4f} [{d2[1]:+.4f},{d2[2]:+.4f}]")


ROI_LINES = (("O/U 1.5", 7.5, "ou_1.5_over", "ou_1.5_under"),
             ("O/U 2.5", 12.5, "ou_2.5_over", "ou_2.5_under"),
             ("starts R2", 5.0, "sr_2_yes", "sr_2_no"),
             ("starts R3", 10.0, "sr_3_yes", "sr_3_no"))


def roi(arms_to_test=("hz_logit", "hz_cat", "direct", "decomp"), thresholds=(0.0, 0.05, 0.10)) -> None:
    """Flat 1-unit bets at BFO closing prices (best across books / median book): per fight
    and line, bet the side with the highest EV if EV >= threshold. Also the naive 'always
    over' and 'always under' baselines."""
    from scripts.method_roi import prices
    m = build_matrix()
    start, _ = folds(len(m))
    ev = m.iloc[start:].reset_index(drop=True)
    ev["fight_id"] = ev["fight_id"].astype(str)
    t, sched, event = labels(ev)
    px = prices()
    rng = np.random.default_rng(0)

    def summary(profit):
        profit = np.asarray(profit, float)
        if not len(profit):
            return "no bets"
        bs = [profit[rng.integers(0, len(profit), len(profit))].mean() for _ in range(1000)]
        return (f"{len(profit):5d} bets  ROI {profit.mean():+6.1%}  "
                f"[{np.percentile(bs, 2.5):+.1%}, {np.percentile(bs, 97.5):+.1%}]")

    print("\nROI at BestFightOdds closing prices, flat stakes")
    for label, line, k_over, k_under in ROI_LINES:
        wide = px[px["key"].isin([k_over, k_under])].pivot_table(
            index="fight_id", columns="key", values=["best", "median"])
        wide.columns = [f"{a}|{b}" for a, b in wide.columns]
        d = ev[["fight_id"]].merge(wide.reset_index(), on="fight_id", how="left")
        over_won = alive(t, event, line)
        print(f"\n{label}")
        for price in ("best", "median"):
            po = d.get(f"{price}|{k_over}", pd.Series(np.nan, index=d.index)).to_numpy(float)
            pu = d.get(f"{price}|{k_under}", pd.Series(np.nan, index=d.index)).to_numpy(float)
            ok = np.isfinite(po) & np.isfinite(pu)
            print(f"  [{price}] always over : {summary(np.where(over_won[ok], po[ok] - 1, -1))}")
            print(f"  [{price}] always under: {summary(np.where(~over_won[ok], pu[ok] - 1, -1))}")
            for a in arms_to_test:
                f = OUT_DIR / f"{a}.csv"
                if not f.exists():
                    continue
                S = ev[["fight_id"]].merge(pd.read_csv(f, dtype={"fight_id": str}), on="fight_id",
                                           how="left")[f"S_{line:g}"].to_numpy(float)
                for thr in thresholds:
                    eo, eu = S * po - 1, (1 - S) * pu - 1
                    bet_o = ok & (eo >= thr) & (eo >= eu)
                    bet_u = ok & (eu >= thr) & (eu > eo)
                    prof = np.concatenate([np.where(over_won[bet_o], po[bet_o] - 1, -1),
                                           np.where(~over_won[bet_u], pu[bet_u] - 1, -1)])
                    print(f"  [{price}] {a:8s} EV>={thr:.0%}: {summary(prof)}  "
                          f"(over {bet_o.sum()}, under {bet_u.sum()})")


def market_stack(arm_filter: str = "+anch") -> None:
    """Residual model on the market: per line, logit(P) = a_line + b*logit(market) +
    c*logit(model). Fit on EARLIER evaluation folds only (expanding), scored on the next
    fold, so no fight is scored by a stack that saw it. c > 0 = the model adds information."""
    from sklearn.linear_model import LogisticRegression
    m = build_matrix()
    start, fl = folds(len(m))
    ev = m.iloc[start:].reset_index(drop=True)
    ev["fight_id"] = ev["fight_id"].astype(str)
    t, sched, event = labels(ev)
    fold_id = np.concatenate([np.full(hi - lo, i) for i, (lo, hi) in enumerate(fl)])
    props = pd.read_csv("data/bfo/props_close.csv", dtype={"fight_id": str})
    mk = ev[["fight_id"]].merge(props, on="fight_id", how="left")
    p_dec = ev[["fight_id"]].merge(method_marginals(), on="fight_id", how="left")["dec"].to_numpy(float)
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    lines = [(n, x, c) for n, x, c, five in LINES if not five]
    print("\nMarket residual stack (fit on earlier folds, scored on folds 2-8)")
    for f in sorted(OUT_DIR.glob("hz_*.csv")):
        S = ev[["fight_id"]].merge(pd.read_csv(f, dtype={"fight_id": str}), on="fight_id",
                                   how="left")[[f"S_{e:g}" for e in EDGES]].to_numpy(float)
        if arm_filter == "+anch":
            S = anchor(S, p_dec, sched)
        rows = []
        for j, (name, x, col) in enumerate(lines):
            q = mk[col].to_numpy(float) if col in mk else np.full(len(ev), np.nan)
            k = int(round(x / BIN))
            ok = np.isfinite(q) & np.isfinite(S[:, k])
            rows.append(pd.DataFrame({"line": j, "fold": fold_id, "ok": ok, "y": alive(t, event, x).astype(float),
                                      "lm": lg(q), "lp": lg(S[:, k])}))
        D = pd.concat(rows, ignore_index=True)
        D = D[D["ok"]]
        out_m, out_s, coefs = [], [], []
        for k in range(1, len(fl)):
            tr, te = D[D["fold"] < k], D[D["fold"] == k]
            if len(te) == 0 or tr["y"].nunique() < 2:
                continue
            X = lambda d: np.column_stack([pd.get_dummies(d["line"]).reindex(columns=range(len(lines)), fill_value=0).to_numpy(float),
                                           d["lm"], d["lp"]])
            st = LogisticRegression(C=10.0, fit_intercept=False, max_iter=2000).fit(X(tr), tr["y"])
            coefs.append(st.coef_[0][-2:])
            ps = st.predict_proba(X(te))[:, 1]
            out_s.append(pd.DataFrame({"line": te["line"], "ls": _ll(ps, te["y"].to_numpy()),
                                       "lk": _ll(1 / (1 + np.exp(-te["lm"])), te["y"].to_numpy())}))
        R = pd.concat(out_s)
        b, c = np.mean(coefs, axis=0)
        d = _boot((R["ls"] - R["lk"]).to_numpy())
        per = " | ".join(f"{lines[j][0].split(' (')[0]} {g['ls'].mean() - g['lk'].mean():+.4f}"
                         for j, g in R.groupby("line"))
        print(f"  {f.stem + arm_filter:28s} stack - market {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]  "
              f"weights: market {b:.2f}, model {c:.2f}   ({per})")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--roi", action="store_true")
    ap.add_argument("--stack", action="store_true")
    a = ap.parse_args()
    if a.rebuild:
        build_matrix(rebuild=True)
    if a.arms:
        run(a.arms.split(","))
    if a.score:
        score()
    if a.roi:
        roi()
    if a.stack:
        market_stack("+anch")
        market_stack("")


if __name__ == "__main__":
    main()
