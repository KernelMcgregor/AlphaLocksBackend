"""Walk-forward bake-off for expected-stat (xs) models, scored as stat FORECASTS.

The only previous evaluation (a scratch script) scored per-minute rates given each bout's
ACTUAL length, tuned on the window it reported, and used Poisson deviance only. This
harness separates three questions:

  rate     Poisson deviance of count ~ rate x actual minutes (what the old script did)
  total    the pre-fight forecast of the bout total: rate combined with the duration
           model's out-of-sample survival curve (rounds_oof4.npz), scored by MAE/RMSE of
           the mean and CRPS of the full NB-mixture distribution, plus PIT coverage
  h2h      log loss of P(A lands more than B), the "X has more sig strikes / takedowns"
           prop, from the same predictive distributions

Windows: dispersion and any tuning use TUNE (2015 -> report start); everything printed
under REPORT is the duration model's walk-forward window (last 40% of decided fights),
the same fights fast_wf / method_wf / rounds_wf score.

Each arm is a function (train_obs, target_obs, asof) -> per-minute rates, refit at
monthly or quarterly boundaries on bouts strictly before the boundary.

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.xs_eval --arms league,marcel,xs
    ... --score                           (reads every saved arm, prints the tables)
    ... --score --only xs,glm_ctx         (subset, first arm listed is the CI reference)
"""
from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import PoissonRegressor

log = logging.getLogger("xs_eval")
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "ufc" / "xs_eval"
FRAMES = OUT_DIR / "frames.pkl"
OOF4 = ROOT / "models" / "ufc" / "method" / "rounds_oof4.npz"
BIN = 1.25
HURDLE_CTRL = True   # control-time distribution: zero hurdle + NB (else plain NB)

#: stat key -> per-fighter count column
STATS = {
    "sig": "sig_str_landed", "td": "td_landed", "kd": "kd", "sub": "sub_att", "ctrl": "ctrl_seconds",
    "sig_att": "sig_str_attempted", "td_att": "td_attempted",
    "dist": "distance_landed", "clinch": "clinch_landed", "gnd": "ground_landed",
}
CORE = ["sig", "td", "kd", "sub", "ctrl"]
COUNT_COLS = sorted(set(STATS.values()) | {
    "total_str_landed", "rev", "distance_attempted", "clinch_attempted", "ground_attempted",
    "head_landed", "body_landed", "leg_landed"})


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _division(wc) -> str:
    from app.services.ufc.model import _classify_weight_class
    return _classify_weight_class(wc)


def load_frames(rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(obs, robs): one row per (bout, attacker) with fight totals, and one per
    (bout, attacker, round). Attacker's counts plus the defender's control seconds."""
    if FRAMES.exists() and not rebuild:
        with open(FRAMES, "rb") as f:
            return pickle.load(f)
    from app.database import engine
    from app.models.ufc import UFC_SCHEMA
    sch = f"{UFC_SCHEMA}." if UFC_SCHEMA else ""
    cols = ", ".join(f"s.{c}" for c in COUNT_COLS)
    q = f"""
        select f.id as fight_id, f.date, f.weight_class, f.time_format, f.fight_time_seconds,
               f.finish_round, f.method, s.fighter_id, s.round_number, {cols}
        from {sch}ufc_fight_stats s join {sch}ufc_fights f on f.id = s.fight_id
        where f.fight_time_seconds > 0
    """
    raw = pd.read_sql(q, engine)
    raw["fight_id"] = raw["fight_id"].astype(str)
    raw["fighter_id"] = raw["fighter_id"].astype(str)
    raw["date"] = pd.to_datetime(raw["date"])
    raw[COUNT_COLS] = raw[COUNT_COLS].fillna(0).astype(float)
    # exactly two fighters per (fight, round)
    n = raw.groupby(["fight_id", "round_number"])["fighter_id"].transform("nunique")
    raw = raw[n == 2]

    def pair(df: pd.DataFrame) -> pd.DataFrame:
        opp = df[["fight_id", "round_number", "fighter_id", "ctrl_seconds"]].rename(
            columns={"fighter_id": "opp_id", "ctrl_seconds": "opp_ctrl"})
        o = df.merge(opp, on=["fight_id", "round_number"])
        return o[o["fighter_id"] != o["opp_id"]].reset_index(drop=True)

    tf = raw["time_format"].fillna("")
    raw["sched_rounds"] = np.where(tf == "5-5-5-5-5", 5, np.where(tf == "5-5-5", 3, 0))
    raw["division"] = raw["weight_class"].map(_division)
    raw["women"] = raw["division"].str.startswith("w_").astype(int)
    raw["title"] = raw["weight_class"].fillna("").str.lower().str.contains("title").astype(int)

    obs = pair(raw[raw["round_number"] == 0].copy())
    obs["minutes"] = obs["fight_time_seconds"] / 60.0

    rob = raw[raw["round_number"] > 0].copy()
    # round length: full 5:00 rounds, the last round truncated at the finish
    std = rob["sched_rounds"] > 0
    rob = rob[std]
    last = rob.groupby("fight_id")["round_number"].transform("max")
    rem = rob["fight_time_seconds"] - 300 * (last - 1)
    rob["minutes"] = np.where(rob["round_number"] < last, 5.0, rem / 60.0)
    rob = pair(rob)
    rob = rob[rob["minutes"] > 0]

    obs = obs.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)
    rob = rob.sort_values(["date", "fight_id", "round_number", "fighter_id"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(FRAMES, "wb") as f:
        pickle.dump((obs, rob), f)
    log.info(f"  frames: {len(obs)} fight rows, {len(rob)} round rows -> {FRAMES}")
    return obs, rob


def report_window() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Duration-model OOF curves: (fight_id, P(T in each finish bin), P(decision)) and the
    window start date. Cumulative incidence per cause at 1.25-min edges, 0..25 min."""
    z = np.load(OOF4)
    cif = z["curves"].sum(axis=1)                      # (n, 21) all-cause cumulative incidence
    fin = np.diff(cif, axis=1).clip(0, None)            # (n, 20) finish mass per bin
    return pd.DataFrame({"fight_id": z["fight_id"].astype(str)}), fin


# ---------------------------------------------------------------------------
# walk-forward driver
# ---------------------------------------------------------------------------

def boundaries(start: str, end: pd.Timestamp, months: int) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start), end + pd.DateOffset(months=months), freq=f"{months}MS"))


def walk_forward(fit_predict, obs: pd.DataFrame, months: int = 3, start: str = "2012-01-01",
                 stats_keys=CORE, robs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rates for every obs row dated >= start. fit_predict(train, target, asof, robs_train)
    returns a DataFrame indexed like target with columns rate_{stat}."""
    out = []
    bnds = boundaries(start, obs["date"].max(), months)
    t0 = time.time()
    for lo, hi in zip(bnds[:-1], bnds[1:]):
        tgt = obs[(obs["date"] >= lo) & (obs["date"] < hi)]
        if tgt.empty:
            continue
        train = obs[obs["date"] < lo]
        rtrain = robs[robs["date"] < lo] if robs is not None else None
        out.append(fit_predict(train, tgt, lo, rtrain))
    log.info(f"  walk-forward {len(bnds) - 1} refits in {time.time() - t0:.0f}s")
    return pd.concat(out)


def decay_w(dates: pd.Series, asof: pd.Timestamp, half_life: float) -> np.ndarray:
    age = (asof - dates).dt.days.to_numpy() / 365.25
    return 0.5 ** (age / half_life)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def arm_league(half_life=1.5):
    def fp(train, tgt, asof, _r):
        w = decay_w(train["date"], asof, half_life)
        res = pd.DataFrame(index=tgt.index)
        for s, c in STATS.items():
            res[f"rate_{s}"] = (train[c] * w).sum() / (train["minutes"] * w).sum()
        return res
    return fp


def arm_marcel(half_life=1.5, k_min=None):
    """KenPom-classic multiplicative matchup of shrunk raw rates:
    rate = L * (off_A / L) * (allowed_B / L), each shrunk with K minutes of league rate."""
    k_min = k_min or {s: 30.0 for s in STATS}

    def fp(train, tgt, asof, _r):
        w = decay_w(train["date"], asof, half_life)
        m = train["minutes"] * w
        res = pd.DataFrame(index=tgt.index)
        for s, c in STATS.items():
            y = train[c] * w
            L = y.sum() / m.sum()
            K = k_min[s]
            off = (y.groupby(train["fighter_id"]).sum() + K * L) / (m.groupby(train["fighter_id"]).sum() + K)
            alw = (y.groupby(train["opp_id"]).sum() + K * L) / (m.groupby(train["opp_id"]).sum() + K)
            o = tgt["fighter_id"].map(off).fillna(L).to_numpy()
            a = tgt["opp_id"].map(alw).fillna(L).to_numpy()
            res[f"rate_{s}"] = o * a / L
        return res
    return fp


def _design(train, tgt, ids, ctx_cols, s_off=1.0, s_def=1.0, s_ctx=10.0):
    """Sparse [+off_A | -def_B | context] design. Column scaling sets the effective ridge
    per block (penalty alpha / scale^2), so context terms are near-unpenalised."""
    def build(df):
        n = len(ids)
        r = np.arange(len(df))
        a = ids.get_indexer(df["fighter_id"])
        d = ids.get_indexer(df["opp_id"])
        blocks = [sparse.csr_matrix((np.full(len(df), s_off), (r, a)), shape=(len(df), n)),
                  sparse.csr_matrix((np.full(len(df), -s_def), (r, d)), shape=(len(df), n))]
        if ctx_cols:
            blocks.append(sparse.csr_matrix(df[ctx_cols].to_numpy(float) * s_ctx))
        return sparse.hstack(blocks).tocsr()
    return build(train), build(tgt)


def context_frame(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """One-hot division (incl. women's) + scheduled-rounds + era dummies."""
    out = pd.DataFrame(index=df.index)
    for d in levels["division"]:
        out[f"div_{d}"] = (df["division"] == d).astype(float)
    out["five_rd"] = (df["sched_rounds"] == 5).astype(float)
    yr = df["date"].dt.year
    for e in levels["era"]:
        out[f"era_{e}"] = ((yr // 3) * 3 == e).astype(float)
    return out


def arm_glm(alpha=0.003, half_life=1.5, ctx=False, s_off=1.0, s_def=1.0, keys=None,
            alpha_by_stat=None, extra=None, s_extra=3.0, ids_on=True):
    """The current expected_stats model (ctx=False, defaults) plus variants.
    extra(df) -> DataFrame of standardised covariates (e.g. pre-fight Glicko of both
    fighters): the fighter effects then shrink toward the covariate prediction instead of 0.
    ids_on=False drops the fighter columns (a covariate-only forecaster)."""
    keys = keys or list(STATS)
    if not ids_on:
        s_off = s_def = 0.0

    def fp(train, tgt, asof, _r):
        train = train[train["minutes"] > 0]
        ids = pd.Index(pd.unique(pd.concat([train["fighter_id"], train["opp_id"],
                                            tgt["fighter_id"], tgt["opp_id"]])))
        ctx_cols, tr, tg = [], train, tgt
        if ctx:
            lv = {"division": sorted(train["division"].unique()),
                  "era": sorted(((train["date"].dt.year // 3) * 3).unique())}
            ctr, ctg = context_frame(train, lv), context_frame(tgt, lv)
            # future era -> latest era seen
            if ctg.filter(like="era_").sum(axis=1).eq(0).any():
                last = f"era_{lv['era'][-1]}"
                ctg.loc[ctg.filter(like="era_").sum(axis=1).eq(0), last] = 1.0
            ctx_cols = list(ctr.columns)
            tr = pd.concat([train, ctr], axis=1)
            tg = pd.concat([tgt, ctg], axis=1)
        Xtr, Xtg = _design(tr, tg, ids, ctx_cols, s_off, s_def)
        if extra is not None:
            etr, etg = extra(train), extra(tgt)
            mu_, sd_ = etr.mean(), etr.std().replace(0, 1)
            etr, etg = ((etr - mu_) / sd_).fillna(0), ((etg - mu_) / sd_).fillna(0)
            Xtr = sparse.hstack([Xtr, sparse.csr_matrix(etr.to_numpy() * s_extra)]).tocsr()
            Xtg = sparse.hstack([Xtg, sparse.csr_matrix(etg.to_numpy() * s_extra)]).tocsr()
        res = pd.DataFrame(index=tgt.index)
        for s in keys:
            hl = half_life[s] if isinstance(half_life, dict) else half_life
            w = decay_w(train["date"], asof, hl) * train["minutes"].to_numpy()
            y = train[STATS[s]].to_numpy(float) / train["minutes"].to_numpy()
            a = (alpha_by_stat or {}).get(s, alpha)
            m = PoissonRegressor(alpha=a, max_iter=300, fit_intercept=True)
            m.fit(Xtr, y, sample_weight=w)
            res[f"rate_{s}"] = m.predict(Xtg)
        return res
    return fp


def _pair_design(df, ids, n):
    """(X_focal, X_other): +1 at the row's fighter, -1 at its opponent (2n columns), and
    the same with the corners swapped."""
    r = np.arange(len(df))
    a = ids.get_indexer(df["fighter_id"])
    b = ids.get_indexer(df["opp_id"])

    def mk(i, j):  # unseen fighters (index -1) get no column: league-average skill
        ki, kj = i >= 0, j >= 0
        return sparse.hstack([sparse.csr_matrix((np.ones(ki.sum()), (r[ki], i[ki])), shape=(len(df), n)),
                              sparse.csr_matrix((-np.ones(kj.sum()), (r[kj], j[kj])), shape=(len(df), n))]).tocsr()
    return mk(a, b), mk(b, a)


def fit_composition(train, asof, lam=1.0, half_life=1.5, ctx=True, ids=None):
    """Multinomial logit on how a bout's time splits into {A controls, B controls, neither}:
        eta_A = c + ctx + top_A - topdef_B,   eta_B = c + ctx + top_B - topdef_A
        share_A = e^eta_A / (1 + e^eta_A + e^eta_B)
    Pseudo-counts are control MINUTES; ridge lam on fighter terms. Returns predict(df)."""
    from scipy.optimize import minimize
    tr = train[train["minutes"] > 0]
    tr = tr.groupby("fight_id").head(1)                      # model is symmetric: one row/bout
    ids = ids if ids is not None else pd.Index(pd.unique(pd.concat([tr["fighter_id"], tr["opp_id"]])))
    n = len(ids)
    Xa, Xb = _pair_design(tr, ids, n)
    lv = {"division": sorted(tr["division"].unique()), "era": sorted(((tr["date"].dt.year // 3) * 3).unique())}
    C = sparse.csr_matrix(context_frame(tr, lv).to_numpy()) if ctx else sparse.csr_matrix((len(tr), 0))
    ca = tr["ctrl_seconds"].to_numpy() / 60
    cb = tr["opp_ctrl"].to_numpy() / 60
    cn = np.clip(tr["minutes"].to_numpy() - ca - cb, 0, None)
    tot = ca + cb + cn
    w = decay_w(tr["date"], asof, half_life)
    k = C.shape[1]

    def unpack(th):
        return th[0], th[1:1 + k], th[1 + k:]

    def f(th):
        c, g, b = unpack(th)
        base = c + C @ g
        ea, eb = base + Xa @ b, base + Xb @ b
        m = np.maximum(np.maximum(ea, eb), 0)
        D = np.exp(-m) + np.exp(ea - m) + np.exp(eb - m)
        logD = m + np.log(D)
        sa, sb = np.exp(ea - logD), np.exp(eb - logD)
        ll = (w * (ca * ea + cb * eb - tot * logD)).sum()
        ga, gb = w * (ca - tot * sa), w * (cb - tot * sb)
        grad_b = Xa.T @ ga + Xb.T @ gb - lam * b
        grad_g = C.T @ (ga + gb)
        grad_c = (ga + gb).sum()
        obj = -(ll - 0.5 * lam * (b @ b))
        return obj, -np.concatenate([[grad_c], grad_g, grad_b])

    th0 = np.zeros(1 + k + 2 * n)
    th0[0] = -2.0
    res = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    c, g, b = unpack(res.x)

    def predict(df):
        Ya, Yb = _pair_design(df, ids, n)
        Cd = sparse.csr_matrix(context_frame(df, lv).to_numpy()) if ctx else sparse.csr_matrix((len(df), 0))
        if ctx:  # future era -> latest seen
            cf = context_frame(df, lv)
            miss = cf.filter(like="era_").sum(axis=1).eq(0)
            cf.loc[miss, f"era_{lv['era'][-1]}"] = 1.0
            Cd = sparse.csr_matrix(cf.to_numpy())
        base = c + Cd @ g
        ea, eb = base + Ya @ b, base + Yb @ b
        D = 1 + np.exp(ea) + np.exp(eb)
        return np.exp(ea) / D, np.exp(eb) / D
    predict.params = (c, g, b, ids)
    return predict


def arm_composition(lam=1.0, ctx=True, half_life=1.5):
    def fp(train, tgt, asof, _r):
        ids = pd.Index(pd.unique(pd.concat([train["fighter_id"], train["opp_id"], tgt["fighter_id"], tgt["opp_id"]])))
        pr = fit_composition(train, asof, lam=lam, ctx=ctx, ids=ids, half_life=half_life)
        sa, _ = pr(tgt)
        return pd.DataFrame({"rate_ctrl": 60.0 * sa}, index=tgt.index)
    return fp


def arm_pace(alpha=0.003, half_life=1.5, keys=None):
    """GLM plus a shared pace block: log rate_A = mu + off_A - def_B + pace_A + pace_B, so
    the opponent's own tempo moves A's volume (KenPom tempo / bivariate-Poisson idea)."""
    keys = keys or list(STATS)

    def fp(train, tgt, asof, _r):
        train = train[train["minutes"] > 0]
        ids = pd.Index(pd.unique(pd.concat([train["fighter_id"], train["opp_id"], tgt["fighter_id"], tgt["opp_id"]])))
        n = len(ids)

        def X(df):
            r = np.arange(len(df))
            a, d = ids.get_indexer(df["fighter_id"]), ids.get_indexer(df["opp_id"])
            one = np.ones(len(df))
            return sparse.hstack([
                sparse.csr_matrix((one, (r, a)), shape=(len(df), n)),
                sparse.csr_matrix((-one, (r, d)), shape=(len(df), n)),
                sparse.csr_matrix((one, (r, a)), shape=(len(df), n)) + sparse.csr_matrix((one, (r, d)), shape=(len(df), n)),
            ]).tocsr()
        Xtr, Xtg = X(train), X(tgt)
        w = decay_w(train["date"], asof, half_life) * train["minutes"].to_numpy()
        res = pd.DataFrame(index=tgt.index)
        for s in keys:
            y = train[STATS[s]].to_numpy(float) / train["minutes"].to_numpy()
            a = alpha[s] if isinstance(alpha, dict) else alpha
            m = PoissonRegressor(alpha=a, max_iter=300).fit(Xtr, y, sample_weight=w)
            res[f"rate_{s}"] = m.predict(Xtg)
        return res
    return fp


_GLICKO = None


def glicko_cov(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-fight Glicko dims of the attacker (a_) and defender (d_), + rounds seen."""
    global _GLICKO
    if _GLICKO is None:
        g = pd.read_pickle(OUT_DIR / "glicko_snaps.pkl")
        dims = [c for c in g.columns if c not in ("fight_id", "fighter_id") and
                (not c.startswith("_meta") or c == "_meta_rounds_seen")]
        _GLICKO = g.set_index(["fight_id", "fighter_id"])[dims]
        _GLICKO["_meta_rounds_seen"] = np.log1p(_GLICKO["_meta_rounds_seen"])
    a = _GLICKO.reindex(pd.MultiIndex.from_arrays([df["fight_id"], df["fighter_id"]]))
    d = _GLICKO.reindex(pd.MultiIndex.from_arrays([df["fight_id"], df["opp_id"]]))
    a.columns = [f"a_{c}" for c in a.columns]
    d.columns = [f"d_{c}" for c in d.columns]
    return pd.concat([a.reset_index(drop=True), d.reset_index(drop=True)], axis=1).set_index(df.index)


def arm_voleff(alpha_att=0.003, alpha_acc=1.0, half_life=1.5, alpha_land=None):
    """Volume x efficiency: landed = attempts (Poisson GLM off/def) x accuracy (binomial
    logit acc_A - evade_B), for sig strikes and takedowns. Also keeps the direct landed
    GLM's other stats so the arm is complete."""
    from sklearn.linear_model import LogisticRegression

    def fp(train, tgt, asof, _r):
        train = train[train["minutes"] > 0]
        ids = pd.Index(pd.unique(pd.concat([train["fighter_id"], train["opp_id"], tgt["fighter_id"], tgt["opp_id"]])))
        Xtr, Xtg = _design(train, tgt, ids, [])
        dw = decay_w(train["date"], asof, half_life)
        w = dw * train["minutes"].to_numpy()
        res = pd.DataFrame(index=tgt.index)
        for land, att in (("sig", "sig_att"), ("td", "td_att")):
            y = train[STATS[att]].to_numpy(float) / train["minutes"].to_numpy()
            a = alpha_att[att] if isinstance(alpha_att, dict) else alpha_att
            ra = PoissonRegressor(alpha=a, max_iter=300).fit(Xtr, y, sample_weight=w).predict(Xtg)
            L, A = train[STATS[land]].to_numpy(float), train[STATS[att]].to_numpy(float)
            k = A > 0
            Xb = sparse.vstack([Xtr[k], Xtr[k]])
            yb = np.r_[np.ones(k.sum()), np.zeros(k.sum())]
            wb = np.r_[L[k] * dw[k], (A[k] - L[k]) * dw[k]]
            C = alpha_acc[land] if isinstance(alpha_acc, dict) else alpha_acc
            lr = LogisticRegression(C=C, max_iter=500).fit(Xb, yb, sample_weight=wb)
            acc = lr.predict_proba(Xtg)[:, 1]
            res[f"rate_{att}"] = ra
            res[f"rate_{land}"] = ra * acc
        return res
    return fp


def arm_position(alpha=0.003, lam=3.0, half_life=1.5, kappa=0.15):
    """Position-aware sig strikes. Exposures from the bout's time split:
        distance ~ minutes with neither fighter in control
        clinch   ~ all minutes
        ground   ~ A's control minutes + kappa * B's control minutes (strikes off the back)
    Each a Poisson off/def GLM on its own exposure; the pre-fight forecast uses the
    composition model's expected shares. rate_sig = sum of the three."""
    def fp(train, tgt, asof, _r):
        train = train[train["minutes"] > 0]
        ids = pd.Index(pd.unique(pd.concat([train["fighter_id"], train["opp_id"], tgt["fighter_id"], tgt["opp_id"]])))
        comp = fit_composition(train, asof, lam=lam, ids=ids)
        sa, sb = comp(tgt)
        Xtr, Xtg = _design(train, tgt, ids, [])
        dw = decay_w(train["date"], asof, half_life)
        m = train["minutes"].to_numpy()
        ca, cb = train["ctrl_seconds"].to_numpy() / 60, train["opp_ctrl"].to_numpy() / 60
        expo = {"dist": np.clip(m - ca - cb, 0, None) + 0.05, "clinch": m, "gnd": ca + kappa * cb + 0.05}
        share = {"dist": np.clip(1 - sa - sb, 0, 1), "clinch": np.ones(len(tgt)), "gnd": sa + kappa * sb}
        res = pd.DataFrame(index=tgt.index)
        tot = np.zeros(len(tgt))
        for k in ("dist", "clinch", "gnd"):
            e = expo[k]
            a = alpha[k] if isinstance(alpha, dict) else alpha
            r = PoissonRegressor(alpha=a, max_iter=300).fit(Xtr, train[STATS[k]].to_numpy(float) / e,
                                                             sample_weight=dw * e).predict(Xtg)
            res[f"rate_{k}"] = r * share[k]
            tot += r * share[k]
        res["rate_sig"] = tot
        res["rate_ctrl"] = 60 * sa
        return res
    return fp


ARMS = {
    "league": lambda: (arm_league(), 3),
    "marcel": lambda: (arm_marcel(), 3),
    "xs": lambda: (arm_glm(), 3),                       # current production settings
    "xs_monthly": lambda: (arm_glm(), 1),
    "glm_ctx": lambda: (arm_glm(ctx=True), 3),
}
#: per-stat ridge chosen on the TUNE window (glm_a* sweep): rare events need ~30x less
#: shrinkage than the single ALPHA=0.003 the production model uses for everything
ALPHA_TUNED = {"sig": 3e-3, "ctrl": 3e-3, "sig_att": 3e-3, "dist": 3e-3, "clinch": 1e-3, "gnd": 1e-3,
               "td": 1e-4, "kd": 1e-4, "sub": 1e-4, "td_att": 1e-4}
ARMS.update({
    "glm_a1e-2": lambda: (arm_glm(alpha=1e-2), 3),
    "glm_tuned": lambda: (arm_glm(alpha_by_stat=ALPHA_TUNED), 3),
    "glm_tuned_ctx": lambda: (arm_glm(alpha_by_stat=ALPHA_TUNED, ctx=True), 3),
    "glm_tuned_glicko": lambda: (arm_glm(alpha_by_stat=ALPHA_TUNED, ctx=True, extra=glicko_cov), 3),
    "glicko_only": lambda: (arm_glm(alpha=1e-4, ctx=True, extra=glicko_cov, ids_on=False), 3),
    "voleff": lambda: (arm_voleff(alpha_att=ALPHA_TUNED, alpha_acc=1.0), 3),
    "pace": lambda: (arm_pace(alpha=ALPHA_TUNED), 3),
    "position": lambda: (arm_position(alpha=ALPHA_TUNED), 3),
    "comp_l2": lambda: (arm_composition(lam=2.0), 3),
    "comp_l5": lambda: (arm_composition(lam=5.0), 3),
})
for _c in ("0.1", "0.01", "0.001"):
    ARMS[f"voleff_c{_c}"] = (lambda c=float(_c): (arm_voleff(alpha_att=ALPHA_TUNED, alpha_acc=c), 3))
for _h in ("0.75", "2.5", "4"):
    ARMS[f"glm_tuned_hl{_h}"] = (lambda h=float(_h): (arm_glm(alpha_by_stat=ALPHA_TUNED, half_life=h), 3))
HL_TUNED = {"sig": 1.5, "kd": 1.5, "sub": 1.5, "td": 2.5, "td_att": 2.5, "sig_att": 2.5, "ctrl": 4.0,
            "dist": 1.5, "clinch": 1.5, "gnd": 1.5}
ARMS.update({
    "v2_glm": lambda: (arm_glm(alpha_by_stat=ALPHA_TUNED, half_life=HL_TUNED, ctx=True, extra=glicko_cov,
                               keys=["sig", "td", "kd", "sub", "sig_att", "td_att"]), 3),
    "v2_glm_noglicko": lambda: (arm_glm(alpha_by_stat=ALPHA_TUNED, half_life=HL_TUNED, ctx=True,
                                        keys=["sig", "td", "kd", "sub", "sig_att", "td_att"]), 3),
    "ctrl_glm_hl4_a1e-2": lambda: (arm_glm(alpha=1e-2, half_life=4.0, keys=["ctrl"]), 3),
    "comp_l2_hl3": lambda: (arm_composition(lam=2.0, half_life=3.0), 3),
    "comp_l2_hl4": lambda: (arm_composition(lam=2.0, half_life=4.0), 3),
    "comp_l4_hl4": lambda: (arm_composition(lam=4.0, half_life=4.0), 3),
    "comp_l2_hl4_noctx": lambda: (arm_composition(lam=2.0, half_life=4.0, ctx=False), 3),
})
for _l in ("0.3", "1", "3", "10"):
    ARMS[f"comp_l{_l}"] = (lambda l=float(_l): (arm_composition(lam=l), 3))
for _a in ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5"):
    ARMS[f"glm_a{_a}"] = (lambda a=float(_a): (arm_glm(alpha=a), 3))


def run_arm(name: str, obs, robs):
    fp, months = ARMS[name]()
    pred = walk_forward(fp, obs, months=months, robs=robs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_pickle(OUT_DIR / f"arm_{name}.pkl")
    log.info(f"  saved arm {name}: {len(pred)} rows")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def pois_dev(y, mu):
    mu = np.clip(mu, 1e-9, None)
    return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))


def nb_theta(y, mu) -> float:
    """Method-of-moments NB size: Var = mu + mu^2/theta."""
    num = ((y - mu) ** 2 - mu).sum()
    den = (mu ** 2).sum()
    k = max(num / den, 1e-4)
    return 1.0 / k


def _nb_pmf_grid(mu: np.ndarray, theta: float, support: np.ndarray) -> np.ndarray:
    """pmf (len(mu), len(support)); mu may be 0."""
    mu = np.clip(mu, 1e-9, None)
    p = theta / (theta + mu)
    return stats.nbinom.pmf(support[None, :], theta, p[:, None])


def exposure_dist(fin: np.ndarray, sched: np.ndarray):
    """Discrete fight-length distribution: finish bins at midpoints + decision at the
    scheduled end. Returns (T values (n,K), probs (n,K))."""
    n, nb = fin.shape
    mids = (np.arange(nb) + 0.5) * BIN
    end = np.where(sched == 5, 25.0, 15.0)
    keep = mids[None, :] < end[:, None]
    pf = np.where(keep, fin, 0.0)
    pdec = (1 - pf.sum(axis=1)).clip(0, None)
    T = np.concatenate([np.broadcast_to(mids, (n, nb)), end[:, None]], axis=1)
    P = np.concatenate([pf, pdec[:, None]], axis=1)
    P = P / P.sum(axis=1, keepdims=True)
    return T, P


def mixture_cdf(rate: np.ndarray, theta: float, T: np.ndarray, P: np.ndarray,
                support: np.ndarray) -> np.ndarray:
    """CDF over support of the NB(rate*T, theta) mixture over T~P. (n, len(support))."""
    cdf = np.zeros((len(rate), len(support)))
    for k in range(T.shape[1]):
        mu = np.clip(rate * T[:, k], 1e-9, None)
        cdf += P[:, [k]] * stats.nbinom.cdf(support[None, :], theta, (theta / (theta + mu))[:, None])
    return cdf


def fit_hurdle(y, mu):
    """Zero hurdle for control time: P(y=0) = sigmoid(a + b log mu), fitted on TUNE rows,
    and the NB size of the positive part (mean mu / (1 - p0))."""
    from sklearn.linear_model import LogisticRegression
    x = np.log(np.clip(mu, 1e-3, None))[:, None]
    lr = LogisticRegression(C=1e3).fit(x, (y == 0).astype(int))
    p0 = lr.predict_proba(x)[:, 1]
    pos = y > 0
    th = nb_theta(y[pos], mu[pos] / (1 - p0[pos]))
    return lr, th


def hurdle_cdf(rate, lr, theta, T, P, support):
    cdf = np.zeros((len(rate), len(support)))
    for k in range(T.shape[1]):
        mu = np.clip(rate * T[:, k], 1e-3, None)
        p0 = lr.predict_proba(np.log(mu)[:, None])[:, 1]
        m = mu / (1 - p0)
        q = theta / (theta + m)
        f0 = stats.nbinom.pmf(0, theta, q)
        Fpos = (stats.nbinom.cdf(support[None, :], theta, q[:, None]) - f0[:, None]) / (1 - f0[:, None])
        cdf += P[:, [k]] * (p0[:, None] + (1 - p0[:, None]) * np.clip(Fpos, 0, 1))
    return cdf


def crps_from_cdf(cdf: np.ndarray, support: np.ndarray, y: np.ndarray) -> np.ndarray:
    step = np.diff(support, append=support[-1] + (support[-1] - support[-2]))
    ind = (support[None, :] >= y[:, None]).astype(float)
    return (((cdf - ind) ** 2) * step[None, :]).sum(axis=1)


SUPPORT = {"sig": np.arange(0, 451), "td": np.arange(0, 26), "kd": np.arange(0, 8),
           "sub": np.arange(0, 16), "ctrl": np.arange(0, 1501, 5), "sig_att": np.arange(0, 1001),
           "td_att": np.arange(0, 41), "dist": np.arange(0, 401), "clinch": np.arange(0, 151),
           "gnd": np.arange(0, 251)}


def _boot(d: np.ndarray, groups: np.ndarray, n=1000, seed=0):
    """Fight-clustered bootstrap CI of mean(d)."""
    rng = np.random.default_rng(seed)
    g, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnt = np.bincount(inv)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    m = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def score_tune(arms: list[str], stats_keys=CORE) -> pd.DataFrame:
    """Rate deviance on the TUNE window only (2015 -> report start). For choosing
    hyper-parameters without touching the report window."""
    obs, _ = load_frames()
    ids_rep, _ = report_window()
    rep_start = obs.loc[obs["fight_id"].isin(set(ids_rep["fight_id"])), "date"].min()
    preds = {a: pd.read_pickle(OUT_DIR / f"arm_{a}.pkl") for a in arms}
    o = obs[(obs["date"] >= "2015-01-01") & (obs["date"] < rep_start)]
    rows = []
    for s in stats_keys:
        y, m = o[STATS[s]].to_numpy(), o["minutes"].to_numpy()
        rows.append({"stat": s, **{a: pois_dev(y, preds[a].loc[o.index, f"rate_{s}"].to_numpy() * m).mean()
                                   for a in arms if f"rate_{s}" in preds[a]}})
    tab = pd.DataFrame(rows).set_index("stat")
    print(f"TUNE window rate deviance (n={len(o)}), best per stat marked")
    print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
    print(tab.idxmin(axis=1).to_string())
    return tab


def score(arms: list[str], stats_keys=CORE):
    obs, _ = load_frames()
    ids_rep, fin = report_window()
    rep_ids = set(ids_rep["fight_id"])
    rep_start = obs.loc[obs["fight_id"].isin(rep_ids), "date"].min()
    log.info(f"report window starts {rep_start.date()} ({len(rep_ids)} fights)")
    preds = {a: pd.read_pickle(OUT_DIR / f"arm_{a}.pkl") for a in arms}
    common = obs.index
    for p in preds.values():
        common = common.intersection(p.dropna(how="all").index)
    o = obs.loc[common]
    tune = (o["date"] >= "2015-01-01") & (o["date"] < rep_start)
    rep = o["fight_id"].isin(rep_ids) & o["sched_rounds"].isin([3, 5])
    # duration distribution per report row
    fpos = {f: i for i, f in enumerate(ids_rep["fight_id"])}
    r_o = o[rep]
    T, P = exposure_dist(fin[r_o["fight_id"].map(fpos).to_numpy()], r_o["sched_rounds"].to_numpy())
    ET = (T * P).sum(axis=1)
    groups = r_o["fight_id"].to_numpy()
    ref = arms[0]
    rng = np.random.default_rng(0)
    rows = []
    h2h = []
    for s in stats_keys:
        col = STATS[s]
        sup = SUPPORT[s]
        y_t, m_t = o.loc[tune, col].to_numpy(), o.loc[tune, "minutes"].to_numpy()
        y_r, m_r = r_o[col].to_numpy(), r_o["minutes"].to_numpy()
        per_arm = {}
        for a in arms:
            rate = preds[a].loc[common, f"rate_{s}"]
            th = nb_theta(y_t, rate[tune].to_numpy() * m_t)
            rr = rate[rep].to_numpy()
            dev = pois_dev(y_r, rr * m_r)
            if s == "ctrl" and HURDLE_CTRL:
                lr, th = fit_hurdle(y_t, rate[tune].to_numpy() * m_t)
                cdf = hurdle_cdf(rr, lr, th, T, P, sup)
            else:
                cdf = mixture_cdf(rr, th, T, P, sup)
            crps = crps_from_cdf(cdf, sup, y_r)
            mean_tot = rr * ET
            # randomised PIT: u ~ U(F(y-), F(y)); for ctrl the grid is 5s, close enough
            j = np.searchsorted(sup, y_r).clip(0, len(sup) - 1)
            hi_ = cdf[np.arange(len(y_r)), j]
            lo_ = np.where(j > 0, cdf[np.arange(len(y_r)), (j - 1).clip(0)], 0.0)
            pit_lo = lo_ + rng.random(len(y_r)) * (hi_ - lo_)
            per_arm[a] = dict(dev=dev, crps=crps, ae=np.abs(y_r - mean_tot), se=(y_r - mean_tot) ** 2,
                              cov80=((pit_lo >= 0.1) & (pit_lo <= 0.9)).astype(float), theta=th, cdf=cdf)
        for a in arms:
            r = per_arm[a]
            row = {"stat": s, "arm": a, "n": len(y_r), "rate_dev": r["dev"].mean(),
                   "tot_mae": r["ae"].mean(), "tot_rmse": np.sqrt(r["se"].mean()),
                   "crps": r["crps"].mean(), "cov80": r["cov80"].mean(), "theta": r["theta"]}
            if a != ref:
                for k in ("dev", "crps"):
                    d = r[k] - per_arm[ref][k]
                    lo, hi = _boot(d, groups)
                    row[f"d_{k}"] = f"{d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]"
            rows.append(row)
        # h2h: P(attacker lands strictly more than defender) from per-row cdfs, pairing
        # the two rows of each fight under a shared T (independent given T)
        if s in ("sig", "td"):
            for a in arms:
                h2h.append(_h2h(r_o, preds[a].loc[r_o.index, f"rate_{s}"].to_numpy(),
                                per_arm[a]["theta"], T, P, sup, col, a, s))
    pd.set_option("display.width", 250)
    tab = pd.DataFrame(rows)
    print(tab.drop(columns=[]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if h2h:
        hd = pd.DataFrame(h2h)
        print("\nhead-to-head: P(fighter lands more), log loss over fights without a tie")
        print(hd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return tab


def _h2h(r_o, rate, theta, T, P, sup, col, arm, s):
    df = r_o[["fight_id", "fighter_id", col]].copy()
    df["i"] = np.arange(len(df))
    first = df.groupby("fight_id").head(1)
    second = df.drop(first.index)
    pr = first.merge(second, on="fight_id", suffixes=("_a", "_b"))
    ia, ib = pr["i_a"].to_numpy(), pr["i_b"].to_numpy()
    pa = np.zeros(len(pr))
    pb = np.zeros(len(pr))
    for k in range(T.shape[1]):
        fa = _nb_pmf_grid(rate[ia] * T[ia, k], theta, sup)
        fb = _nb_pmf_grid(rate[ib] * T[ib, k], theta, sup)
        cb = np.cumsum(fb, axis=1)
        ca = np.cumsum(fa, axis=1)
        # P(A > B) = sum_x fa(x) * Fb(x-1)
        pa += P[ia, k] * (fa[:, 1:] * cb[:, :-1]).sum(axis=1)
        pb += P[ia, k] * (fb[:, 1:] * ca[:, :-1]).sum(axis=1)
    ya, yb = pr[f"{col}_a"].to_numpy(), pr[f"{col}_b"].to_numpy()
    keep = ya != yb
    q = np.clip(pa / (pa + pb), 1e-6, 1 - 1e-6)[keep]
    yy = (ya > yb)[keep]
    ll = -(yy * np.log(q) + (1 - yy) * np.log(1 - q))
    return {"stat": s, "arm": arm, "n": int(keep.sum()), "log_loss": ll.mean(),
            "brier": ((q - yy) ** 2).mean()}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--stats", default=",".join(CORE))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tune", action="store_true", help="score the tune window only")
    ap.add_argument("--no-hurdle", action="store_true", help="plain NB for control time")
    a = ap.parse_args()
    obs, robs = load_frames(a.rebuild)
    global HURDLE_CTRL
    HURDLE_CTRL = not a.no_hurdle
    for name in filter(None, a.arms.split(",")):
        run_arm(name, obs, robs)
    if a.score:
        arms = a.only.split(",") if a.only else sorted(
            p.stem[4:] for p in OUT_DIR.glob("arm_*.pkl"))
        (score_tune if a.tune else score)(arms, a.stats.split(","))


if __name__ == "__main__":
    main()
