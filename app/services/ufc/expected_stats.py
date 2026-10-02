"""Opponent-adjusted expected fight stats (the "expected strikes / takedowns" model).

For each count stat, a Poisson model of the count fighter A lands on fighter B:

    log E[y_AB] = log(minutes) + mu + off_A - def_B  [+ context + covariates]

fit as a ridge-penalised GLM (shrinking every fighter toward the prior) with exponential
time-decay weights, the Maher / Dixon-Coles team-strength form applied to fighters. It is
refit at quarterly boundaries on bouts strictly before the boundary, and predictions for a
bout come from the latest fit before its date, so nothing downstream sees the bout it is
predicting.

Control time (v2) is not a free Poisson rate but a COMPOSITION of the bout's time:
{A controls, B controls, neither}, a multinomial logit with each fighter's "take top" and
"prevent top" skills. Shares are bounded and sum to one, and B's grappling lowers A's share.

v2 (ALOCKS_XS_V2=1) vs v1, measured by scripts/xs_eval.py (tune 2015-05/2022, report
05/2022-09/2026, 2,450 bouts; see alocks-docs models/expected-stats.md):
  - per-stat ridge: v1's single ALPHA=0.003 is mild for strikes but crushes rare events
    (sklearn scales the penalty to the mean deviance); takedowns, knockdowns and sub
    attempts need ~30x less. Takedown deviance -14%, "who lands more takedowns" log loss
    0.680 -> 0.585.
  - per-stat half-life (control 4 years, takedowns 2.5) and division / era / 5-round
    context. Optionally (ALOCKS_XS_V2_GLICKO=1) pre-fight Glicko of both fighters as
    covariates, so a fighter with few bouts is shrunk toward what their ratings predict
    instead of toward the league mean: better stats, worse winner features (see V2_GLICKO).
  - control via the composition model: deviance -2.4%.

Outputs per (fight_id, fighter_id):
  xs_{stat}_for      expected per-minute rate this fighter lands vs this opponent
  xs_{stat}_against  expected per-minute rate this fighter absorbs from this opponent
  xs_{stat}_diff     for - against
  xs_fight_*         (v2) bout-level, identical for both corners: sig/td/sub pace (both
                     fighters' expected rates summed) and expected ground share (share of
                     the bout either fighter spends in control). Emitted once as fight_*.
  xsres_{stat}       actual - expected per minute: a POST-fight quantity, never a feature
                     (the xs_ prefix match in winner_feature_columns excludes it).

Differs from the 15-dimension Glicko, which updates Elo-style per round on transformed
observables: this is a proper count model with exposure, in interpretable units.
"""
from __future__ import annotations

import logging
import os
from datetime import date

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import PoissonRegressor

log = logging.getLogger("expected_stats")

#: stat name -> source column on the load_fight_data frame (per-fighter fight totals).
STATS = {
    "sig": "sig_str_landed",
    "td": "td_landed",
    "kd": "kd",
    "sub": "sub_att",
    "ctrl": "ctrl_seconds",  # v1: Poisson rate of control-seconds per minute; v2: composition
}
# v1. Chosen on predictive Poisson deviance of 2016+ bouts, conditional on actual minutes.
HALF_LIFE_YEARS = 1.5
ALPHA = 0.003  # ridge strength on the per-fighter effects (shrinkage to league mean)
REFIT_MONTHS = 3
FIRST_REFIT = date(2012, 1, 1)

#: Off until the winner model is retrained on it (served features change). Winner walk-
#: forward, 4-member ensemble, 2,514 fights: v2 -0.0003 [-0.0040, +0.0031] vs v1, neutral.
V2 = os.environ.get("ALOCKS_XS_V2") == "1"
#: Pre-fight Glicko of both fighters as covariates. The better STAT forecaster (takedown
#: deviance -17% vs -14% without; "who lands more takedowns" 0.554 vs 0.585) but it feeds
#: Glicko into the winner model a second time: ensemble +0.0041 [+0.0003, +0.0080] worse.
#: Off for the winner features; turn on (=1) only for stat forecasts / props.
V2_GLICKO = os.environ.get("ALOCKS_XS_V2_GLICKO") == "1"
#: v2 settings, each chosen on the TUNE window only (scripts/xs_eval.py --tune).
V2_ALPHA = {"sig": 3e-3, "td": 1e-4, "kd": 1e-4, "sub": 1e-4}
V2_HALF_LIFE = {"sig": 1.5, "td": 2.5, "kd": 1.5, "sub": 1.5}
#: composition model for control: ridge on fighter terms, half-life, no context terms
V2_CTRL_LAMBDA = 2.0
V2_CTRL_HALF_LIFE = 4.0
#: column scale (penalty alpha/scale^2): context and Glicko are lightly penalised
CTX_SCALE, COV_SCALE = 10.0, 3.0


def _boundaries(start: date, end: date) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthBegin(1),
                              freq=f"{REFIT_MONTHS}MS"))


def _observations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (fight, attacker): attacker id, defender id, minutes, stat counts, the
    defender's control seconds, bout context and both fighters' pre-fight Glicko."""
    glicko = [c for c in df.columns if c.startswith("glicko_") and
              (not c.startswith("glicko_meta") or c == "glicko_meta_rounds_seen")]
    ctx = [c for c in ("weight_class", "time_format") if c in df.columns]
    base = df[["fight_id", "date", "stats_fighter_id", "fight_time_seconds"]
              + list(STATS.values()) + ctx + glicko].copy()
    opp = (base[["fight_id", "stats_fighter_id", "ctrl_seconds"] + glicko]
           .rename(columns={"stats_fighter_id": "opp_id", "ctrl_seconds": "opp_ctrl",
                            **{c: f"opp_{c}" for c in glicko}}))
    obs = base.merge(opp, on="fight_id")
    obs = obs[obs["stats_fighter_id"] != obs["opp_id"]]
    obs["minutes"] = obs["fight_time_seconds"].astype(float) / 60.0
    obs["date"] = pd.to_datetime(obs["date"])
    return obs.reset_index(drop=True)


# ---------------------------------------------------------------------------
# design pieces
# ---------------------------------------------------------------------------

def _context(df: pd.DataFrame, levels: dict | None = None) -> tuple[np.ndarray, dict]:
    """One-hot division (men's and women's separate), 5-round flag, 3-year era. Levels are
    taken from the training frame; an unseen future era maps to the latest one."""
    from app.services.ufc.model import _classify_weight_class
    div = df["weight_class"].map(_classify_weight_class) if "weight_class" in df else pd.Series("unknown", df.index)
    era = (df["date"].dt.year // 3) * 3
    if levels is None:
        levels = {"division": sorted(div.unique()), "era": sorted(era.unique())}
    era = era.clip(upper=levels["era"][-1])
    cols = [(div == d).to_numpy(float) for d in levels["division"]]
    five = (df["time_format"].fillna("") == "5-5-5-5-5") if "time_format" in df else pd.Series(False, df.index)
    cols.append(five.to_numpy(float))
    cols += [(era == e).to_numpy(float) for e in levels["era"]]
    return np.column_stack(cols), levels


def _covariates(df: pd.DataFrame, stats: tuple | None = None, use: bool = True):
    """Both fighters' pre-fight Glicko (log rounds seen), standardised on the training frame."""
    cols = [c for c in df.columns if c.startswith("glicko_")] if use else []
    if not cols:
        return np.zeros((len(df), 0)), stats
    X = df[cols + [f"opp_{c}" for c in cols]].astype(float).copy()
    for c in [c for c in X.columns if c.endswith("meta_rounds_seen")]:
        X[c] = np.log1p(X[c].clip(lower=0))
    if stats is None:
        stats = (X.mean(), X.std().replace(0, 1.0))
    return ((X - stats[0]) / stats[1]).fillna(0.0).to_numpy(), stats


def _pair(df: pd.DataFrame, ids: pd.Index, swap: bool = False) -> sparse.csr_matrix:
    """+1 at the attacker's offence column, -1 at the defender's defence column."""
    a = ids.get_indexer(df["opp_id" if swap else "stats_fighter_id"])
    d = ids.get_indexer(df["stats_fighter_id" if swap else "opp_id"])
    r, n = np.arange(len(df)), len(ids)
    return sparse.hstack([
        sparse.csr_matrix((np.ones(len(df)), (r, a)), shape=(len(df), n)),
        sparse.csr_matrix((-np.ones(len(df)), (r, d)), shape=(len(df), n)),
    ]).tocsr()


def _decay(dates: pd.Series, asof: pd.Timestamp, half_life: float) -> np.ndarray:
    return 0.5 ** ((asof - dates).dt.days.to_numpy() / 365.25 / half_life)


# ---------------------------------------------------------------------------
# control-time composition model
# ---------------------------------------------------------------------------

def fit_composition(train: pd.DataFrame, asof: pd.Timestamp, ids: pd.Index,
                    lam: float = V2_CTRL_LAMBDA, half_life: float = V2_CTRL_HALF_LIFE):
    """Multinomial logit on how a bout's minutes split into {A controls, B controls, neither}:

        eta_A = c + top_A - topdef_B,   eta_B = c + top_B - topdef_A
        share_A = e^eta_A / (1 + e^eta_A + e^eta_B)

    Pseudo-counts are control MINUTES, ridge `lam` on the fighter terms. The model is
    symmetric, so each bout enters once. Returns predict(df) -> (share_A, share_B)."""
    tr = train.groupby("fight_id", sort=False).head(1)
    Xa, Xb = _pair(tr, ids), _pair(tr, ids, swap=True)
    ca = tr["ctrl_seconds"].to_numpy(float) / 60
    cb = tr["opp_ctrl"].to_numpy(float) / 60
    tot = np.maximum(tr["minutes"].to_numpy(), ca + cb)       # neither = tot - ca - cb >= 0
    w = _decay(tr["date"], asof, half_life)

    def f(th):
        c, b = th[0], th[1:]
        ea, eb = c + Xa @ b, c + Xb @ b
        m = np.maximum(np.maximum(ea, eb), 0)
        logD = m + np.log(np.exp(-m) + np.exp(ea - m) + np.exp(eb - m))
        sa, sb = np.exp(ea - logD), np.exp(eb - logD)
        ga, gb = w * (ca - tot * sa), w * (cb - tot * sb)
        obj = -(w * (ca * ea + cb * eb - tot * logD)).sum() + 0.5 * lam * (b @ b)
        grad_b = -(Xa.T @ ga + Xb.T @ gb) + lam * b
        return obj, np.concatenate([[-(ga + gb).sum()], grad_b])

    th0 = np.zeros(1 + Xa.shape[1])
    th0[0] = -2.0
    th = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": 500}).x

    def predict(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ea, eb = th[0] + _pair(df, ids) @ th[1:], th[0] + _pair(df, ids, swap=True) @ th[1:]
        D = 1 + np.exp(ea) + np.exp(eb)
        return np.exp(ea) / D, np.exp(eb) / D
    return predict


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _fit_glm(X_tr, y_rate, weight, X_tg, X_tg_swap, alpha):
    m = PoissonRegressor(alpha=alpha, max_iter=300)
    m.fit(X_tr, y_rate, sample_weight=weight)
    return m.predict(X_tg), m.predict(X_tg_swap)


def compute_expected_stats(df: pd.DataFrame, v2: bool | None = None,
                           glicko: bool | None = None) -> pd.DataFrame:
    """Expected stats for every row of a load_fight_data frame (incl. unplayed).
    v2 / glicko default to the ALOCKS_XS_V2 / ALOCKS_XS_V2_GLICKO flags (winner features);
    the display table passes v2=True, glicko=True (best stat forecasts)."""
    v2 = V2 if v2 is None else v2
    glicko = V2_GLICKO if glicko is None else glicko
    obs = _observations(df)
    played = obs[(obs["minutes"] > 0) & obs[STATS["sig"]].notna()]
    ids = pd.Index(pd.unique(pd.concat([obs["stats_fighter_id"], obs["opp_id"]])))
    use_v2 = v2 and "weight_class" in obs

    out = pd.DataFrame({"fight_id": obs["fight_id"], "stats_fighter_id": obs["stats_fighter_id"]})
    for s in STATS:
        out[f"xs_{s}_for"] = np.nan
        out[f"xs_{s}_against"] = np.nan
    if use_v2:
        out["xs_ctrl_share_for"] = np.nan
        out["xs_ctrl_share_against"] = np.nan

    bounds = _boundaries(FIRST_REFIT, obs["date"].max().date())
    log.info(f"  Expected stats ({'v2' if use_v2 else 'v1'}): {len(bounds) - 1} refits x {len(STATS)} stats")
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        target = (obs["date"] >= lo) & (obs["date"] < hi)
        if not target.any():
            continue
        train = played[played["date"] < lo]
        if len(train) < 500:
            continue
        tgt = obs[target]
        # The opponent's view of the same bout: what this fighter absorbs.
        tgt_swap = tgt.assign(stats_fighter_id=tgt["opp_id"], opp_id=tgt["stats_fighter_id"])
        if use_v2:
            gl = [c for c in tgt.columns if c.startswith("glicko_")]
            swap_cols = {c: f"opp_{c}" for c in gl} | {f"opp_{c}": c for c in gl}
            tgt_swap = tgt_swap.rename(columns=swap_cols)
        minutes = train["minutes"].to_numpy()

        if not use_v2:
            X = _pair(train, ids)
            decay = _decay(train["date"], lo, HALF_LIFE_YEARS)
            for s, col in STATS.items():
                # Rate target with exposure as weight == Poisson with log(minutes) offset.
                f, a = _fit_glm(X, train[col].to_numpy(float) / minutes, minutes * decay,
                                _pair(tgt, ids), _pair(tgt_swap, ids), ALPHA)
                out.loc[tgt.index, f"xs_{s}_for"] = f
                out.loc[tgt.index, f"xs_{s}_against"] = a
            continue

        C_tr, lv = _context(train)
        G_tr, gstats = _covariates(train, use=glicko)

        def design(d):
            C, _ = _context(d, lv)
            G, _ = _covariates(d, gstats, use=glicko)
            return sparse.hstack([_pair(d, ids), sparse.csr_matrix(C * CTX_SCALE),
                                  sparse.csr_matrix(G * COV_SCALE)]).tocsr()
        X_tr = sparse.hstack([_pair(train, ids), sparse.csr_matrix(C_tr * CTX_SCALE),
                              sparse.csr_matrix(G_tr * COV_SCALE)]).tocsr()
        X_tg, X_sw = design(tgt), design(tgt_swap)
        for s in ("sig", "td", "kd", "sub"):
            w = minutes * _decay(train["date"], lo, V2_HALF_LIFE[s])
            f, a = _fit_glm(X_tr, train[STATS[s]].to_numpy(float) / minutes, w, X_tg, X_sw, V2_ALPHA[s])
            out.loc[tgt.index, f"xs_{s}_for"] = f
            out.loc[tgt.index, f"xs_{s}_against"] = a
        comp = fit_composition(train, lo, ids)
        sa, sb = comp(tgt)
        out.loc[tgt.index, "xs_ctrl_for"] = 60.0 * sa
        out.loc[tgt.index, "xs_ctrl_against"] = 60.0 * sb
        out.loc[tgt.index, "xs_ctrl_share_for"] = sa
        out.loc[tgt.index, "xs_ctrl_share_against"] = sb

    for s, col in STATS.items():
        out[f"xs_{s}_diff"] = out[f"xs_{s}_for"] - out[f"xs_{s}_against"]
        actual = obs[col].astype(float) / obs["minutes"].where(obs["minutes"] > 0)
        out[f"xsres_{s}"] = actual.to_numpy() - out[f"xs_{s}_for"].to_numpy()
    if use_v2:
        # Bout-level: the same number on both corners (emitted once as fight_xs_fight_*).
        for s in ("sig", "td", "sub"):
            out[f"xs_fight_{s}_pace"] = out[f"xs_{s}_for"] + out[f"xs_{s}_against"]
        out["xs_fight_ground_share"] = out["xs_ctrl_share_for"] + out["xs_ctrl_share_against"]
        out = out.drop(columns=["xs_ctrl_share_for", "xs_ctrl_share_against"])
    return out
