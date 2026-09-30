"""Opponent-adjusted expected fight stats (the "expected strikes / takedowns" model).

For each stat, a Poisson model of the count fighter A lands on fighter B:

    log E[y_AB] = log(minutes) + mu + off_A - def_B

fit as a ridge-penalised GLM (shrinking every fighter toward the league mean) with
exponential time-decay weights, the Maher / Dixon-Coles team-strength form applied to
fighters. It is refit at quarterly boundaries on bouts strictly before the boundary, and
predictions for a bout come from the latest fit before its date, so nothing downstream
sees the bout it is predicting.

Outputs per (fight_id, fighter_id):
  xs_{stat}_for      expected per-minute rate this fighter lands vs this opponent
  xs_{stat}_against  expected per-minute rate this fighter absorbs from this opponent
  xs_{stat}_diff     for - against
  xsres_{stat}       actual - expected per minute: a POST-fight quantity, never a feature
                     (the xs_ prefix match in winner_feature_columns excludes it). It is
                     what the "landed 5 more per minute than expected" display reads.

Differs from the 15-dimension Glicko, which updates Elo-style per round on transformed
observables: this is a proper count model with exposure, in interpretable units.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import PoissonRegressor

log = logging.getLogger("expected_stats")

#: stat name -> source column on the load_fight_data frame (per-fighter fight totals).
STATS = {
    "sig": "sig_str_landed",
    "td": "td_landed",
    "kd": "kd",
    "sub": "sub_att",
    "ctrl": "ctrl_seconds",  # modelled as a Poisson rate of control-seconds per minute
}
# Chosen on predictive Poisson deviance of 2016+ bouts. Raw career averages, which most
# of the winner model's features are built from, did WORSE than the league mean on every
# stat. These beat the league mean by 28% on sig strikes, 19% on control time, 8% on sub
# attempts and 3% on takedowns; knockdowns are essentially unpredictable either way.
HALF_LIFE_YEARS = 1.5
ALPHA = 0.003  # ridge strength on the per-fighter effects (shrinkage to league mean)
REFIT_MONTHS = 3
FIRST_REFIT = date(2012, 1, 1)


def _boundaries(start: date, end: date) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthBegin(1),
                              freq=f"{REFIT_MONTHS}MS"))


def _observations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (fight, attacker): attacker id, defender id, minutes, stat counts."""
    base = df[["fight_id", "date", "stats_fighter_id", "fight_time_seconds"]
              + list(STATS.values())].copy()
    opp = (base[["fight_id", "stats_fighter_id"]]
           .rename(columns={"stats_fighter_id": "opp_id"}))
    obs = base.merge(opp, on="fight_id")
    obs = obs[obs["stats_fighter_id"] != obs["opp_id"]]
    obs["minutes"] = obs["fight_time_seconds"].astype(float) / 60.0
    obs["date"] = pd.to_datetime(obs["date"])
    return obs.reset_index(drop=True)


def compute_expected_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Expected-stat features for every row of a load_fight_data frame (incl. unplayed)."""
    obs = _observations(df)
    played = obs[(obs["minutes"] > 0) & obs[STATS["sig"]].notna()]
    ids = pd.Index(pd.unique(pd.concat([obs["stats_fighter_id"], obs["opp_id"]])))
    n = len(ids)

    out = pd.DataFrame({"fight_id": obs["fight_id"], "stats_fighter_id": obs["stats_fighter_id"]})
    for s in STATS:
        out[f"xs_{s}_for"] = np.nan
        out[f"xs_{s}_against"] = np.nan

    bounds = _boundaries(FIRST_REFIT, obs["date"].max().date())
    log.info(f"  Expected stats: {len(bounds) - 1} refits x {len(STATS)} stats")
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        target = (obs["date"] >= lo) & (obs["date"] < hi)
        if not target.any():
            continue
        train = played[played["date"] < lo]
        if len(train) < 500:
            continue
        a = ids.get_indexer(train["stats_fighter_id"])
        d = ids.get_indexer(train["opp_id"])
        rows = np.arange(len(train))
        X = sparse.hstack([
            sparse.csr_matrix((np.ones(len(train)), (rows, a)), shape=(len(train), n)),
            sparse.csr_matrix((-np.ones(len(train)), (rows, d)), shape=(len(train), n)),
        ]).tocsr()
        age_years = (lo - train["date"]).dt.days.to_numpy() / 365.25
        decay = 0.5 ** (age_years / HALF_LIFE_YEARS)
        minutes = train["minutes"].to_numpy()

        tgt = obs[target]
        ta = ids.get_indexer(tgt["stats_fighter_id"])
        td = ids.get_indexer(tgt["opp_id"])
        for s, col in STATS.items():
            # Rate target with exposure as weight == Poisson with log(minutes) offset.
            y = train[col].to_numpy(float) / minutes
            m = PoissonRegressor(alpha=ALPHA, max_iter=300)
            m.fit(X, y, sample_weight=minutes * decay)
            # X carries +1 in the attacker's column and -1 in the defender's, so the
            # fitted defender coefficient already enters as "- def_B".
            off = m.coef_[:n]
            dfn = m.coef_[n:]
            mu = m.intercept_
            out.loc[tgt.index, f"xs_{s}_for"] = np.exp(mu + off[ta] - dfn[td])
            out.loc[tgt.index, f"xs_{s}_against"] = np.exp(mu + off[td] - dfn[ta])

    for s, col in STATS.items():
        out[f"xs_{s}_diff"] = out[f"xs_{s}_for"] - out[f"xs_{s}_against"]
        actual = obs[col].astype(float) / obs["minutes"].where(obs["minutes"] > 0)
        out[f"xsres_{s}"] = actual.to_numpy() - out[f"xs_{s}_for"].to_numpy()
    return out
