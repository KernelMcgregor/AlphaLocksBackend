"""Competing-risks fight simulator.

WHY THIS EXISTS
---------------
Holmes, McHale & Żychaluk (Int. J. Forecasting 2023) showed that the gain in MMA
forecasting does not come from a better classifier — their logistic regression on career
stat differentials scored 47.7%, worse than a coin flip — but from opponent-adjusted
skill estimates fed into a simulation. This repo already has the skill estimates: the
15-dimensional Glicko ratings, with explicit attack/defence pairs (ko/kod, sub/subd,
td/tdd, str_acc/str_def). What was missing is the layer that turns ratings into a fight.

WHAT IT MODELS
--------------
A competing-risks (multi-state) model rather than a full positional Markov chain. At
each time step an ongoing fight can be ended by one of four absorbing events —
red KO, blue KO, red submission, blue submission — and if none fires before the final
bell the bout goes to a decision scored on accumulated output and control.

This is deliberately simpler than modelling standing/clinch/ground transitions. The
positional data needed to fit transition rates honestly (time in each position) is not
in `ufc_fight_stats`; `est_standing_min`/`est_ground_min` are derived estimates, not
observations. Fitting a richer chain to data that cannot identify it would produce
confident nonsense. Hazards conditioned on Glicko are identifiable from what we have.

WHAT IT PRODUCES
----------------
One simulation yields the joint distribution over winner x method x round, so every
market is a marginal of the same object and the prices cannot contradict each other:

    P(red wins)                      -> moneyline
    P(KO) / P(sub) / P(decision)     -> method
    P(red by KO)                     -> fighter x method
    round distribution               -> round props, fight-goes-the-distance

VALIDATION
----------
Both outputs are scored on REALIZED OUTCOMES (Brier, log loss). The market is a
benchmark reported afterwards, never a training target. The winner side additionally
has a market to compare against; the method side does not yet (the DB holds 55 rows of
method odds from a single snapshot).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("model")

# Absorbing outcomes
KO_RED, KO_BLUE, SUB_RED, SUB_BLUE, DECISION = 0, 1, 2, 3, 4
METHOD_KO, METHOD_SUB, METHOD_DEC = 0, 1, 2

STEP_SECONDS = 10.0
DEFAULT_SIMS = 5000


class HazardRateModel:
    """Maps Glicko ratings to per-minute hazard rates for each ending.

    Fitted with Poisson regression on TRAINING fights only. The natural target is
    "did this fight end this way, and how long did it take", which is exactly a Poisson
    exposure model: event count (0 or 1) with log-exposure offset log(minutes).

    Features are deliberately few. There are ~6,000 modern-era fights and four hazards
    to fit; a wide feature set here would fit noise, which is the failure mode this
    whole project has been correcting.
    """

    # (attacker dimension, defender dimension) driving each hazard
    HAZARD_SPECS = {
        "ko": ("glicko_ko", "glicko_kod"),
        "sub": ("glicko_sub", "glicko_subd"),
    }

    # Glicko dimensions that describe who wins a SCORED fight. A decision resolves
    # ~49% of bouts, so this branch matters as much as both hazards combined; with only
    # three inputs the simulator produced a red_prob std of 0.048 and barely
    # discriminated between fights.
    DECISION_DIMS = (
        "pts", "str_vol", "str_acc", "str_def", "td", "tdd", "ctrl",
        "dist", "clinch", "gnd", "durability",
    )

    # Every column the model reads. NaNs here are filled from TRAINING medians only.
    INPUT_COLS = (
        "red_glicko_ko", "blue_glicko_ko", "red_glicko_kod", "blue_glicko_kod",
        "red_glicko_sub", "blue_glicko_sub", "red_glicko_subd", "blue_glicko_subd",
        "diff_glicko_pts", "is_five_round",
        "diff_output_rate", "diff_control_rate",
    ) + tuple(f"diff_glicko_{d}" for d in DECISION_DIMS)

    # (HAZARD_DIMS is a subset of DECISION_DIMS, so INPUT_COLS already covers it.)

    def __init__(self, ridge: float = 1.0):
        self.ridge = ridge
        self.coef_: dict[str, np.ndarray] = {}
        self.scale_: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.fill_: dict[str, float] = {}
        self.decision_coef_: np.ndarray | None = None
        self.fitted_ = False

    # -- missing values -------------------------------------------------------
    # Debut fighters have no `recent_*` history, so the decision features arrive with
    # NaNs. A single NaN poisons the standardiser's mean, turns the whole design matrix
    # NaN, trips the IRLS finiteness guard on iteration one, and leaves beta at its zero
    # initialisation — a fit that silently "succeeds" while predicting 0.5 for every
    # decision. Filled from training medians, and _fit_* now warns if it never moved.

    def _fit_fill(self, frame: pd.DataFrame) -> None:
        for c in self.INPUT_COLS:
            if c in frame:
                med = float(np.nanmedian(frame[c].to_numpy(float)))
                self.fill_[c] = 0.0 if med != med else med

    def _col(self, frame: pd.DataFrame, name: str) -> np.ndarray:
        v = frame[name].to_numpy(float) if name in frame else np.zeros(len(frame))
        return np.nan_to_num(v, nan=self.fill_.get(name, 0.0))

    # -- scaling --------------------------------------------------------------
    # Glicko dimensions run to std ~230 and control-rate differentials to std ~66.
    # Unstandardised, IRLS overflows on the first step, the finiteness guard trips, and
    # the fit silently returns its zero initialisation — which is what made every
    # simulated decision a coin flip. Standardising also makes one ridge value
    # meaningful across features on wildly different scales.

    def _fit_scaler(self, key: str, X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        mu[0], sd[0] = 0.0, 1.0            # leave the intercept column alone
        sd[sd < 1e-9] = 1.0
        self.scale_[key] = (mu, sd)
        return (X - mu) / sd

    def _apply_scaler(self, key: str, X: np.ndarray) -> np.ndarray:
        mu, sd = self.scale_[key]
        return (X - mu) / sd

    # -- feature construction -------------------------------------------------

    # Dimensions that plausibly drive HOW a fight ends, beyond the finish-specific
    # attack/defence pair. The same list is used for both hazards rather than
    # hand-picking per hazard, which would be fitting the dev split.
    # MEASURED, not assumed: adding ("str_vol","str_acc","str_def","td","ctrl",
    # "durability") here made the hazards unstable — per-fight rates went extreme,
    # red_prob spanned the full [0,1], and method log loss rose from 1.016 to 1.411
    # against a 1.023 base rate. A Poisson hazard with a log link is far more sensitive
    # to extra dimensions than the decision logit, because errors exponentiate. Kept
    # empty; the finish-specific attack/defence pair plus the overall skill gap is what
    # this much data supports.
    HAZARD_DIMS: tuple[str, ...] = ()

    def _hazard_features(self, frame: pd.DataFrame, att: np.ndarray, dfn: np.ndarray,
                         five_round: np.ndarray, sign: float) -> np.ndarray:
        """Features for one side's finishing hazard.

        `sign` is +1 when modelling red and -1 for blue: every diff_* column is
        red-minus-blue, so it must be flipped to read from blue's perspective.
        """
        skill_gap = sign * self._col(frame, "diff_glicko_pts")
        cols = [np.ones_like(att), att, dfn, att - dfn, five_round, skill_gap]
        cols += [sign * self._col(frame, f"diff_glicko_{d}") for d in self.HAZARD_DIMS]
        return np.column_stack(cols)

    def _decision_features(self, frame: pd.DataFrame) -> np.ndarray:
        cols = [np.ones(len(frame)),
                self._col(frame, "diff_output_rate"),
                self._col(frame, "diff_control_rate")]
        cols += [self._col(frame, f"diff_glicko_{d}") for d in self.DECISION_DIMS]
        return np.column_stack(cols)

    # -- fitting --------------------------------------------------------------

    def _fit_poisson(self, X: np.ndarray, y: np.ndarray, exposure_min: np.ndarray) -> np.ndarray:
        """Poisson regression with a log-exposure offset, via IRLS with ridge.

        sklearn's PoissonRegressor has no offset parameter, and the exposure here is
        essential — a fight ending by KO in round 1 is far stronger evidence of a high
        KO hazard than one ending by KO in round 3.
        """
        offset = np.log(np.clip(exposure_min, 1e-3, None))
        beta = np.zeros(X.shape[1])
        beta[0] = np.log(max(y.sum(), 1.0) / max(exposure_min.sum(), 1e-3))

        for _ in range(50):
            eta = offset + X @ beta
            mu = np.exp(np.clip(eta, -30, 30))
            W = np.clip(mu, 1e-8, None)
            z = eta - offset + (y - mu) / W

            XtW = X.T * W
            A = XtW @ X + self.ridge * np.eye(X.shape[1])
            A[0, 0] -= self.ridge          # do not penalise the intercept
            try:
                new = np.linalg.solve(A, XtW @ z)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(new)):
                log.warning("  simulator: Poisson IRLS diverged; keeping last stable beta")
                break
            step = np.max(np.abs(new - beta))
            beta = new
            if step < 1e-8:
                break
        return beta

    def fit(self, train: pd.DataFrame) -> "HazardRateModel":
        """`train` is one row per fight with red_/blue_ Glicko columns and outcome."""
        self._fit_fill(train)
        minutes = np.clip(train["fight_minutes"].to_numpy(float), 0.5, None)
        five = self._col(train, "is_five_round")

        for side, prefix, opp in (("red", "red_", "blue_"), ("blue", "blue_", "red_")):
            for hz, (att_dim, def_dim) in self.HAZARD_SPECS.items():
                att = self._col(train, f"{prefix}{att_dim}")
                dfn = self._col(train, f"{opp}{def_dim}")
                key = f"{side}_{hz}"
                sign = 1.0 if side == "red" else -1.0
                X = self._fit_scaler(
                    key, self._hazard_features(train, att, dfn, five, sign))
                y = train[f"{key}_win"].to_numpy(float)
                self.coef_[key] = self._fit_poisson(X, y, minutes)

        # Decision scoring: among fights that actually reached a decision, how often did
        # red win, as a function of output/control/points differentials.
        dec = train[train["went_to_decision"] == 1]
        if len(dec) >= 50:
            X = self._fit_scaler("decision", self._decision_features(dec))
            y = dec["red_wins"].to_numpy(float)
            self.decision_coef_ = self._fit_logistic(X, y)
        else:
            # Too few decisions to fit: fall back to an even split rather than a
            # spuriously confident one.
            width = 3 + len(self.DECISION_DIMS)
            self.scale_["decision"] = (np.zeros(width), np.ones(width))
            self.decision_coef_ = np.zeros(width)

        self.fitted_ = True
        return self

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        beta = np.zeros(X.shape[1])
        for _ in range(50):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
            W = np.clip(p * (1 - p), 1e-8, None)
            z = X @ beta + (y - p) / W
            XtW = X.T * W
            A = XtW @ X + self.ridge * np.eye(X.shape[1])
            A[0, 0] -= self.ridge
            try:
                new = np.linalg.solve(A, XtW @ z)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(new)):
                log.warning("  simulator: logistic IRLS diverged; keeping last stable beta")
                break
            step = np.max(np.abs(new - beta))
            beta = new
            if step < 1e-8:
                break
        if np.allclose(beta[1:], 0.0):
            log.warning(
                "  simulator: decision model has all-zero slopes — every simulated "
                "decision will be a coin flip. Check for NaNs in the inputs."
            )
        return beta

    # -- prediction -----------------------------------------------------------

    def hazards(self, fights: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-minute hazard rates for each ending, one value per fight."""
        five = self._col(fights, "is_five_round")
        out = {}
        for side, prefix, opp in (("red", "red_", "blue_"), ("blue", "blue_", "red_")):
            for hz, (att_dim, def_dim) in self.HAZARD_SPECS.items():
                att = self._col(fights, f"{prefix}{att_dim}")
                dfn = self._col(fights, f"{opp}{def_dim}")
                key = f"{side}_{hz}"
                sign = 1.0 if side == "red" else -1.0
                X = self._apply_scaler(
                    key, self._hazard_features(fights, att, dfn, five, sign))
                eta = X @ self.coef_[key]
                out[key] = np.exp(np.clip(eta, -30, 5))
        return out

    def decision_red_prob(self, fights: pd.DataFrame) -> np.ndarray:
        X = self._apply_scaler("decision", self._decision_features(fights))
        return 1.0 / (1.0 + np.exp(-np.clip(X @ self.decision_coef_, -30, 30)))


def simulate(
    fights: pd.DataFrame,
    rate_model: HazardRateModel,
    n_sims: int = DEFAULT_SIMS,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate every fight `n_sims` times. Returns one row per fight.

    Vectorised across fights AND simulations: all fights step through time together as
    one flat array. Looping per fight is ~5x slower for no benefit. float32 halves the
    memory traffic, which is the bottleneck at this size.
    """
    n_fights = len(fights)
    if n_fights == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    hz = rate_model.hazards(fights)
    dec_p = rate_model.decision_red_prob(fights)

    # Per-step probability from a per-minute rate: 1 - exp(-rate * dt)
    dt_min = STEP_SECONDS / 60.0
    def step_prob(rate):
        return np.repeat(
            (1.0 - np.exp(-np.clip(rate, 0, None) * dt_min)).astype(np.float32), n_sims
        )

    p_ko_r, p_ko_b = step_prob(hz["red_ko"]), step_prob(hz["blue_ko"])
    p_sb_r, p_sb_b = step_prob(hz["red_sub"]), step_prob(hz["blue_sub"])

    rounds = np.nan_to_num(fights["scheduled_rounds"].to_numpy(float), nan=3.0)
    total_steps = np.repeat(
        (rounds * 5.0 * 60.0 / STEP_SECONDS).astype(np.int32), n_sims
    )
    max_steps = int(total_steps.max())

    N = n_fights * n_sims
    alive = np.ones(N, dtype=bool)
    outcome = np.full(N, DECISION, dtype=np.int8)
    end_step = total_steps.copy()

    for s in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        # Fights whose scheduled time has elapsed stop being at risk
        expired = total_steps[idx] <= s
        if expired.any():
            alive[idx[expired]] = False
            idx = idx[~expired]
            if idx.size == 0:
                break

        u = rng.random((4, idx.size), dtype=np.float32)
        ev_ko_r = u[0] < p_ko_r[idx]
        ev_ko_b = u[1] < p_ko_b[idx]
        ev_sb_r = u[2] < p_sb_r[idx]
        ev_sb_b = u[3] < p_sb_b[idx]

        any_ev = ev_ko_r | ev_ko_b | ev_sb_r | ev_sb_b
        if not any_ev.any():
            continue

        gi = idx[any_ev]
        # Competing risks: if several fire in the same step, pick one at random rather
        # than always preferring the first — order must not create a systematic bias.
        stacked = np.stack([ev_ko_r[any_ev], ev_ko_b[any_ev],
                            ev_sb_r[any_ev], ev_sb_b[any_ev]])
        weights = stacked.astype(np.float32) * rng.random((4, gi.size), dtype=np.float32)
        chosen = np.argmax(weights, axis=0).astype(np.int8)

        outcome[gi] = chosen
        end_step[gi] = s
        alive[gi] = False

    # Decisions: resolve by the fitted decision model
    is_dec = outcome == DECISION
    if is_dec.any():
        dec_draw = rng.random(int(is_dec.sum()), dtype=np.float32)
        red_wins_dec = dec_draw < np.repeat(dec_p.astype(np.float32), n_sims)[is_dec]
    else:
        red_wins_dec = np.zeros(0, dtype=bool)

    red_won = np.zeros(N, dtype=bool)
    red_won[outcome == KO_RED] = True
    red_won[outcome == SUB_RED] = True
    red_won[is_dec] = red_wins_dec

    method = np.full(N, METHOD_DEC, dtype=np.int8)
    method[(outcome == KO_RED) | (outcome == KO_BLUE)] = METHOD_KO
    method[(outcome == SUB_RED) | (outcome == SUB_BLUE)] = METHOD_SUB

    # Aggregate back to one row per fight
    def per_fight(a):
        return a.reshape(n_fights, n_sims)

    rw = per_fight(red_won)
    mt = per_fight(method)
    es = per_fight(end_step)

    end_round = np.floor(es * STEP_SECONDS / 300.0) + 1

    return pd.DataFrame({
        "fight_id": fights["fight_id"].values if "fight_id" in fights else fights.index,
        "red_prob": rw.mean(axis=1),
        "p_ko": (mt == METHOD_KO).mean(axis=1),
        "p_sub": (mt == METHOD_SUB).mean(axis=1),
        "p_dec": (mt == METHOD_DEC).mean(axis=1),
        "p_red_ko": (rw & (mt == METHOD_KO)).mean(axis=1),
        "p_blue_ko": (~rw & (mt == METHOD_KO)).mean(axis=1),
        "p_red_sub": (rw & (mt == METHOD_SUB)).mean(axis=1),
        "p_blue_sub": (~rw & (mt == METHOD_SUB)).mean(axis=1),
        "exp_end_round": end_round.mean(axis=1),
        "p_round1": (end_round == 1).mean(axis=1),
        "p_distance": (mt == METHOD_DEC).mean(axis=1),
    })


def build_simulator_frame(matchup: pd.DataFrame) -> pd.DataFrame:
    """Derive the per-fight columns the hazard model needs from the matchup frame.

    Everything here must be pre-fight. `fight_minutes` and the outcome flags are used
    ONLY as fitting targets on training rows, never as predictors.
    """
    out = pd.DataFrame(index=matchup.index)
    out["fight_id"] = matchup.index.values

    carry = ["red_glicko_ko", "blue_glicko_ko", "red_glicko_kod", "blue_glicko_kod",
             "red_glicko_sub", "blue_glicko_sub", "red_glicko_subd", "blue_glicko_subd"]
    carry += [c for c in matchup.columns if c.startswith("diff_glicko_")]
    for col in carry:
        out[col] = matchup[col].to_numpy(float) if col in matchup else np.nan

    out["is_five_round"] = (
        matchup["fight_is_five_round"].to_numpy(float)
        if "fight_is_five_round" in matchup else 0.0
    )
    out["scheduled_rounds"] = (
        matchup["fight_scheduled_rounds"].to_numpy(float)
        if "fight_scheduled_rounds" in matchup else 3.0
    )

    # Pre-fight output/control differentials used to score simulated decisions
    for src, dest in (("diff_recent_sig_str_landed_per5", "diff_output_rate"),
                      ("diff_recent_ctrl_per5", "diff_control_rate")):
        out[dest] = matchup[src].to_numpy(float) if src in matchup else 0.0

    out["red_wins"] = matchup["red_wins"].to_numpy(float)
    return out


def attach_fit_targets(sim_frame: pd.DataFrame, matchup: pd.DataFrame) -> pd.DataFrame:
    """Add realised outcome columns. Used ONLY for fitting on training rows and for
    scoring on test rows — never as predictors."""
    out = sim_frame.copy()

    out["fight_minutes"] = np.clip(
        matchup["outcome_fight_minutes"].to_numpy(float), 0.5, None
    )
    method = matchup["outcome_method_class"].to_numpy(int)   # 0 KO, 1 SUB, 2 DEC
    red_won = matchup["red_wins"].to_numpy(float) == 1

    out["red_ko_win"] = ((method == METHOD_KO) & red_won).astype(float)
    out["blue_ko_win"] = ((method == METHOD_KO) & ~red_won).astype(float)
    out["red_sub_win"] = ((method == METHOD_SUB) & red_won).astype(float)
    out["blue_sub_win"] = ((method == METHOD_SUB) & ~red_won).astype(float)
    out["went_to_decision"] = (method == METHOD_DEC).astype(float)
    out["method_class"] = method
    return out
