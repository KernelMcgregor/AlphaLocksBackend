"""Fight duration as a survival curve: discrete-time competing-risks hazard.

Time is cut into 2.5-minute bins (half-rounds), so every betting line falls on a bin edge:
    5:00 = end of R1 ("starts round 2")   7:30 = O/U 1.5   10:00 = end of R2
    12:30 = O/U 2.5   15:00 = end of a 3-round fight   17:30 / 22:30 = O/U 3.5 / 4.5

Person-period rows: one row per fight per bin the fight was still going at the start of.
Outcome of a row: 0 = still going at the end of the bin, 1 = KO/TKO in this bin,
2 = submission in this bin. A decision is a fight that survives every bin of its scheduled
length; the scheduled end is known in advance, so it is a non-informative stop, not an
event (P(decision) = S(end)).

    hazard     h_k = P(KO or Sub in bin k | still going at its start)
    survival   S(edge k) = prod_{j<k} (1 - hKO_j - hSub_j)
    P(over line) = S(line)      P(ends in round r) = S(start of r) - S(end of r)

Features per row: the fight's features (corner view) + bin index, round, which half of
the round, 5-round flag. Corner symmetry: rows are built from both corner views and the
predicted curves are averaged over the two views.

Finish-time convention: a finish at exactly 7:30 is UNDER 1.5 (the fight did not pass
7:30), so a finish at time T falls in bin ceil(T / 2.5) - 1; a stoppage at 5:00 of R1
belongs to R1.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

BIN = 2.5  # default slice, minutes; any divisor of 2.5 keeps every line on an edge (1.25, 0.5)
LINES = {"sr_2": 5.0, "ou_1.5": 7.5, "sr_3": 10.0, "ou_2.5": 12.5,
         "sr_4": 15.0, "ou_3.5": 17.5, "sr_5": 20.0, "ou_4.5": 22.5}
TIME_COLS = ("hz_bin", "hz_round", "hz_pos_in_round", "hz_round_start", "hz_five",
             "hz_bin_x_five", "hz_minutes")
ROUND_MIN = 5.0


def finish_bin(t_minutes: float, bin_size: float = BIN) -> int:
    """Bin index a finish at t falls in (a finish exactly on an edge belongs to the bin
    that ends there)."""
    return max(int(math.ceil(round(t_minutes / bin_size, 6))) - 1, 0)


def n_bins(scheduled_minutes: float, bin_size: float = BIN) -> int:
    return int(round(scheduled_minutes / bin_size))


def person_period(X: pd.DataFrame, t: np.ndarray, event: np.ndarray, sched: np.ndarray,
                  bin_size: float = BIN, weights: np.ndarray | None = None):
    """X: one row per fight (a corner view). event: 0 none (decision), 1 KO, 2 Sub.
    Returns (rows with features + time cols, labels, fight row index, row weights)."""
    reps, bins, labels = [], [], []
    for i in range(len(X)):
        K = n_bins(sched[i], bin_size)
        last = finish_bin(t[i], bin_size) if event[i] else K - 1
        last = min(last, K - 1)
        for k in range(last + 1):
            reps.append(i); bins.append(k)
            labels.append(event[i] if (event[i] and k == last) else 0)
    reps = np.array(reps); bins = np.array(bins)
    rows = X.iloc[reps].reset_index(drop=True)
    w = None if weights is None else np.asarray(weights, float)[reps]
    return _add_time(rows, bins, sched[reps], bin_size), np.array(labels), reps, w


def _add_time(rows: pd.DataFrame, bins: np.ndarray, sched: np.ndarray,
              bin_size: float = BIN) -> pd.DataFrame:
    """Clock features. The finish rate is a sawtooth: lowest just after the one-minute
    rest, rising ~2-3x by the bell (checked on 2015+ fights), so position within the
    round and a round-start flag are given explicitly."""
    five = (np.asarray(sched) > 15).astype(float)
    start_min = bins * bin_size
    rnd = np.floor(start_min / ROUND_MIN + 1e-9)
    pos = (start_min - rnd * ROUND_MIN) / ROUND_MIN
    return rows.assign(hz_bin=bins, hz_round=rnd + 1, hz_pos_in_round=pos,
                       hz_round_start=(pos < 1e-9).astype(float), hz_five=five,
                       hz_bin_x_five=bins * five, hz_minutes=start_min)


def all_bins(X: pd.DataFrame, sched: np.ndarray, bin_size: float = BIN):
    """Every bin of every fight (for prediction)."""
    reps, bins = [], []
    for i in range(len(X)):
        for k in range(n_bins(sched[i], bin_size)):
            reps.append(i); bins.append(k)
    reps = np.array(reps); bins = np.array(bins)
    rows = X.iloc[reps].reset_index(drop=True)
    return _add_time(rows, bins, sched[reps], bin_size), reps, bins


CATBOOST_DEFAULT = dict(learning_rate=0.05, depth=5, l2_leaf_reg=10)


class HazardModel:
    """Multinomial hazard {continue, KO, Sub} on person-period rows.

    bin_size      slice length in minutes (2.5, 1.25, ...)
    interactions  (logit only) features multiplied by round and position-in-round, so an
                  effect can grow or fade through the fight (power early, cardio late);
                  CatBoost finds such interactions on its own from the clock columns
    params        CatBoost settings
    """

    def __init__(self, features: list[str], backend: str = "logit", bin_size: float = BIN,
                 interactions: list[str] | None = None, params: dict | None = None):
        self.base_features = list(features)
        self.features = list(features) + list(TIME_COLS)
        self.backend = backend
        self.bin_size = bin_size
        self.interactions = [f for f in (interactions or []) if f in features]
        self.params = {**CATBOOST_DEFAULT, **(params or {})}
        self.model = None
        self.med = None

    def _design(self, rows: pd.DataFrame) -> np.ndarray:
        X = rows.reindex(columns=self.features).to_numpy(float)
        if self.backend != "logit":
            return X
        if self.med is None:
            med = np.nanmedian(X, axis=0)
            self.med = np.where(np.isnan(med), 0.0, med)
        X = np.where(np.isnan(X), self.med, X)
        parts = [X]
        if self.interactions:
            idx = [self.features.index(f) for f in self.interactions]
            rnd = rows["hz_round"].to_numpy(float)[:, None]
            pos = rows["hz_pos_in_round"].to_numpy(float)[:, None]
            parts += [X[:, idx] * rnd, X[:, idx] * pos]
        b = rows["hz_bin"].to_numpy(int)
        nb = n_bins(25.0, self.bin_size)
        D = np.zeros((len(b), nb))
        D[np.arange(len(b)), np.clip(b, 0, nb - 1)] = 1.0  # free-form baseline hazard
        parts.append(D)
        return np.hstack(parts)

    def fit(self, views: list[pd.DataFrame], t, event, sched, weights=None) -> "HazardModel":
        parts = [person_period(v, t, event, sched, self.bin_size, weights) for v in views]
        rows = pd.concat([p[0] for p in parts], ignore_index=True)
        y = np.concatenate([p[1] for p in parts])
        w = None if weights is None else np.concatenate([p[3] for p in parts])
        if self.backend == "logit":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.med = None
            X = self._design(rows)
            self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=4000))
            self.model.fit(X, y, logisticregression__sample_weight=w)
        elif self.backend == "catboost":
            from catboost import CatBoostClassifier
            X = self._design(rows)
            cut = int(len(X) * 0.85)
            p = dict(**self.params, loss_function="MultiClass", random_seed=42, verbose=False,
                     allow_writing_files=False)
            probe = CatBoostClassifier(iterations=2000, od_type="Iter", od_wait=100, **p)
            probe.fit(X[:cut], y[:cut], sample_weight=None if w is None else w[:cut],
                      eval_set=(X[cut:], y[cut:]), use_best_model=True)
            self.best_iter = max(probe.get_best_iteration() or 100, 50)
            self.model = CatBoostClassifier(iterations=self.best_iter, **p)
            self.model.fit(X, y, sample_weight=w)
        else:
            raise ValueError(self.backend)
        return self

    def hazards(self, rows: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(self._design(rows))
        return p[:, 1:3] if p.shape[1] == 3 else np.column_stack([p[:, 1], np.zeros(len(p))])

    def survival(self, views: list[pd.DataFrame], sched: np.ndarray) -> np.ndarray:
        """(n, 11) S at the 2.5-minute edges 0 .. 25, averaged over corner views. Beyond a
        fight's scheduled end S stays at its final value (= P(decision))."""
        step = int(round(BIN / self.bin_size))
        curves = []
        for v in views:
            rows, reps, bins = all_bins(v, sched, self.bin_size)
            h = self.hazards(rows).sum(axis=1)
            S = np.ones((len(v), 11))
            for i in range(len(v)):
                fine = np.concatenate([[1.0], np.cumprod(1 - np.clip(h[reps == i], 0, 0.999))])
                coarse = fine[::step]
                S[i, :len(coarse)] = coarse
                S[i, len(coarse):] = coarse[-1]
            curves.append(S)
        return np.mean(curves, axis=0)


def cause_curves(model: "HazardModel", views: list[pd.DataFrame], sched: np.ndarray):
    """Fine-resolution curves at every bin edge (0, bin, 2*bin, ... 25 min), averaged over
    corner views: S (still going), KO (cumulative KO/TKO incidence), SUB (cumulative
    submission incidence), H (total hazard per bin). Beyond a fight's scheduled end the
    curves stay flat. Shapes: (n, 25/bin + 1) for S/KO/SUB, (n, 25/bin) for H."""
    nb = n_bins(25.0, model.bin_size)
    acc = {k: [] for k in ("S", "KO", "SUB", "H")}
    for v in views:
        rows, reps, bins = all_bins(v, sched, model.bin_size)
        hz = model.hazards(rows)
        S = np.ones((len(v), nb + 1)); KO = np.zeros_like(S); SUB = np.zeros_like(S)
        H = np.zeros((len(v), nb))
        for i in range(len(v)):
            h = hz[reps == i]
            hk, hs = np.clip(h[:, 0], 0, 0.999), np.clip(h[:, 1], 0, 0.999)
            surv = np.concatenate([[1.0], np.cumprod(1 - np.clip(hk + hs, 0, 0.999))])
            k = len(hk)
            S[i, :k + 1] = surv; S[i, k + 1:] = surv[-1]
            KO[i, 1:k + 1] = np.cumsum(surv[:-1] * hk); KO[i, k + 1:] = KO[i, k]
            SUB[i, 1:k + 1] = np.cumsum(surv[:-1] * hs); SUB[i, k + 1:] = SUB[i, k]
            H[i, :k] = hk + hs
        for name, arr in zip(("S", "KO", "SUB", "H"), (S, KO, SUB, H)):
            acc[name].append(arr)
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def anchor(S: np.ndarray, p_decision: np.ndarray, sched: np.ndarray) -> np.ndarray:
    """Rescale a curve's finish mass so S(end) equals an external P(decision) (method_v2),
    keeping the curve's timing shape: S'(t) = 1 - (1 - S(t)) * (1 - p_dec) / (1 - S(end))."""
    end = np.where(np.asarray(sched) <= 15, 6, 10)
    s_end = S[np.arange(len(S)), end]
    p = np.where(np.isfinite(p_decision), p_decision, s_end)
    scale = (1 - p) / np.clip(1 - s_end, 1e-4, None)
    A = 1 - (1 - S) * scale[:, None]
    return np.minimum.accumulate(np.clip(A, 1e-4, 1.0), axis=1)


def over_prob(S: np.ndarray, line_minutes: float) -> np.ndarray:
    """P(fight lasts past line) from edge curves (line must be a multiple of 2.5)."""
    return S[:, int(round(line_minutes / BIN))]


def round_probs(S: np.ndarray, sched: np.ndarray) -> np.ndarray:
    """(n, 6): P(ends in R1..R5), P(decision). R4/R5 are 0 for 3-round fights."""
    out = np.zeros((len(S), 6))
    for r in range(5):
        out[:, r] = S[:, 2 * r] - S[:, 2 * r + 2]
    three = np.asarray(sched) <= 15
    out[three, 3:5] = 0.0
    out[:, 5] = np.where(three, S[:, 6], S[:, 10])
    return out
