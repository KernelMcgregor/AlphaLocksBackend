"""Deserve to win: how often each fighter wins on the cards if the fight is replayed.

MoneyPuck's Deserve to Win O'Meter replays a hockey game shot by shot, each shot a
Bernoulli draw with its expected-goals probability. The MMA version replays a fight
round by round:

  round stats -> P(red wins the round) and P(10-8 | winner)       (round model)
             -> three judges draw verdicts, correlated within a round (shared latent)
             -> cards totalled, referee deductions applied, official rules
             -> P(red), P(draw), P(blue) over N simulations

"Average judges" (no judge-specific lean) is the headline, like MoneyPuck's average
goaltending; passing per-judge logit shifts gives the actual panel's version.

Finished fights are replayed to the scheduled distance: completed rounds are scored
from their stats; rounds never fought are drawn from a per-fight "form" distribution
(a normal-normal posterior over the fight's round logits, prior = the pre-fight edge).

Parameter uncertainty (Holmes, McHale & Zychaluk, EJOR 2023): pass several round-model
fits (bootstrap by fight); each simulated fight uses ONE fit for all its rounds.

This module is the pure simulation; fitting and DB I/O live in scripts.deserve_to_win.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Bout outcomes (red's perspective).
RED, DRAW, BLUE = 1, 0, -1


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def latent_scale(sigma_u: float) -> float:
    """Factor that keeps a judge's MARGINAL P(red) equal to the model's p when a shared
    round latent u ~ N(0, sigma_u^2) is added: E[sigmoid(a + u)] ~= sigmoid(a / k) with
    k = sqrt(1 + pi * sigma_u^2 / 8) (logistic-normal approximation), so use a = z * k."""
    return float(np.sqrt(1.0 + np.pi * sigma_u ** 2 / 8.0))


@dataclass
class TenEight:
    """P(the round's winner gets a 10-8) = sigmoid(a + b * z_w), z_w = the round logit
    signed toward that winner (a fluke round won by the dominated fighter is rarely 10-8).
    Used for rounds without their own hurdle-model probability (extrapolated rounds)."""
    a: float = -6.0
    b: float = 0.8

    def __call__(self, z_toward_winner):
        return sigmoid(self.a + self.b * np.asarray(z_toward_winner, float))


@dataclass
class Form:
    """Distribution of an unfought round's logit for one fight.

    fight mean m ~ N(mean, sd_mean^2); each unfought round's logit ~ N(m, sd_round^2).
    """
    mean: float
    sd_mean: float
    sd_round: float


def form_posterior(observed: np.ndarray, prior_mean: float, tau: float, sigma: float) -> Form:
    """Normal-normal update of a fight's mean round logit.

    tau   = SD of fight means around the pre-fight prior (between fights)
    sigma = SD of round logits around their fight's mean (within a fight)
    """
    z = np.asarray(observed, float)
    n = len(z)
    if n == 0:
        return Form(prior_mean, tau, sigma)
    prec = 1.0 / tau ** 2 + n / sigma ** 2
    mean = (prior_mean / tau ** 2 + z.sum() / sigma ** 2) / prec
    return Form(float(mean), float(np.sqrt(1.0 / prec)), sigma)


def decide(red_cards: np.ndarray, blue_cards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Official result from three judges' totals. Arrays (..., 3).

    Returns (outcome RED/DRAW/BLUE, kind) where kind is one of
    'ud', 'sd', 'md', 'draw' as small ints 0..3 (see KINDS).
    """
    r = (red_cards > blue_cards).sum(-1)
    b = (blue_cards > red_cards).sum(-1)
    e = 3 - r - b
    out = np.where(r >= 2, RED, np.where(b >= 2, BLUE, DRAW))
    w = np.maximum(r, b)
    kind = np.where(out == DRAW, 3,
                    np.where(w == 3, 0, np.where(e == 1, 2, 1)))  # 3-0 ud, 2-0-1 md, 2-1 sd
    return out, kind


KINDS = ("ud", "sd", "md", "draw")


@dataclass
class FightInput:
    """One fight's rounds for simulation.

    round_logits   (B, R_obs) logit P(red wins round) per model fit (B >= 1), for the
                   rounds that were (at least partly) fought
    ten8           (B, R_obs) P(10-8 | red wins the round), or None -> TenEight(z)
    ten8_blue      (B, R_obs) P(10-8 | blue wins the round); defaults to ten8
    partial_weight fraction of the LAST observed round actually fought (a finish); its
                   logit is blended with a form draw: w * z_obs + (1 - w) * z_form
    scheduled      rounds scheduled (3 or 5); rounds R_obs+1..scheduled are extrapolated
    form           Form for extrapolated rounds (required when R_obs < scheduled)
    red_ded, blue_ded  (scheduled,) referee deductions per round (points)
    judge_shift    (3, scheduled) logit shift per judge per round (actual panel), or None
    """
    round_logits: np.ndarray
    scheduled: int
    ten8: np.ndarray | None = None
    ten8_blue: np.ndarray | None = None
    partial_weight: float | None = None
    form: Form | None = None
    red_ded: np.ndarray | None = None
    blue_ded: np.ndarray | None = None
    judge_shift: np.ndarray | None = None


@dataclass
class SimResult:
    p_red: float
    p_draw: float
    p_blue: float
    p_kind: dict[str, float]
    round_p_red: list[float]
    top_cards: list[dict] = field(default_factory=list)
    n: int = 0


def simulate_fight(fi: FightInput, n: int = 10_000, sigma_u: float = 1.0,
                   ten8_fallback: TenEight | None = None, rng=None,
                   top_k: int = 5, sigma_v: float = 0.0) -> SimResult:
    """Replay one fight n times to the scheduled distance with three judges.

    sigma_u: SD of the round latent all three judges share (split-round rate).
    sigma_v: SD of a fight latent shared by every round and judge: what the box score
             misses (damage, "who looked better") that tilts the whole fight.
    """
    rng = np.random.default_rng(rng)
    ten8_fallback = ten8_fallback or TenEight()
    L = np.atleast_2d(np.asarray(fi.round_logits, float))
    B, R_obs = L.shape
    R = int(fi.scheduled)
    if R_obs > R:
        raise ValueError(f"{R_obs} rounds observed but only {R} scheduled")
    partial = fi.partial_weight is not None and fi.partial_weight < 1
    if (R_obs < R or partial) and fi.form is None:
        raise ValueError("form is required to extrapolate unfought rounds")
    k = latent_scale(float(np.hypot(sigma_u, sigma_v)))

    # Round logits per simulation: (n, R)
    b = rng.integers(0, B, size=n)
    z = np.empty((n, R))
    z[:, :R_obs] = L[b]
    if fi.form is not None and (R_obs < R or partial):
        f = fi.form
        m = rng.normal(f.mean, f.sd_mean, size=(n, 1))
        if R_obs < R:
            z[:, R_obs:] = m + rng.normal(0.0, f.sd_round, size=(n, R - R_obs))
        if partial:
            w = float(fi.partial_weight)
            z_rest = m[:, 0] + rng.normal(0.0, f.sd_round, size=n)
            z[:, R_obs - 1] = w * z[:, R_obs - 1] + (1 - w) * z_rest
    # P(10-8 | each side wins the round): model values for fought rounds, else fallback.
    q_red, q_blue = ten8_fallback(z), ten8_fallback(-z)
    if fi.ten8 is not None:
        q_red[:, :R_obs] = np.atleast_2d(np.asarray(fi.ten8, float))[b]
        q_blue[:, :R_obs] = np.atleast_2d(np.asarray(
            fi.ten8 if fi.ten8_blue is None else fi.ten8_blue, float))[b]

    # Judge verdicts: red iff k*z + shift + u_r + e_rj > 0, e_rj ~ Logistic(0, 1)
    a = (k * z)[:, None, :]                                       # (n, 1, R)
    if fi.judge_shift is not None:
        a = a + np.asarray(fi.judge_shift, float)[None, :, :]     # (n, 3, R)
    u = rng.normal(0.0, sigma_u, size=(n, 1, R)) + rng.normal(0.0, sigma_v, size=(n, 1, 1))
    e = rng.logistic(size=(n, 3, R))
    red_round = (a + u + e) > 0                                   # (n, 3, R)

    # 10-8: each judge independently, given their own verdict
    eight = rng.random((n, 3, R)) < np.where(red_round, q_red[:, None, :], q_blue[:, None, :])
    loser_pts = np.where(eight, 8, 9)
    red_pts = np.where(red_round, 10, loser_pts)
    blue_pts = np.where(red_round, loser_pts, 10)
    rd = np.zeros(R) if fi.red_ded is None else np.asarray(fi.red_ded, float)[:R]
    bd = np.zeros(R) if fi.blue_ded is None else np.asarray(fi.blue_ded, float)[:R]
    red_tot = red_pts.sum(-1) - rd.sum()                          # (n, 3)
    blue_tot = blue_pts.sum(-1) - bd.sum()

    out, kind = decide(red_tot, blue_tot)
    p_kind = {name: float((kind == i).mean()) for i, name in enumerate(KINDS)}

    # Most likely card sets, judges sorted so order doesn't matter.
    max_pts = 10 * R
    margin = (red_tot - blue_tot).astype(int)
    lo = (np.minimum(red_tot, blue_tot)).astype(int)
    # Encode each card as (margin, loser points) -> unique int, then sort the 3 cards.
    code = (margin + 4 * R) * (max_pts + 1) + lo
    code.sort(axis=1)
    base = (8 * R + 1) * (max_pts + 1)
    key = (code[:, 0] * base + code[:, 1]) * base + code[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    top = []
    for i in np.argsort(-counts)[:top_k]:
        sel = np.flatnonzero(key == vals[i])[0]
        cards = [f"{int(r)}-{int(bl)}" for r, bl in
                 sorted(zip(red_tot[sel], blue_tot[sel]), key=lambda c: c[1] - c[0])]
        top.append({"cards": cards, "p": float(counts[i] / n),
                    "result": {RED: "red", DRAW: "draw", BLUE: "blue"}[int(out[sel])]})

    return SimResult(
        p_red=float((out == RED).mean()), p_draw=float((out == DRAW).mean()),
        p_blue=float((out == BLUE).mean()), p_kind=p_kind,
        round_p_red=[float(x) for x in red_round.mean(axis=(0, 1))],
        top_cards=top, n=n,
    )


def judge_agreement(p: np.ndarray, sigma_u: float, n: int = 200, rng=None,
                    sigma_v: float = 0.0) -> np.ndarray:
    """Simulated P(all three judges agree) for rounds with model probabilities p.
    Used to fit sigma_u against the empirical 3-0 / 2-1 split of verified cards.
    (Within one round the fight latent acts like extra round latent.)"""
    rng = np.random.default_rng(rng)
    s = float(np.hypot(sigma_u, sigma_v))
    a = latent_scale(s) * logit(p)[:, None, None]
    u = rng.normal(0.0, s, size=(len(p), n, 1))
    e = rng.logistic(size=(len(p), n, 3))
    red = (a + u + e) > 0
    s = red.sum(-1)
    return ((s == 0) | (s == 3)).mean(-1)


def fit_sigma_u(p: np.ndarray, unanimous: np.ndarray,
                grid=np.round(np.arange(0.0, 3.01, 0.1), 2), rng=0) -> tuple[float, dict]:
    """Choose sigma_u by maximum likelihood of each round's judges being unanimous
    (Bernoulli), given the round model's p. Returns (best sigma_u, {sigma: loglik})."""
    y = np.asarray(unanimous, float)
    ll = {}
    for s in grid:
        a = np.clip(judge_agreement(p, float(s), rng=rng), 1e-4, 1 - 1e-4)
        ll[float(s)] = float(np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)))
    return max(ll, key=ll.get), ll
