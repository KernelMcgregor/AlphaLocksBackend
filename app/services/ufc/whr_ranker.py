"""Whole-History Rating (Coulom 2008) and a Bradley-Terry reference.

Why this rather than more Points tuning: ranking_eval measured every constant change to
the Points system and none was statistically distinguishable from the shipped version
(paired 95% CIs all straddling zero on 1440 walk-forward bouts). The limitation is
structural, not parametric. Points is a *results-accumulation* system — it adds up what a
fighter did — where what the rankings need is a *latent-skill estimate*.

WHR is the latent-skill version:

  * Every fighter has a rating that varies over time, with a Wiener-process prior tying
    consecutive appearances together (variance `w2` per day). Ratings move when results
    demand it and drift toward uncertainty when a fighter is idle.
  * The whole history is re-fit at once, so a 2019 rating updates when 2026 reveals that
    the man who beat him became a champion. Elo and Points structurally cannot do this —
    they are single-pass and can only ever use the past.
  * It returns a posterior variance, so a 2-fight prospect is explicitly uncertain rather
    than merely un-scored. Ranking on `mu - k*sigma` puts unproven fighters where they
    belong by construction rather than via a tuned penalty — which is the direct answer
    to a 13-1 prospect outranking Charles Oliveira.

Retrodiction is deliberate here and safe: this module is for the DISPLAY ranking only. It
never writes ufc_glicko_snapshots and never enters build_features. test_leakage asserts
that. The evaluation harness refits it per fold on past fights only, so the numbers it
reports are still honest out-of-sample.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date

from app.services.ufc.fighter_registry import is_decided

log = logging.getLogger("whr")

#: Rating scale. Elo-like: 400 points is a 10:1 odds ratio.
SCALE = 400.0 / math.log(10.0)

#: Wiener drift, rating-variance per day. ~2 elo^2/day lets a fighter move a few dozen
#: points across a year of inactivity without letting the prior wander freely.
DEFAULT_W2 = 2.0

#: Prior on a debutant: mean 0, this variance. Wide enough not to fight the data, tight
#: enough that one upset does not launch someone to the top of a division.
PRIOR_VAR = 220.0 ** 2

#: How many standard deviations to subtract for the displayed rating. 1.0 is TrueSkill's
#: conservative-estimate convention.
CONSERVATISM = 1.0


class WHRRanker:
    """Batch ranker: `fit(fights)` then `rate(fighter_id, as_of)`."""

    name = "whr"

    def __init__(self, w2: float = DEFAULT_W2, conservatism: float = CONSERVATISM,
                 iterations: int = 40):
        self.w2 = w2
        self.conservatism = conservatism
        self.iterations = iterations
        self.days: dict[int, list[date]] = {}
        self.mu: dict[int, list[float]] = {}
        self.var: dict[int, list[float]] = {}
        self._games: dict[int, list[list[tuple]]] = {}

    # ------------------------------------------------------------------ fitting
    def fit(self, fights: list) -> None:
        usable = [f for f in fights
                  if f.date and is_decided(f.method, f.winner_id)
                  and f.red_fighter_id and f.blue_fighter_id]

        appearances: dict[int, set] = defaultdict(set)
        for f in usable:
            appearances[f.red_fighter_id].add(f.date)
            appearances[f.blue_fighter_id].add(f.date)

        self.days = {fid: sorted(ds) for fid, ds in appearances.items()}
        index = {fid: {d: i for i, d in enumerate(ds)} for fid, ds in self.days.items()}
        self.mu = {fid: [0.0] * len(ds) for fid, ds in self.days.items()}
        self.var = {fid: [PRIOR_VAR] * len(ds) for fid, ds in self.days.items()}

        # games[fid][t] = [(opponent_id, opponent_time_index, score), ...]
        games: dict[int, list[list[tuple]]] = {
            fid: [[] for _ in ds] for fid, ds in self.days.items()}
        for f in usable:
            r, b = f.red_fighter_id, f.blue_fighter_id
            ri, bi = index[r][f.date], index[b][f.date]
            red_won = 1.0 if f.winner_id == r else 0.0
            games[r][ri].append((b, bi, red_won))
            games[b][bi].append((r, ri, 1.0 - red_won))
        self._games = games

        for _ in range(self.iterations):
            for fid in self.days:
                self._refine(fid)
        log.info(f"  WHR: {len(self.days)} fighters, "
                 f"{sum(len(v) for v in self.days.values())} rating points")

    def _refine(self, fid: int) -> None:
        """One Newton step on this fighter's whole time series.

        The Hessian is tridiagonal — each rating couples only to its own neighbours in
        time — so this is a linear-time solve, which is why refitting per fold is cheap.
        """
        ds, mus = self.days[fid], self.mu[fid]
        n = len(ds)
        if n == 0:
            return

        g = [0.0] * n           # gradient
        h = [0.0] * n           # diagonal of the Hessian
        off = [0.0] * n         # sub/super-diagonal (time coupling)

        for t in range(n):
            # Bradley-Terry likelihood against each opponent's current rating.
            for opp, opp_t, score in self._games[fid][t]:
                diff = (mus[t] - self.mu[opp][opp_t]) / SCALE
                p = 1.0 / (1.0 + math.exp(-diff)) if diff > -700 else 0.0
                g[t] += (score - p) / SCALE
                h[t] -= p * (1.0 - p) / (SCALE * SCALE)

            # Wiener prior linking consecutive appearances.
            for other in (t - 1, t + 1):
                if 0 <= other < n:
                    dt = abs((ds[t] - ds[other]).days) or 1
                    prec = 1.0 / (self.w2 * dt)
                    g[t] -= (mus[t] - mus[other]) * prec
                    h[t] -= prec
                    if other == t + 1:
                        off[t] = prec

            if n == 1 or True:
                # Weak anchor to the prior mean, so a fighter whose every bout is
                # against one opponent cannot drift arbitrarily far as a pair.
                g[t] -= mus[t] / PRIOR_VAR
                h[t] -= 1.0 / PRIOR_VAR

        # Thomas algorithm on the tridiagonal system H * delta = -g.
        c = [0.0] * n
        d = [0.0] * n
        beta = h[0] if h[0] != 0 else -1e-9
        c[0] = off[0] / beta
        d[0] = -g[0] / beta
        for i in range(1, n):
            beta = h[i] - off[i - 1] * c[i - 1]
            if beta == 0:
                beta = -1e-9
            c[i] = (off[i] / beta) if i < n - 1 else 0.0
            d[i] = (-g[i] - off[i - 1] * d[i - 1]) / beta

        delta = [0.0] * n
        delta[n - 1] = d[n - 1]
        for i in range(n - 2, -1, -1):
            delta[i] = d[i] - c[i] * delta[i + 1]

        for t in range(n):
            step = max(-120.0, min(120.0, delta[t]))   # keep Newton from overshooting
            mus[t] += step
            self.var[fid][t] = min(PRIOR_VAR, 1.0 / max(-h[t], 1.0 / PRIOR_VAR))

    # ------------------------------------------------------------------ rating
    def rate(self, fid: int, as_of: date) -> tuple[float, float] | None:
        ds = self.days.get(fid)
        if not ds:
            return None
        # Most recent appearance at or before as_of.
        lo, hi = 0, len(ds) - 1
        idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if ds[mid] <= as_of:
                idx, lo = mid, mid + 1
            else:
                hi = mid - 1
        if idx < 0:
            return None

        mu = self.mu[fid][idx]
        var = self.var[fid][idx]
        # Idleness widens the posterior, so a long layoff lowers the displayed rating
        # without a hand-tuned inactivity multiplier.
        var += self.w2 * max(0, (as_of - ds[idx]).days)
        sigma = math.sqrt(var)
        return (mu - self.conservatism * sigma, sigma)

    def rating_only(self, fid: int, as_of: date) -> tuple[float, float] | None:
        """`mu` without the conservatism penalty — for diagnostics."""
        r = self.rate(fid, as_of)
        if r is None:
            return None
        return (r[0] + self.conservatism * r[1], r[1])


class BradleyTerryRanker(WHRRanker):
    """WHR without time dynamics: one rating per fighter for their whole career.

    Included as a reference, not a candidate. The WHR-minus-BT gap is exactly what the
    Wiener prior buys, so if the gap is ~0 the time dynamics are not earning their
    complexity.
    """

    name = "bt"

    def __init__(self, **kw):
        # A near-zero drift collapses every appearance onto one rating.
        super().__init__(w2=1e-6, **{k: v for k, v in kw.items() if k != "w2"})
