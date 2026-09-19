"""Textbook Glicko-1, with MMA parameters taken from Fight Matrix.

This is NOT `glicko_service.py`. That module is a 15-dimension, round-level engine whose
output is the radar profile and the ML feature snapshots; it happens to use the Glicko
*update rule* but it is not a rating system in the ordinary sense. This module is a plain
single-dimension Glicko-1 whose only job is to answer "how good is this fighter, and how
sure are we" so the ranking can weight opponents by it.

Why Glicko rather than the Elo backbone it replaces
---------------------------------------------------
The Elo backbone carried `ELO_K_NEWCOMER = 60` for a fighter's first 5 bouts and
`ELO_K_BASE = 40` after. That is a hand-rolled approximation of rating *uncertainty*:
move fast while we know little, slow once we know more. Glicko models that quantity
directly as RD, so the two constants and the 5-fight cutoff all disappear, and RD is then
available to the ranker for free — which is what lets it discount a win over an unproven
opponent without inventing a separate penalty for it.

Parameters are Fight Matrix's published MMA-tuned values (fightmatrix.com/faq), which is
the only public parameterisation of Glicko fitted specifically to this sport. Nothing here
is chosen by us:

    r0 = 1500, RD0 = RDmax = 230      Fight Matrix
    RD floor = 30                     Glickman, "The Glicko system", sec. on RD floors
    c = 87, 180-day grace             Fight Matrix
    q = ln(10)/400                    definitional
    rating period = one fight         Fight Matrix

Glickman's own caveat applies and is worth stating: Glicko assumes a moderate number of
games per rating period, and MMA gives roughly two fights a year. That is the reason the
ranking does not read ratings directly off this module — it buckets them into deciles
(see `tiered_ranking_service`), which is robust to exactly the per-fighter noise this
sparsity produces.
"""

from __future__ import annotations

import math

#: Glicko's logistic scale constant. ln(10)/400 — definitional, not tunable.
Q = math.log(10.0) / 400.0

#: Fight Matrix's published MMA parameterisation.
RATING_START = 1500.0
RD_START = 230.0
RD_MAX = 230.0

#: Glickman recommends a floor "so that ratings can change appreciably even in a
#: relatively short time" — without it a busy fighter's RD collapses and their rating
#: freezes.
RD_FLOOR = 30.0

#: Inactivity: RD is untouched for 180 days, then grows by c per sqrt(day) up to RD_MAX.
#: Note this inflates *uncertainty only* — Glicko never moves the rating itself for
#: inactivity, which is the point of the "decay confidence, not skill" philosophy.
#: The ranking applies its own calendar decay to points; the two are separate mechanisms.
INACTIVITY_GRACE_DAYS = 180
C = 87.0


def g(rd: float) -> float:
    """Attenuation of a result by the *opponent's* uncertainty.

    A win over someone whose rating we barely know moves us less, because it is weaker
    evidence. This is the mechanism that makes an explicit newcomer K-factor unnecessary.
    """
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd * rd / (math.pi * math.pi))


def expected(rating: float, opp_rating: float, opp_rd: float) -> float:
    """Probability `rating` beats `opp_rating`, discounted by the opponent's RD."""
    return 1.0 / (1.0 + 10.0 ** (-g(opp_rd) * (rating - opp_rating) / 400.0))


def inflate_rd(rd: float, days_idle: int) -> float:
    """RD growth over a lay-off. Flat inside the grace window, then sqrt-time growth."""
    if days_idle <= INACTIVITY_GRACE_DAYS:
        return min(rd, RD_MAX)
    excess = days_idle - INACTIVITY_GRACE_DAYS
    return min(math.sqrt(rd * rd + C * C * excess), RD_MAX)


def update(rating: float, rd: float, opp_rating: float, opp_rd: float,
           score: float, weight: float = 1.0) -> tuple[float, float]:
    """One Glicko-1 rating period containing exactly one bout.

    `score` is the outcome value, NOT a bare 1/0. Fight Matrix and BoxRec both feed method
    of victory in exactly here — a unanimous-decision win enters as 0.91, a split as 0.55,
    a finish as 1.00 — rather than as a separate bonus applied afterwards. That is the
    published design and it is the right one: a narrow decision genuinely is weaker
    evidence of superiority than a knockout, so it should move the rating less, and
    folding it in here means it cannot be double-counted downstream.

    `weight` scales how much the bout is allowed to move the rating, which is where
    BoxRec's rounds-scheduled weighting lands: a three-round decision carries (3/5)^2 of a
    championship-distance one. It attenuates the rating move without touching RD's
    shrinkage, so a lightly-weighted bout still counts as evidence seen.
    """
    gj = g(opp_rd)
    e = 1.0 / (1.0 + 10.0 ** (-gj * (rating - opp_rating) / 400.0))

    # d^2 is the variance of the rating estimate from this bout alone. E(1-E) -> 0 for a
    # foregone conclusion, so a heavy favourite winning tells us almost nothing and moves
    # them almost not at all.
    denom = Q * Q * gj * gj * e * (1.0 - e)
    if denom <= 0.0:
        return rating, max(rd, RD_FLOOR)
    d2 = 1.0 / denom

    inv = 1.0 / (rd * rd) + 1.0 / d2
    new_rating = rating + weight * (Q / inv) * gj * (score - e)
    new_rd = max(math.sqrt(1.0 / inv), RD_FLOOR)
    return new_rating, new_rd


class Glicko1:
    """Incremental Glicko-1 state over a stream of bouts in chronological order.

    Deliberately online. The ranking measures opponent strength *as of the bout*, so the
    pre-fight pair returned by `observe` is the product, not a by-product — it is what the
    incumbent computed and then threw away (`points_ranking_service.py:288` stores
    `opp_elo` and `:308` reads the final Elo instead, so a fighter who later declines
    retroactively devalues the wins scored over them).

    Being online also means the eval harness can drive this with the same `observe` calls
    production uses, so there is no second implementation to drift.
    """

    #: BoxRec's published recency rule: a fighter's current rating is capped by their best
    #: performance inside an 18-month window, with earlier performances degraded by x0.5
    #: per further 18 months. It is what stops a rating from being a career museum piece —
    #: Glicko on its own inflates RD when you are idle but never lowers the rating, so a
    #: long-retired champion keeps their peak forever. The UFC's own Meta model decays from
    #: 18 months too, which is the same number arrived at independently.
    PEAK_WINDOW_DAYS = 548

    def __init__(self) -> None:
        self.rating: dict[int, float] = {}
        self.rd: dict[int, float] = {}
        self.last_seen: dict[int, object] = {}
        self.fights: dict[int, int] = {}
        #: (date, rating) after each bout, for the BoxRec peak-in-window readout.
        self.history: dict[int, list[tuple[object, float]]] = {}

    def decayed(self, fid: int, as_of) -> float:
        """Rating with Meta's published inactivity decay: flat for 18 months, then the
        above-baseline part halves per further 18 months.

        Glicko alone never lowers a rating for inactivity — it only grows RD — so a
        long-idle former champion keeps their peak indefinitely. Every published RANKING
        (as opposed to rating) decays: Meta from 18 months, BoxRec x0.5 per 18 months,
        Fight Matrix with an accelerating penalty. This is the ranking-side half.
        """
        r = self.rating.get(fid, RATING_START)
        last = self.last_seen.get(fid)
        if last is None:
            return r
        idle = (as_of - last).days
        if idle <= self.PEAK_WINDOW_DAYS:
            return r
        periods = (idle - self.PEAK_WINDOW_DAYS) / self.PEAK_WINDOW_DAYS
        return RATING_START + (r - RATING_START) * (0.5 ** periods)

    def peak(self, fid: int, as_of) -> float:
        """Best rating held inside the recency window, older peaks halved per window.

        Returns the plain current rating when there is no history to work from.
        """
        hist = self.history.get(fid)
        if not hist:
            return self.rating.get(fid, RATING_START)
        best = None
        for when, r in hist:
            age = (as_of - when).days
            if age <= self.PEAK_WINDOW_DAYS:
                decayed = r
            else:
                # Degrade the ABOVE-BASELINE part, so decay pulls toward the 1500 mean
                # rather than toward zero.
                periods = (age - self.PEAK_WINDOW_DAYS) / self.PEAK_WINDOW_DAYS
                decayed = RATING_START + (r - RATING_START) * (0.5 ** periods)
            best = decayed if best is None else max(best, decayed)
        return best

    def state(self, fid: int, as_of=None) -> tuple[float, float]:
        """Current (rating, RD), with inactivity inflation applied up to `as_of`."""
        r = self.rating.get(fid, RATING_START)
        rd = self.rd.get(fid, RD_START)
        last = self.last_seen.get(fid)
        if as_of is not None and last is not None:
            rd = inflate_rd(rd, (as_of - last).days)
        return r, rd

    def conservative(self, fid: int, as_of=None) -> float:
        """Rating discounted by uncertainty — TrueSkill's published `mu - k*sigma`.

        This is what the ranking tiers on. Two fighters can share a rating while one has
        eight bouts of evidence behind it and the other has three; beating the proven one
        is the better win, and `rating - RD` is how that difference gets expressed without
        a bespoke "experience" term.
        """
        r, rd = self.state(fid, as_of)
        return r - rd

    def observe(self, red_id: int, blue_id: int, red_score: float,
                fight_date=None, weight: float = 1.0) -> dict[int, tuple[float, float]]:
        """Apply one decided bout. Returns each corner's PRE-fight (rating, RD).

        Callers must filter out no-contests, DQs and overturned results before calling —
        `fighter_registry.is_decided` is the predicate. Scoring an NC as a draw would
        shrink both RDs and drag both ratings together on zero information.
        """
        r_red, rd_red = self.state(red_id, fight_date)
        r_blue, rd_blue = self.state(blue_id, fight_date)

        # Both updates read the pre-fight state of the other corner. Sequencing them
        # would make the result depend on which fighter is arbitrarily "red".
        new_red = update(r_red, rd_red, r_blue, rd_blue, red_score, weight)
        new_blue = update(r_blue, rd_blue, r_red, rd_red, 1.0 - red_score, weight)

        self.rating[red_id], self.rd[red_id] = new_red
        self.rating[blue_id], self.rd[blue_id] = new_blue
        if fight_date is not None:
            self.last_seen[red_id] = fight_date
            self.last_seen[blue_id] = fight_date
        self.fights[red_id] = self.fights.get(red_id, 0) + 1
        self.fights[blue_id] = self.fights.get(blue_id, 0) + 1
        if fight_date is not None:
            for fid in (red_id, blue_id):
                self.history.setdefault(fid, []).append((fight_date, self.rating[fid]))

        return {red_id: (r_red, rd_red), blue_id: (r_blue, rd_blue)}
