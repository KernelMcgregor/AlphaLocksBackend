"""UFC Rankings — a reimplementation of the Tapology system.

This is the project's only ranker. Points+Elo, Glicko-tiered, WHR and Bradley-Terry all
existed here before and are gone; see docs/models/rankings.md.

What is copied and what is not
------------------------------
Tapology publishes their RULES in detail (tapology.com/faq_rankings) and explicitly keeps
the SCORING confidential — "the exact details of how the Tapology algorithm works are
proprietary." So this module splits in two, and the split is load-bearing:

  * Every disclosed rule is implemented exactly as stated, cited inline to the FAQ.
  * Every undisclosed number lives in `Weights` and is FITTED against Tapology's own
    published output by `tapology_fit.py`. None of them is chosen by taste.

The one piece of the algorithm Tapology describes in real detail is the **opponent tier**,
and it is described under Strength of Schedule rather than under the ranking itself. That
is still the best available window into how they measure opposition, because the FAQ says
SoS "is built from some of the same ideas used elsewhere in the rankings system." The
recursion they describe is reproduced exactly in `_resume` below, down to the fighter
counts their own example implies (6 opponents -> 36 -> 216 matches).

Why not a rating system
-----------------------
The previous attempt scored opposition by Glicko percentile and failed the eye test. That
is not a tuning problem. Tapology's opponent quality is a *record-based* recursion — how
much their opponents had been winning, against opponents who had themselves been winning —
and it is deliberately blind to rating. A fighter who beats six men on win streaks rates
highly here even if a rating system had never caught up to them. Substituting Glicko for
that changes what the system measures, which is why calibrating it could not rescue it.

Run: python -m app.services.ufc.tapology_rankings --preview
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, timedelta

from app.database import SessionLocal
from app.models.ufc import UFCEvent, UFCFight, UFCFighter
from app.services.ufc.fighter_registry import Eligibility, classify_weight_class

log = logging.getLogger("tapology_rankings")


# ---------------------------------------------------------------------------
# Disclosed rules — quoted from the Tapology FAQ
# ---------------------------------------------------------------------------

#: "Tapology's algorithm looks at the last 6 UFC matches for every fighter."
WINDOW = 6

#: "Active fighters on the UFC roster with at least 1 completed UFC bout in the last 21
#: months are included." Note ONE bout, not the 2-fight/5-round floor the old rankers
#: used — Tapology ranks the whole roster, not just the proven part of it.
ELIGIBILITY_MONTHS = 21
ELIGIBILITY_DAYS = int(ELIGIBILITY_MONTHS * 30.44)              # 639

#: "The fighter will remain displayed for 60 days (though listed as ineligible) in the
#: position they would have occupied if still eligible." They leave the NUMBERED list at
#: 21 months (see `score`); this 60 days only keeps them inside the registry's eligibility
#: bound so the publisher does not reject a fighter we still want to carry data for.
GRACE_DAYS = 60

#: "A fighter must have competed in a weight class within the last 24 months in order to
#: remain eligible for that specific weight class."
DIVISION_WINDOW_DAYS = int(24 * 30.44)                          # 730

#: "The Tapology UFC Rankings encompass 11 divisions: 8 men's weight classes and 3
#: women's." No P4P: "At this time, Tapology does not have official Pound for Pound UFC
#: rankings using this scoring algorithm." Ours had P4P purely by inheritance.
DIVISIONS = [
    "flyweight", "bantamweight", "featherweight", "lightweight",
    "welterweight", "middleweight", "light_heavyweight", "heavyweight",
    "w_strawweight", "w_flyweight", "w_bantamweight",
]

WEIGHT_CLASS_LABELS = {
    "flyweight": "Flyweight", "bantamweight": "Bantamweight",
    "featherweight": "Featherweight", "lightweight": "Lightweight",
    "welterweight": "Welterweight", "middleweight": "Middleweight",
    "light_heavyweight": "Light Heavyweight", "heavyweight": "Heavyweight",
    "w_strawweight": "Strawweight", "w_flyweight": "Flyweight",
    "w_bantamweight": "Bantamweight",
}

#: "Each of the fighter's last 6 UFC opponents is placed into a tier from 1 to 10."
N_TIERS = 10

#: "Fighters removed from the UFC roster are typically set as ineligible", and a fighter
#: who competes elsewhere "is automatically removed from eligibility." We cannot see
#: outside-promotion bouts, but `ufc_fighters.status` carries release and retirement.
EXCLUDE_STATUSES = ("Retired", "Released")

#: Tapology's eligibility, expressed in the registry's vocabulary so the publisher's
#: invariant check and the ranker apply ONE predicate rather than two that disagree. The
#: 1-fight floor is theirs; the old 2-fight/5-round floor was ours.
TAPOLOGY_ELIGIBILITY = Eligibility(
    min_decided_fights=1,
    min_rounds=0,
    max_days_inactive=ELIGIBILITY_DAYS + GRACE_DAYS,
)


# ---------------------------------------------------------------------------
# Undisclosed scoring — every field here is FITTED, see tapology_fit.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Weights:
    """The confidential half of the algorithm.

    These ARE the fitted values, not guesses: coordinate descent against 163 fighters
    across 8 of Tapology's published divisions. Leave-one-division-out held out at 0.773
    against 0.795 in-sample — a gap of 0.022, so this is calibration rather than
    memorisation. Re-derive with `python -m app.services.ufc.tapology_fit`.

    The gauge is fixed at `v_ud = 1.0` — the whole vector is scale-free, so one outcome
    has to be nailed down or the fit wanders along a useless ray.
    """

    # -- Outcome ladder. "Looking at finishes, unanimous/split decisions, or round
    #    numbers are all good ideas" — confirmed as inputs, values withheld.
    v_finish: float = 1.1
    v_ud: float = 1.00          # gauge, not fitted
    v_md: float = 0.8
    v_sd: float = 0.6
    v_draw: float = 0.65

    #: "Is winning in the 1st round better than the 2nd?" Applied to finishes only, as a
    #: fraction added for each round earlier than the last.
    round_bonus: float = 0.05

    # -- Opponent tier -> multiplier. Tier is 1-10 (see `_resume`); the curve maps it to
    #    a credit multiplier. `q_exp` controls how much beating a tier-9 is worth over a
    #    tier-3, which is the single most consequential parameter in the system.
    q_min: float = 0.0
    q_max: float = 3.0
    q_exp: float = 1.0

    #: "The most recent fight is the most important fight, all the way down to the 6th
    #: oldest, which is by far the least important." Geometric over the six slots, so one
    #: parameter instead of six free weights that would certainly overfit.
    pos_decay: float = 0.95

    #: "There are age-related thresholds built into the system... if they are far in the
    #: past, they will be weighted down further and further the older they get." Calendar
    #: age, on top of the positional weighting. Half-life in days.
    age_half_life: float = 1825.0

    #: Loss magnitude relative to a win of the same method and opposition.
    loss_scale: float = 0.5

    #: "A loss to a quality opponent will not hurt a fighter's ranking as much as a loss
    #: to a weak opponent." Fraction of the penalty removed at tier 10 vs tier 1.
    loss_tier_relief: float = 0.75

    #: "The carryover from one weight class to another is not exactly the same (it's not
    #: 1:1)... these rules impact the percentage of points that can be transferred."
    #: Applied to a bout fought in a division other than the one being ranked.
    carry_other_division: float = 1.0


    def clamp(self) -> "Weights":
        return replace(
            self,
            v_finish=max(0.0, self.v_finish), v_ud=1.0,
            v_md=max(0.0, self.v_md), v_sd=max(0.0, self.v_sd),
            v_draw=max(0.0, self.v_draw),
            round_bonus=max(0.0, self.round_bonus),
            q_min=max(0.0, self.q_min),
            q_max=max(self.q_min + 0.01, self.q_max),
            q_exp=max(0.1, self.q_exp),
            pos_decay=min(max(self.pos_decay, 0.05), 0.999),
            age_half_life=max(30.0, self.age_half_life),
            loss_scale=max(0.0, self.loss_scale),
            loss_tier_relief=min(max(self.loss_tier_relief, 0.0), 1.0),
            carry_other_division=min(max(self.carry_other_division, 0.0), 1.0),
        )


FITTED = Weights()


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

def outcome_value(method: str | None, finish_round: int | None, w: Weights) -> float:
    """Value of a WIN by this method, before opponent quality and recency.

    Draws are handled by the caller — `v_draw` is not a win value.
    """
    m = (method or "").lower()
    if "decision" in m:
        if "split" in m:
            return w.v_sd
        if "majority" in m:
            return w.v_md
        return w.v_ud
    if "ko" in m or "tko" in m or "submission" in m:
        # An earlier finish is worth more. Round 1 gets the full bonus, the last
        # possible round gets none.
        r = finish_round or 1
        return w.v_finish * (1.0 + w.round_bonus * max(0, 5 - r))
    # DQ and anything unparsed score as the weakest win rather than as nothing.
    return w.v_sd


def tier_multiplier(tier: int, w: Weights) -> float:
    """Tier 1-10 -> credit multiplier."""
    t = (max(1, min(N_TIERS, tier)) - 1) / (N_TIERS - 1)
    return w.q_min + (w.q_max - w.q_min) * (t ** w.q_exp)


# ---------------------------------------------------------------------------
# The opponent tier — Tapology's own worked example, implemented
# ---------------------------------------------------------------------------
# From the FAQ, using their GSP/Hendricks example:
#
#   "for each of a fighter's last 6 opponents, we look at that opponent's last 6 UFC
#    opponents as of that specific point in time. That creates up to 36 opponents being
#    reviewed. Then, for each of those 36 fighters, Tapology looks at their Win-Loss-Draw
#    record in their own previous 6 UFC fights."
#
# So the tier of opponent O, as of date d, is built from O's last 6 bouts before d: for
# each, whether O won, and how much that opponent P had been winning in P's own previous
# 6 UFC fights. Their summary of Hendricks — "had been winning against opponents who had
# also been mostly winning" — is both halves: O's results AND P's records.
#
# The counts confirm the depth: 6 opponents x 6 of their opponents = 36 fighters, x 6
# previous fights each = "up to 216 UFC matches". Exactly two levels, not a fixed point.

@dataclass(frozen=True)
class _Bout:
    date: date
    fight_id: int
    opponent_id: int
    won: bool
    drew: bool
    method: str | None
    finish_round: int | None
    division: str


def _last_n_before(bouts: list[_Bout], before: date, n: int = WINDOW) -> list[_Bout]:
    """The n most recent bouts strictly before `before`, newest first.

    `bouts` is oldest-first, so a binary search on the date gives the cut point without
    scanning — this runs ~300k times across a full build.
    """
    lo = bisect_left([b.date for b in bouts], before)
    return bouts[max(0, lo - n):lo][::-1]


def _win_rate_6(bouts: list[_Bout], before: date) -> float:
    """P's W-L-D record over P's previous 6 UFC fights, as a rate.

    A draw counts a half. A fighter with no prior UFC bouts returns 0.0 rather than 0.5:
    Tapology treats the UFC newcomer as unproven, not as average — "it is meant to confirm
    that their UFC schedule has been limited and therefore not yet considered strong."
    """
    prior = _last_n_before(bouts, before)
    if not prior:
        return 0.0
    credit = sum(1.0 if b.won else (0.5 if b.drew else 0.0) for b in prior)
    return credit / len(prior)


def _resume(bouts_by_fighter: dict[int, list[_Bout]], fid: int, before: date) -> float:
    """How much fighter `fid` had proven themselves as of `before`.

    Summed, not averaged, over up to 6 bouts — which is what makes a short UFC record
    score low by construction: "a fighter with only 1 or 2 UFC fights will only have 1 or
    2 opponent tier numbers being added together instead of 6."
    """
    own = bouts_by_fighter.get(fid)
    if not own:
        return 0.0
    total = 0.0
    for b in _last_n_before(own, before):
        credit = 1.0 if b.won else (0.5 if b.drew else 0.0)
        opp_strength = _win_rate_6(bouts_by_fighter.get(b.opponent_id, []), b.date)
        total += credit * opp_strength
    return total


# The resume -> tier mapping is ABSOLUTE, not a percentile of the active roster.
#
# This was measured, not assumed. Deciling resume across everyone active puts all six of
# an elite fighter's opponents in the top two deciles by construction, which produced a
# Strength of Schedule of 94 for Gaethje against Tapology's published 67 — and the same
# overshoot for all twenty fighters on their lightweight page, mean error +29.9 with no
# fighter under. Fitting `a + b * resume` against those twenty published values instead
# gives mean error 5.3 and no systematic bias.
#
# An absolute scale is also the more defensible object: a fighter's tier then depends only
# on their own record and their opponents', not on who else happens to be active, so a
# quiet division cannot inflate everybody in it.
#
#: FITTED against the 20 published SoS values on Tapology's lightweight page (2026-09-19).
#: Re-derive with `python -m app.services.ufc.tapology_fit --sos`.
TIER_INTERCEPT = -0.9
TIER_SLOPE = 2.5


def opponent_tier(resume: float) -> int:
    """Resume score -> tier 1-10."""
    return max(1, min(N_TIERS, round(TIER_INTERCEPT + TIER_SLOPE * resume)))


# ---------------------------------------------------------------------------
# History build
# ---------------------------------------------------------------------------

def build_history(db, as_of: date | None = None) -> dict:
    """Everything that does NOT depend on the fitted weights OR on the as-of date.

    Split out for two callers. The fitter rescores hundreds of candidate parameter sets,
    and this pass — every bout in UFC history plus the tier recursion over all of them —
    is identical for all of them. The rank-history backfill replays hundreds of DATES, and
    this pass is identical for those too: an opponent's tier is fixed at the time of the
    bout, so it never depends on when you ask.

    `as_of` is accepted and ignored, for callers that still pass it.
    """
    rows = (
        db.query(
            UFCFight.id, UFCFight.date, UFCFight.red_fighter_id,
            UFCFight.blue_fighter_id, UFCFight.winner_id, UFCFight.method,
            UFCFight.finish_round, UFCFight.weight_class,
            UFCFight.red_result, UFCFight.blue_result,
        )
        .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
        .order_by(UFCFight.date, UFCFight.id)
        .all()
    )

    fighters = db.query(
        UFCFighter.id, UFCFighter.first_name, UFCFighter.last_name, UFCFighter.status,
    ).all()
    names = {f[0]: f"{f[1] or ''} {f[2] or ''}".strip() for f in fighters}
    status = {f[0]: (f[3] or "") for f in fighters}

    bouts_by_fighter: dict[int, list[_Bout]] = defaultdict(list)
    #: EVERY bout date per fighter, including no-contests. Kept as a list rather than a
    #: running maximum so `score` can ask "when did they last compete as of date D" —
    #: which is what lets one build serve a whole historical replay.
    activity: dict[int, list[date]] = defaultdict(list)
    #: (date, division) per bout, for the 24-month per-division eligibility window.
    div_bouts: dict[int, list[tuple[date, str]]] = defaultdict(list)
    #: Chronological list of scoring bouts, for the tier pass below.
    timeline: list[tuple[date, int, int, int]] = []  # (date, fighter, fight, opponent)

    for fight_id, d, red, blue, winner, method, fround, wc, red_res, blue_res in rows:
        # No `d > as_of` filter: the context holds ALL of history and `score` bounds it
        # per call. That is what lets the rank-history backfill build once and replay ~700
        # event dates against the same context instead of rebuilding for each.
        if not d or red is None or blue is None:
            continue

        division = classify_weight_class(wc)

        # "The only thing a No-Contest will do for a fighter is reset their timer on being
        # considered an active fighter." So activity counts every bout, decided or not.
        for fid in (red, blue):
            activity[fid].append(d)
            # "Catchweight fights do not count towards eligibility in any weight class"
            # — classify_weight_class returns "unknown" for them.
            if division != "unknown":
                div_bouts[fid].append((d, division))

        # "When calculating a fighter's ranking, their last 6 fights will be selected with
        # any No-Contests skipped as if they were not there at all." A skipped bout
        # occupies no slot, so the 7th-oldest is pulled in. An overturned result is a
        # no-contest under another name.
        #
        # Draws are NOT skipped: "Draws are eligible and included when selecting a
        # fighter's last 6 UFC bouts."
        #
        # The per-corner result code is the authority for all of this, not the method
        # string. A draw is stored as the DECISION that produced it ("Decision - Majority")
        # with a null winner, so both a method-text check for "draw" and the registry's
        # `is_decided` drop it — which left Chris Padilla (5-0-1) one bout short of
        # Tapology's published six. `is_decided` is correct for a rating engine, which has
        # no winner to update toward; it is wrong for window selection.
        for me, opp, code in ((red, blue, red_res), (blue, red, blue_res)):
            if code == "NC" or not code:
                continue
            bouts_by_fighter[me].append(_Bout(
                date=d, fight_id=fight_id, opponent_id=opp,
                won=(code == "W"), drew=(code == "D"), method=method,
                finish_round=fround, division=division,
            ))
            timeline.append((d, me, fight_id, opp))

    # -- Tier pass -----------------------------------------------------------
    # The tier attached to a bout is the tier its opponent held AT THE TIME — "as of that
    # specific point in time". `_resume` is bounded at the bout date for exactly this
    # reason: a bout's credit must not change years later because the opponent's career
    # continued.
    tiers: dict[tuple[int, int], int] = {}      # (fighter_id, fight_id) -> opponent tier
    for d, fid, fight_id, opp in timeline:
        tiers[(fid, fight_id)] = opponent_tier(_resume(bouts_by_fighter, opp, d))

    from app.services.ufc.champions import load_title_bouts

    return {
        "bouts": bouts_by_fighter, "tiers": tiers, "names": names,
        "status": status, "activity": {f: sorted(ds) for f, ds in activity.items()},
        "div_bouts": div_bouts, "title_bouts": load_title_bouts(db),
    }


def _divisions_as_of(div_bouts: dict[int, list[tuple[date, str]]],
                     as_of: date) -> dict[int, list[str]]:
    """Which division(s) each fighter is ranked in.

    "Fighters are ranked in whichever weight classes they competed in during their most
    recent 2 UFC matches. If both matches were in the same weight class the fighter will
    only be ranked in that class."

    So the answer is a SET, not a single division: when the last two bouts disagree the
    fighter appears in BOTH, which is how Tapology has Usman at welterweight and
    middleweight, Pereira at light heavyweight and heavyweight, and Whittaker, Page, Costa,
    de Ridder, Rakić and Oliveira each in two. Collapsing this to one division was silently
    costing us one fighter in each of eight lists, and shifting everyone below them up.

    The 24-month per-class window is applied first: "a fighter must have competed in a
    weight class within the last 24 months to remain eligible for that specific class",
    which is what stops a long-ago division from being resurrected by one recent bout.
    """
    out: dict[int, list[str]] = {}
    for fid, bouts in div_bouts.items():
        # 0 <= delta, not just delta <= window: a bout AFTER `as_of` yields a negative
        # delta, which sails through an upper-bound-only test and leaks a future division
        # into a historical ranking.
        recent = [d for dt, d in bouts
                  if 0 <= (as_of - dt).days <= DIVISION_WINDOW_DAYS]
        if not recent:
            continue
        # dict.fromkeys keeps the last-two in order while de-duplicating them.
        out[fid] = list(dict.fromkeys(recent[-2:]))
    return out


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

class TapologyRanker:
    """Tapology's disclosed rules, with the confidential scoring fitted."""

    name = "tapology"

    def __init__(self, weights: Weights | None = None) -> None:
        self.w = (weights or FITTED).clamp()

    def rank(self, db, registry: dict, as_of: date, crit=None):
        return self.score(build_history(db, as_of), as_of, registry)

    def score(self, ctx: dict, as_of: date, registry: dict | None = None):
        from app.services.ufc.ranking_publisher import RankingResult

        w = self.w
        bouts_by_fighter = ctx["bouts"]
        tiers = ctx["tiers"]
        names = ctx["names"]
        status = ctx["status"]
        activity = ctx["activity"]
        divisions_now = _divisions_as_of(ctx["div_bouts"], as_of)

        # The context spans all of history, so every read of it here is bounded at
        # `as_of`. Anything below that reached past this date would be using results that
        # had not happened yet — which is exactly how a backfilled historical ranking ends
        # up looking plausible and being wrong.
        cutoff = as_of + timedelta(days=1)

        from app.services.ufc.champions import champions_as_of
        champions = champions_as_of(ctx["title_bouts"], as_of, divisions_now)

        # Keyed by (fighter, division): a fighter ranked in two classes scores separately
        # in each, because the weight-class carryover haircut is applied relative to the
        # class being ranked. `scores` keeps the fighter's best for consumers that still
        # want one number per fighter.
        div_scores: dict[tuple[int, str], float] = {}
        scores: dict[int, float] = {}
        extras: dict[int, dict] = {}
        eligible_divisions: dict[int, set[str]] = {}

        # `status` is a PRESENT-TENSE fact about the roster: it records who is retired or
        # released NOW, not who was on any given past date. Applying it to a historical
        # ranking deletes all 135 currently-retired fighters from every standing they ever
        # held — Khabib Nurmagomedov was the reigning lightweight champion in January 2020
        # and disappeared from that ranking altogether. For past dates the 21-month
        # inactivity rule is the correct instrument, and it is the one that actually
        # removed people at the time.
        apply_status = (date.today() - as_of).days <= GRACE_DAYS

        for fid, bouts in bouts_by_fighter.items():
            if apply_status and status.get(fid) in EXCLUDE_STATUSES:
                continue

            # Past 21 months a fighter is ineligible and leaves the numbered list. The FAQ
            # says they "remain displayed for 60 days in the position they would have
            # occupied", and the page does still show them — but under a `snooze` marker,
            # OUTSIDE the numbering, not at a rank. Shavkat Rakhmonov sat at our
            # welterweight #3 while Tapology had him snoozed and unranked, pushing every
            # fighter below him off by one.
            acts = activity.get(fid, ())
            i = bisect_left(acts, cutoff)
            if i == 0:
                continue                      # had not debuted yet as of this date
            idle = (as_of - acts[i - 1]).days
            if idle > ELIGIBILITY_DAYS:
                continue

            window = _last_n_before(bouts, cutoff)
            if not window:
                continue

            for division in divisions_now.get(fid, []):
                if division not in DIVISIONS:
                    continue

                total, ledger = 0.0, []
                for i, b in enumerate(window):
                    tier = tiers.get((fid, b.fight_id), (N_TIERS + 1) // 2)
                    q = tier_multiplier(tier, w)
                    days_ago = (as_of - b.date).days
                    recency = (w.pos_decay ** i) * (0.5 ** (days_ago / w.age_half_life))
                    carry = 1.0 if b.division == division else w.carry_other_division
                    v = outcome_value(b.method, b.finish_round, w)

                    if b.drew:
                        # "A draw probably does not help as much as a win, but it's not as
                        # bad as a loss." A reduced win, not its own category.
                        pts = w.v_draw * q * recency * carry
                    elif b.won:
                        pts = v * q * recency * carry
                    else:
                        # Losing to a tier-10 costs less than losing to a tier-1.
                        relief = 1.0 - w.loss_tier_relief * ((tier - 1) / (N_TIERS - 1))
                        pts = -w.loss_scale * v * relief * recency * carry

                    total += pts
                    ledger.append({
                        "fight_id": str(b.fight_id), "date": b.date.isoformat(),
                        "opponent_id": str(b.opponent_id),
                        "opponent_name": names.get(b.opponent_id, ""),
                        "won": b.won, "drew": b.drew, "method": b.method,
                        "tier": tier, "points": round(pts, 3),
                    })

                div_scores[(fid, division)] = total
                eligible_divisions.setdefault(fid, set()).add(division)
                if total >= scores.get(fid, float("-inf")):
                    scores[fid] = total
                    extras[fid] = {
                        # "The tier numbers for the last 6 opponents are added together...
                        # then converted from a 1-to-60 scale into a 1-to-99 scale."
                        # Display only: "Strength of Schedule is not a direct input into
                        # the rankings."
                        "sos": max(1, min(99, round(
                            sum(l["tier"] for l in ledger) / (WINDOW * N_TIERS) * 99))),
                        "points": round(total, 3),
                        "eligible": True,
                        "ledger": ledger,
                    }

        order: dict[str, list[int]] = defaultdict(list)
        for (fid, d) in div_scores:
            order[d].append(fid)

        for d in order:
            order[d].sort(key=lambda f: div_scores[(f, d)], reverse=True)
            # "Whichever UFC fighter most recently won the undisputed (non-interim) title
            # in each weight class will always be in the #1 position" — even when their
            # last six bouts scored fewer points than a contender's.
            champ = champions.get(d)
            if champ is not None and (champ, d) in div_scores:
                order[d].remove(champ)
                order[d].insert(0, champ)

        log.info(f"  Tapology: scored {len(scores)} fighters into "
                 f"{len(div_scores)} division slots across {len(order)} divisions")
        return RankingResult(order=dict(order), scores=scores, extras=extras,
                             division_scores=div_scores,
                             eligible_divisions=eligible_divisions)


# ---------------------------------------------------------------------------
# Read side — what /ufc/rankings serves
# ---------------------------------------------------------------------------

#: The 15 Glicko skill dimensions behind the radar chart. These are NOT ranking inputs —
#: they come from `ranking_service.compute_dimension_profiles` and exist only so the UI
#: can show a skill breakdown next to a rank. Keeping them clearly separate is deliberate:
#: conflating the two is what made the old rankings impossible to reason about.
DIMENSIONS = [
    "pts", "ko", "kod", "sub", "subd",
    "td", "tdd", "ctrl",
    "str_vol", "str_acc", "str_def",
    "dist", "clinch", "gnd",
    "durability",
]


#: How far back "movement" looks. A single event only churns the handful of divisions
#: that fought on it, so a since-last-publish delta is blank for most of the board and
#: noisy where it isn't. A quarter is long enough that the arrow reflects a trajectory.
MOVEMENT_DAYS = 90


def _movement_baseline(db, fighter_ids: set[int]) -> dict[tuple[int, str], dict]:
    """Standings as of ~`MOVEMENT_DAYS` ago, keyed by (fighter_id, weight_class).

    Snapshots are written per event date, so there is rarely a row exactly 90 days back.
    This picks the one distinct `as_of` nearest the target and reads the whole board at
    that date — one query, and every fighter is compared against the *same* past
    standings. Comparing each fighter to their own nearest snapshot would mix dates and
    make two fighters' arrows incomparable.
    """
    from sqlalchemy import func, select

    from app.models.ufc import UFCRankingHistory

    if not fighter_ids:
        return {}

    target = date.today() - timedelta(days=MOVEMENT_DAYS)
    as_of = db.execute(
        select(func.max(UFCRankingHistory.as_of)).where(UFCRankingHistory.as_of <= target)
    ).scalar()
    if as_of is None:
        # History does not reach back 90 days yet — fall back to the oldest snapshot
        # there is, so the column degrades to "since we started tracking" rather than
        # reporting every fighter as NEW.
        as_of = db.execute(select(func.min(UFCRankingHistory.as_of))).scalar()
    if as_of is None:
        return {}

    rows = (
        db.query(UFCRankingHistory)
        .filter(
            UFCRankingHistory.as_of == as_of,
            UFCRankingHistory.fighter_id.in_(fighter_ids),
        )
        .all()
    )
    return {
        (r.fighter_id, r.weight_class): {
            "as_of": r.as_of.isoformat(), "rank": r.rank, "sos": r.sos,
        }
        for r in rows
    }


def _next_fights(db, fighter_ids: set[int]) -> dict[int, dict]:
    """The announced upcoming bout per ranked fighter, if any.

    Keyed by fighter id rather than (fighter, division): a fighter has at most one booked
    fight, and it may well be in a division they are not ranked in (a move up, a catchweight),
    which is exactly the case worth surfacing on a rankings board.
    """
    from sqlalchemy.orm import aliased

    from app.models.ufc import UFCFightOdds, UFCFightPrediction

    if not fighter_ids:
        return {}

    # Yesterday, not today: events stay listed through the day after so a UTC offset
    # never hides a card that is still in progress. Mirrors /ufc/upcoming.
    cutoff = date.today() - timedelta(days=1)
    red_f = aliased(UFCFighter)
    blue_f = aliased(UFCFighter)

    rows = (
        db.query(UFCFight, UFCEvent, red_f, blue_f, UFCFightPrediction)
        .join(UFCEvent, UFCEvent.id == UFCFight.event_id)
        .join(red_f, red_f.id == UFCFight.red_fighter_id)
        .join(blue_f, blue_f.id == UFCFight.blue_fighter_id)
        .outerjoin(UFCFightPrediction, UFCFightPrediction.fight_id == UFCFight.id)
        .filter(
            UFCFight.winner_id.is_(None),
            UFCEvent.date >= cutoff,
            (UFCFight.red_fighter_id.in_(fighter_ids)) | (UFCFight.blue_fighter_id.in_(fighter_ids)),
        )
        # Soonest card first, so the dict assignment below keeps the nearest bout for a
        # fighter booked on two future events.
        .order_by(UFCEvent.date.desc())
        .all()
    )

    # Market prices for those same bouts, one query rather than one per fight. The stored
    # implied probabilities are already vig-removed, so averaging across books gives a
    # consensus that is directly comparable to the model's number.
    odds_by_fight: dict[int, list] = {}
    if rows:
        for o in db.query(UFCFightOdds).filter(
            UFCFightOdds.fight_id.in_([f.id for f, *_ in rows])
        ).all():
            odds_by_fight.setdefault(o.fight_id, []).append(o)

    out: dict[int, dict] = {}
    for fight, event, red, blue, pred in rows:
        books = odds_by_fight.get(fight.id, [])
        for fid, opp, is_red in ((red.id, blue, True), (blue.id, red, False)):
            if fid not in fighter_ids:
                continue
            win_prob = None
            if pred is not None and pred.red_prob is not None:
                win_prob = round(pred.red_prob if is_red else 1 - pred.red_prob, 4)

            market_prob, best_odds, best_book = None, None, None
            if books:
                probs = [(b.red_implied_prob if is_red else b.blue_implied_prob) for b in books]
                probs = [p for p in probs if p is not None]
                if probs:
                    market_prob = round(sum(probs) / len(probs), 4)
                prices = [
                    ((b.red_odds if is_red else b.blue_odds), b.bookmaker)
                    for b in books
                    if (b.red_odds if is_red else b.blue_odds) is not None
                ]
                if prices:
                    # Best price for a bettor is the highest American number: +250 pays
                    # more than +180, and -110 pays more than -150.
                    best_odds, best_book = max(prices, key=lambda p: p[0])
            out[fid] = {
                "fight_id": str(fight.id),
                "event_name": event.name,
                "event_date": event.date.isoformat() if event.date else None,
                "event_location": event.location,
                "weight_class": fight.weight_class,
                "opponent_id": str(opp.id),
                "opponent_name": f"{opp.first_name or ''} {opp.last_name or ''}".strip(),
                "opponent_country_code": opp.country_code,
                "opponent_image_url": opp.image_url,
                "opponent_record": [opp.wins, opp.losses, opp.draws],
                "win_probability": win_prob,
                # Vig-removed consensus across books, the best American price on offer,
                # and who offers it. `edge` is the model's disagreement with the market —
                # the whole reason both numbers are shown side by side.
                "market_probability": market_prob,
                "best_odds": best_odds,
                "best_book": best_book,
                "book_count": len(books),
                "edge": (round(win_prob - market_prob, 4)
                         if win_prob is not None and market_prob is not None else None),
            }
    return out


def _last_fight_details(db, fight_ids: set[int]) -> dict[int, dict]:
    """Round, time and event for each fighter's most recent scored bout.

    The ledger is frozen into `feature_profile` at publish time and carries only what the
    scorer needed, so these are joined at read time instead — that keeps "how did it end"
    available on the board without forcing a republish to widen the stored ledger.
    """
    if not fight_ids:
        return {}
    rows = (
        db.query(UFCFight, UFCEvent)
        .outerjoin(UFCEvent, UFCEvent.id == UFCFight.event_id)
        .filter(UFCFight.id.in_(fight_ids))
        .all()
    )
    return {
        fight.id: {
            "finish_round": fight.finish_round,
            "finish_time": fight.finish_time,
            "time_format": fight.time_format,
            "details": fight.details,
            "event_name": event.name if event else None,
        }
        for fight, event in rows
    }


def get_rankings() -> dict:
    import json

    from app.models.ufc import UFCFighterRanking

    db = SessionLocal()
    try:
        rankings = (
            db.query(UFCFighterRanking, UFCFighter)
            .join(UFCFighter, UFCFighterRanking.fighter_id == UFCFighter.id)
            # rank=0 was an old placeholder. 0 sorts ahead of 1, so any surviving row
            # landed at the TOP of its division. Never serve them.
            .filter(UFCFighterRanking.rank > 0)
            .order_by(UFCFighterRanking.weight_class, UFCFighterRanking.rank)
            .all()
        )
        if not rankings:
            return {"weight_classes": [], "method": "none"}

        fighter_ids = {r.fighter_id for r, _ in rankings}
        baseline = _movement_baseline(db, fighter_ids)
        next_fights = _next_fights(db, fighter_ids)

        # Collect every fighter's most recent bout id first so the detail join is one
        # query for the whole board rather than one per row.
        profiles: dict[int, dict] = {}
        last_ids: set[int] = set()
        last_opp_ids: set[int] = set()
        for ranking, _ in rankings:
            try:
                p = json.loads(ranking.feature_profile) if ranking.feature_profile else {}
            except (json.JSONDecodeError, TypeError):
                p = {}
            profiles[ranking.id] = p
            led = p.get("ledger") or []
            if led and str(led[0].get("fight_id", "")).isdigit():
                last_ids.add(int(led[0]["fight_id"]))
            if led and str(led[0].get("opponent_id", "")).isdigit():
                last_opp_ids.add(int(led[0]["opponent_id"]))
        last_details = _last_fight_details(db, last_ids)
        # Portraits for the last-fight opponents. The ledger stores only a name and id,
        # and the board shows a headshot per bout, so the photo is joined here rather
        # than fetched per row on the client.
        opp_images = dict(
            db.query(UFCFighter.id, UFCFighter.image_url)
            .filter(UFCFighter.id.in_(last_opp_ids))
            .all()
        ) if last_opp_ids else {}

        wc_map: dict[str, list] = {}
        for ranking, fighter in rankings:
            profile = profiles.get(ranking.id, {})
            ledger = profile.get("ledger", [])
            prior = baseline.get((ranking.fighter_id, ranking.weight_class))

            last_fight = None
            if ledger:
                fid = str(ledger[0].get("fight_id", ""))
                oid = str(ledger[0].get("opponent_id", ""))
                last_fight = {
                    **ledger[0],
                    **(last_details.get(int(fid), {}) if fid.isdigit() else {}),
                    "opponent_image_url": opp_images.get(int(oid)) if oid.isdigit() else None,
                }

            # Current streak, read off the front of the ledger. A draw ends a streak
            # without starting one, which is how every record line treats it.
            streak, streak_type = 0, None
            if ledger and not ledger[0].get("drew"):
                streak_type = "W" if ledger[0]["won"] else "L"
                for b in ledger:
                    if b.get("drew") or b["won"] != ledger[0]["won"]:
                        break
                    streak += 1
            wc_map.setdefault(ranking.weight_class, []).append({
                "id": str(fighter.id),
                "first_name": fighter.first_name,
                "last_name": fighter.last_name,
                "nickname": fighter.nickname,
                "wins": fighter.wins,
                "losses": fighter.losses,
                "draws": fighter.draws,
                "country_code": fighter.country_code,
                "image_url": fighter.image_url,
                "rank": ranking.rank,
                "score": round(ranking.score, 1),
                "dimensions": {d: profile.get(d, 0) for d in DIMENSIONS},
                "uncertainty": profile.get("uncertainty", 0),
                "sos": profile.get("sos", 0),
                "points": profile.get("points", 0),
                # False once past 21 months — Tapology keeps showing the fighter for
                # another 60 days, marked ineligible, rather than dropping them silently.
                "eligible": profile.get("eligible", True),
                # The per-bout decomposition. The point of a rules-based ranking is that
                # it can be audited, so the breakdown ships with the number.
                "ledger": ledger,
                # `ledger` is ordered most-recent-first (the scorer's recency decay is
                # indexed off position 0), so the head of it IS the last fight. Lifted out
                # as its own field so the row does not have to know that.
                "last_fight": last_fight,
                "streak": streak,
                "streak_type": streak_type,
                # Rank ~90 days ago and the delta from it. `delta` is POSITIVE when the
                # fighter climbed: ranks count down, so it is prior - current, not the
                # other way round. None means they were unranked in this division then.
                "movement": {
                    "prior_rank": prior["rank"] if prior else None,
                    "delta": (prior["rank"] - ranking.rank) if prior else None,
                    "prior_sos": prior["sos"] if prior else None,
                    "as_of": prior["as_of"] if prior else None,
                },
                "next_fight": next_fights.get(ranking.fighter_id),
            })

        return {
            "weight_classes": [
                {"key": wc, "label": WEIGHT_CLASS_LABELS.get(wc, wc), "fighters": wc_map[wc]}
                for wc in DIVISIONS
                if wc in wc_map
            ],
            "method": "tapology",
            "dimensions": DIMENSIONS,
            "window": WINDOW,
            "movement_days": MOVEMENT_DAYS,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def generate_rankings(preview: bool = False):
    from app.services.ufc.ranking_publisher import publish_rankings

    db = SessionLocal()
    try:
        publish_rankings(db, ranker=TapologyRanker(), crit=TAPOLOGY_ELIGIBILITY,
                         preview=preview)
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_rankings(preview="--preview" in sys.argv)
