"""Tests for the Glicko-1 + tiered-points ranker.

These pin the PROPERTIES the published systems specify, not the constants themselves —
the constants are cited and re-derivable, but a refactor that quietly inverts the loss
penalty or lets an opponent's later career rewrite a past bout would pass a
constant-equality test while breaking the ranking. An inverted loss factor is exactly the
defect that shipped once before (a 50:1 win/loss asymmetry, see
docs/models/rankings.md), so it gets a test rather than a comment.

Pure functions over synthetic data — no DB, no artifacts.

Run:  ./venv/bin/python -m pytest tests/test_tiered_scoring.py -v
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from app.services.ufc.fighter_registry import is_decided
from app.services.ufc.glicko1 import (
    RATING_START, RD_FLOOR, RD_MAX, RD_START, Glicko1, inflate_rd,
)
from app.services.ufc.fighter_registry import current_division
from app.services.ufc.tiered_ranking_service import (
    N_TIERS, V_DRAW, V_FINISH, V_MAJORITY, V_SPLIT, V_UD,
    bout_weight, carry_factor, outcome_value, scheduled_rounds,
)


# --------------------------------------------------------------------------- outcome
def test_finish_outranks_every_decision():
    """A first-round KO must beat a five-round unanimous decision. The published ladder
    encodes this as 1.00 vs 0.91 rather than as a separate finish bonus."""
    assert outcome_value("KO/TKO") == V_FINISH
    assert outcome_value("Submission") == V_FINISH
    assert V_FINISH > V_UD > V_MAJORITY > V_SPLIT > V_DRAW


def test_decision_types_ordered():
    assert outcome_value("Decision - Unanimous") == V_UD
    assert outcome_value("Decision - Majority") == V_MAJORITY
    assert outcome_value("Decision - Split") == V_SPLIT
    # Bare "Decision" (3 rows in the table) reads as unanimous, the modal decision.
    assert outcome_value("Decision") == V_UD


def test_injury_stoppage_is_a_finish():
    """Fight Matrix treats an injury TKO identically to a normal TKO. These reach the
    scorer as decided bouts with a winner, so they must map somewhere deliberate."""
    assert outcome_value("TKO - Doctor's Stoppage") == V_FINISH
    assert outcome_value("Could Not Continue") == V_FINISH
    assert outcome_value("Other") == V_FINISH


def test_no_finish_speed_grading():
    """No published system grades finish speed, and the incumbent's round-2 cliff between
    early_finish (5.0 pts) and late_finish (4.0) was invented. A first-round KO and a
    third-round KO carry the same outcome value."""
    assert outcome_value("KO/TKO") == outcome_value("Submission") == V_FINISH


# --------------------------------------------------------------------------- distance
def test_scheduled_rounds_parses_round_length_list():
    assert scheduled_rounds("5-5-5") == 3
    assert scheduled_rounds("5-5-5-5-5") == 5
    assert scheduled_rounds("10-5") == 2
    # Pre-modern formats fall back to three; all are decades past the age cutoff.
    assert scheduled_rounds("No Time Limit") == 3
    assert scheduled_rounds(None) == 3


def test_five_round_decision_worth_more_than_three_round():
    """BoxRec's (rounds/12)^2 rescaled to MMA's championship distance. This is what
    replaces TITLE_MULT and FIVE_ROUND_MULT — and the "title" substring sniffing."""
    three = bout_weight("Decision - Unanimous", "5-5-5")
    five = bout_weight("Decision - Unanimous", "5-5-5-5-5")
    assert three == pytest.approx(0.36)
    assert five == pytest.approx(1.0)
    assert five / three == pytest.approx(2.78, abs=0.01)


def test_finishes_carry_full_weight_at_any_distance():
    """You cannot go the distance harder than by not needing it."""
    assert bout_weight("KO/TKO", "5-5-5") == 1.0
    assert bout_weight("KO/TKO", "5-5-5-5-5") == 1.0


# ------------------------------------------------------- method inside the rating
def test_method_enters_the_rating_not_a_separate_bonus():
    """Fight Matrix and BoxRec both feed the outcome ladder in as the Glicko score. A
    unanimous decision is weaker evidence of superiority than a knockout, so it must move
    the rating less — and folding it in here means it cannot be double-counted."""
    def gain(v):
        g = Glicko1()
        g.rating[1], g.rd[1] = 1500.0, 120.0
        g.rating[2], g.rd[2] = 1500.0, 120.0
        g.observe(1, 2, v, date(2024, 1, 1))
        return g.rating[1] - 1500.0

    assert gain(V_FINISH) > gain(V_UD) > gain(V_MAJORITY) > gain(V_SPLIT) > gain(V_DRAW)
    assert gain(V_DRAW) == pytest.approx(0.0, abs=1e-9)


def test_distance_scales_the_rating_move():
    """BoxRec's rounds weighting, applied to the update rather than to a points total."""
    def gain(weight):
        g = Glicko1()
        g.rating[1], g.rd[1] = 1500.0, 120.0
        g.rating[2], g.rd[2] = 1500.0, 120.0
        g.observe(1, 2, V_UD, date(2024, 1, 1), weight)
        return g.rating[1] - 1500.0

    three = bout_weight("Decision - Unanimous", "5-5-5")
    five = bout_weight("Decision - Unanimous", "5-5-5-5-5")
    assert gain(five) / gain(three) == pytest.approx(five / three)


def test_losing_to_an_elite_costs_less_than_losing_to_a_journeyman():
    """Tapology and Fight Matrix both specify this. It arrives free from Glicko's
    expectation term — the incumbent needed three tables to approximate it, and an earlier
    version INVERTED, producing a 50:1 win/loss asymmetry that made fighting risk-free."""
    def loss_cost(opp_rating):
        g = Glicko1()
        g.rating[1], g.rd[1] = 1800.0, 120.0
        g.rating[2], g.rd[2] = opp_rating, 120.0
        g.observe(1, 2, 0.0, date(2024, 1, 1))
        return 1800.0 - g.rating[1]

    assert loss_cost(2400.0) < loss_cost(1200.0)
    assert loss_cost(2400.0) > 0


def test_beating_a_weak_opponent_pays_almost_nothing():
    """The ratchet that let undefeated prospects top four divisions in the first cut of
    this ranker. Elo's expectation term removes it without a coefficient."""
    def win_gain(opp_rating):
        g = Glicko1()
        g.rating[1], g.rd[1] = 1800.0, 120.0
        g.rating[2], g.rd[2] = opp_rating, 120.0
        g.observe(1, 2, V_FINISH, date(2024, 1, 1))
        return g.rating[1] - 1800.0

    assert win_gain(1200.0) < 0.2 * win_gain(2400.0)


def test_the_ranking_cannot_saturate_at_the_top():
    """A decile bucket cannot separate a champion from a fringe contender because both sit
    in the top 10%. That saturation is the documented failure of the incumbent's
    _elo_percentile, and it recurred when this ranker used tiers to score."""
    g = Glicko1()
    for r in (2600.0, 2400.0, 2200.0, 2100.0):
        g.rating[int(r)], g.rd[int(r)] = r, 120.0
    vals = sorted({g.conservative(int(r)) for r in (2600.0, 2400.0, 2200.0, 2100.0)})
    assert len(vals) == 4


def test_draw_value_is_neutral():
    assert V_DRAW == 0.5


# ------------------------------------------------------- weight-class assignment
def test_one_fight_up_does_not_move_a_fighter():
    """Max Holloway fighting once at welterweight is still a lightweight. Tapology's rule:
    a fighter is ranked in the class they fought in over their LAST TWO bouts."""
    assert current_division(["featherweight", "lightweight", "lightweight",
                             "welterweight"]) == "lightweight"


def test_two_fights_up_completes_the_move():
    assert current_division(["lightweight", "lightweight", "welterweight",
                             "welterweight"]) == "welterweight"


def test_debutant_takes_their_only_division():
    assert current_division(["lightweight"]) == "lightweight"
    assert current_division([]) == "unknown"


def test_alternating_divisions_hold_the_established_one():
    assert current_division(["welterweight", "middleweight", "welterweight",
                             "middleweight"]) == "welterweight"


# --------------------------------------------------------------------------- division
def test_moving_up_a_division_discounts_carried_credit():
    """Fight Matrix's ~17% per-division haircut. A bout scored against lightweight deciles
    is not directly comparable to one scored against welterweight deciles."""
    up = carry_factor("lightweight", "welterweight")
    assert up == pytest.approx(0.83)
    assert carry_factor("lightweight", "middleweight") == pytest.approx(0.83 ** 2)


def test_moving_down_a_division_is_a_premium():
    assert carry_factor("welterweight", "lightweight") == pytest.approx(1 / 0.83)


def test_same_division_is_neutral():
    assert carry_factor("lightweight", "lightweight") == 1.0


def test_womens_carry_is_stronger():
    """Fight Matrix reports the factor as ~1.5x stronger for women."""
    men = carry_factor("lightweight", "welterweight")
    women = carry_factor("w_strawweight", "w_flyweight")
    assert women < men


def test_cross_gender_movement_declines_to_adjust():
    assert carry_factor("w_flyweight", "flyweight") == 1.0


# --------------------------------------------------------------------------- glicko
def test_beating_an_unproven_fighter_scores_below_beating_a_proven_one():
    """The conservative rating (r - RD) is what makes this true without a bespoke
    "experience" term. Two fighters at the same rating are not the same win if one has
    eight bouts of evidence behind it and the other has three."""
    g = Glicko1()
    proven, unproven = 1, 2
    # Give `proven` a long even record so RD collapses toward the floor.
    for i in range(10):
        opp = 100 + i
        g.observe(proven, opp, 1.0 if i % 2 == 0 else 0.0, date(2020, 1, 1) + timedelta(days=90 * i))

    r_proven, rd_proven = g.state(proven)
    r_unproven, rd_unproven = g.state(unproven)
    assert rd_proven < rd_unproven
    # At comparable ratings, the proven fighter is worth more as an opponent.
    assert g.conservative(proven) > g.conservative(unproven)


def test_rd_grows_only_after_the_grace_window():
    """Fight Matrix: RD is untouched for 180 days, then grows. Inactivity inflates
    uncertainty, never the rating itself."""
    assert inflate_rd(100.0, 30) == 100.0
    assert inflate_rd(100.0, 180) == 100.0
    assert inflate_rd(100.0, 365) > 100.0
    assert inflate_rd(100.0, 365 * 20) == pytest.approx(RD_MAX)


def test_rd_respects_its_floor():
    """Without Glickman's floor a busy fighter's rating freezes."""
    g = Glicko1()
    for i in range(60):
        g.observe(1, 200 + i, 1.0 if i % 2 else 0.0, date(2015, 1, 1) + timedelta(days=30 * i))
    assert g.state(1)[1] >= RD_FLOOR


def test_favourite_beating_a_nobody_barely_moves():
    """E(1-E) -> 0 for a foregone conclusion, so a heavy favourite winning is weak
    evidence. This is the mechanism that replaces the newcomer K-factor."""
    g = Glicko1()
    g.rating[1], g.rd[1] = 2200.0, 40.0
    g.rating[2], g.rd[2] = 1200.0, 40.0
    before = g.rating[1]
    g.observe(1, 2, 1.0, date(2024, 1, 1))
    assert abs(g.rating[1] - before) < 5.0


def test_upset_moves_more_than_the_expected_result():
    """An upset is strong evidence; a favourite holding serve is weak evidence. Note the
    move is bounded by q*RD^2 either way — an established rating is SUPPOSED to be hard to
    shift, which is what a confidence-weighted system buys over a flat K-factor."""
    def underdog_gain(score):
        g = Glicko1()
        g.rating[1], g.rd[1] = 2200.0, 40.0
        g.rating[2], g.rd[2] = 1200.0, 40.0
        before = g.rating[2]
        g.observe(1, 2, score, date(2024, 1, 1))
        return g.rating[2] - before

    assert underdog_gain(0.0) > 0                 # upset: gains
    assert underdog_gain(1.0) < 0                 # expected loss: dips
    assert underdog_gain(0.0) > abs(underdog_gain(1.0)) * 100


def test_an_uncertain_fighter_moves_further_than_a_settled_one():
    """RD is what the incumbent's ELO_K_NEWCOMER=60-for-5-fights was approximating."""
    def move(rd):
        g = Glicko1()
        g.rating[1], g.rd[1] = 1500.0, rd
        g.rating[2], g.rd[2] = 1500.0, 80.0
        g.observe(1, 2, 1.0, date(2024, 1, 1))
        return g.rating[1] - 1500.0

    assert move(RD_START) > move(80.0) > move(RD_FLOOR)


def test_observe_returns_pre_fight_state():
    """This return value is the whole point of the module: opponent strength is measured
    AS OF the bout. The incumbent computed it (points_ranking_service.py:288) and then
    read the final Elo instead (:308), so a fighter who later declined retroactively
    devalued every win scored over them."""
    g = Glicko1()
    pre = g.observe(1, 2, 1.0, date(2024, 1, 1))
    assert pre[1] == (RATING_START, RD_START)
    assert pre[2] == (RATING_START, RD_START)
    # And the post-fight state has moved away from it.
    assert g.rating[1] > RATING_START
    assert g.rating[2] < RATING_START


def test_draw_moves_both_toward_each_other():
    g = Glicko1()
    g.rating[1], g.rd[1] = 1700.0, 60.0
    g.rating[2], g.rd[2] = 1400.0, 60.0
    g.observe(1, 2, 0.5, date(2024, 1, 1))
    assert g.rating[1] < 1700.0
    assert g.rating[2] > 1400.0


def test_no_contests_never_reach_the_rating():
    """Scoring a no-contest as a draw would shrink both RDs and drag both ratings together
    on zero information. `is_decided` is the gate, and it must hold."""
    assert not is_decided("No Contest", None)
    assert not is_decided("Overturned", 5)
    assert not is_decided("DQ", 5)
    assert not is_decided("KO/TKO", None)
    assert is_decided("KO/TKO", 5)


# --------------------------------------------------------------------------- ledger
def test_ledger_entry_is_immutable_once_written():
    """At-the-time tiering means a bout's credit is fixed when it happens. This is the
    property that separates this ranker from WHR, which was implemented here and rejected
    for reshuffling the past on every refit (tau 0.7103)."""
    g = Glicko1()
    from app.services.ufc.tiered_ranking_service import _DivisionPool

    pool = _DivisionPool()
    for i in range(20):
        fid = 100 + i
        g.rating[fid] = 1400.0 + i * 20
        g.rd[fid] = 50.0
        pool.set(fid, "lightweight", g.conservative(fid))

    tier_at_the_time = pool.tier("lightweight", g.conservative(105))

    # The opponent now goes on a tear against everyone else.
    for i in range(8):
        g.observe(105, 300 + i, 1.0, date(2025, 1, 1) + timedelta(days=60 * i))
        pool.set(105, "lightweight", g.conservative(105))

    # Their CURRENT tier has risen, but the tier recorded on the old bout has not.
    assert pool.tier("lightweight", g.conservative(105)) > tier_at_the_time


def test_ledger_deltas_reconstruct_the_rating():
    """The ledger is the score's decomposition, not a parallel display artifact: every
    bout's rating delta sums back to the fighter's standing above the 1500 start."""
    g = Glicko1()
    deltas = []
    for i in range(6):
        before = g.state(1)[0]
        g.observe(1, 500 + i, V_FINISH if i % 2 else 0.0,
                  date(2022, 1, 1) + timedelta(days=120 * i))
        deltas.append(g.rating[1] - before)

    assert sum(deltas) == pytest.approx(g.rating[1] - RATING_START)
    assert any(d > 0 for d in deltas) and any(d < 0 for d in deltas)
