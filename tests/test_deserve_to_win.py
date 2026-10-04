"""Deserve-to-win simulator: decision rules, symmetry, extrapolation, deductions."""
import numpy as np
import pytest

from app.services.ufc.deserve_to_win import (
    BLUE, DRAW, RED, FightInput, Form, decide, fit_sigma_u, form_posterior, judge_agreement,
    latent_scale, logit, sigmoid, simulate_fight,
)


@pytest.mark.parametrize("red,blue,out,kind", [
    ([30, 30, 30], [27, 27, 27], RED, 0),     # UD
    ([29, 29, 28], [28, 28, 29], RED, 1),     # SD
    ([29, 29, 28], [28, 28, 28], RED, 2),     # MD (2-0-1)
    ([28, 29, 28], [28, 28, 29], DRAW, 3),    # split draw 1-1-1
    ([29, 28, 28], [28, 28, 28], DRAW, 3),    # majority draw 1-0-2
    ([27, 28, 28], [30, 29, 30], BLUE, 0),
])
def test_decide(red, blue, out, kind):
    o, k = decide(np.array(red), np.array(blue))
    assert o == out and k == kind


def test_even_fight_is_coin_flip():
    r = simulate_fight(FightInput(np.zeros((1, 3)), scheduled=3), n=40_000, rng=1)
    assert abs(r.p_red - r.p_blue) < 0.02
    assert r.p_red + r.p_draw + r.p_blue == pytest.approx(1.0)


def test_dominant_fight_near_certain():
    r = simulate_fight(FightInput(np.full((1, 3), logit(0.99)), scheduled=3), n=5_000, rng=2)
    assert r.p_red > 0.99
    assert r.p_kind["ud"] > 0.95


def test_corner_symmetry():
    L = np.array([[1.2, -0.4, 0.3]])
    a = simulate_fight(FightInput(L, scheduled=3), n=50_000, rng=3)
    b = simulate_fight(FightInput(-L, scheduled=3), n=50_000, rng=3)
    assert a.p_red == pytest.approx(b.p_blue, abs=0.01)
    assert a.p_draw == pytest.approx(b.p_draw, abs=0.01)


def test_seeded_reproducible():
    fi = FightInput(np.array([[0.5, -0.2, 0.1]]), scheduled=3)
    assert simulate_fight(fi, n=2000, rng=7).p_red == simulate_fight(fi, n=2000, rng=7).p_red


def test_marginal_round_prob_preserved_with_latent():
    p = 0.8
    r = simulate_fight(FightInput(np.full((1, 1), logit(p)), scheduled=1), n=200_000,
                       sigma_u=1.5, rng=4)
    assert r.round_p_red[0] == pytest.approx(p, abs=0.02)


def test_latent_raises_agreement():
    p = np.full(500, 0.6)
    assert judge_agreement(p, 2.0, rng=0).mean() > judge_agreement(p, 0.0, rng=0).mean() + 0.1


def test_fit_sigma_u_recovers_truth():
    rng = np.random.default_rng(5)
    p = sigmoid(rng.normal(0, 2, 3000))
    truth = 1.2
    k = latent_scale(truth)
    u = rng.normal(0, truth, (len(p), 1))
    red = (k * logit(p)[:, None] + u + rng.logistic(size=(len(p), 3))) > 0
    unan = (red.sum(1) == 0) | (red.sum(1) == 3)
    s, _ = fit_sigma_u(p, unan)
    assert abs(s - truth) <= 0.4


def test_deduction_flips_close_fight():
    # Red wins every round 10-9 (no 10-8s), but 3 points deducted: 27-27 -> draws dominate.
    L = np.full((1, 3), logit(0.999))
    fi = FightInput(L, scheduled=3, ten8=np.zeros((1, 3)), red_ded=np.array([1, 1, 1]))
    r = simulate_fight(fi, n=5000, rng=6)
    assert r.p_draw > 0.9


def test_five_rounds_and_extrapolation():
    # One dominant round observed, four to extrapolate from a confident form.
    form = Form(mean=2.0, sd_mean=0.3, sd_round=1.0)
    r = simulate_fight(FightInput(np.array([[3.0]]), scheduled=5, form=form), n=20_000, rng=8)
    assert r.p_red > 0.85
    assert len(r.round_p_red) == 5
    with pytest.raises(ValueError):
        simulate_fight(FightInput(np.array([[3.0]]), scheduled=5), n=10)


def test_form_posterior_shrinks():
    f0 = form_posterior(np.array([]), prior_mean=0.5, tau=1.0, sigma=2.0)
    assert (f0.mean, f0.sd_mean) == (0.5, 1.0)
    f = form_posterior(np.array([3.0, 3.0]), prior_mean=0.0, tau=1.0, sigma=2.0)
    assert 0 < f.mean < 3.0 and f.sd_mean < 1.0


def test_parameter_draws_widen_uncertainty():
    # Two model fits disagree on a round; the fight mixes them.
    L = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    r = simulate_fight(FightInput(L, scheduled=3), n=40_000, rng=9)
    assert abs(r.p_red - r.p_blue) < 0.03


def test_partial_round_blends_with_form():
    # Round 1 barely fought (10%) and looked great for red; form says blue is better.
    form = Form(mean=-2.0, sd_mean=0.1, sd_round=0.5)
    fi = FightInput(np.array([[4.0]]), scheduled=1, partial_weight=0.1, form=form)
    assert simulate_fight(fi, n=20_000, rng=11).round_p_red[0] < 0.5


def test_ten8_depends_on_round_winner():
    # Red's round wins are always 10-8, blue's never: a sweep is 30-24 on every card.
    L = np.full((1, 3), logit(0.999))
    fi = FightInput(L, scheduled=3, ten8=np.ones((1, 3)), ten8_blue=np.zeros((1, 3)))
    r = simulate_fight(fi, n=5000, rng=12)
    assert r.top_cards[0]["cards"] == ["30-24", "30-24", "30-24"]
    fi = FightInput(-L, scheduled=3, ten8=np.ones((1, 3)), ten8_blue=np.zeros((1, 3)))
    assert simulate_fight(fi, n=5000, rng=12).top_cards[0]["cards"] == ["27-30"] * 3


def test_top_cards_sum_and_format():
    r = simulate_fight(FightInput(np.array([[1.0, 1.0, -1.0]]), scheduled=3), n=10_000, rng=10)
    assert 0 < sum(c["p"] for c in r.top_cards) <= 1.0
    assert all(len(c["cards"]) == 3 for c in r.top_cards)
