"""Regression tests for the dimension drift that made the radar charts unreadable.

Measured over 21,312 production snapshots, every offence dimension rose with a fighter's
fight count and every paired defence dimension fell:

    ko +0.735 / kod -0.590,  sub +0.599 / subd -0.451,  td +0.505 / tdd -0.496

That antisymmetry is a bug signature, not a skill signal. On the site it showed as
Charles Oliveira (37-11) SUB 100 / SUB DEF 11 and Justin Gaethje (28-5) KO 100 / CHIN 29,
while one-fight newcomers showed CHIN 84-90 and TD DEF 80-99. The defence columns were
substantially an inverted experience counter.

Three independent causes, one test class each. Plus str_acc/str_def, which correlated
+0.93 with each other because they were two monotone functions of the same two numbers.

Run:  ./venv/bin/python -m pytest tests/test_glicko_drift.py -v
"""
from __future__ import annotations

import pytest

from app.services.ufc.glicko_service import (
    DIMENSIONS,
    GlickoParams,
    _centred_expectation,
)


class TestExpectationIsCentred:
    """`_compute_baselines` averages the SAME observable each update scores, so
    E[outcome] == baseline. The old `baseline * exp` gives baseline/2 at parity, leaving
    a residual every single round."""

    @pytest.mark.parametrize("baseline", [0.008, 0.026, 0.05, 0.09])
    def test_equal_fighters_expect_the_league_mean(self, baseline):
        """The whole fix in one assertion: at parity the expectation IS the mean, so
        the expected rating change is zero."""
        assert _centred_expectation(baseline, 0.5) == pytest.approx(baseline, rel=1e-9)

    @pytest.mark.parametrize("baseline", [0.008, 0.026, 0.05, 0.09])
    def test_the_old_form_was_biased_low_by_half(self, baseline):
        """Documents the defect: the superseded expression under-predicts by 2x, which
        is precisely the +K*baseline/2 per-round drift."""
        old = baseline * 0.5
        assert old == pytest.approx(_centred_expectation(baseline, 0.5) / 2, rel=1e-9)

    def test_expectation_rises_with_skill(self):
        b = 0.026
        assert (_centred_expectation(b, 0.2) < _centred_expectation(b, 0.5)
                < _centred_expectation(b, 0.8))

    @pytest.mark.parametrize("baseline", [0.001, 0.026, 0.3, 0.9])
    def test_expectation_stays_a_valid_probability(self, baseline):
        for exp in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert 0.0 <= _centred_expectation(baseline, exp) <= 1.0


class TestFinishBonusIsDisabled:
    """Making the bonus zero-sum was not enough. The transfer runs between DIFFERENT
    dimensions — winner credited on `ko`, loser debited on `kod` — so the pair total is
    conserved while `ko` inflates and `kod` deflates. Ablation over the full corpus:

        finish_bonus_k=5.0 -> ko +0.709  kod -0.591  sub +0.503  subd -0.346
        finish_bonus_k=0.0 -> ko +0.335  kod +0.163  sub +0.236  subd +0.005
    """

    def test_default_is_off(self):
        assert GlickoParams().finish_bonus_k == 0.0, (
            "the finish bonus is the entire offence/defence antisymmetry; finishing "
            "ability is already carried by the per-round kd and sub_att observables")

    def test_it_remains_tunable_for_ablation(self):
        assert GlickoParams(finish_bonus_k=5.0).finish_bonus_k == 5.0


class TestBaselinesMatchTheirObservables:
    """A baseline must be the mean of the quantity it is subtracted from. The td rate
    branch scored `td_landed*15/standing_min/cap` against `bl["td"]`, which is the mean
    of `min(td_landed,5)/5` — a different quantity on a different scale, so td drifted
    even independently of the centring bug."""

    def test_the_rate_branch_has_its_own_baseline(self):
        import inspect

        from app.services.ufc import glicko_service as gs

        src = inspect.getsource(gs._run_glicko)
        assert 'bl["td15s_mean"]' in src, (
            "the td rate branch must be centred on the mean of the capped rate it "
            "observes, not on the mean of the count fallback")

    def test_the_accuracy_baseline_exists(self):
        import inspect

        from app.services.ufc import glicko_service as gs

        assert 'bl["sig_acc"]' in inspect.getsource(gs._run_glicko)

    def test_td15s_mean_includes_rounds_with_no_takedown(self):
        """`wc_td15s_vals` skips zero-takedown rounds so it can produce a percentile cap
        but NOT the mean of the observable; a separate accumulator is required."""
        import inspect

        from app.services.ufc import glicko_service as gs

        src = inspect.getsource(gs._compute_baselines)
        assert "wc_td15s_all" in src


class TestStrAccAndStrDefAreDistinct:
    """`red_acc_share` and `red_def_share` were two monotone functions of the same
    (red_acc, blue_acc) pair, so red_acc_share > 0.5 IFF red_def_share > 0.5, always.
    Measured correlation +0.93: two of fifteen dimensions were one dimension."""

    def test_they_are_cross_paired_like_the_other_offence_defence_pairs(self):
        import inspect

        from app.services.ufc import glicko_service as gs

        src = inspect.getsource(gs._run_glicko)
        assert '_elo_expected(r_mu("str_acc"), b_mu("str_def"))' in src, (
            "str_acc must be scored against the opponent's str_def, as ko/sub/td are")
        assert '_elo_expected(r_mu("str_acc"), b_mu("str_acc"))' not in src, (
            "self-pairing str_acc is what made it a duplicate of str_def")

    def test_the_algebraically_dual_shares_are_gone(self):
        import inspect

        from app.services.ufc import glicko_service as gs

        # Comments describing the old behaviour are expected; only executable lines
        # should be free of it.
        code = "\n".join(
            line for line in inspect.getsource(gs._run_glicko).splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "red_def_share" not in code


class TestSosTransferConserves:
    def test_the_loser_is_debited_what_the_winner_gains(self):
        """It is a TRANSFER. The winner was credited here while the loser was only
        debited by a separate percentage penalty that touches solely dimensions where
        mu > 0, so every decided fight created net rating — the source of pts +0.504."""
        import inspect

        from app.services.ufc import glicko_service as gs

        src = inspect.getsource(gs._run_glicko)
        assert '_update_rating(ratings, loser_id, d, -bonus' in src


class TestDimensionSetIsStable:
    """Renaming or reordering these breaks model.is_glicko_feature, simulator's
    hard-coded tuples, the corner-swap prefix contract, the snapshot columns, and four
    frontend pages. The drift fixes deliberately changed only SEMANTICS, never names."""

    def test_there_are_still_fifteen_in_this_order(self):
        assert DIMENSIONS == [
            "pts", "ko", "kod", "sub", "subd",
            "td", "tdd", "ctrl",
            "str_vol", "str_acc", "str_def",
            "dist", "clinch", "gnd",
            "durability",
        ]
