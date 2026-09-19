"""Regression tests for the rank=0 rows that reached production.

38 rows across 10 divisions were served with rank=0, which sorts ahead of rank 1 and so
appeared at the TOP of every division. Two independent causes, both tested here:

  1. Glicko and Points disagreed about who was ACTIVE, because Points skipped bouts with
     no winner_id and Glicko did not.
  2. They disagreed about which DIVISION a fighter belonged to, for the same reason.

Pure-function tests over synthetic data — no DB, no artifacts.

Run:  ./venv/bin/python -m pytest tests/test_ranking_integrity.py -v
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from app.services.ufc.fighter_registry import (
    Eligibility,
    FighterState,
    build_fighter_registry,
    classify_weight_class,
    is_decided,
    is_rankable,
)
from app.services.ufc.ranking_publisher import (
    RankingIntegrityError,
    RankingResult,
    _check,
)

TODAY = date(2026, 9, 6)


# --------------------------------------------------------------------------- fakes
@dataclass
class FakeFight:
    id: int
    date: date
    red_fighter_id: int
    blue_fighter_id: int
    winner_id: int | None
    weight_class: str
    method: str | None = "Decision - Unanimous"


class _Query:
    def __init__(self, rows): self._rows = rows
    def order_by(self, *_): return self
    def filter(self, *_): return self
    def all(self): return self._rows


class FakeDB:
    """Just enough of a Session for build_fighter_registry."""

    def __init__(self, fights, round_rows):
        self.fights, self.round_rows = fights, round_rows

    def query(self, *entities):
        # build_fighter_registry queries UFCFight, then (fighter_id, fight_id) columns.
        return _Query(self.fights if len(entities) == 1 and hasattr(entities[0], "__tablename__")
                      else self.round_rows)


def _rounds(fight_id: int, fighters: tuple[int, ...], n: int):
    return [(f, fight_id) for f in fighters for _ in range(n)]


class TestNoContestIsNotInactivity:
    """Tom Aspinall: his 2025-10-25 title fight was waved off with no winner. Points read
    his last fight as 2024-07-27 — 771 days, past the 548-day cutoff — and dropped him,
    while Glicko counted the bout and kept him. Result: a rank=0 heavyweight row."""

    def _aspinall_registry(self):
        fights = [
            FakeFight(1, date(2024, 7, 27), 10, 11, 10, "UFC Interim Heavyweight Title Bout"),
            FakeFight(2, date(2023, 9, 2), 10, 12, 10, "Heavyweight Bout"),
            FakeFight(3, date(2023, 1, 14), 10, 13, 10, "Heavyweight Bout"),
            # No winner, no method: the bout that the two services disagreed about.
            FakeFight(4, date(2025, 10, 25), 10, 14, None, "UFC Heavyweight Title Bout",
                      method="Could Not Continue"),
        ]
        rows = (_rounds(1, (10, 11), 4) + _rounds(2, (10, 12), 3)
                + _rounds(3, (10, 13), 3) + _rounds(4, (10, 14), 1))
        return build_fighter_registry(FakeDB(fights, rows))

    def test_last_activity_counts_the_no_contest(self):
        st = self._aspinall_registry()[10]
        assert st.last_activity == date(2025, 10, 25), (
            "activity must come from every bout; walking out to the cage is not a layoff")
        assert st.last_decided == date(2024, 7, 27)

    def test_the_no_contest_does_not_count_as_a_decided_fight(self):
        st = self._aspinall_registry()[10]
        assert st.decided_fights == 3, "a waved-off bout carries no result"

    def test_he_is_rankable(self):
        """The whole point: 316 days since activity, not 771."""
        st = self._aspinall_registry()[10]
        assert is_rankable(st, TODAY), (
            "counting only decided bouts put him past the inactivity cutoff, which is "
            "what left his row at rank=0")


class TestDivisionFollowsTheLatestBout:
    """Ode Osbourne: last decided bout at Bantamweight, last bout at Flyweight. The two
    services filed him under different divisions, so both rows survived the delete.

    The invariant under test is that EVERY bout feeds the division, including the
    no-contest — not that the single latest bout wins. Which of the recent divisions is
    selected is `current_division`'s job and it now takes two bouts to complete a move.
    """

    def _registry(self):
        fights = [
            FakeFight(1, date(2024, 3, 16), 20, 21, 20, "Flyweight Bout"),
            FakeFight(2, date(2025, 8, 9), 20, 22, 20, "Bantamweight Bout"),
            FakeFight(3, date(2026, 7, 11), 20, 23, None, "Flyweight Bout", method=None),
        ]
        rows = _rounds(1, (20, 21), 3) + _rounds(2, (20, 22), 3) + _rounds(3, (20, 23), 4)
        return build_fighter_registry(FakeDB(fights, rows))

    def test_a_single_bout_back_at_flyweight_does_not_move_him(self):
        """One flyweight bout after a bantamweight one is not a division change; it takes
        two. Before the two-fight rule this asserted "flyweight" — the change is
        deliberate, and it is what keeps a one-off appearance from relocating a
        contender out of the division they actually compete in."""
        assert self._registry()[20].division == "bantamweight"

    def test_the_no_contest_still_feeds_the_division(self):
        """The original defect: a bout with no winner told Points nothing about division
        while Glicko counted it. It must still participate — a second flyweight bout
        completes the move even though the first of the two was waved off."""
        fights = [
            FakeFight(1, date(2024, 3, 16), 20, 21, 20, "Bantamweight Bout"),
            FakeFight(2, date(2026, 7, 11), 20, 23, None, "Flyweight Bout", method=None),
            FakeFight(3, date(2026, 8, 11), 20, 24, 20, "Flyweight Bout"),
        ]
        rows = _rounds(1, (20, 21), 3) + _rounds(2, (20, 23), 4) + _rounds(3, (20, 24), 3)
        assert build_fighter_registry(FakeDB(fights, rows))[20].division == "flyweight"

    def test_unknown_divisions_do_not_erase_a_known_one(self):
        """A catchweight must not blank out the division and make a fighter unrankable."""
        fights = [
            FakeFight(1, date(2025, 1, 1), 30, 31, 30, "Lightweight Bout"),
            FakeFight(2, date(2026, 1, 1), 30, 32, 30, "Catch Weight Bout"),
        ]
        rows = _rounds(1, (30, 31), 5) + _rounds(2, (30, 32), 5)
        assert build_fighter_registry(FakeDB(fights, rows))[30].division == "lightweight"


class TestDecidedClassification:
    def test_overturned_results_are_not_decided(self):
        assert not is_decided("Overturned", 5)

    def test_missing_winner_is_not_decided(self):
        assert not is_decided("Decision - Unanimous", None)

    @pytest.mark.parametrize("m", ["No Contest", "DQ", "Draw"])
    def test_undecided_markers(self, m):
        assert not is_decided(m, 5)

    def test_a_normal_decision_is_decided(self):
        assert is_decided("Decision - Unanimous", 7)


class TestEligibilityIsAnded:
    """Each service applied only ONE threshold, so each admitted fighters the other
    rejected — and every such fighter was a candidate rank=0 row.

    The fixtures below sit just under the CURRENT floor (2 decided bouts / 5 rounds). They
    were written against the old 3/10 floor and had to move with it; what is being tested
    is the AND, not the specific numbers.
    """

    def test_enough_fights_but_too_few_rounds_is_not_rankable(self):
        st = FighterState("lightweight", TODAY, decided_fights=4, rounds=3)
        assert not is_rankable(st, TODAY), "4 quick finishes is too little scored material"

    def test_enough_rounds_but_too_few_fights_is_not_rankable(self):
        st = FighterState("lightweight", TODAY, decided_fights=1, rounds=15)
        assert not is_rankable(st, TODAY)

    def test_unknown_division_is_never_rankable(self):
        st = FighterState("unknown", TODAY, decided_fights=9, rounds=30)
        assert not is_rankable(st, TODAY)

    def test_inactive_is_not_rankable(self):
        st = FighterState("lightweight", date(2024, 1, 1), decided_fights=9, rounds=30)
        assert not is_rankable(st, TODAY)


class TestPublisherRefusesBadRankings:
    """The publisher asserts before it commits, so these fail loudly in the pipeline
    instead of quietly on the site."""

    def _ok_state(self):
        return FighterState("lightweight", TODAY, decided_fights=9, rounds=30)

    def test_rejects_a_ranked_fighter_with_no_dimension_profile(self):
        """This is what produced radar charts with all 15 dimensions at 0."""
        reg = {1: self._ok_state(), 2: self._ok_state()}
        result = RankingResult(order={"lightweight": [1, 2]}, scores={1: 5.0, 2: 4.0})
        with pytest.raises(RankingIntegrityError, match="no dimension profile"):
            _check(result, {(1, "lightweight"): {"ko": 50}}, reg, TODAY, Eligibility())

    def test_rejects_a_fighter_ranked_in_the_wrong_division(self):
        reg = {1: self._ok_state(),
               2: FighterState("welterweight", TODAY, decided_fights=9, rounds=30)}
        profiles = {(1, "lightweight"): {}, (2, "lightweight"): {}}
        result = RankingResult(order={"lightweight": [1, 2]}, scores={1: 5.0, 2: 4.0})
        with pytest.raises(RankingIntegrityError, match="registry says"):
            _check(result, profiles, reg, TODAY, Eligibility())

    def test_rejects_an_unrankable_fighter(self):
        reg = {1: self._ok_state(),
               2: FighterState("lightweight", TODAY, decided_fights=1, rounds=2)}
        profiles = {(1, "lightweight"): {}, (2, "lightweight"): {}}
        result = RankingResult(order={"lightweight": [1, 2]}, scores={1: 5.0, 2: 4.0})
        with pytest.raises(RankingIntegrityError, match="not rankable"):
            _check(result, profiles, reg, TODAY, Eligibility())

    def test_rejects_duplicates(self):
        reg = {1: self._ok_state()}
        result = RankingResult(order={"lightweight": [1, 1]}, scores={1: 5.0})
        with pytest.raises(RankingIntegrityError, match="duplicate"):
            _check(result, {(1, "lightweight"): {}}, reg, TODAY, Eligibility())

    def test_accepts_a_well_formed_ranking(self):
        reg = {1: self._ok_state(), 2: self._ok_state()}
        profiles = {(1, "lightweight"): {}, (2, "lightweight"): {}}
        result = RankingResult(order={"lightweight": [1, 2]}, scores={1: 5.0, 2: 4.0})
        _check(result, profiles, reg, TODAY, Eligibility())  # must not raise


class TestServedRanksAreNeverZero:
    def test_get_rankings_filters_out_placeholders(self):
        """rank=0 rows must never reach the API even if the table somehow holds them."""
        import inspect

        from app.services.ufc import tapology_rankings as tr

        src = inspect.getsource(tr.get_rankings)
        assert "rank > 0" in src, (
            "get_rankings must filter placeholder rows; 0 sorts above 1 and lands them "
            "at the top of every division")


class TestWeightClassClassifierIsShared:
    def test_there_is_only_one_classifier(self):
        """Two copies of this function is how the division disagreement started. The
        ranker must not carry its own; it imports the registry's."""
        import inspect

        from app.services.ufc import tapology_rankings as tr

        assert tr.classify_weight_class is classify_weight_class
        assert "def classify_weight_class" not in inspect.getsource(tr)

    def test_womens_featherweight_pools_into_bantamweight(self):
        """Deliberate: a defunct two-fighter division, pooled identically by both
        engines. Asserted so it stays a decision rather than becoming a surprise."""
        assert classify_weight_class("Women's Featherweight Bout") == "w_bantamweight"
