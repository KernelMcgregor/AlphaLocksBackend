"""Regression tests for fight-name matching against the prediction-market venues.

Every case here is a real failure observed while backfilling, not a hypothetical. Matching is the
highest-risk part of this ingestion: a miss loses a fight's prices quietly, and a *wrong* match
attributes one fight's market to another, which is worse because the result looks entirely
plausible on the page.

Pure functions only -- no DB, no network.

Run:  ./venv/bin/python -m pytest tests/test_prediction_market_matching.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.ufc.prediction_markets.common import (
    _fold, hint_agrees, name_matches, split_versus,
)
from app.services.ufc.prediction_markets.polymarket import _slug_hints, classify_market


class TestNameFolding:
    @pytest.mark.parametrize("raw,expected", [
        ("Édgar Cháirez", "edgar chairez"),      # Polymarket accents, ufcstats does not
        ("Morgan Charrière", "morgan charriere"),
        ("Sean King III", "sean king"),          # generational suffix becomes the last token
        ("Raul Rosas Jr.", "raul rosas"),
        ("Duško Todorovic", "dusko todorovic"),
    ])
    def test_fold(self, raw, expected):
        assert _fold(raw) == expected


class TestNameMatches:
    def test_full_name(self):
        assert name_matches("Jessie Rosas", "Jessie", "Rosas")

    @pytest.mark.parametrize("venue,first,last", [
        # Surname-only billing. The sportsbook matcher rejects all of these, because it compares
        # the surname against the fighter's first initial -- it was the single largest source of
        # misses, ~150 of 551 events on the first pass.
        ("Makhachev", "Islam", "Makhachev"),
        ("Della Maddalena", "Jack", "Della Maddalena"),   # multi-token surname
        ("Naurdiev", "Ismail", "Naurdiev"),
        ("Saint Denis", "Benoit", "Saint Denis"),
    ])
    def test_surname_only(self, venue, first, last):
        assert name_matches(venue, first, last)

    def test_suffix_mismatch(self):
        """ufcstats stores 'Sean King III'; both venues bill him as 'Sean King'."""
        assert name_matches("Sean King", "Sean", "King III")

    def test_accents(self):
        assert name_matches("Édgar Cháirez", "Edgar", "Chairez")

    @pytest.mark.parametrize("venue,first,last", [
        ("Makhachev", "Umar", "Nurmagomedov"),
        ("Jones", "Islam", "Makhachev"),
        ("Jessie Rosas", "Raul", "Rosas"),   # same surname, different fighter
    ])
    def test_rejects_wrong_fighter(self, venue, first, last):
        assert not name_matches(venue, first, last)

    def test_requires_a_surname(self):
        assert not name_matches("Makhachev", "Islam", None)


class TestHintAgrees:
    def test_prefix_of_first_name(self):
        assert hint_agrees("isl", "Islam")

    def test_hint_longer_than_first_name(self):
        """`ufc-rod1-bon-...` is Rodolfo Vieira vs **Bo** Nickal.

        The token runs past the whole first name, so a one-directional startswith reads it as a
        contradiction. When hints were applied as a filter this vetoed a correct match outright --
        which is also why they are now only ever used to break ties.
        """
        assert hint_agrees("bon", "Bo")

    def test_absent_hint_is_not_evidence(self):
        assert hint_agrees(None, "Islam")
        assert hint_agrees("isl", None)

    def test_disagrees(self):
        assert not hint_agrees("ale", "Ilia")   # Aleksandre vs Ilia Topuria


class TestSplitVersus:
    @pytest.mark.parametrize("title,expected", [
        ("Noche UFC: Jessie Rosas vs. Sean King (Featherweight, Prelims)",
         ("Jessie Rosas", "Sean King")),
        ("UFC 331: Joshua Van vs Alexandre Pantoja", ("Joshua Van", "Alexandre Pantoja")),
        ("Makhachev vs. Della Maddalena", ("Makhachev", "Della Maddalena")),
    ])
    def test_splits(self, title, expected):
        assert split_versus(title) == expected

    def test_returns_none_without_a_separator(self):
        assert split_versus("Who will Conor McGregor fight next?") is None


class TestSlugHints:
    def test_strips_polymarket_counter(self):
        assert _slug_hints("ufc-isl-jac9-2025-11-15") == ("isl", "jac")

    def test_non_fight_slug(self):
        assert _slug_hints("who-will-paddy-pimblett-fight-next") == (None, None)


class TestClassifyMarket:
    """The 19 Polymarket markets on a fight, as they actually arrive."""

    RED = SimpleNamespace(first_name="Jessie", last_name="Rosas")
    BLUE = SimpleNamespace(first_name="Sean", last_name="King III")

    @pytest.mark.parametrize("question,expected", [
        ("Noche UFC: Jessie Rosas vs. Sean King (Featherweight, Prelims)",
         ("moneyline", None, None)),
        ("Will the fight be won by KO or TKO?", ("method", "ko_tko", None)),
        ("Will the fight be won by submission?", ("method", "submission", None)),
        ("Will the fight be won by decision?", ("method", "decision", None)),
        # Distinct from the above: draw-inclusive. Collapsing the two onto one key leaves two rows
        # answering to the same name with no way to tell which is which.
        ("Will the fight end in a decision or draw?", ("method", "decision_draw", None)),
        ("Fight to Go the Distance?", ("distance", "distance", None)),
        ("O/U 2.5 Rounds", ("round_ou", "ou_2.5", None)),
        ("Will the fight end before Round 2?", ("round_ou", "before_r2", None)),
        ("Will Jessie Rosas win by KO or TKO?", ("fighter_method", "red_ko_tko", "red")),
        ("Will Sean King win by KO or TKO?", ("fighter_method", "blue_ko_tko", "blue")),
        ("Will Jessie Rosas win in Round 1?", ("fighter_round", "red_r1", "red")),
        ("Will Sean King win in Round 3?", ("fighter_round", "blue_r3", "blue")),
    ])
    def test_classification(self, question, expected):
        assert classify_market(question, self.RED, self.BLUE) == expected

    def test_corner_comes_from_the_matched_fighter_not_listing_order(self):
        """Props are attributed by identity, so the venue's listing order cannot flip them.

        This is the failure that would be invisible on the page: a KO prop attributed to the wrong
        fighter still renders as a plausible number.
        """
        swapped = classify_market(f"Will Sean King win by KO or TKO?", self.RED, self.BLUE)
        assert swapped == ("fighter_method", "blue_ko_tko", "blue")

    @pytest.mark.parametrize("question,expected_side", [
        # Billed by surname while the question uses the full name — this stranded 132 of 946
        # fighter-method markets with no corner when props were matched against the event title.
        ("Will Jessie Rosas win in Round 1?", "red"),
        ("Will Rosas win in Round 1?", "red"),
        # Suffix present in our record, absent from the venue's question.
        ("Will Sean King win in Round 1?", "blue"),
    ])
    def test_side_survives_surname_only_and_suffixes(self, question, expected_side):
        assert classify_market(question, self.RED, self.BLUE)[2] == expected_side

    def test_unknown_is_kept_not_dropped(self):
        """Polymarket adds market types; an unrecognised one must stay visible."""
        mt, key, side = classify_market("Will there be a point deduction?", self.RED, self.BLUE)
        assert mt == "unknown"
