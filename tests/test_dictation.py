"""Unit tests for app/services/ufc/dictation.py (no database)."""
import numpy as np
import pytest

from app.services.ufc.dictation import (
    MechParams, MechRound, Mechanics, TugOfWar, TugParams, occupancy, pull,
)


def test_pull_endpoints_and_clip():
    assert pull(2.0, 2.0, 0.0) == 1.0
    assert pull(0.0, 2.0, 0.0) == 0.0
    assert pull(1.0, 2.0, 0.0) == 0.5
    assert pull(10.0, 2.0, 0.0) == 1.5
    assert pull(1.0, 1.0, 1.0) is None


def test_symmetric_splits_the_difference():
    m = TugOfWar(0.0, TugParams(learn_s=False))
    m._get("a").p, m._get("b").p = 2.0, -2.0
    assert m.weight("a", "b", 1) == 0.5
    assert m.predict("a", "b", 1) == pytest.approx(0.0)


def test_negative_direction_rule_goes_to_the_lower_preference():
    m = TugOfWar(0.0, TugParams(learn_s=False, h=-12))
    m._get("a").p, m._get("b").p = 2.0, -2.0
    assert m.predict("a", "b", 1) == pytest.approx(-2.0, abs=1e-3)
    assert m.predict("b", "a", 1) == pytest.approx(-2.0, abs=1e-3)


def test_skill_moves_toward_the_fighter_who_got_their_way():
    m = TugOfWar(0.0, TugParams(k_s=0.5, p_prior=1e9))   # freeze preferences
    m._get("a").p, m._get("b").p = 2.0, -2.0
    for _ in range(20):
        m.update("a", "b", [(1, 1.8, 1.0), (2, 1.8, 1.0)])
    assert m.snapshot("a")[1] > 0 > m.snapshot("b")[1]
    assert m.predict("a", "b", 1) > 1.0


def test_mirror_matchup_does_not_move_skill():
    m = TugOfWar(0.0, TugParams(k_s=1.0))
    m.update("a", "b", [(1, 3.0, 1.0)])     # both start at the league preference
    assert m.snapshot("a")[1] == 0.0 and m.snapshot("b")[1] == 0.0


def test_occupancy_no_entries_and_symmetry():
    assert occupancy(0.0, 1.0, 0.0, 1.0, 300) == pytest.approx((0.0, 0.0))
    a, b = occupancy(0.01, 1 / 60, 0.01, 1 / 60, 300)
    assert a == pytest.approx(b) and 0 < a < 0.5


def test_occupancy_longer_spells_mean_more_control():
    short = occupancy(0.01, 1 / 20, 0.0, 1.0, 300)[0]
    long = occupancy(0.01, 1 / 120, 0.0, 1.0, 300)[0]
    assert long > short


def test_escape_skill_learns_short_spells():
    m = Mechanics(MechParams(lr=0.05, l2=0.0))
    # wrestler lands 3 takedowns a round but only gets 30 s of control: the defender escapes
    for _ in range(30):
        m.update("w", "esc", MechRound(300, 5, 3, 30, 0))
        m.update("w", "stuck", MechRound(300, 5, 3, 240, 0))
    assert m.st["esc"]["esc"] > m.st["stuck"]["esc"]
    assert m.predict_share("w", "esc")[0] < m.predict_share("w", "stuck")[0]
    m_no = Mechanics(MechParams(lr=0.05, l2=0.0, use_escape=False))
    m_no.update("w", "esc", MechRound(300, 5, 3, 30, 0))
    assert m_no.st["esc"]["esc"] == 0.0
