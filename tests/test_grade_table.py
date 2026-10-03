"""Grade table: bucketing, thresholds, monotone grades, family cap, cross-market behaviour."""
import numpy as np
import pytest

from app.services.ufc.grading import (
    GRADE_ORDER, bucket_of, family_table, grade, letter, load_table,
)


def test_thresholds_and_buckets():
    assert letter(0.09) == "A+" and letter(0.04) == "A-" and letter(0.0) == "C+" and letter(-0.2) == "F"
    assert bucket_of(0.0) == 0 and bucket_of(0.05) == 1 and bucket_of(0.10) == 2 and bucket_of(0.5) == 3


def test_grade_is_the_banded_expected_roi():
    rng = np.random.default_rng(0)
    # a market that loses overall, with one tiny band that got lucky
    ev = np.r_[np.full(400, 0.01), np.full(17, 0.20)]
    profit = np.r_[np.where(rng.random(400) < 0.42, 1.0, -1.0), np.where(np.arange(17) < 13, 1.0, -1.0)]
    t = family_table(ev, profit, profit, np.full(len(ev), 2.0))
    lucky = t["buckets"][3]
    assert lucky["roi"] > 0.4                          # raw: a fluke +50%
    assert lucky["expected_roi"] < 0.01                # estimate: pulled back to the market
    assert lucky["grade"] == letter(lucky["expected_roi"])
    # a large consistent winning band keeps most of its size
    ev2 = np.full(1500, 0.05)
    profit2 = np.where(rng.random(1500) < 0.56, 1.0, -1.0)
    t2 = family_table(ev2, profit2, profit2, np.full(1500, 2.0))
    b = t2["buckets"][1]
    assert b["expected_roi"] > 0.5 * b["roi"] > 0


def test_no_pick_and_unrated():
    assert grade("winner_open", -0.01)[0] == "—"
    assert grade("not_a_family", 0.1, {"families": {}})[0] == "NR"


@pytest.mark.skipif(load_table() is None, reason="grade table not built")
def test_moneyline_open_beats_ko_cell_at_same_edge():
    """The owner's example: a 10% edge on the moneyline (opener) grades well above a 10% edge
    on a KO cell, because that band's ROI has been far better."""
    g_ml, _ = grade("winner_open", 0.10)
    for fam in ("sixway_ko_fav", "sixway_ko_dog"):
        g_ko, _ = grade(fam, 0.10)
        assert GRADE_ORDER.index(g_ko) - GRADE_ORDER.index(g_ml) >= 3
