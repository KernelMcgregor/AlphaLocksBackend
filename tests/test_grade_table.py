"""Grade table: bucketing, thresholds, monotone grades, family cap, cross-market behaviour."""
import numpy as np
import pytest

from app.services.ufc.grading import (
    GRADE_ORDER, bucket_of, family_table, grade, letter, load_table,
)


def test_thresholds_and_buckets():
    assert letter(0.09) == "A+" and letter(0.04) == "A-" and letter(0.0) == "C+" and letter(-0.2) == "F"
    assert bucket_of(0.0) == 0 and bucket_of(0.03) == 1 and bucket_of(0.5) == 4


def test_grade_is_the_band_roi():
    ev = np.array([0.01] * 60 + [0.12] * 60)
    profit = np.array([-1.0] * 60 + [1.0] * 40 + [-1.0] * 20)     # band 0: -100%, band 3: +33%
    t = family_table(ev, profit, profit, np.full(120, 2.0))
    b = t["buckets"]
    assert b[0]["roi"] == -1.0 and b[0]["grade"] == "F"
    assert abs(b[3]["roi"] - 1 / 3) < 1e-9 and b[3]["grade"] == "A+"
    assert b[1]["n"] == 0 and b[1]["grade"] is None                 # empty band: not rated
    assert grade("fam", 0.03, {"families": {"fam": t}})[0] == "NR"


def test_no_pick_and_unrated():
    assert grade("winner_open", -0.01)[0] == "—"
    assert grade("not_a_family", 0.1, {"families": {}})[0] == "NR"


@pytest.mark.skipif(load_table() is None, reason="grade table not built")
def test_moneyline_open_beats_ko_cell_at_same_edge():
    """The owner's example: a 10% edge on the moneyline (opener) grades well above a 10% edge
    on a KO cell, because that band's ROI has been far better."""
    g_ml, _ = grade("winner_open", 0.10)
    g_ko, _ = grade("sixway_ko", 0.10)
    assert GRADE_ORDER.index(g_ko) - GRADE_ORDER.index(g_ml) >= 3
