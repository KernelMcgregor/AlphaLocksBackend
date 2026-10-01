"""Picks v2 pure logic: EV, pick side, no-pick rule, badges."""
from app.services.ufc.picks_v2 import american_from_decimal, choose_pick, evaluate_side, graded

TABLE = {"families": {"fam": {"buckets": [
    {"lo": 0.0, "hi": 0.02, "n": 100, "grade": "C", "roi": -0.01, "small_sample": False},
    {"lo": 0.02, "hi": 0.05, "n": 100, "grade": "B", "roi": 0.02, "small_sample": False},
    {"lo": 0.05, "hi": 0.10, "n": 100, "grade": "A-", "roi": 0.04, "small_sample": False},
    {"lo": 0.10, "hi": 0.20, "n": 100, "grade": "A", "roi": 0.06, "small_sample": False},
    {"lo": 0.20, "hi": None, "n": 100, "grade": "A+", "roi": 0.09, "small_sample": False}]}}}


def test_ev_and_conversions():
    e = evaluate_side(0.5, 120, 110)
    assert abs(e["ev_best"] - 0.10) < 1e-9 and abs(e["ev_median"] - 0.05) < 1e-9
    assert american_from_decimal(2.5) == 150 and american_from_decimal(1.5) == -200


def test_pick_is_max_positive_ev_or_none():
    sides = [{"side": "over", "p": 0.55, "best": -110, "med": -115},
             {"side": "under", "p": 0.45, "best": 105, "med": 100}]
    assert choose_pick(sides)["side"] == "over"
    assert choose_pick([{"side": "x", "p": 0.4, "best": 120, "med": 110}]) is None


def test_badges_never_change_the_grade():
    g, _, badges = graded("fam", {"p": 0.5, "best": 200, "med": 110}, None, None, TABLE)
    assert "outlier price" in badges and g == "A+"           # EV at best price = +50%
    g2, _, badges2 = graded("fam", {"p": 0.5, "best": 120, "med": 118}, 0.45, 0.48, TABLE)
    assert "line moved away" in badges2 and g2 == "A"        # EV +10% band, untouched
    _, _, b3 = graded("fam", {"p": 0.2, "best": 600, "med": 600}, None, None, TABLE)
    assert "high variance" in b3
