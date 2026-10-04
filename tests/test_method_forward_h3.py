"""H3 (favourite by decision) in scripts/method_forward_track: decision and report."""

import pytest

from scripts import method_forward_track as mft

OPEN = {"prices": {"red_ko": 0.25, "red_sub": 0.10, "red_dec": 0.30,
                   "blue_ko": 0.15, "blue_sub": 0.05, "blue_dec": 0.15},
        "american": {"red_dec": 250, "blue_dec": 500}}


def test_bets_favourite_by_decision_when_ev_clears_3pct():
    h = mft.h3_decision({"red_dec": 0.36, "blue_dec": 0.10}, OPEN)
    assert h["fav"] == "red" and h["cell"] == "red_dec"
    # blend of 0.36 and 0.30 in log-odds ~0.33; at +250 EV ~ +15%
    assert 0.32 < h["p"] < 0.34 and h["bet"]


def test_no_bet_below_threshold_and_missing_inputs():
    assert not mft.h3_decision({"red_dec": 0.25, "blue_dec": 0.1}, OPEN)["bet"]
    assert mft.h3_decision({"red_dec": 0.36}, None) is None
    no_price = {"prices": OPEN["prices"], "american": {}}
    assert mft.h3_decision({"red_dec": 0.36}, no_price) is None
    partial = {"prices": {"red_dec": 0.3}, "american": {"red_dec": 250}}
    assert mft.h3_decision({"red_dec": 0.36}, partial) is None


def test_favourite_from_market_cells():
    o = {"prices": {**OPEN["prices"], "red_ko": 0.05, "blue_ko": 0.35},   # red 45%, blue 55%
         "american": {"blue_dec": 500, "red_dec": 250}}
    assert mft.h3_decision({"red_dec": 0.3, "blue_dec": 0.2}, o)["fav"] == "blue"


def test_report_roi(capsys):
    def row(win_cls, close):
        return {"h3": {"bet": True, "fav": "red", "cell": "red_dec", "american": 200, "q_open": 0.30},
                "result": {"winner": "red", "class": win_cls}, "close": {"prices": {"red_dec": close}}}
    mft.h3_report([row("dec", 0.33), row("ko", 0.31)])
    out = capsys.readouterr().out
    assert "2 bets" in out and "ROI +50.0%" in out and "CLV +2.00 pts" in out
    mft.h3_report([{"h3": {"bet": False}, "result": {"winner": "red", "class": "dec"}}])
    assert capsys.readouterr().out == ""
