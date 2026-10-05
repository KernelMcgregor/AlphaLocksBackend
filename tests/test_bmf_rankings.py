"""BMF ranking: pure-function tests over synthetic bouts. No database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_bmf_rankings.py
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.services.ufc import bmf_rankings as bmf
from app.services.ufc.alt_rankings_common import (
    BoutView, FighterView, quality_weight, recency_score, recency_weight, scale_quality,
)

TODAY = date(2026, 10, 5)
BASE = {"mu": 0.5, "g": 0.01}


def bout(i, result="W", outcome="ko", rnd=1, tier=5, kd=0, kd_abs=0, head_abs=40,
         days_ago=None):
    d = TODAY - timedelta(days=days_ago if days_ago is not None else 120 * (i + 1))
    return BoutView(
        fight_id=i, date=d, opponent_id=1000 + i, result=result, outcome=outcome,
        finish_round=rnd, division="lightweight", tier=tier, kd=kd, kd_abs=kd_abs,
        head_abs=head_abs, w=recency_weight((TODAY - d).days), q=quality_weight(tier),
    )


def fighter(fid, bouts, division="lightweight"):
    return FighterView(
        fighter_id=fid, name=f"F{fid}", division=division, pool="men", bouts=bouts,
        last_activity=bouts[0].date, bouts_last_730d=3,
    )


def comps(f, base=BASE):
    r = bmf.component_raws(f, base)
    return {
        "finishing": bmf.scale_finishing(r["finishing"]),
        "toughness": bmf.scale_toughness(r["toughness"]),
        "opp_quality": scale_quality(r["opp_quality"]),
        "recency": recency_score((TODAY - f.last_activity).days, f.bouts_last_730d),
    }


def test_ko_artist_over_weak_opposition_is_gated():
    can = fighter(1, [bout(i, tier=1, kd=1) for i in range(5)])
    proven = fighter(2, [bout(i, tier=8, kd=1) for i in range(3)]
                     + [bout(i, outcome="ud", tier=8) for i in range(3, 5)])
    a, b = comps(can), comps(proven)
    # Five straight KOs, but over tier-1 opposition: the gate caps the score however the
    # sliders set the base, and the proven finisher ranks higher at the defaults.
    assert bmf.gate(a["opp_quality"], 1.0) < 0.4
    assert bmf.combine(b) > bmf.combine(a)
    assert bmf.combine(b, {"gamma": 1}) - bmf.combine(a, {"gamma": 1}) \
        > bmf.combine(b) - bmf.combine(a)              # stricter gate, wider gap


def test_finishing_is_relative_to_division():
    f = fighter(1, [bout(i) for i in range(5)])
    light = bmf.component_raws(f, {"mu": 0.4, "g": 0.01})["finishing"]
    heavy = bmf.component_raws(f, {"mu": 0.8, "g": 0.01})["finishing"]
    assert light > heavy          # same record is less special where everyone finishes


def test_injury_stoppage_scores_nothing_doctor_counts():
    assert bmf.bout_finish_points(bout(0, outcome="injury")) == 0
    assert bmf.bout_finish_points(bout(0, outcome="doctor", rnd=3)) == pytest.approx(1.0)
    assert bmf.bout_finish_points(bout(0, outcome="ko", rnd=1)) == pytest.approx(1.3)
    assert bmf.bout_finish_points(bout(0, outcome="sub")) == pytest.approx(0.6)
    # Knockdowns count even in a loss, capped at 3.
    assert bmf.bout_finish_points(bout(0, result="L", outcome="ud", kd=5)) == pytest.approx(1.05)


def test_older_record_counts_less():
    recent = fighter(1, [bout(i, days_ago=100 + 100 * i) for i in range(5)])
    old = fighter(2, [bout(i, outcome="ud") for i in range(1)]
                  + [bout(i, days_ago=1500 + 100 * i) for i in range(1, 5)])
    assert comps(recent)["finishing"] > comps(old)["finishing"]
    vals = [recency_score(d, 1) for d in (0, 100, 365, 700)]
    assert vals == sorted(vals, reverse=True)


def test_chin_and_comebacks():
    clean = fighter(1, [bout(i, outcome="ud", head_abs=80) for i in range(5)])
    dropped = fighter(2, [bout(i, outcome="ud", kd_abs=1, head_abs=80, result="L")
                          for i in range(5)])
    ko_losses = fighter(3, [bout(i, outcome="ko", result="L", head_abs=80) for i in range(5)])
    t = lambda f: bmf.component_raws(f, BASE)["toughness"]
    assert t(clean) > t(dropped) > t(ko_losses)

    # Getting dropped and winning anyway beats getting dropped and losing.
    came_back = fighter(4, [bout(i, outcome="ud", kd_abs=1, head_abs=80) for i in range(5)])
    stopped = fighter(5, [bout(i, outcome="ko", kd_abs=1, head_abs=80, result="L")
                          for i in range(5)])
    assert t(came_back) > t(stopped)


def test_gate_off_and_bounds():
    c = {"finishing": 80, "toughness": 60, "recency": 40, "opp_quality": 10}
    base = (0.45 * 80 + 0.30 * 60 + 0.25 * 40) / 1.0
    assert bmf.combine(c, {"gamma": 0}) == pytest.approx(base)
    assert bmf.combine({**c, "opp_quality": 100}, {"gamma": 1}) == pytest.approx(base)
    for raw in (-5, -0.5, 0, 0.5, 5):
        assert 0 <= bmf.scale_toughness(raw) <= 100
    for raw in (0, 0.5, 1, 3, 50):
        assert 0 <= bmf.scale_finishing(raw) <= 100


def test_combine_matches_hand_computation():
    c = {"finishing": 70, "toughness": 50, "recency": 90, "opp_quality": 64}
    w = {"finishing": 1, "toughness": 1, "recency": 0, "gamma": 0.5}
    expected = 60 * (0.15 + 0.85 * 0.8)
    assert bmf.combine(c, w) == pytest.approx(expected)
