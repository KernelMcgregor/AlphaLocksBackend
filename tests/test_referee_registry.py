"""Referee registry and decision-review helpers in app/services/ufc/officials_service.py."""
import json

import numpy as np
import pandas as pd
import pytest

from app.services.ufc.officials_service import (
    AXES, LEANS, decision_reviews, judge_pairs, norm_name, referee_tables, stable_ids,
    stoppage_expectations, style_map,
)


@pytest.mark.parametrize("raw,norm", [
    ("Herb Dean", "herb dean"),
    ("  HERB   DEAN ", "herb dean"),
    ("Mário Yamasaki", "mario yamasaki"),
    ("Big John McCarthy", "john mccarthy"),
    ("Dan Miragliotta.", "dan miragliotta"),
    (None, ""),
])
def test_norm_name(raw, norm):
    assert norm_name(raw) == norm


def test_stable_ids_keep_existing_and_extend():
    ids = stable_ids({"a": 3, "b": 9}, ["c", "a", "d", "b"])
    assert ids == {"a": 3, "b": 9, "c": 10, "d": 11}
    assert stable_ids({}, ["x", "y"]) == {"x": 1, "y": 2}


def test_judge_pairs_symmetric_counts():
    r = pd.DataFrame(dict(
        fight_id=[1, 1, 1, 2, 2, 2], round=1, judge_id=[1, 2, 3, 1, 2, 3],
        s=[1, 1, -1, -1, -1, -1], p=0.5))
    out = judge_pairs(r, 0.0, min_rounds=1).set_index(["judge_a", "judge_b"])
    assert (out.index.get_level_values(0) < out.index.get_level_values(1)).all()
    assert out.loc[(1, 2)].agree == 1.0 and out.loc[(1, 3)].agree == 0.5
    assert out.loc[(1, 2)].expected == pytest.approx(0.5)
    assert out.n_rounds.tolist() == [2, 2, 2]


def test_style_map_centres_average_judges():
    rng = np.random.default_rng(0)
    cols = LEANS + AXES
    prof = pd.DataFrame(rng.normal(0, 0.1, (12, len(cols))), columns=cols)
    prof.loc[11, cols] = 0.0
    prof["eligible"] = [True] * 10 + [False] * 2
    m = style_map(prof)
    assert m.shape == (12, 2) and m.notna().all().all()
    assert abs(m.loc[11, "map_x"]) < 1e-9          # an all-average judge sits at the origin


def _data(swapped):
    fights = pd.DataFrame([dict(
        fight_id=1, date="2024-01-01", event_name="UFC X", weight_class="Lightweight Bout",
        red=10, blue=11, winner_id=10, method="Decision - Split", details="", referee="Herb Dean",
        finish_round=3, finish_time="5:00", red_name="Red", blue_name="Blue")])
    return dict(
        fights=fights,
        dtw=pd.DataFrame([dict(fight_id=1, round_p_red="[0.5]", p_red=0.2, p_blue=0.8,
                               robbery_score=0.8, official_outcome="red:sd")]),
        decisions=pd.DataFrame([dict(mmad_decision_id=1, fight_id=1, swapped=swapped, fan_a=6,
                                     fan_b=2, fan_draw=2, referee="Herb Dean")]),
        media=pd.DataFrame([dict(fight_id=1, swapped=swapped, pick=p) for p in "aab"]),
    )


@pytest.mark.parametrize("swapped", [False, True])
def test_decision_review_corners(swapped):
    cards = pd.DataFrame([dict(judge_id=j, judge=f"J{j}") for j in (1, 2, 3)])
    fc = pd.DataFrame([dict(fight_id=1, judge_seq=j, judge_id=j, red_pts=rp, blue_pts=bp, w=w)
                       for j, rp, bp, w in ((1, 29, 28, 1), (2, 29, 28, 1), (3, 28, 29, -1))])
    out = decision_reviews(cards, fc, _data(swapped)).iloc[0]
    a_is_red = not swapped
    assert out.fan_red == pytest.approx(0.6 if a_is_red else 0.2)
    assert out.media_red == (2 if a_is_red else 1)
    assert out.kind == "sd" and out.result == 1 and out.robbery_score == 0.8
    assert [c["side"] for c in json.loads(out.cards)] == [1, 1, -1]


def test_referee_tables_rates_and_stoppage():
    rows = []
    for i in range(12):
        method = "KO/TKO" if i < 6 else "Decision - Unanimous"
        rows.append(dict(fight_id=i, date="2024-01-01", event_name="E", weight_class="LW",
                         red=1, blue=2, winner_id=1, method=method, details="", referee="Herb Dean",
                         finish_round=1 if method == "KO/TKO" else 3, finish_time="2:00",
                         red_name="R", blue_name="B"))
    fights = pd.DataFrame(rows)
    stats = pd.DataFrame([dict(fight_id=i, round=1, fighter_id=1, sig_str_landed=5 + i, kd=0)
                          for i in range(6)])
    cards = pd.DataFrame(columns=["fight_id", "round", "red_ded", "blue_ded"])
    regional = pd.DataFrame([dict(referee="Herb Dean", promotion="Bellator", n=4),
                             dict(referee="Herb Dean", promotion="UFC", n=99)])
    refs, rf = referee_tables(dict(fights=fights, stats=stats, cards=cards, regional=regional))
    r = refs.iloc[0]
    assert r.n_fights == 12 and r.n_regional == 4
    assert r.finish_rate == pytest.approx(0.5)
    assert r.stoppage_index == pytest.approx(0.0, abs=1e-6)   # the only referee: no gap to expected
    assert json.loads(r.detail)["mean_finish_seconds"] == 120
    assert rf.absorbed.notna().sum() == 6


def _stoppages(n=600, seed=0, chin_effect=0.0):
    rng = np.random.default_rng(seed)
    sec = rng.integers(10, 300, n)
    chin = rng.normal(50, 10, n)
    y = 0.3 + 0.5 * np.log1p(sec) + chin_effect * (chin - 50) / 10 + rng.normal(0, 0.3, n)
    fr = pd.DataFrame(dict(
        fight_id=np.arange(n), date="2020-01-01", red=1, blue=2, winner=1, winner_fid=1,
        finish_round=1, finish_time=[f"{s // 60}:{s % 60:02d}" for s in sec],
        weight_class="Lightweight Bout", kd_round=0, absorbed=np.expm1(y).round()))
    glicko = pd.DataFrame(
        [dict(fight_id=i, fighter_id=2, ko=40.0, kod=c, durability=30.0) for i, c in enumerate(chin)]
        + [dict(fight_id=i, fighter_id=1, ko=45.0, kod=50.0, durability=30.0) for i in range(n)])
    return fr, sec, glicko


def test_stoppage_model_removes_time_in_round():
    fr, sec, glicko = _stoppages()
    out = stoppage_expectations(fr, glicko)
    assert out.attrs["coef"]["log_sec"] == pytest.approx(0.5, abs=0.08)
    # What's left over no longer depends on how far into the round the stop came.
    assert abs(np.corrcoef(out.absorbed_z, np.log1p(sec))[0, 1]) < 0.05
    assert (out.absorbed_expected > 0).all()


def test_stoppage_model_uses_prefight_chin():
    fr, _, glicko = _stoppages(chin_effect=0.2, seed=1)
    out = stoppage_expectations(fr, glicko)
    assert out.attrs["coef"]["z_l_kod"] == pytest.approx(0.2, abs=0.05)


def test_stoppage_model_treats_zero_snapshots_as_missing():
    fr, _, glicko = _stoppages(seed=2)
    glicko.loc[glicko.fighter_id == 2, ["ko", "kod", "durability"]] = 0.0
    out = stoppage_expectations(fr, glicko)
    assert (out.m_l_kod == 1).all() and (out.z_l_kod == 0).all()
