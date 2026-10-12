"""Hypothetical bouts: the per-request recomputation of opponent- and division-dependent
features, on a hand-built State. No database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_hypothetical.py
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from app.services.ufc import hypothetical as hyp

FIGHTER_COLS = ["elo", "elo_expected", "style_matchup_adv", "age", "xs_sig_for",
                "division_rank", "division_rank_delta", "moved_up", "moved_down",
                "changed_division", "fights_in_current_division", "is_catchweight"]


@pytest.fixture()
def state():
    rows = pd.DataFrame(
        {
            "elo": [1700.0, 1500.0, 1600.0],
            "elo_expected": [0.9, 0.9, 0.9],          # probe-pairing values, must be replaced
            "style_matchup_adv": [0.9, 0.9, 0.9],
            "age": [30.0, 28.0, 33.0],
            "xs_sig_for": [4.2, 3.1, 2.0],
            "division_rank": [13.0, 14.0, np.nan],
            "division_rank_delta": [0.0, 0.0, 0.0],
            "moved_up": [0.0, 0.0, 0.0],
            "moved_down": [0.0, 0.0, 0.0],
            "changed_division": [0.0, 0.0, 0.0],
            # The probe's own count is measured in its red fighter's division: never used.
            "fights_in_current_division": [99.0, 99.0, 99.0],
            "is_catchweight": [0.0, 0.0, 0.0],
            "fighter_style": [0, 1, 1],
            "last_division": ["lightweight", "welterweight", None],
            "n_ufc_fights": [10, 5, 0],
            "div_run": [6.0, 3.0, 0.0],
        },
        index=[11, 22, 33],
    )
    style = np.array([[0.5, 0.62], [0.38, 0.5]])
    return hyp.State(
        built_at=datetime.now(), as_of=date(2026, 10, 5), rows=rows,
        per=pd.DataFrame(index=[11, 22, 33]), fighter_cols=FIGHTER_COLS,
        raw_cols=["elo", "elo_expected", "age"],
        fight_cols=["div_lightweight", "div_welterweight", "is_title_fight", "is_five_round",
                    "scheduled_rounds", "scheduled_minutes"],
        style_adv=style,
    )


def test_opponent_features_recomputed(state):
    a, b = state.rows.loc[11], state.rows.loc[22]
    va = hyp._corner_values(state, a, b, "welterweight")
    vb = hyp._corner_values(state, b, a, "welterweight")
    assert va["elo_expected"] == pytest.approx(1 / (1 + 10 ** (-200 / 400)))
    assert va["elo_expected"] + vb["elo_expected"] == pytest.approx(1.0)
    assert va["style_matchup_adv"] == 0.62 and vb["style_matchup_adv"] == 0.38
    assert np.isnan(va["xs_sig_for"])           # matchup-specific: left to imputation


def test_division_movement(state):
    a, b = state.rows.loc[11], state.rows.loc[22]
    up = hyp._corner_values(state, a, b, "welterweight")      # lightweight moving up
    assert (up["moved_up"], up["moved_down"], up["changed_division"]) == (1.0, 0.0, 1.0)
    assert up["division_rank_delta"] == 1.0
    assert up["fights_in_current_division"] == 0.0
    home = hyp._corner_values(state, b, a, "welterweight")    # at home: count runs on
    assert (home["moved_up"], home["changed_division"]) == (0.0, 0.0)
    assert home["fights_in_current_division"] == 3.0
    cw = hyp._corner_values(state, b, a, "catchweight")
    assert cw["is_catchweight"] == 1.0 and cw["changed_division"] == 1.0
    assert cw["moved_up"] == 0.0 and cw["division_rank_delta"] == 0.0   # catchweight has no rank


def test_debut_has_no_movement(state):
    deb = hyp._corner_values(state, state.rows.loc[33], state.rows.loc[11], "lightweight")
    assert (deb["moved_up"], deb["moved_down"], deb["changed_division"]) == (0.0, 0.0, 0.0)
    assert deb["fights_in_current_division"] == 0.0


def test_frame_is_antisymmetric(state):
    f = hyp._frame(state, state.rows.loc[11], state.rows.loc[22], "welterweight", True, 5)
    g = hyp._frame(state, state.rows.loc[22], state.rows.loc[11], "welterweight", True, 5)
    assert f["diff_elo"].iloc[0] == -g["diff_elo"].iloc[0] == 200.0
    assert f["red_elo"].iloc[0] == g["blue_elo"].iloc[0]
    assert f["fight_div_welterweight"].iloc[0] == 1.0 and f["fight_div_lightweight"].iloc[0] == 0.0
    assert f["fight_is_title_fight"].iloc[0] == 1.0 and f["fight_scheduled_rounds"].iloc[0] == 5.0
    assert np.isnan(f["odds_red_prob"].iloc[0])


def test_unknown_fighter(state, monkeypatch):
    monkeypatch.setattr(hyp, "get_state", lambda: state)
    with pytest.raises(hyp.UnknownFighter):
        hyp._fighter(state, 99)


def test_defaults_and_names():
    assert hyp.default_division("lightweight", "welterweight") == "welterweight"
    assert hyp.default_division("w_flyweight", None) == "w_flyweight"
    assert hyp.default_division(None, "catchweight") == "catchweight"
    assert hyp.bout_name("light_heavyweight", title=True) == "UFC Light Heavyweight Title Bout"
    assert hyp.bout_name("w_strawweight") == "Women's Strawweight Bout"
    from app.services.ufc.model import _classify_weight_class
    for key in hyp.DIVISIONS:   # every name round-trips through the model's classifier
        assert _classify_weight_class(hyp.bout_name(key)) == key
