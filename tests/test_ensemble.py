"""The served ensemble's market blend.

Run:  ./venv/bin/python -m pytest tests/test_ensemble.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.ufc.ensemble import Ensemble, expanding_stack_eval, feature_set, fit_stack


def _ens(stack):
    return Ensemble(members=[], train_means=pd.Series(dtype=float), stack=stack)


def test_unpriced_fights_fall_back_to_the_model():
    e = _ens({"a": 0.0, "b_mkt": 1.0, "b_model": 0.3, "n": 100})
    out = e.final_prob(np.array([0.7, 0.7]), np.array([np.nan, 0.5]))
    assert out[0] == pytest.approx(0.7)
    assert out[1] != pytest.approx(0.7)


def test_pure_market_stack_returns_the_market():
    e = _ens({"a": 0.0, "b_mkt": 1.0, "b_model": 0.0, "n": 100})
    assert e.final_prob(np.array([0.9]), np.array([0.62]))[0] == pytest.approx(0.62, abs=1e-6)


def test_stack_ignores_a_model_that_is_noise():
    rng = np.random.default_rng(0)
    n = 4000
    mkt = rng.uniform(0.2, 0.8, n)
    y = (rng.random(n) < mkt).astype(float)
    oof = pd.DataFrame({"odds_red_prob": mkt, "model_prob": rng.uniform(0.2, 0.8, n),
                        "red_wins": y})
    s = fit_stack(oof)
    assert abs(s["b_model"]) < 0.15
    assert s["b_mkt"] == pytest.approx(1.0, abs=0.15)
    r = expanding_stack_eval(oof)
    assert r["final"] >= r["market"] - 0.003


def test_no_odds_feature_sets_exclude_odds():
    feats = ["diff_elo", "odds_red_prob", "elo_vs_odds", "diff_avg_kd_per5", "diff_xs_sig_diff"]
    assert feature_set("noodds", feats) == ["diff_elo", "diff_avg_kd_per5", "diff_xs_sig_diff"]
    assert feature_set("noodds_noraw", feats) == ["diff_elo", "diff_xs_sig_diff"]
