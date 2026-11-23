"""Fight excitement: the win-probability path, its features and the card logit. No database."""
import json

import numpy as np
import pandas as pd
import pytest

from app.services.ufc.fight_excitement import (
    _win_share, card_score, fight_features, fit_card_logit, path_features, per_round_prior, wp_path,
)


def test_win_share_and_prior():
    assert _win_share([0.5] * 3) == pytest.approx(0.5)
    assert _win_share([1, 1, 0]) == pytest.approx(1.0)
    assert _win_share([0.5, 0.5]) == pytest.approx(0.5)        # a 1-1 split counts half
    q = per_round_prior(0.8, 3)
    assert _win_share([q] * 3) == pytest.approx(0.8, abs=1e-6)


def test_path_moves_with_rounds():
    p = wp_path(0.5, [0.95], 3, None)
    assert p[0] == 0.5 and p[1] > 0.7
    # A finish ends the path at the winner.
    assert wp_path(0.7, [0.2], 3, 0.0)[-1] == 0.0


def test_comeback_swings_more_than_expected_finish():
    favorite_wins = path_features(wp_path(0.8, [0.9], 3, 1.0))
    comeback = path_features(wp_path(0.8, [0.9], 3, 0.0))
    assert comeback["swing"] > 3 * favorite_wins["swing"]
    seesaw = path_features(wp_path(0.5, [0.9, 0.1, 0.1, 0.9, 0.9], 5, None))
    assert seesaw["lead_changes"] == 2


def _row(**kw):
    r = dict(fight_time_seconds=900, max_fight_time_seconds=900, kind="ud", winner_id=1, red_id=1,
             round_p_red=json.dumps([0.6, 0.4, 0.6]), rounds_scheduled=3, finish_round=3,
             red_close=0.6, sig=120, kd=0, sub_att=0, rev=0, gnd=10, ctrl=60, title=False,
             card_position=3)
    r.update(kw)
    return r


def test_fight_features():
    f = fight_features(_row())
    assert f["sig_pm"] == pytest.approx(np.log1p(120 / 17)) and f["ctrl_share"] == pytest.approx(60 / 900)
    assert f["upset"] == pytest.approx(-np.log(0.6))
    assert f["main_card"] == 1.0 and f["main_event"] == 0.0
    stall = fight_features(_row(sig=30, ctrl=700, round_p_red=json.dumps([0.95, 0.95, 0.95])))
    assert stall["round_close"] < f["round_close"] and stall["ctrl_share"] > 0.7
    ko = fight_features(_row(kind="ko", finish_round=1, fight_time_seconds=60, winner_id=2))
    assert ko["quick_finish"] == 1.0 and ko["ko"] == 1.0 and ko["upset"] == pytest.approx(-np.log(0.4))


def test_card_logit_learns_and_normalizes():
    rng = np.random.default_rng(0)
    n_cards, per = 300, 10
    x = rng.normal(size=n_cards * per)
    g = np.repeat(np.arange(n_cards), per)
    y = np.zeros_like(x)
    for c in range(n_cards):          # the card's best x wins most of the time
        m = np.where(g == c)[0]
        u = 2 * x[m]
        p = np.exp(u) / np.exp(u).sum()
        y[m[rng.choice(per, p=p)]] = 1
    X = pd.DataFrame({"x": x, "noise": rng.normal(size=len(x))})
    m = fit_card_logit(X, y, g, ["x", "noise"], l2=0.1)
    assert m.w[0] > 1.5 and abs(m.w[1]) < 0.3
    p = m.probs(X, g)
    assert np.allclose(pd.Series(p).groupby(g).sum(), 1.0)


def test_card_score_rewards_depth():
    assert card_score([95, 90, 85, 80, 75]) > card_score([95, 90, 85, 20, 10])
    assert card_score([]) == 0.0
