"""Tests for app.services.ufc.method_market (decision correction on the method grid)."""

import numpy as np
import pandas as pd
import pytest

from app.services.ufc import method_market as mm


def _corr(hinge=True):
    # a=1, b1=0, b2=-6: only the hinge moves P(dec), and only for big favourites
    return mm.DecisionCorrection(coef=[1.0, 0.0, -6.0] if hinge else [1.0, 0.0],
                                 intercept=0.0, hinge=hinge, n_fit=0)


def test_rows_still_sum_to_one_and_keep_finish_split():
    cond = np.array([[0.3, 0.1, 0.6], [0.5, 0.2, 0.3]])
    out = _corr().adjust(cond, np.array([0.9, 0.95]))
    assert out.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert out[:, 0] / out[:, 1] == pytest.approx(cond[:, 0] / cond[:, 1])
    assert (out[:, 2] < cond[:, 2]).all()


def test_no_odds_rows_unchanged():
    cond = np.array([[0.3, 0.1, 0.6], [0.5, 0.2, 0.3]])
    out = _corr().adjust(cond, np.array([np.nan, 0.6]))
    assert out[0] == pytest.approx(cond[0])
    assert out[1] == pytest.approx(cond[1])   # below the knot, a=1 / b1=0 / c=0: identity


def test_monotone_in_market_prob():
    cond = np.tile([0.3, 0.1, 0.6], (5, 1))
    q = np.array([0.5, 0.7, 0.8, 0.9, 0.97])
    dec = _corr().adjust(cond, q)[:, 2]
    assert (np.diff(dec) <= 1e-12).all()


def test_apply_uses_complement_for_blue():
    cond = np.array([[0.3, 0.1, 0.6]])
    r, b = _corr().apply(cond, cond, np.array([0.9]))
    assert r[0, 2] < 0.6                      # red the big favourite: hinge applies
    assert b[0, 2] == pytest.approx(0.6)      # blue winning as a 10% dog: no hinge


def test_fit_and_hinge_rule():
    rng = np.random.default_rng(0)
    n = 2000
    q = rng.uniform(0.2, 0.95, n)
    p = rng.uniform(0.3, 0.7, n)
    true = 1 / (1 + np.exp(-(mm._logit(p) - 5 * np.clip(q - 0.7, 0, None))))
    rows = pd.DataFrame({"q_w": q, "w_dec": p, "y_dec": (rng.random(n) < true).astype(int),
                         "date": pd.Timestamp("2024-01-01")})
    corr = mm.fit(rows)
    assert corr.hinge and corr.coef[2] < 0     # unscaled hinge: ridge shrinks it, sign holds
    few = rows[np.maximum(rows.q_w, 1 - rows.q_w) < 0.7]
    assert not mm.fit(few).hinge


def test_save_load_roundtrip(tmp_path):
    c = _corr()
    c.save(tmp_path / "m.json")
    assert mm.load(tmp_path / "m.json") == c
    assert mm.load(tmp_path / "missing.json") is None
