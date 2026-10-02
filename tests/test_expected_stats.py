"""expected_stats: walk-forward guarantees and the v2 control-time composition."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.services.ufc.expected_stats as es


def _frame(n_fights: int = 700, n_fighters: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic load_fight_data-style frame, 2009-2013, two rows per bout."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.4, n_fighters)
    dates = pd.date_range("2009-01-01", "2013-06-30", periods=n_fights)
    rows = []
    for i, d in enumerate(dates):
        a, b = rng.choice(n_fighters, 2, replace=False)
        secs = int(rng.choice([300, 600, 900]))
        ca = rng.uniform(0, 0.4) * secs
        cb = rng.uniform(0, 0.9) * (secs - ca) * 0.3
        for f, o, c in ((a, b, ca), (b, a, cb)):
            rows.append({
                "fight_id": i, "date": d.date(), "stats_fighter_id": int(f),
                "fight_time_seconds": secs, "weight_class": "Lightweight Bout",
                "time_format": "5-5-5",
                "sig_str_landed": rng.poisson(4 * np.exp(skill[f] - skill[o]) * secs / 60),
                "td_landed": rng.poisson(0.1 * secs / 60), "kd": rng.poisson(0.02 * secs / 60),
                "sub_att": rng.poisson(0.05 * secs / 60), "ctrl_seconds": float(int(c)),
                "glicko_td": float(skill[f]), "glicko_meta_rounds_seen": float(i % 9),
            })
    return pd.DataFrame(rows)


@pytest.fixture(params=[(False, False), (True, False), (True, True)], ids=["v1", "v2", "v2_glicko"])
def version(request, monkeypatch):
    monkeypatch.setattr(es, "V2", request.param[0])
    monkeypatch.setattr(es, "V2_GLICKO", request.param[1])
    return request.param


def test_a_bouts_own_result_does_not_change_its_expectation(version):
    df = _frame()
    base = es.compute_expected_stats(df)
    late = df["date"] >= pd.Timestamp("2013-01-01").date()
    j = df.index[late][0]
    df2 = df.copy()
    df2.loc[j, ["sig_str_landed", "td_landed", "ctrl_seconds"]] = [500, 20, 900.0]
    after = es.compute_expected_stats(df2)
    cols = [c for c in base.columns if c.startswith("xs_")]
    # Everything up to the end of that bout's refit quarter, the bout itself included,
    # comes from fits that never saw it. Later quarters legitimately learn from it.
    upto = pd.to_datetime(df["date"]) < pd.Timestamp("2013-04-01")
    keep = base["fight_id"].isin(df.loc[upto, "fight_id"])
    pd.testing.assert_frame_equal(base.loc[keep, cols], after.loc[keep, cols])
    assert keep[base["fight_id"] == df.loc[j, "fight_id"]].all()


def test_appending_later_bouts_does_not_change_earlier_expectations(version):
    df = _frame()
    early = df[df["date"] < pd.Timestamp("2013-01-01").date()]
    full = es.compute_expected_stats(df)
    part = es.compute_expected_stats(early)
    cols = [c for c in part.columns if c.startswith("xs_")]
    m = part.merge(full, on=["fight_id", "stats_fighter_id"], suffixes=("", "_f"))
    for c in cols:
        np.testing.assert_allclose(m[c].to_numpy(float), m[f"{c}_f"].to_numpy(float), equal_nan=True)


def test_v2_control_shares_are_a_composition(monkeypatch):
    monkeypatch.setattr(es, "V2", True)
    out = es.compute_expected_stats(_frame()).dropna(subset=["xs_ctrl_for"])
    assert len(out) > 0
    share = out["xs_fight_ground_share"]
    assert ((share > 0) & (share < 1)).all()
    # for + against control (seconds per minute) never exceeds the minute itself
    assert (out["xs_ctrl_for"] + out["xs_ctrl_against"] < 60).all()
    # bout-level columns are identical for both corners
    g = out.groupby("fight_id")
    for c in ("xs_fight_sig_pace", "xs_fight_ground_share"):
        assert ((g[c].max() - g[c].min()) < 1e-12).all()


def test_against_is_the_opponents_for(version):
    out = es.compute_expected_stats(_frame()).dropna(subset=["xs_sig_for"])
    pairs = out.merge(out, on="fight_id", suffixes=("", "_o"))
    pairs = pairs[pairs["stats_fighter_id"] != pairs["stats_fighter_id_o"]]
    np.testing.assert_allclose(pairs["xs_sig_against"], pairs["xs_sig_for_o"], rtol=1e-9)
    np.testing.assert_allclose(pairs["xs_ctrl_against"], pairs["xs_ctrl_for_o"], rtol=1e-9)


def test_display_length_distribution_and_p_more():
    from app.services.ufc.expected_stats_serving import SUPPORT, _length_dist, _p_more
    t = np.arange(0, 25.01, 1.25)
    s = np.clip(1 - 0.02 * t, 0, 1)
    T, P = _length_dist((t, s), 15.0)
    assert abs(P.sum() - 1) < 1e-12 and T.max() == 15.0
    assert abs(P[-1] - (1 - 0.02 * 15) / (1 - 0.02 * 0)) < 1e-9   # decision mass = S(15)
    disp = {"theta": 4.0}
    assert abs(_p_more(5.0, 5.0, T, P, disp, SUPPORT["sig"]) - 0.5) < 1e-9
    a = _p_more(6.0, 4.0, T, P, disp, SUPPORT["sig"])
    b = _p_more(4.0, 6.0, T, P, disp, SUPPORT["sig"])
    assert a > 0.5 and abs(a + b - 1) < 1e-9
