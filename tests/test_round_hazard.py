"""Round / duration survival model: settlement edges, person-period rows, curve shape."""
import numpy as np
import pandas as pd

from app.services.ufc.round_hazard import (
    HazardModel, finish_bin, over_prob, person_period, round_probs,
)


def test_settlement_edges():
    # O/U 1.5 = 7:30. A finish at 7:29 is under, at exactly 7:30 still under, 7:31 over.
    assert finish_bin(7.49) == 2 and finish_bin(7.5) == 2 and finish_bin(7.52) == 3
    # A stoppage at 5:00 of R1 (between rounds) belongs to R1 (bins 0-1).
    assert finish_bin(5.0) == 1 and finish_bin(0.2) == 0


def test_person_period_rows_stop_at_finish():
    X = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    t = np.array([3.0, 15.0, 11.0])          # R1 finish (bin 1); decision; finish in bin 4
    ev = np.array([1, 0, 2])
    sched = np.array([15.0, 15.0, 25.0])
    rows, y, reps, _ = person_period(X, t, ev, sched)
    assert list(np.bincount(reps)) == [2, 6, 5]           # stops at the finish bin
    assert list(y[reps == 0]) == [0, 1] and y[reps == 1].sum() == 0 and y[reps == 2][-1] == 2
    assert rows["hz_five"].tolist()[-1] == 1.0


def test_survival_curve_properties():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"f": rng.normal(size=n)})
    sched = np.where(rng.random(n) < 0.2, 25.0, 15.0)
    t = np.where(rng.random(n) < 0.5, rng.uniform(0.1, 14.9, n), sched)
    ev = np.where(t < sched, rng.integers(1, 3, n), 0)
    m = HazardModel(["f"]).fit([X], t, ev, sched)
    S = m.survival([X], sched)
    assert np.all(np.diff(S, axis=1) <= 1e-12) and np.allclose(S[:, 0], 1.0)
    o15, o25 = over_prob(S, 7.5), over_prob(S, 12.5)
    assert np.all(o25 <= o15 + 1e-12)
    rp = round_probs(S, sched)
    assert np.allclose(rp.sum(axis=1), 1.0)
    assert np.allclose(rp[sched <= 15, 3:5], 0.0)
    assert np.allclose(rp[:, 5], np.where(sched <= 15, S[:, 6], S[:, 10]))   # P(decision) = S(end)


def test_fine_bins_and_anchor():
    from app.services.ufc.round_hazard import anchor, finish_bin
    assert finish_bin(7.5, 1.25) == 5 and finish_bin(7.52, 1.25) == 6
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame({"f": rng.normal(size=n)})
    sched = np.full(n, 15.0)
    t = np.where(rng.random(n) < 0.5, rng.uniform(0.1, 14.9, n), 15.0)
    ev = np.where(t < 15, 1, 0)
    for interactions in (None, ["f"]):
        m = HazardModel(["f"], bin_size=1.25, interactions=interactions).fit([X], t, ev, sched)
        S = m.survival([X], sched)
        assert S.shape == (n, 11) and np.all(np.diff(S, axis=1) <= 1e-12)
    p_dec = np.full(n, 0.6)
    A = anchor(S, p_dec, sched)
    assert np.allclose(A[:, 6], 0.6) and np.all(np.diff(A, axis=1) <= 1e-12)
