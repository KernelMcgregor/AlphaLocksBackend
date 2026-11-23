"""Deserve-to-win blend in the division ranker: synthetic context, no database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_tapology_dtw_blend.py
"""
from __future__ import annotations

from datetime import date

import pytest

from app.services.ufc.tapology_rankings import TapologyRanker, Weights, _Bout

AS_OF = date(2026, 11, 1)
D = date(2026, 6, 1)
A, B = 1, 2


def ctx(method="Decision - Split", dtw=None):
    bouts = {
        A: [_Bout(D, 10, B, won=True, drew=False, method=method, finish_round=3,
                  division="lightweight")],
        B: [_Bout(D, 10, A, won=False, drew=False, method=method, finish_round=3,
                  division="lightweight")],
    }
    return {
        "bouts": bouts, "tiers": {(A, 10): 5, (B, 10): 5}, "names": {A: "A", B: "B"},
        "status": {}, "activity": {A: [D], B: [D]},
        "div_bouts": {A: [(D, "lightweight")], B: [(D, "lightweight")]},
        "title_bouts": [], "dtw": dtw or {},
    }


def scores(c, blend):
    s = TapologyRanker(Weights(dtw_blend=blend)).score(c, AS_OF).division_scores
    return s[(A, "lightweight")], s[(B, "lightweight")]


def test_off_by_default_and_without_dtw():
    robbed = {(A, 10): 0.1, (B, 10): 0.9}
    assert scores(ctx(dtw=robbed), 0.0) == scores(ctx(), 0.0)
    assert scores(ctx(), 0.3) == scores(ctx(), 0.0)


def test_robbery_moves_credit_toward_the_deserving_fighter():
    robbed = {(A, 10): 0.1, (B, 10): 0.9}
    a0, b0 = scores(ctx(dtw=robbed), 0.0)
    a1, b1 = scores(ctx(dtw=robbed), 0.3)
    assert a1 < a0 and b1 > b0
    assert a1 > b1                       # a partial blend never flips who it favors


def test_deserved_win_is_unchanged_at_full_blend_only_if_certain():
    sure = {(A, 10): 1.0, (B, 10): 0.0}
    assert scores(ctx(dtw=sure), 0.5) == pytest.approx(scores(ctx(), 0.0))


def test_finishes_are_never_blended():
    # load_decision_dtw never emits finishes; the ranker uses whatever it is given, so
    # the guard lives in the loader's kinds.
    from app.services.ufc.tapology_rankings import DTW_KINDS
    assert "ko" not in DTW_KINDS and "sub" not in DTW_KINDS


def test_rank_tier_lifts_a_ranked_opponent():
    # C beats X, then A beats C. The record recursion rates C low (one UFC bout); the
    # replay has C ranked #1 the day before, so A's tier for that bout comes up.
    import app.services.ufc.tapology_rankings as tr

    C, X = 3, 4
    d1, d2 = date(2026, 1, 10), date(2026, 5, 10)
    mk = lambda d, fid, opp, won: tr._Bout(d, 20 if d == d1 else 21, opp, won=won, drew=False,
                                           method="KO/TKO", finish_round=1,
                                           division="lightweight")
    bouts = {C: [mk(d1, C, X, True), mk(d2, C, A, False)], X: [mk(d1, X, C, False)],
             A: [mk(d2, A, C, True)]}
    record = {(C, 20): 1, (X, 20): 1, (C, 21): 1, (A, 21): 1}
    ctx = {"bouts": bouts, "tiers": record, "names": {}, "status": {},
           "activity": {f: sorted(b.date for b in v) for f, v in bouts.items()},
           "div_bouts": {f: [(b.date, "lightweight") for b in v] for f, v in bouts.items()},
           "title_bouts": [], "dtw": {}}
    tr._RANK_TIER_CACHE.clear()
    tiers = tr._rank_tiers(ctx)
    assert tiers[(A, 21)] == round(0.5 * 10 + 0.5 * 1)     # C was #1 going in
    assert tiers[(C, 20)] == 1                               # X unranked: rank tier 1
