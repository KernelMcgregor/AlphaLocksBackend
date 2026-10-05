"""P4P ranking, shared eligibility and the alt-rankings publisher checks. No database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_p4p_rankings.py
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from app.services.ufc import p4p_rankings as p4p
from app.services.ufc.alt_rankings_common import (
    BoutView, eligible_fighters, is_eligible, pool_of, recency_weight,
)
from app.services.ufc.alt_rankings_publisher import (
    AltRankingIntegrityError, assign_ranks, check,
)
from app.services.ufc.fighter_registry import FighterState
from app.services.ufc.model import ELO_K, ELO_OUTCOME_SCORES, ELO_W_RESULT
from app.services.ufc.tapology_rankings import _Bout

TODAY = date(2026, 10, 5)


# --------------------------------------------------------------------------- P4P maths

ONLY_D_AND_DEPTH = {k: 0.0 for k in p4p.DEFAULT_WEIGHTS} | {"dominance": 0.8, "depth": 0.2}


def test_deep_division_contender_beats_thin_division_champion():
    # Deep: champion 1800, contender 15 behind, the rest of the top 10 packed at 1750.
    deep = [1800, 1785] + [1750] * 8
    # Thin: a champion who only narrowly leads a weaker field.
    thin = [1700, 1695] + [1650] * 8
    elites = {"deep": deep, "thin": thin}
    depth = p4p.division_depth(elites)

    def score(r, div):
        c = {k: 0.0 for k in p4p.DEFAULT_WEIGHTS}
        c.update(dominance=p4p.dominance(r, elites[div]), depth=depth[div])
        return p4p.combine(c, ONLY_D_AND_DEPTH)

    assert score(1785, "deep") > score(1700, "thin")
    # And the other way: a champion far clear of a weak field still rates as dominant.
    assert p4p.dominance(1700, [1700] + [1560] * 9) > p4p.dominance(1785, deep)


def test_dominance_is_invariant_to_a_division_wide_shift():
    elite = [1700, 1680, 1650, 1640, 1630, 1620, 1610, 1600, 1590, 1580]
    shifted = [e + 250 for e in elite]
    assert p4p.dominance(1700, elite) == pytest.approx(p4p.dominance(1950, shifted))


def test_combine_ignores_unknown_keys_and_handles_zero_weights():
    c = {k: 50.0 for k in p4p.DEFAULT_WEIGHTS} | {"best_wins": 90.0, "not_a_component": 1e9}
    w = p4p.DEFAULT_WEIGHTS
    expected = (sum(w[k] * 50.0 for k in w) + w["best_wins"] * 40.0) / sum(w.values())
    assert p4p.combine(c) == pytest.approx(expected)
    assert p4p.combine(c, {k: 0 for k in w}) == 0.0


def _row(fid, div, score, rank, champ=False, pool="men"):
    return {"fighter_id": fid, "pool": pool, "division": div, "division_rank": rank,
            "is_champion": champ, "default_score": score, "raw": {}}


def test_pav_is_least_squares_nonincreasing():
    assert p4p.pav_nonincreasing([5, 4, 3]) == [5, 4, 3]           # already ordered
    assert p4p.pav_nonincreasing([3, 5]) == [4, 4]                 # violators pooled
    assert p4p.pav_nonincreasing([6, 2, 4, 1]) == [6, 3, 3, 1]
    out = p4p.pav_nonincreasing([1, 9, 2, 8, 3])
    assert all(a >= b for a, b in zip(out, out[1:]))
    assert sum(out) == pytest.approx(1 + 9 + 2 + 8 + 3)            # block means keep the mass


def _merged_order(rows):
    from app.services.ufc.alt_rankings_publisher import assign_ranks
    p4p.division_merge(rows)
    for r in rows:
        r["raw"]["score_raw"] = r.pop("score_raw")
    assign_ranks(rows)
    return [r["fighter_id"] for r in sorted(rows, key=lambda r: r["default_rank"])]


def test_division_merge_never_contradicts_division_order():
    # Lightweight: champion 1, then 2 and 3 — but 3 outscores both on raw score.
    rows = [_row(1, "lightweight", 60, 1, champ=True), _row(2, "lightweight", 55, 2),
            _row(3, "lightweight", 75, 3), _row(4, "welterweight", 70, 1, champ=True),
            _row(5, "welterweight", 66, 2)]
    order = _merged_order(rows)
    pos = {f: k for k, f in enumerate(order)}
    assert pos[1] < pos[2] < pos[3]                     # division order kept
    assert pos[4] < pos[5]
    by = {r["fighter_id"]: r for r in rows}
    # 3's strength lifts the whole block (60, 55, 75 -> 63.3 each) instead of jumping it.
    assert by[1]["default_score"] == pytest.approx(63.333, abs=1e-3)
    # Cross-division is free: the welterweight #2 outranks the lightweight champion.
    assert pos[5] < pos[1]


def test_division_merge_leaves_unranked_alone():
    rows = [_row(1, "lightweight", 50, 1), _row(2, "lightweight", 90, None)]
    p4p.division_merge(rows)
    assert rows[1]["default_score"] == 90


def test_best_wins_rewards_quality_not_quantity():
    win = lambda i, tier: BoutView(fight_id=i, date=TODAY, opponent_id=i, result="W",
                                   outcome="ud", finish_round=3, division="lightweight",
                                   tier=tier)
    three_elite = [win(1, 10), win(2, 10), win(3, 10)]
    many_weak = [win(i, 3) for i in range(10)]
    assert p4p.best_wins(three_elite, TODAY) == pytest.approx(100.0)
    assert p4p.best_wins(many_weak, TODAY) == pytest.approx(30.0)


# --------------------------------------------------------------------------- Elo parity

def _ctx_for_elo():
    stats = lambda sig, kd=0, td=0, ctrl=0, sub=0: {
        "kd": kd, "head_landed": sig, "sig_str_landed": sig, "td_landed": td,
        "ctrl_seconds": ctrl, "sub_att": sub, "sig_str_attempted": sig * 2,
        "td_attempted": td}
    fights = {
        # id: (id, date, red, blue, winner, method, details, secs)
        1: (1, date(2020, 1, 1), 10, 20, 10, "KO/TKO", "", 300),
        2: (2, date(2020, 6, 1), 20, 30, None, "Decision - Majority", "", 900),   # draw
        3: (3, date(2021, 1, 1), 30, 10, 30, "Decision - Unanimous", "", 900),
        4: (4, date(2021, 2, 1), 10, 20, None, "No Contest", "", 60),            # void
        5: (5, date(2021, 3, 1), 20, 30, 20, "Submission", "", 200),             # no stats
    }
    totals = {
        (1, 10): stats(30, kd=1), (1, 20): stats(10),
        (2, 20): stats(50), (2, 30): stats(52),
        (3, 30): stats(60, td=3, ctrl=200), (3, 10): stats(40),
        (4, 10): stats(1), (4, 20): stats(0),
    }
    from app.services.ufc.outcome_types import classify_outcome
    return {"fights": fights, "totals": totals,
            "outcome": {k: classify_outcome(v[5], v[6], v[4]) for k, v in fights.items()}}


def _reference_elo(ctx):
    """model.build_features' loop, transcribed by hand for this fixture."""
    from app.services.ufc.model import ELO_DESERVED_COEF

    elo = {}

    def deserved(fid, red, blue, secs):
        t = ctx["totals"]
        m = max(secs / 60, 0.5)
        z = sum(c * (t[(fid, red)][k] - t[(fid, blue)][k]) / m for k, c in ELO_DESERVED_COEF.items())
        return 1 / (1 + math.exp(-z))

    def update(red, blue, actual):
        r, b = elo.get(red, 1500.0), elo.get(blue, 1500.0)
        e = 1 / (1 + 10 ** ((b - r) / 400))
        elo[red], elo[blue] = r + ELO_K * (actual - e), b - ELO_K * (actual - e)

    update(10, 20, ELO_W_RESULT * ELO_OUTCOME_SCORES["ko"]
           + (1 - ELO_W_RESULT) * deserved(1, 10, 20, 300))
    update(20, 30, 0.5)
    update(30, 10, ELO_W_RESULT * ELO_OUTCOME_SCORES["ud"]
           + (1 - ELO_W_RESULT) * deserved(3, 30, 10, 900))
    # Fight 4 is a no-contest and fight 5 has no stats row: neither moves a rating.
    return elo


def test_elo_matches_model_formula():
    ctx = _ctx_for_elo()
    got = p4p.elo_ratings(ctx, TODAY)
    want = _reference_elo(ctx)
    assert set(got) == set(want)
    for f in want:
        assert got[f] == pytest.approx(want[f], abs=1e-9)


def test_elo_respects_as_of():
    ctx = _ctx_for_elo()
    early = p4p.elo_ratings(ctx, date(2020, 3, 1))
    assert set(early) == {10, 20}


# --------------------------------------------------------------------------- eligibility

def _eligibility_ctx():
    def bouts(fid, n, start_days_ago, division="lightweight"):
        return [
            _Bout(date=TODAY - timedelta(days=start_days_ago + 100 * (n - 1 - i)),
                  fight_id=fid * 100 + i, opponent_id=9999, won=True, drew=False,
                  method="KO/TKO", finish_round=1, division=division)
            for i in range(n)
        ]

    b = {
        1: bouts(1, 5, 30),                                   # eligible
        2: bouts(2, 3, 30),                                   # too few bouts
        3: bouts(3, 5, 900),                                  # inactive
        4: bouts(4, 5, 30),                                   # retired
        5: bouts(5, 5, 30, "w_strawweight"),                  # women's pool
        6: bouts(6, 3, 30, "w_strawweight"),                  # champion on 3 bouts
    }
    registry = {
        fid: FighterState(division=bs[-1].division, last_activity=bs[-1].date,
                          decided_fights=len(bs), rounds=len(bs), result_bouts=len(bs))
        for fid, bs in b.items()
    }
    return {
        "today": TODAY, "registry": registry, "totals": {}, "outcome": {},
        "champions": {"w_strawweight": 6},
        "hist": {
            "bouts": b, "tiers": {}, "names": {f: f"F{f}" for f in b},
            "status": {4: "Retired"},
            "activity": {f: [x.date for x in bs] for f, bs in b.items()},
        },
    }


def test_eligibility_and_pools():
    got = {f.fighter_id: f for f in eligible_fighters(_eligibility_ctx())}
    assert set(got) == {1, 5, 6}
    assert got[6].is_champion and not got[5].is_champion
    assert got[1].pool == "men" and got[5].pool == "women"
    assert got[1].bouts[0].date > got[1].bouts[-1].date        # newest first
    assert pool_of("w_flyweight") == "women" and pool_of("flyweight") == "men"


# --------------------------------------------------------------------------- publisher

def _rows(n_men=10, n_women=10):
    rows = []
    for i in range(n_men + n_women):
        rows.append({"fighter_id": i, "pool": "men" if i < n_men else "women",
                     "components": {"a": 50.0}, "default_score": float(i)})
    return rows


def _registry(ids):
    return {i: FighterState(division="lightweight", last_activity=TODAY,
                            decided_fights=5, rounds=5, result_bouts=5) for i in ids}


def test_publisher_check_passes_and_ranks():
    rows = _rows()
    assign_ranks(rows)
    check("bmf", rows, _registry(range(20)), TODAY)
    top_men = min((r for r in rows if r["pool"] == "men"), key=lambda r: r["default_rank"])
    assert top_men["fighter_id"] == 9 and top_men["default_rank"] == 1


@pytest.mark.parametrize("mutate,msg", [
    (lambda rs: rs.append(dict(rs[0])), "duplicate"),
    (lambda rs: rs[0]["components"].update(a=101.0), "outside"),
    (lambda rs: rs[0]["components"].update(a=float("nan")), "outside"),
    (lambda rs: rs.pop(), "only"),
])
def test_publisher_check_rejects(mutate, msg):
    rows = _rows()
    mutate(rows)
    with pytest.raises(AltRankingIntegrityError, match=msg):
        check("bmf", rows, _registry(range(20)), TODAY)


def test_publisher_check_rejects_ineligible():
    reg = _registry(range(20))
    reg[3] = FighterState(division="lightweight", last_activity=TODAY - timedelta(days=900),
                          decided_fights=5, rounds=5, result_bouts=5)
    with pytest.raises(AltRankingIntegrityError, match="not eligible"):
        check("p4p", _rows(), reg, TODAY)


def test_recency_weight_half_life():
    assert recency_weight(0) == 1.0
    assert recency_weight(3 * 365.25) == pytest.approx(0.5)



def test_champion_eligibility_still_needs_activity():
    st = FighterState(division="w_bantamweight", last_activity=TODAY - timedelta(days=900),
                      decided_fights=3, rounds=3, result_bouts=3)
    assert not is_eligible(st, TODAY, is_champion=True)


# --------------------------------------------------------------------------- belts

def test_elevation_promotes_interim_champion():
    from app.services.ufc import champions as ch

    hw = "UFC Heavyweight Title Bout"
    interim = "UFC Interim Heavyweight Title Bout"
    bouts = [  # newest first, as load_title_bouts returns them
        (date(2026, 6, 14), interim, "KO/TKO", 30, 40, 30),          # Gane interim
        (date(2025, 10, 25), hw, "Could Not Continue", None, 20, 30),  # NC: belt stays
        (date(2024, 11, 16), hw, "KO/TKO", 10, 10, 50),              # Jones defends
        (date(2024, 7, 27), interim, "KO/TKO", 20, 20, 60),          # Aspinall interim
    ]
    at = lambda d: ch.champions_as_of(bouts, d).get("heavyweight")
    assert at(date(2025, 1, 1)) == 10            # Jones
    assert at(date(2025, 7, 1)) == 20            # Jones retires -> Aspinall elevated
    assert at(date(2026, 9, 1)) == 20            # Gane's interim win does not take it
    assert at(date(2026, 10, 1)) == 30           # Aspinall vacates -> Gane elevated
    # A later decided undisputed bout overrides an elevation.
    later = [(date(2026, 11, 14), hw, "KO/TKO", 70, 30, 70)] + bouts
    assert ch.champions_as_of(later, date(2026, 12, 1))["heavyweight"] == 70
