"""Judge profiles (app/services/ufc/officials_service.py): verdicts, carryover columns,
pivotal cards, expected agreement and the lean fit. No database."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from app.services.ufc.deserve_to_win import logit, sigmoid
from app.services.ufc.officials_service import (
    AXES, add_panel_agreement, add_ten8_baseline, build_rounds, fight_cards, fit_leans,
    pair_agree_expected, recalibrate, round_metrics,
)
from scripts.round_effects import RAW


def _cards(fight_id, rows, date=dt.date(2020, 1, 1), red=1, blue=2):
    """rows: (round, judge_seq, red_pts, blue_pts[, red_ded, blue_ded]) -> card frame."""
    out = []
    for r in rows:
        rnd, seq, rp, bp, *ded = r
        rd, bd = ded or (0, 0)
        out.append(dict(fight_id=fight_id, round=rnd, judge_seq=seq, judge_id=100 + seq,
                        judge=f"J{seq}", red_pts=rp, blue_pts=bp, red_ded=rd, blue_ded=bd,
                        date=date, red=red, blue=blue, red_cc="US", blue_cc="BR",
                        location="Las Vegas, Nevada, USA"))
    return pd.DataFrame(out)


def _stats(fight_id, rounds, red=1, blue=2):
    rows = []
    for r in range(1, rounds + 1):
        for f, k in ((red, 1.0), (blue, 0.5)):
            rows.append(dict(fight_id=fight_id, round=r, fighter_id=f,
                             **{c: k * (r + i) for i, c in enumerate(RAW)}))
    return pd.DataFrame(rows)


def _dtw(fight_id, ps, robbery=None, p_red=0.5):
    return pd.DataFrame([dict(fight_id=fight_id, round_p_red=json.dumps(ps), p_red=p_red,
                              p_blue=1 - p_red, robbery_score=robbery, official_outcome=None)])


ODDS = pd.DataFrame([dict(fight_id=1, red_close_prob=0.7)])


def test_build_rounds_verdicts_deductions_and_carry():
    cards = _cards(1, [
        (1, 1, 10, 9), (1, 2, 10, 9), (1, 3, 9, 10),
        (2, 1, 9, 9, 1, 0),          # 10-9 red before the deduction
        (2, 2, 10, 10), (2, 3, 9, 10),
        (3, 1, 10, 8), (3, 2, 9, 10), (3, 3, 9, 10),
    ])
    r = build_rounds(cards, _stats(1, 3), _dtw(1, [0.6, 0.5, 0.4]), ODDS)
    by = r.set_index(["round", "judge_seq"])
    assert by.loc[(2, 1), "s"] == 1                      # deduction added back
    assert by.loc[(2, 2), "s"] == 0                      # 10-10 kept as even
    assert by.loc[(3, 1), "margin"] == 2
    assert by.loc[(1, 1), "carry"] == 0                  # nothing before round 1
    assert by.loc[(2, 3), "carry"] == -1                 # judge 3 gave round 1 to blue
    # Round 2, judge 3: the other two gave round 1 to red.
    assert by.loc[(2, 3), "carry_panel"] == pytest.approx(1.0)
    assert by.loc[(3, 1), "carry_panel"] == pytest.approx((0 - 1) / 2)
    assert by.loc[(2, 1), "prev_logit"] == pytest.approx(logit(0.6))
    assert (r.fav == 1).all()                            # red closed at 70%
    assert (r.home == 1).all()                           # US fighter, US event
    assert by.loc[(1, 1), "p"] == pytest.approx(0.6)


def test_build_rounds_drops_rounds_without_model_p():
    cards = _cards(1, [(1, 1, 10, 9), (2, 1, 10, 9)])
    r = build_rounds(cards, _stats(1, 2), _dtw(1, [0.6]), ODDS)
    assert r["round"].tolist() == [1]


def test_pair_agreement_without_latent_is_independent():
    p = np.array([0.5, 0.7, 0.9])
    assert pair_agree_expected(p, 0.0) == pytest.approx(p ** 2 + (1 - p) ** 2, abs=1e-6)


def test_pair_agreement_symmetric_and_raised_by_latent():
    p = np.array([0.2, 0.5, 0.8])
    a = pair_agree_expected(p, 1.5)
    assert a[0] == pytest.approx(a[2], abs=1e-9)
    assert (a > p ** 2 + (1 - p) ** 2).all()


def test_panel_agreement_counts_other_judges():
    r = pd.DataFrame(dict(fight_id=1, round=1, s=[1, 1, -1], p=0.5))
    out = add_panel_agreement(r, 0.0)
    assert out.agree.tolist() == [0.5, 0.5, 0.0]
    assert out.agree_exp.tolist() == pytest.approx([0.5] * 3)


def test_round_metrics_expected_accuracy():
    d = pd.DataFrame(dict(fight_id=[1, 1, 2, 2], s=[1, -1, 1, 0], p=[0.8, 0.3, 0.4, 0.5],
                          margin=[1, 1, 2, 0], post2017=True, q8=[0.05, 0.05, 0.1, np.nan]))
    m = round_metrics(d)
    assert m["n_decided"] == 3 and m["even_rate"] == pytest.approx(0.25)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["expected_accuracy"] == pytest.approx((0.8 + 0.7 + 0.6) / 3)
    assert m["ten8_post2017"] == pytest.approx(1 / 3)
    assert m["ten8_post2017_exp"] == pytest.approx(0.2 / 3)
    assert m["ten8_pre2017"] is None


def test_recalibrate_recovers_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20_000)
    y = rng.random(z.size) < sigmoid(0.8 * z)
    r = pd.DataFrame(dict(p=sigmoid(z), s=np.where(y, 1, -1)))
    out, b = recalibrate(r)
    assert b == pytest.approx(0.8, abs=0.04)
    assert ((out.p > 0.5) == (r.p > 0.5)).all()         # favourite never changes


def test_ten8_baseline_matches_pooled_rate():
    rng = np.random.default_rng(1)
    n = 5000
    r = pd.DataFrame(dict(s=rng.choice([1, -1], n), p=rng.uniform(0.05, 0.95, n),
                          post2017=rng.random(n) < 0.5))
    r["margin"] = np.where(rng.random(n) < 0.05, 2, 1)
    out = add_ten8_baseline(r)
    assert out.q8.mean() == pytest.approx((r.margin >= 2).mean(), abs=1e-3)


@pytest.mark.parametrize("totals,pivotal", [
    ([(30, 27), (30, 27), (30, 27)], [False, False, False]),   # UD
    ([(29, 28), (29, 28), (28, 29)], [True, True, False]),     # SD: both majority cards
    ([(29, 28), (29, 28), (28, 28)], [True, True, False]),     # MD 2-0-1
    ([(29, 28), (28, 29), (28, 28)], [False, False, False]),   # split draw
])
def test_pivotal_cards(totals, pivotal):
    rows = []
    for seq, (rp, bp) in enumerate(totals, start=1):
        # Spread each total over three rounds; totals are what matter.
        rows += [(1, seq, rp - 20, bp - 20), (2, seq, 10, 10), (3, seq, 10, 10)]
    fc = fight_cards(_cards(1, rows), _dtw(1, [0.5] * 3, robbery=0.8)).sort_values("judge_seq")
    assert fc.pivotal.tolist() == pivotal


def test_robbery_and_against_model_cards():
    rows = [(1, s, 10, 9) for s in (1, 2)] + [(1, 3, 9, 10)]
    fc = fight_cards(_cards(1, rows), _dtw(1, [0.3], robbery=0.75, p_red=0.25))
    fc = fc.sort_values("judge_seq")
    assert fc.robbery_card.tolist() == [True, True, False]
    assert fc.against_model.tolist() == [True, True, False]


def _synthetic_rounds(n_fights=1500, seed=0, carry=0.0, big=1, small=2):
    """Three-round fights judged by a busy judge (in every fight) and a rare one; judge
    `big` carries over with strength `carry` on top of the average judge."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fights):
        judges = [big, 3, small if f % 30 == 0 else 4]
        prev = {j: 0.0 for j in judges}
        prev_z = 0.0
        for rnd in (1, 2, 3):
            z = rng.normal(0, 1.5)
            for seq, j in enumerate(judges, start=1):
                eff = carry * prev[j] if j == big else 0.0
                s = 1 if rng.random() < sigmoid(z + eff) else -1
                rows.append(dict(fight_id=f, round=rnd, judge_seq=seq, judge_id=j,
                                 date=dt.date(2020, 1, 1), s=s, p=float(sigmoid(z)),
                                 fav=0.0, home=0.0, carry=prev[j] if rnd > 1 else 0.0,
                                 prev_logit=prev_z if rnd > 1 else 0.0,
                                 **{f"ax_{a}": 0.0 for a in AXES}))
            prev = {r["judge_id"]: r["s"] for r in rows[-3:]}
            prev_z = z
    d = pd.DataFrame(rows)
    # carry_panel: mean of the other judges' previous-round verdicts.
    d = d.sort_values(["fight_id", "round", "judge_seq"]).reset_index(drop=True)
    tot = d.groupby(["fight_id", "round"]).s.sum().rename("tot").reset_index()
    tot["round"] += 1
    d = d.merge(tot, on=["fight_id", "round"], how="left")
    d["carry_panel"] = np.where(d["round"] > 1, (d.tot.fillna(0) - d.carry) / 2, 0.0)
    return d.drop(columns="tot")


def test_fit_leans_recovers_planted_carryover():
    d = _synthetic_rounds(carry=0.8)
    rows, pooled = fit_leans(d, s_lean=0.3, s_axis=0.3)
    big = rows.set_index("judge_id").loc[1]
    total = pooled["carry"]["effect"] + big["carry"]
    assert total == pytest.approx(0.8, abs=0.25)
    assert big["carry"] > 2 * big["carry_se"]


def test_fit_leans_shrinks_rare_judges_more():
    d = _synthetic_rounds(carry=0.0, seed=3)
    rows, _ = fit_leans(d, s_lean=0.3, s_axis=0.3)
    j = rows.set_index("judge_id")
    assert j.loc[2, "carry_se"] > j.loc[1, "carry_se"]
    assert abs(j.loc[2, "carry"]) < 3 * j.loc[2, "carry_se"]
