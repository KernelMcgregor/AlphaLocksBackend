"""Method v2: 6-way grid coherence, winner orientation, labels, point-in-time ratings."""
from datetime import date

import numpy as np
import pandas as pd

from app.services.ufc.method_ratings import method_class, ufc_features
from app.services.ufc.method_v2 import joint, marginal, orient


def test_joint_sums_and_matches_moneyline():
    rng = np.random.default_rng(0)
    p_red = rng.uniform(0.1, 0.9, 50)
    cr = rng.dirichlet([1, 1, 1], 50)
    cb = rng.dirichlet([1, 1, 1], 50)
    six = joint(p_red, cr, cb)
    assert np.allclose(six.sum(axis=1), 1.0)
    assert np.allclose(six[:, :3].sum(axis=1), p_red)          # agrees with the moneyline
    assert np.allclose(marginal(six).sum(axis=1), 1.0)


def test_orient_mirrors_corners():
    m = pd.DataFrame({"diff_x": [2.0], "red_y": [1.0], "blue_y": [5.0],
                      "fight_is_five_round": [1.0], "red_wins": [1.0],
                      "odds_red_prob": [0.7], "odds_blue_prob": [0.35]})
    a = orient(m, np.array([True]))
    b = orient(m, np.array([False]))
    assert a["diff_x"][0] == 2.0 and b["diff_x"][0] == -2.0
    assert (a["w_y"][0], a["l_y"][0]) == (1.0, 5.0)
    assert (b["w_y"][0], b["l_y"][0]) == (5.0, 1.0)
    assert "w_wins" not in a.columns                          # label never becomes a feature
    assert abs(a["mkt_w_prob"][0] + b["mkt_w_prob"][0] - 1) < 1e-9
    assert a["fight_is_five_round"][0] == b["fight_is_five_round"][0] == 1.0


def test_labels_exclude_draw_dq_nc():
    assert method_class("KO/TKO", "Punch", 1) == "ko"
    assert method_class("TKO - Doctor's Stoppage", "", 1) == "ko"
    assert method_class("Submission", "Rear Naked Choke", 1) == "sub"
    assert method_class("Decision - Split", "", 1) == "dec"
    assert method_class("Decision - Majority", "", None) is None  # draw
    assert method_class("DQ", "Illegal knee", 1) is None
    assert method_class("Overturned", "", None) is None


def _row(fid, d, fighter, corner, winner, method, head=10, kd=0, sub_att=0):
    return dict(fight_id=fid, date=d, stats_fighter_id=fighter, corner=corner,
                red_fighter_id=1, blue_fighter_id=2, winner_id=winner, method=method,
                details="", fight_time_seconds=300, head_landed=head, kd=kd, sub_att=sub_att)


def test_ratings_are_point_in_time():
    rows = [
        _row(10, date(2010, 1, 1), 1, "red", 1, "KO/TKO", head=20, kd=2),
        _row(10, date(2010, 1, 1), 2, "blue", 1, "KO/TKO", head=5),
        _row(11, date(2011, 1, 1), 1, "red", 1, "KO/TKO", head=20, kd=1),
        _row(11, date(2011, 1, 1), 2, "blue", 1, "KO/TKO", head=5),
        # Other fighters, so the pre-2015 prior is not 100% KO.
        _row(5, date(2009, 1, 1), 3, "red", 3, "Decision - Unanimous"),
        _row(5, date(2009, 1, 1), 4, "blue", 3, "Decision - Unanimous"),
        _row(6, date(2009, 2, 1), 3, "red", 3, "Submission"),
        _row(6, date(2009, 2, 1), 4, "blue", 3, "Submission"),
    ]
    df = pd.DataFrame(rows)
    f = ufc_features(df)
    # First fight: no history, so both fighters sit at the prior.
    assert f.loc[0, "m_ufc_n"] == 0 and f.loc[0, "m_power"] == f.loc[1, "m_power"]
    # Second fight sees only the first: fighter 1 more powerful, fighter 2 worse chin.
    assert f.loc[2, "m_ufc_n"] == 1
    assert f.loc[2, "m_power"] > f.loc[0, "m_power"]
    assert f.loc[3, "m_chin"] > f.loc[1, "m_chin"]
    assert f.loc[2, "m_ko_win_share"] > f.loc[0, "m_ko_win_share"]
    assert f.loc[3, "m_ko_loss_share"] > f.loc[1, "m_ko_loss_share"]


_BFO_PAGE = """<table class="odds-table"><tbody>
<tr class="pr"><th scope="row">Smith wins by TKO/KO</th><td data-li="[21,1,500,8,1]">+200&#9650;</td><td data-li="[28,1,500,8,1]">+100</td></tr>
<tr class="pr"><th scope="row">Smith wins by submission</th><td data-li="[21,1,500,9,1]">+600</td></tr>
<tr class="pr"><th scope="row">Smith wins by decision</th><td data-li="[21,1,500,11,1]">+300</td></tr>
<tr class="pr"><th scope="row">Jones wins by TKO/KO</th><td data-li="[21,1,500,8,2]">+400</td></tr>
<tr class="pr"><th scope="row">Jones wins by submission</th><td data-li="[21,1,500,9,2]">+900</td></tr>
<tr class="pr"><th scope="row">Jones wins by decision</th><td data-li="[21,1,500,11,2]">+250</td></tr>
<tr class="pr"><th scope="row">Any other result</th><td data-li="[2,500,8,1]">n/a</td></tr>
<tr class="pr"><th scope="row">Over 2&#189; rounds</th><td data-li="[21,1,500,33,0]">-150</td></tr>
<tr class="pr"><th scope="row">Under 2&#189; rounds</th><td data-li="[21,2,500,33,0]">+120</td></tr>
</tbody></table>"""


def test_bfo_props_parse_and_orient():
    from app.services.ufc.bfo_props import consensus, corner_markets, parse_props
    rows = parse_props(_BFO_PAGE)
    assert {r["book"] for r in rows} == {21}                    # exchange book 28 dropped
    assert {r["key"] for r in rows} >= {"wm_a_ko", "wm_b_dec", "ou_2.5_over", "ou_2.5_under"}
    c = consensus(rows)[500]
    assert c["overround_wm"] > 1
    six = [c[f"wm_{s}_{m}"] for s in "ab" for m in ("ko", "sub", "dec")]
    assert abs(sum(six) - 1) < 1e-9                              # de-vigged together
    red_is_a = corner_markets(c, swapped=False)
    red_is_b = corner_markets(c, swapped=True)
    assert red_is_a["red_ko"]["prob"] == red_is_b["blue_ko"]["prob"] == c["wm_a_ko"]
    assert 0.5 < red_is_a["ou_2.5_over"]["prob"] < 0.6
    assert abs(red_is_a["ou_2.5_under"]["prob"] + red_is_a["ou_2.5_over"]["prob"] - 1) < 1e-9
    # real prices carried through: best price and its book per side
    assert red_is_a["red_ko"]["best_american"] == 200 and red_is_a["red_ko"]["best_book"] == "book_21"
