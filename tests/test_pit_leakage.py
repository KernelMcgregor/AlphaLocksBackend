"""As-of checks for the point-in-time feature blocks (app/services/ufc/pit).

Every block must give the same values for a fight whether or not anything after that fight
exists: computing on a prefix of history matches the full computation, and changing a later
result (or later stats) changes nothing earlier. Synthetic data; no database.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.services.ufc import style_habits
from app.services.ufc.pit import style_pit

STAT_KEYS = ["str_att", "str_landed", "td_att", "td_landed", "sub_att", "rev", "kd", "ctrl_min",
             "head_att", "body_att", "leg_att", "dist_att", "clinch_att", "ground_att",
             "ground_landed", "nonsig_att", "opp_ctrl_min"]


def _synthetic(n_fighters=260, n_bouts=2600, seed=0):
    """Bouts (style_habits.Bout per corner) plus the per-fighter frame the blocks read."""
    rng = np.random.default_rng(seed)
    style = rng.normal(size=(n_fighters, 4))
    start = date(2007, 1, 1)
    bouts, rows = {}, []
    for k in range(n_bouts):
        a, b = rng.choice(n_fighters, 2, replace=False)
        d = start + timedelta(days=int(k * 3650 / n_bouts))
        minutes = float(rng.uniform(3, 15))
        red_wins = rng.random() < 0.5
        for me, opp, corner in ((a, b, "red"), (b, a, "blue")):
            s = style[me]
            c = {key: float(rng.poisson(max(0.2, 3 + s[i % 4]) * minutes / 3)) for i, key in enumerate(STAT_KEYS)}
            c["ctrl_min"] = min(c["ctrl_min"] / 5, minutes / 2)
            c["opp_ctrl_min"] = min(c["opp_ctrl_min"] / 5, minutes / 2)
            c.update(minutes=minutes, standing_min=max(minutes - c["ctrl_min"] - c["opp_ctrl_min"], 0.5),
                     ground_min=c["ctrl_min"] + c["opp_ctrl_min"])
            won = red_wins if corner == "red" else not red_wins
            bouts.setdefault(int(me), []).append(style_habits.Bout(k, int(me), int(opp), d, won, "dec", minutes * 60, c, {}))
            rows.append({"fight_id": k, "stats_fighter_id": int(me), "corner": corner, "date": pd.Timestamp(d),
                         "red_fighter_id": int(a), "blue_fighter_id": int(b), "winner_id": int(a if red_wins else b),
                         "won": float(won), "elo_expected": float(rng.uniform(0.3, 0.7)),
                         "weight_class": "Lightweight Bout", "xsres_sig": float(rng.normal()),
                         "xsres_td": float(rng.normal()), "xsres_ctrl": float(rng.normal())})
    return bouts, pd.DataFrame(rows)


def _cut(bouts, fighters, t):
    b = {f: [x for x in bs if x.date < t] for f, bs in bouts.items()}
    return {f: bs for f, bs in b.items() if bs}, fighters[fighters.date < pd.Timestamp(t)]


@pytest.fixture(scope="module")
def data():
    bouts, fighters = _synthetic()
    full = style_pit.features_from(bouts, fighters).set_index(["fight_id", "fighter_id"]).sort_index()
    return bouts, fighters, full


def test_style_block_has_values(data):
    _, _, full = data
    late = full.loc[full.index.get_level_values(0) > 2000]
    for c in ("sty_cos", "sty_like_resid_p4", "sty_analog_resid_p4", "sty_like_n60"):
        assert late[c].notna().mean() > 0.5, c


def test_style_prefix_matches_full(data):
    bouts, fighters, full = data
    t = date(2013, 6, 15)
    pre = style_pit.features_from(*_cut(bouts, fighters, t)).set_index(["fight_id", "fighter_id"]).sort_index()
    common = full.loc[pre.index, pre.columns]
    pd.testing.assert_frame_equal(pre, common, check_exact=False, rtol=1e-9, atol=1e-12)


def test_style_later_results_change_nothing_earlier(data):
    bouts, fighters, full = data
    t = pd.Timestamp(date(2013, 6, 15))
    f2 = fighters.copy()
    later = f2.date >= t
    f2.loc[later, "won"] = 1.0 - f2.loc[later, "won"]
    f2.loc[later, "elo_expected"] = 0.5
    b2 = {f: [style_habits.Bout(x.fight_id, x.fighter, x.opponent, x.date, x.won, x.method, x.seconds,
                                {**x.c, "td_att": x.c["td_att"] * 3} if x.date >= t.date() else x.c, x.o)
              for x in bs] for f, bs in bouts.items()}
    alt = style_pit.features_from(b2, f2).set_index(["fight_id", "fighter_id"]).sort_index()
    early = full.index[full.index.get_level_values(0).isin(fighters.loc[~later, "fight_id"])]
    pd.testing.assert_frame_equal(alt.loc[early], full.loc[early], check_exact=False, rtol=1e-9, atol=1e-12)


def test_style_debut_has_no_fingerprint(data):
    bouts, fighters, full = data
    first = fighters.sort_values("date").groupby("stats_fighter_id").head(1)
    idx = pd.MultiIndex.from_frame(first[["fight_id", "stats_fighter_id"]])
    assert full.loc[idx, "sty_atyp"].isna().all()


def test_rank_join_never_reads_fight_day():
    from app.services.ufc.pit import rank_pit
    hist = pd.DataFrame({
        "fighter_id": [1, 1, 1, 1], "weight_class": ["lightweight"] * 4,
        "as_of": [date(2020, 1, 1), date(2020, 3, 1), date(2020, 6, 1), date(2020, 9, 1)],
        "rank": [10, 8, 5, 1], "score": [500.0, 600.0, 700.0, 999.0], "total_ranked": [50] * 4, "sos": [40] * 4})
    fighters = pd.DataFrame({"fight_id": [7], "stats_fighter_id": [1], "date": [pd.Timestamp(2020, 9, 1)],
                             "weight_class": ["Lightweight Bout"]})
    r = rank_pit.features_from(hist, fighters).iloc[0]
    assert r.rk_score == 700.0                      # the 09-01 row holds that card's result
    assert r.rk_delta_180 == 700.0 - 600.0          # vs the last row before 2020-03-05 (03-01)


def _synthetic_rounds(n_fighters=120, n_bouts=1500, seed=1):
    rng = np.random.default_rng(seed)
    pref = rng.normal(size=(n_fighters, 3))
    rows = []
    for k in range(n_bouts):
        a, b = rng.choice(n_fighters, 2, replace=False)
        d = pd.Timestamp(date(2009, 1, 1) + timedelta(days=int(k * 4700 / n_bouts)))
        for rnd in (1, 2, 3):
            r = {"fight_id": k, "red": int(a), "blue": int(b), "date": d, "round": rnd, "secs": 300.0}
            for side, f in (("r", a), ("b", b)):
                p = pref[f]
                r.update({f"{side}_sig_str_attempted": float(rng.poisson(30 * np.exp(0.3 * p[0]))),
                          f"{side}_distance_attempted": float(rng.poisson(20)),
                          f"{side}_clinch_attempted": float(rng.poisson(4)),
                          f"{side}_ground_attempted": float(rng.poisson(3 * np.exp(0.5 * p[1]))),
                          f"{side}_td_attempted": float(rng.poisson(1 + max(p[2], 0))),
                          f"{side}_td_landed": float(rng.poisson(0.4)),
                          f"{side}_ctrl_seconds": float(rng.uniform(0, 60))})
            rows.append(r)
    return pd.DataFrame(rows)


def test_tug_states_prefix_after_tuning_window():
    pytest.importorskip("app.services.ufc.dictation")   # untracked research module
    from app.services.ufc.pit import tug_pit
    rounds = _synthetic_rounds()
    full, _ = tug_pit.state_features(rounds)
    t = pd.Timestamp(2020, 6, 1)            # after the 2012-2018 tuning window
    pre, _ = tug_pit.state_features(rounds[rounds.date < t].reset_index(drop=True))
    assert pre and set(pre) <= set(full)
    for key, vals in pre.items():
        for c, v in vals.items():
            assert full[key][c] == pytest.approx(v, rel=1e-9, abs=1e-12), (key, c)


def test_tug_bridge_prefix_matches_full(data):
    pytest.importorskip("app.services.ufc.dictation")   # untracked research module
    from app.services.ufc.pit import tug_pit
    bouts, fighters, _ = data
    bb = {f: [style_habits.Bout(x.fight_id, x.fighter, x.opponent, x.date, x.won, x.method, x.seconds,
                                {**x.c, "dist_l": x.c["str_landed"] * 0.6, "clinch_l": x.c["str_landed"] * 0.2,
                                 "ground_l": x.c["ground_landed"], "ground_min": x.c["ground_min"]}, x.o) for x in bs] for f, bs in bouts.items()}
    fights = (fighters[fighters.corner == "red"][["fight_id", "red_fighter_id", "blue_fighter_id", "date"]]
              .rename(columns={"red_fighter_id": "red", "blue_fighter_id": "blue"}))
    shares = {f: (0.1, 0.15) for f in fights.fight_id}
    full = tug_pit.bridge_features(bb, fights, shares)
    t = date(2013, 6, 15)
    cut = {f: [x for x in bs if x.date < t] for f, bs in bb.items()}
    pre = tug_pit.bridge_features({f: v for f, v in cut.items() if v}, fights[fights.date < pd.Timestamp(t)], shares)
    assert len(pre) > 100
    for key, vals in pre.items():
        for c, v in vals.items():
            assert full[key][c] == pytest.approx(v, rel=1e-9, abs=1e-12), (key, c)


def test_dtw_never_reads_own_fight(data):
    from app.services.ufc.pit import dtw_pit
    _, fighters, _ = data
    rng = np.random.default_rng(3)
    ids = fighters.fight_id.unique()
    dtw = pd.DataFrame({"fight_id": ids, "p_red": rng.uniform(0, 1, len(ids))})
    dtw["p_blue"] = 1 - dtw.p_red
    full = dtw_pit.features_from(dtw, fighters).set_index(["fight_id", "fighter_id"]).sort_index()
    t = pd.Timestamp(2013, 6, 15)
    later = fighters.loc[fighters.date >= t, "fight_id"].unique()
    dtw2 = dtw.copy()
    dtw2.loc[dtw2.fight_id.isin(later), "p_red"] = 0.99      # rewrite every later fight's score
    dtw2["p_blue"] = 1 - dtw2.p_red
    alt = dtw_pit.features_from(dtw2, fighters).set_index(["fight_id", "fighter_id"]).sort_index()
    early = full.index[~full.index.get_level_values(0).isin(later)]
    first_late = fighters[fighters.date >= t].sort_values("date").fight_id.iloc[0]
    keep = early.append(full.index[full.index.get_level_values(0) == first_late])   # own row unread
    pd.testing.assert_frame_equal(alt.loc[keep], full.loc[keep])


def test_resume_prefix_matches_full(data):
    from app.services.ufc.pit import resume_pit
    _, fighters, _ = data
    f = fighters.assign(elo=1500 + 400 * (fighters.elo_expected - 0.5), method="KO/TKO")
    full = resume_pit.features_from(f).set_index(["fight_id", "fighter_id"]).sort_index()
    t = pd.Timestamp(2013, 6, 15)
    pre = resume_pit.features_from(f[f.date < t]).set_index(["fight_id", "fighter_id"]).sort_index()
    pd.testing.assert_frame_equal(pre, full.loc[pre.index, pre.columns])
    assert full["rs_mean"].notna().mean() > 0.5


def test_newcomer_experience_and_prior_use_earlier_dates_only(data):
    from app.services.ufc.pit import newcomer_pit
    _, fighters, _ = data
    f = fighters.assign(avg_x=np.random.default_rng(5).normal(size=len(fighters)))
    dev = set(f.fight_id.unique()[:50].tolist())
    full = newcomer_pit.experience(f, dev).set_index(["fight_id", "fighter_id"]).sort_index()
    t = pd.Timestamp(2013, 6, 15)
    pre = newcomer_pit.experience(f[f.date < t], dev).set_index(["fight_id", "fighter_id"]).sort_index()
    pd.testing.assert_frame_equal(pre, full.loc[pre.index])
    first = f.sort_values("date").groupby("stats_fighter_id").head(1)
    idx = pd.MultiIndex.from_frame(first[["fight_id", "stats_fighter_id"]])
    assert (full.loc[idx, "nu_n_ufc"] == 0).all() and (full.loc[idx, "nu_wpct"] == 0.5).all()
    p_full = newcomer_pit.division_prior(f, ["avg_x"])
    p_pre = newcomer_pit.division_prior(f[f.date < t], ["avg_x"])
    pd.testing.assert_series_equal(p_pre["avg_x"], p_full.loc[p_pre.index, "avg_x"])
    # a row's prior never includes its own date
    d0 = f.date.min()
    assert p_full.loc[f.date == d0, "avg_x"].isna().all()


def _synthetic_sherdog(n_f=300, n_b=6000, seed=7):
    rng = np.random.default_rng(seed)
    skill = rng.normal(size=n_f)
    promos = np.array(["LFA", "CFFC", "Local A", "Local B", "DWCS", "UFC", "TUF"])
    rows = []
    for k in range(n_b):
        a, b = rng.choice(n_f, 2, replace=False)
        p = 1 / (1 + np.exp(-(skill[a] - skill[b])))
        rows.append((int(a) + 1, int(b) + 1, date(2004, 1, 1) + timedelta(days=int(k * 7000 / n_b)),
                     float(rng.random() < p), rng.choice(["KO/TKO", "SUB", "DEC"]), str(rng.choice(promos))))
    return pd.DataFrame(rows, columns=["a", "b", "date", "a_score", "method_class", "promo"])


def test_pro_block_prefix_matches_full():
    from app.services.ufc.pit import pro_pit
    from app.services.ufc.pro_rating import Params
    bouts = _synthetic_sherdog()
    link = {u: u for u in range(1, 301)}
    rows = pd.DataFrame({"fight_id": range(400), "stats_fighter_id": [1 + i % 300 for i in range(400)],
                         "date": [pd.Timestamp(2010, 1, 1) + pd.Timedelta(days=10 * i) for i in range(400)]})
    am = pd.DataFrame({"sherdog_id": [1, 1, 2], "date": pd.to_datetime(["2003-01-01", "2012-01-01", "2003-05-01"]),
                       "result": ["W", "L", "W"]})
    for prm in (Params(), Params(rd=True, promo=True, fin=1.25)):
        full = pro_pit.features_from(bouts, rows, link, am, prm).set_index("fight_id")
        t = date(2013, 1, 1)
        pre = pro_pit.features_from(bouts[bouts.date < t].reset_index(drop=True), rows[rows.date < pd.Timestamp(t)],
                                    link, am[am.date < pd.Timestamp(t)], prm).set_index("fight_id")
        assert pre["pro2_rating"].notna().mean() > 0.9
        pd.testing.assert_frame_equal(pre, full.loc[pre.index, pre.columns], check_exact=False, rtol=1e-9)


def test_mismatch_blend_leans_on_sherdog_for_newcomers():
    from app.services.ufc.pit.mismatch_pit import compute_features
    def corner(n, elo_exp, pro, rd):
        return pd.DataFrame({"nu_n_ufc": [n], "elo_expected": [elo_exp], "pro2_rating": [pro], "pro2_rd": [rd],
                             "elo": [1500.0], "glicko_pts": [0.0], "glicko_meta_sigma": [100.0]})
    # debut vs debut: UFC Elo says 50/50, Sherdog says red is much better -> q_best follows Sherdog
    f = compute_features(corner(0, 0.5, 1800, 80), corner(0, 0.5, 1500, 80))
    assert f["q_best"][0] > 0.8
    # 20-fight veterans: UFC Elo dominates
    f = compute_features(corner(20, 0.7, 1500, 80), corner(20, 0.3, 1500, 80))
    assert 0.65 < f["q_best"][0] < 0.72
    # unknown Sherdog side: falls back to UFC Elo
    f = compute_features(corner(0, 0.6, np.nan, np.nan), corner(0, 0.4, 1500, 80))
    assert abs(f["q_best"][0] - 0.6) < 1e-9
