"""Past Model Results: ledger bet rule and the aggregation views, on a synthetic ledger."""
import pandas as pd
import pytest

from app.services.ufc import backtest_ledger as L
from app.services.ufc import model_results as M


def test_bets_every_positive_side_and_top():
    df = pd.DataFrame([{"fight_id": "1",
                        "p_a": 0.30, "best_a": 300, "med_a": 250, "q_a": None, "won_a": True,     # EV .05
                        "p_b": 0.25, "best_b": 400, "med_b": 350, "q_b": None, "won_b": False,    # EV .125
                        "p_c": 0.10, "best_c": 500, "med_c": 500, "q_c": None, "won_c": False}])  # EV -.4
    sides = [("a", "red_ko"), ("b", "blue_ko"), ("c", "red_sub")]
    b = L.bets(df, sides, lambda s: "fam")
    assert sorted(b["market"]) == ["blue_ko", "red_ko"]                 # every +EV side, not just the best
    assert b.set_index("market")["is_top"].to_dict() == {"red_ko": False, "blue_ko": True}
    a = b.set_index("market").loc["red_ko"]
    assert a["profit_typical"] == pytest.approx(2.5) and a["profit_best"] == pytest.approx(3.0)
    assert len(L.bets(df, sides, lambda s: "fam", keep_all=True)) == 3   # keep_all adds the -EV side
    assert not L.bets(df, sides, lambda s: "fam", keep_all=True).set_index("market").loc["red_sub", "bet"]


def _row(fight, event, date, family, market, basis, ev, won, dec, bet=True, top=True, pos=0, **kw):
    return {"fight_id": fight, "event_id": event, "event_name": f"E{event}", "date": date, "family": family,
            "market": market, "basis": basis, "group": L.group_of(family), "p_raw": 0.5, "p": 0.5, "q": 0.45,
            "ev": ev, "decimal": dec, "decimal_best": dec + 0.1, "won": won, "bet": bet, "is_top": top,
            "profit_typical": dec - 1 if won else -1.0, "profit_best": dec - 0.9 if won else -1.0,
            "clv": kw.get("clv"), "red_name": "Red", "blue_name": "Blue", "weight_class": kw.get("wc", "Lightweight Bout"),
            "card_position": pos, "rounds": 3, "winner_side": "red", "result_cls": "dec",
            "result_method": "Decision - Unanimous", "finish_round": 3, "finish_time": "5:00"}


@pytest.fixture
def ledger():
    rows = [
        _row("f1", "1", "2024-01-06", "winner_open", "ml_red", "open", 0.06, True, 2.0, clv=0.02),
        _row("f1", "1", "2024-01-06", "winner_close", "ml_red", "close", 0.02, True, 1.9),
        _row("f1", "1", "2024-01-06", "winner_close", "ml_blue", "close", -0.1, False, 2.0, bet=False, top=False),
        _row("f1", "1", "2024-01-06", "decision_yes", "dec_yes", "close", 0.10, True, 1.8),
        _row("f2", "1", "2024-01-06", "sixway_ko_dog", "blue_ko", "close", 0.25, False, 6.0, pos=5),
        _row("f2", "1", "2024-01-06", "sixway_ko_fav", "red_ko", "close", 0.04, False, 3.0, top=False, pos=5),
        _row("f3", "2", "2025-03-01", "winner_open", "ml_blue", "open", 0.04, False, 2.5, clv=-0.01,
             wc="UFC Women's Flyweight Title Bout"),
        _row("f3", "2", "2025-03-01", "winner_close", "ml_blue", "close", 0.03, False, 2.4,
             wc="UFC Women's Flyweight Title Bout"),
    ]
    return M.prepare(pd.DataFrame(rows))


def test_basis_and_props_close_only(ledger):
    rows = M.priced(ledger, M.Filters(basis="open"))
    assert set(rows[rows["group"] == "winner"]["basis"]) == {"open"}
    assert rows[rows["group"] != "winner"]["close_only"].all() and not rows[rows["group"] == "winner"]["close_only"].any()
    only_ml = M.priced(ledger, M.Filters(basis="open", props_at_open=False))
    assert set(only_ml["group"]) == {"winner"}
    assert not M.priced(ledger, M.Filters(basis="close"))["close_only"].any()


def test_filters_min_ev_top_only_dates_groups(ledger):
    f = M.Filters(basis="close")
    assert len(M.bets_of(M.priced(ledger, f), f)) == 5
    f = M.Filters(basis="close", min_ev=0.05)
    assert sorted(M.bets_of(M.priced(ledger, f), f)["market"]) == ["blue_ko", "dec_yes"]
    f = M.Filters(basis="close", top_only=True)
    assert "red_ko" not in set(M.bets_of(M.priced(ledger, f), f)["market"])
    f = M.Filters(basis="close", start="2025-01-01")
    assert set(M.priced(ledger, f)["event_id"]) == {"2"}
    f = M.Filters(basis="close", groups=["method"])
    assert set(M.priced(ledger, f)["group"]) == {"method"}


def test_events_summary_money(ledger):
    s = M.events_summary(ledger, M.Filters(basis="close"))
    # bets: ml_red +0.9, dec_yes +0.8, blue_ko -1, red_ko -1, ml_blue (f3) -1
    assert s["totals"]["n"] == 5 and s["totals"]["profit"] == pytest.approx(-1.3)
    assert s["totals"]["roi"] == pytest.approx(-0.26)
    assert [e["event_id"] for e in s["events"]] == ["2", "1"]                # newest first
    assert [c["cumulative"] for c in s["cumulative"]] == pytest.approx([-0.3, -1.3])
    best = M.events_summary(ledger, M.Filters(basis="close", price="best"))
    assert best["totals"]["profit"] == pytest.approx(-1.1)
    assert M.events_summary(ledger, M.Filters(basis="open"))["props_close_only"]


def test_event_detail(ledger):
    d = M.event_detail(ledger, "1", M.Filters(basis="close"))
    assert [f["fight_id"] for f in d["fights"]] == ["f1", "f2"]               # card order
    f1 = d["fights"][0]
    ml = next(m for m in f1["markets"] if m["market"] == "Moneyline")
    assert [s["bet"] for s in ml["sides"]] == [True, False]                  # the -EV side is shown, not bet
    assert ml["sides"][1]["profit"] is None and ml["sides"][0]["price"] == -111
    assert f1["result"] == "Red by Decision"
    assert M.event_detail(ledger, "999", M.Filters()) is None


def test_splits_partition_and_order(ledger):
    s = M.splits(ledger, M.Filters(basis="close"))
    for name, sp in s["splits"].items():
        assert sum(r["n"] for r in sp["rows"]) == s["totals"]["n"], name
    keys = [r["key"] for r in s["splits"]["ev_band"]["rows"]]
    assert keys == sorted(keys, key=lambda k: M._ORDERED["ev_band"].index(k))
    assert {r["key"] for r in s["splits"]["title"]["rows"]} == {"Title fight", "Non-title"}
    assert "Women's Flyweight" in {r["key"] for r in s["splits"]["division"]["rows"]}
    assert s["clv"] is None                                                   # CLV is for opening bets
    o = M.splits(ledger, M.Filters(basis="open"))["clv"]
    assert o["n"] == 2 and o["beat_close"] == pytest.approx(0.5) and o["roi_beat"] == pytest.approx(1.0)


def test_labels_and_american():
    assert M.american(2.5) == 150 and M.american(1.5) == -200 and M.american(None) is None
    assert M.family_label("sixway_dec_fav") == "Favourite by decision"
    assert M.family_label("ou_1_5_under") == "Under 1.5 rounds"
    assert M.family_label("starts_r3_no") == "Starts round 3: no"
    assert M.side_label("itd_blue_no", "A", "B") == "B wins inside distance: no"
    assert M.market_name("ou_2.5_over", "A", "B") == "Total rounds 2.5"


def test_forward_prediction_shapes():
    from app.services.ufc import forward_predictions as fp
    curve = [{"t": t, "s": 1 - t / 30} for t in (0, 1.25, 2.5, 3.75, 5, 7.5, 10, 12.5, 15)]
    g = fp.s_grid(curve)
    assert len(g) == len(fp.S_COLS) and g[0] == 1 and g[2] == pytest.approx(1 - 5 / 30, abs=1e-5)
    assert g[-1] == g[6]                          # a three-round curve carries flat past 15 min
    six = [0.6 * 0.3, 0.6 * 0.2, 0.6 * 0.5, 0.4 * 0.5, 0.4 * 0.1, 0.4 * 0.4]   # p_red x conditional
    assert fp.conditional(six) == pytest.approx([0.3, 0.2, 0.5, 0.5, 0.1, 0.4], abs=1e-5)
    assert fp.conditional([None] * 6) is None


def test_source_split_and_event_flags(ledger):
    df = ledger.copy()
    df.loc[df["event_id"] == "2", "source"] = "live"
    df.loc[df["event_id"] == "2", "price_source"] = "watch"
    df = M.prepare(df)
    s = M.splits(df, M.Filters(basis="close"))
    keys = [r["key"] for r in s["splits"]["source"]["rows"]]
    assert keys == ["Walk-forward backtest", "Live: served before the fight"]
    ev = {e["event_id"]: e for e in M.events_summary(df, M.Filters())["events"]}
    assert ev["2"]["sources"] == ["live"] and ev["2"]["watch_prices"] and not ev["1"]["watch_prices"]
