"""Markets track: per-book line history, sharp vs soft, crowd/model/market, model CLV.

The pure-rule tests need no database. The API tests skip unless DATABASE_URL is SQLite, so they
can never touch the production database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_markets.py
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from app.config import settings
from app.services.ufc.markets_book_history import (
    Quote, book_points, merge_sources, news_spikes, pair_sides, soft_median, stake_steps, weigh_in_at,
)
from app.services.ufc.markets_clv import clv_report, fight_rows
from app.services.ufc.markets_crowd import fan_split

DAY = date(2026, 9, 19)
T = lambda d, h=12: datetime(2026, 9, d, h)

needs_sqlite = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                  reason="needs a scratch SQLite DATABASE_URL")


# --------------------------------------------------------------------------- book history rules

def test_pair_sides_carries_each_side_forward_and_drops_suspended():
    rows = [(T(1), "red", -150, 500.0), (T(1, 13), "blue", 130, None),
            (T(2), "red", -160, None), (T(3), "blue", None, None), (T(4), "blue", 140, 900.0)]
    q = pair_sides(rows)
    assert [(x.red, x.blue) for x in q] == [(-150, 130), (-160, 130), (-160, 140)]
    assert [x.max_stake for x in q] == [500.0, 500.0, 900.0]


def test_merge_prefers_best_source_and_fills_only_outside_its_coverage():
    direct = [Quote(T(10), -150, 130), Quote(T(12), -160, 140)]
    api = [Quote(T(5), -140, 120), Quote(T(11), -999, 600), Quote(T(16), -170, 150)]
    bfo = [Quote(T(3), -130, 110), Quote(T(18), -180, 160)]
    # direct's scraper still read this book on the 15th: its silence to then means "unchanged"
    out, used = merge_sources({"bfo": bfo, "odds_api": api, "direct": direct},
                              {"direct": T(15), "odds_api": T(16)})
    assert [q.t for q in out] == [T(3), T(5), T(10), T(12), T(16), T(18)]
    assert used == ["direct", "odds_api", "bfo"]
    assert all(q.red != -999 for q in out)


def test_book_points_stop_at_cutoff_and_devig():
    cutoff = T(19, 0)
    quotes = [Quote(T(17), -150, 130), Quote(T(18), -150, 130), Quote(T(18, 23), -200, 170),
              Quote(cutoff, -2000, 1000), Quote(T(19, 2), -5000, 2000)]
    pts = book_points(quotes, cutoff)
    assert len(pts) == 2                                     # repeat dropped, cutoff and after dropped
    assert all(datetime.fromisoformat(p["t"]) < cutoff for p in pts)
    assert all(p["days_to_fight"] > 0 for p in pts)
    p = pts[0]
    raw_red, raw_blue = 150 / 250, 100 / 230
    assert p["red_prob"] == round(raw_red / (raw_red + raw_blue), 4)


def test_soft_median_uses_every_book_priced_so_far():
    s = soft_median({
        "FanDuel": [{"t": "a1", "days_to_fight": 3, "red_prob": 0.50}],
        "Bovada": [{"t": "a2", "days_to_fight": 2, "red_prob": 0.60}, {"t": "a4", "days_to_fight": 0.5, "red_prob": 0.40}],
        "BetRivers": [{"t": "a3", "days_to_fight": 1, "red_prob": 0.70}],
    })
    assert [p["red_prob"] for p in s] == [0.5, 0.55, 0.6, 0.5]
    assert [p["n_books"] for p in s] == [1, 2, 3, 3]


def test_stake_steps_only_changes_before_cutoff():
    q = [Quote(T(1), -1, 1, 500.0), Quote(T(2), -1, 1, 500.0), Quote(T(3), -1, 1, 2000.0), Quote(T(5), -1, 1, 9.0)]
    assert [s["max_stake"] for s in stake_steps(q, T(4))] == [500.0, 2000.0]


def test_news_spike_needs_volume_and_a_jump():
    lo, hi = date(2026, 9, 1), date(2026, 9, 18)
    days = {date(2026, 9, d): 2 for d in range(1, 19)}
    days[date(2026, 9, 10)] = 4          # under the floor
    days[date(2026, 9, 15)] = 12
    days[date(2026, 9, 17)] = 30
    days[date(2026, 9, 16)] = 20
    assert news_spikes(days, lo, hi) == [(date(2026, 9, 16), 20), (date(2026, 9, 17), 30)]


def test_weigh_in_is_the_day_before():
    assert weigh_in_at(DAY, None) == datetime(2026, 9, 18, 16)
    assert weigh_in_at(DAY, datetime(2026, 9, 19, 22)) == datetime(2026, 9, 18, 13)


# --------------------------------------------------------------------------- crowd

def test_fan_split_maps_mmad_fighters_to_corners():
    assert fan_split(70, 20, 10, swapped=False) == {"n": 100, "red": 0.7, "blue": 0.2, "draw": 0.1}
    assert fan_split(70, 20, 10, swapped=True)["red"] == 0.2
    assert fan_split(0, 0, None, swapped=False) is None


# --------------------------------------------------------------------------- CLV

def _ledger():
    rows = []

    def fight(fid, date_, p_red, q_open_red, q_close_red, red_won, source="backtest"):
        for basis, q in (("open", q_open_red), ("close", q_close_red)):
            for market, p, qq, won in (("ml_red", p_red, q, red_won), ("ml_blue", 1 - p_red, 1 - q, not red_won)):
                rows.append({"fight_id": fid, "group": "winner", "market": market, "basis": basis, "p_raw": p,
                             "q": qq, "won": won, "source": source, "date": date_, "event_id": "e" + fid,
                             "event_name": "E", "red_name": "R" + fid, "blue_name": "B" + fid})

    fight("1", "2025-01-04", 0.60, 0.50, 0.55, True)                    # model red, line moved to it
    fight("2", "2025-01-18", 0.30, 0.50, 0.53, False)                   # model blue, line moved away
    fight("3", "2025-02-01", 0.50, 0.50, 0.40, True)                    # no view
    fight("4", "2026-11-07", 0.75, 0.60, 0.602, True, source="live")    # flat
    rows.append({"fight_id": "1", "group": "method", "market": "red_ko", "basis": "close", "p_raw": 0.9,
                 "q": 0.1, "won": False, "source": "backtest", "date": "2025-01-04", "event_id": "e1",
                 "event_name": "E", "red_name": "R1", "blue_name": "B1"})
    return pd.DataFrame(rows)


def test_clv_takes_the_models_side_against_the_open():
    x = fight_rows(_ledger()).set_index("fight_id")
    assert list(x.index) == ["1", "2", "4"]
    assert x.loc["1", "side"] == "red" and x.loc["1", "clv"] == pytest.approx(0.05)
    assert x.loc["2", "side"] == "blue" and x.loc["2", "clv"] == pytest.approx(-0.03)
    assert bool(x.loc["1", "toward"]) and bool(x.loc["2", "against"])
    assert not x.loc["4", "toward"] and not x.loc["4", "against"]


def test_clv_report_filters_and_aggregates():
    r = clv_report(_ledger())
    assert r["summary"]["n"] == 3
    assert r["summary"]["toward"] == pytest.approx(1 / 3, abs=1e-4)
    assert [m["month"] for m in r["timeline"]] == ["2025-01", "2026-11"]
    assert r["timeline"][0]["avg_clv"] == pytest.approx(0.01)
    assert r["recent"][0]["fight_id"] == "4"
    live = clv_report(_ledger(), source="live")
    assert live["summary"]["n"] == 1 and live["recent"][0]["source"] == "live"
    assert clv_report(_ledger(), min_edge=0.12)["summary"]["n"] == 2
    assert clv_report(_ledger(), start="2026-01-01", end="2026-12-31")["summary"]["n"] == 1


# --------------------------------------------------------------------------- endpoints over SQLite

BIG = 9_007_199_254_740_993          # past 2**53: must survive as a string
CUT = datetime(2026, 9, 19, 23, 0)   # card start published by the venue = the opener's cutoff


@pytest.fixture()
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.database import Base, SessionLocal, engine
    from app.models.ufc import (
        MMADDecision, UFCBookMarketHistory, UFCCancelledBout, UFCEvent, UFCFight, UFCFighter,
        UFCFightOddsHistory, UFCFightOpenClose, UFCFightPrediction, UFCNewsMention, UFCPredictionMarket,
        UFCPredictionMarketHistory,
    )
    from app.routers import ufc as ufc_router
    from app.services import response_cache

    tables = [UFCFighter.__table__, UFCEvent.__table__, UFCFight.__table__, UFCPredictionMarket.__table__,
              UFCPredictionMarketHistory.__table__, UFCBookMarketHistory.__table__, UFCFightOddsHistory.__table__,
              UFCCancelledBout.__table__, UFCNewsMention.__table__, UFCFightPrediction.__table__,
              UFCFightOpenClose.__table__, MMADDecision.__table__]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)
    s = SessionLocal()
    for fid, last in ((1, "Red"), (2, "Blue"), (3, "Gone")):
        s.add(UFCFighter(id=fid, ufcstats_id=f"u{fid}", first_name="F", last_name=last))
    s.add(UFCEvent(id=1, ufcstats_id="e1", name="Card", date=DAY))
    s.add(UFCFight(id=BIG, ufcstats_id="f1", event_id=1, date=DAY, card_position=0,
                   red_fighter_id=1, blue_fighter_id=2, winner_id=1, method="Decision - Split"))
    s.add(UFCPredictionMarket(id=1, fight_id=BIG, platform="kalshi", external_event_id="x", external_market_id="m1",
                              market_type="moneyline", outcome_key="red", side="red", card_start_at=CUT))

    def side_rows(book, t, red, blue, stake=None, i=[0]):
        for side, am in (("red", red), ("blue", blue)):
            i[0] += 1
            s.add(UFCBookMarketHistory(id=i[0], fight_id=BIG, book=book, external_event_id="x",
                                       external_market_id="m", external_selection_id=side, market_type="ML",
                                       market_name="Moneyline", selection=side, side=side,
                                       market_key=f"moneyline_{side}", american=am, max_stake=stake,
                                       captured_at=t, days_to_fight=0))

    for book, base in (("Pinnacle", -150), ("FanDuel", -160), ("Bovada", -170), ("BetRivers", -140)):
        side_rows(book, T(16), base, 130, 1000.0 if book == "Pinnacle" else None)
        side_rows(book, T(18), base - 20, 150, 4000.0 if book == "Pinnacle" else None)
        side_rows(book, CUT + timedelta(minutes=30), -3000, 1500, 100.0 if book == "Pinnacle" else None)  # in play
    s.add(UFCFightOddsHistory(id=1, fight_id=BIG, bookmaker="DraftKings", red_odds=-155, blue_odds=135,
                              red_implied_prob=0.6, blue_implied_prob=0.4, captured_at=T(17)))
    s.add(UFCFightOddsHistory(id=2, fight_id=BIG, bookmaker="BFO:DraftKings", red_odds=-5000, blue_odds=2000,
                              red_implied_prob=0.97, blue_implied_prob=0.03, captured_at=T(20)))   # post-fight
    s.add(UFCCancelledBout(id=1, ufcstats_id="c1", event_id=1, event_date=DAY, red_fighter_id=1,
                           blue_fighter_id=3, removed_at=T(12), created_at=T(12)))
    s.add(UFCNewsMention(id=1, fighter_id=1, day=date(2026, 9, 17), count=25))
    s.add(UFCFightPrediction(id=1, fight_id=BIG, predicted_winner="red", confidence=0.1, red_prob=0.6,
                             model_prob=0.58, created_at=T(17, 9)))
    s.add(UFCFightOpenClose(id=1, fight_id=BIG, source="bfo", bookmaker="Consensus", red_close_prob=0.62))
    s.add(MMADDecision(id=1, mmad_decision_id=1, fighter_a_name="Blue", fighter_b_name="Red", outcome="win",
                       fight_id=BIG, swapped=True, match_status="verified", fan_n=100, fan_a=60, fan_b=35,
                       fan_draw=5))
    s.commit()
    s.close()
    response_cache.invalidate()

    app = FastAPI()
    app.include_router(ufc_router.router)
    yield TestClient(app)
    response_cache.invalidate()


@needs_sqlite
def test_book_history_endpoint_never_serves_past_the_cutoff(client):
    r = client.get(f"/ufc/fights/{BIG}/book-history")
    assert r.status_code == 200
    d = r.json()
    assert d["fight_id"] == str(BIG) and d["cutoff"] == CUT.isoformat() and d["started"]
    books = {b["book"]: b for b in d["books"]}
    assert set(books) == {"Pinnacle", "FanDuel", "Bovada", "BetRivers", "DraftKings"}
    assert d["books"][0]["book"] == "Pinnacle"
    every = [p for b in d["books"] for p in b["points"]] + d["sharp_soft"]["sharp"] \
        + d["sharp_soft"]["soft_median"] + d["sharp_soft"]["max_stake"]
    assert every and all(datetime.fromisoformat(p["t"]) < CUT and p["days_to_fight"] > 0 for p in every)
    assert all(0.4 < p["red_prob"] < 0.75 for b in d["books"] for p in b["points"])
    assert [s["max_stake"] for s in d["sharp_soft"]["max_stake"]] == [1000.0, 4000.0]
    kinds = {m["kind"]: m for m in d["markers"]}
    assert set(kinds) == {"weigh_in", "opponent_change", "news", "model"}
    assert "Gone" in kinds["opponent_change"]["detail"] and kinds["opponent_change"]["short_notice"]
    assert all(datetime.fromisoformat(m["t"]) < CUT for m in d["markers"])
    assert client.get("/ufc/fights/42/book-history").status_code == 404


@needs_sqlite
def test_crowd_endpoint(client):
    d = client.get(f"/ufc/fights/{BIG}/crowd").json()
    assert d["fight_id"] == str(BIG) and d["winner"] == "red"
    assert d["fans"] == {"n": 100, "red": 0.35, "blue": 0.6, "draw": 0.05}
    assert d["model"]["red_prob"] == 0.6 and d["market"] == {"red_close_prob": 0.62, "source": "consensus"}


@needs_sqlite
def test_clv_endpoint(client, monkeypatch):
    from app.services.ufc import model_results
    df = _ledger()
    monkeypatch.setattr(model_results, "load_ledger", lambda: df)
    d = client.get("/ufc/model/clv?source=live").json()
    assert d["summary"]["n"] == 1 and isinstance(d["recent"][0]["fight_id"], str)
    assert client.get("/ufc/model/clv?source=bogus").status_code == 400
