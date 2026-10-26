"""Bout cutoffs: the last moment an exchange price counts as pre-fight.

The pure-rule tests need no database. The rest skip unless DATABASE_URL is SQLite, so they can
never touch the production database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_bout_cutoffs.py
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app.config import settings
from app.services.ufc.prediction_markets.common import BOUT_LEAD, Bout, bout_cutoffs, parse_ts

DAY = date(2026, 9, 19)
T = lambda h, m=0: datetime(2026, 9, 19, h, m) if h >= 12 else datetime(2026, 9, 20, h, m)

needs_sqlite = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                  reason="needs a scratch SQLite DATABASE_URL")


# --------------------------------------------------------------------------- the rule

def test_bout_is_cut_at_the_previous_bouts_close():
    # card_position 0 = main event; higher = earlier on the card.
    bouts = [Bout(1, 2, T(22, 40), None, DAY), Bout(2, 1, T(23, 30), None, DAY),
             Bout(3, 0, T(1, 10), None, DAY)]
    cut = bout_cutoffs(bouts)
    assert cut[2] == T(22, 40)
    assert cut[3] == T(23, 30)
    # Nothing below the opener and no card start: its own close minus the lead.
    assert cut[1] == T(22, 40) - BOUT_LEAD


def test_card_start_floors_every_bout_and_cuts_the_opener():
    start = T(21, 0)
    bouts = [Bout(1, 1, T(21, 40), start, DAY), Bout(2, 0, None, start, DAY)]
    cut = bout_cutoffs(bouts)
    assert cut[1] == start
    assert cut[2] == T(21, 40)


def test_upcoming_card_with_nothing_known_falls_back_to_event_midnight():
    cut = bout_cutoffs([Bout(1, 0, None, None, DAY)])
    assert cut[1] == datetime(2026, 9, 19)


def test_scheduled_slot_is_never_used():
    """The bug this replaces: an early prelim anchored on Kalshi's 03:00 slot had closed at 22:00."""
    cut = bout_cutoffs([Bout(1, 11, T(22, 0), T(21, 0), DAY)])
    assert cut[1] <= T(22, 0) - timedelta(minutes=10)


def test_bad_card_start_after_the_result_is_ignored():
    cut = bout_cutoffs([Bout(1, 0, T(22, 0), T(23, 0), DAY)])
    assert cut[1] == T(22, 0) - BOUT_LEAD


def test_wrong_card_order_cannot_cut_inside_the_bout():
    # Bout 2 is listed above bout 1 but actually finished first; bout 1's close must not be
    # taken as bout 2's start, since it is after bout 2's own result.
    bouts = [Bout(1, 2, T(23, 30), None, DAY), Bout(2, 1, T(23, 25), None, DAY)]
    assert bout_cutoffs(bouts)[2] == T(23, 25) - BOUT_LEAD


def test_pinned_curve_caps_the_cutoff():
    # A late venue close would put the cutoff after the result; the curve says otherwise.
    bouts = [Bout(1, 1, T(2, 48), None, DAY), Bout(2, 0, T(2, 55), None, DAY, resolved_by=T(0, 0))]
    assert bout_cutoffs(bouts)[2] == T(0, 0) - BOUT_LEAD


@pytest.mark.parametrize("raw, expected", [
    ("2026-09-19T21:00:00Z", datetime(2026, 9, 19, 21)),
    ("2026-09-19 21:00:00+00", datetime(2026, 9, 19, 21)),
    ("2026-10-24 17:53:22.872265+00", datetime(2026, 10, 24, 17, 53, 22, 872265)),
    (None, None),
    ("garbage", None),
])
def test_parse_ts(raw, expected):
    assert parse_ts(raw) == expected


def test_started_open_market_serves_curve_price_not_live_quote():
    from app.services.ufc.prediction_markets.serving import _quote_dict

    class M:
        status, outcome_label = "open", None

    class Q:
        price, bid, ask, volume, open_interest, liquidity, captured_at = 0.9995, None, None, 5.0, None, None, None

    got = _quote_dict(M(), Q(), closing=0.475, started=True)
    assert got["price"] == 0.475 and got["is_closing_line"]
    # Started with nothing on the curve before the cutoff: no usable price at all.
    assert _quote_dict(M(), Q(), closing=None, started=True)["is_settlement"]


# --------------------------------------------------------------------------- against a DB

@pytest.fixture()
def db():
    from app.database import Base, SessionLocal, engine
    from app.models.ufc import (
        UFCEvent, UFCFight, UFCFighter, UFCFightPrediction, UFCPredictionMarket,
        UFCPredictionMarketHistory, UFCPredictionMarketQuote,
    )
    tables = [UFCFighter.__table__, UFCEvent.__table__, UFCFight.__table__,
              UFCPredictionMarket.__table__, UFCPredictionMarketQuote.__table__,
              UFCPredictionMarketHistory.__table__, UFCFightPrediction.__table__]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)
    s = SessionLocal()
    for fid in (1, 2, 3, 4):
        s.add(UFCFighter(id=fid, ufcstats_id=f"u{fid}", first_name="F", last_name=str(fid)))
    s.add(UFCEvent(id=1, ufcstats_id="e1", name="Card", date=DAY))
    s.add(UFCFight(id=10, ufcstats_id="f10", event_id=1, date=DAY, card_position=1,
                   red_fighter_id=1, blue_fighter_id=2))
    s.add(UFCFight(id=11, ufcstats_id="f11", event_id=1, date=DAY, card_position=0,
                   red_fighter_id=3, blue_fighter_id=4))
    s.commit()
    yield s
    s.close()


def _market(db, mid, fight_id, side, closed=None, start=None, platform="kalshi"):
    from app.models.ufc import UFCPredictionMarket
    db.add(UFCPredictionMarket(id=mid, fight_id=fight_id, platform=platform,
                               external_event_id=f"e{fight_id}", external_market_id=f"m{mid}",
                               market_type="moneyline", outcome_key=side, side=side,
                               venue_closed_at=closed, card_start_at=start))


def _curve(db, mid, points, cutoff):
    from app.models.ufc import UFCPredictionMarketHistory
    for i, (ts, price) in enumerate(points):
        db.add(UFCPredictionMarketHistory(id=mid * 100 + i, market_id=mid, price=price, captured_at=ts,
                                          days_to_fight=(cutoff - ts).total_seconds() / 86400))


@needs_sqlite
def test_cutoffs_for_fights_reads_stored_times_and_pinned_tail(db):
    from app.services.ufc.prediction_markets.common import cutoffs_for_fights
    _market(db, 1, 10, "red", closed=T(22, 0), start=T(21, 0))
    _market(db, 2, 11, "red", closed=T(1, 30), start=T(21, 0))
    # The main event's curve pinned at 22:20 -- before even the previous bout's close is any use
    # and hours before its own recorded close.
    _curve(db, 2, [(T(21, 30), 0.55), (T(22, 20), 0.60), (T(22, 21), 0.99), (T(23, 0), 0.999)], T(0, 0))
    db.commit()
    cut = cutoffs_for_fights(db, [10, 11])
    assert cut[10] == T(21, 0)                     # opener: card start
    assert cut[11] == T(22, 20) - BOUT_LEAD        # pinned tail beats the previous bout's close


@needs_sqlite
def test_market_history_stops_at_cutoff(db):
    from app.services.ufc.prediction_markets.serving import market_history
    _market(db, 1, 11, "red")
    _market(db, 2, 11, "blue")
    cutoff = T(0, 0)
    for mid, prices in ((1, (0.48, 0.47, 0.9995)), (2, (0.52, 0.53, 0.0005))):
        _curve(db, mid, list(zip((T(22, 0), T(23, 0), T(1, 0)), prices)), cutoff)
    db.commit()
    series = market_history(db, 11)["kalshi"]
    assert [p["red_prob"] for p in series] == [0.48, 0.47]


@needs_sqlite
def test_served_prediction_survives_rerun_after_fight(db, monkeypatch):
    from app.models.ufc import UFCFightPrediction
    from app.services.ufc import prediction_freeze
    from app.services.ufc.prediction_markets import common

    _market(db, 1, 10, "red", closed=T(22, 0), start=T(21, 0))
    db.add(UFCFightPrediction(id=1, fight_id=10, predicted_winner="red", confidence=0.02, red_prob=0.52,
                              created_at=T(15, 0)))                     # served before the card
    db.add(UFCFightPrediction(id=2, fight_id=11, predicted_winner="red", confidence=0.02, red_prob=0.52,
                              created_at=T(23, 0)))                     # mid-card run: not pre-fight for 10
    db.commit()

    # Before the card starts nothing is frozen; after it, only the pre-fight row is.
    real = common.fight_started
    monkeypatch.setattr(common, "fight_started", lambda c, now=None: real(c, T(20, 0)))
    assert prediction_freeze.frozen_fight_ids(db, UFCFightPrediction) == set()
    monkeypatch.setattr(common, "fight_started", lambda c, now=None: real(c, T(12, 0) + timedelta(days=2)))
    frozen = prediction_freeze.frozen_fight_ids(db, UFCFightPrediction)
    assert frozen == {10}

    prediction_freeze.clear_unfrozen(db, UFCFightPrediction, frozen)
    db.commit()
    assert [p.fight_id for p in db.query(UFCFightPrediction)] == [10]
