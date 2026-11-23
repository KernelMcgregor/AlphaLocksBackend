"""/ufc/exciting/fights, /ufc/exciting/cards and /ufc/events/{id}/bonuses over scratch SQLite.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_highlights_api.py
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from app.config import settings

pytestmark = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                reason="needs a scratch SQLite DATABASE_URL")


@pytest.fixture()
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter
    from app.models.ufc_highlights import UFCFightExcitement
    from app.routers import ufc as ufc_router
    from app.services import response_cache

    tables = [m.__table__ for m in (UFCFighter, UFCEvent, UFCFight, UFCFightExcitement)]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)
    db = SessionLocal()
    for i in range(1, 9):
        db.add(UFCFighter(id=i, ufcstats_id=f"f{i}", first_name="F", last_name=str(i)))
    db.add(UFCEvent(id=1, ufcstats_id="e1", name="UFC 1", date=date(2024, 3, 1)))
    db.add(UFCEvent(id=2, ufcstats_id="e2", name="UFC 2", date=date(2025, 3, 1)))
    comp = json.dumps({"groups": {"action": 0.5}, "raw": {"sig_pm": 9.0}})
    # (fight, event, card position, score, pct, fotn)
    for fid, ev, pos, score, pct, fotn in ((10, 1, 0, 2.0, 99, True), (11, 1, 1, 1.5, 95, False),
                                           (12, 2, 0, -1.0, 20, False), (13, 2, 6, 3.0, 100, True)):
        d = date(2024, 3, 1) if ev == 1 else date(2025, 3, 1)
        a = 2 * (fid - 10) + 1
        db.add(UFCFight(id=fid, ufcstats_id=f"x{fid}", event_id=ev, date=d, card_position=pos,
                        red_fighter_id=a, blue_fighter_id=a + 1, winner_id=a, method="KO/TKO"))
        db.add(UFCFightExcitement(fight_id=fid, model_version="v1", event_id=ev, date=d, score=score,
                                  pct=pct, components=comp, p_fotn=0.3, p_perf=0.5, fotn=fotn, perf=False))
    db.commit()
    db.close()
    response_cache.invalidate()
    app = FastAPI()
    app.include_router(ufc_router.router)
    return TestClient(app)


def test_fights_ranked_and_windowed(client):
    r = client.get("/ufc/exciting/fights").json()
    assert [x["fight_id"] for x in r] == ["13", "10", "11", "12"]
    assert r[0]["red"]["name"] == "F 7" and r[0]["winner"] == "red" and r[0]["fotn"]
    assert [x["fight_id"] for x in client.get("/ufc/exciting/fights?window=year:2024").json()] == ["10", "11"]
    assert client.get("/ufc/exciting/fights?window=nope").status_code == 422
    assert [x["rank"] for x in r] == [1, 2, 3, 4]


def test_fights_search_keeps_window_rank(client):
    r = client.get("/ufc/exciting/fights?q=f 5").json()
    assert [(x["fight_id"], x["rank"]) for x in r] == [("12", 4)]
    assert client.get("/ufc/exciting/fights?q=nobody").json() == []


def test_cards_main_vs_full(client):
    main = client.get("/ufc/exciting/cards").json()
    # Card 2's best bout is a prelim (position 6): only the full card counts it.
    assert [c["event_id"] for c in main] == ["1", "2"]
    full = client.get("/ufc/exciting/cards?scope=full").json()
    assert {c["event_id"]: c["n_fights"] for c in full} == {"1": 2, "2": 2}
    assert full[0]["fights"][0]["pct"] >= full[0]["fights"][-1]["pct"]


def test_event_bonuses(client):
    r = client.get("/ufc/events/1/bonuses").json()
    assert [f["fight_id"] for f in r["fights"]] == ["10", "11"]
    assert client.get("/ufc/events/99/bonuses").status_code == 404
