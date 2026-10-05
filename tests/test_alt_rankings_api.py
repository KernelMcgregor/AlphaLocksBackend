"""/ufc/alt-rankings/{kind} served from a bare FastAPI app over scratch SQLite.

Skips unless DATABASE_URL is SQLite, so it can never touch the production database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_alt_rankings_api.py
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
    from app.models.ufc import UFCAltRanking, UFCFighter
    from app.routers import ufc as ufc_router
    from app.services import response_cache

    tables = [UFCFighter.__table__, UFCAltRanking.__table__]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables + [
        t for t in Base.metadata.sorted_tables if t.name in ("ufc_events", "ufc_fights")])

    db = SessionLocal()
    big = 9_007_199_254_740_993          # past 2**53: must survive as a string
    for i, fid in enumerate((big, 2, 3)):
        db.add(UFCFighter(id=fid, ufcstats_id=f"u{fid}", first_name="F", last_name=str(i),
                          wins=10, losses=1, draws=0))
    rows = [("men", big, 1, 80.0), ("men", 2, 2, 70.0), ("women", 3, 1, 60.0)]
    for i, (pool, fid, rank, score) in enumerate(rows, 1):
        db.add(UFCAltRanking(
            id=i, kind="bmf", fighter_id=fid, pool=pool, division="lightweight",
            as_of=date(2026, 10, 5),
            components=json.dumps({"finishing": 50, "toughness": 50, "recency": 50,
                                   "opp_quality": 50, "raw": {"finishing": 1.0}}),
            default_score=score, default_rank=rank, n_bouts=5,
            last_fight_date=date(2026, 9, 1), ledger=json.dumps([{"fight_id": "1"}]),
        ))
    db.commit()
    db.close()
    response_cache.invalidate()

    app = FastAPI()
    app.include_router(ufc_router.router)
    yield TestClient(app)
    response_cache.invalidate()


def test_bmf_payload_shape(client):
    r = client.get("/ufc/alt-rankings/bmf")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "bmf"
    assert set(body["defaults"]) >= {"finishing", "toughness", "recency", "gamma"}
    assert {c["key"] for c in body["components"]} == {
        "finishing", "toughness", "recency", "opp_quality"}
    men = body["pools"]["men"]
    assert [f["default_rank"] for f in men] == [1, 2]
    assert men[0]["id"] == "9007199254740993"            # id is a string, not a float
    assert "raw" not in men[0]["components"] and men[0]["raw"] == {"finishing": 1.0}
    assert len(body["pools"]["women"]) == 1


def test_p4p_empty_and_unknown_kind(client):
    r = client.get("/ufc/alt-rankings/p4p")
    assert r.status_code == 200 and r.json()["pools"] == {"men": [], "women": []}
    assert client.get("/ufc/alt-rankings/nope").status_code == 404
