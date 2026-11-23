"""Officials endpoints (/ufc/judges, /robberies, /referees, /fights/{id}/officials) served
from a bare FastAPI app over scratch SQLite.

Skips unless DATABASE_URL is SQLite, so it can never touch the production database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_officials_api.py
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from app.config import settings

pytestmark = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                reason="needs a scratch SQLite DATABASE_URL")

BIG = 9_007_199_254_740_993          # past 2**53: fight ids must come back as strings


@pytest.fixture()
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCFighter, UFCJudge, UFCJudgeAlias
    from app.models.ufc_officials import (
        UFCDecisionReview, UFCFightOfficial, UFCJudgeCard, UFCJudgePair, UFCJudgeProfile,
        UFCJudgeRound, UFCOfficialsMeta, UFCReferee, UFCRefereeFight,
    )
    from app.routers import ufc as ufc_router
    from app.services import response_cache

    models = [UFCFighter, UFCJudge, UFCJudgeAlias, UFCJudgeProfile, UFCJudgeRound, UFCJudgeCard,
              UFCDecisionReview, UFCJudgePair, UFCOfficialsMeta, UFCReferee, UFCRefereeFight,
              UFCFightOfficial]
    tables = [m.__table__ for m in models]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)

    db = SessionLocal()
    v = "v1"
    detail = {"n_decided": 300, "red": 0.1, "red_se": 0.05, "pct_red": 70.0, "carry": -0.2,
              "carry_se": 0.08, "fights_by_year": {"2024": 3}, "ten8_post2017": 0.04}
    for jid, name, elig, acc in ((1, "Sal Judge", True, 0.02), (2, "Rare Judge", False, -0.01),
                                 (3, "Other Judge", True, 0.0)):
        db.add(UFCJudge(id=jid, name=name, name_norm=name.lower()))
        db.add(UFCJudgeProfile(judge_id=jid, version=v, name=name, eligible=elig, n_fights=10,
                               n_rounds=300 if elig else 20, accuracy=0.85, expected_accuracy=0.83,
                               acc_above_exp=acc, agree_above_exp=0.01, deciding_cards=2,
                               robbery_cards=1, first_date=date(2020, 1, 1),
                               last_date=date(2024, 1, 1), map_x=0.5, map_y=-0.2,
                               detail=json.dumps(detail)))
    db.add(UFCJudgeAlias(id=1, alias="Sally Judge", alias_norm="sally judge", judge_id=1))
    db.add(UFCJudgePair(version=v, judge_a=1, judge_b=3, n_rounds=40, agree=0.9, expected=0.85))
    db.add(UFCJudgePair(version=v, judge_a=1, judge_b=2, n_rounds=40, agree=0.8, expected=0.85))
    db.add(UFCOfficialsMeta(kind="judges", version=v, payload=json.dumps(
        {"meta": {}, "pool": {"leans": {"carry": {"effect": 0.33, "se": 0.04}}}})))
    for rnd, verdict, p in ((1, 1, 0.9), (2, 1, 0.05), (3, -1, 0.6)):
        db.add(UFCJudgeRound(version=v, judge_id=1, fight_id=BIG, round=rnd, date=date(2024, 1, 1),
                             verdict=verdict, score="10-9" if verdict > 0 else "9-10", p_red=p,
                             red_name="Red A", blue_name="Blue B", event_name="UFC X"))
    cards = [{"judge_id": 1, "name": "Sal Judge", "red": 29, "blue": 28, "side": 1}]
    for fid, score in ((BIG, 0.9), (5, 0.3)):
        db.add(UFCDecisionReview(version=v, fight_id=fid, date=date(2024, 1, 1), event_name="UFC X",
                                 weight_class="Lightweight Bout", red_id=BIG + 2, blue_id=11,
                                 red_name="Red A", blue_name="Blue B", result=1, kind="sd",
                                 p_red=1 - score, p_blue=score, robbery_score=score, media_red=1,
                                 media_blue=5, media_draw=0, fan_red=0.2, fan_blue=0.8,
                                 fan_draw=0.0, cards=json.dumps(cards)))
        db.add(UFCJudgeCard(version=v, judge_id=3, judge_seq=3, fight_id=fid, date=date(2024, 1, 1),
                            red_total=28, blue_total=29, side=-1, result=1, pivotal=False,
                            robbery_card=False, against_model=False))
        db.add(UFCJudgeCard(version=v, judge_id=1, judge_seq=1, fight_id=fid, date=date(2024, 1, 1),
                            red_total=29, blue_total=28, side=1, result=1, pivotal=True,
                            robbery_card=score >= 0.7, against_model=score >= 0.7))
    db.add(UFCReferee(id=7, name="Herb Ref", name_norm="herb ref", n_fights=100, n_regional=5,
                      finish_rate=0.6, stoppage_index=-0.1, deductions_per_100=2.0, dq_rate=0.01,
                      first_date=date(2010, 1, 1), last_date=date(2024, 1, 1),
                      detail=json.dumps({"methods": {"ko": 30}})))
    db.add(UFCRefereeFight(referee_id=7, fight_id=BIG, date=date(2024, 1, 1), event_name="UFC X",
                           weight_class="Lightweight Bout", red_name="Red A", blue_name="Blue B",
                           winner=1, method="KO/TKO", finish_round=2, finish_time="1:00",
                           absorbed=12, absorbed_z=0.4, deductions=0))
    db.add(UFCFightOfficial(fight_id=BIG, role="judge", seq=1, name="Sal Judge", judge_id=1))
    db.add(UFCFightOfficial(fight_id=BIG, role="referee", seq=1, name="Herb Ref", referee_id=7))
    db.commit()
    db.close()
    response_cache.invalidate()

    app = FastAPI()
    app.include_router(ufc_router.router)
    yield TestClient(app)
    response_cache.invalidate()


def test_judges_list_and_pool(client):
    r = client.get("/ufc/judges").json()
    assert [j["judge_id"] for j in r["judges"]][0] == 1          # most rounds first
    assert r["pool"]["leans"]["carry"]["effect"] == 0.33
    j = r["judges"][0]
    assert j["leans"]["red"] == {"effect": 0.1, "se": 0.05, "pct": 70.0}
    assert j["leans"]["fav"]["effect"] is None
    assert j["dissents"] == 0 and j["deciding_cards"] == 2


def test_judge_profile_fights_and_misses(client):
    r = client.get("/ufc/judges/1").json()
    assert r["judge"]["fights_by_year"] == {"2024": 3}
    other = client.get("/ufc/judges").json()["judges"]
    assert {j["judge_id"]: j["dissents"] for j in other} == {1: 0, 3: 2, 2: 0}
    assert {f["fight_id"] for f in r["fights"]} == {str(BIG), "5"}
    assert [(x["judge_id"], x["name"]) for x in r["partners"]] == [(3, "Other Judge"), (2, "Rare Judge")] \
        or [x["judge_id"] for x in r["partners"]] == [2, 3]
    assert next(x for x in r["partners"] if x["judge_id"] == 3)["excess"] == pytest.approx(0.05)
    big = next(f for f in r["fights"] if f["fight_id"] == str(BIG))
    assert big["red"]["id"] == str(BIG + 2) and big["robbery_card"] and big["pivotal"]
    misses = client.get("/ufc/judges/1/rounds?limit=2").json()
    # Round 2 (red given a round the model had at 5%) is the worst miss.
    assert [m["round"] for m in misses] == [2, 3]
    assert client.get("/ufc/judges/99").status_code == 404


def test_agreement_all_judges_flagged(client):
    r = client.get("/ufc/judges/agreement").json()
    assert {n["judge_id"]: n["eligible"] for n in r["nodes"]} == {1: True, 2: False, 3: True}
    assert len(r["edges"]) == 2
    assert next(e for e in r["edges"] if e["b"] == 3)["excess"] == pytest.approx(0.05)


def test_robberies_sorted_and_filtered(client):
    r = client.get("/ufc/robberies?min_score=0.5").json()
    assert [x["fight_id"] for x in r] == [str(BIG)]
    assert r[0]["fans"] == {"red": 0.2, "blue": 0.8, "draw": 0.0}
    assert len(client.get("/ufc/robberies?min_score=0").json()) == 2
    assert len(client.get("/ufc/robberies?min_score=0&window=year:2024").json()) == 2
    assert client.get("/ufc/robberies?min_score=0&window=year:2023").json() == []
    assert client.get("/ufc/robberies?window=bogus").status_code == 422
    # Both fixture decisions went to red with the fans' cards 80% blue: both agree.
    assert len(client.get("/ufc/robberies?min_score=0&fans_agree=true").json()) == 2


def test_referees(client):
    assert client.get("/ufc/referees").json()[0]["name"] == "Herb Ref"
    r = client.get("/ufc/referees/7").json()
    assert r["referee"]["methods"] == {"ko": 30}
    assert r["fights"][0]["fight_id"] == str(BIG)
    assert client.get("/ufc/referees/by-name?name=HERB  REF.").json()["referee_id"] == 7
    assert client.get("/ufc/referees/by-name?name=nobody").status_code == 404


def test_fight_officials(client):
    r = client.get(f"/ufc/fights/{BIG}/officials").json()
    assert r["judges"][0]["profile"]["name"] == "Sal Judge"
    assert r["referee"]["profile"]["referee_id"] == 7
    assert client.get("/ufc/fights/123/officials").json() == {"judges": [], "referee": None}


def test_resolve_official_uses_aliases(client):
    from app.database import SessionLocal
    from app.routers.ufc_officials import resolve_official
    db = SessionLocal()
    try:
        assert resolve_official(db, "judge", "Sal Judge") == (1, None)
        assert resolve_official(db, "judge", "Sally Judge") == (1, None)
        assert resolve_official(db, "judge", "Nobody") == (None, None)
        assert resolve_official(db, "referee", "Herb Ref") == (None, 7)
    finally:
        db.close()


def test_admin_sets_and_reads_panel(client, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter
    from app.routers import admin

    core = [UFCFighter.__table__, UFCEvent.__table__, UFCFight.__table__]
    Base.metadata.drop_all(engine, tables=core)
    Base.metadata.create_all(engine, tables=core)
    db = SessionLocal()
    db.add(UFCEvent(id=50, ufcstats_id="e50", name="UFC Y", date=date(2026, 12, 1)))
    for fid in (60, 61):
        db.add(UFCFighter(id=fid, ufcstats_id=f"u{fid}", first_name="F", last_name=str(fid)))
    db.add(UFCFight(id=70, ufcstats_id="f70", event_id=50, red_fighter_id=60, blue_fighter_id=61))
    db.commit()
    db.close()

    monkeypatch.setattr(admin.settings, "ADMIN_API_KEY", "k")
    app = FastAPI()
    app.include_router(admin.router)
    a = TestClient(app)
    h = {"X-Admin-Key": "k"}
    body = {"judges": ["Sally Judge", "Other Judge", "Mystery Person"], "referee": "Herb Ref"}
    r = a.put("/admin/fights/70/officials", json=body, headers=h).json()
    assert r == {"saved": 4, "unmatched": ["Mystery Person"]}
    rows = a.get("/admin/fights/70/officials", headers=h).json()
    assert [(x["role"], x["judge_id"], x["referee_id"]) for x in rows] == [
        ("judge", 1, None), ("judge", 3, None), ("judge", None, None), ("referee", None, 7)]
    assert a.put("/admin/fights/999/officials", json=body, headers=h).status_code == 404
    assert a.put("/admin/fights/70/officials", json=body, headers={"X-Admin-Key": "x"}).status_code == 403
    # The public view now shows the panel with profiles where matched.
    pub = client.get("/ufc/fights/70/officials").json()
    assert [j["profile"]["name"] if j["profile"] else None for j in pub["judges"]] == \
        ["Sal Judge", "Other Judge", None]


def test_decision_rounds(client):
    r = client.get(f"/ufc/decisions/{BIG}/rounds").json()
    assert [j["judge_id"] for j in r["judges"]] == [1, 3]
    assert [x["round"] for x in r["rounds"]] == [1, 2, 3]
    # Only judge 1 has round rows in the fixture; judge 3's slot is empty.
    assert r["rounds"][1]["scores"] == [{"score": "10-9", "verdict": 1}, None]
    assert r["rounds"][0]["p_red"] == 0.9


def test_fight_round_stats(client):
    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFightStats
    core = [UFCFighter.__table__, UFCEvent.__table__, UFCFight.__table__, UFCFightStats.__table__]
    Base.metadata.drop_all(engine, tables=core)
    Base.metadata.create_all(engine, tables=core)
    db = SessionLocal()
    db.add(UFCEvent(id=50, ufcstats_id="e50", name="UFC Y", date=date(2026, 1, 1)))
    for fid in (60, 61):
        db.add(UFCFighter(id=fid, ufcstats_id=f"u{fid}", first_name="F", last_name=str(fid)))
    db.add(UFCFight(id=70, ufcstats_id="f70", event_id=50, red_fighter_id=60, blue_fighter_id=61, finish_round=2))
    sid = 1
    for rnd in (0, 1, 2, 3):                       # 0 = totals, 3 = zero padding past the finish
        for fid, sig in ((60, 10 * rnd), (61, rnd)):
            db.add(UFCFightStats(id=sid, fight_id=70, fighter_id=fid, round_number=rnd, sig_str_landed=sig,
                                 corner="red" if fid == 60 else "blue",
                                 kd=1 if (rnd == 2 and fid == 60) else 0, td_landed=0, ctrl_seconds=30))
            sid += 1
    db.commit()
    db.close()
    r = client.get("/ufc/fights/70/round-stats").json()
    assert [x["round"] for x in r["rounds"]] == [1, 2]
    assert r["rounds"][1]["red"] == {"sig": 20, "kd": 1, "td": 0, "ctrl": 30}
    assert r["rounds"][1]["blue"]["sig"] == 2
    assert client.get("/ufc/fights/999/round-stats").status_code == 404
