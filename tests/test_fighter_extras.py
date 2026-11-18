"""Fighter-page extras: the pure computations, and the endpoints served from a bare
FastAPI app over scratch SQLite.

The API tests skip unless DATABASE_URL is SQLite, so they can never touch production.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_fighter_extras.py
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from app.config import settings

# ---------------------------------------------------------------------------
# Pure computations
# ---------------------------------------------------------------------------


def test_percentile_mid_rank_and_flip():
    from app.services.ufc.fighter_extras_profile import percentile_of
    vals = [1.0, 2.0, 3.0, 4.0]
    assert percentile_of(vals, 4.0) == 88          # 3.5 of 4 below
    assert percentile_of(vals, 0.5) == 0
    assert percentile_of(vals, 2.0) == 38          # ties count half
    assert percentile_of(vals, 4.0, higher_better=False) == 12
    assert percentile_of([], 1.0) is None


def test_outcome_and_finish():
    from app.services.ufc.fighter_extras_common import is_finish, outcome
    f = SimpleNamespace(winner_id=7, method="KO/TKO")
    assert outcome(f, 7) == "W" and outcome(f, 8) == "L"
    assert outcome(SimpleNamespace(winner_id=None, method="Decision - Majority"), 7) == "D"
    assert outcome(SimpleNamespace(winner_id=None, method="Overturned"), 7) == "NC"
    assert is_finish("Submission") and not is_finish("Decision - Unanimous") and not is_finish(None)


@pytest.mark.parametrize("raw,key", [
    ("American Top Team", "american top"),
    ("Jackson Wink MMA Academy - NM", "jackson wink"),
    ("Jackson-Wink MMA", "jackson wink"),
    ("AKA (American Kickboxing Academy) San Jose", "american kickboxing"),
    ("Team Alpha Male (Urijah Faber's Ultimate Fitness)", "alpha male"),
    ("Nova União", "nova uniao"),
    ("Tristar Gym - Montreal", "tristar"),
    ("4oz. Fight Club", "4oz"),
    ("Freelance", None),
])
def test_camp_key(raw, key):
    from app.services.ufc.fighter_extras_camps import camp_key
    assert camp_key(raw) == key


def test_camp_resolve_folds_prefixes_and_picks_primary():
    from app.services.ufc.fighter_extras_camps import build_rows, resolve
    reg = resolve([
        (1, "ufc", "American Top Team ATT Coconut Creek, FL"),
        (1, "sherdog", "American Top Team"),
        (2, "sherdog", "American Top Team"),
        (3, "ufc", "Xtreme Couture / Overcome Academy"),
        (4, "sherdog", "American Kickboxing Academy"),
        (5, "ufc", "Independent"),
    ])
    rows = build_rows(reg)
    names = {c["id"]: c["name"] for c in rows["camps"]}
    att = next(c for c in rows["camps"] if c["name"] == "American Top Team")
    assert att["n_fighters"] == 2
    # One word in common is not a merge: AKA stays its own camp.
    assert "American Kickboxing Academy" in names.values()
    by_fighter = {}
    for m in rows["members"]:
        by_fighter.setdefault(m["fighter_id"], []).append(m)
    assert len(by_fighter[1]) == 1 and by_fighter[1][0]["source"] == "ufc"
    three = {names[m["camp_id"]]: m["is_primary"] for m in by_fighter[3]}
    assert three == {"Xtreme Couture": True, "Overcome Academy": False}
    assert 5 not in by_fighter


def test_thin_dates():
    from app.services.ufc.fighter_extras_timelapse import thin_dates
    ds = [date(2020, 1, 5), date(2020, 2, 9), date(2020, 4, 1), date(2021, 6, 1)]
    assert thin_dates(ds, "quarter") == [date(2020, 2, 9), date(2020, 4, 1), date(2021, 6, 1)]
    assert thin_dates(ds, "year") == [date(2020, 4, 1), date(2021, 6, 1)]
    assert thin_dates(ds, "event") == ds


def test_age_curve_is_symmetric_and_smoothed():
    from app.services.ufc.fighter_extras_age import build_curves
    dobs = {1: date(1990, 1, 1), 2: date(2000, 1, 1)}
    d = date(2025, 6, 1)      # fighter 1 is 35, fighter 2 is 25
    fights = [(10 + i, d, "Lightweight Bout", 1, 2, 2, "KO/TKO") for i in range(3)]
    fights.append((20, d, "Lightweight Bout", 1, 2, None, "Decision - Split"))  # draw
    rows = build_curves(fights, dobs, {(10, 2): 4.0})
    lw = {r["age"]: r for r in rows if r["division"] == "lightweight"}
    assert lw[25]["fights"] == 4 and lw[25]["wins"] == 3.5 and lw[35]["wins"] == 0.5
    assert lw[25]["rating_delta"] == 4.0 and lw[35]["rating_delta"] is None
    assert {r["division"] for r in rows} == {"lightweight", "all"}
    assert lw[25]["smoothed"] == lw[25]["win_rate"]  # no neighbouring ages


def test_summarise():
    from app.services.ufc.fighter_extras_profile import summarise
    tags = [
        {"result": "W", "finish_win": True, "me": (30, 60), "them": (10, 40), "secs": 600},
        {"result": "L", "finish_win": False, "me": (10, 40), "them": (30, 60), "secs": 600},
        {"result": "NC", "finish_win": False, "me": None, "them": None, "secs": None},
    ]
    s = summarise(tags)
    assert (s["fights"], s["wins"], s["losses"], s["nc"]) == (3, 1, 1, 1)
    assert s["win_pct"] == 50.0 and s["finish_pct"] == 50.0
    assert s["sig_acc"] == 40.0 and s["sig_def"] == 60.0 and s["diff_pm"] == 0.0
    assert summarise([])["win_pct"] is None


# ---------------------------------------------------------------------------
# Endpoints over scratch SQLite
# ---------------------------------------------------------------------------

sqlite_only = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                 reason="needs a scratch SQLite DATABASE_URL")
BIG = 9_007_199_254_740_993          # past 2**53: must survive as a string
OPP = 2


@pytest.fixture()
def client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.database import Base, SessionLocal, engine
    from app.models.ufc import (
        SherdogBout, SherdogFighter, UFCCancelledBout, UFCEvent, UFCFight, UFCFightOpenClose,
        UFCFightStats, UFCFighter, UFCFighterCareerStats, UFCFighterRanking, UFCGlickoSnapshot,
        UFCRankingHistory,
    )
    from app.models.ufc_fighter_extras import UFCAgeCurve, UFCCamp, UFCCampAlias, UFCFighterCamp
    from app.routers import ufc as ufc_router
    from app.services import response_cache

    monkeypatch.delenv("FIGHTER_EXTRAS_DB_URL", raising=False)
    models = [UFCFighter, UFCEvent, UFCFight, UFCFightStats, UFCFighterCareerStats, UFCGlickoSnapshot,
              UFCRankingHistory, UFCFighterRanking, UFCCancelledBout, UFCFightOpenClose, SherdogFighter,
              SherdogBout, UFCAgeCurve, UFCCamp, UFCCampAlias, UFCFighterCamp]
    tables = [m.__table__ for m in models]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)

    db = SessionLocal()
    db.add_all([
        UFCFighter(id=BIG, ufcstats_id="a", first_name="Big", last_name="Id", dob=date(1990, 1, 1),
                   trains_at="City Kickboxing", status="Active", wins=3, losses=1, draws=0),
        UFCFighter(id=OPP, ufcstats_id="b", first_name="Opp", last_name="One", dob=date(1985, 1, 1),
                   trains_at="City Kickboxing", status="Active", wins=1, losses=3, draws=0),
        UFCEvent(id=1, ufcstats_id="e1", name="UFC 1", date=date(2020, 1, 1), location="Las Vegas, Nevada, USA"),
        UFCEvent(id=2, ufcstats_id="e2", name="UFC 2", date=date(2021, 6, 1), location="Mexico City, Distrito Federal, Mexico"),
        UFCEvent(id=3, ufcstats_id="e3", name="UFC 3", date=date(2021, 9, 1), location="Las Vegas, Nevada, USA"),
    ])
    db.flush()
    fights = [
        # id, event, date, winner, method, format
        (11, 1, date(2020, 1, 1), BIG, "KO/TKO", "5-5-5"),
        (12, 2, date(2021, 6, 1), BIG, "Decision - Unanimous", "5-5-5-5-5"),
        (13, 3, date(2021, 9, 1), OPP, "Submission", "5-5-5"),
    ]
    for fid, ev, d, w, m, tf in fights:
        db.add(UFCFight(id=fid, ufcstats_id=f"f{fid}", event_id=ev, date=d, red_fighter_id=BIG,
                        blue_fighter_id=OPP, winner_id=w, method=m, weight_class="Lightweight Bout",
                        time_format=tf, fight_time_seconds=600))
        db.add(UFCFightStats(id=fid * 10, fight_id=fid, fighter_id=BIG, round_number=0, corner="red",
                             sig_str_landed=30, sig_str_attempted=60))
        db.add(UFCFightStats(id=fid * 10 + 1, fight_id=fid, fighter_id=OPP, round_number=0, corner="blue",
                             sig_str_landed=20, sig_str_attempted=50))
        for me, pts in ((BIG, 20.0 + fid), (OPP, 15.0)):
            db.add(UFCGlickoSnapshot(id=fid * 10 + (me == OPP), fight_id=fid, fighter_id=me,
                                     **{d_: pts for d_ in ("pts", "ko", "kod", "sub", "subd", "td", "tdd", "ctrl",
                                                           "str_vol", "str_acc", "str_def", "dist", "clinch",
                                                           "gnd", "durability")}))
    # Fight 11: BIG closed a big underdog. Fight 12: favourite.
    db.add(UFCFightOpenClose(id=1, fight_id=11, source="bfo", bookmaker="Consensus", red_close_prob=0.3))
    db.add(UFCFightOpenClose(id=2, fight_id=12, source="bfo", bookmaker="Consensus", red_close_prob=0.7))
    # OPP had a different bout pulled from card 3: BIG stepped in.
    db.add(UFCCancelledBout(id=1, ufcstats_id="c1", event_date=date(2021, 9, 1), red_fighter_id=OPP,
                            blue_fighter_id=99, removed_at=date(2021, 8, 25)))
    for i, (fid, slpm, fc) in enumerate(((BIG, 5.0, 3), (OPP, 3.0, 3)), 1):
        db.add(UFCFighterCareerStats(id=i, fighter_id=fid, fight_count=fc, slpm=slpm, sapm=3.0,
                                     win_pct=0.5, finish_rate=0.5))
    for i, (d, ranks) in enumerate(((date(2020, 1, 1), (BIG, OPP)), (date(2021, 9, 1), (OPP, BIG))), 1):
        for r, fid in enumerate(ranks, 1):
            db.add(UFCRankingHistory(id=i * 10 + r, fighter_id=fid, as_of=d, weight_class="lightweight",
                                     rank=r, score=1000.0 / r, total_ranked=2))
    db.add(UFCAgeCurve(id=1, division="lightweight", age=31, fights=40, wins=20, win_rate=0.5, smoothed=0.5))
    db.add(UFCAgeCurve(id=2, division="lightweight", age=45, fights=3, wins=1, win_rate=0.33, smoothed=0.33))
    db.commit()
    db.close()

    from app.services.ufc.fighter_extras_camps import build_rows, gather, resolve, write
    db = SessionLocal()
    write(db, build_rows(resolve(gather(db))))
    db.close()
    response_cache.invalidate()

    app = FastAPI()
    app.include_router(ufc_router.router)
    yield TestClient(app)
    response_cache.invalidate()


@sqlite_only
def test_percentiles(client):
    body = client.get(f"/ufc/fighters/{BIG}/percentiles").json()
    assert body["fighter_id"] == str(BIG) and body["division"] == "lightweight"
    stats = {s["key"]: s for g in body["groups"] for s in g["stats"]}
    assert stats["slpm"]["percentile"] == 75 and stats["slpm"]["value"] == 5.0
    assert stats["td15"]["percentile"] is None
    assert client.get("/ufc/fighters/12345/percentiles").status_code == 404


@sqlite_only
def test_rating_history(client):
    body = client.get(f"/ufc/fighters/{BIG}/rating-history").json()
    fights = body["fights"]
    assert [f["result"] for f in fights] == ["W", "W", "L"]
    assert fights[0]["delta"]["pts"] == 1.0 and fights[-1]["delta"] is None
    assert fights[0]["opponent_id"] == str(OPP) and fights[0]["opp_pre"]["pts"] == 15.0
    assert len(body["dims"]) == 15


@sqlite_only
def test_splits(client):
    body = client.get(f"/ufc/fighters/{BIG}/splits").json()
    s = {x["key"]: x for x in body["splits"]}
    assert body["overall"]["wins"] == 2 and body["overall"]["losses"] == 1
    assert s["short_notice"]["a"]["fights"] == 1 and s["short_notice"]["a"]["losses"] == 1
    assert s["layoff"]["a"]["fights"] == 1          # 2020-01 -> 2021-06
    assert s["odds"]["a"]["wins"] == 1 and s["odds"]["b"]["wins"] == 1
    assert s["rounds"]["a"]["fights"] == 1 and s["rounds"]["b"]["fights"] == 2
    assert s["altitude"]["a"]["fights"] == 1
    assert body["overall"]["sig_acc"] == 50.0 and body["overall"]["diff_pm"] == 1.0


@sqlite_only
def test_age_curve(client):
    body = client.get(f"/ufc/fighters/{BIG}/age-curve").json()
    assert body["division"] == "lightweight"
    assert [c["age"] for c in body["curve"]] == [31]       # thin bins dropped
    assert [a["age"] for a in body["ages"]] == [30, 31]
    assert body["bouts"][0]["opponent_id"] == str(OPP)


@sqlite_only
def test_camps(client):
    camps = client.get("/ufc/camps?min_fighters=2").json()["camps"]
    assert len(camps) == 1 and camps[0]["name"] == "City Kickboxing" and camps[0]["fighters"] == 2
    detail = client.get(f"/ufc/camps/{camps[0]['slug']}").json()
    assert {r["id"] for r in detail["roster"]} == {str(BIG), str(OPP)}
    assert client.get(f"/ufc/camps/{camps[0]['id']}").status_code == 200
    mine = client.get(f"/ufc/fighters/{BIG}/camp").json()
    assert mine["camp"]["slug"] == camps[0]["slug"] and mine["teammates"][0]["id"] == str(OPP)
    assert client.get("/ufc/fighters/777/camp").json()["camp"] is None


@sqlite_only
def test_timelapse(client):
    body = client.get("/ufc/rankings/timelapse?division=lightweight&step=event").json()
    assert [f["as_of"] for f in body["frames"]] == ["2020-01-01", "2021-09-01"]
    assert body["frames"][0]["ranks"][0]["id"] == str(BIG)
    assert body["fighters"][str(BIG)]["name"] == "Big Id"
    assert client.get("/ufc/rankings/timelapse?division=nope").status_code == 404
    assert client.get("/ufc/rankings/timelapse?step=week").status_code == 422
