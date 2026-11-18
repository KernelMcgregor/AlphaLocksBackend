"""Fight-flow track: pace profile (services/ufc/fightflow_pace) and clutch rating
(services/ufc/fightflow_clutch), plus their endpoints over a bare app on scratch SQLite.

The unit tests need no database. The API tests skip unless DATABASE_URL is SQLite, so
they can never touch production.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_fightflow.py
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from app.config import settings
from app.services.ufc import fightflow_clutch as fc
from app.services.ufc import fightflow_pace as fp

sqlite_only = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                 reason="needs a scratch SQLite DATABASE_URL")


# ------------------------------------------------------------------- pace


def test_round_seconds_finish_and_padding():
    assert fp.round_seconds("5-5-5", 3, 900) == [300, 300, 300]
    assert fp.round_seconds("5-5-5", 2, 400) == [300, 100]
    # Old formats: a 10-minute first round.
    assert fp.round_seconds("10-5", 2, 780) == [600, 180]
    # Fight time disagrees with the format: keep the scheduled length.
    assert fp.round_seconds("5-5-5", 2, 200) == [300, 300]
    assert fp.round_seconds(None, 2, 0) == [300, 300]
    assert fp.round_seconds("5-5-5", None, 900) == []


def _rows(rates, division="lightweight", fights=1, minutes=5.0):
    """Stat rows for `fights` bouts with per-round landed rates."""
    out = []
    for f in range(fights):
        for i, rate in enumerate(rates, 1):
            out.append({"fight_id": f, "division": division, "fighter_id": 1, "round": i,
                        "minutes": minutes, "landed": rate * minutes})
    return out


NORMS = {"lightweight": {1: 4.0, 2: 4.0, 3: 4.0}, "all": {1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: 4.0}}


def test_profile_labels():
    assert fp.profile(_rows([6, 4, 3], fights=4), NORMS)["label"] == "fader"
    assert fp.profile(_rows([3, 4, 6], fights=4), NORMS)["label"] == "closer"
    assert fp.profile(_rows([4, 4, 4], fights=4), NORMS)["label"] == "steady"
    # One fight: not enough R1 minutes for a label.
    assert fp.profile(_rows([6, 1, 1]), NORMS)["label"] is None


def test_profile_indexes_against_round_norm():
    # Everyone slows: a fighter dropping exactly with the division is steady.
    norms = {"lightweight": {1: 5.0, 2: 4.0, 3: 3.0}}
    p = fp.profile(_rows([5, 4, 3], fights=4), norms)
    assert p["label"] == "steady"
    assert all(abs(r["index"] - 1) < 1e-9 for r in p["rounds"])
    assert p["rounds"][1]["norm"] == pytest.approx(4.0)
    assert p["fights"] == 4


def test_profile_falls_back_to_all_divisions():
    p = fp.profile(_rows([4, 4, 4, 4], division="unknown", fights=4), NORMS)
    assert [r["round"] for r in p["rounds"]] == [1, 2, 3, 4]


# ------------------------------------------------------------------ clutch


def test_side_rounds_drops_partial_and_projected():
    rp = [0.9, 0.2, 0.5, 0.5, 0.5]
    assert len(fc.side_rounds(rp, 3, None)) == 3
    assert len(fc.side_rounds(rp, 3, 90)) == 2
    assert fc.side_rounds([0.0003], 1, None)[0] == pytest.approx(math.log(0.02 / 0.98))


def test_bout_deltas():
    d = fc.bout_deltas([1.0, -1.0, 2.0])
    assert d["late"] == pytest.approx(2.0)           # 2 - mean(1, -1)
    # R3 follows a lost R2: x3 minus the rest without R2 and R3 (= R1).
    assert d["bounce"] == [pytest.approx(1.0)]
    assert d["late_rounds"] == 1 and d["clutch_rounds"] == 1
    d = fc.bout_deltas([-1.0, 0.5, 0.5])
    assert d["clutch_rounds"] == 2                    # R2 after a lost R1, and R3
    assert fc.bout_deltas([1.0, 1.0])["late"] is None


def _bouts(n, red_x):
    return [{"fight_id": i, "red": 1, "blue": 100 + i, "x": list(red_x)} for i in range(n)]


def test_rate_shrinks_and_ranks():
    # Fighter 1 always closes strong; every opponent is the mirror image (fades).
    bouts = _bouts(6, [-0.5, -0.5, 1.5])
    # A few neutral bouts between others so the league has spread.
    bouts += [{"fight_id": 50 + i, "red": 200 + i % 4, "blue": 300 + i % 4,
               "x": [0.2, -0.3, 0.1 * (i % 3)]}
              for i in range(20)]
    res = fc.rate(bouts)
    f1 = res["fighters"][1]["all"]
    assert f1["est"] > 0 and f1["n_fights"] == 6
    # Shrunk: R3 beat R1-R2 by 2.0 logits every time; the estimate is pulled toward 0.
    late = res["fighters"][1]["late"]["est"]
    assert 0 < late < 2.0 - res["meta"]["late_mean"]
    assert res["fighters"][1]["percentile"] == pytest.approx(100.0)
    # One-bout opponents get no percentile.
    assert "percentile" not in res["fighters"][100]
    rows = fc.to_rows(res)
    assert {r["fighter_id"] for r in rows} >= {1, 100}


def test_rate_skill_cancels():
    # A fighter who wins every round by the same margin is not clutch.
    res = fc.rate(_bouts(5, [2.0, 2.0, 2.0]) + _bouts(5, [0.0, 0.0, 0.0]))
    assert abs(res["fighters"][1]["late"]["est"]) < 1e-9


# --------------------------------------------------------------------- API


@pytest.fixture()
def client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFightStats
    from app.models.ufc_fightflow import UFCFighterClutch
    from app.routers import ufc as ufc_router
    from app.services import response_cache

    tables = [UFCFighter.__table__, UFCEvent.__table__, UFCFight.__table__,
              UFCFightStats.__table__, UFCFighterClutch.__table__]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)

    db = SessionLocal()
    big = 9_007_199_254_740_993          # past 2**53: must survive as a string
    for fid in (big, 2):
        db.add(UFCFighter(id=fid, ufcstats_id=f"u{fid}", first_name="F", last_name=str(fid)))
    db.add(UFCEvent(id=1, ufcstats_id="e1", name="E", date=date(2025, 1, 1)))
    db.add(UFCFight(id=big, ufcstats_id="f1", date=date(2025, 1, 1), event_id=1,
                    red_fighter_id=big, blue_fighter_id=2, weight_class="Lightweight Bout",
                    method="KO/TKO", finish_round=2, finish_time="2:30", time_format="5-5-5",
                    fight_time_seconds=450))
    stats = [(big, "red", 1, 20), (2, "blue", 1, 10), (big, "red", 2, 5), (2, "blue", 2, 5),
             (big, "red", 3, 0)]                 # round 3: padding past the finish, ignored
    for i, (fid, corner, rnd, landed) in enumerate(stats, 1):
        db.add(UFCFightStats(id=i, fight_id=big, fighter_id=fid, round_number=rnd, corner=corner,
                             sig_str_landed=landed))
    fc.write(db, [{"fighter_id": big, "clutch": 0.3, "se": 0.2, "late": 0.4, "late_rounds": 6,
                   "bounce": 0.1, "bounce_rounds": 3, "n_rounds": 8, "n_fights": 5, "percentile": 91.0}])
    db.close()
    response_cache.invalidate()
    monkeypatch.setattr(fp, "NORM_MIN_MINUTES", 1.0)

    app = FastAPI()
    app.include_router(ufc_router.router)
    yield TestClient(app)
    response_cache.invalidate()


@sqlite_only
def test_clutch_endpoint(client):
    r = client.get("/ufc/fighters/9007199254740993/clutch")
    assert r.status_code == 200
    body = r.json()
    assert body["fighter_id"] == "9007199254740993"
    assert body["clutch"] == 0.3 and body["percentile"] == 91.0 and body["version"] == fc.VERSION
    assert client.get("/ufc/fighters/2/clutch").status_code == 404


@sqlite_only
def test_pace_endpoint(client):
    r = client.get("/ufc/fights/9007199254740993/pace")
    assert r.status_code == 200
    body = r.json()
    assert body["fight_id"] == "9007199254740993"
    assert body["division"] == "lightweight" and body["scheduled_rounds"] == 3
    red = body["red"]["fight"]
    assert [x["round"] for x in red] == [1, 2]                     # padding round dropped
    assert red[0]["slpm"] == pytest.approx(4.0)
    assert red[1]["minutes"] == pytest.approx(2.5) and red[1]["slpm"] == pytest.approx(2.0)
    # Norm in R1 is the pooled rate of both corners: 30 strikes / 10 minutes.
    assert body["norm"][0]["slpm"] == pytest.approx(3.0)
    assert body["red"]["career"]["label"] is None                 # one bout: no label
    assert client.get("/ufc/fights/12345/pace").status_code == 404
