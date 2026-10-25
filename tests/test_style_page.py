"""Fighter Styles page: era adjustment, frozen placement, lineage, and the two routes.

The unit tests need no database. The route tests skip unless DATABASE_URL is SQLite, so
they can never touch the production database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_style_page.py
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.services.ufc import style_page_service as sp
from app.services.ufc import style_service as ss

needs_sqlite = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                  reason="needs a scratch SQLite DATABASE_URL")


class TestStaysOutOfTheModel:
    def test_page_service_does_not_import_the_model(self):
        import inspect
        src = inspect.getsource(sp)
        assert "from app.services.ufc.model import" not in src
        assert "import app.services.ufc.model" not in src

    def test_no_style_page_column_reaches_winner_features(self):
        from app.services.ufc.model import winner_feature_columns
        frame = pd.DataFrame({"elo": [1500.0], "style_family": [3.0], "style_map_x": [0.2],
                              "style_prime_year": [2019.0]})
        cols = winner_feature_columns(frame)
        for banned in ("style_family", "style_map_x", "style_prime_year"):
            assert banned not in cols


class TestFamilies:
    """Four families fixed by role, plus a calibrated blend per fighter."""

    def _centroids(self):
        col = ss.ALL_FEATURES.index
        C = np.zeros((4, len(ss.ALL_FEATURES)))
        C[0, col("str_att_pm")] = 1.0                        # range: busy, nothing else
        C[1, col("gnp_per_ctrl")] = 2.0                      # pounce
        C[2, col("str_att_pm")] = -2.0                       # patient: low output
        C[3, col("td_att15s")] = 2.0                         # wrestler
        return C

    def test_roles_do_not_depend_on_kmeans_label_order(self):
        C = self._centroids()
        for perm in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
            order = sp.role_order(C[perm])
            assert [perm[k] for k in order] == [sp.RANGE, sp.POUNCE, sp.PATIENT, sp.WRESTLE]

    def test_blends_are_distributions_and_pick_the_nearest(self):
        rng = np.random.default_rng(3)
        m = {"centroids": sp._unit(self._centroids()), "blend_tau": sp.BLEND_TAU}
        U = sp._unit(rng.normal(size=(200, len(ss.ALL_FEATURES))))
        P = sp.blends(U, m)
        assert np.allclose(P.sum(1), 1.0)
        assert np.array_equal(sp.assign_families(U, m), (U @ m["centroids"].T).argmax(1))

    def test_blend_list_is_sorted_and_thresholded(self):
        out = sp.blend_list(np.array([0.05, 0.6, 0.3, 0.05]))
        assert [b["family"] for b in out] == [1, 2]
        assert all(b["share"] >= sp.BLEND_MIN_SHARE for b in out)

    def test_every_family_has_a_name(self):
        assert len(sp.FAMILIES) == sp.N_FAMILIES
        assert all(name and blurb for name, blurb in sp.FAMILIES)


class TestFrozenPlacement:
    def _map(self, n=60, d=37):  # real dimensionality; low-d random anchors crowd together
        rng = np.random.default_rng(1)
        anchors = sp._unit(rng.normal(size=(n, d)))
        return {"anchors": anchors, "anchor_xy": rng.uniform(size=(n, 2))}

    def test_anchors_land_on_their_own_layout_position(self):
        m = self._map()
        xy = sp.place(m["anchors"], m)
        assert np.abs(xy - m["anchor_xy"]).max() < 0.02

    def test_a_new_fighter_lands_next_to_its_twin(self):
        m = self._map()
        twin = sp._unit(m["anchors"][7:8] + 0.01)
        assert np.abs(sp.place(twin, m)[0] - m["anchor_xy"][7]).max() < 0.05

    def test_placement_is_deterministic(self):
        m = self._map()
        assert np.array_equal(sp.place(m["anchors"], m), sp.place(m["anchors"], m))


class TestLineage:
    def test_ancestors_respect_the_era_gap_and_pool(self):
        U = sp._unit(np.array([[1.0, 0.0], [0.99, 0.1], [0.98, 0.2], [0.0, 1.0]]))
        prime = np.array([2024.0, 2021.0, 2010.0, 2008.0])
        anc = sp.ancestors(U, prime, np.ones(4, dtype=bool))
        ids = [j for j, _ in anc[0]]
        assert 1 not in ids                     # only 3 years earlier
        assert ids[0] == 2                       # closest in style among the eligible
        assert anc[3] == []                      # nobody before 2002

        notable_only = sp.ancestors(U, prime, np.array([True, True, False, True]))
        assert [j for j, _ in notable_only[0]] == [3]


@pytest.fixture()
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFighterSimilarity, UFCFighterStyle, UFCStyleFamily
    from app.routers import ufc as ufc_router
    from app.services import response_cache

    tables = [UFCFighter.__table__, UFCFighterStyle.__table__, UFCStyleFamily.__table__,
              UFCFighterSimilarity.__table__, UFCEvent.__table__, UFCFight.__table__]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)

    db = SessionLocal()
    big = 9_007_199_254_740_993          # past 2**53: must survive as a string
    now = datetime(2026, 10, 16)
    for fid, last in ((big, "Big"), (2, "Two"), (3, "Three")):
        db.add(UFCFighter(id=fid, ufcstats_id=f"u{fid}", first_name="F", last_name=last,
                          wins=10, losses=2, draws=0))
    db.add(UFCStyleFamily(id=0, name="Power strikers", blurb="b", traits='["kd15s"]',
                          profile='{"kd15s": 80}', n_fighters=3, map_version="v1", computed_at=now))
    for fid, fam in ((big, 0), (2, 0), (3, 0)):
        db.add(UFCFighterStyle(
            fighter_id=fid, family=fam, division="lightweight", x=0.5, y=0.5,
            first_year=2015, last_year=2025, prime_year=2020.0, active=True, quality=60.0,
            notable=fid == 2,
            profile=('{"kd15s": 70, "_blend": [{"family": 0, "share": 0.7}, {"family": 3, "share": 0.3}]}'
                     if fid == 2 else '{"kd15s": 70}'),
            ancestors=json.dumps([{"id": 3, "sim": 0.7}]), notable_ancestors="[]",
            map_version="v1", computed_at=now,
        ))
    db.add(UFCFighterSimilarity(id=1, fighter_id=big, similar_fighter_id=2, rank=1, similarity=0.9,
                                same_division=True, top_drivers='[{"feature": "kd15s", "z": 1.2}]'))
    db.commit()
    db.close()
    response_cache.invalidate()

    app = FastAPI()
    app.include_router(ufc_router.router)
    yield TestClient(app), big
    response_cache.invalidate()


@needs_sqlite
def test_overview_shape(client):
    c, big = client
    body = c.get("/ufc/styles").json()
    assert body["version"] == "v1"
    assert body["families"][0]["name"] == "Power strikers"
    ids = {f["id"] for f in body["fighters"]}
    assert str(big) in ids                                 # string, not a rounded float
    row = next(f for f in body["fighters"] if f["id"] == str(big))
    assert "profile" not in row
    assert row["ancestors"] == [{"id": "3", "sim": 0.7}]
    assert row["comps"] == [{"id": "2", "similarity": 0.9}]     # inline, no detail request
    assert len(row["habits"]) == len(body["habit_keys"])          # for the area lens
    assert row["ufc_fights"] is None and row["last_fight"] is None  # no bouts in the fixture


@needs_sqlite
def test_detail_has_profiles_and_comps(client):
    c, big = client
    body = c.get(f"/ufc/styles/fighters/{big}").json()
    assert body["fighter"]["profile"] == {"kd15s": 70}
    assert [r["id"] for r in body["comps"]] == ["2"]
    assert body["comps"][0]["similarity"] == 0.9
    assert body["comps"][0]["profile"] == {"kd15s": 70}


@needs_sqlite
def test_detail_404_for_ineligible_fighter(client):
    c, _ = client
    assert c.get("/ufc/styles/fighters/424242").status_code == 404



@needs_sqlite
def test_blend_is_lifted_out_of_the_profile(client):
    c, big = client
    rows = {f["id"]: f for f in c.get("/ufc/styles").json()["fighters"]}
    assert rows["2"]["blend"] == [{"family": 0, "share": 0.7}, {"family": 3, "share": 0.3}]
    assert rows[str(big)]["blend"] == [{"family": 0, "share": 1.0}]      # pre-v5 row: its family
    detail = c.get("/ufc/styles/fighters/2").json()["fighter"]
    assert "_blend" not in detail["profile"]
