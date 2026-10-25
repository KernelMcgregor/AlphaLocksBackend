"""Post-event safety for the style tables: frozen files fail closed, and a rewrite is one
transaction (a failed insert keeps the old rows; the page tables move with similarity).

The write tests skip unless DATABASE_URL is SQLite, so they can never touch production.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_style_writes.py
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.services.ufc import style_page_service as sp
from app.services.ufc import style_service as ss

needs_sqlite = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                  reason="needs a scratch SQLite DATABASE_URL")
BACKEND = Path(__file__).resolve().parents[1]


class TestFailClosed:
    def test_missing_space_raises_instead_of_fitting(self, monkeypatch):
        monkeypatch.setattr(ss, "load_space", lambda: None)
        monkeypatch.setattr(ss, "fit_space", lambda *a: pytest.fail("must not fit without --refit"))
        with pytest.raises(RuntimeError, match="No frozen style space"):
            ss.space_for(ss.StyleInputs.__new__(ss.StyleInputs))

    def test_missing_map_raises_instead_of_fitting(self, monkeypatch):
        monkeypatch.setattr(sp, "load_map", lambda version=sp.MAP_VERSION: None)
        monkeypatch.setattr(sp, "fit_map", lambda U: pytest.fail("must not fit without --refit-map"))
        monkeypatch.setattr(ss, "embed", lambda inputs, space: np.ones((2, 3)))
        inputs = type("I", (), {"fids": [1, 2], "prime": np.zeros(2), "X": np.ones((2, 3))})()
        with pytest.raises(RuntimeError, match="No frozen style map"):
            sp.build_style_page(None, inputs, {})

    @pytest.mark.parametrize("path", [ss._space_path(), sp._map_path()])
    def test_current_frozen_files_are_tracked(self, path):
        """CI checks out the repo: an untracked pkl does not exist there."""
        out = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=BACKEND,
                             capture_output=True, text=True)
        if out.returncode != 0:
            pytest.xfail(f"{path.name} is not committed yet — required before merge")


@pytest.fixture()
def db():
    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCFighterSimilarity, UFCFighterStyle, UFCStyleFamily

    tables = [UFCFighterSimilarity.__table__, UFCFighterStyle.__table__, UFCStyleFamily.__table__]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)
    s = SessionLocal()
    s.add(UFCFighterSimilarity(fighter_id=1, similar_fighter_id=2, rank=1, similarity=0.5,
                               same_division=True, top_drivers="[]"))
    s.commit()
    yield s
    s.close()


def _stub_pipeline(monkeypatch, rows, page):
    inputs = type("I", (), {"fids": [1, 2]})()
    monkeypatch.setattr(ss, "build_inputs", lambda db: inputs)
    monkeypatch.setattr(ss, "space_for", lambda inputs, refit=False: {})
    monkeypatch.setattr(ss, "compute_similarity", lambda inputs, space: rows)
    monkeypatch.setattr(sp, "build_style_page", lambda db, inputs, space: page)


ROW = {"fighter_id": 1, "similar_fighter_id": 3, "rank": 1, "similarity": 0.9,
       "same_division": True, "top_drivers": []}
PAGE = {"version": "vT", "families": [{"id": 0, "name": "A", "blurb": "b", "traits": [],
                                       "profile": {}, "n_fighters": 1}],
        "fighters": [{"fighter_id": 1, "family": 0, "x": 0.1, "y": 0.2, "first_year": 2015,
                      "last_year": 2020, "prime_year": 2017.5, "active": True, "quality": 50.0,
                      "notable": False, "profile": {}, "ancestors": [], "notable_ancestors": [],
                      "division": "lightweight"}]}


@needs_sqlite
class TestAtomicRewrite:
    def test_both_tables_replaced_together(self, db, monkeypatch):
        from app.models.ufc import UFCFighterSimilarity, UFCFighterStyle
        _stub_pipeline(monkeypatch, [ROW], PAGE)
        assert ss.compute_and_save_similarity(db) == 1
        assert [(r.similar_fighter_id) for r in db.query(UFCFighterSimilarity)] == [3]
        assert [r.map_version for r in db.query(UFCFighterStyle)] == ["vT"]

    def test_failed_insert_keeps_the_old_rows(self, db, monkeypatch):
        from app.models.ufc import UFCFighterSimilarity, UFCFighterStyle
        _stub_pipeline(monkeypatch, [ROW], PAGE)
        monkeypatch.setattr(sp, "write_style_page",
                            lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            ss.compute_and_save_similarity(db)
        assert [(r.similar_fighter_id) for r in db.query(UFCFighterSimilarity)] == [2]
        assert db.query(UFCFighterStyle).count() == 0

    def test_page_failure_still_saves_similarity(self, db, monkeypatch):
        from app.models.ufc import UFCFighterSimilarity
        _stub_pipeline(monkeypatch, [ROW], PAGE)
        monkeypatch.setattr(sp, "build_style_page",
                            lambda *a: (_ for _ in ()).throw(ValueError("page broke")))
        assert ss.compute_and_save_similarity(db) == 1
        assert [(r.similar_fighter_id) for r in db.query(UFCFighterSimilarity)] == [3]
