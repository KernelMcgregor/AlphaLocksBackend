"""post_event_chain_stale: the nightly job's catch-up trigger.

Skips unless DATABASE_URL is SQLite, so it can never touch the production database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_post_event_staleness.py
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from app.config import settings

pytestmark = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                reason="needs a scratch SQLite DATABASE_URL")


@pytest.fixture()
def db():
    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFighterCareerStats, UFCFighterSimilarity

    tables = [UFCFighter.__table__, UFCEvent.__table__, UFCFight.__table__,
              UFCFighterSimilarity.__table__, UFCFighterCareerStats.__table__]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)
    s = SessionLocal()
    for fid in (1, 2):
        s.add(UFCFighter(id=fid, ufcstats_id=f"u{fid}", first_name="F", last_name=str(fid)))
    s.add(UFCEvent(id=1, ufcstats_id="e1", name="Card", date=date(2026, 10, 10)))
    s.commit()
    yield s
    s.close()


def _fight(db, fid, method, updated):
    from app.models.ufc import UFCFight
    db.add(UFCFight(id=fid, ufcstats_id=f"f{fid}", event_id=1, date=date(2026, 10, 10),
                    red_fighter_id=1, blue_fighter_id=2, method=method,
                    winner_id=1 if method else None, updated_at=updated))
    db.commit()


def _similarity(db, computed, career=None):
    """A chain run: career stats (early step) then similarity (last step). `career` overrides
    the career-stats time, to model a style-only run."""
    from app.models.ufc import UFCFighterCareerStats, UFCFighterSimilarity
    db.add(UFCFighterCareerStats(id=1, fighter_id=1, computed_at=career or computed))
    db.add(UFCFighterSimilarity(id=1, fighter_id=1, similar_fighter_id=2, rank=1, similarity=0.9,
                                same_division=True, top_drivers="[]", computed_at=computed))
    db.commit()


def test_results_after_last_chain_are_stale(db):
    from app.main import post_event_chain_stale
    _similarity(db, datetime(2026, 10, 4, 9, 0))
    _fight(db, 1, "KO/TKO", datetime(2026, 10, 11, 6, 0))
    assert post_event_chain_stale(db)


def test_chain_after_results_is_current(db):
    from app.main import post_event_chain_stale
    _fight(db, 1, "KO/TKO", datetime(2026, 10, 11, 6, 0))
    _similarity(db, datetime(2026, 10, 11, 9, 5))
    assert not post_event_chain_stale(db)


def test_upcoming_bouts_do_not_count(db):
    # A booked bout has no method; editing it (card order, a rebooking) is not a result.
    from app.main import post_event_chain_stale
    _similarity(db, datetime(2026, 10, 11, 9, 5))
    _fight(db, 2, None, datetime(2026, 10, 12, 1, 0))
    assert not post_event_chain_stale(db)


def test_never_run_with_results_is_stale(db):
    from app.main import post_event_chain_stale
    _fight(db, 1, "Decision - Unanimous", datetime(2026, 10, 11, 6, 0))
    assert post_event_chain_stale(db)


def test_a_style_only_run_does_not_mark_the_chain_fresh(db):
    # Similarity rewritten by hand after the card, but career stats never saw it.
    from app.main import post_event_chain_stale
    _fight(db, 1, "KO/TKO", datetime(2026, 10, 11, 6, 0))
    _similarity(db, datetime(2026, 10, 11, 9, 5), career=datetime(2026, 10, 4, 9, 0))
    assert post_event_chain_stale(db)
