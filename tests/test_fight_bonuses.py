"""Bonus icons off ufcstats pages, and syncing them to ufc_fight_bonuses.

Parsing needs no database. The sync test skips unless DATABASE_URL is SQLite.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_fight_bonuses.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from app.config import settings
from app.services.ufc.fight_bonuses import event_page_bonuses, icons_in

FIX = Path(__file__).parent / "fixtures" / "ufcstats"


def _soup(name):
    return BeautifulSoup((FIX / name).read_text(), "html.parser")


def test_event_page_bonuses_ufc300():
    b = event_page_bonuses(_soup("event_ufc300.html"))
    assert len(b) == 13
    with_bonus = {k: v for k, v in b.items() if v}
    # Holloway-Gaethje: FOTN and Holloway's POTN (the belt icon is not a bonus);
    # Prochazka-Rakic: POTN.
    assert sorted(with_bonus.values()) == [["fotn", "potn"], ["potn"]]


def test_fight_title_icons_ufc140_main():
    title = _soup("fight_ufc140_main.html").select_one("i.b-fight-details__fight-title")
    assert icons_in(title) == ["fotn"]


@pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"), reason="needs a scratch SQLite DATABASE_URL")
def test_sync_replaces_and_credits_winner():
    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter
    from app.models.ufc_highlights import UFCFightBonus
    from app.services.ufc.fight_bonuses import sync_fight_bonuses, try_sync

    tables = [m.__table__ for m in (UFCFighter, UFCEvent, UFCFight, UFCFightBonus)]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)
    db = SessionLocal()
    try:
        db.add_all([UFCFighter(id=1, ufcstats_id="a", first_name="A", last_name="A"),
                    UFCFighter(id=2, ufcstats_id="b", first_name="B", last_name="B"),
                    UFCEvent(id=1, ufcstats_id="e", name="E", date=dt.date(2024, 1, 1)),
                    UFCFight(id=10, ufcstats_id="f", event_id=1, red_fighter_id=1, blue_fighter_id=2,
                             winner_id=2, date=dt.date(2024, 1, 1))])
        db.commit()
        f = db.get(UFCFight, 10)
        sync_fight_bonuses(db, f, ["fotn", "potn"])
        db.commit()
        rows = {b.bonus_type: b.fighter_id for b in db.query(UFCFightBonus)}
        assert rows == {"fotn": None, "potn": 2}
        try_sync(db, f, ["fotn"])          # POTN taken off the page: dropped
        db.commit()
        assert [b.bonus_type for b in db.query(UFCFightBonus)] == ["fotn"]
    finally:
        db.close()
