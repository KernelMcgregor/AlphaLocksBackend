"""Non-UFC bouts for the profile's fight history (services/ufc/pro_record.py)."""
from __future__ import annotations

import datetime as dt

import pytest

from app.config import settings

pytestmark = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                reason="needs a scratch SQLite DATABASE_URL")


@pytest.fixture()
def db():
    from sqlalchemy import event

    from app.database import Base, SessionLocal, engine
    from app.models.ufc import SherdogBout, SherdogFighter, UFCEvent, UFCFight, UFCFighter

    tables = [t.__table__ for t in (UFCFighter, UFCEvent, UFCFight, SherdogFighter, SherdogBout)]
    Base.metadata.drop_all(engine, tables=list(reversed(tables)))
    Base.metadata.create_all(engine, tables=tables)
    counter = iter(range(5000, 10**6))

    def _ids(mapper, conn, target):
        if target.id is None:
            target.id = next(counter)
    for model in (SherdogFighter, SherdogBout):
        if not event.contains(model, "before_insert", _ids):
            event.listen(model, "before_insert", _ids)
    s = SessionLocal()
    s.add_all([UFCFighter(id=1, ufcstats_id="a", first_name="Y", last_name="A"),
               UFCFighter(id=2, ufcstats_id="b", first_name="J", last_name="J"),
               UFCEvent(id=10, ufcstats_id="e", name="UFC 1", date=dt.date(2025, 6, 1))])
    s.flush()
    s.add(UFCFight(id=100, ufcstats_id="f", date=dt.date(2025, 6, 1), event_id=10, red_fighter_id=1, blue_fighter_id=2))
    s.add_all([SherdogFighter(sherdog_id=11, url="u", name="Y A", ufc_fighter_id=1),
               SherdogFighter(sherdog_id=22, url="u", name="J J", ufc_fighter_id=2)])
    bout = dict(fighter_sherdog_id=11, method_class="DEC", round=3, time="5:00")
    s.add_all([
        # The UFC bout as Sherdog lists it, a day off: left out.
        SherdogBout(**bout, opponent_sherdog_id=22, opponent_name="J J", opponent_key="22", result="W",
                    date=dt.date(2025, 6, 2), event_name="UFC 1 - A vs. J", event_key="e1", promotion="UFC"),
        SherdogBout(**bout, opponent_sherdog_id=22, opponent_name="J J", opponent_key="22", result="L",
                    date=dt.date(2023, 1, 1), event_name="Bellator 300 - A vs. J", event_key="e2", promotion="Bellator"),
        SherdogBout(**bout, opponent_sherdog_id=None, opponent_name="Local Guy", opponent_key="local guy", result="W",
                    date=dt.date(2016, 1, 1), event_name="AFC 12 - Arena Fight Championship", event_key="e3", promotion="AFC"),
    ])
    s.commit()
    yield s
    s.close()


def test_external_fights_skip_ufc_and_link_opponents(db):
    from app.services.ufc.pro_record import external_fights
    rows = external_fights(db, 1)
    assert [r["promotion"] for r in rows] == ["Bellator", "AFC"]
    assert rows[0]["opponent_id"] == "2" and rows[0]["method"] == "Decision"
    assert rows[1]["opponent_id"] is None
    assert external_fights(db, 999) == []
