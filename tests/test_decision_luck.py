"""Decision luck (services/ufc/decision_luck.py) on scratch SQLite."""
from __future__ import annotations

import datetime as dt

import pytest

from app.config import settings

pytestmark = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                reason="needs a scratch SQLite DATABASE_URL")


@pytest.fixture()
def db():
    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCDeserveToWin, UFCEvent, UFCFight, UFCFighter

    tables = [t.__table__ for t in (UFCFighter, UFCEvent, UFCFight, UFCDeserveToWin)]
    Base.metadata.drop_all(engine, tables=list(reversed(tables)))
    Base.metadata.create_all(engine, tables=tables)
    from sqlalchemy import event
    counter = iter(range(5000, 10**6))

    def _ids(mapper, conn, target):
        if target.id is None:
            target.id = next(counter)
    if not event.contains(UFCDeserveToWin, "before_insert", _ids):
        event.listen(UFCDeserveToWin, "before_insert", _ids)
    s = SessionLocal()
    s.add_all([UFCFighter(id=i, ufcstats_id=str(i), first_name="F", last_name=str(i)) for i in (1, 2, 3, 4)])
    s.add(UFCEvent(id=10, ufcstats_id="e", name="E", date=dt.date(2025, 1, 1)))
    s.flush()
    day = dt.date(2025, 1, 1)
    # 1 beats 2 by decision while losing most replays (a gift); 3 beats 1 by decision
    # while 1 won most replays (a robbery); 1 knocks out 4 (a finish, left out).
    for fid, red, blue, winner, p_red, outcome in [
        (100, 1, 2, 1, 0.30, "red:sd"), (101, 3, 1, 3, 0.20, "red:ud"), (102, 1, 4, 1, 0.40, "red:ko"),
    ]:
        s.add(UFCFight(id=fid, ufcstats_id=str(fid), date=day, event_id=10, red_fighter_id=red,
                       blue_fighter_id=blue, winner_id=winner))
        s.flush()
        s.add(UFCDeserveToWin(fight_id=fid, model_version="v1", p_red=p_red, p_draw=0.0, p_blue=1 - p_red,
                              rounds_observed=3, rounds_scheduled=3, round_p_red="[]", top_cards="[]",
                              official_outcome=outcome, n_sims=10000))
    s.commit()
    yield s
    s.close()


def test_luck_counts_decisions_only(db):
    from app.services.ufc.decision_luck import decision_luck
    out = decision_luck(db, 1)
    assert out["decisions"] == 2
    assert out["won"] == 1.0
    assert out["deserved"] == pytest.approx(0.3 + 0.8)
    assert out["luck"] == pytest.approx(1.0 - 1.1)
    assert out["robberies"] == 1 and out["gifts"] == 1
    assert {f["opponent_name"] for f in out["fights"]} == {"F 2", "F 3"}
