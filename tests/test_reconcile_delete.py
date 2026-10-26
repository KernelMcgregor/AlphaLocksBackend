"""Deleting a pulled bout must clear every table keyed to it (Oct 2026 post-event failures:
ufc_round_predictions held a foreign key the hand-kept list didn't know about)."""
import datetime as dt

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCRoundPrediction
from app.services.ufc.reconcile_service import _delete_fight, _fight_tables


def _session():
    eng = create_engine("sqlite://")
    event.listen(eng, "connect", lambda c, _: c.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(eng)
    return sessionmaker(bind=eng)()


def test_every_fk_to_fights_is_covered():
    names = {t.name for t, _ in _fight_tables()}
    assert {"ufc_round_predictions", "ufc_expected_stat_predictions", "ufc_prop_odds_history",
            "ufc_fight_predictions", "ufc_fight_stats"} <= names


def test_delete_fight_with_round_and_xs_predictions():
    db = _session()
    db.add_all([UFCFighter(id=1, ufcstats_id="a", first_name="A", last_name="A"),
                UFCFighter(id=2, ufcstats_id="b", first_name="B", last_name="B"),
                UFCEvent(id=10, ufcstats_id="e", name="E", date=dt.date(2026, 10, 31))])
    db.flush()
    f = UFCFight(id=100, ufcstats_id="f", event_id=10, red_fighter_id=1, blue_fighter_id=2)
    db.add(f)
    db.flush()
    db.add(UFCRoundPrediction(id=1, fight_id=100, curve="[]"))
    db.flush()
    _delete_fight(db, f)
    db.flush()
    assert db.scalar(select(func.count()).select_from(UFCFight)) == 0
    assert db.scalar(select(func.count()).select_from(UFCRoundPrediction)) == 0
