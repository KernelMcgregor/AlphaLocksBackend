"""Shared reads for the fighter-page extras (percentiles, age curve, rating history,
splits, camps, rankings time-lapse).

Everything in the fighter_extras_* services is display-only. Nothing here may feed the
prediction model: the camp registry, age curves and splits are descriptive and several
of them (splits by closing odds, camp rosters as they stand today) would leak if used
as features.

`extras_session(db)` is where the precomputed tables (ufc_camps*, ufc_age_curves) are
read and written. It is `db` itself unless FIGHTER_EXTRAS_DB_URL is set, which points
those tables at a separate database — a scratch SQLite for page testing, so the
precompute can be exercised end to end against production reads without a single
production write.
"""
from __future__ import annotations

import os
from collections import defaultdict
from contextlib import contextmanager
from datetime import date

from sqlalchemy import create_engine, or_
from sqlalchemy.orm import Session, sessionmaker

from app.models.ufc import UFCEvent, UFCFight
from app.services.ufc.fighter_registry import classify_weight_class, current_division

EXTRAS_DB_ENV = "FIGHTER_EXTRAS_DB_URL"

_override: dict = {}


def _override_factory():
    url = os.environ.get(EXTRAS_DB_ENV)
    if not url:
        return None
    if _override.get("url") != url:
        eng = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
        # The models carry schema "ufc" when the main DATABASE_URL is Postgres; SQLite has
        # no schemas, so map it away on this engine only.
        if url.startswith("sqlite"):
            eng = eng.execution_options(schema_translate_map={"ufc": None})
        _override.update(url=url, engine=eng, factory=sessionmaker(bind=eng, autoflush=False))
    return _override["factory"]


@contextmanager
def extras_session(db: Session):
    """Session for the precomputed extras tables (see module docstring)."""
    factory = _override_factory()
    if factory is None:
        yield db
        return
    s = factory()
    try:
        yield s
    finally:
        s.close()


def extras_engine(default_engine):
    """Engine the precompute writers create/write tables on."""
    return _override["engine"] if _override_factory() is not None else default_engine


def completed_filter():
    """A bout that happened: dated no later than today, with a winner or a method."""
    return (
        UFCFight.date.isnot(None),
        UFCFight.date <= date.today(),
        or_(UFCFight.winner_id.isnot(None), UFCFight.method.isnot(None)),
    )


def outcome(fight, fighter_id: int) -> str:
    """W / L / D / NC from one fighter's side."""
    if fight.winner_id:
        return "W" if int(fight.winner_id) == int(fighter_id) else "L"
    m = fight.method or ""
    if any(k in m for k in ("No Contest", "Overturned", "Could Not Continue", "DQ")):
        return "NC"
    return "D" if ("Dec" in m or "Draw" in m) else "NC"


def is_finish(method: str | None) -> bool:
    m = method or ""
    return bool(m) and "Dec" not in m and m not in ("Overturned", "Could Not Continue", "DQ", "Other")


def fighter_bouts(db: Session, fighter_id: int) -> list:
    """The fighter's completed UFC bouts, oldest first, with the event location."""
    return (
        db.query(UFCFight, UFCEvent.location)
        .outerjoin(UFCEvent, UFCEvent.id == UFCFight.event_id)
        .filter(or_(UFCFight.red_fighter_id == fighter_id, UFCFight.blue_fighter_id == fighter_id))
        .filter(*completed_filter())
        .order_by(UFCFight.date, UFCFight.id)
        .all()
    )


def division_map(db: Session) -> dict[int, str]:
    """fighter id -> current division (Tapology's last-two-bouts rule, as rankings use)."""
    rows = (
        db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id, UFCFight.weight_class)
        .filter(*completed_filter())
        .order_by(UFCFight.date, UFCFight.id)
        .all()
    )
    seen: dict[int, list[str]] = defaultdict(list)
    for red, blue, wc in rows:
        key = classify_weight_class(wc)
        if key == "unknown":
            continue
        for fid in (red, blue):
            if fid is not None:
                seen[int(fid)].append(key)
    return {fid: current_division(divs) for fid, divs in seen.items()}


def division_label(key: str | None) -> str | None:
    from app.services.ufc.ranking_publisher import WEIGHT_CLASS_LABELS
    if not key:
        return None
    if key == "all":
        return "All divisions"
    base = WEIGHT_CLASS_LABELS.get(key, key.replace("_", " ").title())
    return f"Women's {base}" if key.startswith("w_") else base


def age_on(dob, on) -> float | None:
    """Age in years: whole calendar years (so int() is the age on a birthday) plus the
    fraction of the year since the last birthday."""
    if not dob or not on:
        return None
    years = on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))
    try:
        last = dob.replace(year=dob.year + years)
    except ValueError:  # 29 February
        last = dob.replace(year=dob.year + years, day=28)
    return years + (on - last).days / 366.0
