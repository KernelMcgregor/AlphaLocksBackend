"""Add the right-facing portrait columns to ufc_fighters, idempotently.

The same ALTERs run in app/main.py run_migrations; this applies just these four without
starting the app (whose startup also runs create_all and the scheduler). All nullable
with no default, so on Postgres each is a metadata-only change: no table rewrite.

    PYTHONPATH=. venv/bin/python -m scripts.add_fighter_pose_columns            # apply
    PYTHONPATH=. venv/bin/python -m scripts.add_fighter_pose_columns --dry-run  # print only
"""

from __future__ import annotations

import sys

from sqlalchemy import inspect, text

from app.config import settings
from app.database import engine
from app.models.ufc import FIGHTER_POSE_COLS


def statements() -> list[str]:
    if settings.DATABASE_URL.startswith("sqlite"):
        # SQLite has no ADD COLUMN IF NOT EXISTS: skip the ones already there.
        have = {c["name"] for c in inspect(engine).get_columns("ufc_fighters")}
        return [f"ALTER TABLE ufc_fighters ADD COLUMN {col} {ddl.replace('BYTEA', 'BLOB')}"
                for col, ddl in FIGHTER_POSE_COLS if col not in have]
    return [f"ALTER TABLE ufc.ufc_fighters ADD COLUMN IF NOT EXISTS {col} {ddl}"
            for col, ddl in FIGHTER_POSE_COLS]


def main() -> None:
    dry = "--dry-run" in sys.argv
    stmts = statements()
    for s in stmts:
        print(("would run: " if dry else "run: ") + s)
    if dry or not stmts:
        return
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(text(s))
    have = {c["name"] for c in inspect(engine).get_columns(
        "ufc_fighters", schema=None if settings.DATABASE_URL.startswith("sqlite") else "ufc")}
    missing = [c for c, _ in FIGHTER_POSE_COLS if c not in have]
    print("ok" if not missing else f"MISSING after ALTER: {missing}")


if __name__ == "__main__":
    main()
