"""Pull production-only rows down into the local database, without deleting anything.

WHY THIS EXISTS
---------------
`sync_db.py` replaces a whole table (TRUNCATE + copy). That is correct for building a
fresh local copy, but wrong once local has diverged: the local database holds repaired
fights and fighters that production does not, while production keeps accumulating new
bookings, results and odds from its crons. Neither side is a superset, so making local
current requires a merge.

TRUNCATE is also actively unsafe on the parent tables here -- `TRUNCATE ufc_events
CASCADE` deletes every fight that references those events.

WHAT IT DOES
------------
For each table, compares a natural key between production and local and INSERTs only the
rows production has that local lacks. Nothing is updated and nothing is deleted, so a
local repair can never be clobbered by this script. Row values that differ on both sides
(e.g. a winner filled in on production but not locally) are NOT reconciled here -- run
`rescrape_events.py` for that, which re-derives results from the source of truth.

Order matters: parents before children, so a new fight's event exists before it is
inserted.

Usage:
    python -m scripts.merge_from_prod --target postgresql://localhost/alocks_local
    python -m scripts.merge_from_prod --target ... --apply
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url

from app.config import settings
from app.database import Base
import app.models.ufc  # noqa: F401
import app.models.shared  # noqa: F401

# (table name, natural-key columns). Parents first -- FK order.
MERGE_PLAN = [
    ("ufc_events", ("ufcstats_id",)),
    ("ufc_fighters", ("ufcstats_id",)),
    ("ufc_fights", ("ufcstats_id",)),
    ("ufc_fight_odds", ("fight_id", "bookmaker")),
    ("ufc_fighter_rankings", ("fighter_id", "weight_class")),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="local URL to merge INTO")
    ap.add_argument("--source", default=settings.DATABASE_URL, help="production (read-only)")
    ap.add_argument("--apply", action="store_true", help="write; otherwise report only")
    args = ap.parse_args()

    host = (make_url(args.target).host or "").lower()
    if host not in ("", "localhost", "127.0.0.1", "::1"):
        sys.exit(f"Target {host!r} is not local. This script only writes to a local copy.")

    src = create_engine(args.source, pool_pre_ping=True)
    dst = create_engine(args.target)
    by_name = {t.name: t for t in Base.metadata.sorted_tables}

    print(f"source : production (read-only)\ntarget : {args.target}")
    print(f"mode   : {'APPLY' if args.apply else 'report only'}\n")

    total = 0
    for name, keycols in MERGE_PLAN:
        table = by_name[name]
        cols = [c.name for c in table.columns]

        def keyset(engine):
            with engine.connect() as c:
                rows = c.execute(select(*[table.c[k] for k in keycols])).all()
            return {tuple(r) for r in rows}

        missing = keyset(src) - keyset(dst)
        if not missing:
            print(f"  {name:<28} up to date")
            continue

        with src.connect() as sc:
            where = " OR ".join(
                "(" + " AND ".join(f"{k} = :{k}{i}" for k in keycols) + ")"
                for i in range(len(missing))
            )
            params = {}
            for i, key in enumerate(missing):
                for k, v in zip(keycols, key):
                    params[f"{k}{i}"] = v
            sel = text(f'SELECT {", ".join(cols)} FROM "{table.schema}"."{name}" '
                       f"WHERE {where}")
            rows = [dict(zip(cols, r)) for r in sc.execute(sel, params)]

        print(f"  {name:<28} +{len(rows)} row(s) from production")
        if args.apply and rows:
            with dst.begin() as dc:
                dc.execute(table.insert(), rows)
        total += len(rows)

    src.dispose()
    dst.dispose()
    print(f"\n{total} row(s) {'inserted' if args.apply else 'would be inserted'}.")
    if not args.apply:
        print("Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
