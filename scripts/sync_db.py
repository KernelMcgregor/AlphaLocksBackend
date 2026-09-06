"""Copy the production database to a local one, table by table.

WHY NOT pg_dump
---------------
Railway runs PostgreSQL 18.6; the local client is pg_dump 14.20 (Homebrew
postgresql@14). pg_dump refuses to read a server newer than itself, so the usual
`pg_dump | pg_restore` route needs `brew install postgresql@18` and a second server.

The schema is 17 tables with no Postgres-specific types (no JSONB/ARRAY/UUID), so a
portable row-copy through SQLAlchemy round-trips cleanly and works across versions.

The local schema is created from `Base.metadata`, NOT by replaying `app/migrations/*.sql`.
Those migrations have drifted — several columns were added later by `main.py` — so the ORM
metadata is the only description of the schema that is guaranteed to match what the code
expects.

SAFETY
------
Production is opened read-only and never written. Every write targets the local URL, and
the script refuses to run if the target looks like the production host.

Usage:
    python -m scripts.sync_db --target postgresql://localhost/alocks_local --create
    python -m scripts.sync_db --target postgresql://localhost/alocks_local --truncate
    python -m scripts.sync_db --target ... --only ufc_fights,ufc_fight_stats
"""
from __future__ import annotations

import argparse
import sys
import time

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import make_url

from app.config import settings
from app.database import Base
import app.models.ufc  # noqa: F401  — registers every table on Base.metadata

BATCH = 5000

# Hosts we must never write to.
PROD_HOST_MARKERS = ("rlwy.net", "railway", "cockroachlabs.cloud", "proxy.rlwy")


def _assert_local(target_url: str) -> None:
    url = make_url(target_url)
    host = (url.host or "").lower()
    if any(m in host for m in PROD_HOST_MARKERS):
        sys.exit(f"REFUSING to write to what looks like production: host={host!r}")
    if host not in ("", "localhost", "127.0.0.1", "::1"):
        sys.exit(
            f"Target host {host!r} is not local. Pass an explicit localhost URL — this "
            "script only ever writes to a local copy."
        )


def _create_database_if_needed(target_url: str) -> None:
    """CREATE DATABASE cannot run inside a transaction, so use AUTOCOMMIT on /postgres."""
    url = make_url(target_url)
    dbname = url.database
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": dbname}
        ).scalar()
        if exists:
            print(f"  database {dbname!r} already exists")
        else:
            conn.execute(text(f'CREATE DATABASE "{dbname}"'))
            print(f"  created database {dbname!r}")
    admin.dispose()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="local SQLAlchemy URL to write to")
    ap.add_argument("--source", default=settings.DATABASE_URL,
                    help="defaults to DATABASE_URL (production)")
    ap.add_argument("--create", action="store_true", help="CREATE DATABASE if absent")
    ap.add_argument("--truncate", action="store_true",
                    help="empty target tables first (refresh in place)")
    ap.add_argument("--only", default="", help="comma-separated table subset")
    args = ap.parse_args()

    _assert_local(args.target)

    src_url = make_url(args.source)
    print(f"source : {src_url.host}:{src_url.port}/{src_url.database} (read-only)")
    print(f"target : {args.target}")

    if args.create:
        _create_database_if_needed(args.target)

    src = create_engine(args.source, pool_pre_ping=True)
    dst = create_engine(args.target)

    with src.connect() as c:
        print(f"  source server_version: {c.execute(text('show server_version')).scalar()}")
    with dst.connect() as c:
        print(f"  target server_version: {c.execute(text('show server_version')).scalar()}")

    # Most UFC tables live in a non-public "ufc" schema. create_all() will not create
    # the schema itself, so do it first or every CREATE TABLE fails on InvalidSchemaName.
    schemas = {t.schema for t in Base.metadata.sorted_tables if t.schema}
    if schemas:
        with dst.begin() as conn:
            for s in sorted(schemas):
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{s}"'))
        print(f"  ensured schema(s): {', '.join(sorted(schemas))}")

    print("\nCreating schema from ORM metadata...")
    Base.metadata.create_all(dst)

    tables = list(Base.metadata.sorted_tables)  # FK-safe order
    if args.only:
        want = {t.strip() for t in args.only.split(",") if t.strip()}
        tables = [t for t in tables if t.name in want]
        missing = want - {t.name for t in tables}
        if missing:
            sys.exit(f"unknown table(s): {sorted(missing)}")

    if args.truncate:
        # Reverse order so children are emptied before parents.
        with dst.begin() as conn:
            for t in reversed(tables):
                conn.execute(text(f'TRUNCATE TABLE "{t.name}" CASCADE'))
        print(f"Truncated {len(tables)} target tables")

    print(f"\nCopying {len(tables)} tables...\n")
    t_start = time.time()
    summary = []

    for table in tables:
        cols = [c.name for c in table.columns]
        with src.connect() as sc:
            total = sc.execute(select(func.count()).select_from(table)).scalar() or 0

            copied = 0
            if total:
                # Deterministic order so --resume-style reruns are comparable, and so a
                # partial copy is at least a prefix rather than an arbitrary sample.
                order = [table.c[c] for c in cols if c in ("id",)] or list(table.columns)[:1]
                stmt = select(table).order_by(*order)
                result = sc.execution_options(stream_results=True, yield_per=BATCH).execute(stmt)

                with dst.begin() as dc:
                    for chunk in result.partitions(BATCH):
                        rows = [dict(zip(cols, r)) for r in chunk]
                        if rows:
                            dc.execute(table.insert(), rows)
                            copied += len(rows)
                        # Carriage-return progress only when attached to a terminal;
                        # piping to a file otherwise produces one giant unreadable line.
                        if sys.stdout.isatty():
                            print(f"  {table.name:<32} {copied:>7}/{total:<7}",
                                  end="\r", flush=True)

        with dst.connect() as dc:
            got = dc.execute(select(func.count()).select_from(table)).scalar() or 0

        ok = "OK " if got == total else "MISMATCH"
        summary.append((table.name, total, got, got == total))
        print(f"  {table.name:<32} {copied:>7}/{total:<7}  target={got:<7} {ok}")

    dst.dispose()
    src.dispose()

    bad = [s for s in summary if not s[3]]
    print(f"\nDone in {time.time() - t_start:.0f}s. "
          f"{sum(s[1] for s in summary):,} source rows across {len(summary)} tables.")
    if bad:
        print("\nROW COUNT MISMATCHES:")
        for name, exp, got, _ in bad:
            print(f"  {name:<32} expected {exp:>8}  got {got:>8}")
        return 1
    print("All table row counts match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
