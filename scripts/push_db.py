"""Push the local working database back to production.

This is the mirror of `sync_db.py`, and it is the dangerous direction. `sync_db.py`
refuses to write to a Railway host on purpose; rather than weaken that guard, the
production write lives here where every safety check can be about this one operation.

WHAT MAKES THIS SAFE ENOUGH TO RUN
----------------------------------
1. PREFLIGHT IS MANDATORY AND IS THE DEFAULT. Nothing is written without `--push`.
   The preflight prints, per table, the local and production row counts and the
   delta, plus any column the local ORM has that production's table lacks.

2. SHRINKS ABORT THE RUN. The genuine hazard is not a bad copy -- it is that
   production accumulated rows since the local snapshot was taken (a scrape ran, a
   cron fired), and TRUNCATE + INSERT would silently delete them. Any table where
   local < production stops the push unless `--allow-shrink` is passed explicitly
   for that reason.

3. COLUMN DRIFT ABORTS THE RUN. `create_all()` creates missing *tables* but never
   adds missing *columns* to an existing one. The Glicko work added
   avg_sigma/rounds_seen/fights_seen/days_since_last_update to ufc_glicko_snapshots,
   so pushing into an un-migrated production table would fail mid-copy with half the
   tables already truncated. Detected up front instead, and `--add-columns` issues
   the ALTER TABLE statements.

4. ONE TABLE PER TRANSACTION. TRUNCATE and INSERT for a table share a transaction, so
   a failure leaves that table at its previous contents rather than empty.

There are no user or auth tables in this schema -- it is UFC data plus three shared
prediction tables -- so there is no user-generated content at risk. That is what makes
a wholesale replace acceptable here; it would not be in a schema with accounts.

Usage:
    # preflight (always do this first)
    python -m scripts.push_db --source postgresql://localhost/alocks_local

    # add any columns production is missing, then push
    python -m scripts.push_db --source postgresql://localhost/alocks_local --add-columns
    python -m scripts.push_db --source postgresql://localhost/alocks_local --push
"""
from __future__ import annotations

import argparse
import sys
import time

from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateColumn

from app.config import settings
from app.database import Base
import app.models.ufc  # noqa: F401  — registers every table on Base.metadata
import app.models.shared  # noqa: F401

BATCH = 5000

# The push target must look like production; a typo pointing at localhost would
# quietly do nothing useful, so require the opposite of sync_db.py's assertion.
PROD_HOST_MARKERS = ("rlwy.net", "railway", "cockroachlabs.cloud")


def _assert_production(target_url: str) -> None:
    host = (make_url(target_url).host or "").lower()
    if not any(m in host for m in PROD_HOST_MARKERS):
        sys.exit(
            f"Target host {host!r} does not look like production. This script only "
            "pushes TO production; use sync_db.py to write a local copy."
        )


def _assert_local_source(source_url: str) -> None:
    host = (make_url(source_url).host or "").lower()
    if host not in ("", "localhost", "127.0.0.1", "::1"):
        sys.exit(f"Source host {host!r} is not local. Refusing to copy prod->prod.")


def _missing_columns(engine, table):
    """Columns the ORM defines that the live table does not have.

    Returns [] when the table itself is absent -- create_all() will build it whole.
    """
    insp = inspect(engine)
    if not insp.has_table(table.name, schema=table.schema):
        return []
    live = {c["name"] for c in insp.get_columns(table.name, schema=table.schema)}
    return [c for c in table.columns if c.name not in live]


def _qualified(table) -> str:
    return f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="local SQLAlchemy URL to read from")
    ap.add_argument("--target", default=settings.DATABASE_URL,
                    help="defaults to DATABASE_URL (production)")
    ap.add_argument("--push", action="store_true",
                    help="actually write. Without this the run is preflight-only.")
    ap.add_argument("--add-columns", action="store_true",
                    help="ALTER TABLE ADD COLUMN for any column production is missing")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="proceed even though a table would lose rows")
    ap.add_argument("--only", default="", help="comma-separated table subset")
    args = ap.parse_args()

    _assert_local_source(args.source)
    _assert_production(args.target)

    tgt_url = make_url(args.target)
    print(f"source : {args.source}  (local)")
    print(f"target : {tgt_url.host}:{tgt_url.port}/{tgt_url.database}  (PRODUCTION)")
    print(f"mode   : {'PUSH — WILL WRITE' if args.push else 'preflight only (no writes)'}\n")

    src = create_engine(args.source, pool_pre_ping=True)
    dst = create_engine(args.target, pool_pre_ping=True)

    tables = list(Base.metadata.sorted_tables)  # FK-safe order
    if args.only:
        want = {t.strip() for t in args.only.split(",") if t.strip()}
        tables = [t for t in tables if t.name in want]
        missing = want - {t.name for t in tables}
        if missing:
            sys.exit(f"unknown table(s): {sorted(missing)}")

    # ---- preflight -------------------------------------------------------------
    print(f"{'table':<34}{'local':>10}{'prod':>10}{'delta':>10}   notes")
    print("-" * 78)
    plan, shrinking, drifted = [], [], []
    for table in tables:
        with src.connect() as sc:
            local_n = sc.execute(select(func.count()).select_from(table)).scalar() or 0
        try:
            with dst.connect() as dc:
                prod_n = dc.execute(select(func.count()).select_from(table)).scalar() or 0
            exists = True
        except Exception:
            prod_n, exists = 0, False

        miss = _missing_columns(dst, table) if exists else []
        notes = []
        if not exists:
            notes.append("NEW TABLE")
        if miss:
            notes.append(f"MISSING COLS: {', '.join(c.name for c in miss)}")
            drifted.append((table, miss))
        delta = local_n - prod_n
        if exists and delta < 0:
            notes.append("*** SHRINK ***")
            shrinking.append((table.name, prod_n, local_n))

        print(f"{table.name:<34}{local_n:>10,}{prod_n:>10,}{delta:>+10,}   "
              f"{'; '.join(notes)}")
        plan.append((table, local_n))
    print("-" * 78)

    if drifted and not args.add_columns:
        print("\nPRODUCTION IS MISSING COLUMNS. The push would fail partway through,")
        print("after some tables were already truncated. Re-run with --add-columns")
        print("(safe on its own: ADD COLUMN is additive and touches no rows).")
        return 1

    if args.add_columns and drifted:
        print("\nAdding missing columns to production...")
        for table, cols in drifted:
            with dst.begin() as conn:
                for col in cols:
                    ddl = CreateColumn(col).compile(dst).string
                    conn.execute(text(f"ALTER TABLE {_qualified(table)} ADD COLUMN {ddl}"))
                    print(f"  {table.name}.{col.name}")
        print("Columns added. Re-run preflight, then --push.")
        return 0

    if shrinking and not args.allow_shrink:
        print("\nREFUSING TO PUSH — these tables would lose rows:")
        for name, prod_n, local_n in shrinking:
            print(f"  {name:<34} prod {prod_n:>9,} -> local {local_n:>9,}")
        print("\nProduction most likely gained rows after the local copy was taken.")
        print("Re-sync those tables down first (scripts/sync_db.py --only ...), or pass")
        print("--allow-shrink if the deletion is intended.")
        return 1

    if not args.push:
        print("\nPreflight OK. Nothing was written. Re-run with --push to apply.")
        return 0

    # ---- push ------------------------------------------------------------------
    schemas = {t.schema for t in Base.metadata.sorted_tables if t.schema}
    if schemas:
        with dst.begin() as conn:
            for s in sorted(schemas):
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{s}"'))
    Base.metadata.create_all(dst)

    print(f"\nPushing {len(plan)} tables...\n")
    t_start = time.time()
    summary = []

    for table, total in plan:
        cols = [c.name for c in table.columns]
        copied = 0
        # TRUNCATE and INSERT in ONE transaction: a mid-copy failure rolls the table
        # back to its previous contents instead of leaving it empty.
        with src.connect() as sc, dst.begin() as dc:
            dc.execute(text(f"TRUNCATE TABLE {_qualified(table)} CASCADE"))
            if total:
                order = [table.c[c] for c in cols if c in ("id",)] or list(table.columns)[:1]
                result = sc.execution_options(
                    stream_results=True, yield_per=BATCH
                ).execute(select(table).order_by(*order))
                for chunk in result.partitions(BATCH):
                    rows = [dict(zip(cols, r)) for r in chunk]
                    if rows:
                        dc.execute(table.insert(), rows)
                        copied += len(rows)
                    if sys.stdout.isatty():
                        print(f"  {table.name:<32} {copied:>7}/{total:<7}",
                              end="\r", flush=True)

        with dst.connect() as dc:
            got = dc.execute(select(func.count()).select_from(table)).scalar() or 0
        summary.append((table.name, total, got, got == total))
        print(f"  {table.name:<32} {copied:>7}/{total:<7}  prod={got:<7} "
              f"{'OK ' if got == total else 'MISMATCH'}")

    src.dispose()
    dst.dispose()

    bad = [s for s in summary if not s[3]]
    print(f"\nDone in {time.time() - t_start:.0f}s. "
          f"{sum(s[1] for s in summary):,} rows across {len(summary)} tables.")
    if bad:
        print("\nROW COUNT MISMATCHES:")
        for name, exp, got, _ in bad:
            print(f"  {name:<34} expected {exp:>9,}  got {got:>9,}")
        return 1
    print("All table row counts match production.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
