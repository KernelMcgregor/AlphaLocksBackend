"""Copy prediction-market tables (Polymarket, Kalshi) from one DB to another.

READ-ONLY on the source. Rows are copied with their ids, so re-running is idempotent
(existing ids are skipped). Markets whose fight is missing in the target keep fight_id
NULL rather than failing the foreign key.

Usage:
    python -m scripts.pull_prediction_markets --src "$PROD_URL" --dst postgresql://localhost/alocks_local
"""
from __future__ import annotations

import argparse

from sqlalchemy import create_engine, text

TABLES = (
    ("ufc_prediction_markets", "id"),
    ("ufc_prediction_market_quotes", "id"),
    ("ufc_prediction_market_history", "id"),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src = create_engine(a.src, execution_options={"postgresql_readonly": True})
    dst = create_engine(a.dst)
    with dst.connect() as d:
        local_fights = {r[0] for r in d.execute(text("select id from ufc.ufc_fights"))}
    for table, key in TABLES:
        with dst.connect() as d:
            cols = [r[0] for r in d.execute(text(
                "select column_name from information_schema.columns "
                "where table_schema='ufc' and table_name=:t order by ordinal_position"), {"t": table})]
            have = {r[0] for r in d.execute(text(f"select {key} from ufc.{table}"))}
        with src.connect() as s:
            s.execute(text("SET TRANSACTION READ ONLY"))
            rows = [dict(r._mapping) for r in s.execute(text(f"select {', '.join(cols)} from ufc.{table}"))]
        new = [r for r in rows if r[key] not in have]
        if table == "ufc_prediction_markets":
            for r in new:
                if r["fight_id"] is not None and r["fight_id"] not in local_fights:
                    r["fight_id"] = None
        with dst.begin() as d:
            for i in range(0, len(new), 5000):
                d.execute(text(f"insert into ufc.{table} ({', '.join(cols)}) values "
                               f"({', '.join(':' + c for c in cols)})"), new[i:i + 5000])
            d.execute(text(f"select setval(pg_get_serial_sequence('ufc.{table}', '{key}'), "
                           f"coalesce((select max({key}) from ufc.{table}), 1))"))
        print(f"{table}: {len(rows)} in source, {len(new)} copied")


if __name__ == "__main__":
    main()
