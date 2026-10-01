"""Fetch per-sportsbook line-movement series from BestFightOdds for chosen fights.

The historical crawl kept only BFO's cross-book "Mean" series, which on recent events
includes Polymarket and Kalshi. Lead-lag tests against those exchanges need individual
sportsbooks. Uses bfo_scraper.BFOClient (disk cache, >= 2 s throttle, robots checked).

Usage:
    DATABASE_URL=... python -m scripts.fetch_book_series --kalshi-fights --books 21,25 \\
        --out data/bfo/book_series.csv
"""
from __future__ import annotations

import argparse
import csv

import pandas as pd
from sqlalchemy import create_engine, text

from app.config import settings
from app.services.ufc.bfo_scraper import DEFAULT_CACHE_DIR, BFOClient, series_points


def kalshi_fight_ids() -> set[int]:
    with create_engine(settings.DATABASE_URL).connect() as c:
        return {r[0] for r in c.execute(text(
            "select distinct fight_id from ufc.ufc_prediction_markets "
            "where platform = 'kalshi' and fight_id is not null"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kalshi-fights", action="store_true")
    ap.add_argument("--books", default="21,25")
    ap.add_argument("--out", default="data/bfo/book_series.csv")
    a = ap.parse_args()
    books = [int(b) for b in a.books.split(",")]
    link = (pd.read_csv("data/bfo/odds.csv", dtype={"db_fight_id": str},
                        usecols=["bfo_matchup_id", "bfo_event_slug", "db_fight_id"])
            .dropna(subset=["db_fight_id"]).drop_duplicates("bfo_matchup_id"))
    link["fight_id"] = link["db_fight_id"].str.split(".").str[0].astype("int64")
    if a.kalshi_fights:
        link = link[link["fight_id"].isin(kalshi_fight_ids())]
    client = BFOClient(DEFAULT_CACHE_DIR)
    if not client.robots_allows("/api/"):
        raise SystemExit("robots.txt disallows /api/")
    n = 0
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bfo_matchup_id", "book_id", "side", "ts", "decimal"])
        for mu, slug in zip(link["bfo_matchup_id"], link["bfo_event_slug"]):
            for b in books:
                for side in (1, 2):
                    series = client.chart(int(mu), side, b, referer=f"/events/{slug}")
                    for p in (series_points(series[0]) if series else []):
                        w.writerow([mu, b, side, p.ts.isoformat(), p.decimal])
            n += 1
            if n % 25 == 0:
                print(f"{n}/{len(link)} fights, {client.n_network} requests", flush=True)
    print(f"done: {n} fights, {client.n_network} network requests, {client.n_cache} cached")


if __name__ == "__main__":
    main()
