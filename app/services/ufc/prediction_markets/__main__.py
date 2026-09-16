"""CLI for prediction-market ingestion.

    python -m app.services.ufc.prediction_markets --live
    python -m app.services.ufc.prediction_markets --backfill --dry-run
    python -m app.services.ufc.prediction_markets --backfill --venue kalshi --since 2025-05-01
    python -m app.services.ufc.prediction_markets --backfill --curves moneyline
"""

import argparse
import logging

from app.services.ufc.prediction_markets import run_backfill, run_live

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Kalshi / Polymarket prediction-market ingestion")
    p.add_argument("--venue", choices=["kalshi", "polymarket", "both"], default="both")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true",
                     help="Refresh open markets: quotes + curve extension. Idempotent.")
    mode.add_argument("--backfill", action="store_true",
                     help="Walk all events including settled, building history from venue candles.")
    p.add_argument("--since", help="Only events on/after this date (YYYY-MM-DD)")
    p.add_argument("--curves", choices=["moneyline", "all"], default="all",
                   help="Which Polymarket markets to pull price curves for. Kalshi is moneyline-only.")
    p.add_argument("--dry-run", action="store_true",
                   help="Match fights and report, write nothing. Use this first.")
    args = p.parse_args()

    if args.live:
        stats = run_live(venue=args.venue, curves=args.curves)
    else:
        stats = run_backfill(venue=args.venue, curves=args.curves,
                             since=args.since, dry_run=args.dry_run)

    print("\n=== summary ===")
    for venue, counts in stats.items():
        total = sum(counts.values())
        matched = counts.get("ok", 0) + counts.get("matched", 0)
        rate = f"{matched / total * 100:.1f}%" if total else "n/a"
        print(f"{venue:12s} {counts}  matched={rate}")


if __name__ == "__main__":
    main()
