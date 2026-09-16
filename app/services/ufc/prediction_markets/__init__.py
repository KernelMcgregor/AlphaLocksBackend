"""Prediction-market ingestion (Kalshi, Polymarket).

Exchange prices differ from the sportsbook lines in `odds_scraper` / `bovada_scraper` in two ways
that drive the whole design:

* They are near-no-vig (yes + no ~ 1), so there is no devig assumption to make.
* The venues publish their own price *history*, so curves can be backfilled and a refresh can
  re-request a window rather than only recording "now". The job therefore pulls candles instead
  of polling snapshots, which makes the stored history self-healing and exactly idempotent.

Prices are stored in probability space in `ufc_prediction_market_*`, deliberately separate from
`ufc_fight_odds` -- see the docstring on `UFCPredictionMarket` for why that separation is load
bearing rather than stylistic.
"""

from app.services.ufc.prediction_markets.kalshi import run_kalshi_backfill, run_kalshi_live
from app.services.ufc.prediction_markets.polymarket import (
    run_polymarket_backfill, run_polymarket_live,
)

__all__ = [
    "run_kalshi_backfill", "run_kalshi_live",
    "run_polymarket_backfill", "run_polymarket_live",
    "run_live", "run_backfill",
]


def run_live(venue: str = "both", curves: str = "all") -> dict:
    """Refresh quotes and curves for open markets. Safe to run on any cadence; idempotent."""
    out = {}
    if venue in ("kalshi", "both"):
        out["kalshi"] = run_kalshi_live()
    if venue in ("polymarket", "both"):
        out["polymarket"] = run_polymarket_live(curves=curves)
    return out


def run_backfill(venue: str = "both", curves: str = "all", since: str | None = None,
                 dry_run: bool = False) -> dict:
    out = {}
    if venue in ("kalshi", "both"):
        out["kalshi"] = run_kalshi_backfill(since=since, dry_run=dry_run)
    if venue in ("polymarket", "both"):
        out["polymarket"] = run_polymarket_backfill(since=since, curves=curves, dry_run=dry_run)
    return out
