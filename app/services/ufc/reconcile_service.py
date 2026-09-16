"""Remove bouts that were announced, stored, and then pulled from the card.

ufcstats publishes a bout as soon as it is announced, and simply drops it from the event page if
it is later cancelled or the opponent is changed. Nothing in the scrape pipeline notices that
removal: `upsert_fight` only ever inserts or updates, so a cancelled bout stays in the database
forever. Noche UFC 2026-09-12 is the worked example -- ufcstats lists 13 fights, we stored 16,
and the three extras are Rodriguez vs Silva, Gastelum vs Belgaroui and Jimenez vs Vera, all
rebooked or scrapped. Jean Silva appears twice on the same card in our copy of it.

The damage is not cosmetic. A phantom bout shows up on the upcoming-events page, gets a winner
prediction, a method prediction and SHAP values written against it, can have an AI preview
generated for it, and offers a second candidate when anything tries to match an external feed's
fight to ours.

Detection is by re-fetching the event page and diffing on `ufcstats_id`. Heuristics on the fight
row itself do not work: a cancelled bout carries a full set of 48 `ufc_fight_stats` rows exactly
like a real one, because those are created empty when the bout is first seen and only filled in
afterwards. Presence on the live page is the only authoritative signal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re

from sqlalchemy import delete

from app.database import SessionLocal
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFightOdds, UFCFightOddsHistory, UFCFightPrediction,
    UFCFightPreview, UFCFightShapValue, UFCFightStats, UFCGlickoSnapshot,
    UFCMethodOdds, UFCMethodPrediction,
    UFCPredictionMarket, UFCPredictionMarketHistory, UFCPredictionMarketQuote,
)
from app.services.ufc.scraper import Scraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reconcile")

FIGHT_ID_RE = re.compile(r"fight-details/([0-9a-f]{16})")

#: Every table keyed on a fight, deepest dependency first.
_DEPENDENTS = (
    UFCFightStats, UFCFightPrediction, UFCMethodPrediction, UFCFightShapValue,
    UFCFightOdds, UFCFightOddsHistory, UFCMethodOdds, UFCFightPreview, UFCGlickoSnapshot,
)


def live_fight_ids(scraper: Scraper, event_ufcstats_id: str) -> set[str]:
    """The `ufcstats_id` of every bout currently listed on an event's page."""
    soup = scraper.fetch(f"http://ufcstats.com/event-details/{event_ufcstats_id}")
    if soup is None:
        return set()
    return set(FIGHT_ID_RE.findall(str(soup)))


def _delete_fight(db, fight: UFCFight) -> None:
    """Remove a fight and everything keyed to it."""
    # Prediction-market rows hang off their own catalog table, so the history and quotes have to
    # go before the markets, which go before the fight.
    market_ids = [
        m.id for m in db.query(UFCPredictionMarket)
        .filter(UFCPredictionMarket.fight_id == fight.id).all()
    ]
    if market_ids:
        db.execute(delete(UFCPredictionMarketHistory).where(
            UFCPredictionMarketHistory.market_id.in_(market_ids)))
        db.execute(delete(UFCPredictionMarketQuote).where(
            UFCPredictionMarketQuote.market_id.in_(market_ids)))
        db.execute(delete(UFCPredictionMarket).where(UFCPredictionMarket.id.in_(market_ids)))

    for model in _DEPENDENTS:
        db.execute(delete(model).where(model.fight_id == fight.id))
    db.execute(delete(UFCFight).where(UFCFight.id == fight.id))


def reconcile_event(db, scraper: Scraper, event: UFCEvent, dry_run: bool = True) -> list[UFCFight]:
    """Diff one event against its live page and drop bouts that are no longer on it."""
    live = live_fight_ids(scraper, event.ufcstats_id)

    # An empty result means the fetch failed, the page changed shape, or the PoW challenge was
    # not solved -- never that a real card lost every bout. Deleting on it would wipe the event.
    if not live:
        log.warning(f"  {event.name}: no fights parsed from live page — skipping (fetch failure?)")
        return []

    ours = db.query(UFCFight).filter(UFCFight.event_id == event.id).all()
    stale = [f for f in ours if f.ufcstats_id not in live]
    if not stale:
        return []

    log.info(f"  {event.name} ({event.date}): {len(ours)} stored vs {len(live)} live — "
             f"{len(stale)} to remove")
    for fight in stale:
        red = fight.red_fighter.last_name if fight.red_fighter else "?"
        blue = fight.blue_fighter.last_name if fight.blue_fighter else "?"
        settled = " HAS RESULT" if fight.winner_id else ""
        log.info(f"    - {red} vs {blue} [{fight.ufcstats_id}]{settled}")
        if fight.winner_id:
            # A bout with a recorded winner that has vanished from the page is not a cancellation.
            # Far likelier a scrape or page-shape problem, and deleting a real result is not
            # recoverable from here.
            log.warning("      skipped: has a recorded winner, refusing to delete")
            continue
        if not dry_run:
            _delete_fight(db, fight)

    if not dry_run:
        db.commit()
    return stale


def refresh_affected(previews: bool = True) -> dict[str, str]:
    """Regenerate predictions (and optionally previews) after a reconcile removed something.

    A cancelled bout is almost always replaced rather than simply dropped, and the replacement
    arrives as a brand new fight row with no prediction, no method prediction and no preview. The
    upcoming card would otherwise show the new bout blank until the next nightly pipeline.

    Both prediction passes rebuild every upcoming fight rather than just the affected event: the
    generators are written to sweep, there is no per-event entry point, and the cost is a single
    model load either way.
    """
    import importlib

    results: dict[str, str] = {}
    steps = [
        ("Winner Predictions", "app.services.ufc.model:generate_predictions"),
        ("Method Predictions", "app.services.ufc.method_model:generate_method_predictions"),
    ]
    if previews:
        # Only fills gaps -- `generate_all_upcoming_previews` skips fights that already have one
        # unless forced, so this costs DeepSeek tokens for the replacement bout alone.
        steps.append(
            ("Previews", "app.services.ufc.preview_service:generate_all_upcoming_previews")
        )

    for label, target in steps:
        module_name, func_name = target.split(":")
        try:
            getattr(importlib.import_module(module_name), func_name)()
            log.info(f"{label}: done")
            results[label] = "done"
        except Exception as e:
            # Same reasoning as refresh_after_event: one failed stage must not strand the rest.
            log.exception(f"{label} failed")
            results[label] = f"error: {e}"
    return results


def run_reconcile(days_back: int = 30, days_forward: int = 120, dry_run: bool = True,
                  refresh: bool = True, previews: bool = True) -> dict:
    """Reconcile every event in a window around today.

    Defaults span announced-but-unfought cards plus the recent past, since a bout is usually
    pulled in the weeks before the event and the removal should be picked up on the next pass.
    """
    db = SessionLocal()
    scraper = Scraper()
    today = dt.date.today()
    lo, hi = today - dt.timedelta(days=days_back), today + dt.timedelta(days=days_forward)

    events = (
        db.query(UFCEvent)
        .filter(UFCEvent.date >= lo, UFCEvent.date <= hi)
        .order_by(UFCEvent.date)
        .all()
    )
    log.info(f"Reconciling {len(events)} events from {lo} to {hi}"
             f"{' (dry run — nothing will be deleted)' if dry_run else ''}")

    removed, affected = 0, []
    try:
        for event in events:
            stale = reconcile_event(db, scraper, event, dry_run=dry_run)
            if stale:
                removed += len([f for f in stale if not f.winner_id])
                affected.append(event.name)
    finally:
        db.close()

    log.info(f"Done: {removed} stale fights across {len(affected)} events"
             f"{' (dry run)' if dry_run else ''}")

    out = {"removed": removed, "events": affected, "dry_run": dry_run}
    # Only when something actually changed: regenerating on every no-op pass would burn a model
    # load and a DeepSeek bill on a job that is meant to run every couple of hours.
    if removed and refresh and not dry_run:
        log.info("Regenerating predictions for the replacement bouts")
        out["refresh"] = refresh_affected(previews=previews)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Remove cancelled/rebooked fights from the DB")
    p.add_argument("--days-back", type=int, default=30)
    p.add_argument("--days-forward", type=int, default=120)
    p.add_argument("--apply", action="store_true",
                   help="Actually delete. Without this the run only reports.")
    p.add_argument("--no-refresh", action="store_true",
                   help="Delete only; skip regenerating predictions for replacement bouts.")
    p.add_argument("--no-previews", action="store_true",
                   help="Regenerate predictions but not AI previews (saves DeepSeek tokens).")
    args = p.parse_args()
    run_reconcile(days_back=args.days_back, days_forward=args.days_forward,
                  dry_run=not args.apply, refresh=not args.no_refresh,
                  previews=not args.no_previews)
