"""Opening-line watcher: record each upcoming UFC bout's line the first time it appears.

Backtests put the model's edge at the OPENING line (positive closing-line value there,
~none at the close), and the only odds feed before this checked US books twice a week, by
which time an opener could be days old. This runs every two hours (line-watcher.yml):

  1. Find upcoming UFC events on BestFightOdds (sitemap + homepage, ~2 requests).
  2. Fetch each upcoming event page fresh (~1 request per card).
  3. Match its bouts to our upcoming ufc_fights rows.
  4. Append every sportsbook price that is new or has changed to ufc_fight_odds_history
     as bookmaker "BFO:<book>". Exchanges (Polymarket/Kalshi) are skipped: they trade
     in-play and are tracked separately. The first snapshot of a fight is its observed
     opening line; the last one before the card is its closing line.
  5. Parse the same page's prop rows (winner x method, goes to decision, total rounds),
     de-vig the sportsbook consensus (bfo_props.py) and append changed values to
     ufc_prop_odds_history. The first row of a fight is its opening prop price.
  6. Report which fights were priced for the first time, and which of their fighters have
     no Sherdog record yet, so the workflow can look them up and refresh predictions.

Writes GitHub step outputs (changed / new_fights) when GITHUB_OUTPUT is set.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
from urllib.parse import urlparse

from app.database import SessionLocal
from app.models.ufc import (
    SherdogFighter, UFCEvent, UFCFight, UFCFighter, UFCFightOddsHistory, UFCPropOddsHistory,
)

log = logging.getLogger("line_watcher")

SOURCE_PREFIX = "BFO:"
EXCHANGES = ("polymarket", "kalshi")
HORIZON_DAYS = 60  # ignore cards further out than this


def _normalised(a: int, b: int) -> tuple[float, float]:
    from app.services.ufc.market_anchor import american_to_prob
    pa, pb = american_to_prob(a), american_to_prob(b)
    s = pa + pb
    return pa / s, pb / s


def upcoming_event_refs(client, today: dt.date):
    from app.services.ufc import bfo_props
    from app.services.ufc import bfo_scraper as bfo

    refs = {e.slug: e for e in bfo.list_ufc_events(client, since=today, include_future=True)
            if e.date <= today + dt.timedelta(days=HORIZON_DAYS)}
    # The homepage lists newly announced cards that may not be in the sitemap yet.
    home = client.get("/", max_age_s=0)
    for path in set(re.findall(r'href="(/events/([a-z0-9-]+-\d+))"', home)):
        slug = path[1]
        if slug not in refs and bfo.is_ufc_event_slug(slug):
            refs[slug] = bfo.BFOEventRef(slug=slug, bfo_event_id=None,
                                         url=bfo.BASE_URL + path[0], date=None)
    return list(refs.values())


def watch(max_requests: int = 40) -> dict:
    from app.services.ufc import bfo_props
    from app.services.ufc import bfo_scraper as bfo

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    today = now.date()
    client = bfo.BFOClient(cache_dir=bfo.DEFAULT_CACHE_DIR, max_requests=max_requests)
    if not client.robots_allows("/events/"):
        raise SystemExit("robots.txt disallows /events/; not watching")

    # The prop table is new; create it if this run beats the backend's own migration.
    from app.database import engine
    UFCPropOddsHistory.__table__.create(engine, checkfirst=True)

    db = SessionLocal()
    try:
        fights = (db.query(UFCFight, UFCEvent)
                  .join(UFCEvent, UFCEvent.id == UFCFight.event_id)
                  .filter(UFCEvent.date >= today, UFCFight.winner_id.is_(None)).all())
        names = {f.id: f for f in db.query(UFCFighter).filter(UFCFighter.id.in_(
            {i for fi, _ in fights for i in (fi.red_fighter_id, fi.blue_fighter_id)}))}
        db_fights = [{
            "id": f.id, "fight_date": e.date, "event_name": e.name,
            "red": f"{names[f.red_fighter_id].first_name} {names[f.red_fighter_id].last_name}",
            "blue": f"{names[f.blue_fighter_id].first_name} {names[f.blue_fighter_id].last_name}",
        } for f, e in fights if f.red_fighter_id in names and f.blue_fighter_id in names]
        by_id = {f.id: f for f, _ in fights}

        seen_before = {fid for (fid,) in db.query(UFCFightOddsHistory.fight_id).filter(
            UFCFightOddsHistory.fight_id.in_(list(by_id)),
            UFCFightOddsHistory.bookmaker.like(f"{SOURCE_PREFIX}%")).distinct()}
        last_price: dict[tuple[int, str], tuple[int, int]] = {}
        for h in (db.query(UFCFightOddsHistory)
                  .filter(UFCFightOddsHistory.fight_id.in_(list(seen_before)),
                          UFCFightOddsHistory.bookmaker.like(f"{SOURCE_PREFIX}%"))
                  .order_by(UFCFightOddsHistory.captured_at)):
            last_price[(h.fight_id, h.bookmaker)] = (h.red_odds, h.blue_odds)

        last_prop = {}
        for h in (db.query(UFCPropOddsHistory)
                  .filter(UFCPropOddsHistory.fight_id.in_(list(by_id)),
                          UFCPropOddsHistory.source == "bfo_watch")
                  .order_by(UFCPropOddsHistory.captured_at)):
            last_prop[(h.fight_id, h.market)] = (h.prob, h.best_american)

        new_rows, prop_rows, priced_now = [], [], set()
        for ref in upcoming_event_refs(client, today):
            try:
                html = client.get(urlparse(ref.url).path, max_age_s=0)
                page = bfo.parse_event_page(html)
            except ValueError:
                continue  # card listed but no lines posted yet; checked again next run
            date = page.date or ref.date
            if date is None or date < today:
                continue
            mus = [{"bfo_matchup_id": m.matchup_id, "fighter_a": m.fighter_a,
                    "fighter_b": m.fighter_b, "event_date": date} for m in page.matchups]
            matches = bfo.match_matchups(mus, db_fights)
            cons = bfo_props.consensus(bfo_props.parse_props(html), page.books)
            for mu_id, hit in matches.items():
                c = cons.get(mu_id)
                if not c:
                    continue
                for market, q in bfo_props.corner_markets(c, hit["swapped"]).items():
                    prob = round(q["prob"], 4)
                    key = (hit["fight_id"], market)
                    if last_prop.get(key) == (prob, q["best_american"]):
                        continue   # record a row only when the consensus or best price moves
                    prop_rows.append(UFCPropOddsHistory(
                        fight_id=hit["fight_id"], market=market, prob=prob, n_books=c["n_books"],
                        overround=round(q["overround"], 4) if q["overround"] else None,
                        source="bfo_watch", captured_at=now, best_american=q["best_american"],
                        best_book=q["best_book"], median_american=q["median_american"]))
                    last_prop[key] = (prob, q["best_american"])
            for m in page.matchups:
                hit = matches.get(m.matchup_id)
                if not hit:
                    continue
                fid = hit["fight_id"]
                for book_id, prices in m.page_odds.items():
                    book = page.books.get(book_id, f"book_{book_id}")
                    if any(x in book.lower() for x in EXCHANGES):
                        continue
                    if not prices or None in prices[:2]:
                        continue
                    a, b = prices[0], prices[1]
                    red, blue = (b, a) if hit["swapped"] else (a, b)
                    key = (fid, SOURCE_PREFIX + book)
                    priced_now.add(fid)
                    if last_price.get(key) == (red, blue):
                        continue
                    pr, pb = _normalised(red, blue)
                    new_rows.append(UFCFightOddsHistory(
                        fight_id=fid, bookmaker=key[1], red_odds=red, blue_odds=blue,
                        red_implied_prob=round(pr, 4), blue_implied_prob=round(pb, 4),
                        captured_at=now, days_to_fight=(date - today).days))
                    last_price[key] = (red, blue)
        db.add_all(new_rows)
        db.add_all(prop_rows)
        db.commit()

        new_fights = sorted(priced_now - seen_before)
        linked = {u for (u,) in db.query(SherdogFighter.ufc_fighter_id)
                  .filter(SherdogFighter.ufc_fighter_id.isnot(None))}
        unlinked = sorted({i for fid in new_fights
                           for i in (by_id[fid].red_fighter_id, by_id[fid].blue_fighter_id)}
                          - linked)
    finally:
        db.close()

    out = {"price_rows": len(new_rows), "prop_rows": len(prop_rows), "new_fights": new_fights,
           "unlinked": unlinked,
           "requests": client.n_network}
    log.info(f"Line watcher: {len(new_rows)} new/changed prices, {len(prop_rows)} prop values, "
             f"{len(new_fights)} fights priced "
             f"for the first time, {len(unlinked)} of their fighters without Sherdog records, "
             f"{client.n_network} requests")
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as fh:
            fh.write(f"changed={'true' if new_fights else 'false'}\n")
            fh.write(f"new_fights={len(new_fights)}\n")
            fh.write(f"unlinked={len(unlinked)}\n")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    watch()
