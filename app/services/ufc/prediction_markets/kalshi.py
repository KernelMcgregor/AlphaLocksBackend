"""Kalshi UFC moneyline ingestion.

Kalshi lists one event per fight under series `KXUFCFIGHT`, each holding exactly two mutually
exclusive binary markets -- one per fighter. Moneyline only; Kalshi carries no method or round
props for UFC, which is the gap Polymarket fills.

Everything here is unauthenticated. Market reads need no API key, so this adds no secrets.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from app.models.ufc import UFCPredictionMarket
from app.services.ufc.prediction_markets.common import (
    BOUT_LEAD, NoSuchCard, days_to_fight, fetch_json, last_history_ts, match_fight, open_session,
    parse_ts, realign_cards, record_history, upsert_market, upsert_quote,
)

log = logging.getLogger("prediction_markets.kalshi")

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXUFCFIGHT"
HOST = "kalshi"

#: Kalshi's own labels for a settled binary market.
_RESULT_TO_OUTCOME = {"yes": 1.0, "no": 0.0}


def _f(value) -> float | None:
    """Kalshi returns money and size as decimal *strings* ('0.5400', '81580.10')."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick(d: dict, *keys) -> float | None:
    """First present key, as a float.

    The live and historical candlestick endpoints return the *same data under different names*:
    live gives `price.close_dollars`, `volume_fp`, `open_interest_fp`, while /historical gives
    `price.close`, `volume`, `open_interest`. Reading only the live spelling silently yields an
    empty curve for every market older than the historical cutoff -- which is most of them, and
    the failure is invisible because the request itself succeeds.
    """
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return _f(d[key])
    return None


def _cutoff() -> datetime | None:
    """Boundary before which settled data moves to the /historical mirrors."""
    data = fetch_json(HOST, f"{API}/historical/cutoff")
    if not data:
        return None
    ts = data.get("market_settled_ts")
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def list_fight_events(status: str | None = None) -> list[dict]:
    """Every event in the series, following the cursor to exhaustion."""
    events, cursor = [], ""
    while True:
        params = {"series_ticker": SERIES, "limit": 200}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        data = fetch_json(HOST, f"{API}/events", params)
        if not data:
            break
        batch = data.get("events") or []
        events.extend(batch)
        cursor = data.get("cursor") or ""
        if not cursor or not batch:
            break
    return events


def fetch_markets(event_ticker: str, historical: bool = False) -> list[dict]:
    base = f"{API}/historical/markets" if historical else f"{API}/markets"
    data = fetch_json(HOST, base, {"event_ticker": event_ticker})
    return (data or {}).get("markets") or []


#: Kalshi rejects any candlestick request spanning more than this many periods, with a 400 whose
#: body spells out the limit ("requested time range with candlesticks: 5347.5, max: 5000"). At
#: hourly resolution that is ~208 days, which real markets exceed: a bout announced months ahead
#: opens its market immediately. The request has to be split rather than truncated, or the early
#: part of the curve is lost.
MAX_CANDLES_PER_REQUEST = 5000


def fetch_candles(ticker: str, start_ts: int, end_ts: int, interval: int = 60,
                  historical: bool = False) -> list[dict]:
    """OHLC for price / yes_bid / yes_ask, in windows the API will accept.

    `interval` is minutes: 1, 60, or 1440.
    """
    if historical:
        url = f"{API}/historical/markets/{ticker}/candlesticks"
    else:
        url = f"{API}/series/{SERIES}/markets/{ticker}/candlesticks"

    span = interval * 60 * (MAX_CANDLES_PER_REQUEST - 1)  # seconds per safe chunk
    out: list[dict] = []
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + span, end_ts)
        data = fetch_json(HOST, url, {
            "start_ts": cursor, "end_ts": chunk_end, "period_interval": interval,
        })
        out.extend((data or {}).get("candlesticks") or [])
        cursor = chunk_end + 1
    return out


def _named_markets(markets: list[dict]) -> list[dict] | None:
    """The pair of markets that carry a fighter name, in order.

    The name comes from each market's own `yes_sub_title`, not `no_sub_title`: despite the name,
    Kalshi sets *both* sub-titles to the market's own fighter, so a single market never names its
    opponent. The event's `sub_title` ('Van vs Pantoja') only carries surnames. Reading one full
    name off each of the pair's two markets is the only source that gives both fighters in full.

    Returning the markets rather than bare names keeps each name welded to the market it came
    from -- a market missing a sub-title would otherwise desync the two lists and hand every
    outcome to the wrong corner.
    """
    named = [m for m in markets if m.get("yes_sub_title")]
    return named[:2] if len(named) >= 2 else None


def _event_date(markets: list[dict]) -> date | None:
    for m in markets:
        ts = m.get("occurrence_datetime") or m.get("close_time")
        if ts:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    return None


def closed_at(market: dict) -> datetime | None:
    """When the market really closed -- the bout's end -- or None while it is still open.

    Only a closed market's `close_time` is an observation. On an open market it is a scheduled
    placeholder, and so is `occurrence_datetime`: a per-bout *slot* that can sit hours after the
    bout (an early prelim on 19 Sep 2026 had 03:00 UTC and closed on its result at 22:00).
    """
    if market.get("status") in ("finalized", "settled", "closed"):
        return parse_ts(market.get("close_time"))
    return None


def bout_reference(market: dict) -> datetime | date | None:
    """Provisional anchor for a curve point at write time.

    Only provisional: `realign_cards` rewrites every point against the card-aware cutoff at the
    end of each run. Deliberately ignores `occurrence_datetime` (see `closed_at`) -- with nothing
    observed, the caller falls back to the event date, which is early rather than late.
    """
    close = closed_at(market)
    return close - BOUT_LEAD if close else None


def _is_historical(market: dict, cutoff: datetime | None) -> bool:
    """Whether this market's data has aged out of the live endpoints.

    Keys off `close_time`, not `market_settled_ts`: the latter comes back null even on markets
    Kalshi reports as `finalized`, so it cannot be used to route the request.
    """
    if cutoff is None:
        return False
    ts = market.get("close_time")
    if not ts:
        return False
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) < cutoff


def _ingest_event(db, event: dict, cutoff: datetime | None, *, fine_grained: bool = False,
                  dry_run: bool = False, touched: set | None = None) -> str:
    """Ingest one fight. Returns a status string for the run summary."""
    event_ticker = event["event_ticker"]
    markets = fetch_markets(event_ticker)
    if len(markets) < 2:
        markets = fetch_markets(event_ticker, historical=True)
    if len(markets) < 2:
        return "no_markets"

    pair = _named_markets(markets)
    fight_date = _event_date(markets)
    if not pair or not fight_date:
        return "unparseable"
    names = (pair[0]["yes_sub_title"], pair[1]["yes_sub_title"])

    try:
        matched = match_fight(db, names[0], names[1], fight_date)
    except NoSuchCard:
        return "no_card"
    if not matched:
        log.info(f"  unmatched: {names[0]} vs {names[1]} ({fight_date}) [{event_ticker}]")
        return "unmatched"
    fight, is_swapped = matched

    if dry_run:
        return "matched"

    for idx, market in enumerate(pair):
        # pair[0] is names[0]. is_swapped means the venue listed our blue corner first.
        first_is_red = not is_swapped
        side = ("red" if first_is_red else "blue") if idx == 0 else ("blue" if first_is_red else "red")

        historical = _is_historical(market, cutoff)
        result = (market.get("result") or "").lower()
        status = "settled" if market.get("status") in ("finalized", "settled") else "open"

        row = upsert_market(
            db,
            platform="kalshi",
            external_event_id=event_ticker,
            external_market_id=market["ticker"],
            market_type="moneyline",
            outcome_key=side,
            outcome_label=market.get("yes_sub_title"),
            side=side,
            fight_id=fight.id,
            status=status,
            resolved_outcome=_RESULT_TO_OUTCOME.get(result),
            venue_closed_at=closed_at(market),
        )

        upsert_quote(
            db, row.id,
            price=_f(market.get("last_price_dollars")) or 0.0,
            bid=_f(market.get("yes_bid_dollars")),
            ask=_f(market.get("yes_ask_dollars")),
            volume=_f(market.get("volume_fp")),
            open_interest=_f(market.get("open_interest_fp")),
        )

        _pull_curve(db, row, market, fight_date, historical=historical, fine_grained=fine_grained)

    db.commit()
    if touched is not None:
        touched.add(fight.id)
    return "ok"


def _pull_curve(db, row: UFCPredictionMarket, market: dict, fight_date: date, *,
                historical: bool, fine_grained: bool) -> None:
    """Append the venue's own candles for the window we are missing.

    Asking only for what is missing keeps a refresh cheap, but the request deliberately overlaps
    the last stored point by one period: the most recent candle was still forming when it was
    read, so re-requesting it is how its final value lands. ON CONFLICT DO NOTHING makes the
    overlap free.
    """
    open_ts = market.get("open_time")
    start = (
        datetime.fromisoformat(open_ts.replace("Z", "+00:00"))
        if open_ts else datetime.now(timezone.utc) - timedelta(days=60)
    )

    last = last_history_ts(db, row.id)
    interval = 60
    if last is not None:
        start = max(start, last.replace(tzinfo=timezone.utc) - timedelta(hours=1))

    # In the last day before a fight, hourly candles are too coarse for closing-line work --
    # that window is where the price actually moves and where CLV is decided. One extra request
    # per market buys minute resolution for it.
    if fine_grained:
        fight_dt = datetime.combine(fight_date, datetime.min.time(), tzinfo=timezone.utc)
        if abs((fight_dt - datetime.now(timezone.utc)).total_seconds()) < 86400 * 2:
            interval = 1
            start = max(start, fight_dt - timedelta(days=1))

    end = datetime.now(timezone.utc)
    if start >= end:
        return

    candles = fetch_candles(
        market["ticker"], int(start.timestamp()), int(end.timestamp()),
        interval=interval, historical=historical,
    )

    rows = []
    for c in candles:
        ts = c.get("end_period_ts")
        if not ts:
            continue
        price = c.get("price") or {}
        bid = _pick(c.get("yes_bid") or {}, "close_dollars", "close")
        ask = _pick(c.get("yes_ask") or {}, "close_dollars", "close")

        # An untraded period reports a null or zero close; fall back to the period mean, then to
        # the quoted midpoint, so quiet markets still produce a curve rather than nothing. On a
        # binary contract the midpoint of a live two-sided quote is a perfectly good price.
        p = _pick(price, "close_dollars", "close") or _pick(price, "mean_dollars", "mean")
        if not p:
            p = (bid + ask) / 2 if bid and ask else None
        if not p:
            continue

        captured = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        rows.append({
            "market_id": row.id,
            "price": p,
            "bid": bid,
            "ask": ask,
            "volume": _pick(c, "volume_fp", "volume"),
            "open_interest": _pick(c, "open_interest_fp", "open_interest"),
            "captured_at": captured,
            "days_to_fight": days_to_fight(captured, bout_reference(market) or fight_date),
        })

    if rows:
        record_history(db, rows)


def _run(events: list[dict], *, fine_grained: bool, dry_run: bool, label: str) -> dict:
    db = open_session()
    cutoff = _cutoff()
    stats: dict[str, int] = {}
    touched: set[int] = set()
    try:
        for i, event in enumerate(events, 1):
            try:
                status = _ingest_event(db, event, cutoff, fine_grained=fine_grained, dry_run=dry_run,
                                       touched=touched)
            except Exception as e:
                db.rollback()
                log.warning(f"  {event.get('event_ticker')} failed: {e.__class__.__name__}: {e}")
                status = "error"
            stats[status] = stats.get(status, 0) + 1
            if i % 50 == 0:
                log.info(f"  [{label}] {i}/{len(events)} events — {stats}")
        # Once per run, not per fight: a bout's cutoff depends on its neighbours on the card.
        realigned = realign_cards(db, touched)
        db.commit()
    finally:
        db.close()
    log.info(f"Kalshi {label} done: {stats} ({realigned} curve points realigned)")
    return stats


#: How far back the live run re-reads markets that have closed. They have to be read once more
#: after the result: that is the only time the real close (the bout's end) and the settled status
#: are visible. Without it a card's markets stay "open" forever, still quoting the post-result
#: price, and no bout on the card ever gets a cutoff better than the fallback.
RECENT_CLOSED_DAYS = 4


def run_kalshi_live() -> dict:
    """Refresh open markets plus those that closed in the last few days.

    `status=open` is what keeps this affordable on a two-hourly schedule -- 18 events rather than
    the 689 in the series. Older settled fights have nothing left to learn; backfilling them is
    the separate `--backfill` path.
    """
    since = date.today() - timedelta(days=RECENT_CLOSED_DAYS)
    events = list_fight_events(status="open") + [
        e for status in ("closed", "settled") for e in list_fight_events(status=status)
        if _ticker_date(e.get("event_ticker", "")) >= since
    ]
    events = [e for e in events if not e.get("event_ticker", "").startswith("_")]
    log.info(f"Kalshi live: {len(events)} open or recently closed events")
    return _run(events, fine_grained=True, dry_run=False, label="live")


def run_kalshi_backfill(since: str | None = None, dry_run: bool = False) -> dict:
    """Walk the whole series, including settled fights, and build history from candles."""
    events = list_fight_events()
    if since:
        cutoff_date = date.fromisoformat(since)
        events = [e for e in events if _ticker_date(e.get("event_ticker", "")) >= cutoff_date]
    log.info(f"Kalshi backfill: {len(events)} events{' (dry run)' if dry_run else ''}")
    return _run(events, fine_grained=False, dry_run=dry_run, label="backfill")


def _ticker_date(ticker: str) -> date:
    """Parse the date out of KXUFCFIGHT-26SEP19VANPAN. Returns date.min when unparseable."""
    try:
        part = ticker.split("-")[1]
        return datetime.strptime(part[:7], "%y%b%d").date()
    except (IndexError, ValueError):
        return date.min


def backfill_closed_at() -> dict:
    """One-off repair: record the real close on every stored Kalshi market, then realign.

    Markets ingested before `venue_closed_at` existed have no close recorded, and the history
    writer is ON CONFLICT DO NOTHING, so no re-run of the backfill would revise their curves.
    """
    db = open_session()
    stats = {"events": 0, "closed": 0, "realigned": 0}
    try:
        rows = (
            db.query(UFCPredictionMarket)
            .filter(UFCPredictionMarket.platform == "kalshi",
                    UFCPredictionMarket.fight_id.isnot(None))
            .all()
        )
        by_event: dict[str, list] = {}
        for row in rows:
            by_event.setdefault(row.external_event_id, []).append(row)

        for i, (event_ticker, markets) in enumerate(by_event.items(), 1):
            live = {m["ticker"]: m for m in fetch_markets(event_ticker)}
            if not live:
                live = {m["ticker"]: m for m in fetch_markets(event_ticker, historical=True)}
            stats["events"] += 1
            for row in markets:
                market = live.get(row.external_market_id)
                close = closed_at(market) if market else None
                if close is None:
                    continue
                row.venue_closed_at = close
                if market.get("status") in ("finalized", "settled"):
                    row.status = "settled"
                    outcome = _RESULT_TO_OUTCOME.get((market.get("result") or "").lower())
                    if outcome is not None:
                        row.resolved_outcome = outcome
                stats["closed"] += 1
            if i % 50 == 0:
                db.commit()
                log.info(f"  closed_at {i}/{len(by_event)} events — {stats}")
        db.commit()
        stats["realigned"] = realign_cards(db, {r.fight_id for r in rows})
        db.commit()
    finally:
        db.close()
    log.info(f"Kalshi close backfill done: {stats}")
    return stats
