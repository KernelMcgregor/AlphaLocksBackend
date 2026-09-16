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
    NoSuchCard, align_fight_anchor, days_to_fight, fetch_json, last_history_ts, match_fight, open_session,
    record_history, upsert_market, upsert_quote,
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


#: Subtracted from a settled market's `close_time` to estimate when its bout began. Kalshi closes
#: a UFC market minutes after the result, so close_time is bout *end*; 30 minutes clears a
#: full three-round fight plus walkouts and the settlement delay. Erring early is deliberate --
#: a closing line taken slightly too soon is merely a little stale, while one taken slightly too
#: late is the market reading the result off the screen.
_BOUT_LEAD = timedelta(minutes=30)


def bout_reference(market: dict) -> datetime | date | None:
    """When this specific bout started, as well as Kalshi will say.

    `occurrence_datetime` is the *event's* time and is identical across every bout on the card, so
    it cannot be used to place a prelim -- see `days_to_fight`. A settled market's `close_time`
    is per-bout and is the only bout-specific timestamp Kalshi publishes.
    """
    close = market.get("close_time")
    status = market.get("status")
    if close and status in ("finalized", "settled", "closed"):
        return datetime.fromisoformat(close.replace("Z", "+00:00")) - _BOUT_LEAD
    occ = market.get("occurrence_datetime")
    if occ:
        return datetime.fromisoformat(occ.replace("Z", "+00:00"))
    return None


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
                  dry_run: bool = False) -> str:
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


    # Put both venues on one time reference. Kalshi anchors on a bout-specific close_time
    # and Polymarket on midnight of the event date, which for UFC 331 differ by 31 hours —
    # so without this, `days_to_fight` means something different depending on who wrote the
    # row, and any CLV or line-movement query silently mixes the two. No-op once aligned.
    align_fight_anchor(db, fight.id)
    db.commit()
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
    try:
        for i, event in enumerate(events, 1):
            try:
                status = _ingest_event(db, event, cutoff, fine_grained=fine_grained, dry_run=dry_run)
            except Exception as e:
                db.rollback()
                log.warning(f"  {event.get('event_ticker')} failed: {e.__class__.__name__}: {e}")
                status = "error"
            stats[status] = stats.get(status, 0) + 1
            if i % 50 == 0:
                log.info(f"  [{label}] {i}/{len(events)} events — {stats}")
    finally:
        db.close()
    log.info(f"Kalshi {label} done: {stats}")
    return stats


def run_kalshi_live() -> dict:
    """Refresh open markets: discover new fights, update quotes, extend curves.

    `status=open` is what keeps this affordable on a two-hourly schedule -- 18 events rather than
    the 689 in the series. Settled fights have nothing left to learn: their curve is complete and
    their quote is fixed, so re-walking them each pass would cost ~35 minutes to rewrite data that
    cannot change. Backfilling them is the separate `--backfill` path.
    """
    events = [e for e in list_fight_events(status="open")
              if not e.get("event_ticker", "").startswith("_")]
    log.info(f"Kalshi live: {len(events)} open events")
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


def recompute_days_to_fight() -> dict:
    """Rewrite `days_to_fight` on stored Kalshi curve points against each bout's own start time.

    Needed as a one-off repair, and useful whenever a bout's timing is corrected upstream. The
    normal history write is ON CONFLICT DO NOTHING -- that is what makes re-requesting a window
    free -- so a re-run of the backfill will never revise a column on rows that already exist.

    Only the reference changes; prices are untouched.
    """
    from sqlalchemy import func, literal, update

    from app.models.ufc import UFCPredictionMarketHistory as History

    db = open_session()
    stats = {"markets": 0, "updated": 0, "skipped": 0}
    try:
        rows = (
            db.query(UFCPredictionMarket)
            .filter(UFCPredictionMarket.platform == "kalshi")
            .order_by(UFCPredictionMarket.external_event_id)
            .all()
        )
        by_event: dict[str, list] = {}
        for row in rows:
            by_event.setdefault(row.external_event_id, []).append(row)

        for i, (event_ticker, markets) in enumerate(by_event.items(), 1):
            live = {m["ticker"]: m for m in fetch_markets(event_ticker)}
            if not live:
                live = {m["ticker"]: m for m in fetch_markets(event_ticker, historical=True)}

            for row in markets:
                market = live.get(row.external_market_id)
                ref = bout_reference(market) if market else None
                if ref is None:
                    stats["skipped"] += 1
                    continue
                stats["markets"] += 1
                # One statement per market, not per point. The arithmetic is pushed into SQL
                # because a market can carry thousands of candles and issuing an UPDATE per row
                # over a hosted database turns a two-minute repair into an hours-long one.
                ref_naive = ref.replace(tzinfo=None) if isinstance(ref, datetime) else ref
                result = db.execute(
                    update(History)
                    .where(History.market_id == row.id)
                    .values(days_to_fight=(
                        func.extract("epoch", literal(ref_naive) - History.captured_at) / 86400.0
                    ))
                )
                stats["updated"] += result.rowcount or 0
            db.commit()
            if i % 50 == 0:
                log.info(f"  recompute {i}/{len(by_event)} events — {stats}")
    finally:
        db.close()
    log.info(f"Recompute done: {stats}")
    return stats
