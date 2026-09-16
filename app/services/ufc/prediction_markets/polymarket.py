"""Polymarket UFC ingestion: moneyline plus method, round and distance props.

Polymarket lists one event per fight, slugged `ufc-<f1>-<f2>-<YYYY-MM-DD>`, holding ~19 markets.
That prop coverage -- per-fighter method of victory, win-in-round-N, round totals, goes-the-
distance -- is the reason to carry this venue alongside Kalshi, which is moneyline only. It maps
onto `UFCMethodPrediction` far more richly than the single Bovada book does.

Two operational facts, both measured rather than taken from docs:

* Gamma returns 403 without a `User-Agent`. That is a UA check, not a rate limit.
* Gamma and CLOB both served ~14 rps with zero 429s, so the published "60 requests/minute"
  figure is wrong and prop curves are affordable at every refresh.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone

from app.services.ufc.prediction_markets.common import (
    NoSuchCard, _fold_match, align_fight_anchor, days_to_fight, name_matches, fetch_json, last_history_ts, match_fight,
    open_session, record_history, split_versus, upsert_market, upsert_quote,
)

log = logging.getLogger("prediction_markets.polymarket")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
HOST = "polymarket"

#: Per-fight events only. The `ufc` tag also carries futures ('Who will Conor McGregor fight
#: next?', 'Who will become a UFC champion in 2026?'), which have no fight row to attach to.
FIGHT_SLUG = re.compile(r"^ufc-.+-(\d{4}-\d{2}-\d{2})$")
#: Same slug, split into its two fighter tokens: ufc-<tok_a>-<tok_b>-<date>.
FIGHT_SLUG_PARTS = re.compile(r"^ufc-([a-z]+\d*)-([a-z]+\d*)-\d{4}-\d{2}-\d{2}$")

_ROUND_OU = re.compile(r"^O/U\s+([\d.]+)\s+Rounds?", re.I)
_WIN_IN_ROUND = re.compile(r"^Will\s+(.+?)\s+win in Round\s+(\d+)", re.I)
_WIN_BY_METHOD = re.compile(r"^Will\s+(.+?)\s+win by\s+(KO or TKO|submission|decision)", re.I)
_END_BEFORE_ROUND = re.compile(r"^Will the fight end before Round\s+(\d+)", re.I)

_METHOD_KEYS = {"ko or tko": "ko_tko", "submission": "submission", "decision": "decision"}


def classify_market(question: str, red=None, blue=None):
    """Map a Polymarket question to (market_type, outcome_key, side).

    Ordered most-specific first: 'Will <fighter> win by KO or TKO?' must not be swallowed by the
    fight-level 'Will the fight be won by KO or TKO?' rule.

    `red` and `blue` are the *matched* fighters, not the venue's billing strings, and the corner
    is resolved with the same surname-tolerant matcher used to find the fight. Comparing against
    the billing names instead leaves per-fighter props unattributed whenever the event title is
    billed by surname -- 'Yakhyaev vs Cerqueira' against a question naming 'Abdul-Rakhman
    Yakhyaev'. That silently stranded 132 of 946 fighter-method markets with no corner, which
    makes them unusable without failing anywhere visible.

    Unrecognised questions return `market_type='unknown'` rather than None. Polymarket adds prop
    types over time, and a dropped market is invisible while an 'unknown' row with its original
    `outcome_label` is a thing you can go look at.
    """
    q = (question or "").strip()
    ql = q.lower()

    def side_for(name: str) -> str | None:
        """Which corner a named fighter belongs to."""
        for corner, fighter in (("red", red), ("blue", blue)):
            if fighter is not None and name_matches(name, fighter.first_name, fighter.last_name):
                return corner
        return None

    # Moneyline: the market whose question restates the matchup itself.
    if split_versus(q) is not None:
        return "moneyline", None, None

    m = _WIN_BY_METHOD.match(q)
    if m:
        side = side_for(m.group(1))
        key = _METHOD_KEYS[m.group(2).lower()]
        return "fighter_method", (f"{side}_{key}" if side else key), side

    m = _WIN_IN_ROUND.match(q)
    if m:
        side = side_for(m.group(1))
        rnd = m.group(2)
        return "fighter_round", (f"{side}_r{rnd}" if side else f"r{rnd}"), side

    m = _ROUND_OU.match(q)
    if m:
        return "round_ou", f"ou_{m.group(1)}", None

    m = _END_BEFORE_ROUND.match(q)
    if m:
        return "round_ou", f"before_r{m.group(1)}", None

    if "won by ko or tko" in ql:
        return "method", "ko_tko", None
    if "won by submission" in ql:
        return "method", "submission", None
    # Checked before plain decision: these are two separate markets, and collapsing them onto one
    # outcome_key would leave two rows answering to the same name with no way to tell which is
    # the draw-inclusive one.
    if "decision or draw" in ql:
        return "method", "decision_draw", None
    if "won by decision" in ql:
        return "method", "decision", None
    if "go the distance" in ql or "go to the distance" in ql:
        return "distance", "distance", None

    return "unknown", None, None


def list_fight_events(closed: bool | None = None) -> list[dict]:
    """Per-fight UFC events, paginated. Gamma caps `limit` at 100."""
    events, offset = [], 0
    while True:
        params = {"tag_slug": "ufc", "limit": 100, "offset": offset,
                  "order": "startDate", "ascending": "false"}
        if closed is not None:
            params["closed"] = str(closed).lower()
        batch = fetch_json(HOST, f"{GAMMA}/events", params)
        if not batch:
            break
        events.extend(batch)
        offset += 100
        if len(batch) < 100:
            break
    return [e for e in events if FIGHT_SLUG.match(e.get("slug") or "")]


def fetch_price_history(token_id: str, interval: str = "max", fidelity: int = 60) -> list[dict]:
    data = fetch_json(HOST, f"{CLOB}/prices-history",
                      {"market": token_id, "interval": interval, "fidelity": fidelity})
    return (data or {}).get("history") or []


def _json_list(raw) -> list:
    """`clobTokenIds` and `outcomePrices` arrive as JSON-encoded strings, not arrays."""
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


def _slug_date(slug: str) -> date | None:
    m = FIGHT_SLUG.match(slug or "")
    return date.fromisoformat(m.group(1)) if m else None


def _slug_hints(slug: str) -> tuple[str | None, str | None]:
    """First-name prefixes recovered from the slug, in the same order as the title.

    Polymarket slugs abbreviate *first* names where titles often carry only surnames:
    `ufc-isl-jac9-2025-11-15` is Islam Makhachev vs Jack Della Maddalena, billed as 'Makhachev vs
    Della Maddalena'. That makes the slug the only thing distinguishing, say, Aleksandre Topuria
    (`ale58`) from Ilia. The trailing digits are Polymarket's own uniqueness counter and carry no
    name information, so they are stripped.

    Used strictly as a tie-break in `match_fight`, never as a rejection on its own: the tokens are
    not always first-name-derived, and a hint that fails to match should cost nothing when there
    is only one candidate anyway.
    """
    m = FIGHT_SLUG_PARTS.match(slug or "")
    if not m:
        return None, None
    return (m.group(1).rstrip("0123456789") or None,
            m.group(2).rstrip("0123456789") or None)


def _num(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ingest_event(db, event: dict, *, curves: str, dry_run: bool) -> str:
    slug = event.get("slug") or ""
    fight_date = _slug_date(slug)
    names = split_versus(event.get("title") or "")
    if not fight_date or not names:
        return "unparseable"

    try:
        matched = match_fight(db, names[0], names[1], fight_date, hints=_slug_hints(slug))
    except NoSuchCard:
        return "no_card"
    if not matched:
        log.info(f"  unmatched: {names[0]} vs {names[1]} ({fight_date}) [{slug}]")
        return "unmatched"
    fight, is_swapped = matched
    if dry_run:
        return "matched"

    for market in event.get("markets") or []:
        tokens = _json_list(market.get("clobTokenIds"))
        prices = _json_list(market.get("outcomePrices"))
        outcomes = _json_list(market.get("outcomes"))
        if not tokens:
            continue

        market_type, outcome_key, side = classify_market(
            market.get("question") or "", fight.red_fighter, fight.blue_fighter
        )

        if market_type == "moneyline":
            # The two tokens are the two fighters, in the venue's listing order.
            token_sides = ["blue", "red"] if is_swapped else ["red", "blue"]
            for i, token in enumerate(tokens[:2]):
                _store(db, fight, slug, market, token, market_type,
                       outcome_key=token_sides[i], side=token_sides[i],
                       label=outcomes[i] if i < len(outcomes) else None,
                       price=_num(prices[i]) if i < len(prices) else None,
                       fight_date=fight_date, curves=curves)
        else:
            # Every other market is a Yes/No pair; only the Yes leg carries information, since
            # No is its complement. Storing both would double the row count and the request
            # count for nothing.
            if outcome_key is None:
                outcome_key = (market.get("question") or "")[:60]
            _store(db, fight, slug, market, tokens[0], market_type,
                   outcome_key=outcome_key, side=side,
                   label=market.get("question"),
                   price=_num(prices[0]) if prices else None,
                   fight_date=fight_date, curves=curves)


    # Put both venues on one time reference. Kalshi anchors on a bout-specific close_time
    # and Polymarket on midnight of the event date, which for UFC 331 differ by 31 hours —
    # so without this, `days_to_fight` means something different depending on who wrote the
    # row, and any CLV or line-movement query silently mixes the two. No-op once aligned.
    align_fight_anchor(db, fight.id)
    db.commit()
    return "ok"


def _store(db, fight, event_slug: str, market: dict, token: str, market_type: str, *,
           outcome_key: str, side: str | None, label: str | None, price: float | None,
           fight_date: date, curves: str) -> None:
    closed = bool(market.get("closed"))
    resolved = None
    if closed and price is not None:
        # A settled Polymarket outcome trades at exactly 0 or 1.
        resolved = 1.0 if price > 0.5 else 0.0

    row = upsert_market(
        db,
        platform="polymarket",
        external_event_id=event_slug,
        external_market_id=str(token),
        market_type=market_type,
        outcome_key=outcome_key,
        outcome_label=label,
        side=side,
        fight_id=fight.id,
        status="settled" if closed else "open",
        resolved_outcome=resolved,
    )

    if price is not None:
        upsert_quote(
            db, row.id, price=price,
            volume=_num(market.get("volume")),
            liquidity=_num(market.get("liquidity")),
        )

    if curves == "moneyline" and market_type != "moneyline":
        return

    # Skipping markets already fully captured keeps a refresh proportional to what changed
    # rather than to the size of the card.
    if closed and last_history_ts(db, row.id) is not None:
        return

    history = fetch_price_history(str(token))
    rows = []
    for point in history:
        ts, p = point.get("t"), point.get("p")
        if ts is None or p is None:
            continue
        captured = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        rows.append({
            "market_id": row.id,
            "price": float(p),
            "bid": None, "ask": None,
            "volume": None, "open_interest": None,
            "captured_at": captured,
            "days_to_fight": days_to_fight(captured, fight_date),
        })
    if rows:
        record_history(db, rows)


def _run(events: list[dict], *, curves: str, dry_run: bool, label: str) -> dict:
    db = open_session()
    stats: dict[str, int] = {}
    try:
        for i, event in enumerate(events, 1):
            try:
                status = _ingest_event(db, event, curves=curves, dry_run=dry_run)
            except Exception as e:
                db.rollback()
                log.warning(f"  {event.get('slug')} failed: {e.__class__.__name__}: {e}")
                status = "error"
            stats[status] = stats.get(status, 0) + 1
            if i % 25 == 0:
                log.info(f"  [{label}] {i}/{len(events)} events — {stats}")
    finally:
        db.close()
    log.info(f"Polymarket {label} done: {stats}")
    return stats


def run_polymarket_live(curves: str = "all") -> dict:
    events = list_fight_events(closed=False)
    log.info(f"Polymarket live: {len(events)} open per-fight events")
    return _run(events, curves=curves, dry_run=False, label="live")


def run_polymarket_backfill(since: str | None = None, curves: str = "all",
                            dry_run: bool = False) -> dict:
    events = list_fight_events()
    if since:
        cutoff = date.fromisoformat(since)
        events = [e for e in events if (_slug_date(e.get("slug") or "") or date.min) >= cutoff]
    log.info(f"Polymarket backfill: {len(events)} events{' (dry run)' if dry_run else ''}")
    return _run(events, curves=curves, dry_run=dry_run, label="backfill")
