"""Read-side helpers: turn stored prediction-market rows into API payloads.

Kept out of the router so the shaping rules -- above all how a pair of exchange prices becomes a
probability -- live in one place and can be unit tested.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from app.models.ufc import (
    UFCPredictionMarket, UFCPredictionMarketHistory, UFCPredictionMarketQuote,
)

PLATFORMS = ("kalshi", "polymarket")

#: How far a price must sit from Polymarket's 0.50 seed to count as a real quote. Observed
#: untouched props span 0.49-0.50 (deviation <= 0.01); the thinnest genuinely traded market seen
#: sits at 0.045 out. A market that has truly traded to a coin flip is caught by the volume test.
_SEED_EPSILON = 0.02


def normalise_pair(red_price: float | None, blue_price: float | None) -> tuple[float | None, float | None]:
    """Turn the two sides' traded prices into probabilities that sum to 1.

    On an exchange the pair already sums to roughly 1, so this is a much smaller correction than
    devigging a sportsbook line -- the residual is the bid/ask spread and a little staleness, not
    a deliberate margin. Normalising anyway keeps the output directly comparable to the devigged
    `red_implied_prob` served alongside it.

    Returns (None, None) when only one side is priced: halving the known side would invent a
    number for the other, and a missing probability is more honest than a fabricated one.
    """
    if red_price is None or blue_price is None:
        return None, None
    total = red_price + blue_price
    if total <= 0:
        return None, None
    return red_price / total, blue_price / total


def _quote_dict(market: UFCPredictionMarket, quote: UFCPredictionMarketQuote | None,
                closing: float | None = None) -> dict | None:
    if quote is None:
        return None
    spread = (
        round(quote.ask - quote.bid, 4)
        if quote.ask is not None and quote.bid is not None else None
    )
    # On a settled market the stored quote is post-result; the closing line from the curve is the
    # number that can be compared to anything. Falls back to the quote when no curve point
    # predates the fight, which is the case for a market that only opened after our last refresh.
    price = closing if (market.status == "settled" and closing is not None) else quote.price
    # A settled market with no curve point before the fight has nothing left but its settlement
    # value, which is the *result* dressed as a price -- 0.0 or 1.0. That is not a market view and
    # must never be rendered or averaged as one: a completed fight would otherwise show
    # "Polymarket: 0%" as though the market had been certain.
    #
    # This is not an edge case for Polymarket. Its /prices-history serves only about the last five
    # weeks; every market older than that returns an empty series, so historical fights have a
    # settlement value and nothing else. Kalshi's candlesticks go back the full series, which is
    # why the historical curve coverage rests on Kalshi.
    is_settlement = market.status == "settled" and closing is None
    return {
        "price": price,
        "is_closing_line": market.status == "settled" and closing is not None,
        "is_settlement": is_settlement,
        "last_price": quote.price,
        # Whether this market carries an actual opinion. Polymarket seeds a new prop at 0.50 with
        # no volume and a few dollars of liquidity, so an untouched market quotes a
        # confident-looking "50%" that is not a view about anything -- on an upcoming title fight
        # every prop read 0.49-0.50 while the moneyline had $8.7k behind it. Rendering that beside
        # a real model probability would invent a market consensus that does not exist.
        #
        # Neither obvious test works alone:
        #   * Volume is absent on some genuinely traded markets (a settled prop priced at 0.37
        #     reports no volume at all).
        #   * Presence of a price curve proves nothing, because /prices-history happily returns a
        #     full series for an untouched market -- flat at the seed. The untraded props above
        #     had 188 points and 1-4 distinct prices; traded ones had 10-20.
        # So: real volume, or a price that has actually left the seed.
        # A settlement value is never a tradeable opinion, however much volume the market saw.
        "traded": (not is_settlement) and (bool(quote.volume) or abs(price - 0.5) > _SEED_EPSILON),
        "bid": quote.bid,
        "ask": quote.ask,
        # On an exchange the spread, not vig, is what it actually costs to take the price.
        "spread": spread,
        "volume": quote.volume,
        "open_interest": quote.open_interest,
        "liquidity": quote.liquidity,
        "outcome_label": market.outcome_label,
        "captured_at": quote.captured_at.isoformat() if quote.captured_at else None,
    }


def _rows_for(db: Session, fight_id: int):
    return (
        db.query(UFCPredictionMarket, UFCPredictionMarketQuote)
        .outerjoin(
            UFCPredictionMarketQuote,
            UFCPredictionMarketQuote.market_id == UFCPredictionMarket.id,
        )
        .filter(UFCPredictionMarket.fight_id == fight_id)
        .all()
    )


def _closing_prices(db: Session, market_ids: list[int]) -> dict[int, float]:
    """Last traded price at or before the fight, per market.

    Needed because a settled market's stored quote is *not* a closing line on either venue.
    Kalshi keeps trading for a few minutes after the result is known, so its
    `last_price_dollars` drifts to the outcome -- one settled pair here reads 0.99 / 0.42, which
    sums to 1.41 and means nothing as a probability. Polymarket reports settled outcomes as
    exactly 0 or 1, which sums correctly and is equally useless: it is the result, not a price.

    The real closing line is in the curve, which both venues publish honestly. `days_to_fight`
    is denormalised precisely so it can be found with a comparison instead of a join back to the
    event date.
    """
    if not market_ids:
        return {}
    rows = (
        db.query(UFCPredictionMarketHistory)
        .filter(
            UFCPredictionMarketHistory.market_id.in_(market_ids),
            UFCPredictionMarketHistory.days_to_fight >= 0,
        )
        .order_by(UFCPredictionMarketHistory.captured_at)
        .all()
    )
    # Ordered ascending, so the last write per market wins.
    return {r.market_id: r.price for r in rows}


def fight_payload(db: Session, fight_id: int) -> dict:
    """Everything both venues currently say about one fight, grouped by platform."""
    out: dict[str, dict] = {}
    rows = _rows_for(db, fight_id)
    closing = _closing_prices(
        db, [m.id for m, _ in rows if m.status == "settled"]
    )
    for market, quote in rows:
        platform = out.setdefault(market.platform, {
            "moneyline": None, "method": {}, "rounds": {},
            "fighter_props": {"red": {}, "blue": {}}, "other": {},
        })
        payload = _quote_dict(market, quote, closing.get(market.id))
        if payload is None:
            continue

        if market.market_type == "moneyline":
            ml = platform["moneyline"] or {}
            ml[f"{market.outcome_key}"] = payload
            platform["moneyline"] = ml
        elif market.market_type == "method":
            platform["method"][market.outcome_key] = payload
        elif market.market_type == "round_ou":
            platform["rounds"][market.outcome_key] = payload
        elif market.market_type == "distance":
            platform["method"]["distance"] = payload
        elif market.market_type in ("fighter_method", "fighter_round"):
            side = market.side
            if side in ("red", "blue"):
                # Drop the corner prefix: the corner is already the enclosing key.
                key = market.outcome_key.removeprefix(f"{side}_")
                platform["fighter_props"][side][key] = payload
        else:
            platform["other"][market.outcome_key] = payload

    # Collapse each moneyline into the normalised probabilities the UI actually compares against.
    for platform in out.values():
        ml = platform.get("moneyline")
        if not ml:
            continue
        red, blue = ml.get("red"), ml.get("blue")
        # Settlement values would normalise to a clean 0/1 and read as a market that was certain.
        usable = lambda q: q is not None and not q["is_settlement"]
        red_prob, blue_prob = normalise_pair(
            red["price"] if usable(red) else None,
            blue["price"] if usable(blue) else None,
        )
        platform["moneyline"] = {
            "red": red, "blue": blue,
            "red_prob": red_prob, "blue_prob": blue_prob,
            "volume": sum(q["volume"] or 0 for q in (red, blue) if q) or None,
        }

    return out


def consensus_prob(db: Session, fight_id: int) -> dict | None:
    """One volume-weighted red-corner probability across both venues.

    Weighted by traded volume because the venues differ by an order of magnitude in liquidity on
    any given fight, and an unweighted mean would let a market that traded $200 move the number as
    much as one that traded $2M. Falls back to an equal weighting when no volume is reported,
    which is better than returning nothing.
    """
    payload = fight_payload(db, fight_id)
    weighted, total_w, plain = 0.0, 0.0, []

    for platform in payload.values():
        ml = platform.get("moneyline") or {}
        prob = ml.get("red_prob")
        if prob is None:
            continue
        plain.append(prob)
        vol = ml.get("volume") or 0
        if vol > 0:
            weighted += prob * vol
            total_w += vol

    if total_w > 0:
        return {"red_prob": weighted / total_w, "venues": len(plain), "weighted_by": "volume"}
    if plain:
        return {"red_prob": sum(plain) / len(plain), "venues": len(plain), "weighted_by": "equal"}
    return None


def best_raw_prices(db: Session, fight_id: int) -> dict | None:
    """Cheapest *raw* moneyline price for each corner, across venues, with where it came from.

    Deliberately not the normalised consensus. Normalising forces red + blue to exactly 1, which
    erases precisely the deviation an arbitrage consists of -- comparing two normalised exchange
    quotes always yields a margin of zero no matter how the underlying prices sit. Arbitrage has
    to be computed on the prices actually payable.

    "Cheapest" because each side is bought independently: the arb is whether the two cheapest
    routes to full coverage cost less than the payout.
    """
    best: dict[str, tuple[float, str]] = {}
    for market, quote in _rows_for(db, fight_id):
        if market.market_type != "moneyline" or quote is None or market.side not in ("red", "blue"):
            continue
        if not quote.price or quote.price <= 0:
            continue
        current = best.get(market.side)
        if current is None or quote.price < current[0]:
            best[market.side] = (quote.price, market.platform)

    if "red" not in best or "blue" not in best:
        return None
    return {
        "red_price": best["red"][0], "red_venue": best["red"][1],
        "blue_price": best["blue"][0], "blue_venue": best["blue"][1],
    }


def market_history(db: Session, fight_id: int, market_type: str = "moneyline") -> dict:
    """Per-platform price curves, aligned to the red corner.

    Both corners' series are folded into a single red-corner probability per timestamp, because a
    chart of two complementary lines conveys nothing the one line does not.

    Timestamps are bucketed to the minute rather than matched exactly. Polymarket samples a
    market's two outcome tokens a few seconds apart -- 23:00:14 against 23:00:17 -- so the two
    sides of the same market share *no* exact timestamps at all, and exact matching silently
    dropped every Polymarket point from the chart while leaving Kalshi (whose candles land on
    exact period boundaries) looking fine.
    """
    markets = (
        db.query(UFCPredictionMarket)
        .filter(
            UFCPredictionMarket.fight_id == fight_id,
            UFCPredictionMarket.market_type == market_type,
        )
        .all()
    )
    if not markets:
        return {}

    by_id = {m.id: m for m in markets}
    rows = (
        db.query(UFCPredictionMarketHistory)
        .filter(UFCPredictionMarketHistory.market_id.in_(list(by_id)))
        .order_by(UFCPredictionMarketHistory.captured_at)
        .all()
    )

    # platform -> timestamp -> {side: price}
    series: dict[str, dict] = {}
    for row in rows:
        market = by_id[row.market_id]
        # Truncate to the minute so the two sides of a Polymarket market land in the same bucket.
        minute = row.captured_at.replace(second=0, microsecond=0)
        ts = minute.isoformat()
        bucket = series.setdefault(market.platform, {}).setdefault(
            ts, {"t": ts, "days_to_fight": row.days_to_fight}
        )
        bucket[market.side or market.outcome_key] = row.price

    # One reference time for the whole fight, so the venues' curves line up.
    #
    # Each platform's stored `days_to_fight` is measured against its own idea of when the bout is:
    # Kalshi against `occurrence_datetime`, Polymarket against midnight on the event's date. Those
    # differ by hours, so plotting each series against its own value puts the same wall-clock
    # moment at two different x positions -- Kalshi's last point read 6.31 days out and
    # Polymarket's 5.00, though both were captured minutes apart. The later anchor is taken
    # because it is the one derived from a real event time rather than a date rounded down.
    anchors = [
        dt.datetime.fromisoformat(b["t"]) + dt.timedelta(days=b["days_to_fight"])
        for buckets in series.values() for b in buckets.values()
        if b.get("days_to_fight") is not None
    ]
    anchor = max(anchors) if anchors else None

    out: dict[str, list] = {}
    for platform, buckets in series.items():
        points = []
        for ts in sorted(buckets):
            b = buckets[ts]
            red_prob, blue_prob = normalise_pair(b.get("red"), b.get("blue"))
            if red_prob is None:
                continue
            captured = dt.datetime.fromisoformat(b["t"])
            d2f = ((anchor - captured).total_seconds() / 86400.0
                   if anchor is not None else b["days_to_fight"])
            points.append({
                "t": b["t"],
                "days_to_fight": round(d2f, 4) if d2f is not None else None,
                "red_prob": round(red_prob, 4),
                "blue_prob": round(blue_prob, 4),
            })
        if points:
            out[platform] = points
    return out


def consensus_probs_bulk(db: Session, fight_ids: list[int]) -> dict[int, dict]:
    """`consensus_prob` for many fights in one pass.

    The list endpoints render a whole card at a time, and calling the per-fight version in a loop
    turns one page load into several dozen round trips against a hosted database.

    Settled markets are skipped rather than repaired here: recovering a closing line needs the
    curve, and the endpoints this serves are showing *upcoming* fights, where the live quote is
    already the right number.
    """
    if not fight_ids:
        return {}

    rows = (
        db.query(UFCPredictionMarket, UFCPredictionMarketQuote)
        .join(UFCPredictionMarketQuote,
              UFCPredictionMarketQuote.market_id == UFCPredictionMarket.id)
        .filter(
            UFCPredictionMarket.fight_id.in_(fight_ids),
            UFCPredictionMarket.market_type == "moneyline",
            UFCPredictionMarket.status != "settled",
        )
        .all()
    )

    # (fight, platform) -> {side: (price, volume)}
    grouped: dict[tuple[int, str], dict] = {}
    for market, quote in rows:
        if market.side not in ("red", "blue") or quote.price is None:
            continue
        grouped.setdefault((market.fight_id, market.platform), {})[market.side] = (
            quote.price, quote.volume or 0.0
        )

    per_fight: dict[int, list] = {}
    for (fight_id, _platform), sides in grouped.items():
        if "red" not in sides or "blue" not in sides:
            continue
        red_prob, _ = normalise_pair(sides["red"][0], sides["blue"][0])
        if red_prob is None:
            continue
        per_fight.setdefault(fight_id, []).append((red_prob, sides["red"][1] + sides["blue"][1]))

    out: dict[int, dict] = {}
    for fight_id, entries in per_fight.items():
        total_w = sum(w for _, w in entries)
        if total_w > 0:
            prob = sum(p * w for p, w in entries) / total_w
            weighted_by = "volume"
        else:
            prob = sum(p for p, _ in entries) / len(entries)
            weighted_by = "equal"
        out[fight_id] = {
            "red_prob": prob, "venues": len(entries),
            "weighted_by": weighted_by, "volume": total_w or None,
        }
    return out
