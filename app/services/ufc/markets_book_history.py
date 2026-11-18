"""Per-book moneyline movement for one fight, the sharp-vs-soft split, and what happened around it.

Three stores carry sportsbook moneylines, with different reach:

  direct    ufc_book_market_history: read straight from FanDuel / BetRivers / Bovada / Pinnacle
            (from late Oct 2026), one row per side, written only when that side's price moves.
            The only source with Pinnacle, and Pinnacle's max stake.
  odds_api  ufc_fight_odds_history rows without a prefix (from Sep 2026), both sides per row.
  bfo       ufc_fight_odds_history 'BFO:<book>' rows: the line watcher's BestFightOdds reads.

A book that appears in more than one is one book, so its series are merged rather than drawn
twice: the best source wins wherever it has data, and a lesser source only fills the stretch
before the better one starts. Mixing sources inside the same window would draw a zigzag out of
nothing more than two scrapers sampling a few minutes apart.

Every point is cut at the bout's cutoff (prediction_markets.common.cutoffs_for_fights), the same
line every other pre-fight price on the site stops at. Books keep a market up in-play, and a
point after the bell is the book watching the fight, not pricing it.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.models.ufc import (
    UFCBookMarketHistory, UFCCancelledBout, UFCEvent, UFCFight, UFCFighter, UFCFightOddsHistory,
    UFCFightPrediction, UFCNewsMention,
)
from app.services.ufc.bfo_scraper import devig_pair
from app.services.ufc.prediction_markets.common import cutoffs_for_fights, fight_started

SHARP = "Pinnacle"
SOFT = ("FanDuel", "BetRivers", "Bovada")
SOURCE_RANK = {"direct": 0, "odds_api": 1, "bfo": 2}
#: Never a sportsbook: exchange and aggregate rows that can turn up under a BFO prefix.
_NOT_BOOKS = ("polymarket", "kalshi", "consensus", "mean")
#: A news day counts as a spike at this many headlines and this multiple of the fighter's usual.
SPIKE_MIN, SPIKE_RATIO = 5, 3.0
SPIKES_PER_FIGHTER = 2
#: How far back cancellations and news are looked for (matches short_notice.WINDOW_DAYS for cards).
CARD_WINDOW = timedelta(days=10)
NEWS_WINDOW = timedelta(days=45)


@dataclass(frozen=True)
class Quote:
    t: datetime
    red: int | None      # American; None = suspended
    blue: int | None
    max_stake: float | None = None


def book_name(raw: str) -> tuple[str, str]:
    """(book, source) for an ufc_fight_odds_history bookmaker label."""
    if raw.startswith("BFO:"):
        return raw[4:], "bfo"
    return raw, "odds_api"


def pair_sides(rows: list[tuple[datetime, str, int | None, float | None]]) -> list[Quote]:
    """Fold one book's per-side rows (t, side, american, max_stake) into two-sided quotes.

    The direct scrapers write a side only when it moves, so each side carries forward until it
    changes. A suspended side (american None) blanks the quote until it is priced again: half a
    market has no vig-free probability.
    """
    last: dict[str, int | None] = {}
    stake: float | None = None
    by_t: dict[datetime, Quote] = {}
    for t, side, american, max_stake in sorted(rows, key=lambda r: r[0]):
        if side not in ("red", "blue"):
            continue
        last[side] = american
        if max_stake is not None:
            stake = max_stake
        by_t[t] = Quote(t, last.get("red"), last.get("blue"), stake)
    return [q for q in by_t.values() if q.red is not None and q.blue is not None]


def merge_sources(by_source: dict[str, list[Quote]],
                  alive: dict[str, datetime | None] | None = None) -> tuple[list[Quote], list[str]]:
    """One series per book: the best-ranked source wherever it covers, lesser ones only outside.

    A source covers from its first point to the last time its scraper read that book at all
    (`alive`), not just to its own last point: every store writes on change, so a quiet stretch
    means the price held. Only once a scraper has stopped (the odds API in Nov 2026) does a lesser
    source take over after it.
    """
    alive = alive or {}
    out: list[Quote] = []
    used: list[str] = []
    covered: list[tuple[datetime, datetime]] = []
    for source in sorted(by_source, key=lambda s: SOURCE_RANK.get(s, 9)):
        quotes = sorted(by_source[source], key=lambda q: q.t)
        take = [q for q in quotes if not any(lo <= q.t <= hi for lo, hi in covered)]
        if not take:
            continue
        used.append(source)
        out.extend(take)
        end = max(t for t in (take[-1].t, alive.get(source)) if t is not None)
        covered.append((take[0].t, end))
    out.sort(key=lambda q: q.t)
    return out, used


def days_before(t: datetime, cutoff: datetime) -> float:
    return round((cutoff - t).total_seconds() / 86400.0, 4)


def book_points(quotes: list[Quote], cutoff: datetime) -> list[dict]:
    """Vig-free red-corner points strictly before the cutoff, dropping repeats of the same price."""
    pts: list[dict] = []
    prev = None
    for q in quotes:
        if q.t >= cutoff:
            break
        red, _ = devig_pair(q.red, q.blue)
        if red is None or (q.red, q.blue) == prev:
            continue
        prev = (q.red, q.blue)
        pts.append({"t": q.t.isoformat(), "days_to_fight": days_before(q.t, cutoff),
                    "red_prob": round(red, 4), "red_american": q.red, "blue_american": q.blue})
    return pts


def soft_median(series: dict[str, list[dict]]) -> list[dict]:
    """Median of the soft books' latest prices, re-evaluated whenever any of them moves.

    Each book's last price stands until it changes, so the median at any moment is over every
    soft book that has priced the fight so far -- not just the ones that happened to move.
    """
    events = sorted((p["t"], p["days_to_fight"], book, p["red_prob"])
                    for book, pts in series.items() for p in pts)
    latest: dict[str, float] = {}
    out: list[dict] = []
    for t, d, book, prob in events:
        latest[book] = prob
        point = {"t": t, "days_to_fight": d, "red_prob": round(statistics.median(latest.values()), 4),
                 "n_books": len(latest)}
        if out and out[-1]["t"] == t:
            out[-1] = point
        else:
            out.append(point)
    return out


def stake_steps(quotes: list[Quote], cutoff: datetime) -> list[dict]:
    """Pinnacle's max stake each time it changes, before the cutoff."""
    out: list[dict] = []
    for q in quotes:
        if q.t >= cutoff:
            break
        if q.max_stake is None or (out and out[-1]["max_stake"] == q.max_stake):
            continue
        out.append({"t": q.t.isoformat(), "days_to_fight": days_before(q.t, cutoff), "max_stake": q.max_stake})
    return out


def news_spikes(days: dict[date, int], window_start: date, window_end: date) -> list[tuple[date, int]]:
    """Days a fighter's headline count jumped: >= SPIKE_MIN and >= SPIKE_RATIO x their median day.

    Days with no row count as zero (the table only has rows for days a fighter was named), so the
    median is over the whole window, not just the busy days.
    """
    span = [(window_start + timedelta(days=i)) for i in range((window_end - window_start).days + 1)]
    if not span:
        return []
    base = statistics.median(days.get(d, 0) for d in span)
    hits = [(d, n) for d, n in days.items()
            if window_start <= d <= window_end and n >= SPIKE_MIN and n >= SPIKE_RATIO * max(base, 1)]
    hits.sort(key=lambda x: -x[1])
    return sorted(hits[:SPIKES_PER_FIGHTER])


def weigh_in_at(event_date: date, card_start: datetime | None) -> datetime:
    """Approximate official weigh-in time: the morning before, local to the card.

    The UFC weighs in ~33h before the first bout. With a published start that anchors it for
    cards in any time zone; otherwise 16:00 UTC the day before (Friday morning in the US).
    """
    if card_start is not None:
        return card_start - timedelta(hours=33)
    return datetime.combine(event_date - timedelta(days=1), datetime.min.time()) + timedelta(hours=16)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _alive(db: Session, by_book: dict[str, dict]) -> dict[str, dict[str, datetime | None]]:
    """Per book and source, the last time that scraper read the book for any fight."""
    from sqlalchemy import func

    multi = [b for b, s in by_book.items() if len(s) > 1]
    if not multi:
        return {}
    out: dict[str, dict[str, datetime | None]] = defaultdict(dict)
    for book, t in (db.query(UFCBookMarketHistory.book, func.max(UFCBookMarketHistory.captured_at))
                    .filter(UFCBookMarketHistory.book.in_(multi)).group_by(UFCBookMarketHistory.book)):
        out[book]["direct"] = t
    labels = multi + [f"BFO:{b}" for b in multi]
    for raw, t in (db.query(UFCFightOddsHistory.bookmaker, func.max(UFCFightOddsHistory.captured_at))
                   .filter(UFCFightOddsHistory.bookmaker.in_(labels)).group_by(UFCFightOddsHistory.bookmaker)):
        book, source = book_name(raw)
        out[book][source] = t
    return out


def fight_cutoff(db: Session, fight_id: int, event_date: date | None) -> datetime | None:
    cut = cutoffs_for_fights(db, [fight_id]).get(fight_id)
    if cut is None and event_date is not None:
        cut = datetime.combine(event_date, datetime.min.time())
    return cut


def book_history(db: Session, fight_id: int) -> dict | None:
    """The /fights/{id}/book-history payload, or None for an unknown fight."""
    row = (db.query(UFCFight, UFCEvent).join(UFCEvent, UFCEvent.id == UFCFight.event_id)
           .filter(UFCFight.id == fight_id).first())
    if row is None:
        return None
    fight, event = row
    cutoff = fight_cutoff(db, fight_id, event.date)
    started = fight_started(cutoff)
    # Upcoming: the axis runs to the cutoff, but nothing can exist past now anyway.
    limit = cutoff or _now()

    by_book: dict[str, dict[str, list[Quote]]] = defaultdict(dict)

    direct: dict[str, list] = defaultdict(list)
    for r in (db.query(UFCBookMarketHistory.book, UFCBookMarketHistory.captured_at, UFCBookMarketHistory.side,
                       UFCBookMarketHistory.american, UFCBookMarketHistory.max_stake)
              .filter(UFCBookMarketHistory.fight_id == fight_id,
                      UFCBookMarketHistory.market_key.in_(("moneyline_red", "moneyline_blue")),
                      UFCBookMarketHistory.captured_at < limit)):
        direct[r.book].append((r.captured_at, r.side, r.american, r.max_stake))
    for book, rows in direct.items():
        by_book[book]["direct"] = pair_sides(rows)

    hist: dict[tuple[str, str], list[Quote]] = defaultdict(list)
    for r in (db.query(UFCFightOddsHistory.bookmaker, UFCFightOddsHistory.captured_at,
                       UFCFightOddsHistory.red_odds, UFCFightOddsHistory.blue_odds)
              .filter(UFCFightOddsHistory.fight_id == fight_id, UFCFightOddsHistory.captured_at < limit)):
        book, source = book_name(r.bookmaker)
        if any(x in book.lower() for x in _NOT_BOOKS):
            continue
        hist[(book, source)].append(Quote(r.captured_at, r.red_odds, r.blue_odds))
    for (book, source), quotes in hist.items():
        by_book[book][source] = quotes

    alive = _alive(db, by_book)
    books: list[dict] = []
    merged: dict[str, list[Quote]] = {}
    for book, sources in by_book.items():
        quotes, used = merge_sources(sources, alive.get(book))
        pts = book_points(quotes, limit)
        if not pts:
            continue
        merged[book] = quotes
        books.append({"book": book, "sharp": book == SHARP, "soft": book in SOFT, "sources": used,
                      "points": pts})
    books.sort(key=lambda b: (not b["sharp"], not b["soft"], b["book"]))

    soft = soft_median({b["book"]: b["points"] for b in books if b["soft"]})
    sharp = next((b["points"] for b in books if b["sharp"]), [])
    stakes = stake_steps(merged.get(SHARP, []), limit)

    return {
        "fight_id": str(fight_id),
        "cutoff": cutoff.isoformat() if cutoff else None,
        "started": started,
        # Where "now" sits on the axis (0 once the bout has started): every store writes on
        # change, so each line's last price holds from its last point up to here.
        "now_days": days_before(min(_now(), limit), limit),
        "books": books,
        "sharp_soft": {"sharp": sharp, "soft_median": soft, "max_stake": stakes} if sharp and soft else None,
        "markers": markers(db, fight, event, cutoff, books),
    }


def markers(db: Session, fight: UFCFight, event: UFCEvent, cutoff: datetime | None, books: list[dict]) -> list[dict]:
    """Things that happened in the window the lines cover, placed on the same days-to-fight axis."""
    if cutoff is None:
        return []
    first = min((datetime.fromisoformat(b["points"][0]["t"]) for b in books if b["points"]), default=None)
    start = min(first, cutoff - timedelta(days=14)) if first else cutoff - timedelta(days=14)
    now = _now()
    out: list[dict] = []

    def add(kind: str, t: datetime, label: str, detail: str | None = None, scheduled: bool = False,
            **extra) -> None:
        # Only the weigh-in is placed ahead of time: it is on the schedule, the rest are events.
        if start <= t < cutoff and (t <= now or scheduled):
            out.append({"kind": kind, "t": t.isoformat(), "days_to_fight": days_before(t, cutoff),
                        "label": label, "detail": detail, **extra})

    starts = [t for t in (event.early_prelims_at, event.prelims_at, event.main_card_at) if t]
    if event.date is not None:
        t = weigh_in_at(event.date, min(starts) if starts else None)
        add("weigh_in", t, "Weigh-ins", scheduled=True, upcoming=t > now)

    ids = {fight.red_fighter_id, fight.blue_fighter_id}
    pulled = (db.query(UFCCancelledBout)
              .filter((UFCCancelledBout.red_fighter_id.in_(ids)) | (UFCCancelledBout.blue_fighter_id.in_(ids)))
              .all())
    names = {f.id: f for f in db.query(UFCFighter).filter(UFCFighter.id.in_(
        ids | {x for p in pulled for x in (p.red_fighter_id, p.blue_fighter_id) if x}))}
    last = lambda fid: (names[fid].last_name or names[fid].first_name) if fid in names else "?"
    for p in pulled:
        if p.event_date is None or event.date is None or abs(p.event_date - event.date) > CARD_WINDOW:
            continue
        pair = {p.red_fighter_id, p.blue_fighter_id}
        if pair == ids:
            continue      # the same bout pulled and rebooked: not a new opponent
        kept = next((f for f in pair if f in ids), None)
        gone = next((f for f in pair if f not in ids), None)
        if kept is None:
            continue
        new_opp = fight.blue_fighter_id if kept == fight.red_fighter_id else fight.red_fighter_id
        days_out = (event.date - p.removed_at.date()).days
        add("opponent_change", p.removed_at, "Opponent change",
            f"{last(kept)} vs {last(gone)} pulled; {last(new_opp)} in"
            + (f" on {days_out} days' notice" if 0 <= days_out <= 28 else ""),
            short_notice=0 <= days_out <= 14)

    lo, hi = (cutoff - NEWS_WINDOW).date(), cutoff.date()
    counts: dict[int, dict[date, int]] = defaultdict(dict)
    for m in (db.query(UFCNewsMention).filter(UFCNewsMention.fighter_id.in_(ids),
                                              UFCNewsMention.day >= lo, UFCNewsMention.day <= hi)):
        counts[m.fighter_id][m.day] = m.count
    for fid, days in counts.items():
        for d, n in news_spikes(days, lo, hi):
            add("news", datetime.combine(d, datetime.min.time()) + timedelta(hours=12),
                f"{last(fid)} in the news", f"{n} headlines")

    pred = db.query(UFCFightPrediction.created_at).filter(UFCFightPrediction.fight_id == fight.id).scalar()
    if pred is not None:
        add("model", pred, "Model published")

    out.sort(key=lambda m: m["t"])
    return out
