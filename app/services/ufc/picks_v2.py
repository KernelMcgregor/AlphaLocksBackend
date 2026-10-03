"""Picks v2: a graded pick for every market on an upcoming card.

Per fight, every market we price:
  moneyline (red / blue)                          winner_open | winner_close
  winner x method cells (yes side)                sixway_ko | sixway_sub | sixway_dec
  goes to decision (yes / no)                     decision_yes | decision_no
  fighter wins inside the distance (yes / no)     itd_yes | itd_no
  O/U 1.5 / 2.5 rounds (over / under)             ou_1_5_over ... ou_2_5_under
  fight starts round 2 / 3 (yes / no)             starts_r2_yes ... starts_r3_no
  O/U 3.5 / 4.5 on five-round fights              not graded yet ("NR")

EV per $1 = p x decimal(price) - 1, with p the model blended with the market for props (the
moneyline's p is already market-stacked) and the price the TYPICAL book's (median), which is
what picks and grades use; EV at the best price is reported too. The pick is the side with the higher EV; if
no side has EV > 0 the market is listed with grade "—". The grade is the realised ROI of the
pick's EV band in that market's backtest (app/services/ufc/grading.py, built by
scripts/build_grade_table.py) — nothing else changes it. The moneyline uses the opening-line
history (winner_open) while the line watcher first saw the fight < 24 h ago, else the
closing-line history (winner_close). Informational badges (never change the grade):
  opening line, outlier price (best > 1.15x the median book's price), line moved away
  (market moved > 2 pts away from the pick since it opened), high variance (decimal >= 6),
  small sample (the band has < 50 historical bets), stale price.
Exchanges (Kalshi, Polymarket) are price venues like any book, at the cost of buying the Yes
leg (ask, plus Kalshi's taker fee), but only once a quote has really traded: Polymarket seeds new
props with placeholder prices and no volume. A pick priced on one carries an "exchange price"
badge. Exchanges are not in the grade table's backtest (their history is too short), so the band
ROI behind such a grade comes from sportsbook prices.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from statistics import median

from sqlalchemy.orm import Session

from app.services.ufc.grading import blend_prob, decimal, grade, load_table

STALE_HOURS = 36
OPENING_HOURS = 24
OUTLIER_RATIO = 1.15
MOVED_AWAY_PTS = 0.02
LONGSHOT_DECIMAL = 6.0
EXCHANGE_BOOKS = ("polymarket", "kalshi")
EXCHANGE_NAME = {"kalshi": "Kalshi", "polymarket": "Polymarket"}
#: Taker fee per $1 contract = rate x P x (1 - P). Kalshi charges 0.07; Polymarket's UFC
#: markets charge none.
EXCHANGE_FEE_RATE = {"kalshi": 0.07, "polymarket": 0.0}
MAX_EXCHANGE_SPREAD = 0.10
EXCHANGE_MIN_PRICE = 0.02
#: Smallest EV at the typical price that counts as a pick. Below it the edge is inside the
#: noise of a price moving a few cents, so the market is listed without a pick.
MIN_PICK_EV = 0.03
GRADE_RANK = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F", "NR", "—"]
#: Polymarket outcome -> this page's market key. Only Yes legs are stored, so only Yes sides
#: can be bought. Its "O/U n.5 rounds" markets are left out: which leg is stored isn't recorded.
POLY_PROP_KEY = {"red_ko_tko": "red_ko", "blue_ko_tko": "blue_ko", "distance": "dec_yes",
                 "before_r2": "sr_2_no", "before_r3": "sr_3_no"}
SIXWAY_LABEL = {"ko": "KO/TKO", "sub": "submission", "dec": "decision"}


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def american_from_decimal(d: float) -> int:
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def evaluate_side(p: float | None, best: float | None, med: float | None) -> dict:
    db, dm = decimal(best), decimal(med)
    return {"ev_best": None if p is None or db is None else p * db - 1,
            "ev_median": None if p is None or dm is None else p * dm - 1,
            "decimal_best": db, "decimal_median": dm}


def typical_ev(e: dict) -> float | None:
    """EV at the typical (median) book's price -- what a reader can actually get; the best
    price only where no median is stored."""
    return e["ev_median"] if e["ev_median"] is not None else e["ev_best"]


def choose_pick(sides: list[dict]) -> dict | None:
    """sides: [{side, p, best, med, ...}] -> the side with the highest positive EV at the
    typical price, or None."""
    best = None
    for s in sides:
        ev = typical_ev(evaluate_side(s.get("p"), s.get("best"), s.get("med")))
        if ev is not None and ev >= MIN_PICK_EV and (best is None or ev > best[0]):
            best = (ev, s)
    return None if best is None else best[1]


def graded(family: str, side: dict, q_now: float | None, q_open: float | None,
           table: dict | None = None) -> tuple[str, dict | None, list[str]]:
    """Grade (expected ROI of the EV band at the typical price; grading.py) + its basis +
    informational badges."""
    ev = evaluate_side(side.get("p"), side.get("best"), side.get("med"))
    g, basis = grade(family, typical_ev(ev), table)
    badges = []
    if ev["decimal_best"] and ev["decimal_median"] and ev["decimal_best"] > OUTLIER_RATIO * ev["decimal_median"]:
        badges.append("outlier price")
    if ev["decimal_best"] and ev["decimal_best"] >= LONGSHOT_DECIMAL:
        badges.append("high variance")
    if q_now is not None and q_open is not None and (q_now - q_open) < -MOVED_AWAY_PTS:
        badges.append("line moved away")
    if basis and basis.get("small_sample"):
        badges.append("small sample")
    return g, basis, badges


def top_drivers(shap_rows, n: int = 6) -> list[dict]:
    """The winner model's biggest SHAP contributions for the page's "why" panel. Odds
    features are dropped: the price is what the pick is measured against, so citing it as
    a reason would be circular. shap_value > 0 favours red."""
    rows = [r for r in shap_rows if "odds" not in r.feature_name]
    rows.sort(key=lambda r: abs(r.shap_value), reverse=True)
    return [{"feature_name": r.feature_name, "shap_value": round(r.shap_value, 4),
             "feature_value": None if r.feature_value is None else round(r.feature_value, 4)}
            for r in rows[:n]]


def exchange_american(platform: str, price: float | None) -> int | None:
    """American odds for buying a $1 exchange contract at `price`, after the taker fee."""
    if price is None or not 0 < price < 1:
        return None
    cost = price + EXCHANGE_FEE_RATE.get(platform, 0.0) * price * (1 - price)
    return american_from_decimal(1 / cost) if cost < 1 else None


def exchange_buy_price(bid, ask, last, volume) -> float | None:
    """What it costs to take the Yes leg now, or None when the quote is not a real market:
    the ask when a book is quoted (and not absurdly wide), else the last trade if anything
    has traded. A seeded Polymarket prop has neither and is skipped."""
    # A live order book only: a last trade can be hours old or post-result (a finished
    # fight's Polymarket market sat at 0.0005 and was served as +199900). Prices at the
    # extremes are settled or settling markets, not bets.
    if ask is None or not (EXCHANGE_MIN_PRICE <= ask <= 1 - EXCHANGE_MIN_PRICE):
        return None
    return ask if bid is None or ask - bid <= MAX_EXCHANGE_SPREAD else None


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _exchanges(db: Session, fight_id: int, rows: list | None = None) -> dict:
    """{market_key: {"Kalshi": american, ...}} for traded exchange quotes, plus "ml_q_red", the
    exchange consensus (mean mid per side, normalised) for fights no sportsbook prices."""
    from app.models.ufc import UFCPredictionMarket as M, UFCPredictionMarketQuote as Q
    if rows is None:
        rows = (db.query(M, Q).join(Q, Q.market_id == M.id)
                .filter(M.fight_id == fight_id, M.status == "open").all())
    out: dict = {}
    mids = {"red": [], "blue": []}
    for m, q in rows:
        if m.market_type == "moneyline" and m.side in ("red", "blue"):
            key = f"ml_{m.side}"
        elif m.platform == "polymarket" and m.outcome_key in POLY_PROP_KEY:
            key = POLY_PROP_KEY[m.outcome_key]
        else:
            continue
        buy = exchange_buy_price(q.bid, q.ask, q.price, q.volume)
        a = exchange_american(m.platform, buy)
        if a is None:
            continue
        name = EXCHANGE_NAME.get(m.platform, m.platform)
        cur = out.setdefault(key, {})
        # a venue can list the same fight twice; keep its better price
        if name not in cur or decimal(a) > decimal(cur[name]):
            cur[name] = a
        if key.startswith("ml_"):
            mids[m.side].append(q.price)
    if mids["red"] and mids["blue"]:
        r = sum(mids["red"]) / len(mids["red"]); b = sum(mids["blue"]) / len(mids["blue"])
        out["ml_q_red"] = r / (r + b)
    return out


#: Reference only: its no-vig price is the market probability, but it does not take US
#: customers, so it never sets the best price (and so never the EV or the grade).
SHARP_BOOK = "Pinnacle"

# Only the columns _direct_books uses: a database whose copy of the table predates a later
# column (max_stake) must still serve the page.
_DIRECT_COLS = lambda H: (H.fight_id, H.book, H.external_market_id, H.external_selection_id,
                          H.market_key, H.american, H.line, H.captured_at)


def _direct_books(db: Session, fight_id: int, rows: list | None = None) -> tuple[dict, dict, dict]:
    """Prices read straight from sportsbooks (ufc_book_market_history: FanDuel, Bovada,
    BetRivers, Pinnacle).

    Returns ({market_key: {book: american}}, {book: {market_key: no-vig prob}},
    {market_key: captured_at}). The no-vig prob normalises one book's market over its own
    selections (per line, for markets that list several). A market whose prices sum to under
    100% is skipped: it lists only some outcomes (FanDuel's "what round will the fight end"
    leaves out the decision), and scaling it to 100% would inflate every selection. Moneyline
    keys are renamed ml_red / ml_blue to match the exchange prices they are merged with.
    """
    from app.models.ufc import UFCBookMarketHistory as H
    # Only the columns used here: a database whose copy of the table predates a later column
    # (max_stake) must still serve the page.
    if rows is None:
        rows = (db.query(*_DIRECT_COLS(H)).filter(H.fight_id == fight_id).order_by(H.captured_at).all())
    latest: dict = {}
    for h in rows:
        latest[(h.book, h.external_market_id, h.external_selection_id)] = h
    prices, at, groups = {}, {}, {}
    for h in latest.values():
        if h.american is None:
            continue
        groups.setdefault((h.book, h.external_market_id, h.line), []).append(h)
        if not h.market_key:
            continue
        key = h.market_key.replace("moneyline_", "ml_")
        prices.setdefault(key, {})[h.book] = h.american
        at[key] = max(at.get(key, h.captured_at), h.captured_at)
    q: dict = {}
    for (book, _, _), rows in groups.items():
        implied = [1 / decimal(r.american) for r in rows]
        total = sum(implied)
        if len(rows) < 2 or total < 1:
            continue
        for r, p in zip(rows, implied):
            if r.market_key:
                q.setdefault(book, {}).setdefault(r.market_key.replace("moneyline_", "ml_"), p / total)
    return prices, q, at


def merge_prices(side: dict, extra: dict | None) -> dict:
    """Fold exchange prices into a side's books; best = highest payout across all of them.
    The median stays the sportsbooks' (it is the "typical book"), unless only exchanges price it."""
    if not extra:
        return side
    books = {**(side.get("books") or ({side["book"]: side["best"]} if side.get("best") is not None else {})), **extra}
    book, best = max(books.items(), key=lambda kv: decimal(kv[1]))
    med = side.get("med")
    if med is None:
        med = american_from_decimal(median(decimal(v) for v in books.values()))
    return {**side, "best": best, "book": book, "med": med, "books": books}

def _moneyline(db: Session, fight_id: int, hist: list | None = None,
               fallback: list | None = None) -> dict:
    """Latest price per sportsbook (line watcher 'BFO:<book>' rows, else ufc_fight_odds), the
    best / median per side, the de-vigged median, and the opening consensus. hist / fallback
    are those rows preloaded (ascending captured_at); queried here when not given."""
    from app.models.ufc import UFCFightOdds, UFCFightOddsHistory
    if hist is None:
        hist = (db.query(UFCFightOddsHistory).filter(UFCFightOddsHistory.fight_id == fight_id,
                                                     UFCFightOddsHistory.bookmaker.like("BFO:%"))
                .order_by(UFCFightOddsHistory.captured_at).all())
    latest, first, first_seen = {}, {}, None
    for h in hist:
        book = h.bookmaker.removeprefix("BFO:")
        if any(x in book.lower() for x in EXCHANGE_BOOKS) or not (h.red_odds and h.blue_odds):
            continue
        first.setdefault(book, (h.red_odds, h.blue_odds))
        latest[book] = (h.red_odds, h.blue_odds, h.captured_at)
        first_seen = first_seen or h.captured_at
    if not latest:
        if fallback is None:
            fallback = db.query(UFCFightOdds).filter(UFCFightOdds.fight_id == fight_id).all()
        for o in fallback:
            if o.red_odds and o.blue_odds and not any(x in (o.bookmaker or "").lower() for x in EXCHANGE_BOOKS):
                latest[o.bookmaker] = (o.red_odds, o.blue_odds, getattr(o, "updated_at", None))
    if not latest:
        return {}

    def nv(pairs):
        r = median(1 / decimal(a) for a, _ in pairs); b = median(1 / decimal(b) for _, b in pairs)
        return r / (r + b)
    out = {"q_red": nv([(r, b) for r, b, _ in latest.values()]),
           "q_red_open": nv(list(first.values())) if first else None,
           "first_seen": first_seen,
           "captured_at": max((t for *_, t in latest.values() if t), default=None)}
    for i, side in enumerate(("red", "blue")):
        prices = [(v[i], book) for book, v in latest.items()]
        best = max(prices, key=lambda x: decimal(x[0]))
        out[side] = {"best": best[0], "book": best[1],
                     "med": american_from_decimal(median(decimal(p) for p, _ in prices)),
                     "books": {book: p for p, book in prices}}
    return out


def _round_s(rp, t: float) -> float | None:
    if rp is None:
        return None
    for pt in json.loads(rp.curve):
        if abs(pt["t"] - t) < 1e-6:
            return pt["s"]
    return None


def _event_fights(db: Session, event_id=None, fight_id=None, all_upcoming=False):
    from app.models.ufc import UFCEvent, UFCFight
    q = db.query(UFCFight, UFCEvent).join(UFCEvent, UFCEvent.id == UFCFight.event_id)
    if fight_id is not None:
        return q.filter(UFCFight.id == fight_id).all()
    if all_upcoming:
        # same window as /ufc/upcoming: yesterday on, so a card stays listed through UTC rollover
        return (q.filter(UFCEvent.date >= date.today() - timedelta(days=1), UFCFight.winner_id.is_(None))
                .order_by(UFCEvent.date, UFCFight.card_position.is_(None), UFCFight.card_position).all())
    if event_id is None:
        nxt = (db.query(UFCEvent).filter(UFCEvent.date >= date.today()).order_by(UFCEvent.date).first())
        if nxt is None:
            return []
        event_id = nxt.id
    return (q.filter(UFCFight.event_id == event_id, UFCFight.winner_id.is_(None))
            .order_by(UFCFight.card_position.is_(None), UFCFight.card_position).all())


def _preload(db: Session, fight_ids: list, has_rounds: bool) -> dict:
    """Everything build() reads per fight, in one query per table rather than ~10 per fight:
    against a remote database that is the difference between ~1 s and ~25 s for a slate."""
    from collections import defaultdict

    from app.models.ufc import (
        UFCBookMarketHistory as H, UFCFightOdds, UFCFightOddsHistory, UFCFighter, UFCFight,
        UFCFightPrediction, UFCFightShapValue, UFCMethodPrediction, UFCPredictionMarket as PM,
        UFCPredictionMarketQuote as PQ, UFCPropOddsHistory as P, UFCRoundPrediction,
    )
    ids = list(fight_ids)
    group = lambda rows, attr="fight_id": _grouped(rows, attr, defaultdict(list))
    one = lambda model: {r.fight_id: r for r in db.query(model).filter(model.fight_id.in_(ids))} if ids else {}
    fighter_ids = {i for f in db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id)
                   .filter(UFCFight.id.in_(ids)) for i in f} if ids else set()
    props = (db.query(P).filter(P.fight_id.in_(ids), P.source.in_(("bfo_watch", "bfo_close")))
             .order_by(P.captured_at).all()) if ids else []
    return {
        "fighters": {x.id: x for x in db.query(UFCFighter).filter(UFCFighter.id.in_(fighter_ids))} if fighter_ids else {},
        "pred": one(UFCFightPrediction), "mp": one(UFCMethodPrediction),
        "rp": one(UFCRoundPrediction) if has_rounds else {},
        "shap": group(db.query(UFCFightShapValue).filter(UFCFightShapValue.fight_id.in_(ids)).all() if ids else []),
        "props_watch": group([r for r in props if r.source == "bfo_watch"]),
        "props_close": group([r for r in props if r.source == "bfo_close"]),
        "ml_hist": group(db.query(UFCFightOddsHistory).filter(
            UFCFightOddsHistory.fight_id.in_(ids), UFCFightOddsHistory.bookmaker.like("BFO:%"))
            .order_by(UFCFightOddsHistory.captured_at).all() if ids else []),
        "ml_fallback": group(db.query(UFCFightOdds).filter(UFCFightOdds.fight_id.in_(ids)).all() if ids else []),
        "exchanges": _grouped(db.query(PM, PQ).join(PQ, PQ.market_id == PM.id)
                              .filter(PM.fight_id.in_(ids), PM.status == "open").all() if ids else [],
                              None, defaultdict(list), key=lambda r: r[0].fight_id),
        # the directly-read book table is created by the first scraper run; until then, none
        "direct": group(db.query(*_DIRECT_COLS(H)).filter(H.fight_id.in_(ids))
                        .order_by(H.captured_at).all() if ids and _has_table(H) else []),
    }


def _grouped(rows, attr, out, key=None):
    for r in rows:
        out[key(r) if key else getattr(r, attr)].append(r)
    return out


def _has_table(model) -> bool:
    from sqlalchemy import inspect

    from app.database import engine
    return inspect(engine).has_table(model.__tablename__, schema=model.__table__.schema)


def _has_rounds() -> bool:
    from app.models.ufc import UFCRoundPrediction
    return _has_table(UFCRoundPrediction)


def build(db: Session, event_id=None, fight_id=None) -> dict:
    """One card (default: the next event), or one fight."""
    return _build_rows(db, _event_fights(db, event_id, fight_id))


def build_all(db: Session) -> list[dict]:
    """Every upcoming card, one payload per event (same shape as build), from a single preload."""
    rows = _event_fights(db, all_upcoming=True)
    by_event: dict = {}
    for f, e in rows:
        by_event.setdefault(e.id, []).append((f, e))
    pre = _preload(db, [f.id for f, _ in rows], _has_rounds())
    return [p for p in (_build_rows(db, r, pre) for r in by_event.values()) if p["fights"]]


def _build_rows(db: Session, rows, pre: dict | None = None) -> dict:
    from app.services.ufc.prop_serving import markets_from_rows

    table = load_table()
    if pre is None:
        pre = _preload(db, [f.id for f, _ in rows], _has_rounds())
    now = datetime.utcnow()
    fights_out, event = [], None
    for f, e in rows:
        event = event or {"id": str(e.id), "name": e.name, "date": e.date.isoformat() if e.date else None}
        names = pre["fighters"]
        nm = lambda i: (f"{names[i].first_name} {names[i].last_name}".strip(), names[i].last_name) if i in names else ("?", "?")
        red_full, red_last = nm(f.red_fighter_id); blue_full, blue_last = nm(f.blue_fighter_id)
        last = {"red": red_last, "blue": blue_last}
        pred = pre["pred"].get(f.id)
        mp = pre["mp"].get(f.id)
        rp = pre["rp"].get(f.id)
        props = markets_from_rows(pre["props_watch"].get(f.id, []), pre["props_close"].get(f.id, []))
        ml = _moneyline(db, f.id, pre["ml_hist"].get(f.id, []), pre["ml_fallback"].get(f.id, []))
        ex = _exchanges(db, f.id, pre["exchanges"].get(f.id, []))
        direct, q_by_book, direct_at = _direct_books(db, f.id, pre["direct"].get(f.id, []))
        # Market probability: Pinnacle's no-vig price where it has one (the sharpest line we
        # read), else BestFightOdds' consensus, else any directly-read book's no-vig price.
        sharp_q = q_by_book.get(SHARP_BOOK, {})
        direct_q = {}
        for book, qs in sorted(q_by_book.items(), key=lambda kv: kv[0] != SHARP_BOOK):
            for k, v in qs.items():
                direct_q.setdefault(k, v)
        # One venue map per market key: exchanges plus books read directly. A direct read
        # replaces the aggregator's copy of the same book (it is fresher).
        venues = {k: {**ex.get(k, {}), **{b: a for b, a in direct.get(k, {}).items() if b != SHARP_BOOK}}
                  for k in {*direct, *ex} if k != "ml_q_red"}
        five = (f.max_fight_time_seconds or 0) >= 1500
        markets = []

        def add(family, market_key, label, sides, q_of, captured):
            """sides: [{side, label, p, best, med, book}]; q_of(side) -> market no-vig prob.
            Props: p becomes the model blended with the market (grading.blend_prob), the
            probability the grade table was built on; the raw model is kept as p_model_raw.
            The moneyline's p is already the ensemble's market stack."""
            for sd in sides:
                sd["p_raw"] = sd.get("p")
                if market_key != "moneyline":
                    qs = sharp_q.get(sd.get("key"))
                    sd["p"] = blend_prob(sd.get("p"), qs if qs is not None else q_of(sd["side"]))
            pick = choose_pick(sides)
            ref = pick or max(sides, key=lambda s: (s.get("p") or 0))
            ev = evaluate_side(ref.get("p"), ref.get("best"), ref.get("med"))
            fam = family(ref["side"]) if callable(family) else family
            if pick:
                g, basis, badges = graded(fam, pick, q_of(pick["side"]), q_of(pick["side"], opening=True), table)
            else:
                g, basis, badges = "—", None, []
            if pick and pick.get("book") in EXCHANGE_NAME.values():
                badges = badges + ["exchange price"]
            stale = bool(captured and (now - captured) > timedelta(hours=STALE_HOURS))
            # display probability: Pinnacle where it prices this exact selection. The
            # "line moved away" badge above keeps comparing one source with itself.
            q_sharp = sharp_q.get(ref.get("key"))
            q = q_sharp if q_sharp is not None else q_of(ref["side"])
            source = ("Pinnacle no-vig" if q_sharp is not None
                      else "BFO consensus" if market_key != "moneyline" else "sportsbooks")
            markets.append({
                "family": fam, "market_key": market_key, "label": label,
                "pick": pick["side"] if pick else None, "pick_label": pick["label"] if pick else None,
                "p_model": ref.get("p"), "p_model_raw": ref.get("p_raw"), "q_market": q,
                "prob_basis": "model + market" if market_key != "moneyline" else "model (market-stacked)",
                "best_american": ref.get("best"), "best_book": ref.get("book"),
                "median_american": ref.get("med"),
                # every venue's price for this side: each sportsbook on the moneyline; for props
                # the best sportsbook (only best + median are stored) plus any exchange
                "books": ref.get("books"),
                # Pinnacle's price for the same selection, shown beside the bettable books
                "reference": ({"book": SHARP_BOOK, "american": direct[ref["key"]][SHARP_BOOK]}
                              if SHARP_BOOK in direct.get(ref.get("key"), {}) else None),
                "ev_best": ev["ev_best"], "ev_median": ev["ev_median"], "ev_typical": typical_ev(ev),
                "edge_pts": None if q is None or ref.get("p") is None else round(100 * (ref["p"] - q), 2),
                "grade": g, "grade_basis": basis, "badges": badges + (["stale price"] if stale else []),
                "price_captured_at": captured.isoformat() if captured else None,
                "stale": stale, "source": source})

        # moneyline
        if pred and (ml or venues.get("ml_red") or venues.get("ml_blue")):
            opening = bool(ml.get("first_seen") and (now - ml["first_seen"]) < timedelta(hours=OPENING_HOURS))
            pr = pred.red_prob
            sides = [merge_prices({"side": "red", "key": "ml_red", "label": red_full, "p": pr, **ml.get("red", {})}, venues.get("ml_red")),
                     merge_prices({"side": "blue", "key": "ml_blue", "label": blue_full, "p": 1 - pr, **ml.get("blue", {})}, venues.get("ml_blue"))]
            def q_ml(side, opening=False):
                q = ml.get("q_red_open") if opening else (ml.get("q_red") or direct_q.get("ml_red") or ex.get("ml_q_red"))
                return None if q is None else (q if side == "red" else 1 - q)
            ml_at = [t for t in (ml.get("captured_at"), direct_at.get("ml_red")) if t]
            add("winner_open" if opening else "winner_close", "moneyline", "Moneyline", sides, q_ml,
                max(ml_at) if ml_at else None)
            if opening:
                markets[-1]["badges"].append("opening line")

        def prop_side(market, side, label, p):
            q = props.get(market, {})
            return merge_prices({"side": side, "key": market, "label": label, "p": p, "best": q.get("best_american"),
                                 "med": q.get("median_american"), "book": q.get("best_book")}, venues.get(market))

        def q_prop(market_of):
            def f(side, opening=False):
                q = props.get(market_of(side), {})
                if opening:
                    return q.get("opening_prob")
                return q.get("prob") if q.get("prob") is not None else direct_q.get(market_of(side))
            return f

        def captured(*keys):
            ts = [datetime.fromisoformat(props[k]["captured_at"]) for k in keys
                  if k in props and props[k].get("captured_at")]
            ts += [direct_at[k] for k in keys if k in direct_at]
            return max(ts) if ts else None

        if mp and mp.red_ko_prob is not None:
            cell = {"red_ko": mp.red_ko_prob, "red_sub": mp.red_sub_prob, "red_dec": mp.red_dec_prob,
                    "blue_ko": mp.blue_ko_prob, "blue_sub": mp.blue_sub_prob, "blue_dec": mp.blue_dec_prob}
            # graded separately for the favourite and the underdog (the market misprices them
            # differently; see scripts/build_grade_table.py)
            fav = "red" if (pred.red_prob if pred else 0.5) >= 0.5 else "blue"
            for key, p in cell.items():
                corner, m = key.split("_")
                lbl = f"{last[corner]} by {SIXWAY_LABEL[m]}"
                add(f"sixway_{m}_{'fav' if corner == fav else 'dog'}", key, lbl, [prop_side(key, "yes", lbl, p)],
                    q_prop(lambda s, k=key: k), captured(key))
            dec = mp.distance_prob if mp.distance_prob is not None else mp.dec_prob
            add(lambda s: "decision_yes" if s == "yes" else "decision_no", "decision", "Goes to decision",
                [prop_side("dec_yes", "yes", "Goes to decision", dec),
                 prop_side("dec_no", "no", "Doesn't go to decision", 1 - dec)],
                q_prop(lambda s: "dec_yes" if s == "yes" else "dec_no"), captured("dec_yes", "dec_no"))
            for corner in ("red", "blue"):
                itd = cell[f"{corner}_ko"] + cell[f"{corner}_sub"]
                add(lambda s: "itd_yes" if s == "yes" else "itd_no", f"itd_{corner}",
                    f"{last[corner]} inside the distance",
                    [prop_side(f"itd_{corner}_yes", "yes", f"{last[corner]} inside the distance", itd),
                     prop_side(f"itd_{corner}_no", "no", f"Not {last[corner]} inside the distance", 1 - itd)],
                    q_prop(lambda s, c=corner: f"itd_{c}_{s}"), captured(f"itd_{corner}_yes"))
        if rp:
            lines = [("ou_1_5", "1.5", rp.over_1_5), ("ou_2_5", "2.5", rp.over_2_5)]
            if five:
                lines += [("ou_3_5", "3.5", rp.over_3_5), ("ou_4_5", "4.5", rp.over_4_5)]
            for fam, line, p in lines:
                if p is None:
                    continue
                add(lambda s, f_=fam: f"{f_}_{s}", fam, f"O/U {line} rounds",
                    [prop_side(f"ou_{line}_over", "over", f"Over {line} rounds", p),
                     prop_side(f"ou_{line}_under", "under", f"Under {line} rounds", 1 - p)],
                    q_prop(lambda s, l=line: f"ou_{l}_{s}"), captured(f"ou_{line}_over"))
            for n, t in ((2, 5.0), (3, 10.0)):
                p = _round_s(rp, t)
                if p is None:
                    continue
                add(lambda s, n=n: f"starts_r{n}_{s}", f"starts_r{n}", f"Fight starts round {n}",
                    [prop_side(f"sr_{n}", "yes", f"Starts round {n}", p),
                     prop_side(f"sr_{n}_no", "no", f"Doesn't start round {n}", 1 - p)],
                    q_prop(lambda s, n=n: f"sr_{n}" if s == "yes" else f"sr_{n}_no"), captured(f"sr_{n}"))
        def corner(fid, full):
            x = names.get(fid)
            return {"id": str(fid), "name": full, "nickname": x and x.nickname,
                    "image_url": x and x.image_url, "country_code": x and x.country_code}
        shap = pre["shap"].get(f.id, [])
        if rp:
            for n in range(1, 6 if five else 4):
                p = getattr(rp, f"p_end_r{n}", None)
                if p is None:
                    continue
                # books price only the Yes side; no backtest yet, so this grades "NR"
                add(f"ends_r{n}", f"er_{n}", f"Fight ends in round {n}",
                    [prop_side(f"er_{n}", "yes", f"Fight ends in round {n}", p)],
                    q_prop(lambda s, n=n: f"er_{n}"), captured(f"er_{n}"))
        # One primary pick per fight: the best grade, then the biggest EV at the typical
        # price. Several markets often carry the same opinion (X by decision, goes to
        # decision, not Y inside the distance); the rest are "related".
        picked = [mk for mk in markets if mk["pick"]]
        if picked:
            top = min(picked, key=lambda mk: (GRADE_RANK.index(mk["grade"]) if mk["grade"] in GRADE_RANK
                                              else len(GRADE_RANK), -(mk["ev_typical"] or 0)))
            for mk in markets:
                mk["primary"] = mk is top
        fights_out.append({
            "fight_id": str(f.id), "card_position": f.card_position, "five_round": five,
            "weight_class": f.weight_class,
            "red": corner(f.red_fighter_id, red_full),
            "blue": corner(f.blue_fighter_id, blue_full),
            "red_prob": pred.red_prob if pred else None,
            "method": {k: getattr(mp, f"{k}_prob") for k in
                       ("red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec")}
                      if mp and mp.red_ko_prob is not None else None,
            "drivers": top_drivers(shap),
            "markets": markets})
    return {"event": event, "generated_at": now.isoformat(timespec="seconds"),
            "grade_table": {"version": (table or {}).get("version"), "built_at": (table or {}).get("built_at")},
            "fights": fights_out}
