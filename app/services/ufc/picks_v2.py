"""Picks v2: a graded pick for every market on an upcoming card.

Per fight, every market we price:
  moneyline (red / blue)                          winner_open | winner_close
  winner x method cells (yes side)                sixway_ko | sixway_sub | sixway_dec
  goes to decision (yes / no)                     decision_yes | decision_no
  fighter wins inside the distance (yes / no)     itd_yes | itd_no
  O/U 1.5 / 2.5 rounds (over / under)             ou_1_5_over ... ou_2_5_under
  fight starts round 2 / 3 (yes / no)             starts_r2_yes ... starts_r3_no
  O/U 3.5 / 4.5 on five-round fights              not graded yet ("NR")

EV per $1 = p_model x decimal(best price) - 1. The pick is the side with the higher EV; if
no side has EV > 0 the market is listed with grade "—". The grade is the realised ROI of the
pick's EV band in that market's backtest (app/services/ufc/grading.py, built by
scripts/build_grade_table.py) — nothing else changes it. The moneyline uses the opening-line
history (winner_open) while the line watcher first saw the fight < 24 h ago, else the
closing-line history (winner_close). Informational badges (never change the grade):
  opening line, outlier price (best > 1.15x the median book's price), line moved away
  (market moved > 2 pts away from the pick since it opened), high variance (decimal >= 6),
  small sample (the band has < 50 historical bets), stale price.
Exchanges (Polymarket, Kalshi) never feed the grade.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from statistics import median

from sqlalchemy.orm import Session

from app.services.ufc.grading import decimal, grade, load_table

STALE_HOURS = 36
OPENING_HOURS = 24
OUTLIER_RATIO = 1.15
MOVED_AWAY_PTS = 0.02
LONGSHOT_DECIMAL = 6.0
EXCHANGE_BOOKS = ("polymarket", "kalshi")
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


def choose_pick(sides: list[dict]) -> dict | None:
    """sides: [{side, p, best, med, ...}] -> the side with the highest positive EV, or None."""
    best = None
    for s in sides:
        ev = evaluate_side(s.get("p"), s.get("best"), s.get("med"))["ev_best"]
        if ev is not None and ev > 0 and (best is None or ev > best[0]):
            best = (ev, s)
    return None if best is None else best[1]


def graded(family: str, side: dict, q_now: float | None, q_open: float | None,
           table: dict | None = None) -> tuple[str, dict | None, list[str]]:
    """Grade (ROI of the EV band, at the best price) + its basis + informational badges."""
    ev = evaluate_side(side.get("p"), side.get("best"), side.get("med"))
    g, basis = grade(family, ev["ev_best"], table)
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


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _moneyline(db: Session, fight_id: int) -> dict:
    """Latest price per sportsbook (line watcher 'BFO:<book>' rows, else ufc_fight_odds), the
    best / median per side, the de-vigged median, and the opening consensus."""
    from app.models.ufc import UFCFightOdds, UFCFightOddsHistory
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
        for o in db.query(UFCFightOdds).filter(UFCFightOdds.fight_id == fight_id):
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
                     "med": american_from_decimal(median(decimal(p) for p, _ in prices))}
    return out


def _round_s(rp, t: float) -> float | None:
    if rp is None:
        return None
    for pt in json.loads(rp.curve):
        if abs(pt["t"] - t) < 1e-6:
            return pt["s"]
    return None


def _event_fights(db: Session, event_id=None, fight_id=None):
    from app.models.ufc import UFCEvent, UFCFight
    q = db.query(UFCFight, UFCEvent).join(UFCEvent, UFCEvent.id == UFCFight.event_id)
    if fight_id is not None:
        return q.filter(UFCFight.id == fight_id).all()
    if event_id is None:
        nxt = (db.query(UFCEvent).filter(UFCEvent.date >= date.today()).order_by(UFCEvent.date).first())
        if nxt is None:
            return []
        event_id = nxt.id
    return (q.filter(UFCFight.event_id == event_id, UFCFight.winner_id.is_(None))
            .order_by(UFCFight.card_position.is_(None), UFCFight.card_position).all())


def build(db: Session, event_id=None, fight_id=None) -> dict:
    from app.models.ufc import (
        UFCFighter, UFCFightPrediction, UFCMethodPrediction, UFCRoundPrediction,
    )
    from app.services.ufc.prop_serving import prop_markets
    from app.services.ufc.rounds_v1 import round_payload  # noqa: F401  (table guard lives there)
    from sqlalchemy import inspect

    from app.database import engine

    table = load_table()
    rows = _event_fights(db, event_id, fight_id)
    has_rounds = inspect(engine).has_table(UFCRoundPrediction.__tablename__,
                                           schema=UFCRoundPrediction.__table__.schema)
    now = datetime.utcnow()
    fights_out, event = [], None
    for f, e in rows:
        event = event or {"id": str(e.id), "name": e.name, "date": e.date.isoformat() if e.date else None}
        names = {x.id: x for x in db.query(UFCFighter).filter(UFCFighter.id.in_([f.red_fighter_id, f.blue_fighter_id]))}
        nm = lambda i: (f"{names[i].first_name} {names[i].last_name}".strip(), names[i].last_name) if i in names else ("?", "?")
        red_full, red_last = nm(f.red_fighter_id); blue_full, blue_last = nm(f.blue_fighter_id)
        last = {"red": red_last, "blue": blue_last}
        pred = db.query(UFCFightPrediction).filter(UFCFightPrediction.fight_id == f.id).first()
        mp = db.query(UFCMethodPrediction).filter(UFCMethodPrediction.fight_id == f.id).first()
        rp = db.query(UFCRoundPrediction).filter(UFCRoundPrediction.fight_id == f.id).first() if has_rounds else None
        props = prop_markets(db, f.id)
        ml = _moneyline(db, f.id)
        five = (f.max_fight_time_seconds or 0) >= 1500
        markets = []

        def add(family, market_key, label, sides, q_of, captured):
            """sides: [{side, label, p, best, med, book}]; q_of(side) -> market no-vig prob."""
            pick = choose_pick(sides)
            ref = pick or max(sides, key=lambda s: (s.get("p") or 0))
            ev = evaluate_side(ref.get("p"), ref.get("best"), ref.get("med"))
            fam = family(ref["side"]) if callable(family) else family
            if pick:
                g, basis, badges = graded(fam, pick, q_of(pick["side"]), q_of(pick["side"], opening=True), table)
            else:
                g, basis, badges = "—", None, []
            stale = bool(captured and (now - captured) > timedelta(hours=STALE_HOURS))
            q = q_of(ref["side"])
            markets.append({
                "family": fam, "market_key": market_key, "label": label,
                "pick": pick["side"] if pick else None, "pick_label": pick["label"] if pick else None,
                "p_model": ref.get("p"), "q_market": q,
                "best_american": ref.get("best"), "best_book": ref.get("book"),
                "median_american": ref.get("med"),
                "ev_best": ev["ev_best"], "ev_median": ev["ev_median"],
                "edge_pts": None if q is None or ref.get("p") is None else round(100 * (ref["p"] - q), 2),
                "grade": g, "grade_basis": basis, "badges": badges + (["stale price"] if stale else []),
                "price_captured_at": captured.isoformat() if captured else None,
                "stale": stale, "source": "BFO consensus" if market_key != "moneyline" else "sportsbooks"})

        # moneyline
        if pred and ml:
            opening = bool(ml.get("first_seen") and (now - ml["first_seen"]) < timedelta(hours=OPENING_HOURS))
            pr = pred.red_prob
            sides = [{"side": "red", "label": red_full, "p": pr, **ml["red"]},
                     {"side": "blue", "label": blue_full, "p": 1 - pr, **ml["blue"]}]
            def q_ml(side, opening=False):
                q = ml.get("q_red_open") if opening else ml.get("q_red")
                return None if q is None else (q if side == "red" else 1 - q)
            add("winner_open" if opening else "winner_close", "moneyline", "Moneyline", sides, q_ml,
                ml.get("captured_at"))
            if opening:
                markets[-1]["badges"].append("opening line")

        def prop_side(market, side, label, p):
            q = props.get(market, {})
            return {"side": side, "label": label, "p": p, "best": q.get("best_american"),
                    "med": q.get("median_american"), "book": q.get("best_book")}

        def q_prop(market_of):
            def f(side, opening=False):
                q = props.get(market_of(side), {})
                return q.get("opening_prob") if opening else q.get("prob")
            return f

        def captured(*keys):
            ts = [props[k]["captured_at"] for k in keys if k in props and props[k].get("captured_at")]
            return datetime.fromisoformat(max(ts)) if ts else None

        if mp and mp.red_ko_prob is not None:
            cell = {"red_ko": mp.red_ko_prob, "red_sub": mp.red_sub_prob, "red_dec": mp.red_dec_prob,
                    "blue_ko": mp.blue_ko_prob, "blue_sub": mp.blue_sub_prob, "blue_dec": mp.blue_dec_prob}
            for key, p in cell.items():
                corner, m = key.split("_")
                lbl = f"{last[corner]} by {SIXWAY_LABEL[m]}"
                add(f"sixway_{m}", key, lbl, [prop_side(key, "yes", lbl, p)], q_prop(lambda s, k=key: k),
                    captured(key))
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
        fights_out.append({
            "fight_id": str(f.id), "card_position": f.card_position, "five_round": five,
            "red": {"id": str(f.red_fighter_id), "name": red_full},
            "blue": {"id": str(f.blue_fighter_id), "name": blue_full},
            "markets": markets})
    return {"event": event, "generated_at": now.isoformat(timespec="seconds"),
            "grade_table": {"version": (table or {}).get("version"), "built_at": (table or {}).get("built_at")},
            "fights": fights_out}
