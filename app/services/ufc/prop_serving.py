"""Prop-market prices for the fight API (BestFightOdds consensus, ufc_prop_odds_history).

prop_markets(db, fight_id) -> {market: {prob, best_american, best_book, median_american,
    n_books, captured_at, opening_prob, opening_at, source}}
    Latest line-watcher snapshot per market (source 'bfo_watch'); for a fight the watcher
    never saw, the backfilled closing price ('bfo_close'). opening_prob is the first
    watcher snapshot (the observed opening prop line).
prop_history(db, fight_id) -> {market: [{t, prob, best_american}]}  (watcher snapshots)

Market keys: red_ko ... blue_dec (winner x method), itd_<corner>_yes/_no, dec_yes / dec_no,
ou_<line>_over / _under, sr_<N> / sr_<N>_no (fight starts round N), er_<N> (ends in round N).
`prob` is de-vigged; best/median prices are real American odds.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.ufc import UFCPropOddsHistory


def _rows(db: Session, fight_id: int, source: str):
    return (db.query(UFCPropOddsHistory)
            .filter(UFCPropOddsHistory.fight_id == fight_id, UFCPropOddsHistory.source == source)
            .order_by(UFCPropOddsHistory.captured_at).all())


def prop_markets(db: Session, fight_id: int) -> dict:
    watch = _rows(db, fight_id, "bfo_watch")
    rows, source = (watch, "bfo_watch") if watch else (_rows(db, fight_id, "bfo_close"), "bfo_close")
    out: dict[str, dict] = {}
    for h in rows:                      # ascending time: first seen = opening, last = latest
        cur = out.get(h.market)
        if cur is None:
            cur = out[h.market] = {"opening_prob": h.prob, "opening_at": h.captured_at.isoformat(),
                                   "source": source}
        cur.update({"prob": h.prob, "best_american": h.best_american, "best_book": h.best_book,
                    "median_american": h.median_american, "n_books": h.n_books,
                    "captured_at": h.captured_at.isoformat()})
    return out


def prop_history(db: Session, fight_id: int) -> dict:
    out: dict[str, list] = {}
    for h in _rows(db, fight_id, "bfo_watch"):
        out.setdefault(h.market, []).append(
            {"t": h.captured_at.isoformat(), "prob": h.prob, "best_american": h.best_american})
    return out
