"""Build models/ufc/backtest_ledger.csv.gz: the Past Model Results page's data.

One row per priced side of every market on every backtested fight (walk-forward OOF, see
app/services/ufc/backtest_ledger.py), with `bet` = EV > 0 at the typical book's price and
`is_top` = the grade table's one-bet-per-market rule. Fight, event and result metadata are
baked in so the API serves the page without a database read. Reads the DB only; runs in the
post-event workflow next to build_grade_table.

Usage:
    DATABASE_URL=... python -m scripts.build_backtest_ledger
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine, text

from app.config import settings
from app.services.ufc.backtest_ledger import LEDGER_PATH, build
from app.services.ufc.method_ratings import method_class

log = logging.getLogger("build_backtest_ledger")


def fight_meta(engine) -> pd.DataFrame:
    with engine.connect() as c:
        rows = c.execute(text("""
            select f.id::text, e.id, e.name, e.date,
                   trim(concat_ws(' ', r.first_name, r.last_name)), trim(concat_ws(' ', b.first_name, b.last_name)), f.weight_class, f.card_position,
                   f.max_fight_time_seconds, f.method, f.details, f.finish_round, f.finish_time,
                   case when f.winner_id = f.red_fighter_id then 'red'
                        when f.winner_id = f.blue_fighter_id then 'blue' end
            from ufc.ufc_fights f
            join ufc.ufc_events e on e.id = f.event_id
            join ufc.ufc_fighters r on r.id = f.red_fighter_id
            join ufc.ufc_fighters b on b.id = f.blue_fighter_id""")).all()
    out = []
    for (fid, eid, ename, edate, red, blue, wc, pos, sched, method, details, rnd, ftime, winner) in rows:
        out.append({"fight_id": fid, "event_id": eid, "event_name": ename, "date": edate.isoformat(),
                    "red_name": red, "blue_name": blue, "weight_class": wc or "", "card_position": pos,
                    "rounds": 5 if (sched or 0) >= 1500 else 3, "winner_side": winner,
                    "result_cls": method_class(method, details, None if winner is None else 1) or "",
                    "result_method": method or "", "finish_round": rnd, "finish_time": ftime or ""})
    return pd.DataFrame(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    engine = create_engine(settings.DATABASE_URL)
    led = build(engine, keep_all=True, extended=True)
    led = led.merge(fight_meta(engine), on="fight_id", how="inner")
    for c in ("p_raw", "p", "q", "ev", "decimal", "decimal_best", "profit_typical", "profit_best", "clv"):
        if c in led:
            led[c] = led[c].astype(float).round(5)
    led = led.sort_values(["date", "event_id", "card_position", "fight_id", "family", "market"])
    # mtime 0: an unchanged ledger is byte-identical, so the post-event commit skips it
    led.to_csv(LEDGER_PATH, index=False, compression={"method": "gzip", "mtime": 0})
    bets = led[led["bet"]]
    log.info(f"wrote {LEDGER_PATH.name}: {len(led)} rows, {len(bets)} bets over "
             f"{led['fight_id'].nunique()} fights / {led['event_id'].nunique()} events "
             f"({led['date'].min()} .. {led['date'].max()})")


if __name__ == "__main__":
    main()
