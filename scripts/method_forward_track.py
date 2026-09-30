"""Forward test of the method model (method_v2): every fight logged BEFORE it happens.

Accuracy only for now: no betting rule is registered. Rows are append-only and never
re-priced; settlement fills in the result and closing prices only.

When a fight is logged: the first time the line watcher has recorded BestFightOdds prop
prices for it (its OPENING prop line, ufc_prop_odds_history source 'bfo_watch'). If no
props have appeared by the day before the card, it is logged without them so the model's
prediction is still frozen before the fight.

Logged: the six winner x method cells, KO/Sub/Dec marginals and goes-the-distance as
served (ufc_method_predictions, i.e. combined with the served winner probability), the
model version, and the opening prop prices.
Settled: the result (winner corner + method class), the closing prop prices (the
watcher's last snapshot before the card), and Polymarket's closing KO / distance prices.

Usage:
    python -m scripts.method_forward_track --log
    python -m scripts.method_forward_track --settle
    python -m scripts.method_forward_track --report
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.database import SessionLocal
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter, UFCMethodPrediction, UFCPredictionMarket,
    UFCPredictionMarketHistory, UFCPropOddsHistory,
)

LOG_PATH = Path(os.environ.get("METHOD_FORWARD_LOG_PATH")
                or Path(__file__).resolve().parents[1] / "forward_log_method_v1.jsonl")
LOG_VERSION = "method-v1-2026-09-30"
CELLS = ("red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec")
PROP_MARKETS = CELLS + ("dec_yes", "ou_1.5_over", "ou_2.5_over")


def _read() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]


def _write(entries: list[dict]) -> None:
    LOG_PATH.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))


def _props(db, fight_id: int, first: bool, before: datetime | None = None) -> dict:
    q = db.query(UFCPropOddsHistory).filter(UFCPropOddsHistory.fight_id == fight_id,
                                            UFCPropOddsHistory.source == "bfo_watch")
    if before is not None:
        q = q.filter(UFCPropOddsHistory.captured_at < before)
    out, ts = {}, None
    for h in q.order_by(UFCPropOddsHistory.captured_at.asc() if first
                        else UFCPropOddsHistory.captured_at.desc()):
        if h.market not in out:
            out[h.market] = h.prob
            ts = ts or h.captured_at
    return {"prices": out, "captured_at": ts.isoformat() if ts else None} if out else {}


def _model_version() -> str | None:
    from app.services.ufc import method_v2
    m = method_v2.load()
    return m.meta.get("trained_at") if m else None


def _joint_columns_ready() -> bool:
    """Migration 013 (winner x method columns) runs when the backend restarts; a CI run
    that gets there first logs nothing instead of failing."""
    from sqlalchemy import inspect

    from app.database import engine
    t = UFCMethodPrediction.__table__
    cols = {c["name"] for c in inspect(engine).get_columns(t.name, schema=t.schema)}
    return "red_ko_prob" in cols and inspect(engine).has_table(
        UFCPropOddsHistory.__table__.name, schema=UFCPropOddsHistory.__table__.schema)


def log_upcoming() -> int:
    if not _joint_columns_ready():
        print("method prediction grid columns / prop table missing (migration 013 pending); skipped")
        return 0
    entries = _read()
    logged = {e["fight_id"] for e in entries}
    today = datetime.now(timezone.utc).date()
    version = _model_version()
    db = SessionLocal()
    try:
        rows = (db.query(UFCFight, UFCEvent, UFCMethodPrediction)
                .join(UFCEvent, UFCEvent.id == UFCFight.event_id)
                .join(UFCMethodPrediction, UFCMethodPrediction.fight_id == UFCFight.id)
                .filter(UFCEvent.date >= today, UFCFight.winner_id.is_(None),
                        UFCMethodPrediction.red_ko_prob.isnot(None)).all())
        names = {f.id: f for f in db.query(UFCFighter).filter(UFCFighter.id.in_(
            {i for f, _, _ in rows for i in (f.red_fighter_id, f.blue_fighter_id)}))}
        new = []
        for f, e, mp in rows:
            if str(f.id) in logged:
                continue
            opening = _props(db, f.id, first=True)
            if not opening and e.date > today + timedelta(days=1):
                continue  # wait for the opening prop line
            nm = lambda i: f"{names[i].first_name} {names[i].last_name}".strip() if i in names else "?"
            new.append({
                "log_version": LOG_VERSION, "model_version": version,
                "fight_id": str(f.id), "event": e.name, "event_date": e.date.isoformat(),
                "red": nm(f.red_fighter_id), "blue": nm(f.blue_fighter_id),
                "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": {c: getattr(mp, f"{c}_prob") for c in CELLS}
                         | {"ko": mp.ko_prob, "sub": mp.sub_prob, "dec": mp.dec_prob,
                            "distance": mp.distance_prob},
                "open": opening or None,
                "result": None,
            })
    finally:
        db.close()
    if new:
        _write(entries + new)
    print(f"logged {len(new)} fights ({sum(1 for n in new if n['open'])} at their opening prop line)")
    return len(new)


def _pm_close(db, fight_id: int, key: str) -> float | None:
    m = (db.query(UFCPredictionMarket).filter(UFCPredictionMarket.fight_id == fight_id,
                                             UFCPredictionMarket.platform == "polymarket",
                                             UFCPredictionMarket.outcome_key == key).first())
    if not m:
        return None
    h = (db.query(UFCPredictionMarketHistory)
         .filter(UFCPredictionMarketHistory.market_id == m.id,
                 UFCPredictionMarketHistory.days_to_fight >= 0)
         .order_by(UFCPredictionMarketHistory.captured_at.desc()).first())
    return h.price if h else None


def settle() -> int:
    from app.services.ufc.method_ratings import method_class
    entries = _read()
    db = SessionLocal()
    n = 0
    try:
        for e in entries:
            if e.get("result"):
                continue
            f = db.get(UFCFight, int(e["fight_id"]))
            if f is None or not f.method:
                continue
            k = method_class(f.method, f.details, f.winner_id)
            e["result"] = {"method": f.method, "class": k,
                           "winner": ("red" if f.winner_id == f.red_fighter_id else
                                      "blue" if f.winner_id == f.blue_fighter_id else None)}
            cutoff = datetime.combine(date.fromisoformat(e["event_date"]) + timedelta(days=1),
                                      datetime.min.time())
            e["close"] = _props(db, f.id, first=False, before=cutoff) or None
            e["pm_close"] = {k2: _pm_close(db, f.id, k2) for k2 in ("ko_tko", "distance")}
            n += 1
    finally:
        db.close()
    if n:
        _write(entries)
    print(f"settled {n} fights")
    return n


def _ll(p):
    return float(-np.log(np.clip(p, 1e-6, 1)))


def report(entries: list[dict] | None = None) -> int:
    entries = entries if entries is not None else _read()
    done = [e for e in entries if e.get("result") and e["result"].get("class")]
    print(f"{len(entries)} logged, {len(done)} settled (draws/NC/DQ excluded from scoring)")
    if not done:
        return 0
    rows = {"model": [], "open": [], "close": []}
    for e in done:
        cell = f"{e['result']['winner']}_{e['result']['class']}"
        rows["model"].append(_ll(e["model"][cell]))
        for src in ("open", "close"):
            p = ((e.get(src) or {}).get("prices") or {})
            rows[src].append(_ll(p[cell]) if cell in p and all(c in p for c in CELLS) else np.nan)
    m = np.array(rows["model"])
    print(f"6-way log loss, model: {m.mean():.4f} (n={len(m)})")
    for src in ("open", "close"):
        k = np.array(rows[src])
        ok = ~np.isnan(k)
        if ok.sum():
            print(f"  vs props at {src}: model {m[ok].mean():.4f}  market {k[ok].mean():.4f}  "
                  f"(n={int(ok.sum())})")
    return len(done)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.log:
        log_upcoming()
    if a.settle:
        settle()
    if a.report:
        report()


if __name__ == "__main__":
    main()
