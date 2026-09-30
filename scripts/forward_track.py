"""Forward test of the ensemble: every fight logged BEFORE it happens, settled after.

Pre-registered in PREREGISTRATION_V2.md. This file implements it; do not change the rule
constants below without opening a new log (see that document).

Two things are measured on fights the model has never seen:

1. Accuracy, on EVERY priced fight: log loss of the model alone, of the market blend,
   of the market at logging time and of the market close. This converges far faster
   than betting results and answers "is the model getting closer to Vegas".
2. The betting rule: bet the side whose blended probability gives positive expected value
   at the logged consensus price, flat stakes. Graded on profit AND closing line value
   (CLV: did the price we logged beat the price at the close).

Each fight is logged once, on the first run that finds it priced within LOG_WINDOW_DAYS
of the event, and that row is never re-priced. Settlement only fills result columns.

Usage:
    python -m scripts.forward_track --log      # log upcoming priced fights
    python -m scripts.forward_track --settle   # fill results + closing lines
    python -m scripts.forward_track --report   # summary so far
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
    UFCEvent, UFCFight, UFCFighter, UFCFightOdds, UFCFightOddsHistory,
    UFCPredictionMarket, UFCPredictionMarketHistory,
)
from app.services.ufc.fighter_registry import is_decided
from app.services.ufc.market_anchor import american_to_prob, consensus_american, goto_devig

LOG_PATH = Path(os.environ.get("FORWARD_LOG_PATH")
                or Path(__file__).resolve().parents[1] / "forward_log_v2.jsonl")

# ---- Pre-registered rule (PREREGISTRATION_V2.md). Frozen. ----
RULE_VERSION = "v2.0-2026-09-28"
MIN_EV = 0.0          # bet when expected value per $1 at the logged price exceeds this
STAKE = 100.0         # flat
LOG_WINDOW_DAYS = 7   # log a fight once it is priced and within this many days
BOOKS = ("FanDuel", "DraftKings", "BetMGM", "Bovada", "BetRivers", "Caesars")


def _read() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]


def _write(entries: list[dict]) -> None:
    LOG_PATH.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))


def _decimal(american: float) -> float:
    return 1 + (american / 100 if american > 0 else 100 / abs(american))


def _consensus(rows) -> tuple[float, float] | None:
    rows = [r for r in rows if r.red_odds and r.blue_odds]
    preferred = [r for r in rows if r.bookmaker in BOOKS] or rows
    if not preferred:
        return None
    return (consensus_american([r.red_odds for r in preferred]),
            consensus_american([r.blue_odds for r in preferred]))


def _name(f) -> str:
    return f"{f.first_name} {f.last_name}".strip() if f else "?"


def _short_notice_tags(db, fights, dates) -> dict[int, dict]:
    """Per fight: which corner (if any) is a short-notice replacement, and whose opponent
    changed, from ufc_cancelled_bouts (same logic as short_notice.py)."""
    import pandas as pd
    from sqlalchemy import inspect

    from app.database import engine
    from app.models.ufc import UFCCancelledBout
    from app.services.ufc.short_notice import features_from_rows

    t = UFCCancelledBout.__table__
    if not fights or not inspect(engine).has_table(t.name, schema=t.schema):
        return {}
    pulled = db.query(UFCCancelledBout.event_date, UFCCancelledBout.red_fighter_id,
                      UFCCancelledBout.blue_fighter_id).all()
    base = pd.DataFrame({"fight_id": [f.id for f in fights],
                         "date": [dates[f.id] for f in fights],
                         "red_fighter_id": [f.red_fighter_id for f in fights],
                         "blue_fighter_id": [f.blue_fighter_id for f in fights]})
    red = features_from_rows(base.assign(stats_fighter_id=base["red_fighter_id"]), pulled)
    blue = features_from_rows(base.assign(stats_fighter_id=base["blue_fighter_id"]), pulled)
    return {fid: {"red_replacement": bool(red.iloc[i]["sn_replacement"]),
                  "blue_replacement": bool(blue.iloc[i]["sn_replacement"]),
                  "red_opp_changed": bool(red.iloc[i]["sn_opp_changed"]),
                  "blue_opp_changed": bool(blue.iloc[i]["sn_opp_changed"])}
            for i, fid in enumerate(base["fight_id"])}


def log_upcoming() -> int:
    from app.services.ufc import ensemble as ens_mod
    from app.services.ufc.model import build_features, build_serving_matchup, load_fight_data

    ens = ens_mod.load()
    if ens is None:
        print("No ensemble artifact; nothing logged.")
        return 1
    db = SessionLocal()
    try:
        today = date.today()
        fights = (db.query(UFCFight, UFCEvent)
                  .join(UFCEvent, UFCEvent.id == UFCFight.event_id)
                  .filter(UFCEvent.date > today,
                          UFCEvent.date <= today + timedelta(days=LOG_WINDOW_DAYS),
                          UFCFight.winner_id.is_(None))
                  .all())
        logged = {e["fight_id"] for e in _read()}
        todo = [(f, e) for f, e in fights if f.id not in logged]
        if not todo:
            print("Nothing new to log.")
            return 0

        df, rd = load_fight_data(include_upcoming=True)
        matchup = build_serving_matchup(build_features(df, rd))
        fighters = {f.id: f for f in db.query(UFCFighter).all()}
        tags = _short_notice_tags(db, [f for f, _ in todo], {f.id: e.date for f, e in todo})
        new = []
        for f, e in todo:
            if f.id not in matchup.index:
                continue
            cons = _consensus(db.query(UFCFightOdds).filter(UFCFightOdds.fight_id == f.id).all())
            if cons is None:
                continue  # not priced yet; a later run will log it
            red_am, blue_am = cons
            mkt = goto_devig(american_to_prob(red_am), american_to_prob(blue_am))
            row = matchup.loc[[f.id]]
            model_p = float(ens.model_prob(row)[0])
            final_p = float(ens.final_prob(np.array([model_p]), np.array([mkt]))[0])
            ev_red = final_p * _decimal(red_am) - 1
            ev_blue = (1 - final_p) * _decimal(blue_am) - 1
            side = "red" if ev_red >= ev_blue else "blue"
            ev = max(ev_red, ev_blue)
            new.append({
                "rule_version": RULE_VERSION,
                "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "fight_id": f.id, "event": e.name, "event_date": str(e.date),
                "red_fighter": _name(fighters.get(f.red_fighter_id)),
                "blue_fighter": _name(fighters.get(f.blue_fighter_id)),
                "model_prob_red": round(model_p, 4),
                "final_prob_red": round(final_p, 4),
                "market_prob_red": round(mkt, 4),
                "red_american": round(red_am, 1), "blue_american": round(blue_am, 1),
                "bet": ev > MIN_EV, "bet_side": side if ev > MIN_EV else None,
                "bet_ev": round(ev, 4), "stake": STAKE if ev > MIN_EV else 0.0,
                # Secondary hypothesis H2 (PREREGISTRATION_V2.md): short-notice tags.
                **tags.get(f.id, {}),
                # filled by --settle
                "red_won": None, "profit": None, "close_prob_red": None,
                "close_american_side": None, "clv": None, "pm_close_prob_red": None,
            })
        if new:
            with open(LOG_PATH, "a") as fh:
                for r in new:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"Logged {len(new)} fight(s); {sum(r['bet'] for r in new)} bet(s).")
        for r in new:
            tag = f"BET {r['bet_side']} (EV {r['bet_ev']:+.1%})" if r["bet"] else "no bet"
            print(f"  {r['event_date']} {r['red_fighter']} vs {r['blue_fighter']}: "
                  f"model {r['model_prob_red']:.1%} blend {r['final_prob_red']:.1%} "
                  f"market {r['market_prob_red']:.1%}  {tag}")
        return 0
    finally:
        db.close()


def _closing(db, fight: UFCFight, event_date: date):
    """Last sportsbook consensus captured before the card starts (red, blue American).

    Cut-off 22:00 UTC on the event date: after the fight-day 21:00 snapshot, before US
    prelims. Cards held elsewhere start earlier; their close is then the prior capture.
    """
    cutoff = datetime.combine(event_date, datetime.min.time()) + timedelta(hours=22)
    hist = (db.query(UFCFightOddsHistory)
            .filter(UFCFightOddsHistory.fight_id == fight.id,
                    UFCFightOddsHistory.captured_at < cutoff)
            .order_by(UFCFightOddsHistory.captured_at).all())
    latest = {}
    for h in hist:  # ascending, so the last capture per book wins
        latest[h.bookmaker] = h
    rows = list(latest.values()) or db.query(UFCFightOdds).filter(UFCFightOdds.fight_id == fight.id).all()
    return _consensus(rows)


def _pm_close(db, fight_id: int) -> float | None:
    """Prediction-market closing P(red): mean of Kalshi/Polymarket moneyline closes."""
    mkts = (db.query(UFCPredictionMarket)
            .filter(UFCPredictionMarket.fight_id == fight_id,
                    UFCPredictionMarket.market_type == "moneyline").all())
    vals = []
    for platform in {m.platform for m in mkts}:
        prices = {}
        for m in (m for m in mkts if m.platform == platform):
            last = (db.query(UFCPredictionMarketHistory)
                    .filter(UFCPredictionMarketHistory.market_id == m.id,
                            UFCPredictionMarketHistory.days_to_fight >= 0)
                    .order_by(UFCPredictionMarketHistory.captured_at.desc()).first())
            if last is not None and m.side in ("red", "blue"):
                prices[m.side] = last.price
        if "red" in prices and "blue" in prices and prices["red"] + prices["blue"] > 0:
            vals.append(prices["red"] / (prices["red"] + prices["blue"]))
    return float(np.mean(vals)) if vals else None


def settle() -> int:
    entries = _read()
    db = SessionLocal()
    try:
        changed = 0
        for e in entries:
            if e.get("red_won") is not None or e.get("void"):
                continue
            f = db.get(UFCFight, e["fight_id"])
            if f is None or (f.winner_id is None and not f.method):
                continue  # not fought yet
            if not is_decided(f.method, f.winner_id):
                e["void"] = True  # draw / NC / DQ: no result, stake returned
                e["profit"] = 0.0
                changed += 1
                continue
            red_won = f.winner_id == f.red_fighter_id
            e["red_won"] = bool(red_won)
            if e["bet"]:
                won = red_won if e["bet_side"] == "red" else not red_won
                am = e["red_american"] if e["bet_side"] == "red" else e["blue_american"]
                e["profit"] = round(STAKE * (_decimal(am) - 1) if won else -STAKE, 2)
            else:
                e["profit"] = 0.0
            close = _closing(db, f, date.fromisoformat(e["event_date"]))
            if close is not None:
                cr = goto_devig(american_to_prob(close[0]), american_to_prob(close[1]))
                e["close_prob_red"] = round(cr, 4)
                if e["bet"]:
                    # CLV: fair probability at the close vs the (vigged) price we took.
                    took = e["red_american"] if e["bet_side"] == "red" else e["blue_american"]
                    fair_close = cr if e["bet_side"] == "red" else 1 - cr
                    e["close_american_side"] = round(close[0] if e["bet_side"] == "red" else close[1], 1)
                    e["clv"] = round(fair_close * _decimal(took) - 1, 4)
            e["pm_close_prob_red"] = _pm_close(db, f.id)
            changed += 1
        if changed:
            _write(entries)
        print(f"Settled {changed} fight(s).")
        return report(entries)
    finally:
        db.close()


def report(entries: list[dict] | None = None) -> int:
    entries = entries if entries is not None else _read()
    done = [e for e in entries if e.get("red_won") is not None]
    print(f"\nForward log {RULE_VERSION}: {len(entries)} logged, {len(done)} settled")
    if not done:
        return 0
    y = np.array([float(e["red_won"]) for e in done])

    def ll(key):
        vals = [(e[key], yy) for e, yy in zip(done, y) if e.get(key) is not None]
        if not vals:
            return None, 0
        p = np.clip(np.array([v[0] for v in vals]), 1e-6, 1 - 1e-6)
        t = np.array([v[1] for v in vals])
        return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()), len(vals)

    print("  log loss (lower is better):")
    for k, label in (("model_prob_red", "model alone"), ("final_prob_red", "model+market blend"),
                     ("market_prob_red", "market at log time"), ("close_prob_red", "market close"),
                     ("pm_close_prob_red", "prediction-mkt close")):
        v, n = ll(k)
        if v is not None:
            print(f"    {label:22s} {v:.4f}  (n={n})")
    bets = [e for e in done if e["bet"]]
    if bets:
        staked = STAKE * len(bets)
        profit = sum(e["profit"] for e in bets)
        wins = sum(1 for e in bets if e["profit"] > 0)
        clv = [e["clv"] for e in bets if e.get("clv") is not None]
        print(f"  bets: {len(bets)}  record {wins}-{len(bets) - wins}  "
              f"profit ${profit:+,.0f}  ROI {profit / staked:+.1%}")
        if clv:
            se = np.std(clv, ddof=1) / np.sqrt(len(clv)) if len(clv) > 1 else float("nan")
            print(f"  mean CLV {np.mean(clv):+.2%} (±{1.96 * se:.2%}, n={len(clv)})")
    _report_fade_replacements(done)
    return 0


def _report_fade_replacements(done: list[dict]) -> None:
    """H2: bet AGAINST the short-notice replacement at the logged price (flat stakes)."""
    pnl, clv = [], []
    for e in done:
        rr, br = e.get("red_replacement"), e.get("blue_replacement")
        if bool(rr) == bool(br):
            continue  # no replacement, or both (can't fade either)
        side = "blue" if rr else "red"
        am = e["red_american"] if side == "red" else e["blue_american"]
        won = e["red_won"] if side == "red" else not e["red_won"]
        pnl.append(_decimal(am) - 1 if won else -1.0)
        if e.get("close_prob_red") is not None:
            fair = e["close_prob_red"] if side == "red" else 1 - e["close_prob_red"]
            clv.append(fair * _decimal(am) - 1)
    if not pnl:
        return
    line = f"  H2 fade the replacement: {len(pnl)} fights  ROI {np.mean(pnl):+.1%}"
    if clv:
        line += f"  mean CLV {np.mean(clv):+.2%} (n={len(clv)})"
    print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    rc = 0
    if a.settle:
        rc |= settle()
    if a.log:
        rc |= log_upcoming()
    if a.report:
        rc |= report()
    raise SystemExit(rc)
