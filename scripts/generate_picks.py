"""Apply the pre-registered rule to upcoming fights and append them to the audit log.

The log (`picks_log.jsonl`) is append-only and git-tracked. That is the entire point:
the rule's backtested ROI was obtained by searching the eval set, so it proves nothing.
What will eventually prove or disprove it is an immutable record of picks made BEFORE
the fights happened, which nobody can quietly revise afterwards.

Each entry records the odds and probabilities as they stood when the pick was made, so
settlement later needs only the winner. Re-running for the same event does not duplicate
or overwrite entries -- already-logged fights are skipped, so the first recorded opinion
is the one that counts.

Usage:
    python -m scripts.generate_picks --event "Noche UFC"          # log picks
    python -m scripts.generate_picks --upcoming --dry-run         # preview only
    python -m scripts.generate_picks --settle                     # fill in results
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from app.database import SessionLocal
from app.models.ufc import UFCEvent, UFCFight, UFCFightOdds, UFCFighter
from app.services.ufc.picks import (
    FLAT_STAKE, MARKET_UNCERTAINTY_MAX, MIN_EDGE, RULE_VERSION,
    load_picks_model, select_picks,
)

log = logging.getLogger("generate_picks")
LOG_PATH = Path(__file__).resolve().parents[1] / "picks_log.jsonl"

#: Books to average into a single consensus line, in preference order. Kept short on
#: purpose -- the multi-book consensus-outlier strategy is explicitly out of scope.
CONSENSUS_BOOKS = ("FanDuel", "DraftKings", "BetMGM", "Bovada", "BetRivers")


def _name(f) -> str:
    return f"{f.first_name} {f.last_name}".strip() if f else "?"


def _read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_log(entries: list[dict]) -> None:
    with open(LOG_PATH, "a") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True) + "\n")


def _consensus_odds(db, fight_id: int):
    """Average the American odds across available books, preferring CONSENSUS_BOOKS."""
    rows = db.query(UFCFightOdds).filter(UFCFightOdds.fight_id == fight_id).all()
    if not rows:
        return None, None
    preferred = [r for r in rows if r.bookmaker in CONSENSUS_BOOKS] or rows
    return (float(np.mean([r.red_odds for r in preferred])),
            float(np.mean([r.blue_odds for r in preferred])))


def _build_probs(db, fights: list[UFCFight]) -> dict[int, float]:
    """Model probability that RED wins, for each fight, from the frozen picks model."""
    from app.services.ufc.model import (
        build_features, build_serving_matchup, load_fight_data,
    )

    saved = load_picks_model()
    model, feats, means = saved["model"], saved["features"], saved["train_means"]
    if saved.get("include_odds"):
        raise RuntimeError("picks model was trained with odds features; refusing to use it")

    # The SAME serving frame the site's predictions use. build_matchup_df() would be
    # wrong here: it keeps only decided fights, so every upcoming bout — the only ones
    # a pick can be made on — would be silently dropped.
    df, rd = load_fight_data()
    df = build_features(df, rd)
    matchup = build_serving_matchup(df)

    for c in feats:
        if c not in matchup.columns:
            matchup[c] = np.nan
    matchup[feats] = matchup[feats].fillna(means).fillna(0.0)

    want = {f.id for f in fights}
    sub = matchup[matchup.index.isin(want)]
    if sub.empty:
        return {}
    p = model.predict_proba(sub[feats].to_numpy(dtype=float))[:, 1]
    return dict(zip(sub.index.astype(int), p))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="", help="substring of the event name")
    ap.add_argument("--upcoming", action="store_true", help="every future event")
    ap.add_argument("--dry-run", action="store_true", help="print, do not log")
    ap.add_argument("--settle", action="store_true", help="fill results into the log")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        if args.settle:
            return _settle(db)

        q = db.query(UFCFight).join(UFCEvent, UFCEvent.id == UFCFight.event_id)
        if args.event:
            q = q.filter(UFCEvent.name.ilike(f"%{args.event}%"))
        elif args.upcoming:
            q = q.filter(UFCEvent.date >= date.today())
        else:
            ap.error("pass --event or --upcoming")
        fights = q.all()
        if not fights:
            log.warning("no fights matched")
            return 1

        ev = {e.id: e for e in db.query(UFCEvent).all()}
        fr = {f.id: f for f in db.query(UFCFighter).all()}

        # A pick is only evidence if it was made before the fight. The picks model trains
        # on every decided fight, so "picking" a settled bout is retrodiction against its
        # own training data -- it looks spectacular and means nothing. Selecting by name
        # makes this easy to do by accident: `--event Noche` matches Noche UFC 2024 as
        # readily as the 2026 card, and that mistake silently poisons the audit log.
        today = date.today()
        eligible, rejected = [], []
        for f in fights:
            e_ = ev.get(f.event_id)
            (rejected if (f.winner_id is not None
                          or (e_ and e_.date and e_.date <= today)) else eligible).append(f)
        if rejected:
            names = sorted({ev[f.event_id].name for f in rejected if f.event_id in ev})
            log.warning(
                "Skipping %d already-decided/past fight(s) — a pick on a settled fight "
                "is retrodiction, not a forecast: %s",
                len(rejected), ", ".join(names[:4]) + ("..." if len(names) > 4 else ""))
        fights = eligible
        if not fights:
            log.error("no undecided future fights matched")
            return 1

        probs = _build_probs(db, fights)

        ids, pm, ra, ba, keep = [], [], [], [], []
        for f in fights:
            if f.id not in probs:
                continue
            r, b = _consensus_odds(db, f.id)
            if r is None:
                continue
            ids.append(f.id); pm.append(probs[f.id]); ra.append(r); ba.append(b); keep.append(f)

        print(f"\nRule {RULE_VERSION}: |market-0.5| < {MARKET_UNCERTAINTY_MAX}, "
              f"edge > {MIN_EDGE}, flat ${FLAT_STAKE:.0f}")
        print(f"Evaluated {len(ids)} priced fights\n")

        picks = select_picks(np.array(ids), np.array(pm), np.array(ra), np.array(ba))
        by_id = {f.id: f for f in keep}
        already = {e["fight_id"] for e in _read_log()}

        entries = []
        for p in picks:
            f = by_id[p.fight_id]
            e_ = ev[f.event_id]
            rn, bn = _name(fr.get(f.red_fighter_id)), _name(fr.get(f.blue_fighter_id))
            name = rn if p.side == "red" else bn
            dup = " [already logged]" if p.fight_id in already else ""
            print(f"  {e_.date}  {rn} vs {bn}")
            print(f"     PICK {name} ({p.side})  model {p.model_prob:.1%} vs market "
                  f"{p.market_prob:.1%}  edge {p.edge:+.1%}  @ {p.decimal_odds:.2f}{dup}")
            if p.fight_id in already:
                continue
            entries.append({
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "rule_version": RULE_VERSION,
                "fight_id": p.fight_id, "event": e_.name, "event_date": str(e_.date),
                "red_fighter": rn, "blue_fighter": bn,
                "pick_side": p.side, "pick_fighter": name,
                "model_prob": round(p.model_prob, 4),
                "market_prob": round(p.market_prob, 4),
                "edge": round(p.edge, 4),
                "market_uncertainty": round(p.market_uncertainty, 4),
                "decimal_odds": round(p.decimal_odds, 3),
                "stake": FLAT_STAKE,
                "result": None, "profit": None,
            })

        if not picks:
            print("  (no fights qualified)")
        print(f"\n{len(picks)} pick(s); {len(entries)} new")

        if entries and not args.dry_run:
            _append_log(entries)
            print(f"Appended to {LOG_PATH}")
        elif args.dry_run:
            print("Dry run — nothing written.")
        return 0
    finally:
        db.close()


def _settle(db) -> int:
    """Fill in results for logged picks whose fights have since been decided."""
    entries = _read_log()
    if not entries:
        print("empty log")
        return 0
    fights = {f.id: f for f in db.query(UFCFight).all()}
    changed = 0
    for e in entries:
        if e.get("result") is not None:
            continue
        f = fights.get(e["fight_id"])
        if f is None or f.winner_id is None:
            continue
        won = ((f.winner_id == f.red_fighter_id) if e["pick_side"] == "red"
               else (f.winner_id == f.blue_fighter_id))
        e["result"] = "win" if won else "loss"
        e["profit"] = round(e["stake"] * (e["decimal_odds"] - 1) if won else -e["stake"], 2)
        changed += 1

    if changed:
        with open(LOG_PATH, "w") as fh:
            for e in entries:
                fh.write(json.dumps(e, sort_keys=True) + "\n")

    done = [e for e in entries if e.get("result")]
    print(f"Settled {changed} new; {len(done)}/{len(entries)} total settled")
    if done:
        staked = sum(e["stake"] for e in done)
        profit = sum(e["profit"] for e in done)
        wins = sum(1 for e in done if e["result"] == "win")
        print(f"  record {wins}-{len(done) - wins}  ({wins / len(done):.1%})")
        print(f"  staked ${staked:,.0f}  profit ${profit:+,.2f}  ROI {profit / staked:+.2%}")
        print("\nNOTE: the backtest CI was +/-11 points at n=321. Below ~150 settled")
        print("picks this number is noise and should not be acted on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
