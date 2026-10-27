"""Append-only record of out-of-sample predictions for fights after the walk-forward window.

The backtest ledger (backtest_ledger.py) prices bets off walk-forward OOF files, which are
regenerated only when a model is retrained by hand -- so on their own they stop at the last
retrain. This file carries every completed fight past them, one row per fight and model
(winner / method / rounds), from the predictions the site stored:

  live      the row was written before the bout's cutoff and frozen there
            (prediction_freeze.py): exactly what the site served going in
  forward   not frozen, but produced by a saved model whose training data ends before the
            fight's date, so it is out-of-sample all the same (e.g. the method and rounds
            models trained through 2026-09-26, serving October's cards)

A stored row that is neither -- a model trained on the fight scoring it afterwards -- is
skipped. Rows are never rewritten: a later retrain cannot change what was recorded. Written
by the post-event workflow before scripts/build_backtest_ledger.py.

Values stored per model (the shape of the matching OOF file):
  winner  model_prob   the model alone, before the market blend (ensemble_oof.model_prob)
  method  red_ko..blue_dec, CONDITIONAL on the corner winning (method_oof.csv); the stored
          six-way cells are p_red x conditional, so dividing by each corner's sum recovers it
  rounds  S_0..S_25    still-going probability at 2.5-minute edges (rounds_oof.csv), read
          off the stored curve. That curve is anchored to the served six-way cells; the
          ledger re-anchors its finish mass to the market's decision price where priced.

Usage:
    python -m app.services.ufc.forward_predictions --append
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("forward_predictions")
ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "models" / "ufc" / "forward_predictions.csv"
OOF = {"winner": ROOT / "models/ufc/h2h/ensemble_oof.csv",
       "method": ROOT / "models/ufc/method/method_oof.csv",
       "rounds": ROOT / "models/ufc/method/rounds_oof.csv"}
CELLS = ("red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec")
EDGES = np.arange(0, 25.01, 2.5)
S_COLS = [f"S_{e:g}" for e in EDGES]
COLUMNS = ["fight_id", "model", "source", "event_date", "recorded_at", "model_prob", *CELLS, *S_COLS]


def read() -> pd.DataFrame:
    if not PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(PATH, dtype={"fight_id": str})


def s_grid(curve: list[dict]) -> list[float]:
    """Still-going at 0, 2.5, ..., 25 min from a stored curve ({t, s} every 1.25 min). A
    three-round curve ends at 15: past the final bell nothing changes, so it carries flat."""
    t = np.array([p["t"] for p in curve], float)
    s = np.array([p["s"] for p in curve], float)
    return [round(float(np.interp(e, t, s)), 5) for e in EDGES]


def conditional(cells: list[float]) -> list[float] | None:
    c = np.array(cells, float)
    if np.isnan(c).any():
        return None
    red, blue = c[:3].sum(), c[3:].sum()
    if red <= 0 or blue <= 0:
        return None
    return [round(float(x), 5) for x in np.r_[c[:3] / red, c[3:] / blue]]


def _last_date(path: Path, db) -> str:
    """Date of the latest fight in an OOF file (not every file carries a date column)."""
    from sqlalchemy import text
    ids = pd.read_csv(path, usecols=["fight_id"], dtype={"fight_id": str})["fight_id"].tail(200).tolist()
    d = db.execute(text("""select max(e.date) from ufc.ufc_fights f join ufc.ufc_events e on e.id = f.event_id
                           where f.id::text = any(:ids)"""), {"ids": ids}).scalar()
    return d.isoformat() if d else "1990-01-01"


def _trained_through() -> dict[str, dt.date | None]:
    """Last fight date in each saved model's training data (None = unknown, never 'forward')."""
    out: dict[str, dt.date | None] = {"winner": None, "method": None, "rounds": None}
    try:
        from app.services.ufc import ensemble
        e = ensemble.load()
        out["winner"] = dt.date.fromisoformat(str(e.meta["last_fight"])[:10]) if e else None
    except Exception:  # noqa: BLE001 -- a model that can't be read just isn't used
        log.exception("  winner model meta unreadable")
    meta = ROOT / "models/ufc/method/method_v2_meta.json"
    if meta.exists():
        out["method"] = dt.date.fromisoformat(json.loads(meta.read_text())["last_fight"][:10])
    try:
        from app.services.ufc import rounds_v1
        m = rounds_v1.load()
        out["rounds"] = dt.date.fromisoformat(str(m.meta["last_fight"])[:10]) if m else None
    except Exception:  # noqa: BLE001
        log.exception("  rounds model meta unreadable")
    return out


def append(db) -> int:
    from app.models.ufc import (
        UFCEvent, UFCFight, UFCFightPrediction, UFCMethodPrediction, UFCRoundPrediction,
    )
    from app.services.ufc.prediction_freeze import frozen_fight_ids

    have = read()
    done = set(zip(have["fight_id"].astype(str), have["model"]))
    in_oof = {k: set(pd.read_csv(p, usecols=["fight_id"], dtype={"fight_id": str})["fight_id"])
              if p.exists() else set() for k, p in OOF.items()}
    through = _trained_through()
    log.info(f"  models trained through {through}")
    # older fights are all in the OOF files; the bound (well before the earliest OOF file
    # ends) just keeps the scan small
    oof_end = min(_last_date(p, db) for p in OOF.values() if p.exists())
    start = dt.date.fromisoformat(oof_end[:10]) - dt.timedelta(days=60)
    fights = {str(fid): date for fid, date in (
        db.query(UFCFight.id, UFCEvent.date).join(UFCEvent, UFCEvent.id == UFCFight.event_id)
        .filter(UFCFight.method.isnot(None), UFCFight.method != "", UFCEvent.date >= start))}
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    rows = []
    for model, table in (("winner", UFCFightPrediction), ("method", UFCMethodPrediction),
                         ("rounds", UFCRoundPrediction)):
        frozen = {str(f) for f in frozen_fight_ids(db, table)}
        todo = [f for f in fights if f not in in_oof[model] and (f, model) not in done]
        if not todo:
            continue
        stored = {str(r.fight_id): r for r in db.query(table).filter(table.fight_id.in_([int(f) for f in todo]))}
        for fid in todo:
            r = stored.get(fid)
            if r is None:
                continue
            if fid in frozen:
                source = "live"
            elif through[model] is not None and through[model] < fights[fid]:
                source = "forward"
            else:
                continue          # scored by a model that had seen the fight
            row = {"fight_id": fid, "model": model, "source": source, "event_date": fights[fid].isoformat(),
                   "recorded_at": now}
            if model == "winner":
                if r.model_prob is None:
                    continue
                row["model_prob"] = round(float(r.model_prob), 5)
            elif model == "method":
                cond = conditional([getattr(r, f"{c}_prob") for c in CELLS])
                if cond is None:
                    continue
                row.update(dict(zip(CELLS, cond)))
            else:
                try:
                    row.update(dict(zip(S_COLS, s_grid(json.loads(r.curve)))))
                except (TypeError, ValueError, KeyError):
                    continue
            rows.append(row)
    if rows:
        out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).reindex(columns=COLUMNS)
        out = out.sort_values(["event_date", "fight_id", "model"])
        out.to_csv(PATH, index=False)
    by = pd.DataFrame(rows).groupby(["model", "source"]).size().to_dict() if rows else {}
    log.info(f"  appended {len(rows)} rows {by}; file has {len(have) + len(rows)}")
    return len(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    if a.append:
        from app.database import SessionLocal
        db = SessionLocal()
        try:
            append(db)
        finally:
            db.close()


if __name__ == "__main__":
    main()
