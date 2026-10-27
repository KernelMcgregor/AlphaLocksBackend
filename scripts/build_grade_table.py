"""Build models/ufc/grade_table.json from walk-forward backtests (see app/services/ufc/grading.py).

Bets are rebuilt from data in the repo + DB only, so this runs in CI after each event:
  probabilities  ensemble OOF (models/ufc/h2h/ensemble_oof.csv) with the ensemble's market
                 stacks; method_v2 OOF (models/ufc/method/method_oof.csv); rounds OOF
                 (models/ufc/method/rounds_oof.csv, anchored to the market's decision price when priced,
                  else method_v2's decision prob)
  prices         moneyline open/close: ufc_fight_odds_open_close (opening = the consensus
                 price; closing = best across sportsbooks); props: ufc_prop_odds_history
                 rows with source 'bfo_close' (best and median American price)
  results        ufc_fights

Rule (identical for every family): per fight and market, the side with the highest EV at the
typical book's price if EV > 0; flat 1 unit; ROI graded at that typical price. Prop
probabilities are the model blended 50/50 (log-odds) with the de-vigged closing consensus,
as the picks API serves them; moneyline probabilities are the ensemble's market stack.

Usage:
    DATABASE_URL=... python -m scripts.build_grade_table
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine

from app.config import settings
from app.services.ufc import backtest_ledger
from app.services.ufc.grading import family_table, save_table

log = logging.getLogger("build_grade_table")
ROOT = Path(__file__).resolve().parents[1]


def forward_record() -> dict:
    out = {}
    p = ROOT / "forward_log_v2.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        settled = [r for r in rows if r.get("bet") and r.get("profit") is not None]
        if settled:
            prof = np.array([r["profit"] for r in settled], float) / 100.0  # logged per $100 stake
            out["winner_open"] = {"n": len(settled), "roi": float(prof.mean())}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    engine = create_engine(settings.DATABASE_URL)
    # the bets themselves live in app/services/ufc/backtest_ledger.py (shared with the Past
    # Model Results ledger); grades use one bet per fight and market: the highest-EV side
    allb = backtest_ledger.build(engine)
    allb = allb[allb["is_top"]]
    families = {}
    for fam, g in allb.groupby("family"):
        families[fam] = family_table(g["ev"], g["profit_typical"], g["profit_best"], g["decimal"])
        f = families[fam]
        log.info(f"  {fam:14s} n={f['n']:5d} ROI {f['roi']:+6.1%} (exp {f['expected_roi']:+.1%}) | " + " ".join(
            f"[{b['lo']:.0%}+ n={b['n']} raw {'—' if b['roi'] is None else format(b['roi'], '+.0%')} "
            f"exp {b['expected_roi']:+.1%} {b['grade']}]" for b in f["buckets"]))
    table = save_table(families, forward_record(), {
        "ensemble_oof": "models/ufc/h2h/ensemble_oof.csv", "method_oof": "models/ufc/method/method_oof.csv",
        "rounds_oof": "models/ufc/method/rounds_oof.csv", "props": "ufc_prop_odds_history (bfo_close)",
        "moneyline": "ufc_fight_odds_open_close"})
    log.info(f"wrote grade table version {table['version']} ({len(families)} families)")


if __name__ == "__main__":
    main()
