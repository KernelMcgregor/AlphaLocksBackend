"""Walk-forward "judge appeal" per fighter, for the winner model.

judge appeal = the fighter effect from scripts.round_effects: how many more rounds judges
give this fighter than their round stats say (log-odds per round), as a shrunk random
effect. Leakage-free by construction: at each quarterly cutoff the boosted round model
AND the fighter effects are refitted on judged rounds strictly before the cutoff, and
fights in [cutoff, next cutoff) get the fighters' values from that fit. A fighter with no
judged rounds before the cutoff gets 0 (the prior mean).

The shrinkage strength is tuned once, on rounds before --tune-before (default: the
fast_wf eval window start), so nothing from the evaluated period picks it.

    DATABASE_URL=postgresql://localhost/alocks_local PYTHONPATH=. \
        venv/bin/python -m scripts.judge_appeal_feature --out data/mmad/judge_appeal_wf.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from scripts.round_effects import _ll, fit_effects, load, stats_offsets

log = logging.getLogger("judge_appeal")


def fit_at(rounds: pd.DataFrame, s_f: float) -> pd.Series:
    """fighter_id -> effect (log-odds per round) from all rounds given."""
    off, _ = stats_offsets(rounds, rounds.iloc[:1], "hgb")
    empty = np.zeros((len(rounds), 0))
    _, m, d = fit_effects(rounds, rounds.iloc[:1], off, off[:1], empty, empty[:1],
                          s_f, 0, 0, return_model=True)
    return pd.Series(m.coef_[0][1:1 + len(d.fighters)] * s_f, index=d.fighters)


def tune(rounds: pd.DataFrame, before: dt.date) -> float:
    """Pick the shrinkage on rounds before `before`: train on the earlier 75% of that
    period, score the rest."""
    pre = rounds[rounds.date < before]
    cut = pre.date.sort_values().iloc[int(len(pre) * 0.75)]
    tr, va = pre[pre.date < cut], pre[pre.date >= cut]
    o_tr, o_va = stats_offsets(tr, va, "hgb")
    e_tr, e_va = np.zeros((len(tr), 0)), np.zeros((len(va), 0))
    scores = {s: _ll(va.y, fit_effects(tr, va, o_tr, o_va, e_tr, e_va, s, 0, 0))
              for s in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)}
    log.info("tuning (train < %s, validate %s .. %s): %s", cut, cut, before,
             "  ".join(f"{k}:{v:.4f}" for k, v in scores.items()))
    return min(scores, key=scores.get)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-before", type=dt.date.fromisoformat, default=dt.date(2022, 6, 18))
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2009, 1, 1))
    ap.add_argument("--months", type=int, default=3, help="refit interval")
    ap.add_argument("--out", type=Path, default=Path("data/mmad/judge_appeal_wf.csv"))
    args = ap.parse_args(argv)
    eng = create_engine(os.environ["DATABASE_URL"])

    rounds = load(eng)
    s_f = tune(rounds, args.tune_before)
    log.info("shrinkage s_f = %s", s_f)

    fights = pd.read_sql(text("""
        SELECT f.id AS fight_id, COALESCE(f.date, e.date) AS date,
               f.red_fighter_id AS red, f.blue_fighter_id AS blue
        FROM ufc.ufc_fights f JOIN ufc.ufc_events e ON e.id = f.event_id
    """), eng)
    fights["date"] = pd.to_datetime(fights.date).dt.date
    fights = fights[fights.date >= args.start]
    judged = pd.concat([rounds[["date", "red"]].rename(columns={"red": "f"}),
                        rounds[["date", "blue"]].rename(columns={"blue": "f"})])

    cutoffs = pd.date_range(args.start, dt.date.today() + dt.timedelta(days=92),
                            freq=f"{args.months}MS").date
    out = []
    for c0, c1 in zip(cutoffs[:-1], cutoffs[1:]):
        batch = fights[(fights.date >= c0) & (fights.date < c1)]
        if batch.empty:
            continue
        past = rounds[rounds.date < c0]
        eff = fit_at(past, s_f) if len(past) > 500 else pd.Series(dtype=float)
        n_rounds = judged[judged.date < c0].f.value_counts()
        for side in ("red", "blue"):
            out.append(pd.DataFrame({
                "fight_id": batch.fight_id.values, "corner": side,
                "fighter_id": batch[side].values,
                "judge_appeal": batch[side].map(eff).fillna(0.0).values,
                "judged_rounds": batch[side].map(n_rounds).fillna(0).astype(int).values,
                "fit_cutoff": c0,
            }))
        log.info("cutoff %s: %d past rounds, %d fighters with effects, %d fights scored",
                 c0, len(past), len(eff), len(batch))
    res = pd.concat(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    log.info("wrote %d rows -> %s", len(res), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
