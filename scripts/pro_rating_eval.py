"""Bake-off of whole-career Sherdog ratings (app/services/ufc/pro_rating.py). Read-only.

Every variant is a single date-ordered pass; each bout is predicted from pre-bout state.
Parameters are tuned on 2008-2014 bouts, scored on 2015+:
  all       every Sherdog pro bout from 2015
  ufc_new   UFC bouts where either fighter has at most 1 prior UFC bout (the newcomer
            fights where the winner model loses most to the market)

    PYTHONPATH=. venv/bin/python -m scripts.pro_rating_eval
"""
from __future__ import annotations

import itertools
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.ufc.pro_rating import Params, load_bouts, run

CACHE = Path(__file__).resolve().parents[1] / "data" / "feature_wf" / "sherdog_bouts.pkl"
TUNE = (date(2008, 1, 1), date(2015, 1, 1))
REPORT = date(2015, 1, 1)


def bouts() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        b = load_bouts(db)
    finally:
        db.close()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    b.to_pickle(CACHE)
    return b


def ufc_newcomer_mask(b: pd.DataFrame) -> np.ndarray:
    """UFC bouts where either side has <= 1 earlier UFC bout."""
    seen: dict[int, int] = {}
    out = np.zeros(len(b), dtype=bool)
    for i, (a, o, pr) in enumerate(zip(b["a"], b["b"], b["promo"])):
        if pr == "UFC":
            out[i] = seen.get(a, 0) <= 1 or seen.get(o, 0) <= 1
            seen[a] = seen.get(a, 0) + 1
            seen[o] = seen.get(o, 0) + 1
    return out


def ll(e, y, mask):
    e = np.clip(e[mask], 1e-4, 1 - 1e-4)
    yy = y[mask]
    return float(-(yy * np.log(e) + (1 - yy) * np.log(1 - e)).mean())


def main():
    b = bouts()
    d = np.array(b["date"].tolist())
    y = b["a_score"].to_numpy()
    decided = y != 0.5
    tune = decided & (d >= TUNE[0]) & (d < TUNE[1])
    rep = decided & (d >= REPORT)
    new = rep & ufc_newcomer_mask(b)
    print(f"{len(b):,} bouts; tune {tune.sum():,}, report {rep.sum():,}, UFC newcomer report {new.sum():,}")

    grids = {
        "elo (current pro_elo)": [dict(k0=48.0)],
        "elo + K decay": [dict(k0=k, kd=kd) for k in (32.0, 48.0) for kd in (1.0, 2.0, 4.0)],
        "elo + promotion start": [dict(k0=48.0, promo=True, promo_shrink=s) for s in (10.0, 30.0, 100.0)],
        "elo + finish weight": [dict(k0=48.0, fin=f) for f in (1.25, 1.5)],
        "glicko": [dict(rd=True, rd0=r, c=c) for r in (200.0, 300.0) for c in (1.5, 4.0)],
        "glicko + promo + finish": [dict(rd=True, rd0=r, c=c, promo=True, promo_shrink=30.0, fin=f)
                                    for r in (200.0, 300.0) for c in (1.5, 4.0) for f in (1.0, 1.25)],
    }
    print(f"\n{'variant':28s} {'tuned params':58s} {'tune':>7s} {'all 15+':>8s} {'UFC new':>8s}")
    for name, grid in grids.items():
        best = None
        for kw in grid:
            e, _ = run(b, Params(**kw))
            sc = ll(e, y, tune)
            if best is None or sc < best[0]:
                best = (sc, kw, e)
        sc, kw, e = best
        print(f"{name:28s} {str(kw):58s} {sc:7.4f} {ll(e, y, rep):8.4f} {ll(e, y, new):8.4f}")


if __name__ == "__main__":
    main()
