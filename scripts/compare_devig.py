"""Which way of stripping the bookmaker margin gives the best probabilities for UFC?

Compares, on settled two-way moneylines:
  multiplicative  p_i / sum(p)                     (what odds_scraper stores today)
  additive        p_i - (sum(p) - 1) / 2
  power           p_i ** k, k solved so the pair sums to 1
  shin            Shin (1993) insider-trading model, closed form for two outcomes
  goto            goto_conversion: p_i - se_i * step, se_i = sqrt(p_i (1 - p_i) / p_i)
                  (github.com/gotoConversion/goto_conversion)

goto, power and Shin all shade longshots down, i.e. assume a favourite-longshot bias.
A 2026 study found no such bias in MMA, so multiplicative may be the right answer here;
this measures it rather than assuming. Paired bootstrap vs multiplicative.

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.compare_devig
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from app.database import SessionLocal
from app.models.ufc import UFCFight, UFCFightOdds
from app.services.ufc.fighter_registry import is_decided
from app.services.ufc.market_anchor import american_to_prob


def multiplicative(p):
    return p / p.sum()


def additive(p):
    return p - (p.sum() - 1) / 2


def power(p):
    k = brentq(lambda k: (p ** k).sum() - 1, 0.5, 3.0)
    return p ** k


def shin(p):
    # Two-outcome closed form (Jullien & Salanie 1994 / Strumbelj 2014).
    s = p.sum()
    d = p[0] - p[1]
    z = ((s - 1) * (d ** 2 - s)) / (s * (d ** 2 - 1))
    return (np.sqrt(z ** 2 + 4 * (1 - z) * p ** 2 / s) - z) / (2 * (1 - z))


def goto(p):
    se = np.sqrt((p - p ** 2) / p)
    step = (p.sum() - 1) / se.sum()
    return np.clip(p - se * step, 1e-6, 1 - 1e-6)


METHODS = {"multiplicative": multiplicative, "additive": additive, "power": power,
           "shin": shin, "goto": goto}


def main() -> None:
    db = SessionLocal()
    try:
        rows = (db.query(UFCFightOdds, UFCFight)
                .join(UFCFight, UFCFight.id == UFCFightOdds.fight_id).all())
    finally:
        db.close()
    probs = {m: [] for m in METHODS}
    ys, fav = [], []
    for o, f in rows:
        if not is_decided(f.method, f.winner_id) or not o.red_odds or not o.blue_odds:
            continue
        raw = np.array([american_to_prob(o.red_odds), american_to_prob(o.blue_odds)])
        if raw.sum() <= 1.0:
            continue  # no margin to remove (bad row)
        for m, fn in METHODS.items():
            probs[m].append(fn(raw)[0] / fn(raw).sum())
        ys.append(float(f.winner_id == f.red_fighter_id))
        fav.append(max(raw) / raw.sum())
    y = np.array(ys)
    print(f"{len(y)} book-lines on decided fights\n")

    def ll(p):
        p = np.clip(np.array(p), 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    base = ll(probs["multiplicative"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(2000, len(y)))
    print(f"{'method':15s} {'log_loss':>9s} {'d_vs_mult':>10s}   95% CI")
    for m in METHODS:
        l = ll(probs[m]); d = l - base
        bs = d[idx].mean(axis=1)
        print(f"{m:15s} {l.mean():9.5f} {d.mean():+10.5f}   "
              f"[{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]")

    # Favourite-longshot check: calibration of the multiplicative price by bucket.
    p = np.array(probs["multiplicative"])
    fav_p = np.where(p >= 0.5, p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    print("\nfavourite calibration (multiplicative):")
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)):
        m = (fav_p >= lo) & (fav_p < hi)
        if m.sum():
            print(f"  {lo:.1f}-{hi:.1f}  n={m.sum():5d}  implied={fav_p[m].mean():.3f}  "
                  f"won={fav_won[m].mean():.3f}")


if __name__ == "__main__":
    main()
