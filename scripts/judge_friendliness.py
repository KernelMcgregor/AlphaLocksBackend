"""Is "judge-friendliness" a real fighter trait?

For every judged round (mmadecisions verified cards with UFCStats round stats):

    gap = share of judges who gave the fighter the round  -  round model's P(fighter won it)

The round model (scripts.round_model) is CROSS-FITTED by fight: each round's expected
value comes from a model that never saw that fight, so a fighter's own rounds can't pull
the expectation toward their result. 10-10 judge-rounds are left out of the share.

Two persistence tests (both only ever compare a fighter with themself across fights):

  split-half    each fighter's judged fights in date order, first half vs second half:
                does the early mean gap predict the late one?
  prospective   for every judged fight, the fighter's shrunk mean gap from STRICTLY
                EARLIER fights (sum / (rounds + K)) vs their mean gap in this fight. This
                is exactly how it would enter the winner model as a feature.

Confidence intervals bootstrap over fighters. A slope / correlation whose CI excludes 0
says the gap is partly a trait; one centred on 0 says it is noise (luck, judges, opponents).

    DATABASE_URL=postgresql://localhost/alocks_local PYTHONPATH=. \
        venv/bin/python -m scripts.judge_friendliness --out data/mmad/fighter_gaps.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sqlalchemy import create_engine, text

from scripts.round_model import fit, load_cards, load_round_stats, predict


def cross_fitted_rounds(eng, folds: int = 5) -> pd.DataFrame:
    """One row per judged fight-round: red's judge share and cross-fitted P(red)."""
    rounds = load_round_stats(eng)
    cards = load_cards(eng)
    cards = cards[cards.red != cards.blue].copy()
    cards["y"] = (cards.red > cards.blue).astype(int)
    data = cards.merge(rounds, on=["fight_id", "round"], how="inner").reset_index(drop=True)
    data["p"] = np.nan
    for tr_idx, te_idx in GroupKFold(n_splits=folds).split(data, groups=data["fight_id"]):
        m, mu, sd = fit(data.iloc[tr_idx])
        data.loc[data.index[te_idx], "p"] = predict(m, mu, sd, data.iloc[te_idx])
    return (data.groupby(["fight_id", "round"])
            .agg(share=("y", "mean"), n_judges=("y", "size"), p=("p", "first"),
                 date=("date", "first"), red=("red_fighter_id", "first"),
                 blue=("blue_fighter_id", "first"))
            .reset_index())


def fighter_rounds(r: pd.DataFrame) -> pd.DataFrame:
    """Both corners' view of each round: gap = actual share - expected, for that fighter."""
    red = r.assign(fighter_id=r.red, opp_id=r.blue, gap=r.share - r.p)
    blue = r.assign(fighter_id=r.blue, opp_id=r.red, gap=(1 - r.share) - (1 - r.p))
    return pd.concat([red, blue])[["fight_id", "round", "date", "fighter_id", "opp_id", "gap"]]


def per_fight(fr: pd.DataFrame) -> pd.DataFrame:
    return (fr.groupby(["fighter_id", "fight_id"])
            .agg(date=("date", "first"), gap=("gap", "mean"), n=("gap", "size"),
                 gap_sum=("gap", "sum"))
            .reset_index().sort_values(["fighter_id", "date", "fight_id"]))


def _boot(fighters: np.ndarray, stat, B: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(fighters)
    idx_by = {f: np.flatnonzero(fighters == f) for f in uniq}
    vals = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[f] for f in pick])
        vals.append(stat(rows))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _wcorr(x, y, w) -> float:
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return float(cov / np.sqrt(np.average((x - mx) ** 2, weights=w) *
                               np.average((y - my) ** 2, weights=w)))


def split_half(pf: pd.DataFrame, min_fights: int = 4) -> pd.DataFrame:
    rows = []
    for fid, g in pf.groupby("fighter_id"):
        if len(g) < min_fights:
            continue
        h = len(g) // 2
        e, l = g.iloc[:h], g.iloc[h:]
        rows.append(dict(fighter_id=fid, early=e.gap_sum.sum() / e.n.sum(), n_early=e.n.sum(),
                         late=l.gap_sum.sum() / l.n.sum(), n_late=l.n.sum()))
    return pd.DataFrame(rows)


def prospective(pf: pd.DataFrame, k: float) -> pd.DataFrame:
    """Shrunk prior mean gap (earlier fights only) next to this fight's mean gap."""
    pf = pf.copy()
    g = pf.groupby("fighter_id")
    pf["prior_sum"] = g.gap_sum.cumsum() - pf.gap_sum
    pf["prior_n"] = g.n.cumsum() - pf.n
    pf["feature"] = pf.prior_sum / (pf.prior_n + k)
    return pf[pf.prior_n > 0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/mmad/fighter_gaps.csv"))
    args = ap.parse_args(argv)
    eng = create_engine(os.environ["DATABASE_URL"])

    r = cross_fitted_rounds(eng)
    fr = fighter_rounds(r)
    pf = per_fight(fr)
    print(f"judged rounds: {len(r):,} in {r.fight_id.nunique():,} fights; "
          f"fighters: {fr.fighter_id.nunique():,}")
    print(f"mean gap (should be ~0): {fr.gap.mean():+.4f}; SD per round {fr.gap.std():.3f}")

    # ---- split-half
    sh = split_half(pf)
    w = np.sqrt(sh.n_early * sh.n_late)   # more rounds on both sides = more reliable
    x, y = sh.early.to_numpy(), sh.late.to_numpy()
    c = _wcorr(x, y, w)
    lo, hi = _boot(sh.fighter_id.to_numpy(), lambda i: _wcorr(x[i], y[i], w.to_numpy()[i]))
    print(f"\nSPLIT-HALF (fighters with >= 4 judged fights: {len(sh):,})")
    print(f"  corr(early gap, late gap) = {c:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    # Placebo: pair each fighter's early half with a random other fighter's late half.
    rng = np.random.default_rng(1)
    plc = [_wcorr(x, rng.permutation(y), w) for _ in range(500)]
    print(f"  placebo (shuffled fighters): 95% of corr within "
          f"[{np.percentile(plc, 2.5):+.3f}, {np.percentile(plc, 97.5):+.3f}]")

    # ---- prospective, as a feature would be built
    print("\nPROSPECTIVE (prior fights only -> this fight), by shrinkage K (pseudo-rounds at 0):")
    print(f"  {'K':>4} {'fights':>7} {'slope':>8} {'95% CI':>20} {'corr':>7}")
    for k in (5, 10, 20, 40):
        p = prospective(pf, k)
        X, Y, W = p.feature.to_numpy(), p.gap.to_numpy(), p.n.to_numpy()
        fids = p.fighter_id.to_numpy()

        def slope(i, X=X, Y=Y, W=W):
            xm, ym = np.average(X[i], weights=W[i]), np.average(Y[i], weights=W[i])
            return np.sum(W[i] * (X[i] - xm) * (Y[i] - ym)) / np.sum(W[i] * (X[i] - xm) ** 2)

        allr = np.arange(len(p))
        lo, hi = _boot(fids, slope, B=500)
        print(f"  {k:>4} {len(p):>7,} {slope(allr):>+8.3f}   [{lo:+.3f}, {hi:+.3f}]"
              f" {_wcorr(X, Y, W):>+7.3f}")

    # Size of the effect, if real: spread of the shrunk feature in rounds per 3-round fight.
    p = prospective(pf, 10)
    print(f"\n  feature spread (K=10): SD {p.feature.std():.3f} per round "
          f"= {3 * p.feature.std():.2f} rounds per 3-round fight; "
          f"5th-95th pct [{p.feature.quantile(.05):+.3f}, {p.feature.quantile(.95):+.3f}]")

    names = pd.read_sql(text("SELECT id AS fighter_id, first_name || ' ' || last_name AS name "
                             "FROM ufc.ufc_fighters"), eng)
    career = (pf.groupby("fighter_id").agg(fights=("fight_id", "size"), rounds=("n", "sum"),
                                           gap_sum=("gap_sum", "sum"))
              .reset_index().merge(names, on="fighter_id"))
    career["gap_per_round"] = career.gap_sum / career.rounds
    career["shrunk"] = career.gap_sum / (career.rounds + 10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    career.sort_values("shrunk", ascending=False).to_csv(args.out, index=False)
    top = career[career.rounds >= 20].sort_values("shrunk")
    print("\nmost judge-friendly (>= 20 judged rounds; shrunk gap per round, rounds above expected):")
    for _, row in top.tail(8).iloc[::-1].iterrows():
        print(f"  {row['name']:<26} {row.shrunk:+.3f}  ({row.gap_sum:+.1f} rounds over {int(row.rounds)})")
    print("least:")
    for _, row in top.head(8).iterrows():
        print(f"  {row['name']:<26} {row.shrunk:+.3f}  ({row.gap_sum:+.1f} rounds over {int(row.rounds)})")
    print(f"\nwrote {len(career):,} fighters -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
