"""Score the ensemble against true opening and closing lines (BestFightOdds, 2007+).

Until the BFO backfill, the only benchmark was one US book's line ~a day before the
card, and only from 2022. This uses BFO's consensus open and close:

1. Log loss on the same fights: model alone, market open, market close, and the model
   blended with each (blend weights fit on earlier fights only).
2. Line movement: does (model - open) predict (close - open)? A positive slope means the
   model knew, at the open, something the market only priced in later.
3. Betting at the opening price: bet when the model+open blend has positive EV at the
   consensus open; report ROI and closing-line value (fair close prob x price taken - 1).

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.eval_vs_bfo
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.database import SessionLocal
from app.models.ufc import UFCFightOpenClose
from app.services.ufc.market_anchor import american_to_prob, goto_devig

log = logging.getLogger("eval_vs_bfo")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ci(d, n=2000):
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def _decimal(am):
    am = np.asarray(am, float)
    return np.where(am > 0, 1 + am / 100, 1 + 100 / np.abs(am))


def _expanding_blend(df, mkt_col, blocks=8):
    """Blend model with a market column; each block scored by weights fit on earlier ones."""
    out = np.full(len(df), np.nan)
    idx = np.array_split(np.arange(len(df)), blocks)
    for i in range(1, blocks):
        tr = np.concatenate(idx[:i])
        X = np.c_[_logit(df[mkt_col].values[tr]), _logit(df["model_prob"].values[tr])]
        m = LogisticRegression(C=10).fit(X, df["red_wins"].values[tr])
        te = idx[i]
        out[te] = m.predict_proba(np.c_[_logit(df[mkt_col].values[te]),
                                        _logit(df["model_prob"].values[te])])[:, 1]
    return out


def load_lines() -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = db.query(UFCFightOpenClose).filter(UFCFightOpenClose.bookmaker == "Consensus").all()
    finally:
        db.close()
    recs = []
    for r in rows:
        if None in (r.red_open, r.blue_open, r.red_close, r.blue_close):
            continue
        recs.append({
            "fight_id": r.fight_id,
            "open_prob": goto_devig(american_to_prob(r.red_open), american_to_prob(r.blue_open)),
            "close_prob": goto_devig(american_to_prob(r.red_close), american_to_prob(r.blue_close)),
            "red_open_am": r.red_open, "blue_open_am": r.blue_open,
        })
    return pd.DataFrame(recs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-frac", type=float, default=0.6)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--oof-cache", default="models/ufc/h2h/ensemble_oof_long.csv")
    a = ap.parse_args()

    from pathlib import Path
    cache = Path(a.oof_cache)
    if cache.exists():
        oof = pd.read_csv(cache)
    else:
        from app.services.ufc.ensemble import walk_forward_oof
        from app.services.ufc.glicko_service import run_glicko_inmemory
        from app.services.ufc.model import build_features, build_matchup_df, load_fight_data
        df, rd = load_fight_data()
        df = build_features(df, rd, glicko_snapshots=run_glicko_inmemory())
        matchup, features = build_matchup_df(df)
        matchup = matchup.sort_values("date").reset_index()
        matchup = matchup[matchup["red_wins"].notna()].reset_index(drop=True)
        oof = walk_forward_oof(matchup, features, n_folds=a.folds, eval_frac=a.eval_frac)
        oof.to_csv(cache, index=False)

    lines = load_lines()
    df = oof.merge(lines, on="fight_id", how="inner").dropna(subset=["model_prob"])
    df = df.sort_values("date").reset_index(drop=True)
    y = df["red_wins"].to_numpy(float)
    print(f"\n{len(df)} fights with model + BFO open/close  ({df['date'].min()} -> {df['date'].max()})")

    df["blend_open"] = _expanding_blend(df, "open_prob")
    df["blend_close"] = _expanding_blend(df, "close_prob")
    s = df["blend_open"].notna().to_numpy()
    print(f"\n1. Log loss on {s.sum()} fights (blends scored out of sample):")
    base = _ll(df["close_prob"].values[s], y[s])
    for col, label in (("model_prob", "model alone"), ("open_prob", "market OPEN"),
                       ("close_prob", "market CLOSE"), ("blend_open", "model + open"),
                       ("blend_close", "model + close")):
        l = _ll(df[col].values[s], y[s])
        lo, hi = _ci(l - base)
        print(f"   {label:15s} {l.mean():.4f}   vs close {l.mean() - base.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]")

    mv = _logit(df["close_prob"]) - _logit(df["open_prob"])
    dis = _logit(df["model_prob"]) - _logit(df["open_prob"])
    X = np.c_[np.ones(len(df)), dis]
    beta, *_ = np.linalg.lstsq(X, mv, rcond=None)
    resid = mv - X @ beta
    se = np.sqrt(resid.var(ddof=2) / ((dis - dis.mean()) ** 2).sum())
    agree = np.mean(np.sign(dis[np.abs(dis) > 0.2]) == np.sign(mv[np.abs(dis) > 0.2]))
    print(f"\n2. Line movement: slope of (close-open) on (model-open) = {beta[1]:+.4f} "
          f"(se {se:.4f}, t={beta[1] / se:+.1f})")
    print(f"   when the model disagrees with the open by >0.2 logit, the line moved its way "
          f"{agree:.1%} of the time")

    p = df["blend_open"].values
    ok = ~np.isnan(p)
    dr, db_ = _decimal(df["red_open_am"]), _decimal(df["blue_open_am"])
    ev_r, ev_b = p * dr - 1, (1 - p) * db_ - 1
    for thr in (0.0, 0.03, 0.05):
        bet_r = ok & (ev_r > thr) & (ev_r >= ev_b)
        bet_b = ok & (ev_b > thr) & (ev_b > ev_r)
        pnl = np.r_[np.where(y[bet_r] == 1, dr[bet_r] - 1, -1), np.where(y[bet_b] == 0, db_[bet_b] - 1, -1)]
        clv = np.r_[df["close_prob"].values[bet_r] * dr[bet_r] - 1,
                    (1 - df["close_prob"].values[bet_b]) * db_[bet_b] - 1]
        lo, hi = _ci(pnl) if len(pnl) else (np.nan, np.nan)
        clo, chi = _ci(clv) if len(clv) else (np.nan, np.nan)
        print(f"\n3. Bet at OPEN when EV>{thr:.0%}: {len(pnl)} bets  ROI {pnl.mean():+.1%} "
              f"[{lo:+.1%}, {hi:+.1%}]   mean CLV {clv.mean():+.2%} [{clo:+.2%}, {chi:+.2%}]")
    df.to_csv("models/ufc/h2h/eval_vs_bfo.csv", index=False)


if __name__ == "__main__":
    main()
