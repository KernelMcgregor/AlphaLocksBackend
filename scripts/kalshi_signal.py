"""Does Kalshi carry news the sportsbooks don't? Three tests on fights with both.

1. Sharpness: log loss of Kalshi's closing price vs the sportsbook consensus close.
2. News signal: does Kalshi's LATE movement (T-24h -> close), or its disagreement with the
   books at the close, predict the result beyond the book close? Time-ordered
   cross-validated logistic regression, compared with the book close alone.
3. Lead-lag: hour by hour over the last 72h, does a Kalshi move predict the books' NEXT
   move (Kalshi leads), or the reverse?

Time anchor per fight: Kalshi's own (market close_time - 30 min, stored as
days_to_fight = 0). Kalshi is cut there, so it never sees in-play prices.
Book close (tests 1-2): ufc_fight_odds_open_close "Consensus" rows built as the median of
individual sportsbook closes (close_source median_of_N_books). BFO's "Mean" chart is not
used: on recent events it includes Polymarket and Kalshi themselves.
Book series (test 3): per-sportsbook line-movement charts (FanDuel, BetRivers) fetched by
scripts/fetch_book_series.py into data/bfo/book_series.csv, cut at the same anchor.

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.kalshi_signal
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine, text

from app.config import settings

MOVES = "data/bfo/book_series.csv"
ODDS = "data/bfo/odds.csv"
EXCLUDE_BOOKS = ("mean", "polymarket", "kalshi", "ref")
HOURS = 72
#: Kalshi's anchor is bout END minus 30 min, so a long fight can still be in progress at it
#: (checked: Kalshi's edge over the book close jumps from -0.011 at anchor-30min to -0.019 at
#: the anchor). All Kalshi "close" values are read this long before the anchor.
SAFE_LEAD = timedelta(hours=1)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def kalshi_series(engine) -> tuple[pd.DataFrame, pd.Series]:
    """-> (points: fight_id, ts, p_red), anchor time per fight."""
    q = text("""
        select m.fight_id, m.outcome_key, h.captured_at, h.price, h.days_to_fight
        from ufc.ufc_prediction_markets m
        join ufc.ufc_prediction_market_history h on h.market_id = m.id
        where m.platform = 'kalshi' and m.market_type = 'moneyline'
          and m.fight_id is not null and m.outcome_key in ('red', 'blue')""")
    with engine.connect() as c:
        df = pd.DataFrame(c.execute(q).all(), columns=["fight_id", "side", "ts", "price", "dtf"])
    df["ts"] = pd.to_datetime(df["ts"])
    anchor = (df["ts"] + pd.to_timedelta(df["dtf"], unit="D")).groupby(df["fight_id"]).median()
    df = df[df["dtf"] >= 0]
    wide = df.pivot_table(index=["fight_id", "ts"], columns="side", values="price", aggfunc="last")
    red = wide.get("red")
    blue = wide.get("blue")
    p = np.where(red.notna() & blue.notna(), (red + (1 - blue)) / 2, red.fillna(1 - blue))
    pts = pd.DataFrame({"p_red": p}, index=wide.index).reset_index().dropna()
    pts = pts[(pts["p_red"] > 0.01) & (pts["p_red"] < 0.99)]
    return pts, anchor


def book_series(fights: set) -> pd.DataFrame:
    """-> fight_id, ts, book, p_red (each book de-vigged with its latest price on each side)."""
    import os
    if not os.path.exists(MOVES):
        return pd.DataFrame(columns=["fight_id", "book", "ts", "p_red"])
    link = (pd.read_csv(ODDS, usecols=["bfo_matchup_id", "db_fight_id", "db_swapped"],
                        dtype={"db_fight_id": str})
            .dropna(subset=["db_fight_id"]).drop_duplicates("bfo_matchup_id"))
    link["fight_id"] = link["db_fight_id"].str.split(".").str[0].astype("int64")
    link = link[link["fight_id"].isin(fights)]
    mv = pd.read_csv(MOVES).merge(link, on="bfo_matchup_id")
    mv["ts"] = pd.to_datetime(mv["ts"]).dt.tz_localize(None)
    mv["imp"] = 1 / mv["decimal"]
    out = []
    for (fid, book), g in mv.groupby(["fight_id", "book_id"]):
        s = g.pivot_table(index="ts", columns="side", values="imp", aggfunc="last").sort_index().ffill()
        if 1 not in s or 2 not in s:
            continue
        s = s.dropna()
        p_a = s[1] / (s[1] + s[2])
        swapped = str(g["db_swapped"].iloc[0]) == "True"
        out.append(pd.DataFrame({"fight_id": fid, "book": book, "ts": s.index,
                                 "p_red": (1 - p_a if swapped else p_a).values}))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["fight_id", "book", "ts", "p_red"]))


def book_closes(engine) -> pd.Series:
    """Median-of-sportsbooks close per fight (P(red), de-vigged)."""
    with engine.connect() as c:
        rows = c.execute(text("""
            select fight_id, red_close_prob from ufc.ufc_fight_odds_open_close
            where bookmaker = 'Consensus' and close_source like 'median_of_%'
              and red_close_prob is not null""")).all()
    return pd.Series({r[0]: r[1] for r in rows})


def value_at(pts: pd.DataFrame, t) -> float:
    """Latest value at or before t (NaN if none)."""
    s = pts[pts["ts"] <= t]
    return float(s["p_red"].iloc[-1]) if len(s) else np.nan


def book_consensus_at(books: pd.DataFrame, t) -> float:
    s = books[books["ts"] <= t]
    if s.empty:
        return np.nan
    return float(s.groupby("book")["p_red"].last().median())


def outcomes(engine) -> pd.Series:
    with engine.connect() as c:
        rows = c.execute(text("select id, (winner_id = red_fighter_id)::int from ufc.ufc_fights "
                              "where winner_id is not null")).all()
    return pd.Series({r[0]: r[1] for r in rows})


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = d[np.isfinite(d)]
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main() -> None:
    engine = create_engine(settings.DATABASE_URL)
    kp, anchor = kalshi_series(engine)
    y_all = outcomes(engine)
    closes = book_closes(engine)
    fights = set(kp["fight_id"]) & set(y_all.index) & set(closes.index)
    books = book_series(fights)
    print(f"fights with Kalshi + sportsbook close + result: {len(fights)}; "
          f"with per-book series: {books['fight_id'].nunique()}")

    rows, lead = [], []
    kp_by = dict(tuple(kp.groupby("fight_id")))
    bk_by = dict(tuple(books.groupby("fight_id"))) if len(books) else {}
    for fid in sorted(fights, key=lambda f: anchor[f]):
        a = anchor[fid] - SAFE_LEAD
        k = kp_by[fid].sort_values("ts")
        b = bk_by.get(fid)
        r = {"fight_id": fid, "anchor": a, "y": y_all[fid],
             "k_close": value_at(k, a), "b_close": closes[fid],
             "k_24": value_at(k, a - timedelta(hours=24)), "k_72": value_at(k, a - timedelta(hours=72)),
             "b_24": book_consensus_at(b.sort_values("ts"), a - timedelta(hours=24)) if b is not None else np.nan}
        rows.append(r)
        if b is not None:
            b = b.sort_values("ts")
            grid = [a - timedelta(hours=h) for h in range(HOURS, -1, -1)]
            kv = np.array([value_at(k, t) for t in grid])
            bv = np.array([book_consensus_at(b, t) for t in grid])
            lead.append(pd.DataFrame({"fight_id": fid, "h": range(HOURS, -1, -1),
                                      "k": _logit(kv), "b": _logit(bv)}))
    df = pd.DataFrame(rows).dropna(subset=["k_close", "b_close"])
    y = df["y"].to_numpy(float)

    # 1. Sharpness
    print(f"\n1. SHARPNESS (n={len(df)}), log loss at the close")
    lk, lb = _ll(df["k_close"].values, y), _ll(df["b_close"].values, y)
    avg = 1 / (1 + np.exp(-(_logit(df["k_close"]) + _logit(df["b_close"])) / 2))
    la = _ll(avg, y)
    print(f"  Kalshi close  {lk.mean():.4f}")
    print(f"  Books close   {lb.mean():.4f}")
    print(f"  Average       {la.mean():.4f}")
    m, lo, hi = boot(lk - lb); print(f"  Kalshi - books: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    m, lo, hi = boot(la - lb); print(f"  Average - books: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  mean |Kalshi - books| at close: {np.abs(df.k_close - df.b_close).mean():.3f}")

    # 2. News signal: time-ordered CV, features beyond the book close
    print("\n2. NEWS SIGNAL: does Kalshi add to the book close? (time-ordered 5-fold CV log loss)")
    d2 = df.dropna(subset=["k_24"]).reset_index(drop=True)
    X_base = _logit(d2["b_close"]).reshape(-1, 1)
    feats = {
        "book close only": X_base,
        "+ Kalshi late move (T-24h -> close)": np.column_stack([X_base, _logit(d2.k_close) - _logit(d2.k_24)]),
        "+ Kalshi vs books gap at close": np.column_stack([X_base, _logit(d2.k_close) - _logit(d2.b_close)]),
        "+ Kalshi 3-day move (T-72h -> close)": np.column_stack(
            [X_base, np.nan_to_num(_logit(d2.k_close) - _logit(d2.k_72.fillna(d2.k_24)))]),
    }
    yy = d2["y"].to_numpy(float)
    n = len(d2); folds = np.array_split(np.arange(n), 6)
    losses = {}
    for name, X in feats.items():
        l = np.full(n, np.nan)
        for i in range(1, 6):
            tr = np.concatenate(folds[:i]); te = folds[i]
            m_ = LogisticRegression(C=1.0).fit(X[tr], yy[tr])
            l[te] = _ll(m_.predict_proba(X[te])[:, 1], yy[te])
        losses[name] = l
    base = losses["book close only"]
    for name, l in losses.items():
        ok = np.isfinite(l)
        if name == "book close only":
            print(f"  {name:40s} {l[ok].mean():.4f}  (n={ok.sum()})")
        else:
            m, lo, hi = boot(l - base)
            print(f"  {name:40s} {l[ok].mean():.4f}  diff {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    full = LogisticRegression(C=1.0).fit(feats["+ Kalshi late move (T-24h -> close)"], yy)
    print(f"  coefficient on Kalshi late move (all data): {full.coef_[0][1]:+.3f}")

    # 3. Lead-lag, pooled hourly changes
    print(f"\n3. LEAD-LAG over the last {HOURS}h (hourly changes in logit price, pooled)")
    if not lead:
        print("  no per-book series yet (run scripts/fetch_book_series.py)")
        return
    L = pd.concat(lead).sort_values(["fight_id", "h"], ascending=[True, False])
    L["dk"] = L.groupby("fight_id")["k"].diff(); L["db"] = L.groupby("fight_id")["b"].diff()
    for hz in (1, 3, 6):
        L[f"fb{hz}"] = L.groupby("fight_id")["b"].shift(-hz) - L["b"]
        L[f"fk{hz}"] = L.groupby("fight_id")["k"].shift(-hz) - L["k"]
    Z = L.dropna(subset=["dk", "db", "fb6", "fk6"])
    Z = Z[(Z["dk"].abs() < 2) & (Z["db"].abs() < 2)]
    for hz in (1, 3, 6):
        kb = np.polyfit(Z["dk"], Z[f"fb{hz}"], 1)[0]
        bk = np.polyfit(Z["db"], Z[f"fk{hz}"], 1)[0]
        print(f"  next {hz}h: books follow Kalshi move x{kb:+.3f} | Kalshi follows books move x{bk:+.3f}")
    big = Z[Z["dk"].abs() > 0.15]
    if len(big):
        follow = (np.sign(big["dk"]) * big["fb6"]).mean()
        own = big["dk"].abs().mean()
        print(f"  big Kalshi hourly moves (>0.15 logit, n={len(big)}): books move {follow:+.3f} "
              f"in the same direction over the next 6h (vs {own:.3f} Kalshi move)")
    bigb = Z[Z["db"].abs() > 0.15]
    if len(bigb):
        print(f"  big book hourly moves (n={len(bigb)}): Kalshi moves "
              f"{(np.sign(bigb['db']) * bigb['fk6']).mean():+.3f} the same way over the next 6h")


if __name__ == "__main__":
    main()
