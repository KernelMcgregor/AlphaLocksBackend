"""Backtest: would betting the method model at BestFightOdds closing prices have profited?

Model probabilities are walk-forward out-of-sample (method_oof.csv x a winner probability),
so no fight was predicted by a model that had seen it. Prices are real sportsbook closing
odds parsed from cached BFO event pages (exchanges excluded).

Markets
  sixway    "<fighter> wins by KO / submission / decision"   (6 cells, Yes side)
  decision  "Fight goes to decision" yes / no
  itd       "<fighter> wins inside the distance" yes / no

Winner probability used for the cells
  model     the ensemble's own OOF probability (never saw odds)
  mktwin    the prop market's implied P(red) (so only the method part is the model's)
  blend     50/50 log blend of the mktwin 6-way grid with the market's own 6-way price

Rule: per fight and market, bet the single outcome with the highest EV if EV >= threshold;
flat 1-unit stakes. Price = best across sportsbooks (line shopping) or the median book.
ROI CI: bootstrap over fights.

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.method_roi
"""
from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from app.config import settings
from app.services.ufc.bfo_props import parse_props
from app.services.ufc.bfo_scraper import DEFAULT_CACHE_DIR, _json_ld_date
from app.services.ufc.method_v2 import OOF_PATH
from scripts.method_benchmark import CELLS, outcomes

THRESHOLDS = (0.0, 0.05, 0.10, 0.20)


def _dec(a):
    return 1 + (a / 100 if a > 0 else 100 / -a)


def prices() -> pd.DataFrame:
    """fight_id, key (red/blue terms), best decimal, median decimal, n_books."""
    rows = []
    for f in glob.glob(str(Path(DEFAULT_CACHE_DIR) / "pages" / "events_*.txt")):
        html = Path(f).read_text()
        if _json_ld_date(html) is None:
            continue
        rows.extend(parse_props(html))
    df = pd.DataFrame(rows)
    df["dec"] = df["american"].map(_dec)
    link = (pd.read_csv("data/bfo/odds.csv", dtype={"db_fight_id": str},
                        usecols=["bfo_matchup_id", "db_fight_id", "db_swapped"])
            .dropna(subset=["db_fight_id"]).drop_duplicates("bfo_matchup_id"))
    df = df.merge(link, left_on="matchup", right_on="bfo_matchup_id")
    sw = df["db_swapped"].astype(str) == "True"

    def corner(key, swapped):
        parts = key.split("_")
        if parts[0] in ("wm", "itd"):
            side = parts[1]
            red = (side == "a") != swapped
            return "_".join([parts[0], "red" if red else "blue"] + parts[2:])
        return key
    df["key"] = [corner(k, s) for k, s in zip(df["key"], sw)]
    df["fight_id"] = df["db_fight_id"].str.split(".").str[0]
    g = df.groupby(["fight_id", "key"])["dec"]
    return pd.DataFrame({"best": g.max(), "median": g.median(), "n": g.size()}).reset_index()


def model_probs(oof: pd.DataFrame, props: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per variant: fight_id + probability for each bettable key."""
    ens = pd.read_csv("models/ufc/h2h/ensemble_oof.csv", dtype={"fight_id": str})[["fight_id", "model_prob"]]
    d = oof.merge(ens, on="fight_id").merge(props, on="fight_id")
    cr = d[list(CELLS[:3])].to_numpy(float); cb = d[list(CELLS[3:])].to_numpy(float)
    mk = d[[f"mkt_{c}" for c in CELLS]].to_numpy(float)
    mk_red = mk[:, :3].sum(axis=1)
    out = {}
    for name, p_red in (("model", d["model_prob"].to_numpy(float)), ("mktwin", mk_red)):
        six = np.hstack([p_red[:, None] * cr, (1 - p_red)[:, None] * cb])
        out[name] = six
    lb = np.exp(0.5 * np.log(np.clip(out["mktwin"], 1e-6, 1)) + 0.5 * np.log(np.clip(mk, 1e-6, 1)))
    out["blend"] = lb / lb.sum(axis=1, keepdims=True)
    frames = {}
    for name, six in out.items():
        f = pd.DataFrame({"fight_id": d["fight_id"]})
        for j, c in enumerate(CELLS):
            side, m = c.split("_")
            f[f"wm_{side}_{m}"] = six[:, j]
        dec = six[:, 2] + six[:, 5]
        f["dec_yes"], f["dec_no"] = dec, 1 - dec
        for side, cols in (("red", [0, 1]), ("blue", [3, 4])):
            itd = six[:, cols].sum(axis=1)
            f[f"itd_{side}_yes"], f[f"itd_{side}_no"] = itd, 1 - itd
        frames[name] = f
    return frames


MARKETS = {
    "sixway": [f"wm_{s}_{m}" for s in ("red", "blue") for m in ("ko", "sub", "dec")],
    "decision": ["dec_yes", "dec_no"],
    "itd": ["itd_red_yes", "itd_red_no", "itd_blue_yes", "itd_blue_no"],
}


def won(key: str, y: int, red_won: bool) -> bool:
    winner = "red" if red_won else "blue"
    cls = ("ko", "sub", "dec")[y]
    if key.startswith("wm_"):
        _, side, m = key.split("_")
        return side == winner and m == cls
    if key == "dec_yes":
        return cls == "dec"
    if key == "dec_no":
        return cls != "dec"
    if key.startswith("itd_"):
        _, side, yn = key.split("_")
        hit = side == winner and cls != "dec"
        return hit if yn == "yes" else not hit
    raise ValueError(key)


def backtest(probs: pd.DataFrame, px: pd.DataFrame, res: pd.DataFrame, market: str,
             thr: float, price: str) -> pd.DataFrame:
    keys = MARKETS[market]
    pxw = px[px["key"].isin(keys)].pivot_table(index="fight_id", columns="key", values=price)
    d = probs.merge(res, on="fight_id").merge(pxw.add_prefix("px_").reset_index(), on="fight_id")
    bets = []
    for r in d.itertuples(index=False):
        rd = r._asdict()
        best = None
        for k in keys:
            q = rd.get(f"px_{k}")
            if q is None or q != q:
                continue
            ev = rd[k] * q - 1
            if ev >= thr and (best is None or ev > best[1]):
                best = (k, ev, q)
        if best:
            k, ev, q = best
            w = won(k, rd["y"], rd["red_won"])
            bets.append({"fight_id": rd["fight_id"], "key": k, "ev": ev, "price": q,
                         "won": w, "profit": (q - 1) if w else -1.0})
    return pd.DataFrame(bets)


def summarize(b: pd.DataFrame, n_boot: int = 2000) -> dict:
    if b.empty:
        return {"bets": 0}
    rng = np.random.default_rng(0)
    prof = b["profit"].to_numpy()
    bs = [prof[rng.integers(0, len(prof), len(prof))].mean() for _ in range(n_boot)]
    return {"bets": len(b), "hit": b["won"].mean(), "avg_price": b["price"].mean(),
            "roi": prof.mean(), "ci": f"[{np.percentile(bs, 2.5):+.1%}, {np.percentile(bs, 97.5):+.1%}]",
            "units": prof.sum()}


def main() -> None:
    engine = create_engine(settings.DATABASE_URL)
    res = outcomes(engine)
    oof = pd.read_csv(OOF_PATH, dtype={"fight_id": str})
    props = pd.read_csv("data/bfo/props_close.csv", dtype={"fight_id": str}).dropna(
        subset=[f"mkt_{c}" for c in CELLS])
    px = prices()
    frames = model_probs(oof, props)
    rows = []
    for variant, probs in frames.items():
        for market in MARKETS:
            for price in ("best", "median"):
                for thr in THRESHOLDS:
                    s = summarize(backtest(probs, px, res, market, thr, price))
                    rows.append({"variant": variant, "market": market, "price": price,
                                 "min_ev": thr, **s})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    fmt = t.copy()
    for c in ("hit", "roi"):
        fmt[c] = fmt[c].map(lambda v: f"{v:+.1%}" if c == "roi" and v == v else (f"{v:.1%}" if v == v else ""))
    print(fmt.round(2).to_string(index=False))
    t.to_csv("data/bfo/method_roi.csv", index=False)


if __name__ == "__main__":
    main()
