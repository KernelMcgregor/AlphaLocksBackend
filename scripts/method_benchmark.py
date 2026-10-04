"""Benchmark the method model against prop-market closing prices.

Model: walk-forward OOF conditional probabilities (models/ufc/method/method_oof.csv) --
each fight was predicted by a model that had not seen it.
Markets:
  bfo   BestFightOdds consensus close, de-vigged (data/bfo/props_close.csv; bfo_props.py)
  poly  Polymarket fight-level KO and goes-the-distance closing prices (last pre-fight
        history point; local DB). Its per-fighter markets cover KO only, so no 6-way test.

Scores (log loss, lower is better; paired bootstrap CI on the difference):
  cond      P(method | actual winner). The fair "method knowledge" test: it removes the
            winner question, where the market is known to be better.
  marginal  P(KO / Sub / Dec)
  decision  P(goes to decision) yes/no
  six       6-way cell, with the MARKET moneyline for who wins (how the model would be
            used for winner x method bets) vs the market's own 6-way price
Each score also has a 50/50 log-linear blend of model and market: if the blend beats
the market, the model carries information the market does not.

--corr also scores v2_corr: the method_market decision correction, refit each quarter
from CORR_FROM on rows before the quarter, against raw v2 and the market on the same
fights, overall and by market-favourite band.

Usage:
    DATABASE_URL=postgresql://localhost/alocks_local python -m scripts.method_benchmark [--corr]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from app.config import settings
from app.services.ufc.method_v2 import OOF_PATH

PROPS = "data/bfo/props_close.csv"
CELLS = ("red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec")
CORR_FROM = pd.Timestamp("2024-09-01")   # v2_corr is scored from here (expanding refit)


def _ll(p):
    return -np.log(np.clip(p, 1e-6, 1))


def _norm(a):
    return a / a.sum(axis=1, keepdims=True)


def _blend(p, q, w=0.5):
    return _norm(np.exp(w * np.log(np.clip(p, 1e-6, 1)) + (1 - w) * np.log(np.clip(q, 1e-6, 1))))


def outcomes(engine) -> pd.DataFrame:
    with engine.connect() as c:
        rows = c.execute(text("select id::text, method, details, winner_id, red_fighter_id "
                              "from ufc.ufc_fights where winner_id is not null")).all()
    from app.services.ufc.method_ratings import method_class
    out = []
    for fid, method, details, winner, red in rows:
        k = method_class(method, details, winner)
        if k:
            out.append((fid, {"ko": 0, "sub": 1, "dec": 2}[k], winner == red))
    return pd.DataFrame(out, columns=["fight_id", "y", "red_won"])


def polymarket_binary(engine) -> pd.DataFrame:
    """Closing Yes prices for Polymarket's fight-level KO and goes-the-distance markets
    (its per-fighter markets only cover KO, so no 6-way comparison is possible)."""
    q = text("""
        select m.fight_id::text fight_id, m.outcome_key, h.price
        from ufc.ufc_prediction_markets m
        join lateral (select price from ufc.ufc_prediction_market_history h
                      where h.market_id = m.id and h.days_to_fight >= 0
                      order by h.captured_at desc limit 1) h on true
        where m.platform = 'polymarket' and m.fight_id is not null
          and m.outcome_key in ('ko_tko', 'distance')""")
    with engine.connect() as c:
        df = pd.DataFrame(c.execute(q).all(), columns=["fight_id", "key", "price"])
    return df.pivot_table(index="fight_id", columns="key", values="price", aggfunc="last").reset_index()


def evaluate_binary(df: pd.DataFrame, label: str, n_boot: int = 2000) -> pd.DataFrame:
    """KO yes/no and distance yes/no: model marginal (market moneyline for the winner mix)
    vs the market's Yes price."""
    y = df["y"].to_numpy(int)
    p_red = df["p_red"].to_numpy(float)
    cr, cb = df[list(CELLS[:3])].to_numpy(float), df[list(CELLS[3:])].to_numpy(float)
    m = p_red[:, None] * cr + (1 - p_red[:, None]) * cb
    rng = np.random.default_rng(0)
    out = []
    for name, col, model_p, hit in (("ko_yes", "ko_tko", m[:, 0], y == 0),
                                    ("distance_yes", "distance", m[:, 2], y == 2)):
        if col not in df.columns:
            continue
        ok = df[col].notna().to_numpy() & (df[col] > 0.01).to_numpy() & (df[col] < 0.99).to_numpy()
        pm, pk = model_p[ok], df[col].to_numpy(float)[ok]
        h = hit[ok]
        f = lambda p: _ll(np.where(h, p, 1 - p))
        pb = 1 / (1 + np.exp(-(0.5 * np.log(pm / (1 - pm)) + 0.5 * np.log(pk / (1 - pk)))))
        lm, lk, lb = f(pm), f(pk), f(pb)
        row = {"market": label, "score": name, "n": int(ok.sum()), "model": lm.mean(),
               "market_ll": lk.mean(), "blend": lb.mean()}
        for tag, diff in (("model-mkt", lm - lk), ("blend-mkt", lb - lk)):
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
            row[tag] = diff.mean()
            row[f"{tag}_ci"] = f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
        out.append(row)
    return pd.DataFrame(out)


def _scores(df: pd.DataFrame) -> dict:
    """Per-fight (model, market, blend) probabilities of what happened, per score.
    df: model cond cols (red_ko..blue_dec), mkt_<cell> cols, y, red_won."""
    y = df["y"].to_numpy(int)
    red = df["red_won"].to_numpy(bool)
    rows = np.arange(len(df))
    cr, cb = df[list(CELLS[:3])].to_numpy(float), df[list(CELLS[3:])].to_numpy(float)
    mk = df[[f"mkt_{c}" for c in CELLS]].to_numpy(float)
    mk_red = mk[:, :3].sum(axis=1)
    mc_red, mc_blue = _norm(mk[:, :3]), _norm(mk[:, 3:])  # market conditional on winner
    model_six = np.hstack([mk_red[:, None] * cr, (1 - mk_red)[:, None] * cb])
    truth6 = np.where(red, y, 3 + y)

    scores = {}
    # cond
    m_c = np.where(red[:, None], cr, cb); k_c = np.where(red[:, None], mc_red, mc_blue)
    scores["cond"] = (m_c[rows, y], k_c[rows, y], _blend(m_c, k_c)[rows, y])
    # marginal
    m_m = model_six[:, :3] + model_six[:, 3:]; k_m = mk[:, :3] + mk[:, 3:]
    scores["marginal"] = (m_m[rows, y], k_m[rows, y], _blend(m_m, k_m)[rows, y])
    # decision yes/no
    d = (y == 2)
    m_d = np.column_stack([m_m[:, 2], 1 - m_m[:, 2]])
    k_d = np.column_stack([k_m[:, 2], 1 - k_m[:, 2]])
    idx = np.where(d, 0, 1)
    scores["decision"] = (m_d[rows, idx], k_d[rows, idx], _blend(m_d, k_d)[rows, idx])
    # six
    scores["six"] = (model_six[rows, truth6], mk[rows, truth6], _blend(model_six, mk)[rows, truth6])
    return scores


def evaluate(df: pd.DataFrame, label: str, n_boot: int = 2000) -> pd.DataFrame:
    """df: model cond cols (red_ko..blue_dec), mkt_<cell> cols, y, red_won."""
    scores = _scores(df)
    rng = np.random.default_rng(0)
    out = []
    for name, (pm, pk, pb) in scores.items():
        lm, lk, lb = _ll(pm), _ll(pk), _ll(pb)
        row = {"market": label, "score": name, "n": len(lm), "model": lm.mean(), "market_ll": lk.mean(),
               "blend": lb.mean()}
        for tag, diff in (("model-mkt", lm - lk), ("blend-mkt", lb - lk)):
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
            row[tag] = diff.mean()
            row[f"{tag}_ci"] = f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
        out.append(row)
    return pd.DataFrame(out)


def corrected_oof(oof: pd.DataFrame, matrix: pd.DataFrame, start=CORR_FROM) -> pd.DataFrame:
    """method_market's decision correction, refit quarterly on an expanding window: each
    quarter's fights are corrected by a fit on rows dated before the quarter. OOF rows from
    `start` on, unpriced fights unchanged; adds q_red (de-vigged moneyline)."""
    from app.services.ufc import method_market as mm
    rows = mm.training_rows(oof, matrix)
    m = matrix.assign(fight_id=matrix["fight_id"].astype(str))
    d = oof.assign(fight_id=oof["fight_id"].astype(str)).merge(
        m[["fight_id", "date", "odds_red_prob", "odds_blue_prob"]], on="fight_id")
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] >= start].reset_index(drop=True)
    d["q_red"] = mm.market_red_prob(d)
    for qtr, idx in d.groupby(d["date"].dt.to_period("Q")).groups.items():
        corr = mm.fit(rows[rows["date"] < qtr.start_time])
        cr, cb = corr.apply(d.loc[idx, list(CELLS[:3])].to_numpy(float),
                            d.loc[idx, list(CELLS[3:])].to_numpy(float), d.loc[idx, "q_red"].to_numpy(float))
        d.loc[idx, list(CELLS[:3])] = cr
        d.loc[idx, list(CELLS[3:])] = cb
        print(f"  {qtr}: fit on {corr.n_fit} rows, hinge={corr.hinge}, "
              f"coef={np.round(corr.coef, 2).tolist()} c={corr.intercept:+.2f}")
    return d[["fight_id", "q_red", *CELLS]]


def compare_corrected(raw: pd.DataFrame, corr: pd.DataFrame, n_boot: int = 2000):
    """Same fights, raw v2 vs corrected (v2_corr) vs market: overall per score with paired
    CIs, and the decision score by market-favourite band."""
    s_raw, s_cor = _scores(raw), _scores(corr)
    rng = np.random.default_rng(0)

    def ci(diff):
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
        return f"{diff.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
    rows = []
    for name in ("decision", "marginal", "six", "cond"):
        lr, lc, lk = _ll(s_raw[name][0]), _ll(s_cor[name][0]), _ll(s_raw[name][1])
        rows.append({"score": name, "n": len(lr), "v2": lr.mean(), "v2_corr": lc.mean(), "market": lk.mean(),
                     "corr-v2": ci(lc - lr), "corr-mkt": ci(lc - lk), "v2-mkt": ci(lr - lk)})
    q = corr["q_red"].to_numpy(float)
    band = pd.cut(np.maximum(q, 1 - q), [0, 0.6, 0.7, 0.85, 1.0])
    lr, lc, lk = (_ll(s_raw["decision"][0]), _ll(s_cor["decision"][0]), _ll(s_raw["decision"][1]))
    by = pd.DataFrame({"band": band, "v2": lr, "v2_corr": lc, "market": lk}).groupby("band", observed=True)
    bands = by.mean().assign(n=by.size(), **{"corr-v2": by.mean()["v2_corr"] - by.mean()["v2"]})
    return pd.DataFrame(rows), bands


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", action="store_true",
                    help="also score v2_corr (method_market decision correction, expanding refit)")
    args = ap.parse_args()
    engine = create_engine(settings.DATABASE_URL)
    oof = pd.read_csv(OOF_PATH, dtype={"fight_id": str})
    res = outcomes(engine)
    base = oof.merge(res, on="fight_id")
    tables = []
    bfo = pd.read_csv(PROPS, dtype={"fight_id": str})
    b = base.merge(bfo, on="fight_id").dropna(subset=[f"mkt_{c}" for c in CELLS])
    tables.append(evaluate(b, "bfo"))
    poly = polymarket_binary(engine)
    if len(poly):
        # Winner mix for the model: the BFO moneyline-consistent price where we have it.
        p = base.merge(poly, on="fight_id").merge(
            bfo[["fight_id", "mkt_red_ko", "mkt_red_sub", "mkt_red_dec"]], on="fight_id", how="left")
        p["p_red"] = p[["mkt_red_ko", "mkt_red_sub", "mkt_red_dec"]].sum(axis=1, min_count=3).fillna(0.5)
        if len(p) >= 30:
            tables.append(evaluate_binary(p, "poly"))
        print(f"polymarket fights with closing KO/distance prices in the OOF window: {len(p)}")
    pd.set_option("display.width", 200)
    t = pd.concat(tables)
    print(t.round(4).to_string(index=False))
    t.to_csv("data/bfo/method_benchmark.csv", index=False)
    if args.corr:
        import pickle

        from app.services.ufc.method_v2 import MATRIX_CACHE
        with open(MATRIX_CACHE, "rb") as f:
            matrix = pickle.load(f)
        cor = corrected_oof(oof, matrix)
        c = cor.drop(columns="q_red").merge(res, on="fight_id").merge(bfo, on="fight_id") \
            .dropna(subset=[f"mkt_{c}" for c in CELLS])
        c = c.merge(cor[["fight_id", "q_red"]], on="fight_id")
        c = c[c["q_red"].notna()].reset_index(drop=True)
        r = b.set_index("fight_id").loc[c["fight_id"]].reset_index()
        overall, bands = compare_corrected(r, c)
        print(f"\nv2_corr vs v2 vs market, {CORR_FROM.date()} on, priced fights:")
        print(overall.round(4).to_string(index=False))
        print("\ndecision log loss by market-favourite band:")
        print(bands.round(4).to_string())


if __name__ == "__main__":
    main()
