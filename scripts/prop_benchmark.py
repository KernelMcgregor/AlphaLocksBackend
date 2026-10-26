"""Every prop market: does the model beat, match or trail the BFO consensus close? Read-only.

Model probabilities are out-of-sample (each fight predicted by a model that hadn't seen it):
  method  models/ufc/method/method_oof.csv   P(method | corner wins) per corner (2022-06 on), turned
          into joint cells with the market's de-vigged moneyline for who wins (isolates what
          the model knows about HOW fights end; the site combines them the same way once priced)
  rounds  models/ufc/method/rounds_oof.csv   survival S(t) in fight minutes (anchored)
Market: ufc_prop_odds_history source='bfo_close' (de-vigged consensus close).

Per market (log loss; lower is better):
  market       the close
  model        the model alone
  blend50      the 50/50 log-odds blend the picks page uses (grading.blend_prob)
  stack        logistic on [logit market, logit model], fit on the EARLIER half of fights and
               scored on the later half (so the weight is honest); market scored on the same half
CIs are paired bootstraps on the fight-level difference vs the market.

    PYTHONPATH=. venv/bin/python -m scripts.prop_benchmark [--by-fav]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import text

from app.database import engine

CELLS = ("red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec")
# binary market -> (model column expression, outcome name)
BINARY = {
    "dec_yes": "decision", "itd_red_yes": "red_itd", "itd_blue_yes": "blue_itd",
    "ou_1.5_over": "over_1.5", "ou_2.5_over": "over_2.5", "sr_2": "starts_2", "sr_3": "starts_3",
}


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def load():
    props = pd.read_sql(text("SELECT fight_id, market, prob FROM ufc.ufc_prop_odds_history WHERE source='bfo_close'"), engine)
    props = props.pivot_table(index="fight_id", columns="market", values="prob", aggfunc="last")
    f = pd.read_sql(text("""SELECT f.id AS fight_id, f.red_fighter_id, f.winner_id, f.method, f.fight_time_seconds, f.date
                            FROM ufc.ufc_fights f WHERE f.winner_id IS NOT NULL"""), engine).set_index("fight_id")
    m = pd.read_csv("models/ufc/method/method_oof.csv").set_index("fight_id")
    r = pd.read_csv("models/ufc/method/rounds_oof.csv").set_index("fight_id")
    import pickle
    from app.services.ufc.market_anchor import devig
    mx = pickle.load(open("models/ufc/h2h/feature_wf_matrix_v2.pkl", "rb"))[0]
    mx = mx[mx.odds_red_prob.notna()].set_index("fight_id")
    wp = pd.Series(devig(mx.odds_red_prob.values, mx.odds_blue_prob.values), index=mx.index, name="wp_red")
    d = f.join(props, how="inner").join(m.add_prefix("c_"), how="left").join(r, how="left").join(wp, how="inner")
    for c in CELLS:     # joint = P(corner wins) x P(method | corner wins)
        w = d["wp_red"] if c.startswith("red") else 1 - d["wp_red"]
        d[f"m_{c}"] = w * d[f"c_{c}"]
    meth = d["method"].fillna("").str.upper()
    cls = np.where(meth.str.contains("KO"), "ko", np.where(meth.str.contains("SUB"), "sub", "dec"))
    corner = np.where(d["winner_id"] == d["red_fighter_id"], "red", "blue")
    d["cell"] = [f"{c}_{k}" for c, k in zip(corner, cls)]
    t = d["fight_time_seconds"] / 60
    d["y_decision"] = (cls == "dec").astype(float)
    d["y_red_itd"] = ((corner == "red") & (cls != "dec")).astype(float)
    d["y_blue_itd"] = ((corner == "blue") & (cls != "dec")).astype(float)
    d["y_over_1.5"], d["y_over_2.5"] = (t > 7.5).astype(float), (t > 12.5).astype(float)
    d["y_starts_2"], d["y_starts_3"] = (t > 5).astype(float), (t > 10).astype(float)
    d["p_decision"] = d["m_red_dec"] + d["m_blue_dec"]
    d["p_red_itd"] = d["m_red_ko"] + d["m_red_sub"]
    d["p_blue_itd"] = d["m_blue_ko"] + d["m_blue_sub"]
    d["p_over_1.5"], d["p_over_2.5"] = d["S_7.5"], d["S_12.5"]
    d["p_starts_2"], d["p_starts_3"] = d["S_5"], d["S_10"]
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date")


def score(name, q, p, y, rows):
    """q market, p model, y outcome (aligned arrays). Earlier half fits the stack."""
    n = len(y)
    h = n // 2
    blend = 1 / (1 + np.exp(-(0.5 * lg(q) + 0.5 * lg(p))))
    X = np.column_stack([lg(q), lg(p)])
    st = LogisticRegression(C=10).fit(X[:h], y[:h])
    ps = st.predict_proba(X[h:])[:, 1]
    lm, lp, lb = ll(q, y), ll(p, y), ll(blend, y)
    ls, lm2 = ll(ps, y[h:]), ll(q[h:], y[h:])
    rows.append(dict(market=name, n=n, market_ll=lm.mean(), model_ll=lp.mean(),
                     model_vs_mkt=(lp - lm).mean(), model_ci=ci(lp - lm),
                     blend_vs_mkt=(lb - lm).mean(), blend_ci=ci(lb - lm),
                     stack_vs_mkt=(ls - lm2).mean(), stack_ci=ci(ls - lm2),
                     w_model=st.coef_[0][1], w_mkt=st.coef_[0][0]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-fav", action="store_true", help="split binary markets by moneyline favourite band")
    a = ap.parse_args(argv)
    d = load()
    rows = []
    for mk, oc in BINARY.items():
        s = d[[mk, f"p_{oc}", f"y_{oc}"]].dropna()
        score(mk, s[mk].to_numpy(), s[f"p_{oc}"].to_numpy(), s[f"y_{oc}"].to_numpy(), rows)
    # six-way: multiclass, market cells renormalised
    s = d.dropna(subset=list(CELLS) + [f"m_{c}" for c in CELLS])
    Q = s[list(CELLS)].to_numpy(); Q = Q / Q.sum(1, keepdims=True)
    P = s[[f"m_{c}" for c in CELLS]].to_numpy(); P = P / P.sum(1, keepdims=True)
    idx = np.array([CELLS.index(c) for c in s["cell"]])
    lq = -np.log(np.clip(Q[np.arange(len(s)), idx], 1e-4, 1)); lp = -np.log(np.clip(P[np.arange(len(s)), idx], 1e-4, 1))
    B = np.exp(0.5 * np.log(np.clip(Q, 1e-4, 1)) + 0.5 * np.log(np.clip(P, 1e-4, 1))); B /= B.sum(1, keepdims=True)
    lb = -np.log(np.clip(B[np.arange(len(s)), idx], 1e-4, 1))
    rows.append(dict(market="six-way (multiclass)", n=len(s), market_ll=lq.mean(), model_ll=lp.mean(),
                     model_vs_mkt=(lp - lq).mean(), model_ci=ci(lp - lq), blend_vs_mkt=(lb - lq).mean(),
                     blend_ci=ci(lb - lq), stack_vs_mkt=np.nan, stack_ci=(np.nan, np.nan), w_model=np.nan, w_mkt=np.nan))
    # each six-way cell as a yes/no market
    for c in CELLS:
        sc = s[[c, f"m_{c}"]].copy(); y = (s["cell"] == c).astype(float).to_numpy()
        score(f"cell {c}", sc[c].to_numpy(), sc[f"m_{c}"].to_numpy(), y, rows)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    f = lambda x: f"[{x[0]:+.4f},{x[1]:+.4f}]"
    print(f"{'market':22s} {'n':>5s} {'mkt':>7s} {'model':>7s} {'model-mkt':>10s} {'CI':>18s} {'blend-mkt':>10s} {'CI':>18s} {'stack-mkt':>10s} {'CI':>18s} {'w_model':>7s}")
    for r in out.itertuples():
        print(f"{r.market:22s} {r.n:5d} {r.market_ll:7.4f} {r.model_ll:7.4f} {r.model_vs_mkt:+10.4f} {f(r.model_ci):>18s} "
              f"{r.blend_vs_mkt:+10.4f} {f(r.blend_ci):>18s} {r.stack_vs_mkt:+10.4f} {f(r.stack_ci):>18s} {r.w_model:7.2f}")
    out.to_csv("data/feature_wf/prop_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
