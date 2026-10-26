"""Where does the winner model lose to the market, and where does it add to it? Read-only."""
import numpy as np, pandas as pd
from scripts.feature_wf import build_matrix
from app.services.ufc.market_anchor import devig
m, f, slim = build_matrix(False)
w = pd.read_csv("data/feature_wf/style_wf.csv")
d = w.merge(m.drop(columns=["date", "red_wins", "odds_red_prob"]), on="fight_id")
d = d[d.odds_red_prob.notna() & d.red_wins.notna()].copy()
q = devig(d.odds_red_prob.values, d.odds_blue_prob.values)
y = d.red_wins.values
ll = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
d["gap"] = ll(d.base_ensemble_cal.values) - ll(q)      # + = model alone worse than market
d["add"] = ll(d.base_ensemble_anc.values) - ll(q)      # - = model adds to the market
d["q"], d["p"] = q, d.base_ensemble_cal.values
fav = np.maximum(q, 1 - q)
divs = {c.replace("fight_div_", ""): c for c in d.columns if c.startswith("fight_div_")}
def div_of(r):
    for k, c in divs.items():
        if r[c] == 1: return k
    return "unknown"
d["div"] = d.apply(div_of, axis=1)
grp = {"heavyweight": "HW/LHW", "light_heavyweight": "HW/LHW", "middleweight": "MW/WW", "welterweight": "MW/WW",
       "lightweight": "LW/FW", "featherweight": "LW/FW", "bantamweight": "BW/FLY", "flyweight": "BW/FLY"}
d["divg"] = d["div"].map(lambda x: "women" if x.startswith("w_") else grp.get(x, "other"))
mn = np.minimum(d.red_career_fights, d.blue_career_fights)
segs = {
    "market favourite": pd.cut(fav, [.5, .6, .7, .8, .9, 1.0], labels=["50-60", "60-70", "70-80", "80-90", "90+"], include_lowest=True),
    "least-experienced fighter (prior UFC fights)": pd.cut(mn, [-1, 0, 1, 4, 999], labels=["debut", "1", "2-4", "5+"]),
    "oldest fighter age": pd.cut(np.maximum(d.red_age, d.blue_age), [0, 30, 34, 37, 99], labels=["<=30", "31-34", "35-37", "38+"]),
    "longest layoff (days)": pd.cut(np.maximum(d.red_days_since_last, d.blue_days_since_last), [-9, 200, 365, 730, 99999], labels=["<200", "200-365", "1-2y", "2y+"]),
    "division": d["divg"],
    "five rounds": d.fight_is_five_round.map({1: "yes", 0: "no"}),
    "year": pd.to_datetime(d.date).dt.year,
    "model vs market disagreement": pd.cut(np.abs(d.p - d.q), [0, .05, .10, .15, 1], labels=["<5pt", "5-10", "10-15", "15+"], include_lowest=True),
}
rng = np.random.default_rng(0)
def ci(x):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
tot_gap = d.gap.sum()
print(f"priced fights {len(d)}: market LL {ll(q).mean():.4f}, model {ll(d.p.values).mean():.4f} (gap {d.gap.mean():+.4f}), "
      f"anchored {ll(d.base_ensemble_anc.values).mean():.4f} (adds {d['add'].mean():+.4f})")
for name, s in segs.items():
    print(f"\n== {name}")
    print(f"  {'segment':10s} {'n':>5s} {'gap':>8s} {'share':>6s}  {'adds':>8s} {'95% CI':>18s}")
    for k, g in d.groupby(s, observed=True):
        lo, hi = ci(g["add"].values)
        print(f"  {str(k):10s} {len(g):5d} {g.gap.mean():+8.4f} {g.gap.sum()/tot_gap:6.0%}  {g['add'].mean():+8.4f} [{lo:+.4f},{hi:+.4f}]")
# who is right when they disagree
print("\n== when the model disagrees with the market by 10pt+: actual vs each")
big = d[np.abs(d.p - d.q) >= .10]
for lab, sel in (("model higher on red", big.p > big.q), ("model lower on red", big.p < big.q)):
    g = big[sel]
    print(f"  {lab:20s} n={len(g):4d} market {g.q.mean():.3f} model {g.p.mean():.3f} actual {g.red_wins.mean():.3f}")
# favourite calibration
print("\n== calibration on the market favourite: actual win rate vs market vs model")
d["fav_y"] = np.where(d.q >= .5, d.red_wins, 1 - d.red_wins)
d["fav_q"], d["fav_p"] = fav, np.where(d.q >= .5, d.p, 1 - d.p)
for k, g in d.groupby(segs["market favourite"], observed=True):
    print(f"  {k:6s} n={len(g):4d} actual {g.fav_y.mean():.3f} market {g.fav_q.mean():.3f} model {g.fav_p.mean():.3f}")
