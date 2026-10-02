"""Round scoring with fighter, judge and home effects (steps 0-3 of the round-model ladder).

Each row is one judge's verdict on one round (mmadecisions verified cards joined to the
UFCStats round stats; scores BEFORE referee deductions; 10-10s dropped). The question
for every step: does it predict judges' round verdicts in LATER fights better?

  step 0  linear model on 13 raw stat differences      (scripts.round_model)
  step 1  richer round features: shares, diminishing returns (signed log), accuracy,
          position (distance / clinch / ground), knockdown indicator. Linear (1a) and
          gradient boosting (1b); the better one's logit is the stats offset for 2-3.
  step 2  + fighter effects: one column per fighter, +1 when red, -1 when blue, so a
          positive value = judges give this fighter more than the stats say
  step 3  + judge effects (each judge's own weight on striking / control / takedowns /
          knockdowns, and a red-corner lean) + home effect (fighter's country = event's)

Fighter and judge columns are L2-penalised with their own strength (column scaling in
one LogisticRegression; equivalent to Gaussian random effects at the MAP), so the data
sets the shrinkage. Strengths are tuned on 2021-22 with training before 2021; the final
scoreboard trains on everything before 2023 and scores 2023+.

    DATABASE_URL=postgresql://localhost/alocks_local PYTHONPATH=. \
        venv/bin/python -m scripts.round_effects
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sqlalchemy import create_engine, text

from scripts.round_model import STATS as BASE_STATS

RAW = ["kd", "sig_str_landed", "sig_str_attempted", "total_str_landed", "total_str_attempted",
       "head_landed", "body_landed", "leg_landed", "distance_landed", "clinch_landed",
       "ground_landed", "td_landed", "td_attempted", "sub_att", "rev", "ctrl_seconds"]
COUNTRY = {
    "USA": "US", "Brazil": "BR", "Canada": "CA", "United Kingdom": "GB", "England": "GB",
    "Scotland": "GB", "Wales": "GB", "Northern Ireland": "GB", "United Arab Emirates": "AE",
    "Australia": "AU", "China": "CN", "Mexico": "MX", "Japan": "JP", "Germany": "DE",
    "Singapore": "SG", "Sweden": "SE", "France": "FR", "New Zealand": "NZ", "Ireland": "IE",
    "Russia": "RU", "Poland": "PL", "Saudi Arabia": "SA", "Netherlands": "NL",
    "South Korea": "KR", "Azerbaijan": "AZ", "Puerto Rico": "PR", "Chile": "CL",
    "Argentina": "AR", "Denmark": "DK", "Uruguay": "UY", "Serbia": "RS", "Qatar": "QA",
    "Croatia": "HR", "Philippines": "PH", "Czech Republic": "CZ",
}
#: Judge-specific weights on these standardised differences (the "lean" axes).
JUDGE_AXES = ["d_sig_str_landed", "d_ctrl_seconds", "d_td_landed", "d_kd"]


# ----------------------------------------------------------------------- data


def load(eng) -> pd.DataFrame:
    cols = ", ".join(f"s.{c}" for c in RAW)
    st = pd.read_sql(text(f"""
        SELECT s.fight_id, s.round_number AS round, s.fighter_id, {cols}
        FROM ufc.ufc_fight_stats s WHERE s.round_number >= 1
    """), eng)
    st[RAW] = st[RAW].astype(float).fillna(0.0)
    cards = pd.read_sql(text("""
        SELECT c.fight_id, c.round, c.judge_id, c.red_pts + c.red_ded AS rp,
               c.blue_pts + c.blue_ded AS bp, COALESCE(f.date, e.date) AS date,
               f.red_fighter_id AS red, f.blue_fighter_id AS blue,
               rf.country_code AS red_cc, bf.country_code AS blue_cc, e.location
        FROM ufc.ufc_judge_scorecards c
        JOIN ufc.mmad_decisions d ON d.mmad_decision_id = c.mmad_decision_id
        JOIN ufc.ufc_fights f ON f.id = c.fight_id
        JOIN ufc.ufc_events e ON e.id = f.event_id
        JOIN ufc.ufc_fighters rf ON rf.id = f.red_fighter_id
        JOIN ufc.ufc_fighters bf ON bf.id = f.blue_fighter_id
        WHERE c.source = 'mmad' AND c.round >= 1 AND d.match_status = 'verified'
          AND c.red_pts IS NOT NULL AND c.blue_pts IS NOT NULL AND c.judge_id IS NOT NULL
    """), eng)
    cards = cards[cards.rp != cards.bp].copy()
    cards["y"] = (cards.rp > cards.bp).astype(int)
    cards["date"] = pd.to_datetime(cards["date"]).dt.date

    red = st.rename(columns={c: f"r_{c}" for c in RAW}).rename(columns={"fighter_id": "red"})
    blue = st.rename(columns={c: f"b_{c}" for c in RAW}).rename(columns={"fighter_id": "blue"})
    df = cards.merge(red, on=["fight_id", "round", "red"]).merge(blue, on=["fight_id", "round", "blue"])

    ev_cc = df.location.fillna("").str.split(",").str[-1].str.strip().map(COUNTRY)
    def home(cc):
        return cc.fillna("").str[:2].eq(ev_cc.fillna("--")).astype(float)
    df["home"] = home(df.red_cc) - home(df.blue_cc)
    return df.reset_index(drop=True)


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"d_{c}": df[f"r_{c}"] - df[f"b_{c}"] for c in BASE_STATS})


def rich_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({f"d_{c}": df[f"r_{c}"] - df[f"b_{c}"] for c in RAW})
    for c in ("sig_str_landed", "head_landed", "total_str_landed", "ground_landed",
              "ctrl_seconds", "body_landed", "leg_landed"):
        d = f[f"d_{c}"]
        f[f"slog_{c}"] = np.sign(d) * np.log1p(np.abs(d))
    for c in ("sig_str_landed", "head_landed", "total_str_landed", "ctrl_seconds", "td_landed"):
        r, b = df[f"r_{c}"], df[f"b_{c}"]
        f[f"share_{c}"] = np.where(r + b > 0, r / (r + b).replace(0, 1), 0.5) - 0.5
    for l, a in (("sig_str_landed", "sig_str_attempted"), ("td_landed", "td_attempted")):
        f[f"acc_{l}"] = df[f"r_{l}"] / (df[f"r_{a}"] + 1) - df[f"b_{l}"] / (df[f"b_{a}"] + 1)
    f["kd_any"] = (df.r_kd > 0).astype(float) - (df.b_kd > 0).astype(float)
    # Both must flip sign when the corners swap (every feature here is red-minus-blue).
    # Control that came with ground strikes, per fighter, then differenced.
    f["ctrl_x_ground"] = (df.r_ctrl_seconds * np.tanh(df.r_ground_landed / 5)
                          - df.b_ctrl_seconds * np.tanh(df.b_ground_landed / 5))
    # Missed significant strikes: volume that didn't land.
    f["sig_missed"] = ((df.r_sig_str_attempted - df.r_sig_str_landed)
                       - (df.b_sig_str_attempted - df.b_sig_str_landed))
    return f


# --------------------------------------------------------------------- models


def _std(train: np.ndarray, *others: np.ndarray):
    mu, sd = train.mean(0), train.std(0) + 1e-9
    return [(x - mu) / sd for x in (train, *others)]


def fit_linear(Xtr, ytr, Xs: list, C: float = 1.0):
    Ztr, *Zs = _std(Xtr, *Xs)
    m = LogisticRegression(C=C, max_iter=5000).fit(Ztr, ytr)
    return [m.predict_proba(z)[:, 1] for z in Zs]


def fit_hgb(Xtr, ytr, Xs: list, seed: int = 0):
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=80, l2_regularization=1.0,
                                       random_state=seed)
    # Train on both corners (swap red/blue -> negate diffs, centre shares, flip y) so the
    # model is symmetric apart from what the linear intercept learns.
    m.fit(np.vstack([Xtr, -Xtr]), np.concatenate([ytr, 1 - ytr]))
    return [0.5 * (m.predict_proba(x)[:, 1] + 1 - m.predict_proba(-x)[:, 1]) for x in Xs]


def oof(fitter, X, y, groups, folds: int = 5) -> np.ndarray:
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, groups=groups):
        out[te] = fitter(X[tr], y[tr], [X[te]])[0]
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class EffectsDesign:
    """Sparse columns for fighter (+1 red / -1 blue), judge lean, home; fitted on train ids."""

    def __init__(self, train: pd.DataFrame, axes: np.ndarray):
        self.fighters = pd.Index(pd.unique(pd.concat([train.red, train.blue])))
        self.judges = pd.Index(pd.unique(train.judge_id))
        self.axes_sd = axes.std(0) + 1e-9

    def fighter_block(self, df):
        n = len(df)
        ri, bi = self.fighters.get_indexer(df.red), self.fighters.get_indexer(df.blue)
        rows = np.concatenate([np.arange(n)[ri >= 0], np.arange(n)[bi >= 0]])
        cols = np.concatenate([ri[ri >= 0], bi[bi >= 0]])
        vals = np.concatenate([np.ones((ri >= 0).sum()), -np.ones((bi >= 0).sum())])
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, len(self.fighters)))

    def judge_block(self, df, axes):
        n, k = len(df), axes.shape[1] + 1
        ji = self.judges.get_indexer(df.judge_id)
        ok = ji >= 0
        z = np.hstack([np.ones((n, 1)), axes / self.axes_sd])          # lean on red + axes
        rows = np.repeat(np.arange(n)[ok], k)
        cols = (ji[ok][:, None] * k + np.arange(k)[None, :]).ravel()
        return sp.csr_matrix((z[ok].ravel(), (rows, cols)), shape=(n, len(self.judges) * k))


class OffsetLogit:
    """L2-penalised logistic regression with a fixed offset (weight exactly 1) and an
    unpenalised intercept: P(y=1) = sigmoid(offset + c + X b), penalty |b|^2 / 2.
    scikit-learn can't pin a coefficient; letting it re-scale the stats offset made the
    model overconfident on test (cross-fitted train offsets are blunter than test ones)."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, off, y):
        from scipy.optimize import minimize
        y = np.asarray(y, float)
        X = sp.csr_matrix(X)
        n, k = X.shape

        def f(w):
            c, b = (w[0] if self.fit_intercept else 0.0), w[1:]
            z = off + c + (X @ b if k else 0.0)
            ll = np.sum(np.logaddexp(0.0, z) - y * z)
            r = 1.0 / (1.0 + np.exp(-z)) - y
            grad = np.concatenate([[r.sum() if self.fit_intercept else 0.0],
                                   (X.T @ r if k else np.zeros(0)) + b])
            return ll + 0.5 * b @ b, grad

        w = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B",
                     options={"maxiter": 5000}).x
        self.intercept_ = w[0] if self.fit_intercept else 0.0
        self.b = w[1:]
        # Same layout the callers read: [offset weight, effect columns...]
        self.coef_ = np.concatenate([[1.0], self.b]).reshape(1, -1)
        return self

    def predict(self, X, off):
        z = off + self.intercept_ + (sp.csr_matrix(X) @ self.b if len(self.b) else 0.0)
        return 1.0 / (1.0 + np.exp(-z))


#: Red-corner intercept? The red corner's edge with judges beyond the stats fell from
#: +0.35 log-odds (2009-12) to +0.08 (2023+) (and 2001-08 is a UFCStats artifact: red =
#: winner), so one intercept fitted on history overshoots on new fights. Corner-neutral
#: by default; fighter effects carry any "favourite" quality.
FIT_CORNER = False


def fit_effects(tr, te, off_tr, off_te, axes_tr, axes_te, s_f: float, s_j: float,
                s_h: float, return_model: bool = False):
    """Stats offset (fixed weight 1) + [fighters*s_f | judges*s_j | home*s_h], each column
    with an L2 penalty of 1, so a column scaled by s has prior SD ~ s on its own scale."""
    d = EffectsDesign(tr, axes_tr)
    def X(df, axes):
        blocks = []
        if s_f:
            blocks.append(d.fighter_block(df) * s_f)
        if s_j:
            blocks.append(d.judge_block(df, axes) * s_j)
        if s_h:
            blocks.append(sp.csr_matrix(df.home.to_numpy().reshape(-1, 1) * s_h))
        return sp.hstack(blocks).tocsr() if blocks else sp.csr_matrix((len(df), 0))
    m = OffsetLogit(fit_intercept=FIT_CORNER).fit(X(tr, axes_tr), off_tr, tr.y)
    p = m.predict(X(te, axes_te), off_te)
    return (p, m, d) if return_model else p


# --------------------------------------------------------------------- driver


def _ll(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


def stats_offsets(tr, te, kind: str):
    """Cross-fitted stats logit on train (so effects aren't fitted to in-sample fit) and
    the full-train model's logit on test."""
    X = rich_features if kind != "base" else base_features
    Xtr, Xte = X(tr).to_numpy(), X(te).to_numpy()
    fitter = fit_hgb if kind == "hgb" else fit_linear
    return (_logit(oof(fitter, Xtr, tr.y.to_numpy(), tr.fight_id.to_numpy())),
            _logit(fitter(Xtr, tr.y.to_numpy(), [Xte])[0]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-split", type=dt.date.fromisoformat, default=dt.date(2021, 1, 1))
    ap.add_argument("--split", type=dt.date.fromisoformat, default=dt.date(2023, 1, 1))
    ap.add_argument("--out-dir", type=Path, default=Path("data/mmad"))
    ap.add_argument("--offset", choices=("auto", "linear", "hgb"), default="auto",
                    help="stats model under the effects (auto = best on the tuning split)")
    args = ap.parse_args(argv)
    eng = create_engine(os.environ["DATABASE_URL"])
    df = load(eng)
    print(f"judge-rounds: {len(df):,}  fights: {df.fight_id.nunique():,}  "
          f"judges: {df.judge_id.nunique():,}  home-coded rows: {(df.home != 0).mean():.1%}")

    tune_tr = df[df.date < args.tune_split]
    tune_va = df[(df.date >= args.tune_split) & (df.date < args.split)]
    tr, te = df[df.date < args.split], df[df.date >= args.split]
    seen = te.red.isin(pd.concat([tr.red, tr.blue])) & te.blue.isin(pd.concat([tr.red, tr.blue]))

    def axes(d):
        return rich_features(d)[JUDGE_AXES].to_numpy()

    # --- step 1: choose the stats model on the tuning split
    board = {}
    for kind in ("base", "linear", "hgb"):
        _, va_off = stats_offsets(tune_tr, tune_va, kind)
        board[kind] = _ll(tune_va.y, 1 / (1 + np.exp(-va_off)))
    best = min(("linear", "hgb"), key=board.get) if args.offset == "auto" else args.offset
    print(f"\ntuning split ({args.tune_split} .. {args.split}): stats-only log loss "
          + "  ".join(f"{k} {v:.4f}" for k, v in board.items()) + f"  -> offset: {best}")

    # --- tune effect strengths on the tuning split (coordinate search)
    tr_off, va_off = stats_offsets(tune_tr, tune_va, best)
    ax_tr, ax_va = axes(tune_tr), axes(tune_va)
    grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    s = {"f": 0.0, "j": 0.0, "h": 0.0}
    for key in ("f", "j", "h", "f"):
        scores = {}
        for v in grid:
            t = dict(s, **{key: v})
            p = fit_effects(tune_tr, tune_va, tr_off, va_off, ax_tr, ax_va, t["f"], t["j"], t["h"])
            scores[v] = _ll(tune_va.y, p)
        s[key] = min(scores, key=scores.get)
        print(f"  tune s_{key}: " + "  ".join(f"{v}:{sc:.4f}" for v, sc in scores.items())
              + f"  -> {s[key]}")

    # --- final scoreboard: train < split, test >= split
    print(f"\nSCOREBOARD  train < {args.split}  test >= {args.split}: {len(te):,} judge-rounds "
          f"({seen.mean():.0%} with both fighters seen in training)")
    rows = []
    for kind in ("base", "linear", "hgb"):
        o_tr, o_te = stats_offsets(tr, te, kind)
        rows.append((f"step {'0' if kind == 'base' else '1' + ('a' if kind == 'linear' else 'b')}"
                     f"  stats only ({kind})", 1 / (1 + np.exp(-o_te))))
        if kind == best:
            off_tr, off_te = o_tr, o_te
    ax_tr2, ax_te2 = axes(tr), axes(te)
    rows.append(("step 2  + fighter effects",
                 fit_effects(tr, te, off_tr, off_te, ax_tr2, ax_te2, s["f"], 0, 0)))
    rows.append(("step 3  + judge leans + home",
                 fit_effects(tr, te, off_tr, off_te, ax_tr2, ax_te2, s["f"], s["j"], s["h"])))
    rows.append(("  (3 without fighters)",
                 fit_effects(tr, te, off_tr, off_te, ax_tr2, ax_te2, 0, s["j"], s["h"])))
    y, ys = te.y.to_numpy(), te.y[seen].to_numpy()
    base_ll = _ll(y, rows[0][1])
    print(f"  {'model':<34} {'log loss':>9} {'vs step0':>9} {'acc':>6} {'LL seen':>9}")
    for name, p in rows:
        print(f"  {name:<34} {_ll(y, p):>9.4f} {_ll(y, p) - base_ll:>+9.4f} "
              f"{((p > .5) == y).mean():>6.3f} {_ll(ys, p[seen.to_numpy()]):>9.4f}")
    # Bootstrap the step-2/3 gain over step 1 by fight, so the CI respects clustering.
    p1 = next(p for n, p in rows if n.startswith("step 1") and best in n)
    for name, p in rows[3:5]:
        diff = pd.Series(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
                         + (y * np.log(np.clip(p1, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p1, 1e-6, 1))),
                         index=te.fight_id.to_numpy())
        by_fight = diff.groupby(level=0).agg(["sum", "size"])
        rng = np.random.default_rng(0)
        idx = np.arange(len(by_fight))
        boots = []
        for _ in range(1000):
            b = by_fight.iloc[rng.choice(idx, len(idx))]
            boots.append(b["sum"].sum() / b["size"].sum())
        print(f"  {name.strip():<34} gain over step 1: {diff.mean():+.4f}  "
              f"95% CI [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")

    # --- fit on everything for the estimates people look at
    all_off, _ = stats_offsets(df, df.iloc[:1], best)
    _, m, d = fit_effects(df, df.iloc[:1], all_off, all_off[:1], axes(df), axes(df.iloc[:1]),
                          s["f"], s["j"], s["h"], return_model=True)
    coef = m.coef_[0]
    nf, nj = len(d.fighters) if s["f"] else 0, len(d.judges) if s["j"] else 0
    k = len(JUDGE_AXES) + 1
    names = pd.read_sql(text("SELECT id, first_name || ' ' || last_name AS name FROM ufc.ufc_fighters"), eng)
    jn = pd.read_sql(text("SELECT id, name FROM ufc.ufc_judges"), eng)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if nf:
        rounds = pd.concat([df.red, df.blue]).value_counts()
        fe = pd.DataFrame({"fighter_id": d.fighters, "effect_logit": coef[1:1 + nf] * s["f"]})
        fe["rounds"] = fe.fighter_id.map(rounds)
        # Log-odds at p=0.5 -> extra rounds won per 3 rounds.
        fe["rounds_per_3"] = 3 * 0.25 * fe.effect_logit
        fe = fe.merge(names, left_on="fighter_id", right_on="id").drop(columns="id")
        fe.sort_values("effect_logit", ascending=False).to_csv(args.out_dir / "fighter_effects.csv", index=False)
        show = fe[fe.rounds >= 20].sort_values("effect_logit")
        print("\nfighter effects (>= 20 judged rounds; extra rounds won per 3 vs stats):")
        print("  top:    " + "; ".join(f"{r['name']} {r.rounds_per_3:+.2f}" for _, r in show.tail(8).iloc[::-1].iterrows()))
        print("  bottom: " + "; ".join(f"{r['name']} {r.rounds_per_3:+.2f}" for _, r in show.head(8).iterrows()))
        print(f"  spread: SD {fe.rounds_per_3.std():.3f} rounds per 3")
    if nj:
        jc = coef[1 + nf:1 + nf + nj * k].reshape(nj, k) * s["j"]
        je = pd.DataFrame(jc, columns=["red_lean"] + [f"lean_{a[2:]}" for a in JUDGE_AXES])
        je["judge_id"] = d.judges
        je["rounds"] = je.judge_id.map(df.judge_id.value_counts())
        je = je.merge(jn, left_on="judge_id", right_on="id").drop(columns="id")
        je.sort_values("rounds", ascending=False).to_csv(args.out_dir / "judge_effects.csv", index=False)
        print("\nbusiest judges' leans (log-odds per SD of the difference, vs the average judge):")
        print(je.sort_values("rounds", ascending=False).head(12)[
            ["name", "rounds", "red_lean", "lean_sig_str_landed", "lean_ctrl_seconds",
             "lean_td_landed", "lean_kd"]].round(3).to_string(index=False))
    if s["h"]:
        print(f"\nhome effect: {coef[-1] * s['h']:+.3f} log-odds per round "
              f"(= {3 * 0.25 * coef[-1] * s['h']:+.2f} rounds per 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
