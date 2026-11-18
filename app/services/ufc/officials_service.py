"""Judge profiles: how each judge scores rounds against the deserve-to-win round model.

Every verified mmadecisions round card (the judge's own verdict, before deductions,
10-10s set aside) is lined up with `ufc_deserve_to_win.round_p_red`, the out-of-bag
P(an average judge gives red the round) from the round stats. A judge is never graded
against a number their own cards trained. The bootstrap-mean p runs a little
overconfident on judges' cards, so its logit is rescaled by one pooled slope (no intercept)
first; "expected" then means what an average judge scores on the same rounds.

Per judge:
  accuracy            share of rounds given to the model's favourite
  expected_accuracy   mean max(p, 1-p): what an average judge would hit on the same rounds
  acc_above_exp       accuracy - expected (the headline; close rounds no longer count against)
  ll_above_exp        mean log-likelihood of the verdicts minus its expectation
  agree_above_exp     agreement with the other judges on the panel minus what the shared
                      round latent (sigma_u, sigma_v from the scorer) predicts at the same p
  leans               log-odds shifts beyond the stats toward red, the betting favourite,
                      the home fighter, and the fighter the judge gave the previous round.
                      Carryover is controlled for the previous round's model logit AND the
                      other judges' verdicts on it, so a fighter who was better twice in
                      ways the stats miss isn't counted. Each with a Laplace SE.
  weights             extra weight on sig strikes / control / takedowns / knockdowns vs the
                      average judge (per SD of the difference)
  ten8                observed 10-8 rate vs an average judge's on the same rounds (pooled
                      logistic on the logit toward the side given, by era)
  deciding_cards      decisions where flipping this judge's card changes the result
  robbery_cards       cards for the official winner where the meter gives the loser >= 70%

Leans and weights come from one offset logistic regression (stats offset fixed at weight
1, judge columns L2-shrunk with strengths tuned on later fights), so judges with few
rounds sit near the average judge.

    PYTHONPATH=. venv/bin/python -m app.services.ufc.officials_service --dry-run out.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sqlalchemy import text

from app.services.ufc.deserve_to_win import latent_scale, logit, sigmoid

log = logging.getLogger(__name__)

MODEL_VERSION = "v1"
ERA = dt.date(2017, 1, 1)
#: Judges below this many judged rounds get no percentiles (still listed).
MIN_ROUNDS = 150
LEANS = ["red", "fav", "home", "carry"]
AXES = ["sig", "ctrl", "td", "kd"]
POOLED_CONTROLS = ["prev_logit", "carry_panel"]


@dataclass
class Latent:
    """Shared round + fight latent SDs from the deserve-to-win scorer."""
    sigma_u: float
    sigma_v: float

    @property
    def s_tot(self) -> float:
        return float(np.hypot(self.sigma_u, self.sigma_v))


# ----------------------------------------------------------------------- data


def load(eng) -> dict[str, pd.DataFrame]:
    from scripts.round_effects import RAW
    cols = ", ".join(f"s.{c}" for c in RAW)
    out = {}
    out["cards"] = pd.read_sql(text("""
        SELECT c.fight_id, c.round, c.judge_seq, c.judge_id, j.name AS judge,
               c.red_pts, c.blue_pts, c.red_ded, c.blue_ded,
               COALESCE(f.date, e.date) AS date, f.red_fighter_id AS red,
               f.blue_fighter_id AS blue, rf.country_code AS red_cc,
               bf.country_code AS blue_cc, e.location
        FROM ufc.ufc_judge_scorecards c
        JOIN ufc.mmad_decisions d ON d.mmad_decision_id = c.mmad_decision_id
        JOIN ufc.ufc_judges j ON j.id = c.judge_id
        JOIN ufc.ufc_fights f ON f.id = c.fight_id
        JOIN ufc.ufc_events e ON e.id = f.event_id
        JOIN ufc.ufc_fighters rf ON rf.id = f.red_fighter_id
        JOIN ufc.ufc_fighters bf ON bf.id = f.blue_fighter_id
        WHERE c.source = 'mmad' AND d.match_status = 'verified'
          AND c.red_pts IS NOT NULL AND c.blue_pts IS NOT NULL
    """), eng)
    out["stats"] = pd.read_sql(text(f"""
        SELECT s.fight_id, s.round_number AS round, s.fighter_id, {cols}
        FROM ufc.ufc_fight_stats s WHERE s.round_number >= 1
    """), eng)
    out["dtw"] = pd.read_sql(text("""
        SELECT fight_id, round_p_red, p_red, p_blue, robbery_score, official_outcome
        FROM ufc.ufc_deserve_to_win WHERE model_version = :v
    """), eng, params={"v": MODEL_VERSION})
    out["odds"] = pd.read_sql(text("""
        SELECT fight_id, AVG(red_close_prob) AS red_close_prob
        FROM ufc.ufc_fight_odds_open_close WHERE red_close_prob IS NOT NULL
        GROUP BY fight_id
    """), eng)
    out["fights"] = pd.read_sql(text("""
        SELECT f.id AS fight_id, COALESCE(f.date, e.date) AS date, e.name AS event_name,
               f.weight_class, f.red_fighter_id AS red, f.blue_fighter_id AS blue,
               f.winner_id, f.method, f.details, f.referee, f.finish_round, f.finish_time,
               rf.first_name || ' ' || rf.last_name AS red_name,
               bf.first_name || ' ' || bf.last_name AS blue_name
        FROM ufc.ufc_fights f JOIN ufc.ufc_events e ON e.id = f.event_id
        JOIN ufc.ufc_fighters rf ON rf.id = f.red_fighter_id
        JOIN ufc.ufc_fighters bf ON bf.id = f.blue_fighter_id
        WHERE f.method IS NOT NULL AND f.method <> ''
    """), eng)
    out["decisions"] = pd.read_sql(text("""
        SELECT mmad_decision_id, fight_id, swapped, fan_a, fan_b, fan_draw, referee
        FROM ufc.mmad_decisions WHERE match_status = 'verified' AND fight_id IS NOT NULL
    """), eng)
    out["media"] = pd.read_sql(text("""
        SELECT d.fight_id, d.swapped, m.pick FROM ufc.mmad_media_scores m
        JOIN ufc.mmad_decisions d ON d.mmad_decision_id = m.mmad_decision_id
        WHERE d.match_status = 'verified' AND d.fight_id IS NOT NULL
    """), eng)
    # Pre-fight ratings (Glicko snapshots are written before each bout): the loser's chin
    # and durability, the winner's power, for the stoppage model.
    out["glicko"] = pd.read_sql(text("""
        SELECT fight_id, fighter_id, ko, kod, durability FROM ufc.ufc_glicko_snapshots
    """), eng)
    out["regional"] = pd.read_sql(text("""
        SELECT referee, promotion, COUNT(DISTINCT (date, event_name, method, round, time)) AS n
        FROM ufc.sherdog_bouts WHERE referee IS NOT NULL AND referee <> ''
        GROUP BY referee, promotion
    """), eng)
    return out


def load_latent() -> Latent:
    from scripts.deserve_to_win import SCORER_PATH, _ScorerUnpickler
    with open(SCORER_PATH, "rb") as f:
        d = _ScorerUnpickler(f).load()
    return Latent(float(d["sigma_u"]), float(d["sigma_v"]))


def build_rounds(cards: pd.DataFrame, stats: pd.DataFrame, dtw: pd.DataFrame,
                 odds: pd.DataFrame) -> pd.DataFrame:
    """One row per judge per round (rounds >= 1) with the model p, the judge's verdict
    and the lean covariates. s: +1 red / -1 blue / 0 even, before deductions."""
    from scripts.round_effects import COUNTRY, JUDGE_AXES, RAW, rich_features

    c = cards[cards["round"] >= 1].copy()
    c["rp"], c["bp"] = c.red_pts + c.red_ded, c.blue_pts + c.blue_ded
    c["s"] = np.sign(c.rp - c.bp).astype(int)
    c["margin"] = (c.rp - c.bp).abs()
    c["score"] = c.rp.astype(int).astype(str) + "-" + c.bp.astype(int).astype(str)
    c["date"] = pd.to_datetime(c["date"]).dt.date

    # Model p for the round from the fight's JSON list.
    rp = {int(r.fight_id): json.loads(r.round_p_red) for r in dtw.itertuples()}
    c["p"] = [rp[f][r - 1] if f in rp and r - 1 < len(rp[f]) else np.nan
              for f, r in zip(c.fight_id, c["round"])]
    c = c[c.p.notna()].copy()

    # Betting favourite (vig-free close averaged over books), +1 red.
    fav = odds.set_index("fight_id").red_close_prob
    c["fav"] = np.sign(c.fight_id.map(fav).fillna(0.5) - 0.5)

    # Home fighter: fighter's country = the event's country (round_effects convention).
    ev_cc = c.location.fillna("").str.split(",").str[-1].str.strip().map(COUNTRY)
    def home(cc):
        return cc.fillna("").str[:2].eq(ev_cc.fillna("--")).astype(float)
    c["home"] = home(c.red_cc) - home(c.blue_cc)

    # Carryover: this judge's verdict on the previous round, and the model's logit there.
    c = c.sort_values(["fight_id", "judge_seq", "round"]).reset_index(drop=True)
    g = c.groupby(["fight_id", "judge_seq"])
    prev_ok = g["round"].shift(1) == c["round"] - 1
    c["carry"] = np.where(prev_ok, g.s.shift(1).fillna(0), 0.0)
    c["prev_logit"] = np.where(prev_ok, logit(g.p.shift(1).fillna(0.5)), 0.0)
    # The other judges' verdicts on the previous round: whatever the stats miss about
    # who was better then (and is still true now) shows up here, not as carryover.
    tot = c.groupby(["fight_id", "round"]).s.agg(["sum", "size"])
    tot.index = tot.index.set_names(["fight_id", "prev_round"])
    c["prev_round"] = c["round"] - 1
    c = c.join(tot, on=["fight_id", "prev_round"])
    n_oth = (c["size"] - 1).where(prev_ok & c["size"].notna(), 0)
    c["carry_panel"] = np.where(n_oth > 0, (c["sum"] - c.carry) / n_oth.replace(0, 1), 0.0)
    c = c.drop(columns=["sum", "size", "prev_round"])

    # Judge-axis stat differences, standardised.
    st = stats.copy()
    st[RAW] = st[RAW].astype(float).fillna(0.0)
    red = st.rename(columns={k: f"r_{k}" for k in RAW}).rename(columns={"fighter_id": "red"})
    blue = st.rename(columns={k: f"b_{k}" for k in RAW}).rename(columns={"fighter_id": "blue"})
    c = c.merge(red, on=["fight_id", "round", "red"]).merge(blue, on=["fight_id", "round", "blue"])
    ax = rich_features(c)[JUDGE_AXES].to_numpy(float)
    ax = ax / (ax.std(0) + 1e-9)
    for i, a in enumerate(AXES):
        c[f"ax_{a}"] = ax[:, i]
    c["post2017"] = c.date >= ERA
    keep = ["fight_id", "round", "judge_seq", "judge_id", "judge", "date", "red", "blue",
            "s", "margin", "score", "p", "fav", "home", "carry", "prev_logit", "carry_panel", "post2017"] + \
           [f"ax_{a}" for a in AXES]
    return c[keep].sort_values(["fight_id", "round", "judge_seq"]).reset_index(drop=True)


# -------------------------------------------------------------------- metrics


def recalibrate(r: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Rescale the model logit by one pooled slope so the average judge scores at
    expectation (the bootstrap-mean p runs slightly overconfident on judges' cards).
    No intercept: the corner stays symmetric and the model's favourite never changes."""
    from scipy.optimize import minimize_scalar
    d = r[r.s != 0]
    z, y = logit(d.p.to_numpy()), (d.s > 0).to_numpy()
    b = minimize_scalar(lambda b: np.sum(np.logaddexp(0, b * z) - y * b * z),
                        bounds=(0.3, 2.0), method="bounded").x
    out = r.copy()
    out["p_raw"] = out.p
    out["p"] = sigmoid(b * logit(out.p.to_numpy()))
    return out, float(b)


def add_ten8_baseline(r: pd.DataFrame) -> pd.DataFrame:
    """q8: P(an average judge scores the round 10-8 | the side they gave it to, era),
    a pooled logistic on the logit toward that side. Each judge is compared to it."""
    from sklearn.linear_model import LogisticRegression
    out = r.copy()
    dec = out.s != 0
    zw = np.where(out.s > 0, 1, -1) * logit(out.p.to_numpy())
    era = out.post2017.to_numpy(float)
    X = np.column_stack([zw, era, zw * era])
    m = LogisticRegression(C=1e6, max_iter=1000).fit(X[dec], (out.margin >= 2)[dec])
    out["q8"] = np.where(dec, m.predict_proba(X)[:, 1], np.nan)
    return out


def pair_agree_expected(p: np.ndarray, s_tot: float, nodes: int = 40) -> np.ndarray:
    """P(two judges agree on a round) with model p under the shared latent u ~ N(0, s_tot^2)
    (same parameterisation as deserve_to_win.judge_agreement), by Gauss-Hermite."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    q = sigmoid(latent_scale(s_tot) * logit(p)[:, None] + s_tot * x[None, :])
    return (q ** 2 + (1 - q) ** 2) @ w


def add_panel_agreement(r: pd.DataFrame, s_tot: float) -> pd.DataFrame:
    """Agreement with the other judges on each decided round (rounds with >= 2 decided cards)."""
    d = r[r.s != 0].copy()
    g = d.groupby(["fight_id", "round"]).s
    n = g.transform("size")
    red = g.transform(lambda s: (s > 0).sum())
    others = n - 1
    same = np.where(d.s > 0, red - 1, (n - red) - 1)
    d["others"] = others
    d["agree"] = np.where(others > 0, same / others.replace(0, 1), np.nan)
    d["agree_exp"] = np.where(others > 0, pair_agree_expected(d.p.to_numpy(), s_tot), np.nan)
    return r.join(d[["others", "agree", "agree_exp"]])


def round_metrics(d: pd.DataFrame) -> dict:
    """Accuracy and 10-8 numbers for one judge's (or the pool's) rounds."""
    dec = d[d.s != 0]
    p, y = dec.p.to_numpy(), (dec.s > 0).to_numpy()
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    fav_side = p != 0.5
    acc = float(((p > 0.5) == y)[fav_side].mean()) if fav_side.any() else np.nan
    exp_acc = float(np.maximum(p, 1 - p)[fav_side].mean()) if fav_side.any() else np.nan
    ll = np.where(y, np.log(pc), np.log(1 - pc)).mean()
    exp_ll = (pc * np.log(pc) + (1 - pc) * np.log(1 - pc)).mean()
    q8 = dec.q8.to_numpy()
    is8 = (dec.margin >= 2).to_numpy()
    era = dec.post2017.to_numpy(bool)
    pa = dec[dec.agree.notna()] if "agree" in dec else dec.iloc[:0]
    out = dict(
        n_rounds=int(len(d)), n_decided=int(len(dec)),
        n_fights=int(d.fight_id.nunique()),
        even_rate=float((d.s == 0).mean()) if len(d) else np.nan,
        accuracy=acc, expected_accuracy=exp_acc,
        acc_above_exp=acc - exp_acc if len(dec) else np.nan,
        ll_above_exp=float(ll - exp_ll) if len(dec) else np.nan,
        agree=float(pa.agree.mean()) if len(pa) else np.nan,
        agree_exp=float(pa.agree_exp.mean()) if len(pa) else np.nan,
    )
    out["agree_above_exp"] = out["agree"] - out["agree_exp"] if len(pa) else np.nan
    for name, m in (("pre2017", ~era), ("post2017", era)):
        out[f"ten8_{name}"] = float(is8[m].mean()) if m.any() else None
        out[f"ten8_{name}_exp"] = float(q8[m].mean()) if m.any() else None
        out[f"ten8_{name}_n"] = int(m.sum())
    return out


# ---------------------------------------------------------------------- leans


class _Design:
    """Pooled columns (weakly penalised) + per-judge lean and axis columns."""

    def __init__(self, judges: pd.Index, s_lean: float, s_axis: float, s_pool: float = 10.0):
        self.judges, self.s_lean, self.s_axis, self.s_pool = judges, s_lean, s_axis, s_pool

    @property
    def per_judge(self) -> list[str]:
        return LEANS + AXES

    def matrix(self, d: pd.DataFrame) -> sp.csr_matrix:
        n = len(d)
        lean = np.column_stack([np.ones(n), d.fav, d.home, d.carry]).astype(float)
        axes = d[[f"ax_{a}" for a in AXES]].to_numpy(float)
        pooled = np.column_stack([lean, d.prev_logit.to_numpy(float),
                                  d.carry_panel.to_numpy(float)]) * self.s_pool
        z = np.hstack([lean * self.s_lean, axes * self.s_axis])
        k = z.shape[1]
        ji = self.judges.get_indexer(d.judge_id)
        ok = ji >= 0
        rows = np.repeat(np.arange(n)[ok], k)
        cols = (ji[ok][:, None] * k + np.arange(k)[None, :]).ravel()
        J = sp.csr_matrix((z[ok].ravel(), (rows, cols)), shape=(n, len(self.judges) * k))
        return sp.hstack([sp.csr_matrix(pooled), J]).tocsr()


def _fit(X, off, y):
    from scripts.round_effects import OffsetLogit
    return OffsetLogit(fit_intercept=False).fit(X, off, y)


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune_strengths(d: pd.DataFrame, split: dt.date, grid=(0.05, 0.1, 0.2, 0.3, 0.5)) -> tuple[float, float]:
    """Coordinate search for the lean / axis shrinkage on fights from `split` on."""
    tr, va = d[d.date < split], d[d.date >= split]
    judges = pd.Index(pd.unique(tr.judge_id))
    s = {"lean": 0.1, "axis": 0.1}
    for key in ("lean", "axis", "lean"):
        scores = {}
        for v in grid:
            t = dict(s, **{key: v})
            des = _Design(judges, t["lean"], t["axis"])
            m = _fit(des.matrix(tr), logit(tr.p.to_numpy()), (tr.s > 0).to_numpy())
            scores[v] = _ll((va.s > 0).to_numpy(), m.predict(des.matrix(va), logit(va.p.to_numpy())))
        s[key] = min(scores, key=scores.get)
        log.info("tune s_%s: %s -> %s", key,
                 "  ".join(f"{v}:{sc:.5f}" for v, sc in scores.items()), s[key])
    return s["lean"], s["axis"]


def fit_leans(d: pd.DataFrame, s_lean: float, s_axis: float) -> tuple[pd.DataFrame, dict]:
    """Per-judge effects (log-odds; weights per SD of the stat difference) with Laplace SEs,
    and the pooled effects. Effects are deviations from the average judge, except the
    pooled ones which are the average judge's own shift beyond the stats."""
    d = d[d.s != 0]
    judges = pd.Index(pd.unique(d.judge_id))
    des = _Design(judges, s_lean, s_axis)
    X = des.matrix(d)
    off = logit(d.p.to_numpy())
    m = _fit(X, off, (d.s > 0).to_numpy())
    b = m.b
    # Laplace: H = X' W X + I (penalty 1 on every scaled column).
    q = m.predict(X, off)
    H = (X.T @ sp.diags(q * (1 - q)) @ X).toarray() + np.eye(X.shape[1])
    se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    names = LEANS + POOLED_CONTROLS
    npool = len(names)
    pooled = {name: dict(effect=float(b[i] * des.s_pool), se=float(se[i] * des.s_pool))
              for i, name in enumerate(names)}
    k = len(des.per_judge)
    scale = np.array([s_lean] * len(LEANS) + [s_axis] * len(AXES))
    B = b[npool:].reshape(len(judges), k) * scale
    S = se[npool:].reshape(len(judges), k) * scale
    rows = pd.DataFrame(B, columns=[f"{c}" for c in des.per_judge])
    for i, c in enumerate(des.per_judge):
        rows[f"{c}_se"] = S[:, i]
    rows.insert(0, "judge_id", judges)
    return rows, pooled


# ------------------------------------------------------------------- fights


def fight_cards(cards: pd.DataFrame, dtw: pd.DataFrame, robbery_at: float = 0.7,
                strong: float = 0.7) -> pd.DataFrame:
    """One row per judge per decision: their winner, whether it was pivotal, and how it
    sits with the deserve-to-win meter. Official totals include deductions."""
    c = cards[cards["round"] >= 1]
    tot = c.groupby(["fight_id", "judge_seq", "judge_id"])[["red_pts", "blue_pts"]].sum().reset_index()
    tot["w"] = np.sign(tot.red_pts - tot.blue_pts).astype(int)
    g = tot.groupby("fight_id").w
    tot["n_red"] = g.transform(lambda w: (w > 0).sum())
    tot["n_blue"] = g.transform(lambda w: (w < 0).sum())
    tot["n_cards"] = g.transform("size")

    def result(nr, nb):
        return np.where(nr >= 2, 1, np.where(nb >= 2, -1, 0))

    tot["result"] = result(tot.n_red, tot.n_blue)
    # Flip this judge's card to the other fighter (a draw card flips to the loser's side).
    other = np.where(tot.w != 0, -tot.w, -tot.result)
    nr2 = tot.n_red - (tot.w > 0) + (other > 0)
    nb2 = tot.n_blue - (tot.w < 0) + (other < 0)
    tot["pivotal"] = (tot.n_cards == 3) & (tot.result != 0) & (result(nr2, nb2) != tot.result)
    m = dtw.set_index("fight_id")
    tot["robbery_score"] = tot.fight_id.map(m.robbery_score)
    tot["p_red"] = tot.fight_id.map(m.p_red)
    tot["p_blue"] = tot.fight_id.map(m.p_blue)
    tot["robbery_card"] = (tot.w == tot.result) & (tot.w != 0) & (tot.robbery_score >= robbery_at)
    model_side = np.where(tot.p_red >= strong, 1, np.where(tot.p_blue >= strong, -1, 0))
    tot["against_model"] = (model_side != 0) & (tot.w == -model_side)
    tot["model_strong"] = model_side != 0
    return tot


# ---------------------------------------------------------------------- build


def percentiles(df: pd.DataFrame, cols: list[str], eligible: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        ref = df.loc[eligible, c].dropna().to_numpy()
        out[f"pct_{c}"] = [float((ref < v).mean() + 0.5 * (ref == v).mean()) * 100
                           if eligible[i] and v == v and len(ref) else None
                           for i, v in zip(df.index, df[c])]
    return out


def build_profiles(data: dict[str, pd.DataFrame], lat: Latent, s_lean: float | None = None,
                   s_axis: float | None = None, tune_split: dt.date = dt.date(2022, 1, 1)) -> dict:
    r = build_rounds(data["cards"], data["stats"], data["dtw"], data["odds"])
    r, slope = recalibrate(r)
    r = add_ten8_baseline(r)
    r = add_panel_agreement(r, lat.s_tot)
    if s_lean is None or s_axis is None:
        s_lean, s_axis = tune_strengths(r[r.s != 0], tune_split)
    leans, pooled = fit_leans(r, s_lean, s_axis)
    fc = fight_cards(data["cards"], data["dtw"])

    rows = []
    for (jid, name), d in r.groupby(["judge_id", "judge"]):
        f = fc[fc.judge_id == jid]
        years = pd.Series([x.year for x in d.drop_duplicates("fight_id").date]).value_counts().sort_index()
        rows.append(dict(
            judge_id=int(jid), name=name, **round_metrics(d),
            first_date=str(min(d.date)), last_date=str(max(d.date)),
            fights_by_year={int(k): int(v) for k, v in years.items()},
            deciding_cards=int(f.pivotal.sum()), robbery_cards=int(f.robbery_card.sum()),
            against_model=int(f.against_model.sum()), model_strong=int(f.model_strong.sum()),
        ))
    prof = pd.DataFrame(rows).merge(leans, on="judge_id", how="left")
    eligible = prof.n_decided >= MIN_ROUNDS
    pct_cols = ["acc_above_exp", "agree_above_exp"] + LEANS + AXES
    prof = pd.concat([prof, percentiles(prof, pct_cols, eligible)], axis=1)
    prof["eligible"] = eligible
    prof = pd.concat([prof, style_map(prof)], axis=1)

    pool = round_metrics(r)
    pool["leans"] = pooled
    pool["s_lean"], pool["s_axis"], pool["calibration_slope"] = s_lean, s_axis, slope
    pool["n_judges"] = int(len(prof))
    pool["n_eligible"] = int(eligible.sum())
    return dict(meta=dict(model_version=MODEL_VERSION, min_rounds=MIN_ROUNDS,
                          built_at=dt.datetime.now(dt.timezone.utc).isoformat()),
                pool=pool, judges=prof, rounds=r, fight_cards=fc,
                pairs=judge_pairs(r, lat.s_tot),
                reviews=decision_reviews(data["cards"], fc, data) if "fights" in data else None,
                fights=data.get("fights"))


# ------------------------------------------------------------ decisions page


def decision_reviews(cards: pd.DataFrame, fc: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per verified decision: the meter, each judge's card, media and fan majorities
    in DB corners."""
    from app.services.ufc.outcome_types import classify_outcome
    fights = data["fights"].set_index("fight_id")
    dtw = data["dtw"].set_index("fight_id")
    dec = data["decisions"].drop_duplicates("fight_id").set_index("fight_id")
    names = cards.drop_duplicates("judge_id").set_index("judge_id").judge

    m = data["media"][data["media"].pick.isin(["a", "b", "draw"])].copy()
    sw = m.swapped.fillna(False).astype(bool)
    m["side"] = np.where(m.pick == "draw", "draw",
                         np.where((m.pick == "a") != sw, "red", "blue"))
    media = m.pivot_table(index="fight_id", columns="side", aggfunc="size", fill_value=0)

    rows = []
    for fid, g in fc.groupby("fight_id"):
        if fid not in fights.index:
            continue
        f = fights.loc[fid]
        kind = classify_outcome(f.method, f.details, f.winner_id)
        if kind not in ("ud", "sd", "md", "draw"):
            continue
        g = g.sort_values("judge_seq")
        fan = (None, None, None)
        if fid in dec.index:
            d = dec.loc[fid]
            a, b, dr = (float(x) if x == x and x is not None else 0.0 for x in (d.fan_a, d.fan_b, d.fan_draw))
            tot = a + b + dr
            if tot > 0:
                red, blue = (b, a) if bool(d.swapped) else (a, b)
                fan = (red / tot, blue / tot, dr / tot)
        md = media.loc[fid] if fid in media.index else {}
        result = 1 if f.winner_id == f.red else -1 if f.winner_id == f.blue else 0
        rows.append(dict(
            fight_id=int(fid), date=pd.to_datetime(f.date).date(), event_name=f.event_name,
            weight_class=f.weight_class, red_id=int(f.red), blue_id=int(f.blue),
            red_name=f.red_name, blue_name=f.blue_name, result=result, kind=kind,
            p_red=float(dtw.loc[fid].p_red) if fid in dtw.index else None,
            p_blue=float(dtw.loc[fid].p_blue) if fid in dtw.index else None,
            robbery_score=(float(dtw.loc[fid].robbery_score)
                           if fid in dtw.index and dtw.loc[fid].robbery_score == dtw.loc[fid].robbery_score
                           else None),
            media_red=int(md.get("red", 0)), media_blue=int(md.get("blue", 0)),
            media_draw=int(md.get("draw", 0)),
            fan_red=fan[0], fan_blue=fan[1], fan_draw=fan[2],
            cards=json.dumps([dict(judge_id=int(c.judge_id), name=names.get(c.judge_id, ""),
                                   red=int(c.red_pts), blue=int(c.blue_pts), side=int(c.w))
                              for c in g.itertuples()]),
        ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- pairs and map


def judge_pairs(r: pd.DataFrame, s_tot: float, min_rounds: int = 20) -> pd.DataFrame:
    """Pairwise agreement on shared decided rounds vs the model expectation."""
    d = r[r.s != 0][["fight_id", "round", "judge_id", "s", "p"]]
    x = d.merge(d, on=["fight_id", "round"], suffixes=("_a", "_b"))
    x = x[x.judge_id_a < x.judge_id_b]
    x["agree"] = (x.s_a == x.s_b).astype(float)
    x["expected"] = pair_agree_expected(x.p_a.to_numpy(), s_tot)
    out = x.groupby(["judge_id_a", "judge_id_b"]).agg(
        n_rounds=("agree", "size"), agree=("agree", "mean"), expected=("expected", "mean")).reset_index()
    out = out.rename(columns={"judge_id_a": "judge_a", "judge_id_b": "judge_b"})
    return out[out.n_rounds >= min_rounds].reset_index(drop=True)


def style_map(prof: pd.DataFrame) -> pd.DataFrame:
    """2D map of judges by how they score: PCA of the eight shrunk effects, fitted on
    eligible judges (judges with few rounds sit near the average judge at the centre)."""
    cols = LEANS + AXES
    X = prof[cols].fillna(0.0).to_numpy(float)
    E = X[prof.eligible.to_numpy(bool)]
    if len(E) < 3:
        return pd.DataFrame({"map_x": np.nan, "map_y": np.nan}, index=prof.index)
    sd = E.std(0) + 1e-9
    _, _, vt = np.linalg.svd(E / sd - (E / sd).mean(0), full_matrices=False)
    Z = (X / sd) @ vt[:2].T
    # Fix the signs so refits don't mirror the map: +x = more weight on strikes.
    for k in range(2):
        ref = cols.index("sig") if k == 0 else cols.index("ctrl")
        if vt[k, ref] < 0:
            Z[:, k] *= -1
    return pd.DataFrame({"map_x": Z[:, 0], "map_y": Z[:, 1]}, index=prof.index)


# ------------------------------------------------------------------ referees

#: Spellings that normalise differently but are the same referee.
REFEREE_ALIASES = {
    "big john mccarthy": "john mccarthy",
}
_STRIKE_STOPPAGE = ("KO/TKO",)


def norm_name(name: str | None) -> str:
    import re
    import unicodedata
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return REFEREE_ALIASES.get(s, s)


#: Shrinkage for a referee's stoppage index: mean residual * n / (n + K), K = noise
#: variance / between-referee variance of the residual means. Measured on prod
#: 2026-11-17 (3,691 strike stoppages, 33 referees with 20+): 0.264 / 0.0015 ≈ 170. Once
#: time in the round and era are accounted for, referees differ by only ~±4% in strikes
#: taken, so a referee needs ~170 stoppages before half of their raw mean counts.
STOPPAGE_K = 170.0


def _secs(t) -> int | None:
    try:
        m, s = str(t).split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


def stoppage_expectations(fr: pd.DataFrame, glicko: pd.DataFrame | None) -> pd.DataFrame:
    """For each strike stoppage, the significant strikes the loser would be expected to
    take in the finishing round before the stop, given how far into the round it came,
    which round, the weight class, the era, the winner's knockdowns in that round, and
    the pre-fight chin (kod), durability and power (ko) ratings of the two fighters.

    Adds `absorbed_expected` (strikes) and `absorbed_z` (log(1+taken) - log(1+expected):
    > 0 = more strikes than a similar stoppage, < 0 = fewer). Ratings come from the
    pre-fight Glicko snapshot, standardised within year (durability drifts upward over
    time); an all-zero snapshot (placeholder rounds) counts as missing."""
    fr = fr.copy()
    fr["sec"] = fr.finish_time.map(_secs).astype(float)
    fr["year"] = pd.to_datetime(fr["date"]).dt.year
    fr["loser_fid"] = np.where(fr.winner > 0, fr.blue, fr.red)
    rat = ["ko", "kod", "durability"]
    if glicko is not None and len(glicko):
        gl = glicko.copy()
        gl.loc[(gl[rat].fillna(0) == 0).all(axis=1), rat] = np.nan
        w = gl.rename(columns={"fighter_id": "winner_fid", "ko": "w_ko"})[["fight_id", "winner_fid", "w_ko"]]
        lo = gl.rename(columns={"fighter_id": "loser_fid", "kod": "l_kod", "durability": "l_dur"})[
            ["fight_id", "loser_fid", "l_kod", "l_dur"]]
        fr = fr.merge(w, on=["fight_id", "winner_fid"], how="left").merge(lo, on=["fight_id", "loser_fid"], how="left")
    else:
        fr["w_ko"] = fr["l_kod"] = fr["l_dur"] = np.nan
    cols = []
    for c in ("w_ko", "l_kod", "l_dur"):
        g = fr.groupby("year")[c]
        z = (fr[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        fr[f"z_{c}"] = z.fillna(0.0)
        fr[f"m_{c}"] = z.isna().astype(float)
        cols += [f"z_{c}", f"m_{c}"]
    X = pd.DataFrame({
        "one": 1.0,
        "log_sec": np.log1p(fr.sec.fillna(fr.sec.median())),
        "kd1": (fr.kd_round.fillna(0) == 1).astype(float),
        "kd2": (fr.kd_round.fillna(0) >= 2).astype(float),
        **{c: fr[c] for c in cols},
    })
    rnd = fr.finish_round.clip(upper=4).astype(int).astype(str)
    era = (fr.year // 3 * 3).astype(str)
    wc = fr.weight_class.fillna("?").str.replace(r"^UFC |(Interim )?Title |Bout$", "", regex=True).str.strip()
    top = wc.value_counts()
    wc = wc.where(wc.map(top) >= 30, "other")
    X = pd.concat([X, pd.get_dummies(rnd, prefix="r", drop_first=True, dtype=float),
                   pd.get_dummies(era, prefix="e", drop_first=True, dtype=float),
                   pd.get_dummies(wc, prefix="w", drop_first=True, dtype=float)], axis=1)
    y = np.log1p(fr.absorbed.to_numpy(float))
    A = X.to_numpy(float)
    lam = 1e-3 * np.eye(A.shape[1])
    lam[0, 0] = 0.0
    beta = np.linalg.solve(A.T @ A + lam, A.T @ y)
    pred = A @ beta
    fr["absorbed_expected"] = np.expm1(pred)
    fr["absorbed_z"] = y - pred
    fr.attrs["coef"] = dict(zip(X.columns, beta))
    return fr


def referee_tables(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(referees, referee_fights). UFC fights only for the rates; Sherdog bouts outside
    the UFC are counted as n_regional."""
    from scripts.round_effects import RAW  # noqa: F401  (stats frame has these columns)
    f = data["fights"].copy()
    f["name_norm"] = f.referee.map(norm_name)
    f = f[f.name_norm != ""]
    f["date"] = pd.to_datetime(f["date"]).dt.date
    f["winner"] = np.where(f.winner_id == f.red, 1, np.where(f.winner_id == f.blue, -1, 0))

    # The loser's significant strikes absorbed in the finishing round of strike stoppages.
    st = data["stats"]
    fr = f[f.method.isin(_STRIKE_STOPPAGE) & f.finish_round.notna() & (f.winner != 0)].copy()
    fr["winner_fid"] = np.where(fr.winner > 0, fr.red, fr.blue)
    k = st[["fight_id", "round", "fighter_id", "sig_str_landed"]].rename(
        columns={"round": "finish_round", "fighter_id": "winner_fid"})
    fr["finish_round"] = fr.finish_round.astype(int)
    fr = fr.merge(k, on=["fight_id", "finish_round", "winner_fid"], how="left")
    fr["absorbed"] = fr.sig_str_landed.astype(float)
    kd = st[["fight_id", "round", "fighter_id", "kd"]].rename(
        columns={"round": "finish_round", "fighter_id": "winner_fid", "kd": "kd_round"})
    fr = fr.merge(kd, on=["fight_id", "finish_round", "winner_fid"], how="left")
    fr = fr[fr.absorbed.notna()].copy()
    fr = stoppage_expectations(fr, data.get("glicko"))
    f = f.merge(fr[["fight_id", "absorbed", "absorbed_expected", "absorbed_z"]], on="fight_id", how="left")

    # Point deductions from the round cards (max over the judges in each round).
    c = data["cards"][data["cards"]["round"] >= 1]
    ded = (c.assign(d=c.red_ded + c.blue_ded).groupby(["fight_id", "round"]).d.max()
           .groupby(level=0).sum())
    f["deductions"] = f.fight_id.map(ded).fillna(0).astype(int)
    f["has_cards"] = f.fight_id.isin(ded.index)

    reg = data["regional"].copy()
    reg["name_norm"] = reg.referee.map(norm_name)
    reg = reg[~reg.promotion.fillna("").str.contains("UFC|Ultimate Fighting", case=False)]
    regional = reg.groupby("name_norm").n.sum()

    rows = []
    for i, (nn, g) in enumerate(sorted(f.groupby("name_norm"), key=lambda x: -len(x[1])), start=1):
        meth = g.method.fillna("")
        fin = meth.isin(["KO/TKO", "Submission", "TKO - Doctor's Stoppage"])
        dec = meth.str.startswith("Decision")
        z = g.absorbed_z.dropna()
        shrink = len(z) / (len(z) + STOPPAGE_K) if len(z) else 0.0
        secs = [(int(r) - 1) * 300 + _secs(t) for r, t in zip(g.finish_round[fin], g.finish_time[fin])
                if r == r and _secs(t) is not None]
        years = pd.Series([d.year for d in g.date]).value_counts().sort_index()
        cards_n = int(g.has_cards.sum())
        rows.append(dict(
            id=i, name=g.referee.mode().iloc[0], name_norm=nn, n_fights=int(len(g)),
            n_regional=int(regional.get(nn, 0)),
            finish_rate=float(fin.sum() / max(fin.sum() + dec.sum(), 1)) if (fin | dec).any() else None,
            stoppage_index=float(z.mean() * shrink) if len(z) >= 5 else None,
            deductions_per_100=float(100 * g.deductions[g.has_cards].sum() / cards_n) if cards_n >= 10 else None,
            dq_rate=float((meth == "DQ").mean()),
            first_date=min(g.date), last_date=max(g.date),
            detail=json.dumps(dict(
                fights_by_year={int(k): int(v) for k, v in years.items()},
                methods=dict(ko=int((meth == "KO/TKO").sum()), sub=int((meth == "Submission").sum()),
                             doctor=int((meth == "TKO - Doctor's Stoppage").sum()),
                             decision=int(dec.sum()), dq=int((meth == "DQ").sum()),
                             other=int((~fin & ~dec & (meth != "DQ")).sum())),
                strike_stoppages=int(len(z)),
                stoppage_index_se=float(z.std() / np.sqrt(len(z)) * shrink) if len(z) >= 5 else None,
                mean_finish_seconds=float(np.mean(secs)) if secs else None,
                decisions_with_cards=cards_n,
            )),
        ))
    refs = pd.DataFrame(rows)
    ids = refs.set_index("name_norm").id
    rf = f.assign(referee_id=f.name_norm.map(ids))[[
        "referee_id", "fight_id", "date", "event_name", "weight_class", "red_name", "blue_name",
        "winner", "method", "finish_round", "finish_time", "absorbed", "absorbed_expected", "absorbed_z",
        "deductions"]]
    return refs, rf


# ---------------------------------------------------------------------- write


def write(eng, res: dict, refs: pd.DataFrame, ref_fights: pd.DataFrame) -> None:
    """Replace this version's rows (and the referee tables) in one transaction."""
    from app.models.ufc_officials import (
        UFCDecisionReview, UFCJudgeCard, UFCJudgePair, UFCJudgeProfile, UFCJudgeRound,
        UFCOfficialsMeta, UFCReferee, UFCRefereeFight,
    )
    from app.database import Base
    tables = [m.__table__ for m in (UFCJudgeProfile, UFCJudgeRound, UFCJudgeCard, UFCDecisionReview,
                                    UFCJudgePair, UFCOfficialsMeta, UFCReferee, UFCRefereeFight)]
    Base.metadata.create_all(eng, tables=tables)
    v = MODEL_VERSION
    prof = res["judges"]
    detail_cols = [c for c in prof.columns if c not in (
        "judge_id", "name", "eligible", "n_fights", "n_rounds", "accuracy", "expected_accuracy",
        "acc_above_exp", "agree_above_exp", "deciding_cards", "robbery_cards", "first_date",
        "last_date", "map_x", "map_y")]
    profiles = [dict(
        judge_id=int(p.judge_id), version=v, name=p["name"], eligible=bool(p.eligible),
        n_fights=int(p.n_fights), n_rounds=int(p.n_rounds), accuracy=_f(p.accuracy),
        expected_accuracy=_f(p.expected_accuracy), acc_above_exp=_f(p.acc_above_exp),
        agree_above_exp=_f(p.agree_above_exp), deciding_cards=int(p.deciding_cards),
        robbery_cards=int(p.robbery_cards), first_date=dt.date.fromisoformat(p.first_date),
        last_date=dt.date.fromisoformat(p.last_date), map_x=_f(p.map_x), map_y=_f(p.map_y),
        detail=json.dumps(_jsonable({c: p[c] for c in detail_cols}), default=str),
    ) for _, p in prof.iterrows()]
    r, fights = res["rounds"], res["fights"].set_index("fight_id")
    rounds = [dict(
        version=v, judge_id=int(x.judge_id), fight_id=int(x.fight_id), round=int(x.round),
        date=x.date, verdict=int(x.s), score=x.score, p_red=float(x.p),
        red_name=fights.red_name.get(x.fight_id, ""), blue_name=fights.blue_name.get(x.fight_id, ""),
        event_name=fights.event_name.get(x.fight_id),
    ) for x in r.itertuples()]
    fc = res["fight_cards"]
    dates = fights.date.map(lambda d: pd.to_datetime(d).date())
    cards = [dict(
        version=v, judge_id=int(x.judge_id), judge_seq=int(x.judge_seq), fight_id=int(x.fight_id),
        date=dates.get(x.fight_id), red_total=int(x.red_pts), blue_total=int(x.blue_pts),
        side=int(x.w), result=int(x.result), pivotal=bool(x.pivotal),
        robbery_card=bool(x.robbery_card), against_model=bool(x.against_model),
    ) for x in fc.itertuples() if x.fight_id in dates.index]
    reviews = [dict(version=v, **{k: (None if isinstance(val, float) and val != val else val)
                                  for k, val in row.items()})
               for row in res["reviews"].to_dict("records")]
    pairs = [dict(version=v, judge_a=int(x.judge_a), judge_b=int(x.judge_b),
                  n_rounds=int(x.n_rounds), agree=float(x.agree), expected=float(x.expected))
             for x in res["pairs"].itertuples()]
    meta = dict(kind="judges", version=v,
                payload=json.dumps(_jsonable(dict(meta=res["meta"], pool=res["pool"])), default=str))
    ref_rows = [{k: (None if isinstance(val, float) and val != val else val) for k, val in row.items()}
                for row in refs.to_dict("records")]
    rf_rows = [dict(
        referee_id=int(x.referee_id), fight_id=int(x.fight_id), date=x.date,
        event_name=x.event_name, weight_class=x.weight_class, red_name=x.red_name,
        blue_name=x.blue_name, winner=int(x.winner), method=x.method,
        finish_round=int(x.finish_round) if x.finish_round == x.finish_round else None,
        finish_time=x.finish_time, absorbed=int(x.absorbed) if x.absorbed == x.absorbed else None,
        absorbed_z=_f(x.absorbed_z), absorbed_expected=_f(x.absorbed_expected),
        deductions=int(x.deductions),
    ) for x in ref_fights.itertuples()]

    with eng.begin() as c:
        # ufc_referee_fights gained absorbed_expected after its first prod write;
        # create_all never alters an existing table.
        if eng.dialect.name == "postgresql":
            c.execute(text("ALTER TABLE ufc.ufc_referee_fights ADD COLUMN IF NOT EXISTS absorbed_expected double precision"))
        # Keep each referee's id across runs (profile URLs, entered panels point at it).
        old = dict(c.execute(UFCReferee.__table__.select().with_only_columns(
            UFCReferee.__table__.c.name_norm, UFCReferee.__table__.c.id)).all())
        ids = stable_ids(old, [r["name_norm"] for r in ref_rows])
        remap = {r["id"]: ids[r["name_norm"]] for r in ref_rows}
        for r in ref_rows:
            r["id"] = remap[r["id"]]
        for r in rf_rows:
            r["referee_id"] = remap[r["referee_id"]]
        for t in (UFCJudgeProfile, UFCJudgeRound, UFCJudgeCard, UFCDecisionReview, UFCJudgePair,
                  UFCOfficialsMeta):
            c.execute(t.__table__.delete().where(t.__table__.c.version == v))
        c.execute(UFCRefereeFight.__table__.delete())
        c.execute(UFCReferee.__table__.delete())
        for t, rows in ((UFCJudgeProfile, profiles), (UFCJudgeRound, rounds), (UFCJudgeCard, cards),
                        (UFCDecisionReview, reviews), (UFCJudgePair, pairs),
                        (UFCOfficialsMeta, [meta]), (UFCReferee, ref_rows),
                        (UFCRefereeFight, rf_rows)):
            for i in range(0, len(rows), 5000):
                c.execute(t.__table__.insert(), rows[i:i + 5000])
    log.info("wrote %d judges, %d rounds, %d cards, %d decisions, %d pairs, %d referees, %d referee fights",
             len(profiles), len(rounds), len(cards), len(reviews), len(pairs), len(ref_rows), len(rf_rows))


def stable_ids(existing: dict[str, int], names: list[str]) -> dict[str, int]:
    """Ids for names: the existing id where the name had one, else new ids after the max."""
    nxt = max(existing.values(), default=0)
    out = {}
    for n in names:
        if n in existing:
            out[n] = existing[n]
        else:
            nxt += 1
            out[n] = nxt
    return out


def _f(x):
    return None if x is None or x != x else float(x)


def _jsonable(o):
    """JSON-safe copy: frames to records, numpy scalars to Python, NaN to None."""
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="records", date_format="iso"))
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        o = o.item()
    if isinstance(o, float) and o != o:
        return None
    return o


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", metavar="OUT_JSON", help="write profiles to a JSON file (no DB writes)")
    g.add_argument("--write", action="store_true", help="replace the officials tables")
    ap.add_argument("--db", metavar="URL", help="write to this database instead (e.g. a scratch SQLite)")
    ap.add_argument("--s-lean", type=float)
    ap.add_argument("--s-axis", type=float)
    a = ap.parse_args(argv)
    from app.database import engine as eng
    data = load(eng)
    res = build_profiles(data, load_latent(), a.s_lean, a.s_axis)
    if a.dry_run:
        out = {"meta": res["meta"], "pool": res["pool"], "judges": res["judges"]}
        with open(a.dry_run, "w") as fh:
            json.dump(_jsonable(out), fh, indent=1, default=str)
        log.info("Wrote %s (%d judges)", a.dry_run, len(res["judges"]))
        return 0
    refs, ref_fights = referee_tables(data)
    target = eng
    if a.db:
        from sqlalchemy import create_engine
        target = create_engine(a.db)
        if target.dialect.name == "sqlite":
            # The models carry the Postgres "ufc" schema; SQLite has none.
            target = target.execution_options(schema_translate_map={"ufc": None})
    write(target, res, refs, ref_fights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
