"""Fit, validate and score the deserve-to-win meter (app/services/ufc/deserve_to_win.py).

Steps
  1. Round model: corner-symmetric gradient boosting on round stat differences
     (scripts.round_effects.rich_features) + a post-2017 era flag -> P(a judge gives red
     the round); a 10-8 hurdle (logistic, features signed toward the round's winner).
     B bootstrap fits by fight. A fight that trained a fit never uses that fit
     (out-of-bag), so every historical fight is scored by models that never saw it.
  2. Judge correlation: sigma_u of the shared round latent, by maximum likelihood of
     the three judges agreeing on each verified round.
  3. Form (finished fights): within-fight and between-fight spread of round logits on
     decision fights, for the normal-normal update over unfought rounds.
  4. Validation (--validate): round log loss, 10-8 calibration, judge agreement and
     UD/SD/MD rates, fight calibration vs official decisions, robberies vs fan/media
     cards, and a truncation backtest of the extrapolation.
  5. Scoring (--write): every fight with round stats and 5-minute rounds ->
     ufc.ufc_deserve_to_win (replaces this model_version's rows).

Uses the database in app.config settings (alocks-backend/.env = production).

    PYTHONPATH=. venv/bin/python -m scripts.deserve_to_win --validate --write
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sqlalchemy import text

from app.services.ufc.deserve_to_win import (
    FightInput, Form, TenEight, fit_sigma_u, form_posterior, judge_agreement, logit, sigmoid,
    simulate_fight,
)
from app.services.ufc.outcome_types import classify_outcome
from scripts.round_effects import JUDGE_AXES, RAW, rich_features

MODEL_VERSION = "v1"
ERA = dt.date(2017, 1, 1)
OUT = Path("data/mmad")


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------- data


def load_rounds(eng) -> pd.DataFrame:
    """One row per fight-round (both corners' stats as r_*/b_*) with fight metadata."""
    cols = ", ".join(f"s.{c}" for c in RAW)
    st = pd.read_sql(text(f"""
        SELECT s.fight_id, s.round_number AS round, s.fighter_id, {cols}
        FROM ufc.ufc_fight_stats s WHERE s.round_number >= 1
    """), eng)
    st[RAW] = st[RAW].astype(float).fillna(0.0)
    f = pd.read_sql(text("""
        SELECT f.id AS fight_id, f.red_fighter_id AS red, f.blue_fighter_id AS blue,
               f.winner_id, f.method, f.details, f.finish_round, f.finish_time, f.time_format,
               COALESCE(f.date, e.date) AS date
        FROM ufc.ufc_fights f JOIN ufc.ufc_events e ON e.id = f.event_id
    """), eng)
    f["date"] = pd.to_datetime(f["date"]).dt.date
    red = st.rename(columns={c: f"r_{c}" for c in RAW}).rename(columns={"fighter_id": "red"})
    blue = st.rename(columns={c: f"b_{c}" for c in RAW}).rename(columns={"fighter_id": "blue"})
    df = f.merge(red, on=["fight_id", "red"]).merge(blue, on=["fight_id", "blue", "round"])
    # UFCStats pads some bouts with all-zero rows past the finish (out to round 23;
    # common since mid-2026). Left in, they make observed rounds exceed the scheduled
    # count and fight_inputs drops the fight — which silently stopped scoring every
    # bout after 2026-08-29. finish_round is authoritative for rounds contested.
    df = df[df["finish_round"].isna() | (df["round"] <= df["finish_round"])]
    return df.sort_values(["fight_id", "round"]).reset_index(drop=True)


def load_cards(eng) -> pd.DataFrame:
    """Verified mmad judge-rounds: the judge's own verdict (before deductions)."""
    c = pd.read_sql(text("""
        SELECT c.fight_id, c.round, c.judge_seq, c.judge_id, j.name AS judge,
               c.red_pts + c.red_ded AS rp, c.blue_pts + c.blue_ded AS bp,
               c.red_ded, c.blue_ded
        FROM ufc.ufc_judge_scorecards c
        JOIN ufc.mmad_decisions d ON d.mmad_decision_id = c.mmad_decision_id
        LEFT JOIN ufc.ufc_judges j ON j.id = c.judge_id
        WHERE c.source = 'mmad' AND c.round >= 1 AND d.match_status = 'verified'
          AND c.red_pts IS NOT NULL AND c.blue_pts IS NOT NULL
    """), eng)
    return c


def load_fans(eng) -> pd.DataFrame:
    """Fan and media majority per verified decision, in DB corners."""
    d = pd.read_sql(text("""
        SELECT d.fight_id, d.swapped, d.fan_a, d.fan_b, d.fan_draw, d.mmad_decision_id
        FROM ufc.mmad_decisions d WHERE d.match_status = 'verified'
    """), eng)
    m = pd.read_sql(text("SELECT mmad_decision_id, pick FROM ufc.mmad_media_scores"), eng)
    mc = m.pivot_table(index="mmad_decision_id", columns="pick", aggfunc="size", fill_value=0)
    d = d.merge(mc, left_on="mmad_decision_id", right_index=True, how="left").fillna(0)
    sw = d.swapped.astype(bool)
    for src, (a, b) in {"fan": ("fan_a", "fan_b"), "media": ("a", "b")}.items():
        if a not in d:
            d[a] = 0
        if b not in d:
            d[b] = 0
        d[f"{src}_red"] = np.where(sw, d[b], d[a]).astype(float)
        d[f"{src}_blue"] = np.where(sw, d[a], d[b]).astype(float)
    return d[["fight_id", "fan_red", "fan_blue", "media_red", "media_blue"]]


def scheduled_rounds(time_format) -> int | None:
    """Rounds scheduled for modern 5-minute-round formats; None otherwise."""
    if not isinstance(time_format, str):
        return None
    parts = time_format.split("-")
    return len(parts) if all(p == "5" for p in parts) else None


def seconds(mmss) -> int | None:
    try:
        m, s = str(mmss).split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------- model


def design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(antisymmetric stat-difference features, era flag)."""
    X = rich_features(df).to_numpy(float)
    era = (pd.Series(df["date"]).map(lambda d: d >= ERA)).to_numpy(float)[:, None]
    return X, era


def fit_winner(X, era, y, seed):
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=80, l2_regularization=1.0,
                                       random_state=seed)
    m.fit(np.vstack([np.hstack([X, era]), np.hstack([-X, era])]),
          np.concatenate([y, 1 - y]))
    return m


def predict_winner(m, X, era):
    a = m.predict_proba(np.hstack([X, era]))[:, 1]
    b = m.predict_proba(np.hstack([-X, era]))[:, 1]
    return 0.5 * (a + 1 - b)


class TenEightModel:
    """P(10-8 | features signed toward the round's winner). Standardised logistic."""

    def fit(self, Xw, era, y8):
        Z = np.hstack([Xw, era])
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        self.m = LogisticRegression(C=0.5, max_iter=5000).fit((Z - self.mu) / self.sd, y8)
        return self

    def predict(self, Xw, era):
        Z = np.hstack([Xw, era])
        return self.m.predict_proba((Z - self.mu) / self.sd)[:, 1]


def fit_bootstrap(train: pd.DataFrame, Xtr, era_tr, Xall, era_all, all_fights, B, seed=0):
    """B bootstrap fits by fight. Returns P(red) and 10-8 arrays (B, n_all) and an
    out-of-bag mask (B, n_all): True where the row's fight was not in that fit."""
    rng = np.random.default_rng(seed)
    fights = train.fight_id.unique()
    rows_by_fight = train.groupby("fight_id").indices
    y = train.y.to_numpy()
    s = np.where(y == 1, 1.0, -1.0)[:, None]
    y8 = train.y8.to_numpy()
    P = np.empty((B, len(Xall)))
    Q_red, Q_blue = np.empty_like(P), np.empty_like(P)
    oob = np.empty_like(P, dtype=bool)
    for b in range(B):
        t0 = time.time()
        pick = rng.choice(fights, size=len(fights), replace=True)
        idx = np.concatenate([rows_by_fight[f] for f in pick])
        m = fit_winner(Xtr[idx], era_tr[idx], y[idx], seed + b)
        t8 = TenEightModel().fit(s[idx] * Xtr[idx], era_tr[idx], y8[idx])
        P[b] = predict_winner(m, Xall, era_all)
        Q_red[b] = t8.predict(Xall, era_all)
        Q_blue[b] = t8.predict(-Xall, era_all)
        oob[b] = ~np.isin(all_fights, pick)
        log(f"  fit {b + 1}/{B}  ({time.time() - t0:.0f}s)")
    return P, Q_red, Q_blue, oob


def mean_oob(A, oob):
    """Mean over the fits that never saw each row's fight (all fits if none did)."""
    w = oob.astype(float)
    n = w.sum(0)
    return np.where(n > 0, (A * w).sum(0) / np.maximum(n, 1), A.mean(0))


# ------------------------------------------------------------------ assembly


def fight_inputs(rounds: pd.DataFrame, P, Q_red, Q_blue, oob, cards: pd.DataFrame,
                 judge_fx: pd.DataFrame | None, axes_sd: np.ndarray):
    """Yield (fight_id, meta, FightInput, panel judge_shift or None) per scoreable fight."""
    L = logit(P)
    ded = (cards.groupby(["fight_id", "round"])[["red_ded", "blue_ded"]].max()
           if len(cards) else None)
    panel = {}
    if judge_fx is not None and len(cards):
        jc = cards.drop_duplicates(["fight_id", "judge_seq"])[["fight_id", "judge_seq", "judge"]]
        for fid, g in jc.groupby("fight_id"):
            if len(g) == 3:
                panel[fid] = g.sort_values("judge_seq").judge.tolist()
    axes = rich_features(rounds)[JUDGE_AXES].to_numpy(float) / axes_sd
    fx = judge_fx.set_index("name") if judge_fx is not None else None

    for fid, idx in rounds.groupby("fight_id").indices.items():
        g = rounds.iloc[idx]
        meta = g.iloc[0]
        R = scheduled_rounds(meta.time_format)
        if R is None:
            continue
        obs = g["round"].to_numpy()
        n_obs = len(obs)
        if n_obs == 0 or n_obs > R or not np.array_equal(obs, np.arange(1, n_obs + 1)):
            continue
        method = meta.method if isinstance(meta.method, str) else ""
        is_dec = method.startswith("Decision")
        if is_dec and n_obs != R:
            continue
        unseen = oob[:, idx[0]]                     # fits that never saw this fight
        use = np.flatnonzero(unseen) if unseen.any() else np.arange(P.shape[0])
        partial_w, partial_s = None, None
        if not is_dec:
            fr = meta.finish_round
            if fr != fr or int(fr) != n_obs:
                continue
            el = seconds(meta.finish_time)
            if el is not None and el < 300:
                partial_w, partial_s = max(el, 1) / 300.0, el
        rd = bd = None
        if ded is not None and is_dec:
            try:
                dd = ded.loc[fid].reindex(range(1, R + 1)).fillna(0)
                rd, bd = dd.red_ded.to_numpy(), dd.blue_ded.to_numpy()
            except KeyError:
                pass
        fi = FightInput(round_logits=L[use][:, idx], scheduled=R,
                        ten8=Q_red[use][:, idx], ten8_blue=Q_blue[use][:, idx],
                        partial_weight=partial_w, red_ded=rd, blue_ded=bd)
        shift = None
        if fx is not None and fid in panel and all(n in fx.index for n in panel[fid]):
            ax = axes[idx]                          # (R, 4)
            shift = np.stack([fx.loc[n, "red_lean"]
                              + ax @ fx.loc[n, [f"lean_{a[2:]}" for a in JUDGE_AXES]].to_numpy(float)
                              for n in panel[fid]])
        out = classify_outcome(meta.method, meta.details, meta.winner_id)
        side = ("red" if meta.winner_id == meta.red else "blue"
                if meta.winner_id == meta.blue else "none")
        yield fid, dict(date=meta.date, R=R, n_obs=n_obs, is_dec=is_dec, partial_s=partial_s,
                        outcome=out, side=side), fi, shift


def fit_form(z_by_fight: list[np.ndarray]) -> tuple[float, float]:
    """(tau, sigma): SD of fight-mean round logits around 0, and of rounds around their
    fight mean, by method of moments on full-distance fights."""
    within = np.concatenate([z - z.mean() for z in z_by_fight])
    dof = sum(len(z) - 1 for z in z_by_fight)
    sigma2 = (within ** 2).sum() / dof
    means = np.array([z.mean() for z in z_by_fight])
    ns = np.array([len(z) for z in z_by_fight])
    tau2 = max(np.mean(means ** 2 - sigma2 / ns), 1e-3)
    return float(np.sqrt(tau2)), float(np.sqrt(sigma2))


# ------------------------------------------------------------------- driver


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=30, help="bootstrap fits")
    ap.add_argument("--n", type=int, default=10_000, help="simulations per fight")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--write", action="store_true", help="write ufc_deserve_to_win")
    args = ap.parse_args(argv)

    from app.database import engine as eng
    t0 = time.time()
    rounds = load_rounds(eng)
    cards = load_cards(eng)
    log(f"fight-rounds: {len(rounds):,} ({rounds.fight_id.nunique():,} fights)  "
        f"judge-rounds: {len(cards):,} ({cards.fight_id.nunique():,} verified decisions)")

    # --- training rows: judge-rounds with stats, 10-10s dropped
    tr = cards[cards.rp != cards.bp].merge(rounds, on=["fight_id", "round"])
    tr["y"] = (tr.rp > tr.bp).astype(int)
    tr["y8"] = (np.minimum(tr.rp, tr.bp) <= 8).astype(int)
    tr = tr.reset_index(drop=True)
    Xtr, era_tr = design(tr)
    Xall, era_all = design(rounds)
    axes_sd = rich_features(tr)[JUDGE_AXES].to_numpy(float).std(0) + 1e-9
    log(f"training judge-rounds: {len(tr):,}  10-8 rate {tr.y8.mean():.3%}  B={args.b}")

    P, Qr, Qb, oob = fit_bootstrap(tr, Xtr, era_tr, Xall, era_all,
                                   rounds.fight_id.to_numpy(), args.b)
    trained = rounds.fight_id.isin(tr.fight_id).to_numpy()
    p_hat = np.where(trained, mean_oob(P, oob), P.mean(0))
    qr_hat = np.where(trained, mean_oob(Qr, oob), Qr.mean(0))
    qb_hat = np.where(trained, mean_oob(Qb, oob), Qb.mean(0))
    rounds["p_red"], rounds["q_red"], rounds["q_blue"] = p_hat, qr_hat, qb_hat
    key = rounds.set_index(["fight_id", "round"])[["p_red", "q_red", "q_blue", "date"]]
    trp = tr.join(key, on=["fight_id", "round"], rsuffix="_k")

    # --- judge correlation
    full = cards[cards.rp != cards.bp].copy()
    full["red_won"] = full.rp > full.bp
    agg = full.groupby(["fight_id", "round"]).red_won.agg(["sum", "size"])
    agg = agg[agg["size"] == 3].join(key)
    agg = agg[agg.p_red.notna()]
    agg["unan"] = (agg["sum"] == 0) | (agg["sum"] == 3)
    sigma_u, ll = fit_sigma_u(agg.p_red.to_numpy(), agg.unan.to_numpy())
    log(f"\njudge correlation: sigma_u = {sigma_u:.2f}  "
        f"(rounds with 3 judges: {len(agg):,}, unanimous {agg.unan.mean():.1%})")

    # --- 10-8 fallback for extrapolated rounds: P(10-8) vs logit toward the winner
    zw = np.where(trp.y == 1, 1, -1) * logit(trp.p_red.to_numpy())
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(zw[:, None], trp.y8)
    t8 = TenEight(a=float(lr.intercept_[0]), b=float(lr.coef_[0, 0]))
    log(f"10-8 fallback: sigmoid({t8.a:.2f} + {t8.b:.2f} * z_toward_winner)")

    # --- form: spread of round logits within / between full-distance fights
    judge_fx = None
    fx_path = OUT / "judge_effects.csv"
    if fx_path.exists():
        judge_fx = pd.read_csv(fx_path).drop_duplicates("name")
    inputs = list(fight_inputs(rounds, P, Qr, Qb, oob, cards, judge_fx, axes_sd))
    dec_z = [np.atleast_2d(fi.round_logits).mean(0) for _, m, fi, _ in inputs if m["is_dec"]]
    tau, sigma = fit_form(dec_z)
    log(f"form: tau (fight mean) = {tau:.2f}  sigma (round to round) = {sigma:.2f}  "
        f"from {len(dec_z):,} decisions")
    log(f"scoreable fights: {len(inputs):,} "
        f"({sum(m['is_dec'] for _, m, _, _ in inputs):,} decisions)")

    # --- fight latent: the round agreement pins the TOTAL shared SD within a round
    # (sigma_u^2 + sigma_v^2 = s_tot^2); split it by fight-level log loss on 2009+
    # decisions (pre-2009 UFCStats lists the winner in the red corner).
    s_tot = sigma_u
    cal = [(m, fi) for _, m, fi, _ in inputs if m["is_dec"] and m["side"] in ("red", "blue")
           and m["outcome"] in ("ud", "sd", "md") and m["date"] >= dt.date(2009, 1, 1)]
    y_cal = np.array([m["side"] == "red" for m, _ in cal], float)
    best = None
    for sv in np.round(np.arange(0.0, s_tot, 0.25), 2):
        su = float(np.sqrt(s_tot ** 2 - sv ** 2))
        g = np.random.default_rng(3)
        p = np.array([(lambda r: r.p_red / max(r.p_red + r.p_blue, 1e-9))(
            simulate_fight(fi, n=2000, sigma_u=su, sigma_v=sv, ten8_fallback=t8, rng=g))
            for _, fi in cal])
        ll_v = _ll(y_cal, p)
        log(f"  sigma_v {sv:.2f} (sigma_u {su:.2f}): fight log loss {ll_v:.4f}")
        if best is None or ll_v < best[0]:
            best = (ll_v, sv, su)
    _, sigma_v, sigma_u = best
    log(f"fight latent: sigma_v = {sigma_v:.2f}, round latent sigma_u = {sigma_u:.2f}")

    def form_for(fi, meta):
        z = np.atleast_2d(fi.round_logits).mean(0)
        # A partial finishing round is weighted by the share of it fought.
        if meta["partial_s"] is not None:
            w = np.ones(len(z))
            w[-1] = fi.partial_weight
            prec = 1 / tau ** 2 + w.sum() / sigma ** 2
            mean = (w * z).sum() / sigma ** 2 / prec
            return Form(float(mean), float(np.sqrt(1 / prec)), sigma)
        return form_posterior(z, 0.0, tau, sigma)

    # --- simulate everything
    rows = []
    rng = np.random.default_rng(42)
    t1 = time.time()
    for i, (fid, meta, fi, shift) in enumerate(inputs):
        if fi.partial_weight is not None or meta["n_obs"] < meta["R"]:
            fi.form = form_for(fi, meta)
        r = simulate_fight(fi, n=args.n, sigma_u=sigma_u, sigma_v=sigma_v, ten8_fallback=t8, rng=rng)
        rp = None
        if shift is not None:
            fi.judge_shift = shift
            rp = simulate_fight(fi, n=args.n, sigma_u=sigma_u, sigma_v=sigma_v, ten8_fallback=t8, rng=rng)
            fi.judge_shift = None
        rows.append(dict(fight_id=int(fid), meta=meta, r=r, panel=rp))
        if (i + 1) % 2000 == 0:
            log(f"  simulated {i + 1:,}/{len(inputs):,} ({time.time() - t1:.0f}s)")
    res = pd.DataFrame([dict(
        fight_id=x["fight_id"], date=x["meta"]["date"], R=x["meta"]["R"],
        n_obs=x["meta"]["n_obs"], is_dec=x["meta"]["is_dec"],
        partial_s=x["meta"]["partial_s"], outcome=x["meta"]["outcome"], side=x["meta"]["side"],
        p_red=x["r"].p_red, p_draw=x["r"].p_draw, p_blue=x["r"].p_blue,
        p_ud=x["r"].p_kind["ud"], p_sd=x["r"].p_kind["sd"], p_md=x["r"].p_kind["md"],
        panel_p_red=x["panel"].p_red if x["panel"] else None,
        panel_p_draw=x["panel"].p_draw if x["panel"] else None,
        panel_p_blue=x["panel"].p_blue if x["panel"] else None,
        round_p_red=json.dumps([round(v, 4) for v in x["r"].round_p_red]),
        top_cards=json.dumps(x["r"].top_cards),
    ) for x in rows])
    res["official_outcome"] = res.side + ":" + res.outcome
    res["robbery_score"] = np.where(
        res.is_dec & (res.side == "red"), res.p_blue,
        np.where(res.is_dec & (res.side == "blue"), res.p_red, np.nan))

    OUT.mkdir(parents=True, exist_ok=True)
    res.drop(columns=["round_p_red", "top_cards"]).to_csv(OUT / "dtw_results.csv", index=False)
    if args.validate:
        validate(trp, agg, sigma_u, sigma_v, res, inputs, tau, sigma, t8, eng)

    if args.write:
        write(eng, res, args.n)
    log(f"\ndone in {time.time() - t0:.0f}s")
    return 0


# ---------------------------------------------------------------- validation


def _ll(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])


def validate(trp, agg, sigma_u, sigma_v, res, inputs, tau, sigma, t8, eng):
    OUT.mkdir(parents=True, exist_ok=True)
    log("\n" + "=" * 70 + "\nVALIDATION")

    # (a) round level, out-of-bag
    y, p = trp.y.to_numpy(), trp.p_red.to_numpy()
    recent = (trp.date >= dt.date(2023, 1, 1)).to_numpy()
    log("\n(a) round model, out-of-bag, vs each judge's verdict")
    log(f"  all:   log loss {_ll(y, p):.4f}  acc {((p > .5) == y).mean():.3f}  n={len(y):,}")
    log(f"  2023+: log loss {_ll(y[recent], p[recent]):.4f}  "
        f"acc {((p[recent] > .5) == y[recent]).mean():.3f}  n={recent.sum():,}")
    maj = agg.assign(maj=agg["sum"] >= 2)
    log(f"  agreement with the 3-judge majority: {((maj.p_red > .5) == maj.maj).mean():.3f}")
    q = np.where(y == 1, trp.q_red, trp.q_blue)
    bins = pd.qcut(q, 10, duplicates="drop")
    cal = pd.DataFrame({"pred": q, "act": trp.y8}).groupby(bins, observed=True).mean()
    log("  10-8 calibration (deciles of predicted P(10-8 | actual winner)):")
    log("    " + "  ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(cal.pred, cal.act)) + "  (pred/actual)")

    # (b) judge agreement by bucket, and decision types
    log(f"\n(b) judge agreement (sigma_u = {sigma_u:.2f}, sigma_v = {sigma_v:.2f}); "
        "P(3-0 round) by model p bucket")
    b = pd.cut(np.abs(agg.p_red - 0.5), [0, .1, .2, .3, .4, .5], include_lowest=True)
    sim = judge_agreement(agg.p_red.to_numpy(), sigma_u, n=400, rng=1, sigma_v=sigma_v)
    t = pd.DataFrame({"emp": agg.unan.to_numpy(), "sim": sim,
                      "indep": judge_agreement(agg.p_red.to_numpy(), 0.0, n=400, rng=1)}
                     ).groupby(b.to_numpy(), observed=True).agg(["mean", "size"])
    for idx, row in t.iterrows():
        log(f"  |p-.5| in {idx}: actual {row[('emp', 'mean')]:.3f}  model {row[('sim', 'mean')]:.3f}  "
            f"(independent judges {row[('indep', 'mean')]:.3f})  n={int(row[('emp', 'size')]):,}")
    dec = res[res.is_dec & res.outcome.isin(["ud", "sd", "md", "draw"])]
    actual = dec.outcome.value_counts(normalize=True)
    log("  decision types, actual vs mean simulated:")
    for k in ("ud", "sd", "md", "draw"):
        simk = dec.p_draw.mean() if k == "draw" else dec[f"p_{k}"].mean()
        log(f"    {k}: actual {actual.get(k, 0):.3f}  simulated {simk:.3f}")

    # (c) fight calibration vs official decisions
    # By corner, so 2009+ only (pre-2009 UFCStats lists the winner in the red corner).
    d = dec[dec.side.isin(["red", "blue"]) & (dec.date >= dt.date(2009, 1, 1))].copy()
    d["y"] = (d.side == "red").astype(int)
    d["p"] = d.p_red / (d.p_red + d.p_blue).clip(lower=1e-9)
    log(f"\n(c) fight level, {len(d):,} decisions with a winner: P(red wins | not a draw)")
    log(f"  log loss {_ll(d.y, d.p):.4f}  Brier {((d.p - d.y) ** 2).mean():.4f}  "
        f"acc {((d.p > .5) == d.y).mean():.3f}")
    d["bin"] = pd.cut(d.p, [0, .1, .25, .4, .6, .75, .9, 1], include_lowest=True)
    for bn, g in d.groupby("bin", observed=True):
        log(f"    P in {bn}: n={len(g):5d}  predicted {g.p.mean():.3f}  red won {g.y.mean():.3f}")
    fav = np.maximum(d.p, 1 - d.p)
    for cut in (0.5, 0.75, 0.9):
        s = fav > cut
        log(f"  deserved-favourite above {cut:.0%}: official winner {((d.p > .5) == d.y)[s].mean():.3f} "
            f"of {s.sum():,}")
    if d.panel_p_red.notna().any():
        e = d[d.panel_p_red.notna()]
        pp = e.panel_p_red / (e.panel_p_red + e.panel_p_blue).clip(lower=1e-9)
        log(f"  actual panel (judge leans), n={len(e):,}: log loss {_ll(e.y, pp):.4f} "
            f"vs average judges {_ll(e.y, e.p):.4f}")

    # (d) robberies vs fans / media
    fans = load_fans(eng)
    r = d.merge(fans, on="fight_id", how="left")
    r["winner_share"] = np.where(r.y == 1, r.p, 1 - r.p)
    for src in ("fan", "media"):
        n = r[f"{src}_red"] + r[f"{src}_blue"]
        ok = n >= (20 if src == "fan" else 3)
        fr = (r[f"{src}_red"] / n.where(n > 0))
        agree = ((fr > .5) == (r.p > .5))[ok]
        log(f"\n(d) model vs {src} majority (n={ok.sum():,}): agree {agree.mean():.3f}")
        rob = ok & (r.winner_share < 0.25)
        fan_rob = ((fr > .5) != (r.y == 1))[rob]
        log(f"  model robberies (official winner < 25%): {rob.sum():,}; "
            f"{src} majority also had the official loser: {fan_rob.mean():.3f}")
    names = pd.read_sql(text("""
        SELECT f.id AS fight_id, rf.first_name || ' ' || rf.last_name AS red_name,
               bf.first_name || ' ' || bf.last_name AS blue_name, e.name AS event
        FROM ufc.ufc_fights f JOIN ufc.ufc_events e ON e.id = f.event_id
        JOIN ufc.ufc_fighters rf ON rf.id = f.red_fighter_id
        JOIN ufc.ufc_fighters bf ON bf.id = f.blue_fighter_id
    """), eng)
    rob = r.merge(names, on="fight_id").sort_values("winner_share")
    rob["winner"] = np.where(rob.y == 1, rob.red_name, rob.blue_name)
    rob["loser"] = np.where(rob.y == 1, rob.blue_name, rob.red_name)
    rob["fan_pct_winner"] = np.where(rob.y == 1, rob.fan_red, rob.fan_blue) / (
        rob.fan_red + rob.fan_blue).where(lambda x: x > 0)
    cols = ["date", "event", "winner", "loser", "outcome", "winner_share", "fan_pct_winner",
            "panel_p_red", "fight_id"]
    rob[cols].to_csv(OUT / "dtw_robberies.csv", index=False)
    log("  biggest robberies (official winner's deserve-to-win share):")
    for _, x in rob.head(15).iterrows():
        fp = f"{x.fan_pct_winner:.0%}" if x.fan_pct_winner == x.fan_pct_winner else "n/a"
        log(f"    {x.date}  {x.winner} def. {x.loser} ({x.outcome}): {x.winner_share:.0%}"
            f"  fans for winner {fp}")

    # (e) symmetry of the round model is by construction; spot-check the simulator
    log("\n(e) symmetry: winner model averages f(x) and 1-f(-x); unit tests cover the sim")

    # extrapolation backtest: truncate decisions after k rounds
    log("\nextrapolation backtest: decisions truncated after k rounds (2,000 sims each)")
    full = {fid: (m, fi) for fid, m, fi, _ in inputs if m["is_dec"] and m["outcome"] in ("ud", "sd", "md")}
    target = d.set_index("fight_id")
    rng = np.random.default_rng(7)
    for R in (3, 5):
        for k in range(1, R):
            out = {"form": [], "coin": [], "full": [], "y": []}
            for fid, (m, fi) in full.items():
                if m["R"] != R or fid not in target.index:
                    continue
                Lk = np.atleast_2d(fi.round_logits)[:, :k]
                z = Lk.mean(0)
                arms = {"form": form_posterior(z, 0.0, tau, sigma),
                        "coin": Form(0.0, 1e-9, 1e-9)}
                for arm, f in arms.items():
                    s = simulate_fight(FightInput(Lk, R, ten8=fi.ten8[:, :k] if fi.ten8 is not None else None,
                                                  ten8_blue=fi.ten8_blue[:, :k] if fi.ten8_blue is not None else None,
                                                  form=f),
                                       n=2000, sigma_u=sigma_u, sigma_v=sigma_v,
                                       ten8_fallback=t8, rng=rng)
                    out[arm].append(s.p_red / max(s.p_red + s.p_blue, 1e-9))
                out["full"].append(target.loc[fid, "p"])
                out["y"].append(target.loc[fid, "y"])
            if not out["y"]:
                continue
            yv, fv = np.array(out["y"]), np.array(out["full"])
            msg = []
            for arm in ("form", "coin"):
                pv = np.array(out[arm])
                msg.append(f"{arm}: LL vs official {_ll(yv, pv):.4f}, "
                           f"MAE vs full-fight {np.abs(pv - fv).mean():.3f}")
            log(f"  {R}-round fights, first {k} round(s), n={len(yv):,}:  " + "  |  ".join(msg))

    # finished fights: how the meter treats them
    fin = res[~res.is_dec & res.side.isin(["red", "blue"])]
    if len(fin):
        ws = np.where(fin.side == "red", fin.p_red, fin.p_blue)
        log(f"\nfinished fights (n={len(fin):,}): finisher's deserve-to-win median {np.median(ws):.2f}; "
            f"share where the finisher was the deserved favourite {np.mean(ws > .5):.3f}")


# -------------------------------------------------------------------- write


def write(eng, res: pd.DataFrame, n_sims: int) -> None:
    from app.models.ufc import UFCDeserveToWin
    UFCDeserveToWin.__table__.create(bind=eng, checkfirst=True)
    now = dt.datetime.now(dt.timezone.utc)
    out = pd.DataFrame({
        "fight_id": res.fight_id, "model_version": MODEL_VERSION,
        "p_red": res.p_red, "p_draw": res.p_draw, "p_blue": res.p_blue,
        "panel_p_red": res.panel_p_red, "panel_p_draw": res.panel_p_draw,
        "panel_p_blue": res.panel_p_blue,
        "p_ud": res.p_ud, "p_sd": res.p_sd, "p_md": res.p_md,
        "rounds_observed": res.n_obs, "rounds_scheduled": res.R,
        "extrapolated": (~res.is_dec) & ((res.n_obs < res.R) | res.partial_s.notna()),
        "partial_round_seconds": res.partial_s.astype("Int64"),
        "round_p_red": res.round_p_red, "top_cards": res.top_cards,
        "official_outcome": res.official_outcome.str[:16],
        "robbery_score": res.robbery_score, "n_sims": n_sims,
        "created_at": now, "updated_at": now,
    })
    out = out.astype(object).where(out.notna(), None)
    with eng.begin() as c:
        c.execute(text("DELETE FROM ufc.ufc_deserve_to_win WHERE model_version = :v"),
                  {"v": MODEL_VERSION})
        recs = out.to_dict("records")
        cols = list(out.columns)
        sql = text(f"INSERT INTO ufc.ufc_deserve_to_win ({', '.join(cols)}) "
                   f"VALUES ({', '.join(':' + k for k in cols)})")
        for i in range(0, len(recs), 1000):
            c.execute(sql, recs[i:i + 1000])
    log(f"\nwrote {len(out):,} rows to ufc.ufc_deserve_to_win (model_version={MODEL_VERSION})")


if __name__ == "__main__":
    raise SystemExit(main())
