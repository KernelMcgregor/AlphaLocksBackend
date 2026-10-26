"""Stage-1 evaluation of fight dictation (app/services/ufc/dictation.py). Read-only.

Questions, each answered on rounds the models had not seen yet (pre-bout state):
  1. Tug-of-war: does a per-fighter dictation skill predict where a round is fought (ground
     share, clinch share) and how fast (pace) better than "split the difference" between
     the two fighters' preferences?
  2. Is the pull a stable trait? Split-half (odd vs even bouts) reliability, and does it
     persist in bouts the fighter LOST (or is it just a side effect of winning)?
  3. Round dynamics: does the grappler's pull grow or fade from R1 to R3, and is a
     fighter's drift itself reliable?
  4. Mechanics: does splitting control into shot rate x success x hold (with an escape
     skill) predict control share better than a one-parameter offence/defence model,
     especially for wrestlers against good escapers?

Windows: fit from the first bout; tune on TUNE_START..REPORT_START, report REPORT_START on.

Usage:
    PYTHONPATH=. venv/bin/python -m scripts.dictation_eval            (cached frame)
    PYTHONPATH=. venv/bin/python -m scripts.dictation_eval --rebuild  (re-read prod)
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.ufc.dictation import (
    MechParams, MechRound, Mechanics, TugOfWar, TugParams, logit, pull, sigmoid,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "ufc" / "dictation"
FRAME = OUT_DIR / "rounds.pkl"
TUNE_START = pd.Timestamp("2012-01-01")
REPORT_START = pd.Timestamp("2019-01-01")
COLS = ["sig_str_attempted", "sig_str_landed", "distance_attempted", "clinch_attempted",
        "ground_attempted", "td_attempted", "td_landed", "ctrl_seconds", "rev", "kd"]
H_GRID = (-12, -6, -3, -1, 0, 1, 3, 6, 12)
SHARE_PRIOR = 4.0     # pseudo-attempts of the league share in each round's observed share


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_rounds(rebuild: bool = False) -> pd.DataFrame:
    """One row per (bout, round) of 5-minute-round bouts, red stats r_*, blue stats b_*."""
    if FRAME.exists() and not rebuild:
        return pd.read_pickle(FRAME)
    from sqlalchemy import text
    from app.database import engine
    cols = ", ".join(f"s.{c}" for c in COLS)
    st = pd.read_sql(text(f"""
        SELECT s.fight_id::text AS fight_id, s.round_number AS round, s.fighter_id::text AS fighter_id, {cols}
        FROM ufc.ufc_fight_stats s WHERE s.round_number >= 1
    """), engine)
    st[COLS] = st[COLS].astype(float).fillna(0.0)
    f = pd.read_sql(text("""
        SELECT f.id::text AS fight_id, f.red_fighter_id::text AS red, f.blue_fighter_id::text AS blue,
               f.winner_id::text AS winner_id, f.finish_round, f.fight_time_seconds, f.time_format,
               COALESCE(f.date, e.date) AS date
        FROM ufc.ufc_fights f JOIN ufc.ufc_events e ON e.id = f.event_id
        WHERE f.fight_time_seconds > 0
    """), engine)
    tf = f["time_format"].fillna("")
    f = f[tf.str.len().gt(0) & tf.map(lambda s: all(p == "5" for p in s.split("-")))]
    red = st.rename(columns={c: f"r_{c}" for c in COLS}).rename(columns={"fighter_id": "red"})
    blue = st.rename(columns={c: f"b_{c}" for c in COLS}).rename(columns={"fighter_id": "blue"})
    df = f.merge(red, on=["fight_id", "red"]).merge(blue, on=["fight_id", "blue", "round"])
    # ufcstats pads some bouts with all-zero rounds past the finish
    df = df[df["finish_round"].isna() | (df["round"] <= df["finish_round"])]
    last = df.groupby("fight_id")["round"].transform("max")
    df["secs"] = np.where(df["round"] < last, 300.0, df["fight_time_seconds"] - 300.0 * (last - 1))
    df["secs"] = df["secs"].clip(upper=300.0)
    df = df[df["secs"] > 5]
    df["date"] = pd.to_datetime(df["date"])
    for c in ("fight_id", "red", "blue", "winner_id"):
        df[c] = df[c].astype(str)
    df = df.sort_values(["date", "fight_id", "round"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(FRAME)
    return df


def add_dimensions(df: pd.DataFrame) -> dict:
    """Observed per-round value (transformed) and weight for each tug-of-war dimension."""
    pos = {k: df[f"r_{k}_attempted"] + df[f"b_{k}_attempted"] for k in ("distance", "clinch", "ground")}
    tot = pos["distance"] + pos["clinch"] + pos["ground"]
    early = df["date"] < TUNE_START
    dims = {}
    for k in ("ground", "clinch"):
        m = pos[k][early].sum() / tot[early].sum()
        share = (pos[k] + SHARE_PRIOR * m) / (tot + SHARE_PRIOR)
        dims[k] = (logit(share.to_numpy()), (tot / (tot + 20) * df["secs"] / 300).to_numpy(), logit(m))
    pace = np.log((df["r_sig_str_attempted"] + df["b_sig_str_attempted"]) / (df["secs"] / 60) + 1)
    dims["pace"] = (pace.to_numpy(), (df["secs"] / 300).to_numpy(), float(pace[early].mean()))
    return dims


def bouts(df: pd.DataFrame):
    """Yield (fight_id, red, blue, winner, date, row indices) in time order."""
    for fid, g in df.groupby("fight_id", sort=False):
        yield fid, g["red"].iat[0], g["blue"].iat[0], g["winner_id"].iat[0], g["date"].iat[0], g.index.to_numpy()


# ---------------------------------------------------------------------------
# tug-of-war
# ---------------------------------------------------------------------------

def run_tug(df, z, wt, league, prm: TugParams):
    """Pre-bout predictions for every round, plus per-round symmetric-preference pulls."""
    m = TugOfWar(league, prm)
    pred = np.full(len(df), np.nan)
    snap = np.full((len(df), 4), np.nan)    # p_red, p_blue, s_red - s_blue, n_min
    rnd = df["round"].to_numpy()
    for _, a, b, _, _, ix in bouts(df):
        pa, sa, _, na = m.snapshot(a)
        pb, sb, _, nb = m.snapshot(b)
        for i in ix:
            pred[i] = m.predict(a, b, int(rnd[i]))
            snap[i] = (pa, pb, sa - sb, min(na, nb))
        m.update(a, b, [(int(rnd[i]), z[i], wt[i]) for i in ix])
    return pred, snap, m


def wmse(y, p, w, mask):
    return float(np.sum(w[mask] * (y[mask] - p[mask]) ** 2) / np.sum(w[mask]))


def _tune(df, z, wt, league, mask, grid):
    """Best TugParams from `grid` (list of kwargs) by weighted MSE on `mask`."""
    return min(grid, key=lambda kw: wmse(z, run_tug(df, z, wt, league, TugParams(**kw))[0], wt, mask))


def _gain_ci(df, z, wt, base, new, rep):
    """Bootstrap-by-bout relative MSE gain of `new` over `base`: (mean, 2.5%, 97.5%)."""
    fid = df["fight_id"].to_numpy()[rep]
    d_err = (wt * ((z - base) ** 2 - (z - new) ** 2))[rep]
    uniq, inv = np.unique(fid, return_inverse=True)
    per, wsum = np.bincount(inv, d_err), np.bincount(inv, wt[rep])
    m0 = wmse(z, base, wt, rep)
    rng = np.random.default_rng(0)
    bs = [per[s].sum() / wsum[s].sum() / m0 for s in (rng.integers(0, len(uniq), len(uniq)) for _ in range(500))]
    return np.mean(bs), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def tug_report(df, dims):
    date = df["date"]
    tune = ((date >= TUNE_START) & (date < REPORT_START)).to_numpy()
    rep = (date >= REPORT_START).to_numpy()
    out = {}
    print("\n=== 1. Tug-of-war: predicting each round's ground share / clinch share / pace ===")
    print("weighted MSE on the transformed scale (logit share, log pace); lower is better")
    print("direction rule h: < 0 the fighter wanting LESS of it gets their way, > 0 the one wanting MORE")
    s_grid = [dict(k_s=ks, s_l2=l2) for ks in (0.4, 1.6, 3.2, 6.4) for l2 in (0.0, 0.01)]
    for k, (z, wt, league) in dims.items():
        sym, sym_snap, _ = run_tug(df, z, wt, league, TugParams(learn_s=False))
        sk = _tune(df, z, wt, league, tune, s_grid)
        skill, _, _ = run_tug(df, z, wt, league, TugParams(**sk))
        h = _tune(df, z, wt, league, tune, [dict(learn_s=False, h=h) for h in H_GRID])["h"]
        dirp, dir_snap, _ = run_tug(df, z, wt, league, TugParams(learn_s=False, h=h))
        dsk = _tune(df, z, wt, league, tune, [dict(g, h=h) for g in s_grid])
        both, _, model = run_tug(df, z, wt, league, TugParams(**dsk))
        kd = _tune(df, z, wt, league, tune, [dict(dsk, k_d=kd) for kd in (0.03, 0.1, 0.3, 1.0)])
        drift, _, _ = run_tug(df, z, wt, league, TugParams(**kd))
        g1, g2 = _gain_ci(df, z, wt, sym, skill, rep), _gain_ci(df, z, wt, dirp, both, rep)
        print(f"\n[{k}]  h={h}  skill {sk}")
        print(f"  league mean                  {wmse(z, np.full(len(df), league), wt, rep):.4f}")
        print(f"  split the difference         {wmse(z, sym, wt, rep):.4f}   (w = 0.5)")
        print(f"  + dictation skill            {wmse(z, skill, wt, rep):.4f}   vs split {g1[0]:+.2%} "
              f"(95% CI {g1[1]:+.2%} .. {g1[2]:+.2%})")
        print(f"  direction rule only          {wmse(z, dirp, wt, rep):.4f}")
        print(f"  direction rule + skill       {wmse(z, both, wt, rep):.4f}   vs rule {g2[0]:+.2%} "
              f"(95% CI {g2[1]:+.2%} .. {g2[2]:+.2%})")
        print(f"  + in-fight drift             {wmse(z, drift, wt, rep):.4f}")
        out[k] = dict(z=z, wt=wt, sym_snap=sym_snap, dir_snap=dir_snap, h=h, model=model)
    return out


# ---------------------------------------------------------------------------
# trait checks on the pull
# ---------------------------------------------------------------------------

def pull_table(df, z, wt, snap, h: float = 0.0, min_sep: float = 0.3) -> pd.DataFrame:
    """Per (bout, round, fighter): pull toward this fighter's preference (t) and the pull
    beyond what the direction rule expects (r = t - w_rule), from pre-bout preferences."""
    rows = []
    rnd = df["round"].to_numpy()
    for fid, a, b, win, date, ix in bouts(df):
        pa, pb = snap[ix[0], 0], snap[ix[0], 1]
        sep = pa - pb
        if abs(sep) < min_sep or snap[ix[0], 3] < 3:
            continue
        w_rule = float(sigmoid(h * np.tanh(2.0 * sep)))
        for i in ix:
            t = pull(z[i], pa, pb)
            w = wt[i] * sep ** 2
            if w <= 0:
                continue
            for f, o, tt, wr in ((a, b, t, w_rule), (b, a, 1 - t, 1 - w_rule)):
                rows.append((fid, f, o, date, int(rnd[i]), tt, tt - wr, w, win == f, win == o))
    return pd.DataFrame(rows, columns=["fight_id", "fighter", "opp", "date", "round", "t", "r", "w", "won", "lost"])


def wmean(d: pd.DataFrame, keys, value: str = "t", extra=()) -> pd.DataFrame:
    """Weighted mean of `value` (weights w) per group; `extra` columns take the first value."""
    g = d.assign(_tw=d[value] * d["w"]).groupby(keys, sort=False)
    out = g[["_tw", "w"]].sum()
    for c in extra:
        out[c] = g[c].first()
    out[value] = out.pop("_tw") / out["w"]
    return out[out["w"] > 0].reset_index()


def split_half(per_bout: pd.DataFrame, min_w: float, value: str = "t") -> tuple[float, int]:
    """Correlation of a fighter's weighted-mean value on odd vs even bouts."""
    per_bout = per_bout.sort_values("date")
    per_bout["k"] = per_bout.groupby("fighter").cumcount() % 2
    agg = per_bout.assign(tw=per_bout[value] * per_bout["w"]).groupby(["fighter", "k"])[["tw", "w"]].sum()
    agg["m"] = agg["tw"] / agg["w"]
    wide = agg["m"].unstack()
    ok = agg["w"].unstack().min(axis=1) >= min_w
    wide = wide[ok].dropna()
    if len(wide) < 10:
        return float("nan"), len(wide)
    return float(np.corrcoef(wide[0], wide[1])[0, 1]), len(wide)


def _trait_lines(pt: pd.DataFrame, value: str) -> tuple[float, float, float, int, int]:
    """(mean when winning, mean when losing, split-half r, n, wins-vs-losses r) for `value`."""
    pb = wmean(pt, ["fight_id", "fighter", "date"], value=value, extra=("won", "lost"))
    min_w = pb["w"].median() * 2
    r_all, n_all = split_half(pb, min_w, value)
    won, lost = pb["won"].astype(bool), pb["lost"].astype(bool)
    agg = lambda m: pb[m].assign(tw=pb[value] * pb["w"]).groupby("fighter")[["tw", "w"]].sum()
    j = agg(won).join(agg(lost), lsuffix="_w", rsuffix="_l").dropna()
    j = j[(j["w_w"] >= min_w / 2) & (j["w_l"] >= min_w / 2)]
    r_wl = np.corrcoef(j["tw_w"] / j["w_w"], j["tw_l"] / j["w_l"])[0, 1] if len(j) > 10 else np.nan
    mw = np.average(pb.loc[won, value], weights=pb.loc[won, "w"])
    ml = np.average(pb.loc[lost, value], weights=pb.loc[lost, "w"])
    return mw, ml, r_all, n_all, r_wl


def trait_report(df, tug):
    print("\n=== 2. Is the pull a trait? (pull = how far toward a fighter's own preference the round landed) ===")
    print("raw = vs split-the-difference; beyond rule = pull minus what the direction rule expects")
    for k in ("ground", "clinch", "pace"):
        d = tug[k]
        raw = pull_table(df, d["z"], d["wt"], d["sym_snap"])
        res = pull_table(df, d["z"], d["wt"], d["dir_snap"], h=d["h"])
        print(f"\n[{k}]  bouts with a real style gap: {raw.fight_id.nunique()}")
        for label, pt, v in (("raw", raw, "t"), ("beyond rule", res, "r")):
            mw, ml, r_all, n_all, r_wl = _trait_lines(pt, v)
            print(f"  {label:11s}  winning {mw:+.2f} / losing {ml:+.2f}   split-half r = {r_all:.2f} "
                  f"(n = {n_all})   wins-vs-losses r = {r_wl:.2f}")
        tug[k]["pull"] = raw
    return tug


def drift_report(df, tug):
    print("\n=== 3. Round dynamics (orientation: the fighter who prefers MORE ground / clinch / pace) ===")
    for k in ("ground", "clinch", "pace"):
        pt = tug[k]["pull"]
        z = tug[k]["sym_snap"]
        # orient on the fighter with the higher preference: their pull is t when they are red
        pref = pd.DataFrame({"fight_id": df["fight_id"], "red": df["red"], "pr": z[:, 0], "pb": z[:, 1]}) \
            .drop_duplicates("fight_id")
        pref["hi"] = np.where(pref["pr"] >= pref["pb"], pref["red"], "")
        hi = pt.merge(pref[["fight_id", "red", "pr", "pb"]], on="fight_id")
        hi = hi[((hi.fighter == hi.red) & (hi.pr >= hi.pb)) | ((hi.fighter != hi.red) & (hi.pb > hi.pr))]
        hi = hi[hi["round"] <= 5]
        by = wmean(hi, ["round"]).set_index("round").sort_index()
        by["n"] = hi.groupby("round").size()
        print(f"\n[{k}]  pull of the higher-preference fighter by round")
        print("  " + "   ".join(f"R{int(r)} {v.t:.2f} (n={int(v.n)})" for r, v in by.iterrows()))
        # per-fighter drift reliability: late (R2-3) minus R1 pull, split-half
        pt2 = pt[pt["round"] <= 3].copy()
        pt2["late"] = pt2["round"] > 1
        g = wmean(pt2, ["fight_id", "fighter", "date", "late"]).set_index(
            ["fight_id", "fighter", "date", "late"])
        t = g["t"].unstack().dropna()
        w = g["w"].unstack().loc[t.index].min(axis=1)
        dd = pd.DataFrame({"t": t[True] - t[False], "w": w}).reset_index()
        r, n = split_half(dd, dd["w"].median() * 2)
        print(f"  per-fighter drift (R2-3 minus R1) split-half r = {r:.2f}  (n = {n} fighters)")


# ---------------------------------------------------------------------------
# mechanics
# ---------------------------------------------------------------------------

def mech_rounds(df):
    """(red half, blue half) MechRound per row."""
    out = []
    for r in df.itertuples(index=False):
        out.append((MechRound(r.secs, r.r_td_attempted, r.r_td_landed, r.r_ctrl_seconds, r.b_ctrl_seconds),
                    MechRound(r.secs, r.b_td_attempted, r.b_td_landed, r.b_ctrl_seconds, r.r_ctrl_seconds)))
    return out


def run_mech(df, mr, prm: MechParams, keep_snap: bool = False):
    m = Mechanics(prm)
    early = (df["date"] < TUNE_START).to_numpy()
    m.init_league([h for i, pair in enumerate(mr) if early[i] for h in pair])
    pred = np.full((len(df), 2), np.nan)
    one = np.full((len(df), 2), np.nan)
    snap = [None] * len(df) if keep_snap else None
    secs = df["secs"].to_numpy()
    for _, a, b, _, _, ix in bouts(df):
        for i in ix:
            pred[i] = m.predict_share(a, b, secs[i])
            one[i] = m.rates(a, b)["ctl"], m.rates(b, a)["ctl"]
            if keep_snap:
                snap[i] = (m.rates(a, b), m.rates(b, a), dict(m.st[a]), dict(m.st[b]), m.c_hold)
        for i in ix:
            m.update(a, b, mr[i][0])
            m.update(b, a, mr[i][1])
    return pred, one, snap


def mech_report(df):
    print("\n=== 4. Control mechanics: shot rate x success x hold, with an escape skill ===")
    mr = mech_rounds(df)
    y = np.column_stack([df["r_ctrl_seconds"] / df["secs"], df["b_ctrl_seconds"] / df["secs"]]).clip(0, 1)
    w = (df["secs"] / 300).to_numpy()[:, None].repeat(2, 1)
    date = df["date"]
    tune = ((date >= TUNE_START) & (date < REPORT_START)).to_numpy()
    rep = (date >= REPORT_START).to_numpy()

    def score(p, mask):
        return float(np.sum(w[mask] * (y[mask] - p[mask]) ** 2) / np.sum(w[mask]))

    best = None
    for lr in (0.01, 0.03, 0.06):
        for l2 in (0.0, 0.002, 0.01):
            p, one, _ = run_mech(df, mr, MechParams(lr=lr, l2=l2))
            sc = score(p, tune)
            if best is None or sc < best[0]:
                best = (sc, lr, l2)
    _, lr, l2 = best
    full, one, snap = run_mech(df, mr, MechParams(lr=lr, l2=l2), keep_snap=True)
    noesc, _, _ = run_mech(df, mr, MechParams(lr=lr, l2=l2, use_escape=False))
    league = np.full_like(y, np.average(y[tune], weights=w[tune]))
    print(f"tuned lr={lr} l2={l2}; MSE of each fighter's control share per round (x1000), {rep.sum()} rounds")
    for name, p in (("league mean", league), ("one-parameter off/def", one),
                    ("mechanics, no escape skill", noesc), ("mechanics + escape skill", full)):
        print(f"  {name:28s} {1000 * score(p, rep):.3f}")

    # wrestler vs escaper: attacker's entry threat high and defender's escape skill high
    rows = []
    for i in np.where(rep)[0]:
        ra, rb, sa, sb, c_hold = snap[i]
        for side, (r_att, s_def) in enumerate(((ra, sb), (rb, sa))):
            threat = r_att["shoot"] * r_att["succ"] * 300
            rows.append((threat, s_def["esc"], y[i, side], full[i, side], one[i, side], noesc[i, side], w[i, side]))
    t = pd.DataFrame(rows, columns=["threat", "esc", "y", "full", "one", "noesc", "w"])
    hi_t = t["threat"] >= t["threat"].quantile(0.8)
    for label, sel in (("elite TD threat vs good escaper", hi_t & (t["esc"] >= t["esc"].quantile(0.8))),
                       ("elite TD threat vs poor escaper", hi_t & (t["esc"] <= t["esc"].quantile(0.2)))):
        s = t[sel]
        print(f"  {label:34s} n={len(s):4d}  actual {s.y.mean():.3f}  "
              f"mechanics {s.full.mean():.3f}  no-escape {s.noesc.mean():.3f}  one-param {s.one.mean():.3f}")

    # escape-skill reliability: defender's hold residual, odd vs even bouts
    rows = []
    for i in range(len(df)):
        if snap[i] is None or df["date"].iat[i] < TUNE_START:
            continue
        ra, rb, sa, sb, c_hold = snap[i]
        for att, dfd, half, s_att in ((df.red.iat[i], df.blue.iat[i], mr[i][0], sa),
                                       (df.blue.iat[i], df.red.iat[i], mr[i][1], sb)):
            n = half.td_land if half.td_land > 0 else (1.0 if half.ctrl >= 5 else 0.0)
            if n == 0:
                continue
            resid = np.log((half.ctrl + 1) / n) - (c_hold + s_att["top"])
            rows.append((df.fight_id.iat[i], dfd, df.date.iat[i], -resid, n))
    e = pd.DataFrame(rows, columns=["fight_id", "fighter", "date", "t", "w"])
    e = wmean(e, ["fight_id", "fighter", "date"])
    r, n = split_half(e, 4)
    print(f"  escape skill (shorter spells than the attacker usually gets) split-half r = {r:.2f}  (n = {n})")
    return snap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    df = load_rounds(a.rebuild)
    print(f"{len(df)} rounds, {df.fight_id.nunique()} bouts, {df.date.min().date()} .. {df.date.max().date()}; "
          f"report window from {REPORT_START.date()}: {(df.date >= REPORT_START).sum()} rounds")
    dims = add_dimensions(df)
    tug = tug_report(df, dims)
    tug = trait_report(df, tug)
    drift_report(df, tug)
    mech_report(df)
    with open(OUT_DIR / "tug_models.pkl", "wb") as f:
        pickle.dump({k: v["model"].st for k, v in tug.items()}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
