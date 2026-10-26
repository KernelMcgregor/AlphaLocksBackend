"""Paired walk-forward test of a candidate feature block in the winner model.

One harness for every as-of feature module in app/services/ufc/pit/. Same folds,
imputation, isotonic calibration and market anchor as scripts/fast_wf.py; the 4 ensemble
members (ensemble.MEMBERS) are run on the baseline features and on baseline + the block.
Ensemble = sigmoid of the mean of the members' calibrated logits, as served.

A module may instead expose transform(matchup, features, slim, session_factory, variant)
-> (new columns, columns replaced in place): for arms that rewrite existing features
(e.g. newcomer imputation). The baseline arm always runs on an untouched copy.

A module exposes
    PREFIX                      column prefix, e.g. "sty_"
    compute(ctx) -> DataFrame   keyed (fight_id, fighter_id): one row per corner, values as
                                of BEFORE the fight. Columns listed in FIGHT_LEVEL (optional)
                                are symmetric matchup values and enter once as fight_<col>;
                                every other column enters as diff_<col> (red minus blue), and
                                also red_/blue_<col> for those in RAW (optional).
ctx is a FeatureContext: the slim per-fighter frame from build_features (elo_expected,
xsres_*, results) and a read-only DB session factory.

The matrix is cached separately from fast_wf's (its file is shared with other experiments):
    models/ufc/h2h/feature_wf_matrix.pkl   (--rebuild re-reads prod, ~15-20 min, read-only)

Usage:
    PYTHONPATH=. venv/bin/python -m scripts.feature_wf --build-only --rebuild
    PYTHONPATH=. venv/bin/python -m scripts.feature_wf style            # all of a module's cols
    PYTHONPATH=. venv/bin/python -m scripts.feature_wf style --cols sty_like_resid_p1 sty_analog_resid_p1
    PYTHONPATH=. venv/bin/python -m scripts.feature_wf style --drop diff_style_matchup_adv
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("feature_wf")

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "models" / "ufc" / "h2h" / "feature_wf_matrix_v2.pkl"
OUT_DIR = ROOT / "data" / "feature_wf"
ARMS = ["noodds:catboost:0", "noodds_noraw:catboost:0", "noodds:hgb:0", "noodds_nosn:logit:0"]
EVAL_FRAC = 0.4
COMBINE = "logit"      # how members combine into the ensemble; set from --combine

#: Per-fighter columns kept from build_features for the pit modules.
SLIM_COLS = ["fight_id", "stats_fighter_id", "corner", "date", "red_fighter_id", "blue_fighter_id",
             "winner_id", "won", "elo", "elo_expected", "method", "weight_class", "_unplayed"]


@dataclass
class FeatureContext:
    fighters: pd.DataFrame        # slim per-fighter frame (one row per fighter per fight)
    matchup: pd.DataFrame         # the cached matchup matrix (fight_id, date, red_wins, ...)

    @staticmethod
    def session():
        from app.database import SessionLocal
        return SessionLocal()


def build_matrix(rebuild: bool = False):
    """(matchup, features, slim per-fighter frame). Reads prod, writes only the local cache."""
    if CACHE.exists() and not rebuild:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    from app.services.ufc.glicko_service import run_glicko_inmemory
    from app.services.ufc.model import build_features, build_matchup_df, load_fight_data
    snaps = run_glicko_inmemory()
    df, rd = load_fight_data()
    df = build_features(df, rd, glicko_snapshots=snaps)
    # the full per-fighter frame: newcomer / mismatch arms need each corner's own values,
    # which the matchup matrix only carries as red-minus-blue diffs
    slim = df.copy()
    matchup, features = build_matchup_df(df)
    matchup = matchup.sort_values("date").reset_index()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump((matchup, features, slim), f)
    return matchup, features, slim


def attach(matchup: pd.DataFrame, slim: pd.DataFrame, block: pd.DataFrame,
           fight_level=(), raw=()) -> list[str]:
    """Add a module's per-corner block to the matchup frame; returns the new column names."""
    corners = slim[["fight_id", "stats_fighter_id", "corner"]].drop_duplicates(["fight_id", "corner"])
    b = block.rename(columns={"fighter_id": "stats_fighter_id"}).merge(
        corners, on=["fight_id", "stats_fighter_id"], how="inner")
    cols = [c for c in block.columns if c not in ("fight_id", "fighter_id")]
    red = b[b.corner == "red"].set_index("fight_id")
    blue = b[b.corner == "blue"].set_index("fight_id")
    new = []
    for c in cols:
        r = matchup.fight_id.map(red[c]) if c in red else np.nan
        bl = matchup.fight_id.map(blue[c]) if c in blue else np.nan
        if c in fight_level:
            matchup[f"fight_{c}"] = r
            new.append(f"fight_{c}")
            continue
        matchup[f"diff_{c}"] = r - bl
        new.append(f"diff_{c}")
        if c in raw:
            matchup[f"red_{c}"], matchup[f"blue_{c}"] = r, bl
            new += [f"red_{c}", f"blue_{c}"]
    return new


def _logit_mean(frame: pd.DataFrame) -> pd.Series:
    """Members combined as prod does (ensemble.Ensemble.model_prob): sigmoid of mean logit."""
    p = frame.clip(1e-4, 1 - 1e-4)
    z = np.log(p / (1 - p)).mean(axis=1)
    return 1 / (1 + np.exp(-z))


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(matchup, base_feats, cand_feats, arms, label):
    """Run both variants on one frame; returns (results on the eval window, summary dict)."""
    return evaluate_pair(matchup, matchup, base_feats, cand_feats, arms, label)


def evaluate_pair(base_frame, cand_frame, base_feats, cand_feats, arms, label, cand_arms=None):
    """Baseline on base_frame, candidate on cand_frame (same rows, same order)."""
    cand_arms = cand_arms or arms
    from scripts.fast_wf import run_arm
    assert (base_frame.fight_id.to_numpy() == cand_frame.fight_id.to_numpy()).all()
    matchup = cand_frame
    n = len(matchup)
    eval_start = int(n * (1 - EVAL_FRAC))
    res = matchup[["fight_id", "date", "red_wins", "odds_red_prob"]].copy()
    for variant, feats, frame, members in (("base", base_feats, base_frame, arms),
                                           ("cand", cand_feats, cand_frame, cand_arms)):
        for spec in members:
            raw, cal, anc = run_arm(spec, frame, feats)
            tag = f"{variant}_{spec.replace(':', '_')}"
            res[f"{tag}_cal"], res[f"{tag}_anc"] = cal, anc
        for kind in ("cal", "anc"):
            cols = [f"{variant}_{s.replace(':', '_')}_{kind}" for s in members]
            res[f"{variant}_ensemble_{kind}"] = (_logit_mean(res[cols]) if COMBINE == "logit"
                                                 else res[cols].mean(axis=1))
    res = res.iloc[eval_start:]
    y = res.red_wins.to_numpy(float)
    ok = ~np.isnan(y)
    priced = ok & res.odds_red_prob.notna().to_numpy()
    summary = {"label": label, "n_eval": int(ok.sum()), "n_priced": int(priced.sum()), "rows": []}
    for tag in ([s.replace(":", "_") for s in arms] if cand_arms == arms else []) + ["ensemble"]:
        for kind, mask in (("cal", ok), ("anc", priced)):
            b = _ll(res[f"base_{tag}_{kind}"].to_numpy(float)[mask], y[mask])
            c = _ll(res[f"cand_{tag}_{kind}"].to_numpy(float)[mask], y[mask])
            lo, hi = _ci(c - b)
            summary["rows"].append({"member": tag, "kind": kind, "base": float(b.mean()),
                                    "cand": float(c.mean()), "delta": float((c - b).mean()),
                                    "lo": lo, "hi": hi})
    return res, summary


def print_summary(s):
    print(f"\nPAIRED: {s['label']}  (negative = better)  eval {s['n_eval']:,}, priced {s['n_priced']:,}")
    print(f"  {'member':<26} {'':4} {'base':>8} {'cand':>8} {'delta':>8} {'95% CI':>20}")
    for r in s["rows"]:
        kind = "all" if r["kind"] == "cal" else "anc"
        print(f"  {r['member']:<26} {kind:4} {r['base']:>8.4f} {r['cand']:>8.4f} {r['delta']:>+8.4f}"
              f"   [{r['lo']:+.4f}, {r['hi']:+.4f}]")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("module", nargs="?", help="pit module name, e.g. style, rank, tug, dtw")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the matrix from prod (read-only)")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--recompute", action="store_true", help="recompute the module's block")
    ap.add_argument("--cols", nargs="*", help="only these module columns (names without diff_)")
    ap.add_argument("--drop", nargs="*", default=[], help="baseline columns removed in the candidate arm")
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--variant", default=None, help="passed to a module's transform()")
    ap.add_argument("--cand-arms", nargs="*", default=None,
                    help="members for the candidate arm only (capacity tests); default = --arms")
    ap.add_argument("--combine", choices=("logit", "prob"), default="logit",
                    help="how members are combined into the ensemble (prod: logit)")
    a = ap.parse_args(argv)

    matchup, features, slim = build_matrix(a.rebuild)
    log.info("matrix: %d fights, %d features", len(matchup), len(features))
    global COMBINE
    COMBINE = a.combine
    if a.build_only:
        return
    if not a.module:            # members-only comparison: same features, different members
        if not a.cand_arms:
            return
        tag = a.tag or "members"
        res, summary = evaluate_pair(matchup, matchup, features, features, a.arms,
                                     f"{tag}: members {a.cand_arms}", cand_arms=a.cand_arms)
        print_summary(summary)
        res.to_csv(OUT_DIR / f"{tag}_wf.csv", index=False)
        (OUT_DIR / f"{tag}.json").write_text(json.dumps(summary, indent=1))
        return

    mod = importlib.import_module(f"app.services.ufc.pit.{a.module}_pit")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(mod, "transform"):
        base = matchup.copy()
        new, replaced = mod.transform(matchup, features, slim, FeatureContext.session,
                                      **({"variant": a.variant} if a.variant else {}))
        tag = a.tag or f"{a.module}_{a.variant or 'default'}"
        ev = matchup.iloc[int(len(matchup) * (1 - EVAL_FRAC)):]
        evb = base.iloc[int(len(base) * (1 - EVAL_FRAC)):]
        for c in new:
            log.info("  new %-34s coverage %5.1f%%  sd %.4f", c, 100 * ev[c].notna().mean(), ev[c].std())
        if replaced:
            log.info("  replaced %d columns; coverage %.1f%% -> %.1f%%", len(replaced),
                     100 * evb[replaced].notna().to_numpy().mean(), 100 * ev[replaced].notna().to_numpy().mean())
        cand_feats = [f for f in features if f not in a.drop] + new
        res, summary = evaluate_pair(base, matchup, features, cand_feats, a.arms,
                                     f"{tag}: +{len(new)} cols, {len(replaced)} rewritten")
        summary.update(columns=new, replaced=replaced)
        print_summary(summary)
        res.to_csv(OUT_DIR / f"{tag}_wf.csv", index=False)
        (OUT_DIR / f"{tag}.json").write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {OUT_DIR / (tag + '.json')}")
        return
    block_path = OUT_DIR / f"{a.module}_block.pkl"
    if block_path.exists() and not a.recompute:
        block = pd.read_pickle(block_path)
    else:
        block = mod.compute(FeatureContext(fighters=slim, matchup=matchup))
        block.to_pickle(block_path)
    if a.cols:
        block = block[["fight_id", "fighter_id"] + [c for c in block.columns if c in a.cols]]
    new = attach(matchup, slim, block, getattr(mod, "FIGHT_LEVEL", ()), getattr(mod, "RAW", ()))
    ev = matchup.iloc[int(len(matchup) * (1 - EVAL_FRAC)):]
    for c in new:
        log.info("  %-36s coverage %5.1f%%  sd %.4f", c, 100 * ev[c].notna().mean(), ev[c].std())
    cand = [f for f in features if f not in a.drop] + new
    tag = a.tag or a.module
    res, summary = evaluate(matchup, features, cand, a.arms, f"{tag}: +{len(new)} cols, -{len(a.drop)}")
    summary["columns"] = new
    summary["dropped"] = a.drop
    print_summary(summary)
    res.to_csv(OUT_DIR / f"{tag}_wf.csv", index=False)
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT_DIR / (tag + '.json')}")


if __name__ == "__main__":
    main()
