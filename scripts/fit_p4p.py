"""Fit the P4P component weights to UFC.com's own pound-for-pound lists.

For every archived monthly snapshot (data/ufc_rankings_history.json, built by
scripts/ufc_rankings_snapshots.py) this rebuilds our P4P candidate components as of that
date and fits non-negative weights so the weighted score orders fighters the way UFC
did: each listed fighter above every fighter listed below them, and above every eligible
fighter UFC left off. Pairwise logistic loss, weights through softplus so none goes
negative, normalised to sum to 1.

Validation is out of time: fit on 2024-2025 snapshots, score on 2026 ones. Reported:
  top15   mean overlap between our top 15 and UFC's
  rho     Spearman between our order and UFC's, over UFC's 15
  pairs   share of UFC-implied pairs we order the same way
All three after the division merge (nobody above a fighter their own division ranks
higher), unless --no-merge.

Read-only. Run: python -m scripts.fit_p4p [--split 20260101] [--features a,b,c]
"""
from __future__ import annotations

import argparse
import json
import logging
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

from app.database import SessionLocal
from app.services.ufc import p4p_rankings as p4p
from app.services.ufc.alt_rankings_common import champions_at, eligible_fighters, load_context
from app.services.ufc.fighter_registry import build_fighter_registry

#: Apply the division-order merge before scoring a list (--no-merge to compare).
MERGE = True

HISTORY = Path(__file__).resolve().parent.parent / "data" / "ufc_rankings_history.json"
POOL_TITLES = {"men": "Men's Pound-for-Pound", "women": "Women's Pound-for-Pound"}
ALL_FEATURES = ["dominance", "resume", "depth", "recency", "champion", "titles",
                "title_fights", "streak", "strength", "best_wins", "resume_long", "champ_wins", "elo_wins", "div_points"]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").replace(".", "").split())


def build_dataset(db, history: dict) -> list[dict]:
    """One entry per (snapshot, pool): feature matrix, UFC's order, champion info."""
    base = load_context(db)
    by_name: dict[str, list[int]] = {}
    for fid, n in base["hist"]["names"].items():
        by_name.setdefault(norm(n), []).append(fid)

    out = []
    for ds, snap in history.items():
        d = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))
        ctx = {**base, "today": d, "registry": build_fighter_registry(db, as_of=d),
               "champions": champions_at(base["hist"], d),
               **p4p.division_context(base["hist"], d)}
        fighters = eligible_fighters(ctx)
        feats = p4p.features(ctx, fighters)
        ranks = p4p.division_ranks(ctx, fighters)
        for pool, title in POOL_TITLES.items():
            members = [f for f in fighters if f.pool == pool]
            ids = [f.fighter_id for f in members]
            pos = {fid: i for i, fid in enumerate(ids)}
            listed, missing = [], []
            for name in snap.get(title, {}).get("ranked", []):
                cands = [c for c in by_name.get(norm(name), []) if c in pos]
                (listed.append(pos[cands[0]]) if cands else missing.append(name))
            out.append({
                "date": ds, "pool": pool, "ids": ids,
                "X": np.array([[feats[f][k] for k in ALL_FEATURES] for f in ids], float),
                "listed": listed, "missing": missing,
                "division": [ranks.get(f.fighter_id, (f.division, None))[0] for f in members],
                "division_rank": [ranks.get(f.fighter_id, (None, None))[1] for f in members],
                "champion": np.array([f.is_champion for f in members]),
            })
    return out


def pairs(entry) -> tuple[np.ndarray, np.ndarray]:
    """(winner_idx, loser_idx): listed above listed-below, listed above unlisted."""
    listed = entry["listed"]
    unlisted = np.setdiff1d(np.arange(len(entry["ids"])), listed)
    w, l = [], []
    for i, a in enumerate(listed):
        for b in listed[i + 1:]:
            w.append(a); l.append(b)
        w.extend([a] * len(unlisted)); l.extend(unlisted)
    return np.array(w, int), np.array(l, int)


def weights_from(theta):
    w = np.log1p(np.exp(theta[:-1]))
    return w / w.sum(), np.exp(theta[-1])


def fit(train: list[dict], cols: list[int], fixed: dict[int, float] | None = None) -> np.ndarray:
    """`fixed` pins some columns (by position in `cols`) to a weight; the rest share the
    remaining mass."""
    fixed = fixed or {}
    free = [i for i in range(len(cols)) if i not in fixed]
    mass = 1.0 - sum(fixed.values())
    prepared = [(e["X"][:, cols] / 100.0, *pairs(e)) for e in train]

    def full(theta):
        wf, s = weights_from(theta)
        w = np.zeros(len(cols))
        w[free] = wf * mass
        for i, v in fixed.items():
            w[i] = v
        return w, s

    def loss(theta):
        w, s = full(theta)
        tot, n = 0.0, 0
        for X, a, b in prepared:
            z = s * (X[a] - X[b]) @ w
            tot += np.logaddexp(0, -z).sum()
            n += len(a)
        return tot / n

    theta0 = np.r_[np.zeros(len(free)), np.log(10.0)]
    res = minimize(loss, theta0, method="L-BFGS-B")
    return full(res.x)[0]


def ordered(entry, cols, w) -> list[int]:
    score = entry["X"][:, cols] @ w
    rows = [{"i": i, "pool": entry["pool"], "division": entry["division"][i],
             "division_rank": entry["division_rank"][i],
             "is_champion": bool(entry["champion"][i]), "default_score": float(score[i])}
            for i in range(len(score))]
    if MERGE:
        p4p.division_merge(rows)
    rows.sort(key=lambda r: (-r["default_score"], r["division_rank"] or 999,
                             -r.get("score_raw", r["default_score"])))
    return [r["i"] for r in rows]


def evaluate(entries, cols, w) -> dict:
    top, rho, pr = [], [], []
    for e in entries:
        if len(e["listed"]) < 10:
            continue
        order = ordered(e, cols, w)
        rank = {i: k for k, i in enumerate(order)}
        top.append(len(set(order[:15]) & set(e["listed"])) / 15)
        rho.append(spearmanr(range(len(e["listed"])), [rank[i] for i in e["listed"]])[0])
        a, b = pairs(e)
        pr.append(np.mean([rank[x] < rank[y] for x, y in zip(a, b)]))
    return {"top15": float(np.mean(top)), "rho": float(np.mean(rho)), "pairs": float(np.mean(pr))}


def main() -> None:
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="20260101")
    ap.add_argument("--features", default=",".join(ALL_FEATURES))
    ap.add_argument("--show", default="20261002", help="snapshot to print side by side")
    ap.add_argument("--fix", default="", help="pin weights, e.g. champion=0.05")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    global MERGE
    MERGE = not a.no_merge

    history = json.loads(HISTORY.read_text())
    data = build_dataset(SessionLocal(), history)
    gaps = sorted({m for e in data for m in e["missing"]})
    if gaps:
        print(f"UFC-listed fighters not eligible / not matched on some dates: {len(gaps)}: "
              + ", ".join(gaps[:25]))

    train = [e for e in data if e["date"] < a.split]
    test = [e for e in data if e["date"] >= a.split]
    feats = a.features.split(",")
    cols = [ALL_FEATURES.index(f) for f in feats]

    current = [ALL_FEATURES.index(k) for k in p4p.DEFAULT_WEIGHTS]
    cur_w = np.array([p4p.DEFAULT_WEIGHTS[k] for k in p4p.DEFAULT_WEIGHTS])
    print(f"\ncurrent defaults  train {evaluate(train, current, cur_w)}")
    print(f"                  test  {evaluate(test, current, cur_w)}")

    fixed = {feats.index(k): float(v) for k, v in
             (kv.split("=") for kv in a.fix.split(",") if kv)}
    w = fit(train, cols, fixed)
    print(f"\nfitted on {len(train)} pool-snapshots, tested on {len(test)}")
    for f, v in sorted(zip(feats, w), key=lambda t: -t[1]):
        print(f"  {f:<13} {v:.3f}")
    print(f"fitted            train {evaluate(train, cols, w)}")
    print(f"                  test  {evaluate(test, cols, w)}")
    latest = [e for e in data if e["date"] == a.show and e["pool"] == "men"]
    if latest:
        top8 = ordered(latest[0], cols, w)[:8]
        print(f"champions in our men's top 8 on {a.show}: "
              f"{int(sum(latest[0]['champion'][i] for i in top8))}")

    for e in data:
        if e["date"] == a.show:
            order = ordered(e, cols, w)
            ufc = [e["ids"][i] for i in e["listed"]]
            ours = [e["ids"][i] for i in order[:15]]
            db = SessionLocal()
            from app.models.ufc import UFCFighter
            nm = {f.id: f"{f.first_name} {f.last_name}" for f in
                  db.query(UFCFighter).filter(UFCFighter.id.in_(set(ufc) | set(ours)))}
            print(f"\n{e['pool']} {e['date']}:  {'UFC':<26} ours")
            for k in range(15):
                u = nm.get(ufc[k], "?") if k < len(ufc) else ""
                print(f"  {k + 1:>2}  {u:<26} {nm.get(ours[k], '?')}")


if __name__ == "__main__":
    main()
