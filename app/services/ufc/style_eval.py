"""Style space benchmark and gate — the numbers quoted in style_service / style_page_service.

READ-ONLY: one read-only DB session, then everything runs in memory. Nothing is written
except --out (a JSON of the results).

What it measures, for the FROZEN space and map in models/ufc/style/:

  self-recognition  Split each fighter (6+ bouts) into two halves of their fights —
                    odd/even AND early/late career — fingerprint each half through the
                    frozen space, and rank the fighter's other half among every candidate.
                    Reported on a fixed held-out half of fighters (crc32(id) odd); the other
                    half is where settings were tuned.
  quality R^2       5-fold ridge R^2 of overall Glicko quality from the style vector. How
                    much skill leaks into "style"; lower is better.
  labels            UFC.com fighting_style, bucketed (wrestler / bjj / kickboxer / boxer;
                    generic "Striker" only at the coarse striker-vs-grappler level): share of
                    a labelled fighter's 10 nearest labelled neighbours with the same bucket,
                    as a multiple of chance. Noisy (SD ~0.035 fine, ~0.02 coarse).
  transfer          Common-opponent agreement, residualised on a 10 x 5 grid of (|quality
                    gap|, pair mean quality): top-decile lift. Quality alone scores ~0.
  --families        Per-family bootstrap Jaccard of KMeans(k) for k = 2..8 against a Gaussian
                    null with the data's covariance, and the blend-temperature calibration
                    against fight-bootstrap assignment frequencies.

GATE: self-recognition on the held-out odd/even split must find the fighter first at least
25x chance and put the median rank inside the top 5% of the pool. Exit status 1 on FAIL.

Run:  python -m app.services.ufc.style_eval [--families] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import zlib
from collections import defaultdict

import numpy as np
from sqlalchemy import text

from app.database import SessionLocal
from app.models.ufc import UFC_SCHEMA
from app.services.ufc import style_habits as sh
from app.services.ufc import style_page_service as sp
from app.services.ufc import style_service as ss
from app.services.ufc.fighter_registry import is_decided

log = logging.getLogger("style_eval")
MIN_SPLIT = 6

BUCKETS = {
    "Wrestling": "wrestle", "Wrestler": "wrestle", "Sambo": "wrestle",
    "Jiu-Jitsu": "bjj", "Brazilian Jiu-Jitsu": "bjj", "Grappler": "bjj", "Judo": "bjj",
    "Muay Thai": "kick", "Kickboxer": "kick", "Karate": "kick", "Taekwondo": "kick",
    "Kung Fu": "kick", "Kung-Fu": "kick",
    "Boxing": "box", "Boxer": "box", "Brawler": "box",
    "Striker": "striker",
}


def load(db):
    inputs = ss.build_inputs(db)
    bouts = sh.load_bouts(db)
    model = sh.fit_opponent_model(bouts)
    P = f"{UFC_SCHEMA}." if UFC_SCHEMA else ""
    labels = {int(r.id): r.fighting_style for r in db.execute(text(
        f"SELECT id, fighting_style FROM {P}ufc_fighters WHERE coalesce(fighting_style,'') <> ''"))}
    fights = db.execute(text(
        f"SELECT red_fighter_id, blue_fighter_id, winner_id, method, date FROM {P}ufc_fights")).fetchall()
    return inputs, bouts, model, labels, fights


def unit(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def vectors(rows_bouts, keys, inputs, model, space):
    X = np.array([[sh.style_features(bs, model)[c] for c in ss.ALL_FEATURES] for bs in rows_bouts])
    prime = np.array([inputs.prime[inputs.fids.index(k)] for k in keys])
    return unit(ss.transform(X, prime, [inputs.divisions[k] for k in keys], space))


def self_recognition(inputs, bouts, model, space):
    pool = [f for f in inputs.fids if len(bouts[f]) >= MIN_SPLIT]
    held = np.array([zlib.crc32(str(f).encode()) % 2 == 1 for f in pool])
    out = {"n": len(pool)}
    for split in ("oddeven", "chrono"):
        halves = []
        for h in (0, 1):
            hb = []
            for f in pool:
                bs = bouts[f]
                n = len(bs) // 2
                hb.append(bs[h::2] if split == "oddeven" else (bs[:n] if h == 0 else bs[n:]))
            halves.append(vectors(hb, pool, inputs, model, space))
        S = halves[0] @ halves[1].T
        d = np.diag(S)
        ranks = np.concatenate([(S > d[:, None]).sum(1) + 1, (S.T > d[:, None]).sum(1) + 1])
        ev = np.concatenate([held, held])
        out[split] = {"held_out_top1": round(float(np.mean(ranks[ev] == 1)), 4),
                      "held_out_median_rank": float(np.median(ranks[ev])),
                      "tuning_top1": round(float(np.mean(ranks[~ev] == 1)), 4),
                      "tuning_median_rank": float(np.median(ranks[~ev]))}
    return out


def quality(inputs):
    return np.array([np.mean([inputs.profiles[f][c] for c in ss.SKILL_FEATURES]) for f in inputs.fids])


def quality_r2(Z, q):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), Z, q, cv=KFold(5, shuffle=True, random_state=0))
    return float(1 - np.sum((q - pred) ** 2) / np.sum((q - q.mean()) ** 2))


def label_lift(U, fids, labels, coarse):
    lab = {f: BUCKETS.get(labels.get(f)) for f in fids}
    pos = {f: i for i, f in enumerate(fids)}
    keep = [f for f in fids if lab[f] and (coarse or lab[f] != "striker")]
    y = np.array([("S" if lab[f] in ("kick", "box", "striker") else "G") if coarse else lab[f] for f in keep])
    V = U[[pos[f] for f in keep]]
    S = V @ V.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :10]
    obs = np.mean(y[nn] == y[:, None])
    _, cnt = np.unique(y, return_counts=True)
    chance = float(np.sum(cnt / cnt.sum() * (cnt - 1) / (cnt.sum() - 1)))
    return round(float(obs / chance), 3)


def transfer(U, fids, q, fights):
    eligible, pos = set(fids), {f: i for i, f in enumerate(fids)}
    res = defaultdict(dict)
    for f in sorted(fights, key=lambda x: (x.date is None, x.date)):
        if not is_decided(f.method, f.winner_id) or f.red_fighter_id is None or f.blue_fighter_id is None:
            continue
        r, b, w = int(f.red_fighter_id), int(f.blue_fighter_id), int(f.winner_id)
        if r in eligible:
            res[b][r] = w == r
        if b in eligible:
            res[r][b] = w == b
    agree = defaultdict(lambda: [0, 0])
    for seen in res.values():
        it = sorted(seen.items())
        for i in range(len(it)):
            for j in range(i + 1, len(it)):
                rec = agree[(it[i][0], it[j][0])]
                rec[1] += 1
                rec[0] += it[i][1] == it[j][1]
    a = np.array([pos[x] for x, _ in agree]); b = np.array([pos[y] for _, y in agree])
    v = np.array(list(agree.values()), dtype=float)
    gap, mean = np.abs(q[a] - q[b]), (q[a] + q[b]) / 2
    cell = (np.digitize(gap, np.quantile(gap, np.linspace(0, 1, 11)[1:-1])) * 5
            + np.digitize(mean, np.quantile(mean, np.linspace(0, 1, 6)[1:-1])))
    cm = {c: v[cell == c, 0].sum() / v[cell == c, 1].sum() for c in np.unique(cell)}
    resid = v[:, 0] / v[:, 1] - np.array([cm[c] for c in cell])
    s = np.sum(U[a] * U[b], 1)
    top = s >= np.quantile(s, 0.9)
    return round(float(np.sum(resid[top] * v[top, 1]) / v[top, 1].sum()), 4)


def clusterboot(X, k, B=30, seed=0):
    """Hennig-style per-cluster stability: mean over bootstraps of max Jaccard."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    base = KMeans(k, n_init=10, random_state=0).fit(X).labels_
    J = np.zeros((B, k))
    for b in range(B):
        idx = np.unique(rng.integers(0, len(X), len(X)))
        lab = KMeans(k, n_init=10, random_state=b + 1).fit(X[idx]).labels_
        for c in range(k):
            C = set(idx[base[idx] == c])
            J[b, c] = max((len(C & set(idx[lab == c2])) / len(C | set(idx[lab == c2])) for c2 in range(k)), default=0)
    return J.mean(0)


def families(Z, inputs, bouts, model, space, m):
    """Z: the space before unit-normalising; the null is a Gaussian with its mean and covariance."""
    rng = np.random.default_rng(1)
    U = unit(Z)
    N = unit(rng.multivariate_normal(Z.mean(0), np.cov(Z.T), size=len(Z)))
    stab = {}
    for k in range(2, 9):
        jr, jn = clusterboot(U, k), clusterboot(N, k)
        stab[k] = {"real_min": round(float(jr.min()), 2), "real_median": round(float(np.median(jr)), 2),
                   "null_min": round(float(jn.min()), 2), "null_median": round(float(np.median(jn)), 2)}
        log.info(f"  k={k}: real {stab[k]['real_min']}/{stab[k]['real_median']}  null {stab[k]['null_min']}/{stab[k]['null_median']}")
    # Blend calibration: how often each fighter lands in each family when their own fights
    # are resampled, vs the softmax shares at several temperatures.
    freq = np.zeros((len(inputs.fids), sp.N_FAMILIES))
    B = 40
    for _ in range(B):
        rs = [[bouts[f][i] for i in rng.integers(0, len(bouts[f]), len(bouts[f]))] for f in inputs.fids]
        Ub = vectors(rs, inputs.fids, inputs, model, space)
        freq[np.arange(len(Ub)), sp.assign_families(Ub, m)] += 1
    freq /= B
    calib = {}
    for tau in (0.05, 0.10, 0.15, 0.20, 0.30):
        P = sp.blends(U, dict(m, blend_tau=tau))
        calib[tau] = round(float(np.mean(np.abs(P - freq))), 3)
    P = sp.blends(U, m)
    top = P.max(1)
    return {"stability": stab, "blend_gap_by_tau": calib,
            "share_clear_ge_0.8": round(float(np.mean(top >= 0.8)), 3),
            "share_hybrid_lt_0.6": round(float(np.mean(top < 0.6)), 3),
            "family_sizes": np.bincount(P.argmax(1), minlength=sp.N_FAMILIES).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", action="store_true", help="also run family stability + blend calibration")
    ap.add_argument("--out", help="write results JSON here")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db = SessionLocal()
    db.execute(text("SET TRANSACTION READ ONLY")) if db.get_bind().dialect.name == "postgresql" else None
    try:
        inputs, bouts, model, labels, fights = load(db)
    finally:
        db.rollback()
        db.close()
    space = ss.load_space()
    if space is None:
        raise SystemExit(f"No frozen style space {ss.SPACE_VERSION}")
    U = unit(ss.embed(inputs, space))
    q = quality(inputs)
    res = {"space": space["version"], "n_fighters": len(inputs.fids),
           "self_recognition": self_recognition(inputs, bouts, model, space),
           "quality_r2": round(quality_r2(ss.embed(inputs, space), q), 3),
           "label_lift_fine": label_lift(U, inputs.fids, labels, coarse=False),
           "label_lift_coarse": label_lift(U, inputs.fids, labels, coarse=True),
           "transfer_resid_lift": transfer(U, inputs.fids, q, fights)}
    if a.families:
        m = sp.load_map()
        res["map"] = m["version"]
        res["families"] = families(ss.embed(inputs, space), inputs, bouts, model, space, m)
    sr = res["self_recognition"]
    n = sr["n"]
    ok = sr["oddeven"]["held_out_top1"] >= 25.0 / n and sr["oddeven"]["held_out_median_rank"] <= 0.05 * n
    res["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)
    return res["verdict"]


if __name__ == "__main__":
    raise SystemExit(0 if main() == "PASS" else 1)
