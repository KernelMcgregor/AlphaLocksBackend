"""Fighter style similarity — "who fights like this guy?"

WHY THIS IS NOT JUST k-NN OVER THE GLICKO RATINGS
-------------------------------------------------
The obvious approach is to treat the 15 Glicko dimensions as an embedding and take
nearest neighbours. That returns the wrong answer, for two independent reasons:

1. Glicko dimensions are SKILL ratings, and their dominant axis of variation is overall
   quality — an elite fighter is above average in nearly every dimension at once. Raw
   nearest-neighbours in that space answers "who is roughly as good as X", not "who
   fights like X". Jon Jones's neighbours come out as other elites with no stylistic
   relationship to him.
2. Glicko measures SUCCESS, not PROPENSITY. A fighter who never shoots a takedown and a
   fighter who shoots ten a round and gets stuffed both land on a low `td` mu. Tendency
   is invisible to it.

So the vector here is built in two blocks:

  Block A — TENDENCY, from ufc_fighter_career_stats. Rates and shares only, never raw
            volume totals (which just re-encode career length). This is what a fighter
            *does*: where he throws, from what position, how often he shoots, whether he
            finishes.
  Block B — SHAPE, from the Glicko percentiles already published in
            ufc_fighter_rankings.feature_profile, MEAN-CENTRED PER FIGHTER. Subtracting a
            fighter's own mean across dimensions removes the overall-quality axis and
            leaves "relative to his own level, where is he over- and under-weighted".
            That residual is style, and unlike Block A it is opponent-adjusted — the one
            thing raw career stats can never be.

Block B calls `compute_dimension_profiles` — the same function that builds the radar
chart — rather than reading the published `ufc_fighter_rankings.feature_profile`. Reading
the table would be free, since `publish_rankings` runs immediately before this service in
`refresh_after_event`, but that table is deliberately ACTIVE-ONLY: `Eligibility` drops
anyone more than 548 days idle, which is correct for a ranking and fatal here. It would
mean the answer to "who fights like Khabib" is that Khabib does not exist. So this service
recomputes profiles under STYLE_CRIT, which keeps the sample-size floors and drops the
recency gate. The cost is one extra Glicko replay per event; the benefit is that the
retired fighters people most want comparables for are in the pool.

WHY THE SPACE IS FROZEN
-----------------------
This runs after every event. If the scalers were refit each run, the basis would
move under the data: every fighter's scores would drift for reasons unrelated to the
fights that just happened, and `previous_rank` — the whole point of which is to show what
an event changed — would be measuring the basis moving rather than the fighters moving.
So the space is fit once, persisted to models/ufc/style/, and subsequent runs only
transform through it. Refitting is an explicit `--refit`, treated as a versioned change
with style_eval rerun against it.

DISPLAY-ONLY. Every input is career-to-date, so a fighter's vector reflects fights that
had not happened at the time of any given historical bout. This is the same hazard class
as whr_ranker and must never reach model.build_features(); tests/test_leakage.py asserts
it. It is also why nothing here touches model.py — the picks feature set is frozen by
PREREGISTRATION.md and changing model inputs would force a versioned reset of the live
betting experiment.

MEASURED (style_eval, 2026-09-07, 1,515 eligible fighters, 53,844 common-opponent pairs)
-----------------------------------------------------------------------------------------
Outcome agreement against shared opposition, by similarity decile:

    decile   1     2     3     4     5     6     7     8     9    10   baseline
    agree  .491  .506  .512  .513  .513  .524  .526  .529  .534  .563   .522

    monotonicity (corr of decile vs agreement)  +0.928
    top-decile lift over baseline               +0.041       VERDICT: PASS

READ THAT +0.041 WITH CARE — it is partly a fighter-QUALITY artefact. Similar-style pairs
also tend to be similar-quality pairs (corr of style similarity with |quality gap| =
-0.172), and quality alone is the stronger predictor of outcome transfer:

    quality gap alone (trivial baseline)        +0.059
    style similarity, raw                       +0.041
    style similarity, WITHIN quality strata     +0.025   <- the honest number

The stratified figure is what this space actually contributes: hold overall quality fixed
and style still predicts transfer, consistently across all three tertiles. Small, but real
and independent. It should be small — outcomes are binary and most pairs share only one or
two opponents, so the metric's ceiling is low regardless.

For reference, the v1 space (PCA-10 whitened) scored +0.032 raw / +0.018 stratified on the
same data. The stratified pair is the like-for-like comparison; see the PCA note below.

NOT ADOPTED — PCA (was shipped, then removed 2026-09-07). The original design ran
PCA(n_components=10, whiten=True) on the reasoning that slpm/head_pm/dist_pct are
collinear and unwhitened distance triple-counts "strikes a lot". That reasoning is wrong
here. Held-out over 30 random pair splits, on the quality-controlled metric:

    shipped PCA-10 whitened   +0.0175 (sd .0043)
    PCA-20 whitened           +0.0199              beats PCA-10 in 63% of splits
    PCA-37 whitened (no trunc)+0.0206                                    73%
    no PCA (standardised)     +0.0246 (sd .0044)                         97%

Monotone in the amount of PCA applied, so both the truncation AND the whitening cost
accuracy. The collinearity is signal: several correlated measures of "distance striker"
reinforce a real trait, and whitening equalises them against low-variance directions that
are mostly noise. Truncation then discards ~30% of variance that was carrying information.

NOT ADOPTED — supervised per-feature weights (ridge on |feature gap| vs agreement). Best
train score of anything tried (+0.0346) and no held-out gain (+0.0209 vs +0.0215 for plain
cosine). Overfit.

NOT ADOPTED — projecting the quality direction out of the embedding. Actively harmful
(+0.0083 held-out): the quality axis carries real style information and removing it
destroys signal. Mean-centring Block B already removes as much as can be removed cheaply.

NOT ADOPTED — raising SHRINK_K (5/15/30) or the fight-count floor (3 -> 6). No consistent
effect in either direction; median neighbour fight-count stays 9-11 regardless. The
"low-sample fighters dominate the neighbour lists" hypothesis is not supported.

NOT ADOPTED: reading Block B from ufc_fighter_rankings.feature_profile instead of
recomputing it. Free, and it kept the panel consistent with the radar chart by
construction, but that table is active-only and the pool came out at 532 fighters with
every retired fighter missing — "who fights like Khabib" returned nothing. See STYLE_CRIT.

WHERE THE HEADROOM IS: the supervised run drove 15 of the 37 features to zero weight
(body_pct, leg_pct, ground_pct, td15s, sub_att15g, gnp15g, finish_rate and 8 Glicko dims
contributed nothing measurable to transfer). The informative subspace is much smaller than
the vector, so the ceiling is set by the FEATURE SET, not the geometry. Further gains need
new information — career-arc/temporal features, opponent-adjusted tendencies (Block A is
raw), round-level sequence data — not a richer model over these same columns.

Run: python -m app.services.ufc.style_service [--refit]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.database import SessionLocal
from app.models.ufc import UFCFighterCareerStats, UFCFighterSimilarity
from app.services.ufc.fighter_registry import Eligibility, build_fighter_registry, is_rankable
from app.services.ufc.glicko_service import DIMENSIONS
from app.services.ufc.ranking_service import _percentile_profile, compute_dimension_profiles

log = logging.getLogger("style_service")

#: Eligibility for the similarity pool. The sample-size floors are kept — a two-fight
#: fighter has no stable style and should not be anyone's comparable — but the 548-day
#: inactivity gate that Eligibility() applies is dropped. A ranking must exclude retired
#: fighters; a style comparison must not, or the feature cannot answer the questions it
#: exists for ("who is the next Khabib", "who fights like prime Silva"). This is the one
#: place in the codebase that deliberately diverges from the default criteria, and it
#: diverges on exactly one field.
STYLE_CRIT = Eligibility(min_decided_fights=3, min_rounds=10, max_days_inactive=10**6)

STYLE_DIR = Path(__file__).parent.parent.parent.parent / "models" / "ufc" / "style"

#: v2 dropped the PCA stage (see the module docstring). Bumped rather than overwritten so
#: a rollback is a one-line change and the v1 artifact stays on disk for comparison.
SPACE_VERSION = "v2"

#: Block A. Rates, accuracies and distribution shares drawn from ufc_fighter_career_stats.
#: Deliberately excludes every raw total (sig_str_landed, fight_count, total_fight_min):
#: those measure how long a career has been, not how it is fought, and they would put
#: veterans next to veterans. `finish_rate` and the two method shares are included because
#: how a fighter ENDS fights is stylistic information that no rate column carries.
TENDENCY_FEATURES = [
    # Striking volume, accuracy and defence
    "slpm", "sapm", "sig_acc", "sig_def",
    # Target distribution — head/body/leg selection is the single most legible style axis
    "head_pct", "body_pct", "leg_pct",
    # Position distribution — where the offence actually happens
    "dist_pct", "clinch_pct", "ground_pct",
    # Grappling: entries per standing minute, and what happens once it is on the mat
    "td15s", "td_acc", "td_def", "ctrl15g", "sub_att15g", "gnp15g",
    # Damage and scrambles
    "kd15s", "rev15",
    # Outcome shape
    "finish_rate", "avg_fight_sec",
]

#: Derived on the fly from counts, since the method mix is a share not a rate.
DERIVED_FEATURES = ["ko_win_share", "sub_win_share"]

BLOCK_A_FEATURES = TENDENCY_FEATURES + DERIVED_FEATURES
BLOCK_B_FEATURES = [f"glicko_{d}" for d in DIMENSIONS]
ALL_FEATURES = BLOCK_A_FEATURES + BLOCK_B_FEATURES

#: Shrinkage strength. A fighter's tendency vector is a small-sample estimate: three fights
#: of leg kicks is not a leg-kicker. Each Block A feature is pulled toward its division
#: mean with weight n/(n+K); K=5 means a 5-fight fighter sits halfway between his own
#: numbers and his division's, and a 20-fight fighter is 80% his own. Chosen to be near
#: the Eligibility floor of 3 decided fights — smaller values let 3-fight fighters
#: dominate the extremes of every axis, which is exactly the failure this prevents.
SHRINK_K = 5.0

#: Relative weight of Block B against Block A after each is standardised. 1.0 = the
#: opponent-adjusted shape and the raw tendency contribute equally. Tune with style_eval.
BLOCK_B_WEIGHT = 1.0

#: Neighbours stored per fighter.
TOP_K = 20

#: Drivers reported per pair.
N_DRIVERS = 3

#: A frozen space fit this long ago is warned about — the sport drifts, and at some point
#: the basis stops describing the current roster.
SPACE_STALE_DAYS = 400


def _space_path(version: str = SPACE_VERSION) -> Path:
    return STYLE_DIR / f"style_space_{version}.pkl"


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def _load_rows(db):
    """Return {fighter_id: {feature: raw_value}}, division map, and fight counts.

    Eligibility and division come from fighter_registry, never re-derived here. The
    header of that module documents what happened the last time two services each worked
    out divisions for themselves.
    """
    registry = build_fighter_registry(db)
    today = date.today()
    crit = STYLE_CRIT

    # Block B: Glicko dimension percentiles under the relaxed criteria, so retired
    # fighters are in the pool. Same function the radar chart uses, so the two views of
    # a fighter's profile cannot drift apart.
    profiles = compute_dimension_profiles(db, registry, today, crit)
    glicko: dict[int, dict[str, float]] = {}
    for (fid, _division), profile in profiles.items():
        vals = {f"glicko_{d}": profile[d] for d in DIMENSIONS if d in profile}
        if len(vals) == len(DIMENSIONS):
            glicko[int(fid)] = vals

    raw: dict[int, dict[str, float]] = {}
    divisions: dict[int, str] = {}
    fight_counts: dict[int, int] = {}

    for cs in db.query(UFCFighterCareerStats).all():
        fid = int(cs.fighter_id)
        state = registry.get(fid)
        if state is None or not is_rankable(state, today, crit):
            continue
        if fid not in glicko:
            # No published ranking means no opponent-adjusted half of the vector.
            continue

        n = cs.fight_count or 0
        if n <= 0:
            continue

        feats: dict[str, float] = {}
        missing = False
        for col in TENDENCY_FEATURES:
            v = getattr(cs, col, None)
            if v is None:
                missing = True
                break
            feats[col] = float(v)
        if missing:
            continue

        feats["ko_win_share"] = (cs.ko_wins or 0) / n
        feats["sub_win_share"] = (cs.sub_wins or 0) / n
        feats.update(glicko[fid])

        raw[fid] = feats
        divisions[fid] = state.division
        fight_counts[fid] = n

    log.info(f"  {len(raw)} eligible fighters with both career stats and a published profile")
    return raw, divisions, fight_counts


def _shrink_block_a(raw, divisions, fight_counts) -> None:
    """Pull each Block A feature toward its division mean by n/(n+SHRINK_K). In place."""
    by_div: dict[str, list[int]] = defaultdict(list)
    for fid, div in divisions.items():
        by_div[div].append(fid)

    for div, fids in by_div.items():
        for col in BLOCK_A_FEATURES:
            vals = [raw[f][col] for f in fids]
            mean = sum(vals) / len(vals)
            for f in fids:
                n = fight_counts[f]
                w = n / (n + SHRINK_K)
                raw[f][col] = w * raw[f][col] + (1.0 - w) * mean


def _percentile_block_a(raw, divisions) -> None:
    """Convert Block A to within-division ordinal percentiles. In place.

    Without this, "similar style" collapses into "similar weight class": heavyweights
    throw fewer strikes per minute than flyweights, so raw rates cluster by division and
    the cross-division comparison the feature exists to make becomes impossible. Reuses
    ranking_service._percentile_profile so ties are handled the same way the radar chart
    handles them.

    Block B is already percentile-valued — compute_dimension_profiles emits it that way.
    """
    by_div: dict[str, list[int]] = defaultdict(list)
    for fid, div in divisions.items():
        by_div[div].append(fid)

    for div, fids in by_div.items():
        for col in BLOCK_A_FEATURES:
            pct = _percentile_profile({f: raw[f][col] for f in fids})
            for f, v in pct.items():
                raw[f][col] = v


def _centre_block_b(raw) -> None:
    """Subtract each fighter's own mean across the Glicko dimensions. In place.

    This is the step that turns a skill rating into a style descriptor. Before it, the
    vector says "he is good"; after it, "he is good AT THESE THINGS relative to how good
    he is overall" — which is what makes an elite and a journeyman with the same
    tendencies come out as neighbours.
    """
    for feats in raw.values():
        vals = [feats[c] for c in BLOCK_B_FEATURES]
        mean = sum(vals) / len(vals)
        for c in BLOCK_B_FEATURES:
            feats[c] -= mean


def build_matrix(db):
    """Assemble the raw feature matrix.

    Returns (fighter_ids, X, divisions) with X columns ordered as ALL_FEATURES.
    """
    raw, divisions, fight_counts = _load_rows(db)
    if not raw:
        return [], np.empty((0, len(ALL_FEATURES))), {}

    _shrink_block_a(raw, divisions, fight_counts)
    _percentile_block_a(raw, divisions)
    _centre_block_b(raw)

    fids = sorted(raw)
    X = np.array([[raw[f][c] for c in ALL_FEATURES] for f in fids], dtype=float)
    return fids, X, divisions


# ---------------------------------------------------------------------------
# The frozen space
# ---------------------------------------------------------------------------

def fit_space(X: np.ndarray) -> dict:
    """Fit the per-block scalers and return the artifact dict.

    There is no dimension reduction here, and that is a measured choice rather than an
    omission — see the NOT ADOPTED note in the module docstring.
    """
    n_a = len(BLOCK_A_FEATURES)
    scaler_a = StandardScaler().fit(X[:, :n_a])
    scaler_b = StandardScaler().fit(X[:, n_a:])

    log.info(
        f"  Fit style space on {X.shape[0]} fighters x {X.shape[1]} features "
        f"({n_a} tendency + {X.shape[1] - n_a} glicko, block B weight {BLOCK_B_WEIGHT})"
    )

    return {
        "version": SPACE_VERSION,
        "features": list(ALL_FEATURES),
        "n_block_a": n_a,
        "scaler_a": scaler_a,
        "scaler_b": scaler_b,
        "block_b_weight": BLOCK_B_WEIGHT,
        "shrink_k": SHRINK_K,
        "fitted_on": datetime.now(timezone.utc).isoformat(),
        "n_fit_rows": int(X.shape[0]),
    }


def _apply_scalers(X: np.ndarray, scaler_a, scaler_b, weight: float = BLOCK_B_WEIGHT):
    n_a = scaler_a.n_features_in_
    Za = scaler_a.transform(X[:, :n_a])
    Zb = scaler_b.transform(X[:, n_a:]) * weight
    return np.hstack([Za, Zb])


def save_space(space: dict) -> Path:
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    path = _space_path(space["version"])
    with open(path, "wb") as f:
        pickle.dump(space, f)
    log.info(f"  Saved style space to {path}")
    return path


def load_space() -> dict | None:
    path = _space_path()
    if not path.exists():
        return None
    with open(path, "rb") as f:
        space = pickle.load(f)

    if space.get("features") != list(ALL_FEATURES):
        log.warning(
            "  Frozen style space was fit on a different feature list; refusing to use it. "
            "Rerun with --refit (and rerun style_eval against the new space)."
        )
        return None

    try:
        fitted = datetime.fromisoformat(space["fitted_on"])
        age = (datetime.now(timezone.utc) - fitted).days
        if age > SPACE_STALE_DAYS:
            log.warning(
                f"  Style space was fit {age} days ago. The roster has turned over since; "
                f"consider --refit."
            )
    except (KeyError, ValueError):
        pass
    return space


def transform(X: np.ndarray, space: dict) -> np.ndarray:
    return _apply_scalers(X, space["scaler_a"], space["scaler_b"], space["block_b_weight"])


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _drivers(Z: np.ndarray, i: int, j: int) -> list[dict]:
    """The traits two fighters most strongly share.

    Scored on the standardised features, which since v2 are also the retrieval space, so
    the reported reason and the computed distance can no longer disagree. (Under v1 these
    were the pre-PCA features while retrieval ran on the components — nameable, but a step
    removed from what actually produced the match.) Requires the same sign — two fighters
    both far BELOW average on leg kicks is a real shared trait, two on opposite sides is
    not.
    """
    a, b = Z[i], Z[j]
    scored = []
    for k, name in enumerate(ALL_FEATURES):
        if a[k] * b[k] <= 0:
            continue
        scored.append((name, float(np.sign(a[k]) * min(abs(a[k]), abs(b[k])))))
    scored.sort(key=lambda kv: -abs(kv[1]))
    return [{"feature": n, "z": round(z, 2)} for n, z in scored[:N_DRIVERS]]


def compute_similarity(db, refit: bool = False):
    """Build the top-K neighbour table. Returns list of dicts ready to persist."""
    log.info("Building fighter style similarity...")
    fids, X, divisions = build_matrix(db)
    if len(fids) < TOP_K + 1:
        log.warning(f"  Only {len(fids)} eligible fighters; nothing to do")
        return []

    space = None if refit else load_space()
    if space is None:
        log.info("  Fitting a new style space" + (" (--refit)" if refit else " (none on disk)"))
        space = fit_space(X)
        save_space(space)
    else:
        log.info(f"  Using frozen style space fit {space['fitted_on']} on {space['n_fit_rows']} rows")

    # Retrieval runs directly on the standardised features — no projection step, so the
    # embedding and the space the drivers are read from are one and the same.
    Z = _apply_scalers(X, space["scaler_a"], space["scaler_b"], space["block_b_weight"])

    k = min(TOP_K + 1, len(fids))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(Z)
    dist, idx = nn.kneighbors(Z)

    out = []
    for i, fid in enumerate(fids):
        rank = 0
        for d, j in zip(dist[i], idx[i]):
            if j == i:
                continue
            rank += 1
            if rank > TOP_K:
                break
            out.append({
                "fighter_id": fid,
                "similar_fighter_id": fids[j],
                "rank": rank,
                # cosine distance -> similarity. Clipped because the embedding is
                # mean-centred and can go negative; a negative "similarity" is not
                # something the UI can render meaningfully.
                "similarity": round(max(0.0, 1.0 - float(d)), 4),
                "same_division": divisions.get(fid) == divisions.get(fids[j]),
                "top_drivers": _drivers(Z, i, j),
            })

    log.info(f"  {len(out)} neighbour rows for {len(fids)} fighters")
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def compute_and_save_similarity(db=None, refit: bool = False) -> int:
    """Recompute the similarity table, preserving prior ranks for churn reporting."""
    owns_db = db is None
    db = db or SessionLocal()
    try:
        rows = compute_similarity(db, refit=refit)
        if not rows:
            return 0

        # Read prior ranks BEFORE the delete so `previous_rank` survives the rewrite.
        # This is what makes "whose comparables did this event change" answerable at all.
        prior = {
            (int(r.fighter_id), int(r.similar_fighter_id)): r.rank
            for r in db.query(UFCFighterSimilarity).all()
        }

        now = datetime.now(timezone.utc)
        objs = []
        new_entrants = defaultdict(int)
        for r in rows:
            key = (r["fighter_id"], r["similar_fighter_id"])
            prev = prior.get(key)
            if prior and prev is None:
                new_entrants[r["fighter_id"]] += 1
            objs.append(UFCFighterSimilarity(
                fighter_id=r["fighter_id"],
                similar_fighter_id=r["similar_fighter_id"],
                rank=r["rank"],
                similarity=r["similarity"],
                same_division=r["same_division"],
                top_drivers=json.dumps(r["top_drivers"]),
                previous_rank=prev,
                computed_at=now,
            ))

        db.query(UFCFighterSimilarity).delete()
        db.commit()
        db.bulk_save_objects(objs)
        db.commit()

        if prior:
            log.info(
                f"  {len(new_entrants)} fighters had comparables change "
                f"({sum(new_entrants.values())} new entries in top-{TOP_K})"
            )
        log.info(f"  Saved {len(objs)} similarity rows")
        return len(objs)
    finally:
        if owns_db:
            db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fighter style similarity")
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Refit and overwrite the frozen style space instead of transforming through it. "
             "Changes every stored score; rerun style_eval afterwards.",
    )
    args = parser.parse_args()
    n = compute_and_save_similarity(refit=args.refit)
    log.info(f"Done — {n} rows")
