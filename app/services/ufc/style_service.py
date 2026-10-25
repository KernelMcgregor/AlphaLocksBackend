"""Fighter style similarity — "who fights like this guy?"

SPACE v5 (2026-10-18): HABITS ONLY
----------------------------------
The question this answers is "who fights like X" — the choices a fighter makes, not how
good they are. The vector is 15 opponent-adjusted habits from `style_habits.style_features`
in 8 equally weighted concept blocks: pace, target mix, position mix, takedown attempts,
submission attempts, non-significant strike share, control (own and controlled), ground
strikes per control minute. Mixes are centred log-ratios; rare events are floored logs.

Each column is z-scored within the fighter's division (heavyweights throw less than
flyweights), a linear era trend in fight-weighted prime year is removed, the residual is
standardised and each block scaled to equal weight. Cosine similarity, top 20.

Efficacy (accuracy, defence, finishes) and the Glicko skill ratings are NOT in the vector;
they are percentiles on screen only (`profiles`).

MEASURED 2026-10-18 (app/services/ufc/style_eval.py; self-recognition = find a fighter's other half
of fights among 1,284; reported on the held-out half of fighters):

                     odd/even fights    early/late career   quality R^2   transfer
    v4 (shipped)        11.8% / 35          6.4% / 66          0.248       +0.016
    v5                  15.8% / 24          8.8% / 42          0.169       +0.021

(found first / median rank.) Paired bootstrap: v5 better on both splits, CIs exclude 0.
"quality R^2" is how well Glicko quality can be predicted from the vector — lower means
less skill leaking into style. Cosine beat Euclidean; whitening raised self-recognition but
lowered label agreement and transfer (it finds what is unique, not what is shared).

WHY THE SPACE IS FROZEN
-----------------------
This runs after every event. If the era fit and scalers were refit each run, the basis
would move under the data and `previous_rank` — which exists to show what an event
changed — would measure the basis moving. So they are fit once (`--refit`), persisted to
models/ufc/style/, and later runs only transform through them. A missing file is an error,
never a cue to refit. The opponent model is NOT frozen: like Glicko it is a rating of the
data and is refit every run.

DISPLAY-ONLY. Every input is career-to-date, so a fighter's vector reflects fights that had
not happened at the time of any given historical bout. Same hazard class as whr_ranker; it
must never reach model.build_features() (tests/test_leakage.py, test_style_similarity.py).

HISTORY
  v1 PCA-10 whitened; v2 standardised; v3 attempts instead of landed; v4 opponent-adjusted
  habits + half-weight efficacy, percentiles, cubic era. v5 above.

Run: python -m app.services.ufc.style_service [--refit]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text

from app.database import SessionLocal
from app.models.ufc import UFCFighterSimilarity
from app.services.ufc import style_habits
from app.services.ufc.fighter_registry import Eligibility, build_fighter_registry, is_rankable
from app.services.ufc.glicko_service import DIMENSIONS
from app.services.ufc.ranking_service import _percentile_profile, compute_dimension_profiles

log = logging.getLogger("style_service")

#: Style eligibility: the default sample-size floors (3 decided fights, 10 rounds) but no
#: inactivity gate. A ranking must drop retired fighters; a style comparison must not, or
#: "who fights like Khabib" has no Khabib. The one place that diverges from Eligibility().
STYLE_CRIT = Eligibility(min_decided_fights=3, min_rounds=10, max_days_inactive=10**6)

STYLE_DIR = Path(__file__).parent.parent.parent.parent / "models" / "ufc" / "style"

#: Bumped, never overwritten, so a rollback is one line and older artifacts stay for comparison.
SPACE_VERSION = "v5"

#: The retrieval space's columns and their concept blocks (see style_habits).
ALL_FEATURES = list(style_habits.STYLE_FEATURES)
FEATURE_BLOCKS = style_habits.FEATURE_BLOCKS

#: Shown, never matched on: how it goes, and the Glicko skill ratings.
EFFICACY_FEATURES = [
    "sig_acc", "td_acc", "sig_def", "td_def", "sapm", "kd15s",
    "finish_rate", "ko_win_share", "sub_win_share", "avg_fight_sec",
]
SKILL_FEATURES = [f"glicko_{d}" for d in DIMENSIONS]
#: Every percentile column a profile carries.
PROFILE_FEATURES = ALL_FEATURES + EFFICACY_FEATURES + SKILL_FEATURES

#: Neighbours stored per fighter, and shared traits reported per pair.
TOP_K = 20
N_DRIVERS = 3

#: A frozen space fit this long ago is warned about — the sport drifts.
SPACE_STALE_DAYS = 400


def _space_path(version: str = SPACE_VERSION) -> Path:
    return STYLE_DIR / f"style_space_{version}.pkl"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class StyleInputs:
    fids: list[int]
    X: np.ndarray                      # ALL_FEATURES, raw (logs / log-ratios), one row per fid
    divisions: dict[int, str]
    prime: np.ndarray                  # fight-weighted mean date (decimal year) per fid
    years: dict[int, tuple[int, int]]  # first, last UFC year
    profiles: dict[int, dict[str, float]]  # readable 0-99 percentiles: PROFILE_FEATURES
    #: First UFC fight (decimal year) per fid: the lineage's time axis and ancestor rule.
    debut: np.ndarray = field(default_factory=lambda: np.empty(0))


def fight_weighted_prime(dates) -> float:
    """Mean fight date as a decimal year — where a career's weight actually sits."""
    return float(np.mean([d.year + (d.timetuple().tm_yday - 1) / 366.0 for d in dates]))


def _load_rows(db):
    """Eligible fighters' style features, display values, divisions, Glicko, years, primes.

    Eligibility and division come from fighter_registry, never re-derived here. Glicko is
    replayed under STYLE_CRIT so retired fighters have skill profiles, WITHOUT rewriting
    ufc_glicko_snapshots — publish_rankings already wrote them, and the winner model
    serves from that table.
    """
    registry = build_fighter_registry(db)
    today = date.today()
    profiles = compute_dimension_profiles(db, registry, today, STYLE_CRIT, persist_snapshots=False)
    glicko: dict[int, dict[str, float]] = {}
    for (fid, _division), profile in profiles.items():
        vals = {f"glicko_{d}": profile[d] for d in DIMENSIONS if d in profile}
        if len(vals) == len(DIMENSIONS):
            glicko[int(fid)] = vals

    bouts = style_habits.load_bouts(db)
    model = style_habits.fit_opponent_model(bouts)

    feats, shown = {}, {}
    divisions: dict[int, str] = {}
    years: dict[int, tuple[int, int]] = {}
    primes: dict[int, float] = {}
    debuts: dict[int, float] = {}
    for fid, bs in bouts.items():
        state = registry.get(fid)
        if state is None or not is_rankable(state, today, STYLE_CRIT) or fid not in glicko:
            continue
        dated = [b.date for b in bs if b.date is not None]
        if not dated:
            continue
        feats[fid] = style_habits.style_features(bs, model)
        shown[fid] = {**style_habits.display_rates(bs, model), **style_habits.efficacy_features(bs)}
        divisions[fid] = state.division
        years[fid] = (min(dated).year, max(dated).year)
        primes[fid] = fight_weighted_prime(dated)
        debuts[fid] = fight_weighted_prime([min(dated)])

    log.info(f"  {len(feats)} eligible fighters with bouts and a skill profile")
    return feats, shown, divisions, years, primes, debuts, glicko


def _percentile_within_division(raw, divisions, cols) -> None:
    """Each column -> ordinal percentile within the fighter's division. In place. Display only."""
    by_div: dict[str, list[int]] = defaultdict(list)
    for fid, div in divisions.items():
        by_div[div].append(fid)
    for fids in by_div.values():
        for col in cols:
            pct = _percentile_profile({f: raw[f][col] for f in fids})
            for f, v in pct.items():
                raw[f][col] = v


def build_inputs(db) -> StyleInputs:
    feats, shown, divisions, years, primes, debuts, glicko = _load_rows(db)
    if not feats:
        return StyleInputs([], np.empty((0, len(ALL_FEATURES))), {}, np.empty(0), {}, {})
    fids = sorted(feats)
    X = np.array([[feats[f][c] for c in ALL_FEATURES] for f in fids], dtype=float)
    prime = np.array([primes[f] for f in fids], dtype=float)
    _percentile_within_division(shown, divisions, ALL_FEATURES + EFFICACY_FEATURES)
    profiles = {f: {**{c: shown[f][c] for c in ALL_FEATURES + EFFICACY_FEATURES}, **glicko[f]} for f in fids}
    debut = np.array([debuts[f] for f in fids], dtype=float)
    return StyleInputs(fids, X, divisions, prime, years, profiles, debut)


# ---------------------------------------------------------------------------
# Space
# ---------------------------------------------------------------------------

def _era_basis(prime: np.ndarray) -> np.ndarray:
    t = (np.asarray(prime, dtype=float) - 2015.0) / 10.0
    return np.vstack([np.ones_like(t), t]).T


def _block_weights() -> np.ndarray:
    """Equal weight per concept: a block's columns are scaled by 1/sqrt(its degrees of freedom)."""
    w = np.ones(len(ALL_FEATURES))
    for name, cols in FEATURE_BLOCKS.items():
        dof = style_habits.BLOCK_DOF.get(name, len(cols))
        for c in cols:
            w[ALL_FEATURES.index(c)] = 1.0 / np.sqrt(dof)
    return w


def _division_z(X: np.ndarray, divisions: list[str], stats: dict) -> np.ndarray:
    Z = np.empty_like(X, dtype=float)
    for i, d in enumerate(divisions):
        mean, sd = stats.get(d, stats["_all"])
        Z[i] = (X[i] - mean) / sd
    return Z


def fit_space(X: np.ndarray, prime: np.ndarray, divisions: list[str]) -> dict:
    """Fit per-division scaling, the era trend and the residual scale; return the artifact."""
    divisions = list(divisions)
    stats = {"_all": (X.mean(0), X.std(0) + 1e-9)}
    for d in set(divisions):
        m = np.array([x == d for x in divisions])
        stats[d] = (X[m].mean(0), X[m].std(0) + 1e-9)
    Zd = _division_z(X, divisions, stats)
    beta, *_ = np.linalg.lstsq(_era_basis(prime), Zd, rcond=None)
    sd = (Zd - _era_basis(prime) @ beta).std(0) + 1e-9
    log.info(f"  Fit style space {SPACE_VERSION} on {X.shape[0]} fighters x {X.shape[1]} features")
    return {
        "version": SPACE_VERSION,
        "features": list(ALL_FEATURES),
        "division_stats": stats,
        "era_beta": beta,
        "resid_sd": sd,
        "block_weights": _block_weights(),
        "lambda_fights": style_habits.LAMBDA_FIGHTS,
        "rare_floor": style_habits.RARE_FLOOR,
        "fitted_on": datetime.now(timezone.utc).isoformat(),
        "n_fit_rows": int(X.shape[0]),
    }


def transform(X: np.ndarray, prime: np.ndarray, divisions: list[str], space: dict) -> np.ndarray:
    """Raw features -> the retrieval space (division-scaled, era-detrended, block-weighted)."""
    Zd = _division_z(X, list(divisions), space["division_stats"])
    return (Zd - _era_basis(prime) @ space["era_beta"]) / space["resid_sd"] * space["block_weights"]


def embed(inputs: "StyleInputs", space: dict) -> np.ndarray:
    return transform(inputs.X, inputs.prime, [inputs.divisions[f] for f in inputs.fids], space)


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
        log.warning("  Frozen style space was fit on a different feature list; refusing to use it. "
                    "Rerun with --refit (and rerun style_eval against the new space).")
        return None
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(space["fitted_on"])).days
        if age > SPACE_STALE_DAYS:
            log.warning(f"  Style space was fit {age} days ago; consider --refit.")
    except (KeyError, ValueError):
        pass
    return space


def space_for(inputs: StyleInputs, refit: bool = False) -> dict:
    """The frozen space; a new one is fitted (and saved) only on an explicit --refit.

    A missing pkl is an error, not a cue to fit: on a CI runner the fitted file lands on
    throwaway disk, so every event would silently refit and move the basis."""
    space = None if refit else load_space()
    if space is None:
        if not refit:
            raise RuntimeError(f"No frozen style space {SPACE_VERSION} at {_space_path()}; "
                               f"commit it, or run style_service --refit deliberately")
        log.info("  Fitting a new style space (--refit)")
        space = fit_space(inputs.X, inputs.prime, [inputs.divisions[f] for f in inputs.fids])
        save_space(space)
    else:
        log.info(f"  Using frozen style space {space['version']} fit {space['fitted_on']} on {space['n_fit_rows']} rows")
    return space


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _drivers(Z: np.ndarray, i: int, j: int) -> list[dict]:
    """The traits two fighters most strongly share, scored in the retrieval space itself, so
    the stated reason and the computed similarity cannot disagree. Same sign only — both
    far below average on leg kicks is a real shared trait, opposite sides is not."""
    a, b = Z[i], Z[j]
    scored = []
    for k, name in enumerate(ALL_FEATURES):
        if a[k] * b[k] <= 0:
            continue
        scored.append((name, float(np.sign(a[k]) * min(abs(a[k]), abs(b[k])))))
    scored.sort(key=lambda kv: -abs(kv[1]))
    return [{"feature": n, "z": round(z, 2)} for n, z in scored[:N_DRIVERS]]


def compute_similarity(inputs: StyleInputs, space: dict) -> list[dict]:
    """The top-K neighbour table as dicts ready to persist."""
    fids = inputs.fids
    if len(fids) < TOP_K + 1:
        log.warning(f"  Only {len(fids)} eligible fighters; nothing to do")
        return []
    Z = embed(inputs, space)
    nn = NearestNeighbors(n_neighbors=min(TOP_K + 1, len(fids)), metric="cosine", algorithm="brute").fit(Z)
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
                # Cosine distance -> similarity, clipped: the space is centred and can go
                # negative, which the UI cannot render meaningfully.
                "similarity": round(max(0.0, 1.0 - float(d)), 4),
                "same_division": inputs.divisions.get(fid) == inputs.divisions.get(fids[j]),
                "top_drivers": _drivers(Z, i, j),
            })
    log.info(f"  {len(out)} neighbour rows for {len(fids)} fighters")
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

#: Key for the Postgres advisory lock shared by every writer of the style tables
#: (post-event.yml and the Railway chain can overlap).
STYLE_WRITE_LOCK = 0x5717E


def _stored_map_version(db) -> str | None:
    from app.models.ufc import UFCStyleFamily
    row = db.query(UFCStyleFamily.map_version).first()
    return row[0] if row else None


def _current_map_version() -> str:
    from app.services.ufc.style_page_service import MAP_VERSION
    return MAP_VERSION


def _try_write_lock(db) -> bool:
    """Take a transaction-scoped advisory lock on Postgres; other dialects always succeed."""
    if db.get_bind().dialect.name != "postgresql":
        return True
    return bool(db.execute(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": STYLE_WRITE_LOCK}).scalar())


def compute_and_save_similarity(db=None, refit: bool = False) -> int:
    """Recompute the similarity table (and the Fighter Styles page), preserving prior ranks."""
    owns_db = db is None
    db = db or SessionLocal()
    try:
        log.info("Building fighter style similarity...")
        inputs = build_inputs(db)
        if not inputs.fids:
            return 0
        space = space_for(inputs, refit=refit)
        rows = compute_similarity(inputs, space)
        if not rows:
            return 0

        # Build the page before writing anything, so both tables are replaced in one
        # transaction. A page failure still never costs the similarity table.
        page = None
        try:
            from app.services.ufc.style_page_service import build_style_page
            page = build_style_page(db, inputs, space)
        except Exception:
            db.rollback()   # clear any failed read so the write below starts clean
            log.exception("  Style page step failed; similarity rows will still be saved")

        if not _try_write_lock(db):
            log.warning("  Another style write holds the lock; skipping this run")
            return 0
        # Read prior ranks inside the lock so `previous_rank` survives the rewrite — but only
        # from the same space: across a version change, rank moves are the basis moving.
        # The stored map version identifies it (each map is fit on exactly one space).
        prior = {}
        if _stored_map_version(db) == _current_map_version():
            prior = {(int(r.fighter_id), int(r.similar_fighter_id)): r.rank
                     for r in db.query(UFCFighterSimilarity).all()}
        else:
            log.info("  Style space/map version changed: previous_rank starts fresh")
        now = datetime.now(timezone.utc)
        objs = []
        new_entrants = defaultdict(int)
        for r in rows:
            prev = prior.get((r["fighter_id"], r["similar_fighter_id"]))
            if prior and prev is None:
                new_entrants[r["fighter_id"]] += 1
            objs.append(UFCFighterSimilarity(
                fighter_id=r["fighter_id"], similar_fighter_id=r["similar_fighter_id"],
                rank=r["rank"], similarity=r["similarity"], same_division=r["same_division"],
                top_drivers=json.dumps(r["top_drivers"]), previous_rank=prev, computed_at=now,
            ))
        # One transaction: readers see the old rows or the new ones, never an empty table,
        # and a failed insert rolls back to the old rows.
        try:
            db.query(UFCFighterSimilarity).delete()
            db.bulk_save_objects(objs)
            if page is not None:
                from app.services.ufc.style_page_service import write_style_page
                write_style_page(db, page, now)
            db.commit()
        except Exception:
            db.rollback()
            raise
        if prior:
            log.info(f"  {len(new_entrants)} fighters had comparables change "
                     f"({sum(new_entrants.values())} new entries in top-{TOP_K})")
        log.info(f"  Saved {len(objs)} similarity rows"
                 + (f", {len(page['fighters'])} style rows" if page is not None else ""))
        return len(objs)
    finally:
        if owns_db:
            db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fighter style similarity")
    parser.add_argument("--refit", action="store_true",
                        help="Refit and overwrite the frozen style space. Changes every stored score; "
                             "rerun style_eval afterwards.")
    args = parser.parse_args()
    n = compute_and_save_similarity(refit=args.refit)
    log.info(f"Done — {n} rows")
