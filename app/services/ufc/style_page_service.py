"""Fighter Styles page data: four style families with blends, a frozen 2D map, lineage.

Reads the same inputs and frozen space as `style_service` (one Glicko replay and one
opponent-model fit serve both) and writes `ufc_fighter_style` + `ufc_style_families`.
Display-only, like the similarity table. One fingerprint drives the comps, the map, the
families and the lineage.

FOUR FAMILIES, AND BLENDS INSTEAD OF MORE BOXES
-----------------------------------------------
Measured 2026-10-18 on the v5 space (app/services/ufc/style_eval.py --families): per-family
bootstrap Jaccard (Hennig clusterboot; >= 0.75 stable) against the same pipeline on a
Gaussian with the data's covariance — a cloud with no real groups:

    KMeans k   real min / median Jaccard   null min / median
        3            0.86 / 0.86              0.64 / 0.76
        4            0.87 / 0.90              0.61 / 0.63   <- used
        5            0.63 / 0.67              0.67 / 0.75
        8            0.34 / 0.53              0.36 / 0.44

Four families are real; from five up the data is no more clustered than the null, and no
sub-split inside any of the four beats its own null. Style past that point is a continuum,
so finer detail comes from each fighter's BLEND — a softmax over cosine similarity to the
four centroids, with its temperature calibrated so the shares match how often the fighter
lands in each family when their own fights are resampled (mean abs gap 0.046). 65% of
fighters are >= 80% one family; 12% are genuine hybrids (< 60%).

Family ids are fixed by ROLE, not KMeans label order, so names and colours survive refits:
the centroid with the most takedown attempts is Wrestle-control, then most ground strikes
per control minute is Striker-finishers, then lowest pace is Patient, and the last is Range.

FROZEN LAYOUT
-------------
`style_map_<MAP_VERSION>.pkl` freezes the KMeans model, the role order, the blend
temperature and a cosine t-SNE of the anchor fighters. Each run places a fighter at a
softmax-weighted mean of their closest anchors' positions, so a fighter moves only when
their own vector moves. Refitting is `--refit-map` with a new MAP_VERSION; a missing map is
an error, never a cue to refit.

Run:
    python -m app.services.ufc.style_page_service --dry-run out.json   # no DB writes
    python -m app.services.ufc.style_page_service                      # write tables
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from datetime import date, datetime, timezone

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from app.database import SessionLocal
from app.models.ufc import UFCFighterStyle, UFCStyleFamily
from app.services.ufc import style_service as ss
from app.services.ufc.fighter_registry import Eligibility, build_fighter_registry, is_rankable

log = logging.getLogger("style_page")

MAP_VERSION = "v4"
N_FAMILIES = 4
#: Blend temperature on cosine similarity to the centroids. Calibrated 2026-10-18 against
#: fight-bootstrap assignment frequencies (tau 0.05 / 0.10 / 0.15 / 0.20 / 0.30: mean abs
#: gap 0.067 / 0.050 / 0.046 / 0.055 / 0.086 — app/services/ufc/style_eval.py --families).
BLEND_TAU = 0.15
#: Blend entries below this share are dropped from the stored blend.
BLEND_MIN_SHARE = 0.10
#: An ancestor's prime must be at least this many years before the fighter's own.
#: An ancestor must have debuted in the UFC at least this many years before the fighter.
#: Debut, not "prime": one fixed, explainable date, and you can only be influenced by
#: people who were already fighting when you started. (Was a 6-year gap on prime.)
LINEAGE_GAP = 4.0
N_ANCESTORS = 3
#: "Notable" = top quarter of the eligible pool on overall quality.
NOTABLE_QUANTILE = 0.75
#: Placement: softmax over cosine similarity to the ANCHOR_K closest anchors.
ANCHOR_K = 10
ANCHOR_TAU = 0.05
#: Centroid traits reported per family.
N_TRAITS = 3

#: Family id -> (name, blurb). Ids are roles (see `role_order`), so these hold across refits
#: of the same space. Blurbs quote the v5 family medians.
FAMILIES: list[tuple[str, str]] = [
    ("Range strikers",
     "High-volume striking from range: about 8 strikes thrown a minute, nearly 90% of them at "
     "distance. Rarely shoot and rarely go to the ground; half their wins go to the cards."),
    ("Striker-finishers",
     "Strike to finish: the fewest takedown attempts of any family, and when it hits the floor "
     "they follow it down with the most ground strikes per minute on top. Almost half their "
     "wins are knockouts."),
    ("Patient tacticians",
     "Low output, about 5 strikes a minute, with the most clinch work and short strikes and "
     "grappling exchanges on both sides. The highest share of submission wins."),
    ("Wrestle-control",
     "Shoot often, about 12 takedown attempts per 15 standing minutes, then hold position — a "
     "quarter of fight time in control — and strike from top."),
]
RANGE, POUNCE, PATIENT, WRESTLE = range(4)


def _map_path(version: str = MAP_VERSION):
    return ss.STYLE_DIR / f"style_map_{version}.pkl"


def _unit(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


# ---------------------------------------------------------------------------
# Fit (once per MAP_VERSION)
# ---------------------------------------------------------------------------

def role_order(centroids: np.ndarray) -> list[int]:
    """KMeans cluster index for each family id (RANGE, POUNCE, PATIENT, WRESTLE)."""
    col = ss.ALL_FEATURES.index
    left = list(range(len(centroids)))
    wrestle = max(left, key=lambda c: centroids[c, col("td_att15s")]); left.remove(wrestle)
    pounce = max(left, key=lambda c: centroids[c, col("gnp_per_ctrl")]); left.remove(pounce)
    patient = min(left, key=lambda c: centroids[c, col("str_att_pm")]); left.remove(patient)
    return [left[0], pounce, patient, wrestle]


def fit_map(U: np.ndarray) -> dict:
    """Four families + t-SNE layout on unit vectors (KMeans on the unit sphere is cosine
    clustering). Centroids are stored unit-normalised in family-id order."""
    km = KMeans(n_clusters=N_FAMILIES, n_init=20, random_state=0).fit(U)
    order = role_order(km.cluster_centers_)
    xy = TSNE(n_components=2, metric="cosine", perplexity=30, init="pca", random_state=0).fit_transform(U)
    lo, hi = xy.min(0), xy.max(0)
    return {
        "version": MAP_VERSION,
        "space_version": ss.SPACE_VERSION,
        "centroids": _unit(km.cluster_centers_[order]),
        "blend_tau": BLEND_TAU,
        "anchors": U,
        "anchor_xy": (xy - lo) / (hi - lo),
        "fitted_on": date.today().isoformat(),
        "n_fit_rows": len(U),
    }


def save_map(m: dict) -> None:
    path = _map_path(m["version"])
    if path.exists():
        raise FileExistsError(f"{path} exists — bump MAP_VERSION to refit (see module docstring)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(m, fh)
    log.info(f"  Saved style map to {path}")


def load_map(version: str = MAP_VERSION) -> dict | None:
    path = _map_path(version)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        m = pickle.load(fh)
    if m.get("space_version") != ss.SPACE_VERSION:
        raise ValueError(f"{path} was fit on style space {m.get('space_version')}, "
                         f"not {ss.SPACE_VERSION} — bump MAP_VERSION and refit")
    return m


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def blends(U: np.ndarray, m: dict) -> np.ndarray:
    """Calibrated soft membership, rows sum to 1, columns in family-id order."""
    S = U @ m["centroids"].T
    e = np.exp((S - S.max(1, keepdims=True)) / m["blend_tau"])
    return e / e.sum(1, keepdims=True)


def assign_families(U: np.ndarray, m: dict) -> np.ndarray:
    """Primary family per row: the largest blend share."""
    return blends(U, m).argmax(1)


def blend_list(p: np.ndarray) -> list[dict]:
    order = np.argsort(-p)
    return [{"family": int(k), "share": round(float(p[k]), 3)} for k in order if p[k] >= BLEND_MIN_SHARE]


def place(U: np.ndarray, m: dict) -> np.ndarray:
    """Map coordinates: softmax-weighted mean of the closest anchors' frozen positions."""
    S = U @ m["anchors"].T
    k = min(ANCHOR_K, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    sims = np.take_along_axis(S, idx, axis=1)
    w = np.exp((sims - sims.max(1, keepdims=True)) / ANCHOR_TAU)
    w /= w.sum(1, keepdims=True)
    return np.einsum("nk,nkd->nd", w, m["anchor_xy"][idx])


def ancestors(U: np.ndarray, debut: np.ndarray, pool: np.ndarray) -> list[list[tuple[int, float]]]:
    """Per row, the N_ANCESTORS most similar rows in `pool` who debuted LINEAGE_GAP+ years earlier."""
    S = U @ U.T
    eligible = (debut[None, :] <= debut[:, None] - LINEAGE_GAP) & pool[None, :]
    S = np.where(eligible, S, -np.inf)
    out = []
    for i in range(len(U)):
        top = np.argsort(-S[i])[:N_ANCESTORS]
        out.append([(int(j), float(S[i, j])) for j in top if np.isfinite(S[i, j])])
    return out


def build_style_page(db, inputs: "ss.StyleInputs", space: dict, refit_map: bool = False) -> dict:
    """Everything the page needs, as plain data. No DB writes except a first map fit's file."""
    fids, prime = inputs.fids, inputs.prime
    U = _unit(ss.embed(inputs, space))
    m = None if refit_map else load_map()
    if m is None:
        if not refit_map:
            # Never fit on the fly: on a CI runner the file is lost and every event would
            # refit, reshuffling families and their names.
            raise RuntimeError(f"No frozen style map {MAP_VERSION} at {_map_path()}; "
                               f"commit it, or run with --refit-map deliberately")
        log.info("  Fitting style map (--refit-map)")
        m = fit_map(U)
        save_map(m)
    else:
        log.info(f"  Using frozen style map {m['version']} fit {m['fitted_on']} on {m['n_fit_rows']} rows")

    P = blends(U, m)
    fam = P.argmax(1)
    xy = place(U, m)
    profiles = inputs.profiles
    quality = np.array([np.mean([profiles[f][c] for c in ss.SKILL_FEATURES]) for f in fids])
    notable = quality >= np.quantile(quality, NOTABLE_QUANTILE)
    debut = inputs.debut if len(inputs.debut) == len(fids) else prime
    anc_any = ancestors(U, debut, np.ones(len(fids), dtype=bool))
    anc_notable = ancestors(U, debut, notable)

    registry = build_fighter_registry(db)
    today = date.today()
    active = [bool(registry.get(f) and is_rankable(registry[f], today, Eligibility())) for f in fids]

    shown = ss.PROFILE_FEATURES
    families = []
    for c, (name, blurb) in enumerate(FAMILIES):
        idx = np.where(fam == c)[0]
        traits = [ss.ALL_FEATURES[k] for k in np.argsort(-m["centroids"][c])[:N_TRAITS]]
        families.append({"id": c, "name": name, "blurb": blurb, "traits": traits,
                         "n_fighters": int(len(idx)),
                         "profile": {col: round(float(np.mean([profiles[fids[i]][col] for i in idx])), 1)
                                     for col in shown} if len(idx) else {}})

    fighters = []
    for i, f in enumerate(fids):
        first, last = inputs.years[f]
        fighters.append({
            "fighter_id": f, "family": int(fam[i]),
            "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
            "first_year": first, "last_year": last, "prime_year": float(prime[i]),
            "active": active[i], "quality": round(float(quality[i]), 1), "notable": bool(notable[i]),
            # The blend rides in the profile JSON (no schema change); the router lifts it
            # out into its own field.
            "profile": {**{c: round(float(profiles[f][c]), 1) for c in shown}, "_blend": blend_list(P[i])},
            "ancestors": [{"id": fids[j], "sim": round(s, 3)} for j, s in anc_any[i]],
            "notable_ancestors": [{"id": fids[j], "sim": round(s, 3)} for j, s in anc_notable[i]],
            "division": inputs.divisions.get(f),
        })
    log.info(f"  Style page: {len(fighters)} fighters, {sum(active)} active, {int(notable.sum())} notable")
    return {"version": m["version"], "families": families, "fighters": fighters}


def save_style_page(db, page: dict) -> None:
    try:
        write_style_page(db, page, datetime.now(timezone.utc))
        db.commit()
    except Exception:
        db.rollback()
        raise


def write_style_page(db, page: dict, now) -> None:
    """Replace both page tables inside the caller's transaction (no commit)."""
    db.query(UFCFighterStyle).delete()
    db.query(UFCStyleFamily).delete()
    db.bulk_save_objects([
        UFCStyleFamily(id=f["id"], name=f["name"], blurb=f["blurb"], traits=json.dumps(f["traits"]),
                       profile=json.dumps(f["profile"]), n_fighters=f["n_fighters"],
                       map_version=page["version"], computed_at=now)
        for f in page["families"]
    ])
    db.bulk_save_objects([
        UFCFighterStyle(
            fighter_id=r["fighter_id"], family=r["family"], x=r["x"], y=r["y"],
            first_year=r["first_year"], last_year=r["last_year"], prime_year=r["prime_year"],
            active=r["active"], quality=r["quality"], notable=r["notable"],
            profile=json.dumps(r["profile"]), ancestors=json.dumps(r["ancestors"]),
            notable_ancestors=json.dumps(r["notable_ancestors"]),
            division=r["division"], map_version=page["version"], computed_at=now,
        )
        for r in page["fighters"]
    ])
    log.info(f"  Saved {len(page['fighters'])} style rows, {len(page['families'])} families")


def compute_and_save_style_page(db, inputs, space, refit_map: bool = False) -> int:
    """Page tables only (CLI and admin). The post-event path writes them together with the
    similarity table in `style_service.compute_and_save_similarity`."""
    page = build_style_page(db, inputs, space, refit_map=refit_map)
    save_style_page(db, page)
    return len(page["fighters"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Fighter Styles page data")
    p.add_argument("--refit-map", action="store_true", help="Fit a new frozen map (needs a new MAP_VERSION)")
    p.add_argument("--dry-run", metavar="OUT_JSON", help="Write the page data to a JSON file instead of the DB")
    a = p.parse_args()
    session = SessionLocal()
    try:
        inputs_ = ss.build_inputs(session)
        space_ = ss.load_space()
        if space_ is None:
            raise SystemExit(f"No frozen style space {ss.SPACE_VERSION}; run style_service --refit first")
        page_ = build_style_page(session, inputs_, space_, refit_map=a.refit_map)
        if a.dry_run:
            with open(a.dry_run, "w") as fh:
                json.dump(page_, fh, default=str)
            log.info(f"Wrote {a.dry_run}")
        else:
            save_style_page(session, page_)
    finally:
        session.close()
