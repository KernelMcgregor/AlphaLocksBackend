"""Clutch rating: how a fighter does in the rounds that test it — late rounds (3+) and
the round right after losing one — relative to their own other rounds in the same bout.

Every round is read on the deserve-to-win round model's scale: x = logit P(a judge
gives the round to this fighter), from ufc_deserve_to_win.round_p_red (v1), observed
full rounds only (the projected rounds of a finish and its partial last round are
dropped). Comparisons stay inside a bout, so how good the fighter is (and how good
the opponent was) cancels out. Per bout and corner, with at least three full rounds:

  late    mean(x_3..) - mean(x_1, x_2)
  bounce  for each round r after a lost round (x_{r-1} < 0): x_r minus the mean of the
          bout's rounds other than r and r-1. The lost round itself is left out of the
          baseline: picking rounds after a loss selects on that round's noise.

Each is centred on its league mean, and a bout's clutch value is the mean of the two
(whichever exist). Bout values are averaged per fighter and shrunk toward 0 with an
empirical-Bayes prior (between-fighter variance by method of moments). Units are
logit points of round-win probability: +0.4 is roughly +10 points of P(win the
round) at an even round.

  --dry-run OUT.json   prod reads, ratings to a JSON file, no writes
  --from-json IN.json  load ratings from a dry run instead of computing
  --write              replace this version's rows in ufc_fighter_clutch

  PYTHONPATH=. venv/bin/python -m app.services.ufc.fightflow_clutch --dry-run out.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
from collections import defaultdict

import numpy as np
from sqlalchemy import inspect
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

VERSION = "v1"
DTW_VERSION = "v1"
#: Round probabilities are clipped before the logit: the round model puts a few
#: one-sided rounds at 0.0003, and a -8 would swamp a fighter's average.
P_CLIP = 0.02
#: Bouts (of three or more full rounds) needed for a percentile.
MIN_FIGHTS = 3


def logit(p: float) -> float:
    p = min(1 - P_CLIP, max(P_CLIP, p))
    return math.log(p / (1 - p))


def side_rounds(round_p_red: list[float], rounds_observed: int,
                partial_round_seconds: int | None) -> list[float]:
    """Red's observed full-round logits (blue's are the negatives)."""
    n = rounds_observed - (1 if partial_round_seconds is not None else 0)
    return [logit(p) for p in round_p_red[:max(0, n)]]


def bout_deltas(x: list[float]) -> dict:
    """One corner's round logits in one bout -> {late, bounce: [..], late_rounds,
    clutch_rounds}. Fewer than three full rounds -> late None (nothing to compare)."""
    out = {"late": None, "bounce": [], "late_rounds": 0, "clutch_rounds": 0}
    if len(x) < 3:
        return out
    out["late"] = sum(x[2:]) / len(x[2:]) - (x[0] + x[1]) / 2
    out["late_rounds"] = len(x) - 2
    # Distinct clutch rounds: the late ones, plus R2 when it followed a lost R1.
    out["clutch_rounds"] = out["late_rounds"] + (1 if x[0] < 0 else 0)
    for r in range(1, len(x)):              # 0-based: round r+1 follows round r
        if x[r - 1] < 0:
            rest = [v for j, v in enumerate(x) if j not in (r - 1, r)]
            out["bounce"].append(x[r] - sum(rest) / len(rest))
    return out


def bout_rows(bouts: list[dict]) -> list[dict]:
    """One row per (bout, corner) with three or more full rounds. `bouts`:
    {fight_id, red, blue, x} with x red's round logits (blue's are the negatives)."""
    rows = []
    for b in bouts:
        for corner, sign in (("red", 1.0), ("blue", -1.0)):
            d = bout_deltas([sign * v for v in b["x"]])
            if d["late"] is not None:
                rows.append({"fight_id": b["fight_id"], "fighter_id": b[corner], **d})
    return rows


def _shrink(means: dict, counts: dict, sigma2: float) -> tuple[float, dict]:
    """tau^2 by method of moments over fighters with >= MIN_FIGHTS bouts, then the
    posterior mean and SD of each fighter's effect."""
    pool = [k for k in means if counts[k] >= MIN_FIGHTS]
    if len(pool) >= 2:
        m = np.array([means[k] for k in pool])
        noise = np.mean([sigma2 / counts[k] for k in pool])
        tau2 = max(float(np.var(m) - noise), 1e-4)
    else:
        tau2 = 1e-4
    out = {}
    for k, mu in means.items():
        v = sigma2 / counts[k]
        w = tau2 / (tau2 + v)
        out[k] = (w * mu, math.sqrt(w * v))
    return tau2, out


def rate(bouts: list[dict]) -> dict:
    """Clutch ratings from {fight_id, red, blue, x} bouts. Returns {meta, fighters}."""
    rows = bout_rows(bouts)
    if not rows:
        return {"meta": {"n_bouts": 0}, "fighters": {}}
    late_mu = float(np.mean([r["late"] for r in rows]))
    bounces = [v for r in rows for v in r["bounce"]]
    bounce_mu = float(np.mean(bounces)) if bounces else 0.0

    # Per (fighter, bout): each component and their mean, plus the rounds behind them.
    obs = {"all": defaultdict(list), "late": defaultdict(list), "lost": defaultdict(list)}
    n_rounds = {"all": defaultdict(int), "late": defaultdict(int), "lost": defaultdict(int)}
    for r in rows:
        f = r["fighter_id"]
        late = r["late"] - late_mu
        parts = [late]
        obs["late"][f].append(late)
        n_rounds["late"][f] += r["late_rounds"]
        if r["bounce"]:
            bounce = float(np.mean(r["bounce"])) - bounce_mu
            parts.append(bounce)
            obs["lost"][f].append(bounce)
            n_rounds["lost"][f] += len(r["bounce"])
        obs["all"][f].append(sum(parts) / len(parts))
        n_rounds["all"][f] += r["clutch_rounds"]

    fighters: dict = defaultdict(dict)
    meta = {"late_mean": late_mu, "bounce_mean": bounce_mu, "n_bouts": len(rows)}
    for kind, by_fighter in obs.items():
        sigma2 = float(np.var([v for vs in by_fighter.values() for v in vs]))
        means = {f: float(np.mean(v)) for f, v in by_fighter.items()}
        counts = {f: len(v) for f, v in by_fighter.items()}
        tau2, post = _shrink(means, counts, sigma2)
        meta[kind] = {"sigma2": sigma2, "tau2": tau2}
        for f, (est, se) in post.items():
            fighters[f][kind] = {"est": est, "se": se, "n_fights": counts[f], "n_rounds": n_rounds[kind][f]}

    # Percentile of the headline among fighters with enough bouts.
    ranked = [f for f, d in fighters.items() if d["all"]["n_fights"] >= MIN_FIGHTS]
    vals = np.array([fighters[f]["all"]["est"] for f in ranked])
    for f in ranked:
        v = fighters[f]["all"]["est"]
        below = (vals < v).sum() + 0.5 * ((vals == v).sum() - 1)
        fighters[f]["percentile"] = float(100 * below / (len(vals) - 1)) if len(vals) > 1 else 50.0
    return {"meta": meta, "fighters": dict(fighters)}


# ----------------------------------------------------------------------- data


def load_bouts(db: Session) -> list[dict]:
    from app.models.ufc import UFCDeserveToWin, UFCFight
    q = (db.query(UFCDeserveToWin.fight_id, UFCDeserveToWin.round_p_red,
                  UFCDeserveToWin.rounds_observed, UFCDeserveToWin.partial_round_seconds,
                  UFCFight.red_fighter_id, UFCFight.blue_fighter_id)
         .join(UFCFight, UFCFight.id == UFCDeserveToWin.fight_id)
         .filter(UFCDeserveToWin.model_version == DTW_VERSION))
    out = []
    for fid, rp, obs, partial, red, blue in q:
        x = side_rounds(json.loads(rp) if rp else [], obs or 0, partial)
        if len(x) >= 2:
            out.append({"fight_id": fid, "red": red, "blue": blue, "x": x})
    return out


def to_rows(result: dict) -> list[dict]:
    rows = []
    for f, d in result["fighters"].items():
        a = d["all"]
        late = d.get("late") or {}
        lost = d.get("lost") or {}
        rows.append({
            "fighter_id": int(f), "clutch": a["est"], "se": a["se"],
            "late": late.get("est"), "late_rounds": late.get("n_rounds", 0),
            "bounce": lost.get("est"), "bounce_rounds": lost.get("n_rounds", 0),
            "n_rounds": a["n_rounds"], "n_fights": a["n_fights"],
            "percentile": d.get("percentile"),
        })
    return rows


def write(db: Session, rows: list[dict]) -> int:
    from app.models.ufc_fightflow import UFCFighterClutch
    UFCFighterClutch.__table__.create(db.get_bind(), checkfirst=True)
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    db.query(UFCFighterClutch).filter(UFCFighterClutch.version == VERSION).delete()
    db.bulk_insert_mappings(UFCFighterClutch, [
        {**r, "version": VERSION, "computed_at": now} for r in rows])
    db.commit()
    return len(rows)


def payload(db: Session, fighter_id: int) -> dict | None:
    from app.models.ufc_fightflow import UFCFighterClutch
    t = UFCFighterClutch.__table__
    if not inspect(db.get_bind()).has_table(t.name, schema=t.schema):
        return None
    r = (db.query(UFCFighterClutch)
         .filter(UFCFighterClutch.fighter_id == fighter_id, UFCFighterClutch.version == VERSION)
         .first())
    if r is None:
        return None
    return {
        "fighter_id": str(r.fighter_id), "version": r.version,
        "clutch": r.clutch, "se": r.se, "percentile": r.percentile,
        "late": r.late, "late_rounds": r.late_rounds,
        "bounce": r.bounce, "bounce_rounds": r.bounce_rounds,
        "n_rounds": r.n_rounds, "n_fights": r.n_fights,
        "computed_at": r.computed_at.isoformat() if r.computed_at else None,
    }


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", metavar="OUT_JSON", help="write ratings to a JSON file (no DB writes)")
    ap.add_argument("--from-json", metavar="IN_JSON", help="ratings from an earlier dry run")
    ap.add_argument("--write", action="store_true", help="replace this version's rows in ufc_fighter_clutch")
    a = ap.parse_args(argv)
    if not (a.dry_run or a.write):
        ap.error("pass --dry-run OUT.json and/or --write")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        if a.from_json:
            with open(a.from_json) as fh:
                rows = json.load(fh)["rows"]
        else:
            result = rate(load_bouts(db))
            rows = to_rows(result)
            log.info("%d fighters rated (%d with a percentile)", len(rows),
                     sum(r["percentile"] is not None for r in rows))
            if a.dry_run:
                with open(a.dry_run, "w") as fh:
                    json.dump({"meta": result["meta"], "rows": sorted(rows, key=lambda r: -r["clutch"])},
                              fh, indent=1, default=str)
                log.info("Wrote %s", a.dry_run)
        if a.write:
            log.info("Wrote %d rows to ufc_fighter_clutch", write(db, rows))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
