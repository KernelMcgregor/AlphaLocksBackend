"""Fit, check and write the fight excitement model (app/services/ufc/fight_excitement.py).

    python -m scripts.fight_excitement --eval     # fit on cards before 2023, test 2023+; no writes
    python -m scripts.fight_excitement --write    # fit on every labelled card, score all, write

Writes ufc.ufc_fight_excitement (a version's rows replaced wholesale) and the fitted
weights to models/ufc/fight_excitement/model.json. Scoring every bout takes seconds, so
the post-event job simply reruns --write.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.ufc.fight_excitement import (
    CONTEXT, EXCITE, GROUPS, MODEL_VERSION, PERF, card_log_loss, fit_card_logit,
    group_contributions, load_fights, topk_hit,
)

log = logging.getLogger("fight_excitement")
OUT = Path(__file__).resolve().parents[1] / "models" / "ufc" / "fight_excitement"

#: Cards before this have spotty stats/odds and a different bonus menu.
FIT_FROM = dt.date(2008, 1, 1)
#: Performance of the Night replaced KO / Submission of the Night here.
POTN_FROM = dt.date(2014, 2, 15)
TEST_FROM = dt.date(2023, 1, 1)
FOTN_FEATS = EXCITE + CONTEXT
PERF_FEATS = PERF + CONTEXT


def labelled(df: pd.DataFrame, col: str, start: dt.date) -> pd.DataFrame:
    """Cards from `start` on that have at least one `col` bonus recorded."""
    d = df[df["date"] >= start]
    has = d.groupby("event_id")[col].transform("any")
    return d[has]


def perf_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["has_winner"]]


def evaluate(df: pd.DataFrame) -> None:
    fo = labelled(df, "fotn", FIT_FROM)
    tr, te = fo[fo["date"] < TEST_FROM], fo[fo["date"] >= TEST_FROM]
    print(f"FOTN: {tr.event_id.nunique()} train cards / {te.event_id.nunique()} test cards")
    y_te, g_te = te["fotn"].to_numpy(float), te["event_id"].to_numpy()
    for name, feats in (("card position only", CONTEXT), ("excitement only", EXCITE), ("full", FOTN_FEATS)):
        m = fit_card_logit(tr, tr["fotn"].to_numpy(float), tr["event_id"].to_numpy(), feats)
        p = m.probs(te, g_te)
        print(f"  {name:20s} logloss {card_log_loss(p, y_te, g_te):.3f}  top1 {topk_hit(p, y_te, g_te, 1):.1%}"
              f"  top2 {topk_hit(p, y_te, g_te, 2):.1%}  top3 {topk_hit(p, y_te, g_te, 3):.1%}")
    n = te.groupby("event_id").size()
    print(f"  {'uniform':20s} logloss {np.log(n).mean():.3f}")

    pf = perf_rows(labelled(df, "perf", POTN_FROM))
    tr, te = pf[pf["date"] < TEST_FROM], pf[pf["date"] >= TEST_FROM]
    y_te, g_te = te["perf"].to_numpy(float), te["event_id"].to_numpy()
    print(f"POTN: {tr.event_id.nunique()} train cards / {te.event_id.nunique()} test cards")
    for name, feats in (("card position only", CONTEXT), ("full", PERF_FEATS)):
        m = fit_card_logit(tr, tr["perf"].to_numpy(float), tr["event_id"].to_numpy(), feats)
        p = m.probs(te, g_te)
        print(f"  {name:20s} logloss {card_log_loss(p, y_te, g_te):.3f}  top1 {topk_hit(p, y_te, g_te, 1):.1%}"
              f"  top2 {topk_hit(p, y_te, g_te, 2):.1%}")


def fit_all(df: pd.DataFrame):
    fo = labelled(df, "fotn", FIT_FROM)
    fotn = fit_card_logit(fo, fo["fotn"].to_numpy(float), fo["event_id"].to_numpy(), FOTN_FEATS)
    pf = perf_rows(labelled(df, "perf", POTN_FROM))
    perf = fit_card_logit(pf, pf["perf"].to_numpy(float), pf["event_id"].to_numpy(), PERF_FEATS)
    k_perf = float(pf.groupby("event_id")["perf"].sum().mean())
    return fotn, perf, k_perf


def score(df: pd.DataFrame, fotn, perf, k_perf: float) -> pd.DataFrame:
    g = df["event_id"].to_numpy()
    out = df[["fight_id", "event_id", "date", "fotn", "perf"]].copy()
    out["score"] = fotn.utility(df, only=EXCITE)
    out["pct"] = out["score"].rank(pct=True) * 100
    out["p_fotn"] = fotn.probs(df, g)
    pw = df["has_winner"].to_numpy()
    p = np.full(len(df), np.nan)
    sub = df[pw]
    share = perf.probs(sub, sub["event_id"].to_numpy())
    # About k_perf awards a card: the chance this winner is among them.
    p[pw] = 1 - (1 - share) ** k_perf
    out["p_perf"] = p
    contrib = group_contributions(fotn, df)
    raw = df[EXCITE].round(4)
    out["components"] = [json.dumps({"groups": {k: round(float(v), 3) for k, v in c.items()},
                                     "raw": r.to_dict()})
                         for (_, c), (_, r) in zip(contrib.iterrows(), raw.iterrows())]
    return out


def show_weights(m, title: str) -> None:
    print(title)
    for f, w in sorted(zip(m.features, m.w), key=lambda t: -abs(t[1])):
        print(f"  {f:16s} {w:+.3f}")


def write(scored: pd.DataFrame) -> None:
    from app.database import SessionLocal, engine
    from app.models.ufc_highlights import UFCFightExcitement

    UFCFightExcitement.__table__.create(bind=engine, checkfirst=True)
    db = SessionLocal()
    try:
        db.query(UFCFightExcitement).filter_by(model_version=MODEL_VERSION).delete()
        db.bulk_insert_mappings(UFCFightExcitement, [
            {"fight_id": int(r.fight_id), "model_version": MODEL_VERSION, "event_id": int(r.event_id),
             "date": r.date, "score": float(r.score), "pct": float(r.pct), "components": r.components,
             "p_fotn": float(r.p_fotn), "p_perf": None if np.isnan(r.p_perf) else float(r.p_perf),
             "fotn": bool(r.fotn), "perf": bool(r.perf)}
            for r in scored.itertuples()])
        db.commit()
    finally:
        db.close()
    log.info(f"wrote {len(scored)} rows ({MODEL_VERSION})")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--top", type=int, default=25, help="print the top N fights")
    a = ap.parse_args()

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        df = load_fights(db)
    finally:
        db.close()
    log.info(f"{len(df)} UFC bouts, {df.event_id.nunique()} cards")

    if a.eval:
        evaluate(df)
    fotn, perf, k_perf = fit_all(df)
    show_weights(fotn, "FOTN weights (standardized)")
    show_weights(perf, "POTN weights (standardized)")
    scored = score(df, fotn, perf, k_perf)
    if a.top:
        top = scored.nlargest(a.top, "score").merge(df[["fight_id", "event_name", "kind"]], on="fight_id")
        print(top[["date", "event_name", "kind", "pct", "p_fotn", "fotn"]].to_string(index=False))
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "model.json").write_text(json.dumps({
            "version": MODEL_VERSION, "fotn": json.loads(fotn.to_json()),
            "perf": json.loads(perf.to_json()), "k_perf": k_perf, "groups": GROUPS}, indent=1))
        write(scored)


if __name__ == "__main__":
    main()
