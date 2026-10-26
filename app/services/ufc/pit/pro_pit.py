"""Step 2: pre-UFC / whole-career quality from Sherdog, as of before each UFC fight.

The winner model's newcomer view is thin: pro_elo (a plain K=48 Elo where a regional win
counts like a UFC win) and record counts. This block uses app/services/ufc/pro_rating.py
(PARAMS: the best variant from scripts/pro_rating_eval.py) plus the bout-level detail it
records (each bout's pre-bout ratings and the promotion's as-of strength):

  pro2_rating, pro2_rd    rating and uncertainty on the all-promotions scale
  pro2_opp_mean           mean pre-bout rating of opponents faced (all bouts)
  pro2_best_win           best opponent pre-bout rating beaten
  pro2_worst_loss         weakest opponent pre-bout rating lost to (NaN if never lost)
  pro2_qwins              quality-weighted record: mean over bouts of (score - expected)
  pro2_promo_mean         mean as-of strength of the promotions fought on outside the UFC
  pro2_top_tier           bouts on promotions at least TOP_TIER rating points above 1500
  pro2_dwcs_win, pro2_tuf contender-series win / any TUF bout
  pro2_streak             current unbeaten run
  pro2_days_since_loss    days since the last loss (capped at 3650; 3650 = never lost)
  pro2_years_pro          years since the first pro bout
  pro2_active2y           bouts in the previous two years
  pro2_finish_rate        finishes / wins, shrunk with 3 pseudo-wins at 0.5
  pro2_am_n, pro2_am_wpct amateur bouts and shrunk win rate (cache re-parse; optional)

Raw red_/blue_ copies for pro2_rating and pro2_rd so the model sees each corner's level and
certainty, not only the gap.
"""
from __future__ import annotations

import json
import logging
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.ufc.pro_rating import INIT, Params, run

log = logging.getLogger("pro_pit")

PREFIX = "pro2_"
FIGHT_LEVEL = ()
RAW = ("pro2_rating", "pro2_rd")
TOP_TIER = 60.0
ROOT = Path(__file__).resolve().parents[4]
BOUTS = ROOT / "data" / "feature_wf" / "sherdog_bouts.pkl"
AMATEUR = ROOT / "data" / "feature_wf" / "sherdog_amateur.pkl"
PARAMS_FILE = ROOT / "data" / "feature_wf" / "pro_rating_best.json"


def _params() -> Params:
    if PARAMS_FILE.exists():
        return Params(**json.loads(PARAMS_FILE.read_text()))
    return Params(rd=True, rd0=300.0, c=4.0, promo=True, promo_shrink=30.0, fin=1.0)


def fighter_lines(bouts: pd.DataFrame, det: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    """Per sherdog id, its bouts in date order with own/opponent pre-bout ratings."""
    a = pd.DataFrame({"f": bouts["a"], "date": bouts["date"], "score": bouts["a_score"], "me": det["pre_a"],
                      "opp": det["pre_b"], "mc": bouts["method_class"], "promo": bouts["promo"],
                      "pstr": det["promo_strength"]})
    b = pd.DataFrame({"f": bouts["b"], "date": bouts["date"], "score": 1 - bouts["a_score"], "me": det["pre_b"],
                      "opp": det["pre_a"], "mc": bouts["method_class"], "promo": bouts["promo"],
                      "pstr": det["promo_strength"]})
    allb = pd.concat([a, b], ignore_index=True).sort_values(["f", "date"], kind="stable")
    out = {}
    for f, g in allb.groupby("f", sort=False):
        out[int(f)] = {k: g[k].to_numpy() for k in ("date", "score", "me", "opp", "mc", "promo", "pstr")}
    return out


def features_from(bouts: pd.DataFrame, rows: pd.DataFrame, link: dict[int, int],
                  amateur: pd.DataFrame | None = None, params: Params | None = None) -> pd.DataFrame:
    """rows: fight_id, stats_fighter_id, date. link: ufc fighter id -> sherdog id."""
    params = params or _params()
    _, hist, det = run(bouts, params, details=True)
    lines = fighter_lines(bouts, det)
    am = defaultdict(list)
    if amateur is not None and len(amateur):
        for sid, d, r in zip(amateur["sherdog_id"], amateur["date"], amateur["result"]):
            if pd.notna(d):
                am[int(sid)].append((d.date(), r))
    out = []
    for fid, uid, d in zip(rows["fight_id"], rows["stats_fighter_id"], pd.to_datetime(rows["date"]).dt.date):
        row = {"fight_id": int(fid), "fighter_id": int(uid)}
        sid = link.get(uid)
        line = lines.get(sid) if sid is not None else None
        if sid is not None:
            ams = [r for dd, r in am.get(sid, []) if dd < d]
            row["pro2_am_n"] = float(len(ams))
            row["pro2_am_wpct"] = (sum(r == "W" for r in ams) + 1) / (len(ams) + 2)
        if line is None:
            out.append(row)
            continue
        k = bisect_left(list(line["date"]), d)
        if k == 0:
            out.append(row)
            continue
        h = hist[sid]
        j = bisect_left(h["date"], d)
        r, rd, n, _ = h["state"][j - 1]
        sc, me, opp = line["score"][:k], line["me"][:k], line["opp"][:k]
        mc, promo, pstr, dates = line["mc"][:k], line["promo"][:k], line["pstr"][:k], line["date"][:k]
        exp = 1 / (1 + 10 ** ((opp - me) / 400.0))
        wins = sc == 1.0
        nonufc = np.array([p not in ("UFC", "TUF") for p in promo])
        losses = np.where(sc == 0.0)[0]
        streak = k - 1 - losses[-1] if len(losses) else k
        fin = wins & np.isin(mc, ["KO/TKO", "SUB"])
        row.update(
            pro2_rating=r, pro2_rd=rd if params.rd else np.nan,
            pro2_opp_mean=float(opp.mean()),
            pro2_best_win=float(opp[wins].max()) if wins.any() else np.nan,
            pro2_worst_loss=float(opp[sc == 0.0].min()) if len(losses) else np.nan,
            pro2_qwins=float((sc - exp).mean()),
            pro2_promo_mean=float(pstr[nonufc].mean()) if nonufc.any() else np.nan,
            pro2_top_tier=float((pstr >= INIT + TOP_TIER).sum()),
            pro2_dwcs_win=float(any(p == "DWCS" and s == 1.0 for p, s in zip(promo, sc))),
            pro2_tuf=float(any(p == "TUF" for p in promo)),
            pro2_streak=float(streak),
            pro2_days_since_loss=float(min((d - dates[losses[-1]]).days, 3650)) if len(losses) else 3650.0,
            pro2_years_pro=(d - dates[0]).days / 365.25,
            pro2_active2y=float(sum((d - x).days <= 730 for x in dates)),
            pro2_finish_rate=(fin.sum() + 1.5) / (wins.sum() + 3),
        )
        out.append(row)
    return pd.DataFrame(out)


def compute(ctx) -> pd.DataFrame:
    from sqlalchemy import text
    bouts = pd.read_pickle(BOUTS)
    amateur = pd.read_pickle(AMATEUR) if AMATEUR.exists() else None
    db = ctx.session()
    try:
        link = {int(u): int(s) for s, u in db.execute(text(
            "SELECT sherdog_id, ufc_fighter_id FROM ufc.sherdog_fighters WHERE ufc_fighter_id IS NOT NULL"))}
    finally:
        db.close()
    rows = ctx.fighters[["fight_id", "stats_fighter_id", "date"]]
    log.info("  pro2 params: %s", _params())
    return features_from(bouts, rows, link, amateur)
