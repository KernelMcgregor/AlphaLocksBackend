"""Per-fighter finishing propensities for the method model, point-in-time.

Everything is counted over bouts strictly BEFORE the fight, and every rate is shrunk
toward the pre-2015 population rate with pseudo-counts, so a 2-fight fighter reads close
to average instead of 0% or 100%.

Two kinds of evidence (process stats generalise better than outcome counts: Wheatcroft
2020 GAP ratings; Holmes et al. 2023 attack/defence skill models):

UFC process stats (per exposure)
  m_power        (knockdowns landed + KO wins) per head strike landed
  m_chin         (knockdowns absorbed + KO losses) per head strike absorbed (higher = worse)
  m_sub_threat   (sub attempts + 2 x sub wins) per fight minute
  m_sub_vuln     (sub attempts against + 2 x sub losses) per fight minute
UFC outcome shares
  m_ko_win_share, m_sub_win_share, m_dec_win_share      share of wins by each method
  m_ko_loss_share, m_sub_loss_share, m_dec_loss_share   share of losses by each method
  m_finish_rate  share of decided fights that ended inside the distance (either way)
  m_ufc_n        decided UFC fights
Sherdog whole-career outcome shares (all promotions; NaN if not linked)
  pro_ko_win_share, pro_sub_win_share, pro_ko_loss_share, pro_sub_loss_share,
  pro_finish_rate
"""
from __future__ import annotations

import logging
from bisect import bisect_left
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from app.services.ufc.outcome_types import classify_outcome

log = logging.getLogger("method_ratings")

UFC_FEATURES = ("m_power", "m_chin", "m_sub_threat", "m_sub_vuln",
                "m_ko_win_share", "m_sub_win_share", "m_dec_win_share",
                "m_ko_loss_share", "m_sub_loss_share", "m_dec_loss_share",
                "m_finish_rate", "m_ufc_n")
PRO_FEATURES = ("pro_ko_win_share", "pro_sub_win_share", "pro_ko_loss_share",
                "pro_sub_loss_share", "pro_finish_rate")
FEATURES = UFC_FEATURES + PRO_FEATURES

PRIOR_CUTOFF = date(2015, 1, 1)
A = 3.0      # pseudo-fights for outcome shares
H = 150.0    # pseudo head strikes for power / chin
M = 30.0     # pseudo minutes for sub rates

_METHOD = {"ko": "ko", "doctor": "ko", "injury": "ko", "sub": "sub",
           "ud": "dec", "md": "dec", "sd": "dec"}


def method_class(method, details, winner_id) -> str | None:
    """'ko' | 'sub' | 'dec' for a decided bout, else None (draw, NC, DQ, overturned)."""
    return _METHOD.get(classify_outcome(method, details, winner_id))


def _num(x) -> float:
    return float(x) if x is not None and x == x else 0.0


def _shares(c: dict, pri: dict) -> list[float]:
    w, l, n = c["w"], c["l"], c["w"] + c["l"]
    return [
        (c["ko_w"] + A * pri["ko"]) / (w + A), (c["sub_w"] + A * pri["sub"]) / (w + A),
        (c["dec_w"] + A * pri["dec"]) / (w + A),
        (c["ko_l"] + A * pri["ko"]) / (l + A), (c["sub_l"] + A * pri["sub"]) / (l + A),
        (c["dec_l"] + A * pri["dec"]) / (l + A),
        (c["ko_w"] + c["sub_w"] + c["ko_l"] + c["sub_l"] + A * (1 - pri["dec"])) / (n + A),
    ]


def ufc_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: load_fight_data frame (one row per fighter per bout). Aligned to df.index."""
    d = df.assign(_date=pd.to_datetime(df["date"]).dt.date)
    # Opponent's row in the same bout, for strikes / knockdowns / sub attempts absorbed.
    opp = {}
    for fid, g in d.groupby("fight_id"):
        if len(g) == 2:
            a, b = g.index
            opp[a], opp[b] = b, a

    def cls(r):
        return method_class(r["method"], r["details"], r["winner_id"])

    fights = d.drop_duplicates("fight_id")
    played = fights[fights["method"].fillna("") != ""]
    early = played[played["_date"] < PRIOR_CUTOFF]
    classes = early.apply(cls, axis=1).dropna()
    pri = {k: float((classes == k).mean()) for k in ("ko", "sub", "dec")}
    e_rows = d[d["_date"] < PRIOR_CUTOFF]
    head = e_rows["head_landed"].map(_num).sum()
    ko_w = sum(1 for i, r in e_rows.iterrows()
               if cls(r) == "ko" and r["winner_id"] == r["stats_fighter_id"])
    g_pow = (e_rows["kd"].map(_num).sum() + ko_w) / max(head, 1.0)
    mins = (e_rows["fight_time_seconds"].map(_num) / 60.0).sum()
    sub_w = sum(1 for i, r in e_rows.iterrows()
                if cls(r) == "sub" and r["winner_id"] == r["stats_fighter_id"])
    g_sub = (e_rows["sub_att"].map(_num).sum() + 2 * sub_w) / max(mins, 1.0)
    log.info(f"  method priors (pre-2015): KO {pri['ko']:.3f} SUB {pri['sub']:.3f} "
             f"DEC {pri['dec']:.3f}; power {g_pow:.4f}/head strike; sub {g_sub:.4f}/min")

    zero = lambda: dict(w=0, l=0, ko_w=0, sub_w=0, dec_w=0, ko_l=0, sub_l=0, dec_l=0,
                        head_l=0.0, head_a=0.0, kd_l=0.0, kd_a=0.0, sa=0.0, sa_a=0.0,
                        mins=0.0)
    state = defaultdict(zero)
    out = np.full((len(d), len(UFC_FEATURES)), np.nan)
    pos = {ix: k for k, ix in enumerate(d.index)}
    for fid, g in d.sort_values(["_date", "fight_id"], kind="stable").groupby(
            "fight_id", sort=False):
        for ix, r in g.iterrows():  # pre-fight snapshot
            c = state[r["stats_fighter_id"]]
            out[pos[ix]] = [
                (c["kd_l"] + c["ko_w"] + H * g_pow) / (c["head_l"] + H),
                (c["kd_a"] + c["ko_l"] + H * g_pow) / (c["head_a"] + H),
                (c["sa"] + 2 * c["sub_w"] + M * g_sub) / (c["mins"] + M),
                (c["sa_a"] + 2 * c["sub_l"] + M * g_sub) / (c["mins"] + M),
                *_shares(c, pri), c["w"] + c["l"],
            ]
        first = g.iloc[0]
        if not (first["method"] or ""):
            continue  # upcoming bout: nothing to learn from
        k = cls(first)
        for ix, r in g.iterrows():
            c = state[r["stats_fighter_id"]]
            o = d.loc[opp[ix]] if ix in opp else None
            c["head_l"] += _num(r["head_landed"]); c["kd_l"] += _num(r["kd"])
            c["sa"] += _num(r["sub_att"]); c["mins"] += _num(r["fight_time_seconds"]) / 60.0
            if o is not None:
                c["head_a"] += _num(o["head_landed"]); c["kd_a"] += _num(o["kd"])
                c["sa_a"] += _num(o["sub_att"])
            if k is None:
                continue
            if r["winner_id"] == r["stats_fighter_id"]:
                c["w"] += 1; c[f"{k}_w"] += 1
            else:
                c["l"] += 1; c[f"{k}_l"] += 1
    return pd.DataFrame(out, columns=UFC_FEATURES, index=d.index)


_PRO_CLASS = {"KO/TKO": "ko", "SUB": "sub", "DEC": "dec"}


def pro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sherdog whole-career method shares before each bout; NaN when not linked."""
    from sqlalchemy import inspect

    from app.database import SessionLocal, engine
    from app.models.ufc import SherdogBout, SherdogFighter

    out = pd.DataFrame(np.nan, index=df.index, columns=PRO_FEATURES)
    if not inspect(engine).has_table(SherdogBout.__table__.name, schema=SherdogBout.__table__.schema):
        return out
    db = SessionLocal()
    try:
        link = {u: s for s, u in db.query(SherdogFighter.sherdog_id, SherdogFighter.ufc_fighter_id)
                if u is not None}
        rows = db.query(SherdogBout.fighter_sherdog_id, SherdogBout.date, SherdogBout.result,
                        SherdogBout.method_class).filter(SherdogBout.date.isnot(None)).all()
    finally:
        db.close()
    per = defaultdict(list)
    for sid, d, res, mc in rows:
        k = _PRO_CLASS.get(mc)
        if k and res in ("W", "L"):
            per[sid].append((d, res, k))
    pri = {"ko": 0.0, "sub": 0.0, "dec": 0.0}
    allk = [k for v in per.values() for _, _, k in v]
    for k in pri:
        pri[k] = allk.count(k) / max(len(allk), 1)
    hist = {}
    for sid, v in per.items():
        v.sort()
        dates, snaps = [], []
        c = dict(w=0, l=0, ko_w=0, sub_w=0, dec_w=0, ko_l=0, sub_l=0, dec_l=0)
        for d, res, k in v:
            dates.append(d)
            if res == "W":
                c["w"] += 1; c[f"{k}_w"] += 1
            else:
                c["l"] += 1; c[f"{k}_l"] += 1
            snaps.append(dict(c))
        hist[sid] = (dates, snaps)
    empty = dict(w=0, l=0, ko_w=0, sub_w=0, dec_w=0, ko_l=0, sub_l=0, dec_l=0)
    vals = []
    for fid, d in zip(df["stats_fighter_id"], pd.to_datetime(df["date"]).dt.date):
        sid = link.get(fid)
        if sid is None:
            vals.append([np.nan] * len(PRO_FEATURES)); continue
        dates, snaps = hist.get(sid, ([], []))
        k = bisect_left(dates, d)
        s = _shares(snaps[k - 1] if k else empty, pri)
        vals.append([s[0], s[1], s[3], s[4], s[6]])
    out.loc[:, :] = np.array(vals, dtype=float)
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([ufc_features(df), pro_features(df)], axis=1)


# ---------------------------------------------------------------------------
# Fight-duration history (for the rounds / over-under models). Kept out of FEATURES so
# method_v2's feature set is unchanged; scripts/rounds_wf.py attaches these separately.
# ---------------------------------------------------------------------------

TIMING_FEATURES = ("t_early_rate", "t_mid_rate", "t_avg_minutes", "t_r1_fin_share",
                   "pro_t_r1_rate", "pro_t_early_rate")
T_EARLY, T_MID = 7.5, 12.5   # the O/U 1.5 and 2.5 lines, in fight minutes
TA = 4.0                     # pseudo-fights for the duration shares


def timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per fighter, over bouts strictly before this one (draws count; NC/DQ/overturned do not):
      t_early_rate    share of their fights that ended before 7:30 (under 1.5)
      t_mid_rate      share that ended before 12:30 (under 2.5)
      t_avg_minutes   mean fight length, 3-round-equivalent (capped at 15)
      t_r1_fin_share  share of their finishes (won or lost) that came in round 1
      pro_t_r1_rate / pro_t_early_rate  the same from every Sherdog pro bout (NaN if unlinked)
    Shrunk toward pre-2015 population rates."""
    d = df.assign(_date=pd.to_datetime(df["date"]).dt.date)
    fights = d.drop_duplicates("fight_id")
    played = fights[fights["method"].fillna("") != ""]

    def minutes(r):
        return _num(r["fight_time_seconds"]) / 60.0

    def usable(r):
        return classify_outcome(r["method"], r["details"], r["winner_id"]) != "void"

    early = played[(played["_date"] < PRIOR_CUTOFF)]
    early = early[early.apply(usable, axis=1)]
    em = early.apply(minutes, axis=1)
    fin = early.apply(lambda r: method_class(r["method"], r["details"], r["winner_id"]) in ("ko", "sub"), axis=1)
    pri = {"early": float((em < T_EARLY).mean()), "mid": float((em < T_MID).mean()),
           "avg": float(em.clip(upper=15).mean()),
           "r1": float((em[fin] <= 5.0).mean()) if fin.any() else 0.5}
    log.info(f"  timing priors (pre-2015): under 1.5 {pri['early']:.3f}, under 2.5 {pri['mid']:.3f}, "
             f"mean {pri['avg']:.1f} min, R1 share of finishes {pri['r1']:.3f}")

    state = defaultdict(lambda: dict(n=0, early=0, mid=0, mins=0.0, fin=0, fin_r1=0))
    out = np.full((len(d), 4), np.nan)
    pos = {ix: k for k, ix in enumerate(d.index)}
    for fid, g in d.sort_values(["_date", "fight_id"], kind="stable").groupby("fight_id", sort=False):
        for ix, r in g.iterrows():
            c = state[r["stats_fighter_id"]]
            out[pos[ix]] = [(c["early"] + TA * pri["early"]) / (c["n"] + TA),
                            (c["mid"] + TA * pri["mid"]) / (c["n"] + TA),
                            (c["mins"] + TA * pri["avg"]) / (c["n"] + TA),
                            (c["fin_r1"] + TA * pri["r1"]) / (c["fin"] + TA)]
        first = g.iloc[0]
        if not (first["method"] or "") or not usable(first):
            continue
        t = minutes(first)
        is_fin = method_class(first["method"], first["details"], first["winner_id"]) in ("ko", "sub")
        for ix, r in g.iterrows():
            c = state[r["stats_fighter_id"]]
            c["n"] += 1; c["early"] += t < T_EARLY; c["mid"] += t < T_MID; c["mins"] += min(t, 15.0)
            if is_fin:
                c["fin"] += 1; c["fin_r1"] += t <= 5.0
    res = pd.DataFrame(out, columns=TIMING_FEATURES[:4], index=d.index)
    return pd.concat([res, _pro_timing(df)], axis=1)


def _pro_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Sherdog: share of pro bouts ending in round 1, and before 7:30 (R1, or R2 by 2:30)."""
    from sqlalchemy import inspect

    from app.database import SessionLocal, engine
    from app.models.ufc import SherdogBout, SherdogFighter

    cols = ["pro_t_r1_rate", "pro_t_early_rate"]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    if not inspect(engine).has_table(SherdogBout.__table__.name, schema=SherdogBout.__table__.schema):
        return out
    db = SessionLocal()
    try:
        link = {u: s for s, u in db.query(SherdogFighter.sherdog_id, SherdogFighter.ufc_fighter_id)
                if u is not None}
        rows = db.query(SherdogBout.fighter_sherdog_id, SherdogBout.date, SherdogBout.method_class,
                        SherdogBout.round, SherdogBout.time).filter(SherdogBout.date.isnot(None)).all()
    finally:
        db.close()

    def clock(s):
        try:
            m, sec = str(s).split(":")
            return int(m) + int(sec) / 60.0
        except (ValueError, AttributeError):
            return None
    per = defaultdict(list)
    for sid, d, mc, rnd, tm in rows:
        if mc not in ("KO/TKO", "SUB", "DEC", "DRAW") or rnd is None:
            continue
        fin = mc in ("KO/TKO", "SUB")
        c = clock(tm)
        r1 = fin and rnd == 1
        early = fin and (rnd == 1 or (rnd == 2 and c is not None and c <= 2.5))
        per[sid].append((d, r1, early))
    tot = [x for v in per.values() for x in v]
    p_r1 = sum(x[1] for x in tot) / max(len(tot), 1)
    p_early = sum(x[2] for x in tot) / max(len(tot), 1)
    hist = {}
    for sid, v in per.items():
        v.sort(key=lambda x: x[0])
        dates, cum = [], []
        n = r1 = e = 0
        for d, a, b in v:
            n += 1; r1 += a; e += b
            dates.append(d); cum.append((n, r1, e))
        hist[sid] = (dates, cum)
    vals = []
    for fid, d in zip(df["stats_fighter_id"], pd.to_datetime(df["date"]).dt.date):
        sid = link.get(fid)
        if sid is None:
            vals.append((np.nan, np.nan)); continue
        dates, cum = hist.get(sid, ([], []))
        k = bisect_left(dates, d)
        n, r1, e = cum[k - 1] if k else (0, 0, 0)
        vals.append(((r1 + TA * p_r1) / (n + TA), (e + TA * p_early) / (n + TA)))
    out.loc[:, :] = np.array(vals, dtype=float)
    return out
