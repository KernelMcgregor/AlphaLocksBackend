"""Round-by-round pace and fight-context features for the duration (survival) model.

Pace, per fighter, from every PRIOR bout fought with 5-minute rounds (other formats such as
10-5, 3-3-3 or 12-3 are skipped; their rounds are not comparable). Rates are per minute of
time actually fought in that round: full rounds count 5 minutes, the round a fight ended
in counts only up to the finish. Each rate is shrunk toward the pre-2015 population rate
for that round with PACE_PRIOR_MIN pseudo-minutes.

  pc_att_r1, pc_att_r2, pc_att_r3   significant strikes attempted per minute (pace)
  pc_fade                           (R2+R3 pace) / R1 pace; < 1 = slows down (cardio)
  pc_land_r1, pc_land_late          sig. strikes landed per minute, R1 vs R2-R3
  pc_abs_r1, pc_abs_late            sig. strikes absorbed per minute, R1 vs R2-R3
  pc_kd_r1, pc_kd_late              knockdowns landed per minute (power early vs late)
  pc_kdabs_r1, pc_kdabs_late        knockdowns suffered per minute (chin early vs late)
  pc_td_r1, pc_td_late              takedown attempts per minute
  pc_ctrl_r1, pc_ctrl_late          share of round time in control
  pc_att_r45                        pace in championship rounds (4-5)
  pc_r45_minutes                    minutes fought in rounds 4-5 (five-round experience)
  pc_five_fights                    prior bouts scheduled for five rounds

Context
  sn_replacement, sn_opp_changed    short notice (short_notice.py): this fighter took the
                                    fight late / this fighter's opponent changed
  fight_altitude_m, fight_high_altitude   event altitude (Mexico City, Denver, ...)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

log = logging.getLogger("round_features")

PACE_FEATURES = ("pc_att_r1", "pc_att_r2", "pc_att_r3", "pc_fade", "pc_land_r1", "pc_land_late",
                 "pc_abs_r1", "pc_abs_late", "pc_kd_r1", "pc_kd_late", "pc_kdabs_r1",
                 "pc_kdabs_late", "pc_td_r1", "pc_td_late", "pc_ctrl_r1", "pc_ctrl_late",
                 "pc_att_r45", "pc_r45_minutes", "pc_five_fights")
PACE_PRIOR_MIN = 15.0
PRIOR_CUTOFF = date(2015, 1, 1)
# Event altitude, metres (cities that have hosted UFC events at elevation).
ALTITUDE = {"mexico city": 2240, "bogot": 2640, "quito": 2850, "denver": 1609, "broomfield": 1640,
            "albuquerque": 1619, "salt lake": 1288, "johannesburg": 1753, "calgary": 1045,
            "colorado springs": 1839}
_STATS = ("att", "land", "abs", "kd", "kdabs", "td", "ctrl")


def _five_minute_rounds(time_format) -> bool:
    tf = str(time_format or "")
    return bool(tf) and all(p == "5" for p in tf.split("-"))


def _num(x) -> float:
    return float(x) if x is not None and x == x else 0.0


def pace_features(df: pd.DataFrame, round_data: pd.DataFrame) -> pd.DataFrame:
    """df: load_fight_data frame; round_data: its per-round stats. Aligned to df.index."""
    d = df.assign(_date=pd.to_datetime(df["date"]).dt.date)
    rs = {}
    for r in round_data.itertuples(index=False):
        rs[(r.fight_id, r.stats_fighter_id, int(r.round_number))] = r
    opp = {}
    for fid, g in d.groupby("fight_id"):
        if len(g) == 2:
            a, b = g["stats_fighter_id"].tolist()
            opp[(fid, a)], opp[(fid, b)] = b, a

    def bout_rounds(row):
        """[(round, minutes fought, {stat: value})] for one fighter in one bout."""
        if not _five_minute_rounds(row["time_format"]):
            return []
        total = _num(row["fight_time_seconds"]) / 60.0
        if total <= 0:
            return []
        last = int(np.ceil(round(total / 5.0, 6)))
        fid, me = row["fight_id"], row["stats_fighter_id"]
        o = opp.get((fid, me))
        out = []
        for r in range(1, last + 1):
            mine, theirs = rs.get((fid, me, r)), rs.get((fid, o, r)) if o is not None else None
            if mine is None:
                continue
            mins = 5.0 if r < last else total - 5.0 * (last - 1)
            if mins <= 0.05:
                continue
            out.append((r, mins, {
                "att": _num(mine.r_sig_str_attempted), "land": _num(mine.r_sig_str_landed),
                "abs": _num(theirs.r_sig_str_landed) if theirs is not None else 0.0,
                "kd": _num(mine.r_kd), "kdabs": _num(theirs.r_kd) if theirs is not None else 0.0,
                "td": _num(mine.r_td_attempted), "ctrl": _num(mine.r_ctrl_seconds) / 60.0}))
        return out

    groups = ("r1", "r2", "r3", "late", "r45")

    def group_of(r):
        return ["r1"] if r == 1 else (["r2", "late"] if r == 2 else ["r3", "late"] if r == 3 else ["r45"])

    # population per-minute rates by round group, pre-2015
    pop = {g: defaultdict(float) for g in groups}
    for _, row in d[d["_date"] < PRIOR_CUTOFF].iterrows():
        for r, mins, st in bout_rounds(row):
            for g in group_of(r):
                pop[g]["min"] += mins
                for k in _STATS:
                    pop[g][k] += st[k]
    rate0 = {g: {k: pop[g][k] / max(pop[g]["min"], 1.0) for k in _STATS} for g in groups}
    log.info("  pace priors (pre-2015, sig. attempted/min): " +
             ", ".join(f"{g} {rate0[g]['att']:.2f}" for g in groups))

    state = defaultdict(lambda: {g: defaultdict(float) for g in groups} | {"five": 0})
    out = np.full((len(d), len(PACE_FEATURES)), np.nan)
    pos = {ix: k for k, ix in enumerate(d.index)}

    def rate(c, g, k):
        return (c[g][k] + PACE_PRIOR_MIN * rate0[g][k]) / (c[g]["min"] + PACE_PRIOR_MIN)

    for fid, g in d.sort_values(["_date", "fight_id"], kind="stable").groupby("fight_id", sort=False):
        for ix, row in g.iterrows():
            c = state[row["stats_fighter_id"]]
            r1, late = rate(c, "r1", "att"), rate(c, "late", "att")
            out[pos[ix]] = [
                r1, rate(c, "r2", "att"), rate(c, "r3", "att"), late / max(r1, 1e-6),
                rate(c, "r1", "land"), rate(c, "late", "land"),
                rate(c, "r1", "abs"), rate(c, "late", "abs"),
                rate(c, "r1", "kd"), rate(c, "late", "kd"),
                rate(c, "r1", "kdabs"), rate(c, "late", "kdabs"),
                rate(c, "r1", "td"), rate(c, "late", "td"),
                rate(c, "r1", "ctrl"), rate(c, "late", "ctrl"),
                rate(c, "r45", "att"), c["r45"]["min"], c["five"]]
        if not (g.iloc[0]["method"] or ""):
            continue  # upcoming bout
        for ix, row in g.iterrows():
            c = state[row["stats_fighter_id"]]
            if _num(row.get("max_fight_time_seconds")) >= 1500:
                c["five"] += 1
            for r, mins, st in bout_rounds(row):
                for grp in group_of(r):
                    c[grp]["min"] += mins
                    for k in _STATS:
                        c[grp][k] += st[k]
    return pd.DataFrame(out, columns=PACE_FEATURES, index=d.index)


def short_notice_features(df: pd.DataFrame) -> pd.DataFrame:
    from app.services.ufc import short_notice
    out = short_notice.compute(df)
    if out is None:
        return pd.DataFrame({"sn_replacement": np.nan, "sn_opp_changed": np.nan}, index=df.index)
    return out


def altitude_by_fight() -> pd.DataFrame:
    """fight_id -> fight_altitude_m, fight_high_altitude (>= 1,000 m)."""
    from app.database import SessionLocal
    from app.models.ufc import UFCEvent, UFCFight
    db = SessionLocal()
    try:
        rows = db.query(UFCFight.id, UFCEvent.location).join(UFCEvent, UFCEvent.id == UFCFight.event_id).all()
    finally:
        db.close()

    def alt(loc):
        loc = str(loc or "").lower()
        return next((m for k, m in ALTITUDE.items() if k in loc), 0)
    out = pd.DataFrame([(f, alt(l)) for f, l in rows], columns=["fight_id", "fight_altitude_m"])
    out["fight_high_altitude"] = (out["fight_altitude_m"] >= 1000).astype(float)
    return out
