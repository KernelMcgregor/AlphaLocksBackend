"""Tug-of-war / control mechanics as winner features, as of before each fight.

Two column groups, tested as separate arms:

tug_st_*  (C0) the Stage-1 dictation states as they are (app/services/ufc/dictation.py),
    snapshotted before every bout: preference p and dictation skill s for ground share,
    clinch share and pace (TugOfWar, direction rule h and step k_s tuned on 2012-2018 only),
    and the control-mechanics offsets (shoot/deter, tdo/tdd, top/esc, ctl/ctld). Plus the
    matchup's predicted control share for this fighter.

tug_br_*  (C3) the bridge: WHERE this fight is likely to be fought x what each fighter does
    THERE. The winner model has habits and ratings, but not their product, which a boosted
    model on ~6k fights struggles to learn from raw inputs.
      * position mix: the mechanics chain's expected share of time with A on top, B on top,
        and standing (dictation.occupancy) for this pairing;
      * position-specific output, opponent-adjusted: landed significant strikes at distance
        and in the clinch per standing minute, knockdowns per standing minute, ground strikes
        landed per ground minute, submission attempts per minute (prior BRIDGE_LAMBDA fights). Each fitted as
        E[count] = mu * exposure * a_fighter * b_opponent (style_habits.fit_opponent_model,
        refit every quarter on bouts before it), then evaluated against THIS opponent's b.
    Features (A minus B margins, per minute):
      tug_br_stand    standing share x (distance + clinch landed margin)
      tug_br_kd       standing share x knockdown margin
      tug_br_gnp      ground share x (A's - B's ground strikes landed per ground minute)
      tug_br_sub      submission-attempt margin
      tug_br_edge     stand + gnp + KD_WEIGHT x kd   (KD_WEIGHT fixed at 10 in advance)
      tug_br_top      A's expected top-control share against B

League means come from pre-2012 rounds and tuning from 2012-2018, both before the
walk-forward window (~2021 on).
"""
from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from datetime import date, datetime

import numpy as np
import pandas as pd

from app.services.ufc import style_habits
from app.services.ufc.dictation import MechParams, MechRound, Mechanics, TugOfWar, TugParams, occupancy

log = logging.getLogger("tug_pit")

PREFIX = "tug_"
FIGHT_LEVEL = ()
RAW = ()

TUNE_START, TUNE_END = pd.Timestamp("2012-01-01"), pd.Timestamp("2019-01-01")
FIRST_BOUNDARY = date(2008, 1, 1)
KD_WEIGHT = 10.0
#: prior for the bridge fits, in average fights: style's 0.25 is far too weak for rare events
#: (knockdowns) over short exposures (a 20-second KO gave 15x the league KD rate)
BRIDGE_LAMBDA = 2.0
H_GRID = (-12, -6, -3, 0, 3)
S_GRID = (0.4, 1.6, 6.4)

#: position-specific output habits for the bridge: (count key, exposure key, scale)
BRIDGE_HABITS = {
    "dist_l": ("dist_l", "standing_min", 1.0),
    "clinch_l": ("clinch_l", "standing_min", 1.0),
    "kd": ("kd", "standing_min", 1.0),
    "gnp_l": ("ground_l", "ground_min", 1.0),
    "sub": ("sub_att", "minutes", 1.0),
}


def _d(x) -> date:
    return x.date() if isinstance(x, (datetime, pd.Timestamp)) else x


# ---------------------------------------------------------------------------
# C0: Stage-1 states
# ---------------------------------------------------------------------------

def _dims(rounds: pd.DataFrame) -> dict:
    """Same dimensions as scripts/dictation_eval.add_dimensions."""
    from scripts.dictation_eval import add_dimensions
    return add_dimensions(rounds)


def _run_tug(rounds, z, wt, league, prm):
    """Pre-bout (p, s) for both corners per bout, and the weighted error on the tune window."""
    m = TugOfWar(league, prm)
    snaps = {}
    rnd = rounds["round"].to_numpy()
    tune = ((rounds["date"] >= TUNE_START) & (rounds["date"] < TUNE_END)).to_numpy()
    err = wsum = 0.0
    for fid, g in rounds.groupby("fight_id", sort=False):
        ix = g.index.to_numpy()
        a, b = g["red"].iat[0], g["blue"].iat[0]
        pa, sa, _, _ = m.snapshot(a)
        pb, sb, _, _ = m.snapshot(b)
        snaps[fid] = (pa, sa, pb, sb)
        for i in ix:
            if tune[i]:
                e = z[i] - m.predict(a, b, int(rnd[i]))
                err += wt[i] * e * e
                wsum += wt[i]
        m.update(a, b, [(int(rnd[i]), z[i], wt[i]) for i in ix])
    return snaps, err / max(wsum, 1e-9)


def state_features(rounds: pd.DataFrame) -> tuple[dict, dict]:
    """{(fight_id, fighter): {col: value}} for C0, and per-bout mechanics shares (used by C3)."""
    rounds = rounds.sort_values(["date", "fight_id", "round"]).reset_index(drop=True)
    out: dict = defaultdict(dict)
    for k, (z, wt, league) in _dims(rounds).items():
        h = min(H_GRID, key=lambda h: _run_tug(rounds, z, wt, league, TugParams(learn_s=False, h=h))[1])
        ks = min(S_GRID, key=lambda s: _run_tug(rounds, z, wt, league, TugParams(h=h, k_s=s))[1])
        snaps, _ = _run_tug(rounds, z, wt, league, TugParams(h=h, k_s=ks))
        log.info("  tug %s: h=%s k_s=%s", k, h, ks)
        for fid, g in rounds.groupby("fight_id", sort=False):
            pa, sa, pb, sb = snaps[fid]
            a, b = g["red"].iat[0], g["blue"].iat[0]
            out[(fid, a)].update({f"tug_st_{k}_p": pa, f"tug_st_{k}_s": sa})
            out[(fid, b)].update({f"tug_st_{k}_p": pb, f"tug_st_{k}_s": sb})

    mech = Mechanics(MechParams())
    early = rounds["date"] < TUNE_START
    halves = [(MechRound(r.secs, r.r_td_attempted, r.r_td_landed, r.r_ctrl_seconds, r.b_ctrl_seconds),
               MechRound(r.secs, r.b_td_attempted, r.b_td_landed, r.b_ctrl_seconds, r.r_ctrl_seconds))
              for r in rounds.itertuples(index=False)]
    mech.init_league([h for i, pair in enumerate(halves) if early.iat[i] for h in pair])
    shares = {}
    for fid, g in rounds.groupby("fight_id", sort=False):
        a, b = g["red"].iat[0], g["blue"].iat[0]
        top_a, top_b = mech.predict_share(a, b, 300.0)
        shares[fid] = (top_a, top_b)
        for me, other, top in ((a, b, top_a), (b, a, top_b)):
            st = mech.st[me]
            out[(fid, me)].update({f"tug_st_m_{key}": st[key] for key in
                                   ("shoot", "deter", "tdo", "tdd", "top", "esc", "ctl", "ctld")})
            out[(fid, me)]["tug_st_ctrl_pred"] = top
        for i in g.index:
            mech.update(a, b, halves[i][0])
            mech.update(b, a, halves[i][1])
    return out, shares


# ---------------------------------------------------------------------------
# C3: the bridge
# ---------------------------------------------------------------------------

def load_position_bouts(db) -> dict[int, list[style_habits.Bout]]:
    """Fight totals with landed strikes by position, as style_habits.Bout lists."""
    from sqlalchemy import text
    from app.models.ufc import UFC_SCHEMA
    schema = f"{UFC_SCHEMA}." if UFC_SCHEMA else ""
    fights = db.execute(text(
        f"SELECT f.id, f.red_fighter_id, f.blue_fighter_id, f.winner_id, f.fight_time_seconds, "
        f"COALESCE(f.date, e.date) AS date FROM {schema}ufc_fights f JOIN {schema}ufc_events e ON e.id = f.event_id "
        f"WHERE f.fight_time_seconds > 0")).fetchall()
    rows = db.execute(text(
        "SELECT fight_id, fighter_id, kd, sub_att, ctrl_seconds, distance_landed, clinch_landed, ground_landed "
        f"FROM {schema}ufc_fight_stats WHERE round_number = 0")).fetchall()
    stats = {(r.fight_id, r.fighter_id): r for r in rows}
    out = defaultdict(list)
    g = lambda r, k: float(getattr(r, k) or 0) if r is not None else 0.0  # noqa: E731
    for f in fights:
        for me, opp in ((f.red_fighter_id, f.blue_fighter_id), (f.blue_fighter_id, f.red_fighter_id)):
            s, t = stats.get((f.id, me)), stats.get((f.id, opp))
            if s is None or f.date is None:
                continue
            minutes = f.fight_time_seconds / 60.0
            ctrl, octrl = g(s, "ctrl_seconds") / 60.0, g(t, "ctrl_seconds") / 60.0
            c = {"minutes": minutes, "ctrl_min": min(ctrl, minutes), "opp_ctrl_min": octrl,
                 "ground_min": min(ctrl + octrl, minutes),
                 "standing_min": max(minutes - ctrl - octrl, 0.25),
                 "dist_l": g(s, "distance_landed"), "clinch_l": g(s, "clinch_landed"),
                 "ground_l": g(s, "ground_landed"), "kd": g(s, "kd"), "sub_att": g(s, "sub_att")}
            won = None if f.winner_id is None else f.winner_id == me
            out[int(me)].append(style_habits.Bout(int(f.id), int(me), int(opp), _d(f.date), won, "",
                                                  float(f.fight_time_seconds), c, {}))
    for bs in out.values():
        bs.sort(key=lambda b: b.date)
    return out


def bridge_features(bouts: dict[int, list[style_habits.Bout]], fights: pd.DataFrame,
                    shares: dict) -> dict:
    """fights: fight_id, red, blue, date. shares: {fight_id: (top_red, top_blue)}."""
    flat = sorted((b for bs in bouts.values() for b in bs), key=lambda b: b.date)
    flat_dates = [b.date for b in flat]
    last = flat_dates[-1]
    quarters, d = [], FIRST_BOUNDARY
    while d <= last:
        quarters.append(d)
        m = d.month + 2
        d = date(d.year + m // 12, m % 12 + 1, 1)
    models = {}
    saved = style_habits.LAMBDA_FIGHTS
    style_habits.LAMBDA_FIGHTS = BRIDGE_LAMBDA
    try:
        for q in quarters:
            by_f = defaultdict(list)
            for b in flat[:bisect.bisect_left(flat_dates, q)]:
                by_f[b.fighter].append(b)
            models[q] = style_habits.fit_opponent_model(by_f, BRIDGE_HABITS) if by_f else None
    finally:
        style_habits.LAMBDA_FIGHTS = saved
    dates = {f: [b.date for b in bs] for f, bs in bouts.items()}

    def rates(f, opp, d, model):
        n = bisect.bisect_left(dates.get(f, []), d)
        if n < 1:
            return None
        r = style_habits.habit_rates(bouts[f][:n], model, BRIDGE_HABITS)
        return {h: r[h] * model.b[h].get(opp, 1.0) for h in BRIDGE_HABITS}

    out = {}
    for row in fights.itertuples(index=False):
        d = _d(row.date)
        i = bisect.bisect_right(quarters, d) - 1
        model = models[quarters[i]] if i >= 0 else None
        if model is None or row.fight_id not in shares:
            continue
        ra, rb = rates(row.red, row.blue, d, model), rates(row.blue, row.red, d, model)
        if ra is None or rb is None:
            continue
        top_r, top_b = shares[row.fight_id]
        stand = max(1.0 - top_r - top_b, 0.05)
        m_stand = stand * (ra["dist_l"] + ra["clinch_l"] - rb["dist_l"] - rb["clinch_l"])
        m_kd = stand * (ra["kd"] - rb["kd"])
        # ground output per minute of ground time, times how much of the fight is on the ground
        m_gnp = (top_r + top_b) * (ra["gnp_l"] - rb["gnp_l"])
        m_sub = ra["sub"] - rb["sub"]
        red = {"tug_br_stand": m_stand, "tug_br_kd": m_kd, "tug_br_gnp": m_gnp, "tug_br_sub": m_sub,
               "tug_br_edge": m_stand + m_gnp + KD_WEIGHT * m_kd, "tug_br_top": top_r}
        out[(row.fight_id, row.red)] = red
        out[(row.fight_id, row.blue)] = {k: (top_b if k == "tug_br_top" else -v) for k, v in red.items()}
    return out


def compute(ctx) -> pd.DataFrame:
    import scripts.dictation_eval as de
    from scripts.feature_wf import OUT_DIR
    # own copy of the per-round frame (5-minute rounds, string ids), so the research cache in
    # models/ufc/dictation is never overwritten; rebuilt from prod (read-only) when missing
    de.FRAME = OUT_DIR / "tug_rounds.pkl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rounds = de.load_rounds(not de.FRAME.exists())
    for c in ("fight_id", "red", "blue"):
        rounds[c] = rounds[c].astype("int64")
    state, shares = state_features(rounds)
    db = ctx.session()
    try:
        bouts = load_position_bouts(db)
    finally:
        db.close()
    fights = (ctx.fighters[ctx.fighters.corner == "red"][["fight_id", "red_fighter_id", "blue_fighter_id", "date"]]
              .rename(columns={"red_fighter_id": "red", "blue_fighter_id": "blue"}))
    bridge = bridge_features(bouts, fights, shares)
    keys = set(state) | set(bridge)
    rows = [{"fight_id": int(f), "fighter_id": int(p), **state.get((f, p), {}), **bridge.get((f, p), {})}
            for f, p in keys]
    return pd.DataFrame(rows)
