"""Opponent-adjusted fighting habits — the "who fights like X" fingerprint (space v5).

A habit here is a choice a fighter makes: how many strikes they throw, how often they
shoot, where they aim, where the fight goes. Counted raw, a habit is confounded by who the
fighter fought: a natural wrestler shoots less against an elite wrestler, a volume striker
throws less against a counter-puncher. So every habit is fitted as a two-way
multiplicative model over all fights at once, the same idea as KenPom's adjusted tempo:

    E[count] = mu * exposure * a_fighter * b_opponent

`a` is the fighter's habit against an average opponent; `b` is how much more or less of
that habit the opponent usually lets people show. Both carry a gamma-Poisson prior of
LAMBDA_FIGHTS average fights at the league rate.

Everything is ATTEMPTS — what a fighter tries, not what lands. Accuracy, defence and
results are skill and live in `efficacy_features`, which is DISPLAY ONLY since v5.

V5 FEATURES (`style_features`), measured 2026-10-18 with app/services/ufc/style_eval.py:
  * The head/body/leg and distance/clinch/ground mixes are built from opponent-adjusted
    PER-MINUTE counts, then closed and centred-log-ratio transformed. Adjusting the shares
    themselves (v4) broke their sum-to-one and counted each mix 1.5x.
  * Rare events get log(rate + RARE_FLOOR * league rate). Without the floor, a fighter who
    never attempts a submission is placed by how many minutes they have fought.
  * Submission attempts are per fight minute, not per "ground minute" (v4's ground minutes
    were both fighters' control time, so a pure striker's sub rate was the league mean).
  * New habits from columns ufcstats already gives: own control and time controlled (both
    opponent-adjusted), non-significant strike share (clinch/ground pitter-patter), ground
    strikes per minute of own control.
  * Tested negative: reversals and round-3/round-1 pace fade (noise), recency weighting
    with a 3-year half-life (hurt recognition).

ADAPTABILITY (fight-to-fight departure from the model's prediction) is NOT SHIPPED: a
fighter's odd-fight and even-fight values correlate only 0.20.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import text

from app.models.ufc import UFC_SCHEMA

#: Prior strength, in average fights at the league rate. Tuned 2026-10-18 on the tuning half
#: of fighters (0.25 / 0.5 / 1: v5 odd-even found-first 10.9% / 10.4% / 9.4%).
LAMBDA_FIGHTS = 0.25
#: Floor added before logging a rate, as a fraction of the league rate.
RARE_FLOOR = 0.25
#: Alternating-fit iterations; converges well inside this.
N_ITER = 30

#: habit -> (count key, exposure key, display scale). Rate = scale * count / exposure.
HABITS: dict[str, tuple[str, str, float]] = {
    "str_pm": ("str_att", "minutes", 1.0),               # sig strikes thrown per minute
    "head_pm": ("head_att", "minutes", 1.0),             # by target ...
    "body_pm": ("body_att", "minutes", 1.0),
    "leg_pm": ("leg_att", "minutes", 1.0),
    "dist_pm": ("dist_att", "minutes", 1.0),             # ... and by position
    "clinch_pm": ("clinch_att", "minutes", 1.0),
    "ground_pm": ("ground_att", "minutes", 1.0),
    "td_s": ("td_att", "standing_min", 15.0),            # takedowns attempted per 15 standing min
    "sub_pm": ("sub_att", "minutes", 15.0),              # submission attempts per 15 min
    "ctrl": ("ctrl_min", "minutes", 1.0),                # share of fight time in control
    "ctrld": ("opp_ctrl_min", "minutes", 1.0),           # share of fight time controlled
    "nonsig_pm": ("nonsig_att", "minutes", 1.0),         # non-significant strikes thrown
    "gnp_c": ("ground_att", "ctrl_min", 1.0),            # ground strikes per minute of own control
}

#: The retrieval space's columns, grouped into concepts. Each block carries equal weight
#: (a block's columns are scaled by 1/sqrt(its degrees of freedom)); the two mixes have 2
#: degrees of freedom in 3 columns. Keys reuse the display vocabulary where the concept is
#: the same, so labels on the site carry over.
FEATURE_BLOCKS: dict[str, list[str]] = {
    "pace": ["str_att_pm"],
    "target": ["head_att_share", "body_att_share", "leg_att_share"],
    "position": ["dist_att_share", "clinch_att_share", "ground_att_share"],
    "takedowns": ["td_att15s"],
    "submissions": ["sub_att15"],
    "nonsig": ["nonsig_share"],
    "control": ["ctrl_share", "ctrld_share"],
    "gnp": ["gnp_per_ctrl"],
}
BLOCK_DOF = {"target": 2, "position": 2, "control": 2}
STYLE_FEATURES = [c for cols in FEATURE_BLOCKS.values() for c in cols]

#: Habits whose fight-to-fight deviations define adaptability: the gameplan levers.
ADAPT_HABITS = ["str_pm", "td_s", "dist_pm", "clinch_pm", "ground_pm"]


@dataclass
class Bout:
    fight_id: int
    fighter: int
    opponent: int
    date: object
    won: bool | None
    method: str
    seconds: float
    c: dict = field(default_factory=dict)   # counts and exposures (own)
    o: dict = field(default_factory=dict)   # opponent's counts (for defence / absorbed)


def load_bouts(db) -> dict[int, list[Bout]]:
    """{fighter_id: [Bout, ...]} oldest first, over timed bouts with a stats row — the same
    fights career_stats_service aggregates."""
    schema = f"{UFC_SCHEMA}." if UFC_SCHEMA else ""
    fights = db.execute(text(
        f"SELECT id, red_fighter_id, blue_fighter_id, winner_id, method, fight_time_seconds, date "
        f"FROM {schema}ufc_fights WHERE fight_time_seconds IS NOT NULL AND fight_time_seconds > 0"
    )).fetchall()
    rows = db.execute(text(
        "SELECT fight_id, fighter_id, kd, sig_str_landed, sig_str_attempted, total_str_attempted, "
        "td_landed, td_attempted, "
        "sub_att, rev, ctrl_seconds, head_attempted, body_attempted, leg_attempted, "
        "distance_attempted, clinch_attempted, ground_attempted, ground_landed "
        f"FROM {schema}ufc_fight_stats WHERE round_number = 0"
    )).fetchall()
    stats = {(r.fight_id, r.fighter_id): r for r in rows}

    out: dict[int, list[Bout]] = defaultdict(list)
    for f in fights:
        for me, opp in ((f.red_fighter_id, f.blue_fighter_id), (f.blue_fighter_id, f.red_fighter_id)):
            s = stats.get((f.id, me))
            if s is None or me is None or opp is None:
                continue
            t = stats.get((f.id, opp))
            minutes = f.fight_time_seconds / 60.0
            ctrl_both = ((s.ctrl_seconds or 0) + ((t.ctrl_seconds or 0) if t else 0)) / 60.0
            ground_min = min(ctrl_both, minutes)
            g = lambda r, k: float(getattr(r, k) or 0) if r is not None else 0.0  # noqa: E731
            c = {
                "minutes": minutes, "standing_min": max(minutes - ground_min, 0.0), "ground_min": ground_min,
                "str_att": g(s, "sig_str_attempted"), "str_landed": g(s, "sig_str_landed"),
                "td_att": g(s, "td_attempted"), "td_landed": g(s, "td_landed"),
                "sub_att": g(s, "sub_att"), "rev": g(s, "rev"), "kd": g(s, "kd"),
                "ctrl_min": g(s, "ctrl_seconds") / 60.0,
                "head_att": g(s, "head_attempted"), "body_att": g(s, "body_attempted"), "leg_att": g(s, "leg_attempted"),
                "dist_att": g(s, "distance_attempted"), "clinch_att": g(s, "clinch_attempted"),
                "ground_att": g(s, "ground_attempted"), "ground_landed": g(s, "ground_landed"),
                "nonsig_att": max(g(s, "total_str_attempted") - g(s, "sig_str_attempted"), 0.0),
                "opp_ctrl_min": g(t, "ctrl_seconds") / 60.0,
            }
            c["target_att"] = c["head_att"] + c["body_att"] + c["leg_att"]
            c["position_att"] = c["dist_att"] + c["clinch_att"] + c["ground_att"]
            o = {"str_att": g(t, "sig_str_attempted"), "str_landed": g(t, "sig_str_landed"),
                 "td_att": g(t, "td_attempted"), "td_landed": g(t, "td_landed")}
            won = None if f.winner_id is None else (f.winner_id == me)
            out[int(me)].append(Bout(int(f.id), int(me), int(opp), f.date, won, (f.method or "").lower(),
                                     float(f.fight_time_seconds), c, o))
    for bouts in out.values():
        bouts.sort(key=lambda b: (b.date is None, b.date))
    return out


@dataclass
class OpponentModel:
    mu: dict[str, float]                 # league rate per habit
    lam: dict[str, float]                # prior, in count units
    a: dict[str, dict[int, float]]       # fighter effects
    b: dict[str, dict[int, float]]       # opponent "allow" effects


def fit_opponent_model(bouts: dict[int, list[Bout]], habits=HABITS) -> OpponentModel:
    """Alternating fit of E[count] = mu * exposure * a[fighter] * b[opponent] per habit."""
    flat = [x for bs in bouts.values() for x in bs]
    ids = sorted({x.fighter for x in flat} | {x.opponent for x in flat})
    pos = {f: i for i, f in enumerate(ids)}
    n = len(ids)
    fi = np.array([pos[x.fighter] for x in flat], dtype=int)
    oi = np.array([pos[x.opponent] for x in flat], dtype=int)
    seen = np.bincount(oi, minlength=n) > 0
    mu, lam, A, B = {}, {}, {}, {}
    for h, (ck, ek, _) in habits.items():
        c = np.array([x.c[ck] for x in flat], dtype=float)
        e = np.array([x.c[ek] for x in flat], dtype=float)
        m = c.sum() / e.sum() if e.sum() else 0.0
        lam_h = LAMBDA_FIGHTS * c.sum() / max(len(flat), 1)   # LAMBDA_FIGHTS average fights of count
        num_f, num_o = np.bincount(fi, c, n), np.bincount(oi, c, n)
        a, b = np.ones(n), np.ones(n)
        for _ in range(N_ITER):
            a = (num_f + lam_h) / (np.bincount(fi, m * e * b[oi], n) + lam_h)
            b = (num_o + lam_h) / (np.bincount(oi, m * e * a[fi], n) + lam_h)
            # Identify the scale: geometric mean of b over opponents is 1.
            if seen.any():
                b = b / math.exp(np.log(b[seen]).mean())
        mu[h], lam[h] = m, lam_h
        A[h] = {f: float(a[pos[f]]) for f in ids}
        B[h] = {f: float(b[pos[f]]) for f in ids if seen[pos[f]]}
    return OpponentModel(mu, lam, A, B)


def habit_rates(bouts: list[Bout], model: OpponentModel | None, habits=HABITS) -> dict[str, float]:
    """A fighter's habits over `bouts` against an average opponent (or raw, model=None),
    each shrunk toward the league rate by the model's prior."""
    out = {}
    for h, (ck, ek, scale) in habits.items():
        if model is None:
            num = sum(x.c[ck] for x in bouts)
            den = sum(x.c[ek] for x in bouts)
            out[h] = scale * num / den if den else 0.0
            continue
        m, lam_h, b = model.mu[h], model.lam[h], model.b[h]
        num = sum(x.c[ck] for x in bouts) + lam_h
        den = sum(m * x.c[ek] * b.get(x.opponent, 1.0) for x in bouts) + lam_h
        out[h] = scale * m * num / den
    return out


def style_features(bouts: list[Bout], model: OpponentModel) -> dict[str, float]:
    """The v5 retrieval-space columns (STYLE_FEATURES), before division scaling."""
    r = habit_rates(bouts, model)
    fl = lambda h: r[h] + RARE_FLOOR * HABITS[h][2] * model.mu[h]  # noqa: E731
    out = {"str_att_pm": math.log(max(r["str_pm"], 1e-9))}
    for keys, names in ((("head_pm", "body_pm", "leg_pm"), FEATURE_BLOCKS["target"]),
                        (("dist_pm", "clinch_pm", "ground_pm"), FEATURE_BLOCKS["position"])):
        logs = [math.log(fl(h)) for h in keys]
        centre = sum(logs) / len(logs)
        out.update({n: v - centre for n, v in zip(names, logs)})
    out["td_att15s"] = math.log(fl("td_s"))
    out["sub_att15"] = math.log(fl("sub_pm"))
    out["nonsig_share"] = math.log(fl("nonsig_pm") / max(r["str_pm"], 1e-9))
    out["ctrl_share"] = math.log(fl("ctrl"))
    out["ctrld_share"] = math.log(fl("ctrld"))
    out["gnp_per_ctrl"] = math.log(fl("gnp_c"))
    return out


def display_rates(bouts: list[Bout], model: OpponentModel) -> dict[str, float]:
    """The same habits in readable units (per minute, %, per 15), for percentiles on screen."""
    r = habit_rates(bouts, model)
    tgt = r["head_pm"] + r["body_pm"] + r["leg_pm"] or 1e-9
    pos = r["dist_pm"] + r["clinch_pm"] + r["ground_pm"] or 1e-9
    return {
        "str_att_pm": r["str_pm"],
        "head_att_share": r["head_pm"] / tgt, "body_att_share": r["body_pm"] / tgt,
        "leg_att_share": r["leg_pm"] / tgt,
        "dist_att_share": r["dist_pm"] / pos, "clinch_att_share": r["clinch_pm"] / pos,
        "ground_att_share": r["ground_pm"] / pos,
        "td_att15s": r["td_s"], "sub_att15": r["sub_pm"],
        "nonsig_share": r["nonsig_pm"] / (r["nonsig_pm"] + r["str_pm"] or 1e-9),
        "ctrl_share": r["ctrl"], "ctrld_share": r["ctrld"], "gnp_per_ctrl": r["gnp_c"],
    }


def efficacy_features(bouts: list[Bout]) -> dict[str, float]:
    """How well it works — skill, kept out of the habits fingerprint."""
    s = lambda k: sum(x.c[k] for x in bouts)  # noqa: E731
    so = lambda k: sum(x.o[k] for x in bouts)  # noqa: E731
    minutes = s("minutes") or 1e-9
    standing = s("standing_min") or 1e-9
    wins = [x for x in bouts if x.won]
    nw = len(wins) or 1
    return {
        "sig_acc": s("str_landed") / s("str_att") if s("str_att") else 0.0,
        "td_acc": s("td_landed") / s("td_att") if s("td_att") else 0.0,
        "sig_def": 1 - so("str_landed") / so("str_att") if so("str_att") else 0.0,
        "td_def": 1 - so("td_landed") / so("td_att") if so("td_att") else 1.0,
        "sapm": so("str_landed") / minutes,
        "kd15s": 15 * s("kd") / standing,
        "finish_rate": sum(1 for x in wins if "dec" not in x.method) / nw,
        "ko_win_share": sum(1 for x in wins if "ko" in x.method) / (len(bouts) or 1),
        "sub_win_share": sum(1 for x in wins if "sub" in x.method) / (len(bouts) or 1),
        "avg_fight_sec": sum(x.seconds for x in bouts) / (len(bouts) or 1),
    }


def adaptability(bouts: list[Bout], model: OpponentModel, min_bouts: int = 4) -> float | None:
    """Mean over ADAPT_HABITS of the exposure-weighted SD of log(actual / predicted) per
    fight — how much the gameplan moves beyond what opponents explain. None below
    `min_bouts` fights: a spread needs a few points."""
    if len(bouts) < min_bouts:
        return None
    sds = []
    for h in ADAPT_HABITS:
        ck, ek, _ = HABITS[h]
        a = model.a[h].get(bouts[0].fighter, 1.0)
        m, b = model.mu[h], model.b[h]
        pts = []
        for x in bouts:
            pred = m * x.c[ek] * a * b.get(x.opponent, 1.0)
            if x.c[ek] <= 0 or pred <= 0:
                continue
            pts.append((math.log((x.c[ck] + 0.5) / (pred + 0.5)), x.c[ek]))
        if len(pts) < min_bouts:
            continue
        w = sum(p[1] for p in pts)
        mean = sum(p[0] * p[1] for p in pts) / w
        sds.append(math.sqrt(sum(p[1] * (p[0] - mean) ** 2 for p in pts) / w))
    return sum(sds) / len(sds) if sds else None
