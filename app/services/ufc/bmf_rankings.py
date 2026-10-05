"""BMF rankings: violence times quality, across every division.

In the spirit of the UFC's "Baddest Motherf***er" belt rather than a pound-for-pound
list: who finishes people, who can't be put away, and whether they have done it against
anyone. Four components, each 0-100, recombined on the page with the viewer's weights:

  F  finishing   — KO/TKO and submission wins and knockdowns landed, each bout weighted
                   by its opponent's tier, as a multiple of the fighter's own division's
                   rate
  T  toughness   — knockdowns and KO losses per head strike absorbed (a chin rate) relative
                   to the division, plus credit for getting dropped and surviving
  Q  opposition  — mean opponent tier, credited by result (opp_quality_raw)
  R  activity    — absolute recency score

F, T and R average into a base; Q multiplies it as a gate:

    score = base * (GATE_FLOOR + (1 - GATE_FLOOR) * (Q/100) ** gamma)

A gate rather than a fourth additive term because the failure it prevents is specific: a
knockout artist who has only beaten tier-1 opposition. Added, a 99 in F buys back a 10 in
Q; multiplied, it cannot. gamma = 0 switches the gate off.

Run: python -m app.services.ufc.alt_rankings_publisher --preview
"""

from __future__ import annotations

import math

from app.services.ufc.alt_rankings_common import (
    BoutView, FighterView, bout_views, ledger_entry, opp_quality_raw, recency_score,
    safe_log, scale_quality,
)

# -- Finishing ------------------------------------------------------------------
KO_WIN_PTS, SUB_WIN_PTS, KD_PTS, KD_CAP = 1.0, 0.6, 0.35, 3
#: A round-1 KO is worth 30% more than a round-3-or-later one.
EARLY_FINISH_BONUS = 0.15
#: Pseudo-bouts at the division's mean, so two early KOs do not make a fighter #1.
F_PRIOR_BOUTS = 2.0

# -- Toughness ------------------------------------------------------------------
#: Pseudo head strikes absorbed at the division's chin rate (method_ratings' prior).
CHIN_PRIOR_HEAD = 150.0
KO_LOSS_KD_EQUIV = 2.0
COMEBACK_WEIGHT = 0.25
COMEBACK_PRIOR_BOUTS = 2.0

# -- Combination ----------------------------------------------------------------
GATE_FLOOR = 0.15
DEFAULT_WEIGHTS = {"finishing": 0.45, "toughness": 0.30, "recency": 0.25, "gamma": 0.5}

#: How far back the division baselines look.
BASELINE_YEARS = 10

#: Raw -> 0-100 scales. Fixed curves rather than percentiles, for the reason given in
#: `alt_rankings_common.scale_quality`: percentiles put every contender at 95-100.
#: Finishing is a multiple of the division rate (1.0 = average) -> 100*(1-exp(-x/F_SCALE)),
#: so average is ~49 and 3x average ~86, and it never clips. Toughness is a log-ratio
#: centred on 0 (= the division's chin) -> logistic with T_SCALE.
F_SCALE = 1.5
T_SCALE = 0.4

COMPONENTS = [
    {"key": "finishing", "label": "Finishing", "type": "weight",
     "help": "KO/TKO and submission wins plus knockdowns, each fight weighted by the "
             "opponent's tier and how recent it was. Measured against the fighter's own "
             "division, so heavyweights don't win on size."},
    {"key": "toughness", "label": "Toughness", "type": "weight",
     "help": "How rarely they get dropped or stopped per head strike absorbed, against the "
             "division norm, plus credit for getting knocked down and surviving."},
    {"key": "recency", "label": "Activity", "type": "weight",
     "help": "How recently and how often they have fought. Absolute, not a percentile."},
    {"key": "opp_quality", "label": "Opposition", "type": "gate",
     "help": "Mean opponent tier (1-10), counted even in losses, plus a bonus for wins over "
             "elite-tier opponents. Multiplies the score: the 'strictness' slider sets how "
             "hard it bites."},
]


def bout_finish_points(b: BoutView) -> float:
    pts = KD_PTS * min(b.kd, KD_CAP)
    if b.result == "W":
        if b.outcome in ("ko", "doctor"):
            r = b.finish_round or 3
            pts += KO_WIN_PTS * (1 + EARLY_FINISH_BONUS * (3 - min(r, 3)))
        elif b.outcome == "sub":
            pts += SUB_WIN_PTS
        # "injury" stoppages are not violence the winner dealt.
    return pts


def chin_events(b: BoutView) -> float:
    return b.kd_abs + (KO_LOSS_KD_EQUIV if b.result == "L" and b.outcome in ("ko", "doctor")
                       else 0.0)


def division_baselines(ctx: dict) -> dict[str, dict]:
    """Per division over the last BASELINE_YEARS: mean quality-weighted finish points per
    bout (`mu`) and chin events per head strike absorbed (`g`)."""
    from datetime import timedelta

    today = ctx["today"]
    start = today - timedelta(days=int(365.25 * BASELINE_YEARS))
    cutoff = today + timedelta(days=1)
    acc: dict[str, list[float]] = {}
    for fid in ctx["hist"]["bouts"]:
        for b in bout_views(ctx, fid, cutoff, n=None):
            if b.date < start:
                break                      # newest first: everything after is older
            if b.division == "unknown":
                continue
            a = acc.setdefault(b.division, [0.0, 0, 0.0, 0.0])
            a[0] += b.q * bout_finish_points(b)
            a[1] += 1
            a[2] += chin_events(b)
            a[3] += b.head_abs
    return {
        d: {"mu": (a[0] / a[1]) if a[1] else 0.5,
            "g": (a[2] / a[3]) if a[3] else 0.01}
        for d, a in acc.items()
    }


def component_raws(f: FighterView, base: dict) -> dict:
    """The fighter's raw F, T and Q before scaling. `base` is their division's."""
    mu, g = max(base["mu"], 1e-6), max(base["g"], 1e-6)
    bouts = f.bouts
    sw = sum(b.w for b in bouts)

    f_raw = (sum(b.w * b.q * bout_finish_points(b) for b in bouts) + F_PRIOR_BOUTS * mu) \
        / (sw + F_PRIOR_BOUTS)

    chin = (sum(b.w * chin_events(b) for b in bouts) + CHIN_PRIOR_HEAD * g) \
        / (sum(b.w * b.head_abs for b in bouts) + CHIN_PRIOR_HEAD)
    comebacks = sum(b.w for b in bouts if b.kd_abs >= 1 and (b.result == "W" or b.went_distance))
    comeback_rate = comebacks / (sw + COMEBACK_PRIOR_BOUTS)

    return {
        "finishing": f_raw / mu,
        "toughness": -safe_log(chin / g) + COMEBACK_WEIGHT * comeback_rate,
        "opp_quality": opp_quality_raw(bouts),
        "chin_rate": chin,
        "comeback_rate": comeback_rate,
    }


def scale_finishing(raw: float) -> float:
    return round(100.0 * (1.0 - math.exp(-max(raw, 0.0) / F_SCALE)), 2)


def scale_toughness(raw: float) -> float:
    return round(100.0 / (1.0 + math.exp(-raw / T_SCALE)), 2)


def gate(q: float, gamma: float) -> float:
    return GATE_FLOOR + (1 - GATE_FLOOR) * (max(q, 0.0) / 100.0) ** max(gamma, 0.0)


def combine(c: dict, w: dict | None = None) -> float:
    """Final BMF score from 0-100 components. Mirrored exactly by the page's JS."""
    w = {**DEFAULT_WEIGHTS, **(w or {})}
    tot = w["finishing"] + w["toughness"] + w["recency"]
    if tot <= 0:
        base = 0.0
    else:
        base = (w["finishing"] * c["finishing"] + w["toughness"] * c["toughness"]
                + w["recency"] * c["recency"]) / tot
    return base * gate(c["opp_quality"], w["gamma"])


def compute_bmf(ctx: dict, fighters: list[FighterView]) -> list[dict]:
    """Rows for every eligible fighter, both pools. Pure given `ctx`."""
    today = ctx["today"]
    names = ctx["hist"]["names"]
    baselines = division_baselines(ctx)
    fallback = {"mu": 0.5, "g": 0.01}

    raws = {f.fighter_id: component_raws(f, baselines.get(f.division, fallback))
            for f in fighters}
    rows = []
    for pool in ("men", "women"):
        members = [f for f in fighters if f.pool == pool]
        if not members:
            continue
        for f in members:
            r = raws[f.fighter_id]
            comps = {
                "finishing": scale_finishing(r["finishing"]),
                "toughness": scale_toughness(r["toughness"]),
                "opp_quality": scale_quality(r["opp_quality"]),
                "recency": round(recency_score((today - f.last_activity).days,
                                               f.bouts_last_730d), 2),
            }
            rows.append({
                "fighter_id": f.fighter_id, "pool": pool, "division": f.division,
                "is_champion": f.is_champion,
                "components": comps,
                "raw": {k: round(v, 4) for k, v in r.items()},
                "default_score": round(combine(comps), 3),
                "n_bouts": len(f.bouts),
                "last_fight_date": f.bouts[0].date,
                "ledger": [ledger_entry(b, names, f_pts=round(bout_finish_points(b), 3))
                           for b in f.bouts],
            })
    return rows
