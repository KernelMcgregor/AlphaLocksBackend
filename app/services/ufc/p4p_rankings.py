"""Pound-for-pound rankings, fitted to UFC's own P4P lists.

The usual objection to a cross-division P4P list is that divisions barely fight each
other, so one global rating cannot be trusted to put a flyweight and a heavyweight on one
scale (docs/models/rankings.md). The main components avoid needing one:

  dominance     P(fighter beats the median of their division's current top-10), from the
                prediction model's Elo. Size-neutral: both sides weigh the same.
  div_points    the division ranker's own points (absolute opponent tiers, so comparable
                across divisions) — Fight Matrix's "divisional normalisation" idea
  best_wins     mean opponent tier of their three best wins, slow (6-year) decay
  depth         how strong the division's top-10 is, standardised across the pool. The
                only cross-division term.
  champion      100 for a reigning champion
  title_fights  recency-weighted title-bout appearances, won or lost
  recency       absolute activity score

    score = sum(w_k * c_k) / sum(w_k)

Weights are FITTED, not chosen: `scripts/fit_p4p.py` orders fighters against 33 monthly
UFC.com P4P snapshots (2024-01 .. 2026-10), trained on 2024-25 and scored on 2026. Out
of time, after the division merge, it matches 80% of UFC's top 15 (the first hand-set
weights: 75%), Spearman 0.75 (0.67).

Consistency with the division rankings is a hard constraint: nobody is ranked above a
fighter their own division ranks higher (`division_merge`, isotonic regression per
division). Champions are division #1, so a champion is never below their own contender,
but they are not pinned to the top: a contender from a deep division can still outrank
another division's champion. UFC's own P4P list breaks its own division order in 10.6% of
same-division pairs (2024-26 snapshots), so this list is deliberately stricter than UFC's.
"""

from __future__ import annotations

import math
from datetime import date
from statistics import mean, median, pstdev

from app.services.ufc.alt_rankings_common import (
    FighterView, ledger_entry, opp_quality_raw, recency_score, scale_quality,
)

TOP_N = 10
#: Recency-weighted title-bout wins (or appearances) at which the component reaches ~63.
TITLE_SCALE = 1.5
STREAK_CAP = 6
DEPTH_SPREAD = 20.0          # 0-100 points per standard deviation of division depth
#: Fitted by scripts/fit_p4p.py (2026-10-05, train 2024-25 / test 2026, scored after the
#: division merge), rounded, with `champion` pinned at 0.05 (--fix champion=0.05). Left
#: free the fit takes 0.10 and puts all eight men's champions in the top eight.
DEFAULT_WEIGHTS = {"dominance": 0.29, "div_points": 0.21, "best_wins": 0.20,
                   "title_fights": 0.12, "depth": 0.10, "champion": 0.05, "recency": 0.03}

COMPONENTS = [
    {"key": "best_wins", "label": "Best wins", "type": "weight",
     "help": "How good their three best wins were (opponent tier at the time, 1-10), "
             "fading slowly over six years. Three wins over tier-10 opponents = 100."},
    {"key": "dominance", "label": "Dominance", "type": "weight",
     "help": "Chance they beat the median top-10 fighter in their own division, from the "
             "model's Elo. Compares a fighter only with fighters their size."},
    {"key": "div_points", "label": "Division points", "type": "weight",
     "help": "The division rankings' own score: results over the last six fights, weighted "
             "by opponent tier and recency. Opponent tiers are on one scale for every "
             "division, so these points compare across divisions."},
    {"key": "depth", "label": "Division depth", "type": "weight",
     "help": "How strong the division's top 10 is compared with the other divisions."},
    {"key": "champion", "label": "Champion", "type": "weight",
     "help": "100 for a reigning champion. Whatever the weights, nobody is ranked above a "
             "fighter their own division ranks higher — so no champion sits below their "
             "own contender."},
    {"key": "title_fights", "label": "Title fights", "type": "weight",
     "help": "Recent championship fights, won or lost — fighting at title level counts."},
    {"key": "recency", "label": "Activity", "type": "weight",
     "help": "How recently and how often they have fought. Absolute, not a percentile."},
]


# ---------------------------------------------------------------------------
# Elo — model.py's rating, without the training frame
# ---------------------------------------------------------------------------

def elo_ratings(ctx: dict, as_of: date, at_fight: dict | None = None) -> dict[int, float]:
    """Current Elo for every fighter, as of `as_of`.

    The same rating `model.build_features` computes (fitted K, Fight Matrix outcome
    scores, ~50/50 blend with a stats-based "deserved" probability), reproduced over the
    rows `load_context` already read so a ranking publish does not have to build the
    whole training DataFrame. The default (non-scorecard) path only. Parity with
    model.py is checked by tests/test_p4p_rankings.py on a fixture and was verified on
    prod at build time.

    model.py's frame keeps only round-0 stats rows with any recorded activity (the
    placeholder filter in load_fight_data), and a bout enters its Elo loop only through
    such a row; both are mirrored here.
    """
    from app.services.ufc.model import (
        ELO_DESERVED_COEF, ELO_K, ELO_OUTCOME_SCORES, ELO_W_RESULT,
    )

    totals = ctx["totals"]

    def _live(t):
        return t is not None and (t["sig_str_landed"] > 0 or t["sig_str_attempted"] > 0
                                  or t["td_attempted"] > 0)

    order = sorted(
        (row for row in ctx["fights"].values() if row[1] and row[1] <= as_of),
        key=lambda r: (r[1], r[0]),
    )
    elo: dict[int, float] = {}
    for fight_id, _d, red, blue, winner, _method, _details, secs in order:
        t_red, t_blue = totals.get((fight_id, red)), totals.get((fight_id, blue))
        red_live, blue_live = _live(t_red), _live(t_blue)
        if not (red_live or blue_live):
            continue
        otype = ctx["outcome"].get(fight_id, "void")
        if otype == "void":
            continue
        r_elo, b_elo = elo.get(red, 1500.0), elo.get(blue, 1500.0)
        if at_fight is not None:              # pre-fight ratings, for opponent quality
            at_fight[(fight_id, red)], at_fight[(fight_id, blue)] = r_elo, b_elo
        expected_r = 1 / (1 + 10 ** ((b_elo - r_elo) / 400))
        if otype == "draw":
            actual_r = 0.5
        else:
            s_win = ELO_OUTCOME_SCORES[otype]
            actual_r = s_win if winner == red else 1.0 - s_win
            deserved = None
            if red_live and secs is not None:
                minutes = max(float(secs) / 60.0, 0.5)
                opp = t_blue if blue_live else t_red
                z = sum(coef * (t_red[col] - opp[col]) / minutes
                        for col, coef in ELO_DESERVED_COEF.items())
                deserved = 1.0 / (1.0 + math.exp(-z))
            if deserved is not None:
                actual_r = ELO_W_RESULT * actual_r + (1 - ELO_W_RESULT) * deserved
        elo[red] = r_elo + ELO_K * (actual_r - expected_r)
        elo[blue] = b_elo - ELO_K * (actual_r - expected_r)
    return elo


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def win_prob(a: float, b: float) -> float:
    return 1 / (1 + 10 ** ((b - a) / 400))


def division_elites(elo: dict[int, float], fighters: list[FighterView]) -> dict[str, list[float]]:
    """Top-N Elo values per division among eligible fighters, best first."""
    by_div: dict[str, list[float]] = {}
    for f in fighters:
        by_div.setdefault(f.division, []).append(elo.get(f.fighter_id, 1500.0))
    return {d: sorted(v, reverse=True)[:TOP_N] for d, v in by_div.items()}


def dominance(rating: float, elite: list[float]) -> float:
    """0-100: P(beats the median of the division's top-N)."""
    return 100.0 * win_prob(rating, median(elite)) if elite else 50.0


def division_depth(elites: dict[str, list[float]]) -> dict[str, float]:
    """0-100 per division: 50 + DEPTH_SPREAD * z of the mean top-N Elo, within the pool."""
    means = {d: mean(v) for d, v in elites.items() if v}
    if len(means) < 2:
        return {d: 50.0 for d in means}
    mu, sd = mean(means.values()), pstdev(means.values()) or 1.0
    return {d: max(0.0, min(100.0, 50.0 + DEPTH_SPREAD * (m - mu) / sd))
            for d, m in means.items()}


def combine(c: dict, w: dict | None = None) -> float:
    """Final P4P score from 0-100 components. Mirrored exactly by the page's JS."""
    w = {**DEFAULT_WEIGHTS, **(w or {})}
    tot = sum(w[k] for k in DEFAULT_WEIGHTS)
    if tot <= 0:
        return 0.0
    return sum(w[k] * c[k] for k in DEFAULT_WEIGHTS) / tot


def _streak(bouts) -> int:
    n = 0
    for b in bouts:                      # newest first
        if b.result != "W":
            break
        n += 1
    return n


def features(ctx: dict, fighters: list[FighterView], elo: dict[int, float] | None = None
             ) -> dict[int, dict]:
    """Every candidate P4P component, 0-100, per fighter. `DEFAULT_WEIGHTS` decides
    which ones the published list uses; `scripts/fit_p4p.py` scores all of them against
    UFC's own P4P lists."""
    today = ctx["today"]
    pre: dict = {}
    if elo is None or "elo_at_fight" not in ctx:
        elo = elo_ratings(ctx, today, pre)
    else:
        pre = ctx["elo_at_fight"]
    titles = ctx.get("title_fights", set())
    first_title = title_holders(ctx)
    out: dict[int, dict] = {}
    for pool in ("men", "women"):
        members = [f for f in fighters if f.pool == pool]
        if not members:
            continue
        elites = division_elites(elo, members)
        depth = division_depth(elites)
        pool_elite = sorted((elo.get(f.fighter_id, 1500.0) for f in members), reverse=True)[:TOP_N]
        for f in members:
            r = elo.get(f.fighter_id, 1500.0)
            t_wins = sum(b.w for b in f.bouts if b.fight_id in titles and b.result == "W")
            t_apps = sum(b.w for b in f.bouts if b.fight_id in titles)
            out[f.fighter_id] = {
                "dominance": round(dominance(r, elites.get(f.division, [])), 2),
                "resume": scale_quality(opp_quality_raw(f.bouts)),
                "depth": round(depth.get(f.division, 50.0), 2),
                "recency": round(recency_score((today - f.last_activity).days,
                                               f.bouts_last_730d), 2),
                "champion": 100.0 if f.is_champion else 0.0,
                "titles": round(100.0 * (1 - math.exp(-t_wins / TITLE_SCALE)), 2),
                "title_fights": round(100.0 * (1 - math.exp(-t_apps / TITLE_SCALE)), 2),
                "streak": round(100.0 * min(_streak(f.bouts), STREAK_CAP) / STREAK_CAP, 2),
                "strength": round(dominance(r, pool_elite), 2),
                "best_wins": best_wins(f.bouts, today),
                "div_points": div_points(ctx, f),
                "champ_wins": round(100.0 * (1 - math.exp(-sum(
                    _slow(b, today) for b in f.bouts if b.result == "W"
                    and first_title.get(b.opponent_id, date.max) < b.date) / TITLE_SCALE)), 2),
                "elo_wins": elo_wins(f.bouts, pre),
                "resume_long": scale_quality(opp_quality_raw(_reweight(f.bouts, today, LONG_HALF_LIFE))),
                "_raw": {"elo": round(r, 1),
                         "division_median_elo": round(median(elites[f.division]), 1),
                         "opp_quality": round(opp_quality_raw(f.bouts), 4),
                         "title_wins_w": round(t_wins, 3)},
            }
    return out


#: Half-life, years, for the long-memory components. UFC's own list forgets slowly: a
#: title-fight loss barely moves a former champion, which a 3-year decay cannot express.
LONG_HALF_LIFE = 6.0
BEST_WINS_N = 3


def _reweight(bouts, today, half_life_years):
    from dataclasses import replace as _r
    return [_r(b, w=0.5 ** ((today - b.date).days / 365.25 / half_life_years)) for b in bouts]


def _slow(b, today) -> float:
    return 0.5 ** ((today - b.date).days / 365.25 / LONG_HALF_LIFE)


def title_holders(ctx: dict) -> dict[int, date]:
    """fighter -> date of their first UFC title-bout win (undisputed or interim)."""
    from app.services.ufc.fighter_registry import is_decided

    out: dict[int, date] = {}
    for d, wc, method, winner, _r, _b in ctx["hist"].get("title_bouts", []):
        low = wc.lower()
        if wc.startswith("UFC ") and "title bout" in low and "tournament" not in low \
                and is_decided(method, winner):
            out[winner] = min(out.get(winner, date.max), d)
    return out


#: Opponent pre-fight Elo that maps to 50 in `elo_wins`: roughly a top-15 contender.
ELO_WINS_REF = 1750.0


def elo_wins(bouts, pre: dict) -> float:
    """0-100: P(an ELO_WINS_REF fighter beats them), inverted, averaged over the
    BEST_WINS_N strongest opponents beaten by pre-fight Elo. Missing slots count as a
    1500 opponent."""
    opp = sorted((pre.get((b.fight_id, b.opponent_id), 1500.0) for b in bouts
                  if b.result == "W"), reverse=True)[:BEST_WINS_N]
    opp += [1500.0] * (BEST_WINS_N - len(opp))
    return round(100.0 * sum(win_prob(o, ELO_WINS_REF) for o in opp) / BEST_WINS_N, 2)


def best_wins(bouts, today) -> float:
    """0-100: mean opponent tier of the fighter's best BEST_WINS_N wins (slow decay),
    counting missing slots as zero. Three wins over tier-10 opponents = 100."""
    vals = sorted((b.tier * 0.5 ** ((today - b.date).days / 365.25 / LONG_HALF_LIFE)
                   for b in bouts if b.result == "W"), reverse=True)[:BEST_WINS_N]
    return round(100.0 * sum(vals) / (10.0 * BEST_WINS_N), 2)


#: Division-ranker points at which `div_points` reaches ~63 (a top-5 contender's level).
POINTS_SCALE = 6.0


def div_points(ctx: dict, f: FighterView) -> float:
    """0-100 from the division ranker's own score (Tapology-style points over the last
    six bouts, opponent tiers on an absolute scale). The fighter's best division."""
    pts = (ctx.get("division_points") or {}).get(f.fighter_id)
    if pts is None:
        return 0.0
    return round(100.0 * (1 - math.exp(-max(pts, 0.0) / POINTS_SCALE)), 2)


def division_context(hist: dict, as_of) -> dict:
    """The division ranker's order and per-fighter best points, as of a date."""
    from app.services.ufc.tapology_rankings import TapologyRanker

    res = TapologyRanker().score(hist, as_of)
    best: dict[int, float] = {}
    for (fid, _d), v in res.division_scores.items():
        best[fid] = max(best.get(fid, float("-inf")), v)
    return {"division_order": res.order, "division_points": best}


def pav_nonincreasing(values: list[float]) -> list[float]:
    """Isotonic regression (pool-adjacent-violators): the closest sequence, in least
    squares, to `values` that never increases. Barlow, Bartholomew, Bremner & Brunk,
    *Statistical Inference under Order Restrictions* (1972)."""
    blocks: list[list[float]] = []          # [sum, count]
    for v in values:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s_, n_ = blocks.pop()
            blocks[-1][0] += s_
            blocks[-1][1] += n_
    out: list[float] = []
    for s_, n_ in blocks:
        out.extend([s_ / n_] * n_)
    return out


def division_merge(rows: list[dict], key: str = "default_score") -> None:
    """Make the P4P scores consistent with the division rankings, then they can be sorted.

    A P4P list that respects every division's order is a merge of those ordered lists.
    Each division's scores, taken in division-rank order, are replaced by their isotonic
    regression — the least-squares closest scores that never rise as the division rank
    falls. Where a fighter outscores someone ranked above them in their own division, the
    two are averaged into one block: the lower-ranked fighter's strength lifts the
    higher-ranked one instead of jumping them. Sorting the adjusted scores (ties broken by
    division rank) gives the merged list. The champion is division #1, so no champion is
    ever below their own contender. Fighters with no division rank are unconstrained.
    Mirrored by lib/altRankings.js.
    """
    chains: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        r["score_raw"] = r[key]
        if r.get("division_rank"):
            chains.setdefault((r["pool"], r["division"]), []).append(r)
    for chain in chains.values():
        chain.sort(key=lambda r: r["division_rank"])
        for r, v in zip(chain, pav_nonincreasing([r[key] for r in chain])):
            r[key] = round(v, 3)


def division_ranks(ctx: dict, fighters: list[FighterView]) -> dict[int, tuple[str, int]]:
    """fighter -> (P4P division, rank in that division's published ranking).

    From the Tapology ranker — the same order the Rankings page shows, champion at #1.
    Tapology ranks a fighter in both classes when their last two bouts differ; P4P puts
    them in one: their belt's division if a champion, otherwise the one they rank higher
    in."""
    order = ctx.get("division_order") or {}
    pos: dict[int, list[tuple[int, str]]] = {}
    for div, fids in order.items():
        for k, fid in enumerate(fids, 1):
            pos.setdefault(fid, []).append((k, div))
    out = {}
    for f in fighters:
        cands = pos.get(f.fighter_id, [])
        if not cands:
            continue
        own = [c for c in cands if c[1] == f.division]
        k, div = own[0] if f.is_champion and own else min(cands)
        out[f.fighter_id] = (div, k)
    return out


def compute_p4p(ctx: dict, fighters: list[FighterView]) -> list[dict]:
    from app.services.ufc.alt_rankings_common import pool_of

    names = ctx["hist"]["names"]
    feats = features(ctx, fighters)
    ranks = division_ranks(ctx, fighters)
    rows = []
    for f in fighters:
        ft = dict(feats[f.fighter_id])
        raw = ft.pop("_raw")
        comps = {k: ft[k] for k in DEFAULT_WEIGHTS}
        div, k = ranks.get(f.fighter_id, (f.division, None))
        rows.append({
            "fighter_id": f.fighter_id, "pool": pool_of(div), "division": div,
            "division_rank": k,
            "is_champion": f.is_champion,
            "components": comps, "raw": raw,
            "default_score": round(combine(comps), 3),
            "n_bouts": len(f.bouts),
            "last_fight_date": f.bouts[0].date,
            "ledger": [ledger_entry(b, names, title=b.fight_id in ctx.get("title_fights", ()))
                       for b in f.bouts],
        })
    division_merge(rows)
    for r in rows:
        r["raw"]["score_raw"] = r.pop("score_raw")
        r["raw"]["division_rank"] = r["division_rank"]
    return rows
