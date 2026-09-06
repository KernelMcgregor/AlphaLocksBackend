"""Rankers for the evaluation harness, including the two reference brackets.

Every candidate must be read against these. `always_red` is the floor — anything near it
carries no information. `market` is the practical ceiling: the de-vigged closing line is
the best publicly available forecast, and a ranking built from fight results alone should
not beat it. A candidate outside the bracket means the harness is wrong, not that the
ranking is brilliant.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

from app.services.ufc.fighter_registry import classify_weight_class
from app.services.ufc.points_ranking_service import (
    ELO_K_BASE, ELO_K_NEWCOMER, ELO_METHOD_K, ELO_NEWCOMER_FIGHTS, ELO_START,
    FIGHTS_WINDOW, LOSS_METHOD_MULT, LOSS_PENALTY_BASE, MIN_DIVISOR,
    PRIOR_STRENGTH, ELO_PRIOR_SCALE, RECENCY_WEIGHTS, TITLE_MULT, FIVE_ROUND_MULT,
    WIN_POINTS, _classify_method, _elo_expected, _elo_percentile,
    _loss_opponent_factor, _quality_from_percentile,
)


#: The shipped loss multipliers, before the win:loss ratio was bounded.
#: Elo above ELO_START that maps to the full career bonus. 400 = the Elo scale's 10:1
#: odds interval, so a 1900-rated fighter reaches the cap and a 1500 gets nothing.
ELO_LINEAR_SPREAD = 400.0

LEGACY_LOSS_METHOD_MULT = {
    "early_finish": 1.5, "late_finish": 1.2, "ud": 0.8,
    "majority": 0.6, "split": 0.4, "decision": 0.7,
}


class AlwaysRed:
    """Floor. Red corner is the promotion's favourite often enough that this is not 0.50
    exactly, which is itself worth seeing."""

    name = "always_red"

    def rate(self, fid, as_of):
        return (0.0, 0.0)

    def observe(self, fight):
        pass


class MarketBaseline:
    """Ceiling. Rates each corner with its de-vigged closing implied probability.

    Not a ranking — it cannot order a division — but it is the number every ranking is
    ultimately competing against, so it belongs in the table.
    """

    name = "market"

    def __init__(self, db=None):
        self._by_fight: dict[int, float] = {}
        if db is not None:
            from app.models.ufc import UFCFightOdds
            from app.services.ufc.market_anchor import devig

            # red_odds/blue_odds are AMERICAN odds (+175 / -222); the implied
            # probabilities live in their own columns. Averaging the American numbers
            # and taking r/(r+b) produced garbage that clamped to the bounds, which is
            # why this baseline scored a degenerate 0.5000.
            agg: dict[int, list[tuple[float, float]]] = defaultdict(list)
            for o in db.query(UFCFightOdds).all():
                if o.red_implied_prob is not None and o.blue_implied_prob is not None:
                    agg[o.fight_id].append(
                        (float(o.red_implied_prob), float(o.blue_implied_prob)))
            for fid, rows in agg.items():
                r = sum(x[0] for x in rows) / len(rows)
                b = sum(x[1] for x in rows) / len(rows)
                self._by_fight[fid] = float(devig(r, b))
        self._fight = None

    # The harness rates corners independently, so carry the current fight's price
    # through a per-fight hook rather than per-fighter state.
    def bind(self, fight):
        self._fight = fight

    def rate(self, fid, as_of):
        f = self._fight
        if f is None:
            return None
        p = self._by_fight.get(f.id)
        if p is None:
            return None
        # log-odds so differences are additive, matching the sigmoid link
        p = min(max(p, 1e-6), 1 - 1e-6)
        lo = math.log(p / (1 - p))
        return (lo / 2 if fid == f.red_fighter_id else -lo / 2, 0.0)

    def observe(self, fight):
        pass


class EloRanker:
    """The Points system's Phase 1 alone — career Elo, no points window."""

    name = "elo"

    def __init__(self):
        self.elo: dict[int, float] = defaultdict(lambda: ELO_START)
        self.n: dict[int, int] = defaultdict(int)

    def rate(self, fid, as_of):
        if self.n[fid] == 0:
            return None
        return (self.elo[fid], 0.0)

    def observe(self, fight):
        r, b = fight.red_fighter_id, fight.blue_fighter_id
        cat = _classify_method(fight.method or "", fight.finish_round)
        mk = ELO_METHOD_K.get(cat, 1.0)
        exp_r = _elo_expected(self.elo[r], self.elo[b])
        kr = (ELO_K_NEWCOMER if self.n[r] < ELO_NEWCOMER_FIGHTS else ELO_K_BASE) * mk
        kb = (ELO_K_NEWCOMER if self.n[b] < ELO_NEWCOMER_FIGHTS else ELO_K_BASE) * mk
        red_won = fight.winner_id == r
        self.elo[r] += kr * ((1.0 if red_won else 0.0) - exp_r)
        self.elo[b] += kb * ((0.0 if red_won else 1.0) - (1.0 - exp_r))
        self.n[r] += 1
        self.n[b] += 1


class PointsRankerOnline:
    """The production Points+Elo score, evaluated causally.

    Shares the constants and helpers with points_ranking_service so the harness scores
    the shipping rule, not a re-implementation of it.
    """

    name = "points"

    #: The four independent changes to the shipped scoring, so each can be measured
    #: alone rather than judged as a bundle. Shipping them together made the top-15
    #: slice worse while improving AUC, which is unreadable without an ablation.
    ALL_CHANGES = frozenset({"divisor", "loss", "prior", "perfighter_k",
                             "elo_linear", "window10"})

    #: Shipping configuration, chosen by ablation.
    SHIPPING = frozenset({"loss", "perfighter_k"})

    def __init__(self, legacy: bool = False, changes: frozenset | None = None):
        #: `legacy` reproduces the scoring exactly as it shipped: a SUM over the window
        #: rather than a floored average, an opponent factor on losses that INVERTED
        #: with quality, a flat additive ELO_BONUS_MAX, and a single K taken as
        #: min() over both corners.
        self.changes = frozenset() if legacy else (
            self.ALL_CHANGES if changes is None else changes)
        self.legacy = legacy
        if legacy:
            self.name = "points_legacy"
        elif changes is not None:
            self.name = "points+" + "-".join(sorted(changes)) if changes else "points_none"
        self.elo: dict[int, float] = defaultdict(lambda: ELO_START)
        self.n: dict[int, int] = defaultdict(int)
        self.hist: dict[int, list[dict]] = defaultdict(list)
        self._sorted_elos: list[float] = []
        self._dirty = True

    def _active(self) -> list[float]:
        if self._dirty:
            self._sorted_elos = sorted(
                self.elo[f] for f in self.elo if self.n[f] >= 3)
            self._dirty = False
        return self._sorted_elos

    def rate(self, fid, as_of):
        window = 10 if "window10" in self.changes else FIGHTS_WINDOW
        recent = self.hist[fid][-window:][::-1]
        if len(recent) < 3:
            return None
        active = self._active()
        total = weight = 0.0
        for i, f in enumerate(recent):
            rec = RECENCY_WEIGHTS[i] if i < len(RECENCY_WEIGHTS) else 0.3
            opp_mult = _quality_from_percentile(_elo_percentile(f["opp_elo"], active))
            ctx = TITLE_MULT if f["is_title"] else (FIVE_ROUND_MULT if f["is_5rd"] else 1.0)
            if f["won"]:
                v = WIN_POINTS.get(f["cat"], 2.5) * opp_mult * ctx
            elif "loss" in self.changes:
                v = (LOSS_PENALTY_BASE * LOSS_METHOD_MULT.get(f["cat"], 1.0)
                     * _loss_opponent_factor(opp_mult) * ctx)
            else:
                v = (-1.5 * LEGACY_LOSS_METHOD_MULT.get(f["cat"], 1.0)
                     * max(2.0 - opp_mult, 0.3) * ctx)
            total += v * rec
            weight += rec

        # Career-strength term. The percentile SATURATES at the top: with 563 eligible
        # fighters every contender sits at 0.94-1.00, so this is a near-constant ~33-35
        # across an entire top 15 and cannot separate them. Charles Oliveira's 97-Elo
        # lead over Quillan Salkilld converts to 1.93 points of score, against a +9.1
        # fight-points gap — so the ranking is decided almost entirely by recent form,
        # which is the reported symptom. The linear form keeps elite separation.
        if "elo_linear" in self.changes:
            pct = (self.elo[fid] - ELO_START) / ELO_LINEAR_SPREAD
            pct = max(0.0, min(1.0, pct))
        else:
            pct = _elo_percentile(self.elo[fid], active)
        form = total / max(weight, MIN_DIVISOR) if "divisor" in self.changes else total

        if "prior" in self.changes:
            k = len(recent)
            return ((k / (k + PRIOR_STRENGTH)) * form
                    + (PRIOR_STRENGTH / (k + PRIOR_STRENGTH)) * (pct * ELO_PRIOR_SCALE), 0.0)
        # Flat additive bonus. Note this is on a very different scale once `divisor` is
        # on: 35 points added to an average of ~5 swamps the fight record entirely.
        return (form + pct * 35.0, 0.0)

    def observe(self, fight):
        r, b = fight.red_fighter_id, fight.blue_fighter_id
        cat = _classify_method(fight.method or "", fight.finish_round)
        mk = ELO_METHOD_K.get(cat, 1.0)
        pre_r, pre_b = self.elo[r], self.elo[b]
        exp_r = _elo_expected(pre_r, pre_b)
        if "perfighter_k" in self.changes:
            kr = (ELO_K_NEWCOMER if self.n[r] < ELO_NEWCOMER_FIGHTS else ELO_K_BASE) * mk
            kb = (ELO_K_NEWCOMER if self.n[b] < ELO_NEWCOMER_FIGHTS else ELO_K_BASE) * mk
        else:
            # Shipped behaviour: min() over BOTH corners, so a veteran facing a debutant
            # also updated at the newcomer rate.
            shared = (ELO_K_NEWCOMER if min(self.n[r], self.n[b]) < ELO_NEWCOMER_FIGHTS
                      else ELO_K_BASE) * mk
            kr = kb = shared
        red_won = fight.winner_id == r
        self.elo[r] += kr * ((1.0 if red_won else 0.0) - exp_r)
        self.elo[b] += kb * ((0.0 if red_won else 1.0) - (1.0 - exp_r))
        self.n[r] += 1
        self.n[b] += 1
        self._dirty = True

        is_title = "title" in (fight.weight_class or "").lower()
        is_5rd = bool(fight.time_format and fight.time_format.count("-") >= 4)
        self.hist[r].append({"won": red_won, "cat": cat, "opp_elo": pre_b,
                             "is_title": is_title, "is_5rd": is_5rd})
        self.hist[b].append({"won": not red_won, "cat": cat, "opp_elo": pre_r,
                             "is_title": is_title, "is_5rd": is_5rd})


def build_ranker(name: str, db=None):
    if name == "always_red":
        return AlwaysRed()
    if name == "market":
        return MarketBaseline(db)
    if name == "elo":
        return EloRanker()
    if name == "points":
        return PointsRankerOnline()
    if name == "points_legacy":
        return PointsRankerOnline(legacy=True)
    if name.startswith("points+"):
        # "-" separates flags because "," already separates rankers on the CLI.
        flags = frozenset(x for x in name[len("points+"):].split("-") if x)
        unknown = flags - PointsRankerOnline.ALL_CHANGES
        if unknown:
            raise SystemExit(f"unknown scoring change(s): {sorted(unknown)}")
        return PointsRankerOnline(changes=flags)
    if name.startswith("whr"):
        from app.services.ufc.whr_ranker import WHRRanker
        # "whr@w2=4:c=0.5" — w2 is the Wiener drift, c the conservatism in sigmas.
        kw = {}
        if "@" in name:
            for part in name.split("@", 1)[1].split(":"):
                k, _, v = part.partition("=")
                kw[{"w2": "w2", "c": "conservatism"}[k]] = float(v)
        r = WHRRanker(**kw)
        if kw:
            r.name = name
        return r
    if name == "bt":
        from app.services.ufc.whr_ranker import BradleyTerryRanker
        return BradleyTerryRanker()
    raise SystemExit(f"unknown ranker: {name}")
