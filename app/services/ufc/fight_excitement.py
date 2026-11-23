"""How exciting a fight was, and who the bonuses should have gone to.

The parts follow the sports excitement literature (Ely, Frankel & Kamenica's suspense
and surprise; inpredictable's KL-divergence excitement index), measured off what we
already have per round:

* action    pace of significant strikes, knockdowns, submission attempts, reversals,
            ground strikes, and how much of the fight was spent held down;
* suspense  how close each round was (deserve-to-win's P(red takes the round)) and how
            close the fight stood after each round;
* surprise  how far the win probability moved: a path from the closing line through
            each round's decision state to the result (a finish ends it at 0 or 1),
            scored with KL divergence so a 10% -> 1% move counts like 50% -> 90%;
* finish    KO / submission, and when it came;
* upset     -log of the winner's closing probability.

The weights are not hand-set: a conditional logit across the bouts of each card is fitted
to the real Fight of the Night picks. The context terms (title fight, main event, main
card) are in that fit so they don't leak into the other weights, and are then left out of
the excitement score, so a dull title fight is not lifted by its belt. A second
conditional logit, across each card's winners, does the same for Performance of the Night.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MODEL_VERSION = "v1"

EPS = 0.005

#: Excitement parts (in the score) and context (bonus odds only).
ACTION = ["sig_pm", "kd", "sub_att", "rev", "gnd_pm", "ctrl_share", "minutes"]
SUSPENSE = ["round_close", "path_close"]
SURPRISE = ["swing", "lead_changes"]
FINISH = ["ko", "sub", "late_finish", "quick_finish"]
UPSET = ["upset"]
EXCITE = ACTION + SUSPENSE + SURPRISE + FINISH + UPSET
CONTEXT = ["title", "main_event", "co_main", "main_card"]

#: The groups shown on the page (feature -> group). Action is split in two for display:
#: pace (how much fighting) and danger (moments that could end it).
GROUPS = {"pace": ["sig_pm", "gnd_pm", "ctrl_share", "minutes"], "danger": ["kd", "sub_att", "rev"],
          "suspense": SUSPENSE, "swings": SURPRISE, "finish": FINISH, "upset": UPSET}


# ---------------------------------------------------------------------------
# The win-probability path
# ---------------------------------------------------------------------------

def _win_share(ps: list[float]) -> float:
    """P(red wins more of these rounds than blue) + half the chance of a tie, with each
    round an independent coin of the given probability (Poisson-binomial)."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    n = len(ps)
    k = np.arange(n + 1)
    return float(dist[2 * k > n].sum() + 0.5 * dist[2 * k == n].sum())


def per_round_prior(p_fight: float, n: int) -> float:
    """The per-round chance q at which red wins an n-round decision with p_fight."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _win_share([mid] * n) < p_fight:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wp_path(prior: float, round_ps: list[float], n_sched: int, end: float | None) -> list[float]:
    """P(red wins) before the fight, after each complete round, and at the end.

    `round_ps` are the complete rounds' P(red takes the round); rounds still to come are
    drawn at the per-round rate that matches the closing line. `end` is 1/0 for a finish
    (red/blue), None for a decision: the path then stops at the scorecards' expectation
    rather than the official result, so a bad decision isn't scored as drama."""
    q = per_round_prior(min(max(prior, EPS), 1 - EPS), n_sched)
    path = [prior]
    for k in range(1, len(round_ps) + 1):
        path.append(_win_share(list(round_ps[:k]) + [q] * (n_sched - k)))
    if end is not None:
        path.append(end)
    return path


def _kl(p: float, q: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    q = min(max(q, EPS), 1 - EPS)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def path_features(path: list[float]) -> dict:
    swing = sum(_kl(b, a) for a, b in zip(path, path[1:]))
    # Who's ahead flips (after the opening line; exact 50% is no one's lead).
    sides = [np.sign(p - 0.5) for p in path[1:] if abs(p - 0.5) > 1e-9]
    flips = sum(1 for a, b in zip(sides, sides[1:]) if a != b)
    inner = path[:-1] if len(path) > 1 else path
    close = float(np.mean([4 * p * (1 - p) for p in inner]))
    return {"swing": math.log1p(swing), "lead_changes": flips, "path_close": close}


# ---------------------------------------------------------------------------
# Features per fight
# ---------------------------------------------------------------------------

def fight_features(r: dict) -> dict:
    """One fight's features from a loaded row (see load_fights)."""
    secs = max(r["fight_time_seconds"] or 0, 1)
    mins = secs / 60
    kind = r["kind"]
    fin = kind in ("ko", "sub", "doctor")
    red_won = r["winner_id"] == r["red_id"] if r["winner_id"] is not None else None

    ps = json.loads(r["round_p_red"]) if r.get("round_p_red") else []
    n_sched = r["rounds_scheduled"] or (5 if (r["max_fight_time_seconds"] or 0) > 900 else 3)
    full = (r["finish_round"] or 1) - 1 if fin else len(ps[: n_sched])
    full_ps = ps[:full]
    prior = r["red_close"] if r["red_close"] is not None else 0.5
    end = (1.0 if red_won else 0.0) if fin and red_won is not None else None
    pf = path_features(wp_path(prior, full_ps, n_sched, end))

    p_win = None
    if red_won is not None and r["red_close"] is not None:
        p_win = r["red_close"] if red_won else 1 - r["red_close"]
    frac = secs / r["max_fight_time_seconds"] if r["max_fight_time_seconds"] else 1.0

    return {
        # Counts and rates on a log scale (a 30-a-minute pace is not twice as good as 15),
        # and rates steadied with two extra minutes so a short fight's pace isn't noise.
        "sig_pm": math.log1p((r["sig"] or 0) / (mins + 2)),
        "kd": math.log1p(r["kd"] or 0),
        "sub_att": math.log1p(r["sub_att"] or 0),
        "rev": math.log1p(r["rev"] or 0),
        "gnd_pm": math.log1p((r["gnd"] or 0) / (mins + 2)),
        "ctrl_share": min(1.0, (r["ctrl"] or 0) / secs),
        "round_close": float(np.mean([4 * p * (1 - p) for p in ps[: max(full, 1)]])) if ps else 0.5,
        **pf,
        "ko": float(kind in ("ko", "doctor")),
        "sub": float(kind == "sub"),
        "late_finish": float(fin) * frac,
        "quick_finish": float(fin and secs <= 90),
        "upset": -math.log(max(p_win, EPS)) if p_win is not None else 0.0,
        "title": float(bool(r["title"])),
        "main_event": float(r["card_position"] == 0),
        "co_main": float(r["card_position"] == 1),
        "main_card": float(r["card_position"] is not None and r["card_position"] <= 4),
        "minutes": mins,
    }


# ---------------------------------------------------------------------------
# Conditional logit across a card
# ---------------------------------------------------------------------------

@dataclass
class CardLogit:
    """softmax(X @ w) within each group, fitted with an L2 penalty on standardized X."""
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    w: np.ndarray

    def utility(self, X: pd.DataFrame, only: list[str] | None = None) -> np.ndarray:
        Z = (X[self.features].to_numpy(float) - self.mean) / self.scale
        w = self.w if only is None else np.where(np.isin(self.features, only), self.w, 0.0)
        return Z @ w

    def probs(self, X: pd.DataFrame, groups: np.ndarray) -> np.ndarray:
        u = self.utility(X)
        out = np.empty_like(u)
        for g in np.unique(groups):
            m = groups == g
            e = np.exp(u[m] - u[m].max())
            out[m] = e / e.sum()
        return out

    def to_json(self) -> str:
        return json.dumps({"features": self.features, "mean": self.mean.tolist(),
                           "scale": self.scale.tolist(), "w": self.w.tolist()})

    @classmethod
    def from_json(cls, s: str) -> "CardLogit":
        d = json.loads(s)
        return cls(d["features"], np.array(d["mean"]), np.array(d["scale"]), np.array(d["w"]))


def fit_card_logit(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, features: list[str],
                   l2: float = 1.0) -> CardLogit:
    """Each positive is one draw from its card's softmax (cards with two FOTNs count both)."""
    A = X[features].to_numpy(float)
    mean = A.mean(0)
    scale = A.std(0)
    scale[scale == 0] = 1.0
    Z = (A - mean) / scale
    # Index the groups once.
    _, gidx = np.unique(groups, return_inverse=True)
    G = gidx.max() + 1
    npos = np.bincount(gidx, weights=y, minlength=G)

    def nll(w):
        u = Z @ w
        gmax = np.full(G, -np.inf)
        np.maximum.at(gmax, gidx, u)
        e = np.exp(u - gmax[gidx])
        s = np.bincount(gidx, weights=e, minlength=G)
        lse = np.log(s) + gmax
        ll = (y * u).sum() - (npos * lse).sum()
        p = e / s[gidx]
        grad = Z.T @ (y - npos[gidx] * p)
        return -ll + 0.5 * l2 * (w @ w), -grad + l2 * w

    res = minimize(nll, np.zeros(len(features)), jac=True, method="L-BFGS-B")
    return CardLogit(features, mean, scale, res.x)


def card_log_loss(p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Mean -log p of each card's actual bonus bouts."""
    return float(-np.log(np.clip(p[y > 0], 1e-9, 1)).mean())


def topk_hit(p: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int) -> float:
    """Share of cards where an actual bonus bout is among the model's top k."""
    hits = []
    for g in np.unique(groups):
        m = groups == g
        if y[m].sum() == 0:
            continue
        top = np.argsort(-p[m])[:k]
        hits.append(float(y[m][top].sum() > 0))
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

def card_score(pcts: list[float], top: int = 3, depth_at: float = 70.0) -> float:
    """A card's score from its bouts' excitement percentiles: the mean of its best `top`
    plus a depth bonus (share of bouts above `depth_at`, up to 10 points), so a single
    great fight doesn't carry an otherwise flat card."""
    if not pcts:
        return 0.0
    s = sorted(pcts, reverse=True)
    best = float(np.mean(s[:top]))
    depth = float(np.mean([p >= depth_at for p in s]))
    return best * 0.9 + depth * 10.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

#: UFC cards only: ufcstats also carries Pride, WEC, Strikeforce, DWCS and others, which
#: have no bonus icons and aren't what the best-of lists rank.
UFC_EVENT = r"^(UFC|The Ultimate Fighter)"

_SQL = """
with s as (
  select st.fight_id,
         sum(st.sig_str_landed)::float sig, sum(st.kd)::float kd, sum(st.sub_att)::float sub_att, sum(st.rev)::float rev,
         sum(st.ground_landed)::float gnd, sum(st.ctrl_seconds)::float ctrl,
         sum(case when st.fighter_id = f.red_fighter_id then st.sig_str_landed end)::float red_sig,
         sum(case when st.fighter_id = f.red_fighter_id then st.kd end)::float red_kd,
         sum(case when st.fighter_id = f.red_fighter_id then st.sub_att end)::float red_sub_att
  from ufc.ufc_fight_stats st join ufc.ufc_fights f on f.id = st.fight_id
  where st.round_number = 0 group by st.fight_id
), b as (
  select fight_id, bool_or(bonus_type = 'fotn') fotn,
         bool_or(bonus_type in ('potn', 'kotn', 'sotn')) perf
  from ufc.ufc_fight_bonuses group by fight_id
)
select f.id fight_id, f.date, f.event_id, e.name event_name, f.card_position, f.weight_class,
       f.method, f.details, f.winner_id, f.red_fighter_id red_id, f.blue_fighter_id blue_id,
       f.finish_round, f.fight_time_seconds, f.max_fight_time_seconds,
       s.sig, s.kd, s.sub_att, s.rev, s.gnd, s.ctrl, s.red_sig, s.red_kd, s.red_sub_att,
       d.round_p_red, d.rounds_scheduled,
       (select o.red_close_prob from ufc.ufc_fight_odds_open_close o
         where o.fight_id = f.id and o.bookmaker = 'Consensus' and o.red_close_prob is not null
         limit 1) red_close,
       coalesce(b.fotn, false) fotn, coalesce(b.perf, false) perf
from ufc.ufc_fights f
join ufc.ufc_events e on e.id = f.event_id
join s on s.fight_id = f.id
left join ufc.ufc_deserve_to_win d on d.fight_id = f.id and d.model_version = 'v1'
left join b on b.fight_id = f.id
where f.method is not null and f.date < current_date and e.name ~ :ufc
order by f.date, f.event_id, f.card_position
"""


def load_rows(db) -> list[dict]:
    """Every completed UFC bout (no-contests dropped), with its outcome kind and title flag."""
    from sqlalchemy import text

    from app.services.ufc.model import _is_title_bout
    from app.services.ufc.outcome_types import classify_outcome

    out = []
    for r in db.execute(text(_SQL), {"ufc": UFC_EVENT}):
        r = dict(r._mapping)
        r["kind"] = classify_outcome(r["method"], r["details"], r["winner_id"])
        if r["kind"] == "void":
            continue
        r["title"] = _is_title_bout(r["weight_class"])
        out.append(r)
    return out


def load_fights(db, rows: list[dict] | None = None) -> pd.DataFrame:
    """Every completed UFC bout with its features, bonus labels and identifiers."""
    rows = load_rows(db) if rows is None else rows
    out = []
    for r in rows:
        feats = fight_features(r)
        out.append({**{k: r[k] for k in ("fight_id", "date", "event_id", "event_name", "card_position",
                                         "winner_id", "red_id", "blue_id", "kind", "red_close",
                                         "fotn", "perf")},
                    **feats, **winner_features(r, feats)})
    return pd.DataFrame(out)


#: Performance of the Night: the winner's own night.
PERF = ["ko", "sub", "quick_finish", "late_finish", "win_kd", "win_sub_att", "win_sig_share", "upset"]


def winner_features(r: dict, feats: dict) -> dict:
    red = r["winner_id"] is not None and r["winner_id"] == r["red_id"]
    tot = r["sig"] or 0
    red_sig = r["red_sig"] or 0
    sig = red_sig if red else tot - red_sig
    kd = (r["red_kd"] or 0) if red else (r["kd"] or 0) - (r["red_kd"] or 0)
    sa = (r["red_sub_att"] or 0) if red else (r["sub_att"] or 0) - (r["red_sub_att"] or 0)
    return {"win_kd": float(kd), "win_sub_att": float(sa), "win_sig_share": sig / tot if tot else 0.5,
            "has_winner": r["winner_id"] is not None}


def group_contributions(model: CardLogit, X: pd.DataFrame) -> pd.DataFrame:
    """Each excitement group's share of the score, per fight (standardized units)."""
    return pd.DataFrame({g: model.utility(X, only=f) for g, f in GROUPS.items()}, index=X.index)
