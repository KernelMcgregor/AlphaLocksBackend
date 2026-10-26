"""Fight dictation: who imposes where and how fast a fight is fought.

Research module (Stage 1, see scripts/dictation_eval.py). Two online models, both fitted
forward in time from per-round ufcstats rows. Every prediction for a bout uses the state
before that bout; the bout's rounds update the state afterwards.

Tug-of-war (shared properties of a round: where it is fought, how fast)
    Each fighter has a preference p (where they want the round, on a logit/log scale) and
    a dictation skill s. A round between A and B lands at

        z_hat = w * p_A + (1 - w) * p_B,   w = sigmoid(s_A - s_B + (d_A - d_B) * (round - 1))

    so equal skill splits the difference and a big skill gap puts the round on the
    stronger dictator's terms. d is in-fight drift: d > 0 takes over as the fight goes on.
    Fitted by online gradient steps on the weighted squared error. A round only moves s
    in proportion to (p_A - p_B), so mirror matchups carry almost no dictation signal.

Mechanics (who controls whom, opponent-aware)
    Control time is the product of three contests, each one fighter's offence against the
    other's matching defence:
        shot rate   takedown attempts / standing minute   log  = c + shoot_A - deter_B
        success     takedowns landed / attempted          logit = c + tdo_A - tdd_B
        hold        control seconds per entry (spell)     log  = c + top_A - esc_B
    plus a small "other entry" rate (control gained without a takedown: clinch, scrambles,
    knockdowns). The three combine as a continuous-time chain over standing / A on top /
    B on top, started standing, so a great wrestler against a great escaper predicts many
    takedowns but short spells.

Caveats: ufcstats control includes clinch control against the cage, and there are no
get-up counts, so a spell is estimated as control seconds / entries and the last spell of
a round is censored at the bell (ignored here).

Stage-1 findings (scripts/dictation_eval.py, report window 2019+, 2026-10-25)
    Location (ground, clinch): no individual dictation skill. A global direction rule
    (h = -12, i.e. the round lands at the LOWER of the two preferences: the fighter who
    wants less ground/clinch decides) beats split-the-difference by ~5% and absorbs all of
    the per-fighter skill gain (+0.04%, CI spans 0). Pull beyond the rule: split-half r 0.11.
    Pace: no robust direction rule; per-fighter pace imposition is a stable trait (split-half
    r 0.54 beyond the rule, persists in losses, r 0.27) and winners impose more (+0.06 vs
    -0.06), but forecasting gain for round pace is small (+0.5%, CI spans 0).
    Round drift: the pace pusher gets more of their way after R1 (pull 0.60 -> 0.76), but a
    fighter's own drift is not a trait (split-half r ~0.1).
    Mechanics: escape skill is real (split-half r 0.32; elite TD threat vs good escaper 26%
    control share vs 35% against a poor one) but the split model does not beat a one-
    parameter opponent-adjusted control model (MSE 56.3 vs 55.1 x1e-3).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

ROUND_SEC = 300.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# tug-of-war
# ---------------------------------------------------------------------------

@dataclass
class TugParams:
    k_s: float = 0.05        # dictation-skill step
    k_d: float = 0.0         # in-fight drift step (0 = drift off)
    p_prior: float = 6.0     # pseudo-rounds of league mean in each preference
    s_l2: float = 0.0        # per-update shrink of s toward 0
    learn_s: bool = True     # False = symmetric baseline (w = 0.5 always)
    h: float = 0.0           # global direction effect: > 0 the fighter wanting MORE of the
                             # dimension gets their way, < 0 the one wanting LESS does


@dataclass
class _Tug:
    p: float
    n: float = 0.0
    s: float = 0.0
    d: float = 0.0


class TugOfWar:
    """Online tug-of-war for one round-level dimension on a transformed scale."""

    def __init__(self, league: float, params: TugParams | None = None):
        self.league = league
        self.prm = params or TugParams()
        self.st: dict = {}

    def _get(self, f) -> _Tug:
        if f not in self.st:
            self.st[f] = _Tug(p=self.league)
        return self.st[f]

    def snapshot(self, f) -> tuple[float, float, float, float]:
        t = self._get(f)
        return t.p, t.s, t.d, t.n

    def weight(self, a, b, rnd: int) -> float:
        A, B = self._get(a), self._get(b)
        x = self.prm.h * np.tanh(2.0 * (A.p - B.p))
        if self.prm.learn_s:
            x += A.s - B.s + (A.d - B.d) * (rnd - 1)
        return float(sigmoid(x))

    def predict(self, a, b, rnd: int) -> float:
        A, B = self._get(a), self._get(b)
        w = self.weight(a, b, rnd)
        return w * A.p + (1 - w) * B.p

    def update(self, a, b, rounds: list[tuple[int, float, float]]) -> None:
        """rounds: [(round number, observed z, weight)] for one bout, all predicted from the
        pre-bout state; applied together after the bout."""
        A, B = self._get(a), self._get(b)
        gp_a = gp_b = ws_a = ws_b = 0.0
        gs = gd = 0.0
        for rnd, z, wt in rounds:
            w = self.weight(a, b, rnd)
            e = z - (w * A.p + (1 - w) * B.p)
            # preference: running-mean style step, weighted by how much of the round was
            # on that fighter's terms (a dominated round says little about the loser's wish)
            gp_a += wt * w * e
            ws_a += wt * w
            gp_b += wt * (1 - w) * e
            ws_b += wt * (1 - w)
            g = (A.p - B.p) * w * (1 - w)
            gs += wt * e * g
            gd += wt * e * g * (rnd - 1)
        k = self.prm.p_prior
        if ws_a > 0:
            A.p += gp_a / (A.n + ws_a + k)
            A.n += ws_a
        if ws_b > 0:
            B.p += gp_b / (B.n + ws_b + k)
            B.n += ws_b
        if self.prm.learn_s:
            A.s += self.prm.k_s * gs
            B.s -= self.prm.k_s * gs
            A.d += self.prm.k_d * gd
            B.d -= self.prm.k_d * gd
            if self.prm.s_l2:
                A.s *= 1 - self.prm.s_l2
                B.s *= 1 - self.prm.s_l2


def pull(z: float, p_a: float, p_b: float) -> float | None:
    """Share of the way from B's preference to A's that the round landed (1 = A's terms)."""
    if abs(p_a - p_b) < 1e-9:
        return None
    return float(np.clip((z - p_b) / (p_a - p_b), -0.5, 1.5))


# ---------------------------------------------------------------------------
# mechanics
# ---------------------------------------------------------------------------

@dataclass
class MechParams:
    lr: float = 0.03          # fighter-offset step
    lr_c: float = 0.002       # league-intercept step
    l2: float = 0.002         # per-update shrink of every offset toward 0
    use_escape: bool = True   # False = hold time depends on top skill only (no escape skill)
    use_split: bool = True    # False = one-parameter control model (ctl_A - ctld_B)


MECH_KEYS = ("shoot", "deter", "tdo", "tdd", "top", "esc", "oth", "othd", "ctl", "ctld")


@dataclass
class MechRound:
    """One attacker's half of a round."""
    secs: float          # round length fought
    td_att: float
    td_land: float
    ctrl: float          # attacker's control seconds
    opp_ctrl: float      # defender's control seconds


@dataclass
class Mechanics:
    prm: MechParams = field(default_factory=MechParams)
    # league intercepts (per-second rates and log seconds), set by init_league
    c_shoot: float = np.log(1.0 / 300)
    c_tdo: float = -0.7
    c_hold: float = np.log(60.0)
    c_oth: float = np.log(0.3 / 300)
    c_ctl: float = np.log(0.15)
    st: dict = field(default_factory=lambda: defaultdict(lambda: dict.fromkeys(MECH_KEYS, 0.0)))

    def init_league(self, rows: list[MechRound]) -> None:
        stand = sum(max(r.secs - r.ctrl - r.opp_ctrl, 30.0) for r in rows)
        att = sum(r.td_att for r in rows)
        land = sum(r.td_land for r in rows)
        entries = sum(_entries(r) for r in rows)
        oth = sum(_entries(r) - r.td_land for r in rows)
        ctrl = sum(r.ctrl for r in rows)
        secs = sum(r.secs for r in rows)
        self.c_shoot = np.log(max(att, 1) / stand)
        self.c_tdo = float(logit(land / max(att, 1)))
        self.c_hold = np.log(max(ctrl, 1) / max(entries, 1))
        self.c_oth = np.log(max(oth, 1) / stand)
        self.c_ctl = float(logit(ctrl / secs))

    # -- rates for attacker a against defender b
    def rates(self, a, b) -> dict:
        A, B = self.st[a], self.st[b]
        esc = B["esc"] if self.prm.use_escape else 0.0
        return {
            "shoot": np.exp(self.c_shoot + A["shoot"] - B["deter"]),       # attempts / standing sec
            "succ": float(sigmoid(self.c_tdo + A["tdo"] - B["tdd"])),
            "hold": np.exp(self.c_hold + A["top"] - esc),                   # mean spell seconds
            "oth": np.exp(self.c_oth + A["oth"] - B["othd"]),               # non-TD entries / sec
            "ctl": float(sigmoid(self.c_ctl + A["ctl"] - B["ctld"])),       # one-parameter share
        }

    def predict_share(self, a, b, secs: float = ROUND_SEC) -> tuple[float, float]:
        """Expected share of the round with a on top and with b on top."""
        ra, rb = self.rates(a, b), self.rates(b, a)
        if not self.prm.use_split:
            return ra["ctl"], rb["ctl"]
        return occupancy(ra["shoot"] * ra["succ"] + ra["oth"], 1.0 / ra["hold"],
                         rb["shoot"] * rb["succ"] + rb["oth"], 1.0 / rb["hold"], secs)

    def update(self, a, b, r: MechRound) -> None:
        """Gradient step on the attacker-a half of one round (call twice per round)."""
        A, B = self.st[a], self.st[b]
        lr, lc = self.prm.lr, self.prm.lr_c
        ra = self.rates(a, b)
        stand = max(r.secs - r.ctrl - r.opp_ctrl, 30.0)
        # shot rate: Poisson
        e = r.td_att - ra["shoot"] * stand
        A["shoot"] += lr * e
        B["deter"] -= lr * e
        self.c_shoot += lc * e / 10
        # success: binomial
        if r.td_att > 0:
            e = r.td_land - r.td_att * ra["succ"]
            A["tdo"] += lr * e
            B["tdd"] -= lr * e
            self.c_tdo += lc * e / 10
        # other entries: Poisson
        n_oth = _entries(r) - r.td_land
        e = n_oth - ra["oth"] * stand
        A["oth"] += lr * e
        B["othd"] -= lr * e
        self.c_oth += lc * e / 10
        # hold: exponential spells, d loglik / d log(mean) = -n + T/mean
        n = _entries(r)
        if n > 0:
            e = float(np.clip(-n + r.ctrl / ra["hold"], -3, 3))
            A["top"] += lr * e
            if self.prm.use_escape:
                B["esc"] -= lr * e
            self.c_hold += lc * e / 10
        # one-parameter control share (baseline): logistic on the share of round time
        e = r.ctrl / r.secs - ra["ctl"]
        A["ctl"] += lr * 5 * e
        B["ctld"] -= lr * 5 * e
        self.c_ctl += lc * e
        if self.prm.l2:
            for k in MECH_KEYS:
                A[k] *= 1 - self.prm.l2
                B[k] *= 1 - self.prm.l2


def _entries(r: MechRound) -> float:
    """Ground/control entries: every landed takedown, or one if control came without one."""
    return r.td_land if r.td_land > 0 else (1.0 if r.ctrl >= 5 else 0.0)


def occupancy(in_a: float, out_a: float, in_b: float, out_b: float, secs: float) -> tuple[float, float]:
    """Expected share of [0, secs] spent in A-top and B-top for the chain
    standing -> A-top (rate in_a), A-top -> standing (out_a), likewise for B, started standing."""
    Q = np.array([[-(in_a + in_b), in_a, in_b],
                  [out_a, -out_a, 0.0],
                  [out_b, 0.0, -out_b]])
    M = np.zeros((6, 6))
    M[:3, :3] = Q * secs
    M[:3, 3:] = np.eye(3) * secs
    occ = expm(M)[0, 3:] / secs
    return float(occ[1]), float(occ[2])
