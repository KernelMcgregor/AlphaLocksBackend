"""Whole-career rating over every Sherdog pro bout, with the pieces career_rating.py lacks.

career_rating.pro_elo is a plain Elo (K=48, everyone starts at 1500): a regional win counts
the same as a UFC win, a 2-fight newcomer is as "known" as a 40-fight veteran, and method is
ignored. This engine adds, each switchable so scripts/pro_rating_eval.py can test them:

  rd        Glicko-style uncertainty: each fighter carries (rating, RD); RD starts at RD0,
            shrinks with fights and grows with inactivity (C per sqrt-day). Expected score
            uses the combined RD, so uncertain matchups are pulled toward 50%.
  k_decay   K = K0 * (1 + KD / (1 + n)) for an Elo run (faster learning for newcomers).
  promo     newcomers start at their debut promotion's as-of strength instead of 1500:
            the shrunk mean pre-bout rating of fighters who fought on that promotion in the
            previous PROMO_WINDOW days (earlier dates only).
  finish    a finish moves ratings FIN times as far as a decision.

One pass in date order over canonical bouts. For each bout it records the pre-bout expected
score (for scoring) and each fighter's state after the bout (for features), so every read
is strictly before the date being predicted.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

INIT = 1500.0
Q = math.log(10) / 400.0


@dataclass
class Params:
    k0: float = 48.0
    kd: float = 0.0          # K decay strength (0 = plain Elo)
    rd: bool = False
    rd0: float = 250.0
    rd_min: float = 50.0
    c: float = 2.5           # RD growth per sqrt(day) of inactivity
    promo: bool = False
    promo_window: int = 3 * 365
    promo_shrink: float = 30.0   # pseudo-fighters at INIT in a promotion's mean
    fin: float = 1.0         # finish multiplier on the update


def load_bouts(db) -> pd.DataFrame:
    """Canonical Sherdog bouts: a, b (sherdog ids), date, a_score, method_class, promo."""
    import zlib

    from sqlalchemy import text

    from app.models.ufc import UFC_SCHEMA
    from app.services.ufc.promotions import clean
    schema = f"{UFC_SCHEMA}." if UFC_SCHEMA else ""
    b = pd.read_sql(text(
        "SELECT fighter_sherdog_id AS a, opponent_sherdog_id AS b, opponent_key AS b_key, date, result, "
        f"method_class, event_name, promotion FROM {schema}sherdog_bouts "
        "WHERE date IS NOT NULL AND result IN ('W','L','D')"), db.bind)
    synthetic = np.array([-(zlib.crc32(str(k).encode()) % 10**9) - 1 for k in b["b_key"]], dtype=np.int64)
    b["b"] = np.where(b["b"].isna(), synthetic, b["b"].fillna(0)).astype(np.int64)
    b["a"] = b["a"].astype(np.int64)
    b["lo"], b["hi"] = np.minimum(b["a"], b["b"]), np.maximum(b["a"], b["b"])
    b = b.drop_duplicates(["lo", "hi", "date"]).sort_values(["date", "lo", "hi"]).reset_index(drop=True)
    b["a_score"] = b["result"].map({"W": 1.0, "L": 0.0, "D": 0.5})
    b["promo"] = [clean(e, p) for e, p in zip(b["event_name"], b["promotion"])]
    b["date"] = pd.to_datetime(b["date"]).dt.date
    return b[["a", "b", "date", "a_score", "method_class", "promo"]]


def run(bouts: pd.DataFrame, p: Params, details: bool = False):
    """Returns (pre-bout expected score for side a per bout, history per fighter), plus with
    details=True a frame of per-bout pre-bout ratings and the promotion's as-of strength."""
    rating = {}
    rd = {}
    n = defaultdict(int)
    last = {}
    # promotion strength: recent pre-bout ratings of fighters appearing on the promotion
    promo_hist: dict[str, deque] = defaultdict(deque)
    promo_sum: dict[str, float] = defaultdict(float)
    hist: dict[int, dict[str, list]] = defaultdict(lambda: {"date": [], "state": []})
    expect = np.empty(len(bouts))
    pre_a, pre_b, promo_str = (np.empty(len(bouts)) for _ in range(3))
    pending = []          # promo observations for the current date (applied when the date moves)
    cur_date = None

    def promo_mean(pr, d):
        dq = promo_hist[pr]
        while dq and (d - dq[0][0]).days > p.promo_window:
            promo_sum[pr] -= dq.popleft()[1]
        m = len(dq)
        return (promo_sum[pr] + p.promo_shrink * INIT) / (m + p.promo_shrink)

    for i, (a, b, d, s, mc, pr) in enumerate(bouts.itertuples(index=False)):
        if d != cur_date:
            for pr_, d_, r_ in pending:
                promo_hist[pr_].append((d_, r_))
                promo_sum[pr_] += r_
            pending = []
            cur_date = d
        for f in (a, b):
            if f not in rating:
                rating[f] = promo_mean(pr, d) if p.promo else INIT
                rd[f] = p.rd0
            elif p.rd:
                gap = (d - last[f]).days
                rd[f] = min(math.sqrt(rd[f] ** 2 + p.c ** 2 * max(gap, 0)), p.rd0)
        ra, rb = rating[a], rating[b]
        pre_a[i], pre_b[i] = ra, rb
        if details:
            promo_str[i] = promo_mean(pr, d)
        if p.rd:
            g = 1.0 / math.sqrt(1 + 3 * Q * Q * (rd[a] ** 2 + rd[b] ** 2) / math.pi ** 2)
            e = 1.0 / (1.0 + 10 ** (-g * (ra - rb) / 400.0))
        else:
            e = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        expect[i] = e
        mult = p.fin if mc in ("KO/TKO", "SUB") and s != 0.5 else 1.0
        pending.append((pr, d, ra))
        pending.append((pr, d, rb))
        if p.rd:
            # Glicko-1 single-game update for each side
            for me, opp, sc in ((a, b, s), (b, a, 1 - s)):
                gj = 1.0 / math.sqrt(1 + 3 * Q * Q * rd[opp] ** 2 / math.pi ** 2)
                ej = 1.0 / (1.0 + 10 ** (-gj * (rating[me] - rating[opp]) / 400.0))
                d2 = 1.0 / (Q * Q * gj * gj * ej * (1 - ej))
                denom = 1.0 / rd[me] ** 2 + 1.0 / d2
                new_r = rating[me] + mult * Q / denom * gj * (sc - ej)
                new_rd = max(math.sqrt(1.0 / denom), p.rd_min)
                if me == a:
                    upd_a = (new_r, new_rd)
                else:
                    upd_b = (new_r, new_rd)
            (rating[a], rd[a]), (rating[b], rd[b]) = upd_a, upd_b
        else:
            ka = p.k0 * (1 + p.kd / (1 + n[a])) * mult
            kb = p.k0 * (1 + p.kd / (1 + n[b])) * mult
            rating[a] = ra + ka * (s - e)
            rating[b] = rb - kb * (s - e)
        for me, opp_r in ((a, rb), (b, ra)):
            n[me] += 1
            last[me] = d
            hist[me]["date"].append(d)
            hist[me]["state"].append((rating[me], rd[me], n[me], opp_r))
    if details:
        return expect, hist, pd.DataFrame({"pre_a": pre_a, "pre_b": pre_b, "promo_strength": promo_str})
    return expect, hist


def state_before(hist, sid, d):
    """(rating, rd, n, last opponent rating) strictly before date d, or None."""
    h = hist.get(sid)
    if not h:
        return None
    k = bisect_left(h["date"], d)
    return h["state"][k - 1] if k else None
