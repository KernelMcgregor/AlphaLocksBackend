"""Style against the opponent, as of before each fight (v5 habits space, continuous).

The served style space (style_service) is one snapshot built from whole careers, so it can't
be a model input. This rebuilds the same v5 fingerprint as of any date:

  * habits: style_habits.style_features over the fighter's bouts BEFORE the date;
  * opponent model (the KenPom-style "allow" effects): refit every quarter on bouts before
    the quarter boundary, the one fit that would otherwise leak later results;
  * space (division scaling, era trend, residual scale): style_service.fit_space refit every
    year on the vectors of fighters with MIN_BOUTS_FIT prior bouts at that date;
  * prime year from prior bouts, division = most common division over the last 5 prior bouts.

Similarity is cosine in the full 15-column space: no families (they are labels on a
continuum; silhouette 0.17).

Features for fighter A facing B (mirrored for B; the harness takes red minus blue):

  "How has A fared against opponents like B?"  Each earlier opponent O_j of A is scored
  s_j = cos(O_j, B), with O_j's style as known at tonight's date.
    like_n60, like_n75      earlier opponents at least 0.60 / 0.75 similar to B
    like_max                the closest A has come to facing B's style
    like_resid_p1, _p4      sum w_j (won_j - elo_expected_j) / (sum w_j + K), w = max(s, 0)^p
    like_top3_resid         the same over A's 3 most similar opponents, unweighted
    like_xs_{sig,td,ctrl}   the w(p=4)-weighted mean of A's actual-minus-expected stats

  "Has anyone shaped like A fought anyone shaped like B?"  Every earlier UFC bout X vs Y
  (both orientations), with both fingerprints as of that bout:
    analog_resid_p1, _p4    sum w (won_X - elo_expected_X) / (sum w + K),
                            w = (max(cos(X, A), 0) * max(cos(Y, B), 0))^p
    analog_neff_p4          effective number of analog bouts, (sum w)^2 / sum w^2

  Context: cos (A vs B, fight-level), atyp (A's distance from the division-era centre).

Thresholds (0.60/0.75), p in {1, 4} and K = 3 were fixed before any results were seen.
"""
from __future__ import annotations

import bisect
import logging
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import pandas as pd

from app.services.ufc import style_habits
from app.services.ufc.fighter_registry import classify_weight_class
from app.services.ufc.style_service import fight_weighted_prime, fit_space, transform

log = logging.getLogger("style_pit")

PREFIX = "sty_"
FIGHT_LEVEL = ("sty_cos", "sty_analog_neff_p4")
RAW = ()

FIRST_BOUNDARY = date(2008, 1, 1)
MIN_BOUTS = 2            # prior bouts for a fingerprint at all
MIN_BOUTS_FIT = 3        # prior bouts to enter a space fit
FIT_ACTIVE_YEARS = 4     # ... and fought within this many years of the fit date
MIN_DIVISION_FIT = 30    # fighters for a division to get its own scaling
K = 3.0
#: ~90th / 97th percentile of cos between matched opponents (set from the distribution of
#: cosines on 2026-10-26, before any outcome was looked at; 0.70 / 0.80 were too rare)
THRESHOLDS = (0.60, 0.75)
XS_STATS = ("sig", "td", "ctrl")


def _boundaries(last: date, months: int) -> list[date]:
    out, d = [], FIRST_BOUNDARY
    while d <= last:
        out.append(d)
        m = d.month - 1 + months
        d = date(d.year + m // 12, m % 12 + 1, 1)
    return out


def _as_date(x) -> date:
    """datetime / Timestamp -> date; dates pass through."""
    from datetime import datetime
    return x.date() if isinstance(x, datetime) else x


class Fingerprints:
    """Fighter fingerprints as of any date, with the quarter's opponent model and the year's
    space. Cached by (fighter, number of prior bouts, quarter)."""

    def __init__(self, bouts: dict[int, list[style_habits.Bout]], division_of: dict[int, str]):
        self.bouts = {f: [b for b in bs if b.date is not None] for f, bs in bouts.items()}
        self.dates = {f: [_as_date(b.date) for b in bs] for f, bs in self.bouts.items()}
        self.division_of = division_of
        last = max(d for ds in self.dates.values() for d in ds)
        flat = sorted((b for bs in self.bouts.values() for b in bs), key=lambda b: _as_date(b.date))
        flat_dates = [_as_date(b.date) for b in flat]
        self.models = {}
        for q in _boundaries(last, 3):
            prior = flat[:bisect.bisect_left(flat_dates, q)]
            by_f = defaultdict(list)
            for b in prior:
                by_f[b.fighter].append(b)
            self.models[q] = style_habits.fit_opponent_model(by_f) if prior else None
        self.quarters = sorted(self.models)
        self._raw: dict = {}
        self.spaces = {}
        for y in _boundaries(last, 12):
            self.spaces[y] = self._fit_space(y)
        self.years = sorted(self.spaces)

    def n_prior(self, f: int, d: date) -> int:
        return bisect.bisect_left(self.dates.get(f, []), d)

    def _raw_at(self, f: int, d: date):
        """(raw feature vector, prime, division) from f's bouts before d, or None."""
        n = self.n_prior(f, d)
        if n < MIN_BOUTS:
            return None
        i = bisect.bisect_right(self.quarters, d) - 1
        if i < 0:
            return None
        q = self.quarters[i]
        key = (f, n, q)
        if key not in self._raw:
            bs = self.bouts[f][:n]
            model = self.models[q]
            if model is None:
                self._raw[key] = None
            else:
                feats = style_habits.style_features(bs, model)
                x = np.array([feats[c] for c in style_habits.STYLE_FEATURES], dtype=float)
                divs = [self.division_of.get(b.fight_id, "unknown") for b in bs[-5:]]
                # never "_all": fit_space would overwrite the league-wide scaling with it
                divs = [v for v in divs if v != "unknown"] or ["_unknown"]
                self._raw[key] = (x, fight_weighted_prime(self.dates[f][:n]),
                                  Counter(divs).most_common(1)[0][0])
        return self._raw[key]

    def _fit_space(self, y: date):
        rows = []
        for f, ds in self.dates.items():
            n = bisect.bisect_left(ds, y)
            if n >= MIN_BOUTS_FIT and (y - ds[n - 1]).days <= 365 * FIT_ACTIVE_YEARS:
                r = self._raw_at(f, y)
                if r is not None:
                    rows.append(r)
        if len(rows) < 100:
            return None
        X = np.vstack([r[0] for r in rows])
        divs = [r[2] for r in rows]
        space = fit_space(X, np.array([r[1] for r in rows]), divs)
        # a division with a handful of fighters has a near-zero spread and blows the scaled
        # values up; those fall back to the league-wide scaling in vec()
        counts = Counter(divs)
        g_sd = space["division_stats"]["_all"][1]
        for dv in list(space["division_stats"]):
            if dv == "_all":
                continue
            if dv == "_unknown" or counts.get(dv, 0) < MIN_DIVISION_FIT:
                del space["division_stats"][dv]
            else:
                mean, sd = space["division_stats"][dv]
                space["division_stats"][dv] = (mean, np.maximum(sd, 0.25 * g_sd))
        return space

    def _space_at(self, d: date):
        i = bisect.bisect_right(self.years, d) - 1
        # only spaces fit at or before d; never fall forward to a later fit
        for y in reversed(self.years[:i + 1]):
            if self.spaces[y] is not None:
                return self.spaces[y]
        return None

    def vec(self, f: int, d: date):
        """(unit vector, distance from centre) as of d, or None."""
        r = self._raw_at(f, d)
        space = self._space_at(d)
        if r is None or space is None:
            return None
        div = r[2] if r[2] in space["division_stats"] else "_all"
        z = transform(r[0][None, :], np.array([r[1]]), [div], space)[0]
        norm = float(np.linalg.norm(z))
        return (z / norm if norm > 0 else z), norm


def features_from(bouts: dict[int, list[style_habits.Bout]], fighters: pd.DataFrame) -> pd.DataFrame:
    """The pure core. `fighters` has one row per fighter per fight: fight_id, stats_fighter_id,
    date, red_fighter_id, blue_fighter_id, won, elo_expected, optional xsres_*. Returns the
    block keyed (fight_id, fighter_id) for the fights in `fighters`."""
    fr = fighters.copy()
    fr["d"] = pd.to_datetime(fr["date"]).dt.date
    fr["opp"] = np.where(fr["stats_fighter_id"] == fr["red_fighter_id"], fr["blue_fighter_id"], fr["red_fighter_id"])
    division_of = {}
    if "weight_class" in fr:
        division_of = {int(k): classify_weight_class(v) for k, v in
                       fr.drop_duplicates("fight_id").set_index("fight_id")["weight_class"].items()}
    fp = Fingerprints(bouts, division_of)

    # per (fight, fighter): pre-fight residual and process residuals, for the history lookups
    fr["resid"] = fr["won"].astype(float) - fr["elo_expected"].astype(float)
    if "winner_id" in fr:
        fr.loc[fr["winner_id"].isna(), "resid"] = np.nan     # draws / no contests
    xs_cols = [f"xsres_{s}" for s in XS_STATS if f"xsres_{s}" in fr]
    info = fr.set_index(["fight_id", "stats_fighter_id"])[["resid", *xs_cols]].to_dict("index")

    # Analog bouts: every played bout, both orientations, fingerprints as of the bout.
    an_rows = []
    for r in fr.itertuples(index=False):
        if pd.isna(r.resid):
            continue
        vx, vy = fp.vec(int(r.stats_fighter_id), r.d), fp.vec(int(r.opp), r.d)
        if vx is None or vy is None:
            continue
        an_rows.append((r.d, vx[0], vy[0], float(r.resid)))
    an_rows.sort(key=lambda t: t[0])
    an_dates = [t[0] for t in an_rows]
    UX = np.vstack([t[1] for t in an_rows]) if an_rows else np.zeros((0, len(style_habits.STYLE_FEATURES)))
    UY = np.vstack([t[2] for t in an_rows]) if an_rows else UX
    AR = np.array([t[3] for t in an_rows])
    log.info("  %d analog bout orientations with fingerprints", len(an_rows))

    hist = defaultdict(list)          # fighter -> [(date, fight_id, opponent)]
    for r in fr.sort_values("d").itertuples(index=False):
        hist[int(r.stats_fighter_id)].append((r.d, int(r.fight_id), int(r.opp)))

    out = []
    for r in fr.itertuples(index=False):
        a, b, d = int(r.stats_fighter_id), int(r.opp), r.d
        row = {"fight_id": int(r.fight_id), "fighter_id": a}
        va, vb = fp.vec(a, d), fp.vec(b, d)
        if va is not None:
            row["sty_atyp"] = va[1]
        if va is not None and vb is not None:
            row["sty_cos"] = float(va[0] @ vb[0])
        if vb is not None:
            ub = vb[0]
            sims, res, xs = [], [], []
            for d_j, fid_j, o_j in hist[a]:
                if d_j >= d:
                    break
                vo = fp.vec(o_j, d)
                if vo is None:
                    continue
                inf = info.get((fid_j, a), {})
                sims.append(float(vo[0] @ ub))
                res.append(inf.get("resid", np.nan))
                xs.append([inf.get(c, np.nan) for c in xs_cols])
            if sims:
                s = np.array(sims); rr = np.array(res, dtype=float)
                for t in THRESHOLDS:
                    row[f"sty_like_n{int(t * 100)}"] = float((s >= t).sum())
                row["sty_like_max"] = float(s.max())
                ok = ~np.isnan(rr)
                for p in (1, 4):
                    w = np.clip(s, 0, None) ** p
                    row[f"sty_like_resid_p{p}"] = float((w[ok] * rr[ok]).sum() / (w[ok].sum() + K))
                top = np.argsort(-s)[:3]
                top = top[ok[top]]
                row["sty_like_top3_resid"] = float(rr[top].sum() / (len(top) + K))
                if xs_cols:
                    X = np.array(xs, dtype=float)
                    w = np.clip(s, 0, None) ** 4
                    for k, c in enumerate(xs_cols):
                        m = ~np.isnan(X[:, k])
                        if w[m].sum() > 0:
                            row[f"sty_like_xs_{c.split('_', 1)[1]}"] = float((w[m] * X[m, k]).sum() / (w[m].sum() + K))
            else:
                for t in THRESHOLDS:
                    row[f"sty_like_n{int(t * 100)}"] = 0.0
        if va is not None and vb is not None and an_rows:
            m = bisect.bisect_left(an_dates, d)
            if m:
                sa = np.clip(UX[:m] @ va[0], 0, None)
                sb = np.clip(UY[:m] @ vb[0], 0, None)
                base = sa * sb
                for p in (1, 4):
                    w = base ** p
                    row[f"sty_analog_resid_p{p}"] = float((w * AR[:m]).sum() / (w.sum() + K))
                w = base ** 4
                row["sty_analog_neff_p4"] = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
        out.append(row)
    return pd.DataFrame(out)


def compute(ctx) -> pd.DataFrame:
    db = ctx.session()
    try:
        bouts = style_habits.load_bouts(db)
    finally:
        db.close()
    fighters = ctx.fighters
    fighters = fighters[~fighters.get("_unplayed", pd.Series(False, index=fighters.index)).fillna(False).astype(bool)]
    return features_from(bouts, fighters)
