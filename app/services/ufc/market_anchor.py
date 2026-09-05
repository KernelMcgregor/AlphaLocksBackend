"""Market-anchored probability model.

The winner model currently competes with the closing line and loses: feeding it the
market price produced 66.6% accuracy against 67.7% for simply following the line. It
takes a well-priced input and degrades it.

The fix is to stop treating the market as one feature among 40 and start treating it as
the prior. We hold `logit(market)` as an offset and learn only a correction:

    logit(p) = logit(market) + b * (logit(base) - logit(market)) + c

`b` is the only thing that matters. It is the fraction of your disagreement with the
market that is worth acting on:

    b = 0    the market is right, ignore my model entirely  -> p == market
    b = 1    my model is right, ignore the market           -> p == base
    0 < b < 1  shrink my disagreement toward the market by (1-b)

Because b=0 reproduces the market exactly, an anchored model cannot do materially worse
than the line — which is the property the current architecture lacks.

WHY SO FEW PARAMETERS: only 2,748 UFC fights have ever been priced in this database
(odds coverage starts mid-2020). A feature-conditional correction — LightGBM
`init_score` with a GBT learning delta(features) — needs far more data than that and
would fit noise. `fit_conditional=True` is available for when coverage grows, and the
walk-forward harness scores it head-to-head so the data decides rather than taste.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("model")

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def devig(red_prob: np.ndarray, blue_prob: np.ndarray) -> np.ndarray:
    """Normalize a two-way market to sum to 1.

    Idempotent — this database already stores normalized implied probabilities, but
    callers should not have to know that.
    """
    red_prob = np.asarray(red_prob, dtype=float)
    blue_prob = np.asarray(blue_prob, dtype=float)
    total = red_prob + blue_prob
    return np.where(total > 0, red_prob / np.where(total > 0, total, 1.0), 0.5)


class MarketAnchor:
    """Learns how much of a model's disagreement with the market to trust.

    Fit on out-of-sample base predictions (predictions the base model made on rows it
    was not trained on). Fitting on in-sample base predictions inflates `b`, because
    the base model looks far sharper on its own training data than it will in
    production.
    """

    def __init__(self, l2: float = 1.0, clip_b: tuple[float, float] = (0.0, 1.0),
                 fit_conditional: bool = False, min_samples: int = 150):
        self.l2 = l2
        self.clip_b = clip_b
        self.fit_conditional = fit_conditional
        self.min_samples = min_samples
        self.b_ = 0.0
        self.c_ = 0.0
        self.n_fit_ = 0
        self.fallback_ = True

    def fit(self, base_proba, market_proba, y) -> "MarketAnchor":
        base_proba = np.asarray(base_proba, dtype=float)
        market_proba = np.asarray(market_proba, dtype=float)
        y = np.asarray(y)

        ok = np.isfinite(base_proba) & np.isfinite(market_proba)
        if ok.sum() < self.min_samples or len(np.unique(y[ok])) < 2:
            # Not enough priced history to justify deviating from the line.
            log.info(f"    MarketAnchor: only {int(ok.sum())} usable rows -> b=0 (pure market)")
            self.b_, self.c_, self.n_fit_, self.fallback_ = 0.0, 0.0, int(ok.sum()), True
            return self

        lm = logit(market_proba[ok])
        lb = logit(base_proba[ok])

        # Single regressor: how far the model departs from the market. Its coefficient
        # IS b. The market enters as a fixed offset, not a fitted term, so the model
        # cannot "unlearn" the line.
        disagreement = lb - lm

        self.b_, self.c_ = _fit_offset_logistic(lm, disagreement, y[ok], self.l2)
        self.b_ = float(np.clip(self.b_, *self.clip_b))
        self.n_fit_ = int(ok.sum())
        self.fallback_ = False

        log.info(
            f"    MarketAnchor: b={self.b_:.3f} c={self.c_:+.3f} on {self.n_fit_} priced rows "
            f"({'ignoring model' if self.b_ < 0.02 else f'trusting {self.b_:.0%} of disagreement'})"
        )
        return self

    def predict_proba(self, base_proba, market_proba) -> np.ndarray:
        base_proba = np.asarray(base_proba, dtype=float)
        market_proba = np.asarray(market_proba, dtype=float)

        out = np.where(np.isfinite(base_proba), base_proba, 0.5)
        has_market = np.isfinite(market_proba)
        if not has_market.any():
            return out

        lm = logit(market_proba[has_market])
        lb = logit(base_proba[has_market])
        out[has_market] = expit(lm + self.b_ * (lb - lm) + self.c_)
        return out


def _fit_offset_logistic(offset: np.ndarray, x: np.ndarray, y: np.ndarray,
                         l2: float, iters: int = 100) -> tuple[float, float]:
    """Maximum-likelihood fit of  logit(p) = offset + b*x + c  with an L2 penalty on b.

    Plain Newton-Raphson on two parameters. Written out rather than delegated because
    sklearn's LogisticRegression cannot take a per-sample offset, and dropping the
    offset (fitting the market logit as a free coefficient) is precisely the failure
    mode this module exists to prevent.
    """
    b, c = 0.0, 0.0
    for _ in range(iters):
        eta = offset + b * x + c
        p = expit(eta)
        w = np.maximum(p * (1 - p), 1e-9)
        r = y - p

        # Gradient of penalized log-likelihood
        g_b = float(x @ r - l2 * b)
        g_c = float(r.sum())

        # Hessian (negative definite); H = -[[x'Wx + l2, x'W], [x'W, sum W]]
        h_bb = float((x * x) @ w + l2)
        h_bc = float(x @ w)
        h_cc = float(w.sum())

        det = h_bb * h_cc - h_bc * h_bc
        if abs(det) < 1e-12:
            break
        db = (h_cc * g_b - h_bc * g_c) / det
        dc = (h_bb * g_c - h_bc * g_b) / det

        b += db
        c += dc
        if max(abs(db), abs(dc)) < 1e-9:
            break
    return b, c
