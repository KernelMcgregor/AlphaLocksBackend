"""Decorrelation-penalised model — trading accuracy for independence from the market.

THE ARGUMENT
------------
Hubacek & Sir (Int. J. Forecasting 2022, arXiv:2010.12508) prove that a bettor whose
estimates coincide with the market's has EXACTLY ZERO profitability, regardless of how
accurate either is (their Example 3.3). Profit requires the market to err AND the bettor
to err in the opposite direction — their Definition 3.1, in which accuracy does not
appear. Maximum profitability is at Corr[T, M | R] = -1.

Measured on this repo's walk-forward: `market_anchored` scores 68.4% with correlation
0.95 to the closing line, and loses ~7% at every betting threshold — the vig showing
through because no independent signal remains. Meanwhile `no_odds` is decorrelated (0.57)
but 5 points less accurate, so it errs in the wrong direction and also loses.

We have accuracy OR independence, never both. This module makes the trade CONTROLLED,
starting from a high-accuracy model and giving up only as much accuracy as buys real
independence. In the authors' NBA experiment, deliberately surrendering 1.3 accuracy
points (68.8% -> 67.5%) turned their worst result into their best.

THE LOSS (their Equation 59, verbatim)
--------------------------------------
    MSE*(R, M, T) = (1/|omega|) * SUM [ (t_i - r_i)^2 + gamma * (t_i - r_i)(m_i - r_i) ]

with gamma > 0, where t = our estimate, r = the realised outcome, m = the market price.
The penalty is the COVARIANCE OF RESIDUALS, not a correlation. That distinction is the
whole ballgame:

    d/dt [ (t-r)^2 + gamma*(t-r)(m-r) ] = 0   =>   t = r - gamma*(m-r)/2

The optimum is the truth, pushed away from the market in proportion to the market's OWN
error, and the objective stays convex in t for every gamma. A normalised correlation
penalty has no such anchor -- it is scale-free, so a model can drive correlation to -1
while being useless. Measured here on synthetic data: a correlation penalty at gamma=0.3
collapsed accuracy from 0.86 to 0.40 (below chance) with corr -0.66. Equation 59 does not
do that, because pushing t away from r is itself penalised by the (t-r)^2 term.

The authors are explicit that this "might seem counter-intuitive... the additional term
will only hurt performance by pushing it away, not only from the market price, but from
the true value, too". Losing accuracy is the mechanism, not a bug.

The penalty needs the REALISED outcome r, so it is a training-time-only term; nothing
changes at inference.

STAKING, AND WHY IT IS LOAD-BEARING
-----------------------------------
The authors' own Section 3.1 concedes that under growth-optimal (Kelly) staking, profit
STILL requires strictly lower cross-entropy than the market. The decorrelation result
exists only under flat/uniform stakes. Evaluate this arm with flat stakes or the effect
is mathematically guaranteed to vanish.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("model")

EPS = 1e-6


def _logit_t(p: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(p, EPS, 1 - EPS)
    return torch.log(p / (1 - p))


def partial_corr_given_y(a: torch.Tensor, b: torch.Tensor,
                         y: torch.Tensor) -> torch.Tensor:
    """DIAGNOSTIC ONLY -- not the training loss (that is Equation 59, see module docstring).

    Correlation of `a` and `b` after removing what the binary outcome `y` explains.

    Residualising on a binary variable is just centring within each class. Returns 0 when
    either side has no within-class variance, so a degenerate batch contributes no
    gradient rather than a NaN.
    """
    out = a.new_zeros(())
    total = 0.0
    for cls in (0.0, 1.0):
        m = (y == cls)
        n = int(m.sum())
        if n < 8:
            continue
        av = a[m] - a[m].mean()
        bv = b[m] - b[m].mean()
        denom = torch.sqrt((av * av).sum() * (bv * bv).sum())
        if float(denom) < EPS:
            continue
        out = out + n * ((av * bv).sum() / denom)
        total += n
    return out / total if total > 0 else out


class _MLP(nn.Module):
    """Deliberately small: ~5,000 training rows does not support anything wider."""

    def __init__(self, n_in: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DecorrelatedModel:
    """MLP trained with a market-decorrelation penalty.

    `gamma=0` reduces exactly to plain BCE, which makes it its own control: the
    gamma-sweep's first rung is the undecorrelated baseline.
    """

    def __init__(self, gamma: float = 0.4, hidden: int = 64, dropout: float = 0.3,
                 lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 200,
                 batch_size: int = 512, patience: int = 20, seed: int = 42):
        self.gamma = gamma
        self.hidden, self.dropout = hidden, dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.seed = seed
        self.model_ = None
        self.mu_ = self.sd_ = None
        self.best_epoch_ = 0

    # -- scaling: NNs need it, unlike the GBT this replaces --------------------
    def _fit_scaler(self, X):
        self.mu_ = np.nanmean(X, axis=0)
        self.sd_ = np.nanstd(X, axis=0)
        self.sd_[self.sd_ < 1e-9] = 1.0

    def _prep(self, X):
        Xs = (np.nan_to_num(X, nan=0.0) - self.mu_) / self.sd_
        return torch.tensor(np.clip(Xs, -10, 10), dtype=torch.float32)

    def fit(self, X, y, market, X_val=None, y_val=None, market_val=None):
        """`market` is the de-vigged market P(red), NaN where the fight was unpriced."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._fit_scaler(X)
        Xt = self._prep(X)
        yt = torch.tensor(np.asarray(y, dtype=np.float32))
        mt = torch.tensor(np.nan_to_num(np.asarray(market, dtype=np.float32), nan=0.5))
        has_mkt = torch.tensor(np.isfinite(np.asarray(market, dtype=np.float64)))

        self.model_ = _MLP(Xt.shape[1], self.hidden, self.dropout)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        # Equation 59 is defined on probabilities and squared error, not logits/BCE.

        use_val = X_val is not None and len(X_val) > 0
        if use_val:
            Xv, yv = self._prep(X_val), torch.tensor(np.asarray(y_val, dtype=np.float32))

        n = len(Xt)
        best_val, best_state, since = float("inf"), None, 0

        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 32:
                    continue
                p = torch.sigmoid(self.model_(Xt[idx]))
                yb = yt[idx]
                loss = ((p - yb) ** 2).mean()          # the MSE / Brier term

                if self.gamma > 0:
                    # Penalty only over priced rows -- an unpriced fight has no market
                    # opinion whose residual ours could be redundant with.
                    sub = has_mkt[idx]
                    if int(sub.sum()) >= 32:
                        rt = p[sub] - yb[sub]           # our residual
                        rm = mt[idx][sub] - yb[sub]     # the market's residual
                        loss = loss + self.gamma * (rt * rm).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            if use_val:
                self.model_.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(self.model_(Xv))
                    vl = float(((pv - yv) ** 2).mean())
                # Early stopping tracks the UNPENALISED Brier, not the full objective:
                # we want the epoch that predicts best, then let gamma decide how far
                # from the market to sit. Stopping on the penalised loss would simply
                # select whichever epoch drifted furthest from the market.
                if vl < best_val - 1e-5:
                    best_val, since = vl, 0
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                    self.best_epoch_ = epoch
                else:
                    since += 1
                    if since >= self.patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model_(self._prep(X))).numpy().astype(float)

    # -- persistence -----------------------------------------------------------
    # Until this existed the MLP was trained inside the walk-forward harness, measured,
    # and discarded -- there was no way to actually serve it. Everything needed to
    # reproduce a prediction is stored: weights, the feature ORDER (a different order
    # silently produces garbage), and the scaler/imputation constants fitted on train.

    def save(self, path, features: list[str], train_means=None) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "kind": "decorrelated_mlp",
                "version": 1,
                "state_dict": self.model_.state_dict(),
                "n_in": self.model_.net[0].in_features,
                "hidden": self.hidden,
                "dropout": self.dropout,
                "gamma": self.gamma,
                "mu": self.mu_,
                "sd": self.sd_,
                "features": list(features),
                "train_means": train_means,
                "best_epoch": self.best_epoch_,
            }, f)

    @classmethod
    def load(cls, path) -> tuple["DecorrelatedModel", dict]:
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d.get("kind") != "decorrelated_mlp":
            raise ValueError(f"{path} is not a decorrelated MLP artifact")
        m = cls(gamma=d["gamma"], hidden=d["hidden"], dropout=d["dropout"])
        m.model_ = _MLP(d["n_in"], d["hidden"], d["dropout"])
        m.model_.load_state_dict(d["state_dict"])
        m.model_.eval()
        m.mu_, m.sd_ = d["mu"], d["sd"]
        m.best_epoch_ = d.get("best_epoch", 0)
        return m, d
