"""Pre-registered betting picks: the frozen selection rule and its model.

READ PREREGISTRATION.md BEFORE CHANGING ANYTHING IN THIS FILE.

The rule below was chosen by searching the walk-forward eval set, which means its
backtested +14.2% ROI is an in-sample rule-selection figure and is NOT evidence of an
edge. The only thing that can turn it into evidence is applying it unchanged, going
forward, to cards it was never fitted on -- and that only works if the parameters stop
moving. Every constant in the RULE block is frozen as of 2026-09-06.

WHY A SEPARATE MODEL FROM THE ONE THE SITE SERVES
-------------------------------------------------
Production serves `mlp_v1.pkl`, which takes the betting odds as input features. That
model cannot be used to measure edge against the market: it has already seen the line,
so `model_prob - market_prob` is partly the model echoing its own input rather than an
independent disagreement. Measured on the walk-forward eval, the odds-using model is
0.83 correlated with the closing line.

Picks therefore use their own artifact -- a gradient-boosted tree fit with
`include_odds=False`, so the market price is never in its feature set. This matches the
`no_odds` walk-forward arm the rule was measured on, so the live results are comparable
to the backtest rather than to a different model that happens to be nearby.

WHY THE RULE HAS TWO GATES
--------------------------
Betting wherever the model disagrees with the market is catastrophic, because the model's
confident disagreements are its worst predictions:

    edge > 0.05 anywhere    n=757   accuracy 0.464   ROI  -0.35%
    edge > 0.10 anywhere    n=560   accuracy 0.418   ROI  -2.87%
    edge > 0.20 anywhere    n=218   accuracy 0.294   ROI -14.69%

Accuracy falls further below chance the harder the model argues. Contradicting a
confident market is where this model is reliably wrong.

The second gate is what changes the sign. Restricting to fights the market itself cannot
call -- where the de-vigged line sits near 50/50, so the market has no strong opinion to
contradict -- leaves room for a weaker but genuinely independent signal:

    near-even only              n=497   accuracy 0.575   ROI  +8.57%
    near-even AND edge > 0.05   n=321   accuracy 0.586   ROI +14.16%   CI [+3.3, +24.7]

Both gates are necessary. Neither alone is enough.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.ufc.market_anchor import devig

log = logging.getLogger("picks")

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "ufc" / "h2h"
PICKS_MODEL_PATH = MODEL_DIR / "picks_noodds_gbt_v1.pkl"

# ===========================================================================
# THE RULE -- FROZEN 2026-09-06. Do not tune. See PREREGISTRATION.md.
# ===========================================================================
RULE_VERSION = "v1.0-2026-09-06"

#: How near 50/50 the de-vigged market price must be for a fight to qualify.
#: 0.08 admits roughly the middle 24% of priced fights (~70/year).
MARKET_UNCERTAINTY_MAX = 0.08

#: How far the model's probability must exceed the market's on the side it picks.
MIN_EDGE = 0.05

#: Flat stake. Fractional Kelly is deliberately NOT used: the decorrelation result this
#: work grew out of (Hubacek & Sir 2022, sec. 3.1) holds only under uniform staking, and
#: Kelly sizing on an uncalibrated model with an unproven edge compounds estimation error.
FLAT_STAKE = 100.0


@dataclass(frozen=True)
class Pick:
    fight_id: int
    side: str            # 'red' or 'blue'
    model_prob: float    # model's probability for the picked side
    market_prob: float   # de-vigged market probability for the picked side
    edge: float          # model_prob - market_prob
    decimal_odds: float
    market_uncertainty: float  # |devigged red prob - 0.5|


def american_to_decimal(a: float) -> float:
    a = float(a)
    return 1.0 + 100.0 / (-a) if a < 0 else 1.0 + a / 100.0


def american_to_prob(a: float) -> float:
    a = float(a)
    return (-a) / ((-a) + 100.0) if a < 0 else 100.0 / (a + 100.0)


def select_picks(
    fight_ids: np.ndarray,
    model_red_prob: np.ndarray,
    red_american: np.ndarray,
    blue_american: np.ndarray,
) -> list[Pick]:
    """Apply the frozen rule. This function is the rule; keep it small and literal.

    A fight qualifies when BOTH gates pass:
      1. |devig(market) - 0.5| < MARKET_UNCERTAINTY_MAX   -- the market has no strong view
      2. model_prob(picked side) - market_prob(picked side) > MIN_EDGE

    The picked side is whichever one the model favours, decided before the edge test, so
    the rule can never "shop" for whichever side happens to clear the threshold.
    """
    picks: list[Pick] = []
    for fid, p_model, ra, ba in zip(fight_ids, model_red_prob, red_american, blue_american):
        if not np.isfinite(p_model) or ra is None or ba is None:
            continue
        if not (np.isfinite(float(ra)) and np.isfinite(float(ba))):
            continue

        pr, pb = american_to_prob(ra), american_to_prob(ba)
        mkt_red = devig(np.array([pr]), np.array([pb]))[0]
        uncertainty = abs(mkt_red - 0.5)
        if uncertainty >= MARKET_UNCERTAINTY_MAX:
            continue

        side = "red" if p_model >= 0.5 else "blue"
        p_side = p_model if side == "red" else 1.0 - p_model
        m_side = mkt_red if side == "red" else 1.0 - mkt_red
        edge = p_side - m_side
        if edge <= MIN_EDGE:
            continue

        picks.append(Pick(
            fight_id=int(fid), side=side,
            model_prob=float(p_side), market_prob=float(m_side), edge=float(edge),
            decimal_odds=american_to_decimal(ra if side == "red" else ba),
            market_uncertainty=float(uncertainty),
        ))
    return picks


def load_picks_model() -> dict:
    if not PICKS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{PICKS_MODEL_PATH} missing. Train it with:\n"
            "  python -m app.services.ufc.picks --train"
        )
    with open(PICKS_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def train_picks_model(fresh_glicko: bool = True) -> dict:
    """Fit the no-odds GBT used for picks, on every decided fight, and persist it.

    Deliberately mirrors the `no_odds` walk-forward arm: same learner, same feature
    selection with `include_odds=False`, same chronological early stopping. The rule's
    backtest is only meaningful if the live model is the same kind of model.
    """
    from app.services.ufc.model import (
        _fillna_from_train, build_features, build_matchup_df, fit_gbt,
        load_fight_data, run_glicko_inmemory, select_winner_features,
    )

    log.info("=" * 60)
    log.info("TRAINING PICKS MODEL (no odds features)")
    log.info("=" * 60)

    snaps = run_glicko_inmemory() if fresh_glicko else None
    df, round_data = load_fight_data()
    probe = build_matchup_df(build_features(df.copy(), round_data.copy(),
                                            glicko_snapshots=snaps))[0]
    cutoff = probe.sort_values("date")["date"].iloc[int(len(probe) * 0.6)]
    df = build_features(df, round_data, style_cutoff_date=cutoff, glicko_snapshots=snaps)
    matchup, features = build_matchup_df(df)
    matchup = matchup.sort_values("date").reset_index(drop=True)
    matchup = matchup[matchup["red_wins"].notna()].reset_index(drop=True)

    n = len(matchup)
    train_mask = np.ones(n, dtype=bool)
    matchup, train_means = _fillna_from_train(matchup, features, train_mask)
    selected = select_winner_features(matchup, features, train_mask,
                                      top_n=39, include_odds=False)

    odds_leaked = [f for f in selected if "odds" in f.lower()]
    if odds_leaked:
        raise RuntimeError(
            f"Odds features reached the picks model: {odds_leaked}. The edge measure "
            "would be circular. Refusing to save."
        )

    X = matchup[selected].to_numpy(dtype=float)
    y = matchup["red_wins"].to_numpy(dtype=int)
    log.info(f"  Train {n} fights ({matchup['date'].min()} to {matchup['date'].max()})")
    log.info(f"  Features: {len(selected)} (0 odds, verified)")

    model, best_iter = fit_gbt(X, y, selected)

    with open(PICKS_MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model, "features": selected, "train_means": train_means,
            "best_iter": best_iter, "rule_version": RULE_VERSION, "include_odds": False,
        }, f)
    log.info(f"  Saved to {PICKS_MODEL_PATH}")
    return {"model": model, "features": selected, "train_means": train_means}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="fit and save the picks model")
    a = ap.parse_args()
    if a.train:
        train_picks_model()
    else:
        ap.print_help()
