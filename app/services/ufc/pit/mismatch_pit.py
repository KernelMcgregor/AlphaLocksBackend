"""Step 3C: mismatch features -- how big is the quality gap, and how sure are we?

The model ranks fights worse than the market (AUC 0.70 vs 0.74) and is therefore calibrated
toward 50% on big favourites. The only non-linear gap input today is elo_expected (UFC Elo,
which starts every debutant at 1500). These combine every quality source with its certainty:

  mm_q_best        P(win) blending the UFC Elo expectation and the Sherdog Glicko expectation
                   (pro2 rating + RD), weighted by UFC experience w = n / (n + K_BLEND) with n
                   the lesser UFC fight count of the pair: newcomers lean on Sherdog, veterans
                   on UFC Elo
  mm_glicko_z      glicko_pts gap / sqrt(sigma_a^2 + sigma_b^2 + S0^2): the gap in units of
                   its own uncertainty (glicko_pts has its own scale, so no win formula)
  mm_gap_x_exp     |UFC Elo gap| x log(1 + lesser UFC fight count)
  mm_progap_cert   pro2 rating gap x certainty (1 - combined RD / RD_MAX)
  mm_opp_beaten    opponent's best win minus this fighter's rating (pro2 scale): has the
                   opponent beaten someone better than me?
  mm_lost_to_weak  this fighter's worst loss minus the opponent's rating: have I lost to
                   someone weaker than tonight's opponent?
  fight_mm_elo_sq, fight_mm_pro_sq   squared gaps (orientation-free) for the linear member

Needs the pro block (feature_wf pro, data/feature_wf/pro_block.pkl) and newcomer experience.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.ufc.pit.newcomer_pit import dev_fight_ids, experience

PREFIX = "mm_"
K_BLEND = 3.0
S0 = 10.0
RD_MAX = 300.0 * np.sqrt(2)
PRO_BLOCK = "data/feature_wf/pro_block.pkl"


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def corner_values(matchup, slim, pro, exp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per fight, red and blue frames of the inputs, aligned to matchup.fight_id."""
    cols = ["fight_id", "stats_fighter_id", "corner", "elo", "elo_expected", "glicko_pts", "glicko_meta_sigma"]
    s = slim[cols].drop_duplicates(["fight_id", "corner"])
    s = s.merge(pro.rename(columns={"fighter_id": "stats_fighter_id"}), on=["fight_id", "stats_fighter_id"], how="left")
    s = s.merge(exp.rename(columns={"fighter_id": "stats_fighter_id"}), on=["fight_id", "stats_fighter_id"], how="left")
    red = s[s.corner == "red"].set_index("fight_id").reindex(matchup.fight_id)
    blue = s[s.corner == "blue"].set_index("fight_id").reindex(matchup.fight_id)
    return red, blue


def compute_features(red: pd.DataFrame, blue: pd.DataFrame) -> dict[str, np.ndarray]:
    n_min = np.minimum(red["nu_n_ufc"].to_numpy(float), blue["nu_n_ufc"].to_numpy(float))
    w = n_min / (n_min + K_BLEND)
    e_ufc = red["elo_expected"].to_numpy(float)
    gap = red["pro2_rating"].to_numpy(float) - blue["pro2_rating"].to_numpy(float)
    rd2 = red["pro2_rd"].to_numpy(float) ** 2 + blue["pro2_rd"].to_numpy(float) ** 2
    q = np.log(10) / 400
    g = 1 / np.sqrt(1 + 3 * q * q * rd2 / np.pi ** 2)
    e_pro = 1 / (1 + 10 ** (-g * gap / 400))
    z = np.where(np.isnan(e_pro), _logit(e_ufc), w * _logit(e_ufc) + (1 - w) * _logit(e_pro))
    out = {"q_best": 1 / (1 + np.exp(-z))}
    sig = np.sqrt(red["glicko_meta_sigma"].to_numpy(float) ** 2 + blue["glicko_meta_sigma"].to_numpy(float) ** 2 + S0 ** 2)
    out["glicko_z"] = (red["glicko_pts"].to_numpy(float) - blue["glicko_pts"].to_numpy(float)) / sig
    elo_gap = red["elo"].to_numpy(float) - blue["elo"].to_numpy(float)
    out["gap_x_exp"] = np.abs(elo_gap) * np.log1p(np.nan_to_num(n_min))
    out["progap_cert"] = gap * (1 - np.sqrt(rd2) / RD_MAX)
    out["elo_sq"] = (elo_gap / 100) ** 2
    out["pro_sq"] = (gap / 100) ** 2
    return out


def transform(matchup, features, slim, session_factory, variant: str = "all"):
    db = session_factory()
    try:
        exp = experience(slim, dev_fight_ids(db))
    finally:
        db.close()
    pro = pd.read_pickle(PRO_BLOCK)[["fight_id", "fighter_id", "pro2_rating", "pro2_rd", "pro2_best_win",
                                     "pro2_worst_loss"]]
    red, blue = corner_values(matchup, slim, pro, exp)
    f = compute_features(red, blue)
    new = []
    # oriented (red's view; blue's is the mirror), so they enter as red-minus-blue style diffs
    matchup["diff_mm_q_best"] = 2 * f["q_best"] - 1
    matchup["diff_mm_glicko_z"] = f["glicko_z"]
    matchup["diff_mm_progap_cert"] = f["progap_cert"]
    rr, br = red["pro2_rating"].to_numpy(float), blue["pro2_rating"].to_numpy(float)
    beaten_r = blue["pro2_best_win"].to_numpy(float) - rr     # blue has beaten someone vs red's level
    beaten_b = red["pro2_best_win"].to_numpy(float) - br
    lost_r = red["pro2_worst_loss"].to_numpy(float) - br
    lost_b = blue["pro2_worst_loss"].to_numpy(float) - rr
    matchup["diff_mm_opp_beaten"] = beaten_r - beaten_b
    matchup["diff_mm_lost_to_weak"] = lost_r - lost_b
    new += ["diff_mm_q_best", "diff_mm_glicko_z", "diff_mm_progap_cert", "diff_mm_opp_beaten", "diff_mm_lost_to_weak"]
    matchup["fight_mm_gap_x_exp"] = f["gap_x_exp"]
    matchup["fight_mm_elo_sq"] = f["elo_sq"]
    matchup["fight_mm_pro_sq"] = f["pro_sq"]
    new += ["fight_mm_gap_x_exp", "fight_mm_elo_sq", "fight_mm_pro_sq"]
    if variant == "q":           # the blended win probability alone
        for c in new[1:]:
            del matchup[c]
        new = new[:1]
    return new, []
