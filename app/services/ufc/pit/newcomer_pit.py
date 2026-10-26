"""Step 1: represent UFC newcomers properly (matrix transforms, no new data).

How a debutant reaches the model today (model.build_features):
  * no debut flag; career_win_pct = 0, which reads like an 0-N record;
  * every avg_/recent_/last3_ rolling stat is emitted only as red-minus-blue, so if EITHER
    fighter is debuting the diff is NaN and gets the training-mean fill -- the veteran's
    stats are thrown away too;
  * Sherdog pro_* only as diffs (an unlinked side blanks the whole diff);
  * DWCS / Road to UFC bouts sit in ufc_fights and count as UFC experience.

Variants (feature_wf --variant):
  flags    raw red_/blue_ nu_n_ufc (prior UFC fights excluding DWCS/RtU), nu_is_debut,
           nu_wpct = (W+1)/(n+2) over those fights, and raw pro_elo / pro_fights / pro_known
  impute   rebuild every rolling-stat diff after filling a fighter's missing value with the
           as-of mean of their division (fights on earlier dates only), so the veteran's side
           survives; no new columns
  both     flags + impute
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.ufc.fighter_registry import classify_weight_class

PREFIX = "nu_"
ROLLING = ("avg_", "recent_", "last3_")
DEV_SERIES = ("dwcs", "contender series", "road to ufc")


def dev_fight_ids(db) -> set[int]:
    """Fights on Contender Series / Road to UFC events."""
    from sqlalchemy import text
    from app.models.ufc import UFC_SCHEMA
    schema = f"{UFC_SCHEMA}." if UFC_SCHEMA else ""
    rows = db.execute(text(f"SELECT f.id, e.name FROM {schema}ufc_fights f "
                           f"JOIN {schema}ufc_events e ON e.id = f.event_id")).fetchall()
    return {int(r.id) for r in rows if any(k in (r.name or "").lower() for k in DEV_SERIES)}


def experience(slim: pd.DataFrame, dev: set[int]) -> pd.DataFrame:
    """Per (fight, fighter): prior UFC fights excluding development series, debut flag and a
    shrunk win rate over them. Earlier dates only."""
    s = slim[["fight_id", "stats_fighter_id", "date", "winner_id"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s["is_dev"] = s["fight_id"].isin(dev)
    s["win"] = (s["winner_id"] == s["stats_fighter_id"]).astype(float)
    s["decided"] = s["winner_id"].notna() & ~s["is_dev"]
    s = s.sort_values(["stats_fighter_id", "date", "fight_id"])
    g = s.groupby("stats_fighter_id")
    # counts strictly before this fight's date
    for col, val in (("n", (~s["is_dev"]).astype(float)), ("w", s["win"] * s["decided"]),
                     ("d", s["decided"].astype(float))):
        s[f"_{col}"] = val
        cum = g[f"_{col}"].cumsum() - s[f"_{col}"]
        # same-date rows (rare) must not see each other
        same = s.groupby(["stats_fighter_id", "date"])[f"_{col}"].transform("cumsum") - s[f"_{col}"]
        s[col] = cum - same
    return pd.DataFrame({"fight_id": s["fight_id"], "fighter_id": s["stats_fighter_id"],
                         "nu_n_ufc": s["n"], "nu_is_debut": (s["n"] == 0).astype(float),
                         "nu_wpct": (s["w"] + 1) / (s["d"] + 2)})


def division_prior(slim: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """As-of division mean of each column over rows on EARLIER dates (NaN-aware)."""
    s = slim[["fight_id", "stats_fighter_id", "date", "weight_class", *cols]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s["div"] = s["weight_class"].map(classify_weight_class)
    vals = s[cols]
    daily_sum = vals.fillna(0).groupby([s["div"], s["date"]]).sum()
    daily_cnt = vals.notna().astype(float).groupby([s["div"], s["date"]]).sum()
    cum_sum = daily_sum.groupby(level=0).cumsum() - daily_sum        # exclude the same date
    cum_cnt = daily_cnt.groupby(level=0).cumsum() - daily_cnt
    prior = (cum_sum / cum_cnt.replace(0, np.nan))
    idx = pd.MultiIndex.from_arrays([s["div"], s["date"]])
    out = prior.reindex(idx)
    out.index = s.index
    return out


def transform(matchup: pd.DataFrame, features: list[str], slim: pd.DataFrame, session_factory,
              variant: str = "both") -> tuple[list[str], list[str]]:
    """Mutates `matchup`; returns (new feature columns, replaced-in-place columns)."""
    corners = slim[["fight_id", "stats_fighter_id", "corner"]].drop_duplicates(["fight_id", "corner"])
    new, replaced = [], []

    def by_corner(block: pd.DataFrame, cols):
        b = block.merge(corners.rename(columns={"stats_fighter_id": "fighter_id"}), on=["fight_id", "fighter_id"])
        red, blue = b[b.corner == "red"].set_index("fight_id"), b[b.corner == "blue"].set_index("fight_id")
        for c in cols:
            matchup[f"red_{c}"] = matchup.fight_id.map(red[c])
            matchup[f"blue_{c}"] = matchup.fight_id.map(blue[c])
            new.extend([f"red_{c}", f"blue_{c}"])

    if variant in ("flags", "both"):
        db = session_factory()
        try:
            dev = dev_fight_ids(db)
        finally:
            db.close()
        by_corner(experience(slim, dev), ["nu_n_ufc", "nu_is_debut", "nu_wpct"])
        pro = slim[["fight_id", "stats_fighter_id"] + [c for c in ("pro_elo", "pro_fights", "pro_known") if c in slim]]
        pro = pro.rename(columns={"stats_fighter_id": "fighter_id", "pro_elo": "nu_pro_elo",
                                  "pro_fights": "nu_pro_fights", "pro_known": "nu_pro_known"})
        by_corner(pro, [c for c in pro.columns if c.startswith("nu_")])

    if variant in ("impute", "both"):
        roll = [c for c in slim.columns if c.startswith(ROLLING) and f"diff_{c}" in matchup.columns]
        prior = division_prior(slim, roll)
        filled = slim[roll].fillna(prior)
        f = pd.concat([slim[["fight_id", "corner"]], filled], axis=1)
        red = f[f.corner == "red"].drop_duplicates("fight_id").set_index("fight_id")
        blue = f[f.corner == "blue"].drop_duplicates("fight_id").set_index("fight_id")
        for c in roll:
            matchup[f"diff_{c}"] = matchup.fight_id.map(red[c]) - matchup.fight_id.map(blue[c])
            replaced.append(f"diff_{c}")
    return new, replaced
