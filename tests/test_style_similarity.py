"""Guards for fighter style similarity.

Two jobs:

1. Keep the similarity features OUT of the winner model. Every input to the style vector
   is career-to-date, so a fighter's embedding reflects fights that had not happened at
   the time of any historical bout. Feeding it to `build_features` would be the same
   class of leak as feeding it WHR, and it would additionally invalidate the frozen
   picks experiment (PREREGISTRATION.md).

2. Pin the pure transforms — shrinkage, percentile normalisation, mean-centring, and the
   driver explanation — that decide whether the space measures style or something else.
   No DB, no artifacts, so these run in milliseconds.

Run:  ./venv/bin/python -m pytest tests/test_style_similarity.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.ufc.style_service import (
    ALL_FEATURES,
    BLOCK_A_FEATURES,
    BLOCK_B_FEATURES,
    SHRINK_K,
    STYLE_CRIT,
    _centre_block_b,
    _drivers,
    _percentile_block_a,
    _shrink_block_a,
)


class TestStyleStaysOutOfTheModel:
    """The similarity table is display-only. This is the guard that keeps it that way."""

    def test_no_style_column_reaches_winner_features(self):
        from app.services.ufc.model import winner_feature_columns

        # A frame that looks like a real matchup frame but also carries style columns,
        # as it would if someone joined the similarity table into build_features.
        frame = pd.DataFrame({
            "avg_x": [1.0], "elo": [1500.0], "age": [30.0], "days_since_last": [100.0],
            "glicko_pts": [10.0], "div_lightweight": [1.0], "is_title_fight": [0.0],
            "style_similarity": [0.8],
            "style_embedding_0": [0.1],
            "similar_fighter_rank": [3.0],
            "style_cluster_distance": [0.4],
        })
        cols = winner_feature_columns(frame)
        for banned in ("style_similarity", "style_embedding_0",
                       "similar_fighter_rank", "style_cluster_distance"):
            assert banned not in cols, (
                f"{banned} reached the winner model. The similarity space is built from "
                "career-to-date stats and is retrodictive; it must stay display-only."
            )

    def test_style_service_does_not_import_the_model(self):
        """A one-way dependency is what keeps the leak structurally impossible."""
        import inspect

        from app.services.ufc import style_service

        src = inspect.getsource(style_service)
        assert "from app.services.ufc.model import" not in src
        assert "import app.services.ufc.model" not in src


class TestEligibility:
    def test_retired_fighters_are_kept(self):
        """A ranking excludes the inactive; a style comparison must not, or the feature
        cannot answer "who fights like Khabib" — its flagship question."""
        from datetime import date, timedelta

        from app.services.ufc.fighter_registry import FighterState, is_rankable

        retired = FighterState(
            division="lightweight",
            last_activity=date(2020, 10, 24),
            decided_fights=13,
            rounds=40,
        )
        today = date(2020, 10, 24) + timedelta(days=2000)
        assert is_rankable(retired, today, STYLE_CRIT)

    def test_sample_size_floors_are_still_enforced(self):
        """Dropping the recency gate must not drop the sample-size gate: a two-fight
        fighter has no stable style and should not be anyone's comparable."""
        from datetime import date

        from app.services.ufc.fighter_registry import FighterState, is_rankable

        thin = FighterState(
            division="lightweight",
            last_activity=date(2026, 1, 1),
            decided_fights=2,
            rounds=4,
        )
        assert not is_rankable(thin, date(2026, 2, 1), STYLE_CRIT)


def _raw(n_by_fighter, value_by_fighter, col="slpm"):
    return {f: {col: value_by_fighter[f]} for f in n_by_fighter}


class TestShrinkage:
    def test_low_sample_fighter_is_pulled_toward_the_division_mean(self):
        divisions = {1: "lw", 2: "lw", 3: "lw"}
        counts = {1: 3, 2: 3, 3: 30}          # 1 and 2 thin, 3 established
        raw = {
            1: {c: 0.0 for c in ALL_FEATURES},
            2: {c: 0.0 for c in ALL_FEATURES},
            3: {c: 0.0 for c in ALL_FEATURES},
        }
        raw[1]["slpm"] = 10.0                  # extreme, on 3 fights
        raw[2]["slpm"] = 1.0
        raw[3]["slpm"] = 1.0
        mean = (10.0 + 1.0 + 1.0) / 3

        _shrink_block_a(raw, divisions, counts)

        w_thin = 3 / (3 + SHRINK_K)
        assert raw[1]["slpm"] == pytest.approx(w_thin * 10.0 + (1 - w_thin) * mean)
        # The thin fighter moved much further toward the mean than the veteran did.
        assert abs(raw[1]["slpm"] - 10.0) > abs(raw[3]["slpm"] - 1.0)

    def test_shrinkage_uses_the_fighters_own_division(self):
        """Heavyweight rates must not be averaged into a flyweight's prior."""
        divisions = {1: "hw", 2: "hw", 3: "fw", 4: "fw"}
        counts = {i: 4 for i in divisions}
        raw = {i: {c: 0.0 for c in ALL_FEATURES} for i in divisions}
        raw[1]["slpm"] = raw[2]["slpm"] = 2.0     # heavyweights: low volume
        raw[3]["slpm"] = raw[4]["slpm"] = 8.0     # flyweights: high volume

        _shrink_block_a(raw, divisions, counts)

        # Each fighter equals their own division mean, so nothing crossed over.
        assert raw[1]["slpm"] == pytest.approx(2.0)
        assert raw[3]["slpm"] == pytest.approx(8.0)


class TestPercentileNormalisation:
    def test_divisions_are_normalised_independently(self):
        """Without this, "similar style" degenerates into "similar weight class":
        heavyweights strike less per minute than flyweights, so raw rates cluster by
        division and the cross-division comparison becomes impossible."""
        divisions = {1: "hw", 2: "hw", 3: "fw", 4: "fw"}
        raw = {i: {c: 0.0 for c in ALL_FEATURES} for i in divisions}
        raw[1]["slpm"], raw[2]["slpm"] = 1.0, 3.0     # low-volume division
        raw[3]["slpm"], raw[4]["slpm"] = 6.0, 9.0     # high-volume division

        _percentile_block_a(raw, divisions)

        # The top of each division lands in the same place despite a 3x raw gap.
        assert raw[2]["slpm"] == raw[4]["slpm"]
        assert raw[1]["slpm"] == raw[3]["slpm"]
        assert raw[2]["slpm"] > raw[1]["slpm"]

    def test_block_b_is_untouched(self):
        """Block B arrives already percentile-valued from compute_dimension_profiles;
        percentiling it twice would flatten it."""
        divisions = {1: "lw", 2: "lw"}
        raw = {i: {c: 0.0 for c in ALL_FEATURES} for i in divisions}
        raw[1][BLOCK_B_FEATURES[0]] = 42.0

        _percentile_block_a(raw, divisions)

        assert raw[1][BLOCK_B_FEATURES[0]] == 42.0


class TestBlockBCentring:
    def test_uniform_quality_is_removed(self):
        """This is the step that turns a skill rating into a style descriptor. An elite
        and a journeyman with the same PROFILE SHAPE must land on the same vector."""
        elite = {c: 0.0 for c in ALL_FEATURES}
        journeyman = {c: 0.0 for c in ALL_FEATURES}
        for i, c in enumerate(BLOCK_B_FEATURES):
            elite[c] = 50.0 + i          # good at everything, better at later dims
            journeyman[c] = 10.0 + i     # bad at everything, same shape

        raw = {1: elite, 2: journeyman}
        _centre_block_b(raw)

        for c in BLOCK_B_FEATURES:
            assert raw[1][c] == pytest.approx(raw[2][c]), (
                "Mean-centring must remove the overall-quality axis, otherwise "
                "nearest-neighbours returns 'who is as good as X', not 'who fights like X'."
            )

    def test_shape_differences_survive(self):
        grappler = {c: 0.0 for c in ALL_FEATURES}
        striker = {c: 0.0 for c in ALL_FEATURES}
        for c in BLOCK_B_FEATURES:
            grappler[c] = striker[c] = 50.0
        grappler["glicko_td"] = 90.0
        striker["glicko_str_vol"] = 90.0

        raw = {1: grappler, 2: striker}
        _centre_block_b(raw)

        assert raw[1]["glicko_td"] > raw[2]["glicko_td"]
        assert raw[2]["glicko_str_vol"] > raw[1]["glicko_str_vol"]


class TestDrivers:
    def _z(self, **overrides):
        v = np.zeros(len(ALL_FEATURES))
        for name, val in overrides.items():
            v[ALL_FEATURES.index(name)] = val
        return v

    def test_shared_direction_only(self):
        """Two fighters on OPPOSITE sides of a trait do not share it. Reporting the
        magnitude alone would let the panel justify a match with a trait that in fact
        separates the pair."""
        Z = np.vstack([
            self._z(leg_pct=2.0, td15s=1.5),
            self._z(leg_pct=-2.0, td15s=1.5),
        ])
        names = [d["feature"] for d in _drivers(Z, 0, 1)]
        assert "leg_pct" not in names
        assert "td15s" in names

    def test_shared_low_is_a_real_trait(self):
        """Both far BELOW average is as much a shared style as both above."""
        Z = np.vstack([self._z(ground_pct=-2.0), self._z(ground_pct=-1.8)])
        drivers = _drivers(Z, 0, 1)
        assert drivers[0]["feature"] == "ground_pct"
        assert drivers[0]["z"] < 0

    def test_strength_is_the_weaker_of_the_two(self):
        """A trait is only as shared as the fighter who exhibits it less."""
        Z = np.vstack([self._z(leg_pct=3.0), self._z(leg_pct=1.0)])
        assert _drivers(Z, 0, 1)[0]["z"] == pytest.approx(1.0)

    def test_features_are_pre_pca_and_nameable(self):
        """Drivers must be columns a reader can check on the stat table; a principal
        component is not."""
        Z = np.vstack([self._z(sig_acc=2.0), self._z(sig_acc=2.0)])
        for d in _drivers(Z, 0, 1):
            assert d["feature"] in ALL_FEATURES


class TestFeatureBlocks:
    def test_blocks_partition_the_feature_list(self):
        assert ALL_FEATURES == BLOCK_A_FEATURES + BLOCK_B_FEATURES
        assert not set(BLOCK_A_FEATURES) & set(BLOCK_B_FEATURES)

    def test_no_raw_volume_totals_in_block_a(self):
        """Career totals measure how long a career has been, not how it is fought, and
        would put veterans next to veterans."""
        for banned in ("fight_count", "total_fight_min", "sig_str_landed",
                       "est_ground_min", "est_standing_min", "wins", "losses"):
            assert banned not in BLOCK_A_FEATURES

    def test_block_b_covers_every_glicko_dimension(self):
        from app.services.ufc.glicko_service import DIMENSIONS

        assert BLOCK_B_FEATURES == [f"glicko_{d}" for d in DIMENSIONS]


class TestPostEventChain:
    """The chain exists because two of its steps were unreachable from the running app:
    compute_all_career_stats had no call site at all, and compute_all_derived_stats ran
    only from the full scrape, never from the post-event update."""

    def test_chain_is_in_dependency_order(self):
        from app.main import POST_EVENT_CHAIN

        labels = [label for label, _ in POST_EVENT_CHAIN]
        assert labels.index("Derived Fight Stats") < labels.index("Career Stats"), \
            "career aggregates read the derived per-fight columns"
        assert labels.index("Career Stats") < labels.index("Fighter Similarity"), \
            "Block A of the style vector reads career stats"
        assert labels.index("Generate Rankings") < labels.index("Fighter Similarity"), \
            "similarity runs after rankings so both read the same Glicko state"

    def test_every_chain_target_resolves(self):
        import importlib

        from app.main import POST_EVENT_CHAIN

        for label, target in POST_EVENT_CHAIN:
            module_name, func_name = target.split(":")
            mod = importlib.import_module(module_name)
            assert callable(getattr(mod, func_name)), f"{label} -> {target} is not callable"

    def test_recent_update_reports_new_events(self):
        """scheduled_scrape gates the expensive chain on this return value; if it goes
        back to returning None the chain silently stops running after every event."""
        import inspect

        from app.services.ufc.scraper import run_recent_update

        sig = inspect.signature(run_recent_update)
        assert sig.return_annotation is not inspect.Signature.empty
        src = inspect.getsource(run_recent_update)
        assert "return processed_events" in src
