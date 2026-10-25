"""Guards for fighter style similarity.

Two jobs:

1. Keep the similarity features OUT of the winner model. Every input to the style vector
   is career-to-date, so a fighter's embedding reflects fights that had not happened at
   the time of any historical bout. Feeding it to `build_features` would be the same
   class of leak as feeding it WHR, and it would additionally invalidate the frozen
   picks experiment (PREREGISTRATION.md).

2. Pin the pure transforms — the opponent-adjusted habit model, the v5 features (log-ratio
   mixes, floored rare events), division scaling, the era fit, the block weighting and the
   driver explanation — that decide whether the space measures how someone fights or
   something else. No DB, no artifacts.

Run:  ./venv/bin/python -m pytest tests/test_style_similarity.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.ufc import style_habits as sh
from app.services.ufc.style_service import (
    ALL_FEATURES,
    EFFICACY_FEATURES,
    FEATURE_BLOCKS,
    SKILL_FEATURES,
    STYLE_CRIT,
    _drivers,
    _percentile_within_division,
    fit_space,
    transform,
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


def _bout(fighter, opponent, **counts):
    c = {k: 0.0 for k in ("minutes", "standing_min", "ground_min", "str_att", "str_landed", "td_att",
                          "td_landed", "sub_att", "rev", "kd", "ctrl_min", "head_att", "body_att",
                          "leg_att", "dist_att", "clinch_att", "ground_att", "ground_landed",
                          "target_att", "position_att", "nonsig_att", "opp_ctrl_min")}
    c.update({"minutes": 15.0, "standing_min": 15.0, "ground_min": 0.0})
    c.update(counts)
    return sh.Bout(fight_id=hash((fighter, opponent)) % 10**9, fighter=fighter, opponent=opponent,
                   date=None, won=True, method="decision", seconds=900.0, c=c,
                   o={"str_att": 0.0, "str_landed": 0.0, "td_att": 0.0, "td_landed": 0.0})


class TestOpponentAdjustedHabits:
    """A habit is a choice, and choices bend to the opponent: a wrestler shoots less against
    an elite wrestler. The two-way model credits the fighter, not the schedule."""

    def _league(self):
        # Fighters 1 and 2 shoot identically; fighter 1 only ever faced opponent 10, who
        # suppresses takedowns (everyone shoots 1 against him, 4 against opponent 11).
        bouts = {1: [], 2: [], 3: [], 4: []}
        for _ in range(6):
            bouts[1].append(_bout(1, 10, td_att=2.0))
            bouts[2].append(_bout(2, 11, td_att=2.0))
            bouts[3].append(_bout(3, 10, td_att=1.0))
            bouts[3].append(_bout(3, 11, td_att=4.0))
            bouts[4].append(_bout(4, 10, td_att=1.0))
            bouts[4].append(_bout(4, 11, td_att=4.0))
        return bouts

    def test_a_tough_schedule_is_credited_back(self):
        bouts = self._league()
        habits = {"td_s": sh.HABITS["td_s"]}
        model = sh.fit_opponent_model(bouts, habits)
        adj1 = sh.habit_rates(bouts[1], model, habits)["td_s"]
        adj2 = sh.habit_rates(bouts[2], model, habits)["td_s"]
        raw1 = sh.habit_rates(bouts[1], None, habits)["td_s"]
        raw2 = sh.habit_rates(bouts[2], None, habits)["td_s"]
        assert raw1 == pytest.approx(raw2)        # identical raw numbers...
        assert adj1 > adj2                         # ...but 1 did it against the stingier opponent

    def test_the_prior_pulls_short_careers_toward_the_league_rate(self):
        habits = {"td_s": sh.HABITS["td_s"]}
        bouts = {1: [_bout(1, 9, td_att=12.0)], 2: [_bout(2, 9, td_att=1.0) for _ in range(30)]}
        model = sh.fit_opponent_model(bouts, habits)
        one_fight = sh.habit_rates(bouts[1], model, habits)["td_s"]
        raw_one = sh.habit_rates(bouts[1], None, habits)["td_s"]
        league = model.mu["td_s"] * 15
        assert league < one_fight < raw_one

    def test_habits_are_attempts_never_landed(self):
        for name, (count, _exposure, _scale) in sh.HABITS.items():
            assert "landed" not in count, f"{name} counts landed: that is efficacy, not a habit"


class TestPercentileNormalisation:
    def test_divisions_are_normalised_independently(self):
        """Without this, "similar style" degenerates into "similar weight class":
        heavyweights throw less per minute than flyweights."""
        divisions = {1: "hw", 2: "hw", 3: "fw", 4: "fw"}
        raw = {i: {c: 0.0 for c in ALL_FEATURES} for i in divisions}
        raw[1]["str_att_pm"], raw[2]["str_att_pm"] = 1.0, 3.0     # low-volume division
        raw[3]["str_att_pm"], raw[4]["str_att_pm"] = 6.0, 9.0     # high-volume division

        _percentile_within_division(raw, divisions, ALL_FEATURES)

        assert raw[2]["str_att_pm"] == raw[4]["str_att_pm"]
        assert raw[1]["str_att_pm"] == raw[3]["str_att_pm"]
        assert raw[2]["str_att_pm"] > raw[1]["str_att_pm"]


class TestV5Features:
    def _league(self):
        bouts = {}
        for f in range(1, 9):
            bouts[f] = [_bout(f, 100 + k, str_att=60.0 + f, head_att=40.0, body_att=10.0 + f, leg_att=10.0,
                              dist_att=50.0, clinch_att=8.0, ground_att=2.0 + f, td_att=f % 3,
                              sub_att=float(f == 1), ctrl_min=f / 4, opp_ctrl_min=1.0, nonsig_att=10.0)
                        for k in range(f)]          # fighter f has f fights
        return bouts, sh.fit_opponent_model(bouts)

    def test_mixes_are_centred_log_ratios(self):
        bouts, model = self._league()
        feats = sh.style_features(bouts[5], model)
        for block in ("target", "position"):
            assert sum(feats[c] for c in FEATURE_BLOCKS[block]) == pytest.approx(0.0, abs=1e-9)

    def test_display_mixes_sum_to_one(self):
        bouts, model = self._league()
        d = sh.display_rates(bouts[5], model)
        assert d["head_att_share"] + d["body_att_share"] + d["leg_att_share"] == pytest.approx(1.0)
        assert d["dist_att_share"] + d["clinch_att_share"] + d["ground_att_share"] == pytest.approx(1.0)

    def test_never_attempting_is_not_spread_by_minutes(self):
        """With no floor, a fighter who never shoots for a submission is placed by how long
        their career is (log of a shrunk zero). The floor keeps non-attempters together."""
        bouts, model = self._league()
        never = [sh.style_features(bouts[f], model)["sub_att15"] for f in (2, 8)]
        attempts = sh.style_features(bouts[1], model)["sub_att15"]
        assert abs(never[0] - never[1]) < 0.5 * abs(attempts - never[1])

    def test_no_feature_counts_landed(self):
        for name, (count, _exposure, _scale) in sh.HABITS.items():
            assert "landed" not in count


class TestSpace:
    def _fit(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, len(ALL_FEATURES)))
        prime = np.repeat(rng.uniform(2000, 2025, 150), 2)     # each division sees the same eras
        divisions = ["hw" if i % 2 else "fw" for i in range(300)]
        X[:, 0] += (prime - 2012) * 0.2                       # one feature drifts with era
        X[np.array(divisions) == "hw", 0] -= 3.0               # heavyweights throw less
        return X, prime, divisions, fit_space(X, prime, divisions)

    def test_era_trend_is_removed(self):
        X, prime, div, space = self._fit()
        Z = transform(X, prime, div, space)
        assert abs(np.corrcoef(Z[:, 0], prime)[0, 1]) < 0.05

    def test_divisions_are_scaled_independently(self):
        """Without this "similar style" degenerates into "similar weight class"."""
        X, prime, div, space = self._fit()
        Z = transform(X, prime, div, space)
        hw = np.array(div) == "hw"
        assert abs(Z[hw, 0].mean() - Z[~hw, 0].mean()) < 0.1

    def test_each_concept_carries_equal_weight(self):
        """A three-column mix must not outweigh a one-column habit three to one."""
        X, prime, div, space = self._fit()
        Z = transform(X, prime, div, space)
        var = {b: sum(np.var(Z[:, ALL_FEATURES.index(c)]) for c in cols) for b, cols in FEATURE_BLOCKS.items()}
        dof = {b: sh.BLOCK_DOF.get(b, len(cols)) for b, cols in FEATURE_BLOCKS.items()}
        per_dof = [var[b] * dof[b] / len(FEATURE_BLOCKS[b]) for b in FEATURE_BLOCKS]
        assert max(per_dof) == pytest.approx(min(per_dof), rel=0.05)
        assert var["target"] == pytest.approx(1.5, rel=0.05)   # 3 columns / 2 degrees of freedom
        assert var["pace"] == pytest.approx(1.0, rel=0.05)

    def test_transform_is_frozen(self):
        """Transforming new rows must not refit anything: the basis cannot move between events."""
        X, prime, div, space = self._fit()
        before = transform(X[:5], prime[:5], div[:5], space)
        transform(X[::-1] * 2, prime[::-1], div[::-1], space)
        assert np.allclose(before, transform(X[:5], prime[:5], div[:5], space))


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
            self._z(leg_att_share=2.0, td_att15s=1.5),
            self._z(leg_att_share=-2.0, td_att15s=1.5),
        ])
        names = [d["feature"] for d in _drivers(Z, 0, 1)]
        assert "leg_att_share" not in names
        assert "td_att15s" in names

    def test_shared_low_is_a_real_trait(self):
        """Both far BELOW average is as much a shared style as both above."""
        Z = np.vstack([self._z(ground_att_share=-2.0), self._z(ground_att_share=-1.8)])
        drivers = _drivers(Z, 0, 1)
        assert drivers[0]["feature"] == "ground_att_share"
        assert drivers[0]["z"] < 0

    def test_strength_is_the_weaker_of_the_two(self):
        """A trait is only as shared as the fighter who exhibits it less."""
        Z = np.vstack([self._z(leg_att_share=3.0), self._z(leg_att_share=1.0)])
        assert _drivers(Z, 0, 1)[0]["z"] == pytest.approx(1.0)

    def test_features_are_pre_pca_and_nameable(self):
        """Drivers must be columns a reader can check on the stat table; a principal
        component is not."""
        Z = np.vstack([self._z(td_att15s=2.0), self._z(td_att15s=2.0)])
        for d in _drivers(Z, 0, 1):
            assert d["feature"] in ALL_FEATURES


class TestFeatureBlocks:
    def test_blocks_partition_the_feature_list(self):
        assert ALL_FEATURES == [c for cols in FEATURE_BLOCKS.values() for c in cols]
        assert len(set(ALL_FEATURES)) == len(ALL_FEATURES)

    def test_no_raw_volume_totals(self):
        """Career totals measure how long a career has been, not how it is fought."""
        for banned in ("fight_count", "total_fight_min", "sig_str_landed",
                       "est_ground_min", "est_standing_min", "wins", "losses"):
            assert banned not in ALL_FEATURES

    def test_skill_and_efficacy_are_display_only(self):
        """v5 answers "who fights like X": how well it works is shown, not matched on."""
        assert not set(SKILL_FEATURES) & set(ALL_FEATURES)
        assert not set(EFFICACY_FEATURES) & set(ALL_FEATURES)


class TestPostEventChain:
    """The chain exists because two of its steps were unreachable from the running app:
    compute_all_career_stats had no call site at all, and compute_all_derived_stats ran
    only from the full scrape, never from the post-event update."""

    def test_chain_is_in_dependency_order(self):
        from app.main import POST_EVENT_CHAIN

        labels = [label for label, _ in POST_EVENT_CHAIN]
        assert labels.index("Derived Fight Stats") < labels.index("Career Stats"), \
            "career aggregates read the derived per-fight columns"
        assert labels.index("Derived Fight Stats") < labels.index("Fighter Similarity"), \
            "the habit model reads the per-fight stats"
        assert labels.index("Glicko Ratings + Rankings") < labels.index("Fighter Similarity"), \
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


class TestIntentFeatures:
    """Volume is measured as ATTEMPTS. Landed rates fold success into volume, so a fighter
    who shoots ten takedowns and lands two looked like one who shoots three."""

    def test_volume_is_attempts_not_landed(self):
        assert "str_att_pm" in ALL_FEATURES and "td_att15s" in ALL_FEATURES
        assert "slpm" not in ALL_FEATURES and "td15s" not in ALL_FEATURES

    def test_efficacy_is_shown_alongside(self):
        for col in ("sig_acc", "td_acc", "sig_def", "td_def"):
            assert col in EFFICACY_FEATURES


class TestDoesNotRewriteModelSnapshots:
    """The style step runs after publish_rankings, which already wrote
    ufc_glicko_snapshots. Rewriting them here was redundant and briefly emptied the table
    the winner model serves from."""

    def test_style_service_replays_without_persisting(self):
        import inspect
        from app.services.ufc import style_service
        assert "persist_snapshots=False" in inspect.getsource(style_service._load_rows)

    def test_persist_false_skips_the_write(self, monkeypatch):
        from app.services.ufc import glicko_service as g
        saved = []
        monkeypatch.setattr(g, "_load_data", lambda db: ({}, {}, {}, {}))
        monkeypatch.setattr(g, "_compute_baselines", lambda fm, rbf: {})
        monkeypatch.setattr(g, "sherdog_pre_ufc_records", lambda fm: {})
        monkeypatch.setattr(g, "_run_glicko", lambda *a, **k: ({}, {}, {}, {}, {}, [], {(1, 2): {}}))
        monkeypatch.setattr(g, "_save_snapshots", lambda db, snaps: saved.append(snaps))
        g.compute_and_save_snapshots(None, persist=False)
        assert saved == []
        g.compute_and_save_snapshots(None)
        assert len(saved) == 1
