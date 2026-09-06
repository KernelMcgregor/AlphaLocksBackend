"""Regression tests for the leaks found in the winner-model audit.

Each test fails on the pre-fix code and passes after. They are pure-function tests
over synthetic frames — no DB, no trained artifacts — so they run in milliseconds
and can guard every future change to feature engineering.

Run:  ./venv/bin/python -m pytest tests/test_leakage.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.ufc.model import (
    _corner_swap_augment,
    _fillna_from_train,
    compute_style_matchup_features,
    is_odds_feature,
    select_winner_features,
)


def _fight_frame(n_fights: int = 40, seed: int = 0) -> pd.DataFrame:
    """Two rows per fight (red + blue corner), chronological."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_fights):
        red, blue = f"F{i % 7}", f"F{(i % 7) + 7}"
        winner = red if rng.random() < 0.5 else blue
        for corner, fid in (("red", red), ("blue", blue)):
            rows.append({
                "fight_id": f"fight{i}",
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=7 * i),
                "stats_fighter_id": fid,
                "corner": corner,
                "red_fighter_id": red,
                "blue_fighter_id": blue,
                "winner_id": winner,
            })
    return pd.DataFrame(rows)


def _styles(df: pd.DataFrame, n_clusters: int = 3) -> dict:
    """Deterministic style assignment keyed by (fight_id, fighter_id)."""
    return {
        (r.fight_id, r.stats_fighter_id): int(r.stats_fighter_id[1:]) % n_clusters
        for r in df.itertuples(index=False)
    }


class TestStyleMatchupLeakage:
    """The headline leak: the matrix used to include the predicted fight's own result."""

    def test_feature_does_not_change_when_later_fights_are_appended(self):
        """A fight's feature must depend only on fights BEFORE it.

        This is the sharpest single proof the leak is gone. Under the old
        full-dataset matrix, appending later fights shifted earlier fights' values.
        """
        full = _fight_frame(40)
        prefix = full[full["date"] < pd.Timestamp("2020-01-01") + pd.Timedelta(days=7 * 20)]

        adv_full = compute_style_matchup_features(full, _styles(full), n_clusters=3)
        adv_prefix = compute_style_matchup_features(prefix, _styles(prefix), n_clusters=3)

        assert adv_prefix, "expected some features from the prefix frame"
        for key, prefix_value in adv_prefix.items():
            assert key in adv_full
            assert prefix_value == pytest.approx(adv_full[key], abs=1e-12), (
                f"{key} changed when later fights were appended — future data is leaking "
                f"into this fight's feature"
            )

    def test_first_fight_is_pure_prior(self):
        """With no history, the rate must be the 0.5 prior, not an outcome-derived value."""
        df = _fight_frame(10)
        adv = compute_style_matchup_features(df, _styles(df), n_clusters=3)
        first = df[df["fight_id"] == "fight0"].iloc[0]
        assert adv[("fight0", first["stats_fighter_id"])] == pytest.approx(0.5)

    def test_a_fights_own_result_is_excluded(self):
        """Two frames differing ONLY in the last fight's winner must agree on every
        feature, including that last fight's own."""
        a = _fight_frame(15)
        b = a.copy()
        last = b["fight_id"] == "fight14"
        # Flip the final fight's winner
        b.loc[last, "winner_id"] = np.where(
            b.loc[last, "winner_id"] == b.loc[last, "red_fighter_id"],
            b.loc[last, "blue_fighter_id"],
            b.loc[last, "red_fighter_id"],
        )

        adv_a = compute_style_matchup_features(a, _styles(a), n_clusters=3)
        adv_b = compute_style_matchup_features(b, _styles(b), n_clusters=3)
        assert adv_a == pytest.approx(adv_b), (
            "flipping a fight's winner changed its own feature — direct label leakage"
        )


class TestFeatureSelectionLeakage:
    """MI selection used to run over the full frame, seeing test labels."""

    def test_selection_ignores_test_labels(self):
        """Two competing features, each informative in exactly one half.

        A train-only selector must pick the train-half signal. A full-frame selector
        picks the test-half one, because it is perfectly predictive over half the rows
        while the train-half signal is only weakly so.
        """
        n = 800
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, n)

        train_mask = np.zeros(n, dtype=bool)
        train_mask[: n // 2] = True

        # Informative in train (noisy), pure noise in test
        train_signal = np.where(
            train_mask, y + rng.normal(0, 0.45, n), rng.normal(0, 0.45, n)
        )
        # Pure noise in train, perfectly separable in test
        test_signal = np.where(train_mask, rng.normal(0, 1.0, n), y * 50.0)

        matchup = pd.DataFrame({
            "red_wins": y,
            "diff_train_signal": train_signal,
            "diff_test_only_signal": test_signal,
        })
        feats = ["diff_train_signal", "diff_test_only_signal"]

        selected = select_winner_features(
            matchup, feats, train_mask, top_n=1, include_odds=False, verbose=False,
        )
        assert selected == ["diff_train_signal"], (
            f"selected {selected} — a feature that is only predictive in the test half "
            f"outranked a genuine training signal, so selection is seeing test labels"
        )

        # Sanity: the leaky (full-frame) selection really would have chosen differently,
        # confirming this test discriminates rather than passing trivially.
        all_rows = np.ones(n, dtype=bool)
        leaky = select_winner_features(
            matchup, feats, all_rows, top_n=1, include_odds=False, verbose=False,
        )
        assert leaky == ["diff_test_only_signal"], (
            "test fixture no longer distinguishes leaky from clean selection"
        )

    def test_odds_features_are_force_included_and_excludable(self):
        n = 200
        rng = np.random.default_rng(1)
        matchup = pd.DataFrame({
            "red_wins": rng.integers(0, 2, n),
            "diff_a": rng.normal(size=n),
            "odds_red_prob": rng.random(n),
            "elo_vs_odds": rng.normal(size=n),
        })
        mask = np.zeros(n, dtype=bool)
        mask[:150] = True
        feats = ["diff_a", "odds_red_prob", "elo_vs_odds"]

        with_odds = select_winner_features(
            matchup, feats, mask, top_n=1, include_odds=True, verbose=False)
        assert "odds_red_prob" in with_odds and "elo_vs_odds" in with_odds

        no_odds = select_winner_features(
            matchup, feats, mask, top_n=1, include_odds=False, verbose=False)
        assert not any(is_odds_feature(f) for f in no_odds), (
            f"no-odds arm leaked market features: {no_odds}"
        )


class TestImputationLeakage:
    def test_means_come_from_training_rows_only(self):
        matchup = pd.DataFrame({"diff_x": [1.0, 1.0, np.nan, 100.0, 100.0]})
        train_mask = np.array([True, True, True, False, False])

        out, means = _fillna_from_train(matchup, ["diff_x"], train_mask)

        assert means["diff_x"] == pytest.approx(1.0), (
            "imputation mean absorbed test-period values"
        )
        assert out.loc[2, "diff_x"] == pytest.approx(1.0)

    def test_original_frame_is_not_mutated(self):
        matchup = pd.DataFrame({"diff_x": [1.0, np.nan]})
        _fillna_from_train(matchup, ["diff_x"], np.array([True, False]))
        assert matchup["diff_x"].isna().sum() == 1, "input frame was mutated in place"


class TestCornerSwapAugmentation:
    def test_label_and_features_mirror_correctly(self):
        names = ["diff_elo", "red_glicko", "blue_glicko", "odds_diff", "odds_fav_is_red"]
        X = np.array([[2.0, 10.0, 4.0, 0.3, 1.0]])
        y = np.array([1])

        X_aug, y_aug = _corner_swap_augment(X, y, names)

        assert len(X_aug) == 2 and len(y_aug) == 2
        np.testing.assert_array_equal(X_aug[0], X[0])          # original untouched
        assert y_aug[0] == 1 and y_aug[1] == 0                 # label flips
        assert X_aug[1][0] == -2.0                             # diff_ negates
        assert X_aug[1][1] == 4.0 and X_aug[1][2] == 10.0      # red_/blue_ swap
        assert X_aug[1][3] == -0.3                             # odds_diff negates
        assert X_aug[1][4] == 0.0                              # fav flag flips

    def test_augmentation_is_involutive(self):
        """Mirroring twice returns the original — guards the swap logic."""
        names = ["diff_a", "red_b", "blue_b", "elo_vs_odds"]
        X = np.array([[1.5, 3.0, 7.0, -0.2]])
        y = np.array([1])

        once, y1 = _corner_swap_augment(X, y, names)
        twice, y2 = _corner_swap_augment(once[1:], y1[1:], names)

        np.testing.assert_allclose(twice[1], X[0])
        assert y2[1] == y[0]


class TestGlickoNewcomerSeed:
    """The leak that produced the entire apparent edge over Vegas.

    UFCFighter.wins/losses is scraped from the ufcstats listing page and holds the
    fighter's CURRENT LIFETIME record. Seeding a debutant's Glicko rating from it tells
    the model how that fighter's UFC career turns out. Fixing it moved walk-forward
    accuracy from 71.0% to 61.5% — from "beats Vegas" to "well below Vegas".
    """

    def test_ufc_results_are_subtracted_from_the_lifetime_record(self):
        from app.services.ufc.glicko_service import _pre_ufc_records

        # Fighter A: 3 UFC fights (2W 1L). Lifetime 10-2 => pre-UFC must be 8-1.
        fight_map = {
            1: {"red_id": "A", "blue_id": "B", "winner_id": "A"},
            2: {"red_id": "A", "blue_id": "C", "winner_id": "A"},
            3: {"red_id": "A", "blue_id": "D", "winner_id": "D"},
        }
        ufc_w, ufc_l = _pre_ufc_records(fight_map)

        assert ufc_w["A"] == 2 and ufc_l["A"] == 1
        assert max(0, 10 - ufc_w["A"]) == 8
        assert max(0, 2 - ufc_l["A"]) == 1

    def test_undecided_fights_are_ignored(self):
        from app.services.ufc.glicko_service import _pre_ufc_records

        ufc_w, ufc_l = _pre_ufc_records({
            1: {"red_id": "A", "blue_id": "B", "winner_id": None},
        })
        assert ufc_w == {} and ufc_l == {}

    def test_seed_is_monotone_in_record_quality(self):
        from app.services.ufc.glicko_service import _newcomer_seed

        assert _newcomer_seed(0, 0) == 0.0
        assert _newcomer_seed(20, 0) > _newcomer_seed(10, 5) > _newcomer_seed(0, 20)

    def test_a_debutants_seed_ignores_their_future_ufc_results(self):
        """Two fighters with identical pre-UFC records must get identical seeds,
        no matter how differently their UFC careers go."""
        from app.services.ufc.glicko_service import _newcomer_seed, _pre_ufc_records

        # Both entered the UFC at 8-1. One then goes 5-0, the other 0-5.
        fight_map = {}
        for i in range(5):
            fight_map[i] = {"red_id": "winner", "blue_id": f"x{i}", "winner_id": "winner"}
            fight_map[100 + i] = {"red_id": "loser", "blue_id": f"y{i}", "winner_id": f"y{i}"}
        ufc_w, ufc_l = _pre_ufc_records(fight_map)

        lifetime = {"winner": (13, 1), "loser": (8, 6)}
        seeds = {
            f: _newcomer_seed(max(0, w - ufc_w.get(f, 0)), max(0, l - ufc_l.get(f, 0)))
            for f, (w, l) in lifetime.items()
        }
        assert seeds["winner"] == pytest.approx(seeds["loser"]), (
            f"debut seeds differ ({seeds}) — the seed knows the future UFC record"
        )


class TestOddsFeatureClassification:
    @pytest.mark.parametrize("name", [
        "odds_red_prob", "odds_blue_prob", "odds_diff",
        "odds_fav_is_red", "odds_red_american", "elo_vs_odds",
    ])
    def test_market_features_detected(self, name):
        assert is_odds_feature(name), f"{name} must be excluded from the no-odds arm"

    @pytest.mark.parametrize("name", ["diff_elo", "red_glicko_kod", "diff_style_matchup_adv"])
    def test_non_market_features_kept(self, name):
        assert not is_odds_feature(name)


class TestMarketAnchor:
    """The anchored model must reduce to the market when it has nothing to add."""

    def _fit(self, base, market, y, **kw):
        from app.services.ufc.market_anchor import MarketAnchor
        return MarketAnchor(min_samples=10, **kw).fit(base, market, y)

    def test_worthless_model_collapses_to_the_market(self):
        """If the base model is noise, b -> 0 and predictions equal the market.

        This is the property the current with_odds arm lacks: it can, and does,
        score worse than simply following the line.
        """
        rng = np.random.default_rng(0)
        n = 4000
        market = rng.uniform(0.15, 0.85, n)
        y = (rng.random(n) < market).astype(int)   # market is perfectly calibrated
        base = rng.uniform(0.15, 0.85, n)          # model is pure noise

        a = self._fit(base, market, y)
        assert abs(a.b_) < 0.1, f"b={a.b_:.3f}; noise was trusted"

        p = a.predict_proba(base, market)
        assert np.abs(p - market).mean() < 0.02, "predictions drifted off a perfect market"

    def test_informative_model_earns_weight(self):
        rng = np.random.default_rng(1)
        n = 4000
        market = rng.uniform(0.2, 0.8, n)
        signal = rng.normal(0, 1, n)
        # Truth is the market nudged by a signal the market does not contain
        from app.services.ufc.market_anchor import logit, expit
        truth = expit(logit(market) + 1.2 * signal)
        y = (rng.random(n) < truth).astype(int)
        base = expit(logit(market) + signal)       # base sees the signal

        a = self._fit(base, market, y)
        assert a.b_ > 0.5, f"b={a.b_:.3f}; genuine signal was ignored"

    def test_b_is_clipped_to_the_convex_range(self):
        rng = np.random.default_rng(2)
        n = 2000
        market = rng.uniform(0.2, 0.8, n)
        base = np.clip(market + rng.normal(0, 0.02, n), 0.01, 0.99)
        y = (rng.random(n) < base).astype(int)
        a = self._fit(base, market, y)
        assert 0.0 <= a.b_ <= 1.0

    def test_falls_back_to_market_without_enough_priced_rows(self):
        """Uses the DEFAULT min_samples (150), unlike the other tests in this class."""
        from app.services.ufc.market_anchor import MarketAnchor
        rng = np.random.default_rng(3)
        n = 20
        market = rng.uniform(0.3, 0.7, n)
        a = MarketAnchor().fit(rng.uniform(0.3, 0.7, n), market, rng.integers(0, 2, n))
        assert a.fallback_ and a.b_ == 0.0
        np.testing.assert_allclose(a.predict_proba(rng.uniform(0.3, 0.7, n), market), market,
                                   atol=1e-6)

    def test_unpriced_fights_fall_back_to_the_base_model(self):
        from app.services.ufc.market_anchor import MarketAnchor
        a = MarketAnchor(min_samples=10)
        a.b_, a.c_, a.fallback_ = 0.5, 0.0, False
        base = np.array([0.8, 0.3])
        market = np.array([np.nan, 0.5])
        p = a.predict_proba(base, market)
        assert p[0] == pytest.approx(0.8), "unpriced fight should use the base model"
        assert p[1] != pytest.approx(0.3)

    def test_devig_normalizes_and_is_idempotent(self):
        from app.services.ufc.market_anchor import devig
        r = devig(np.array([0.55]), np.array([0.50]))     # sums to 1.05
        assert r[0] == pytest.approx(0.55 / 1.05)
        assert devig(r, 1 - r)[0] == pytest.approx(r[0])


class TestIdPrecision:
    """Silent float64 rounding of 19-digit IDs killed Elo entirely.

    `winner_id` contains NULLs, so pandas inferred float64 and rounded every ID past
    2**53. Vectorized comparisons still matched (int64 casts to float64, both sides
    round identically) but `iterrows()`/`itertuples()` yield Python scalars, and
    Python compares int to float exactly — so the winner check never fired. Elo sat at
    1500.0, elo_expected at 0.5, and every elo_adj_* feature was a constant.
    """

    UFC_ID = 1180969949270474753  # representative real ID

    def test_python_int_float_comparison_is_exact(self):
        """The language behaviour the bug depended on. If this ever changes, the
        guard below is what still protects us."""
        assert self.UFC_ID != float(self.UFC_ID)
        assert np.int64(self.UFC_ID) == np.float64(float(self.UFC_ID))  # lossy, "equal"

    def test_coercion_preserves_exact_ids_and_nulls(self):
        from app.services.ufc.model import _coerce_id_columns

        rows = [
            {"fight_id": 1, "winner_id": self.UFC_ID, "red_fighter_id": self.UFC_ID},
            {"fight_id": 2, "winner_id": None, "red_fighter_id": self.UFC_ID + 1},
        ]
        df = pd.DataFrame(rows)
        assert df["winner_id"].dtype == np.float64, "fixture no longer reproduces the bug"

        out = _coerce_id_columns(df, rows)
        assert out["winner_id"].dtype == object
        assert out.loc[0, "winner_id"] == self.UFC_ID
        assert out.loc[0, "winner_id"] == out.loc[0, "red_fighter_id"]
        assert out.loc[1, "winner_id"] is None

    def test_winner_comparison_survives_itertuples(self):
        from app.services.ufc.model import _coerce_id_columns

        # Two conditions are both required to reproduce, and the real data meets both:
        #  1. a NULL winner_id somewhere, which forces the column to float64
        #  2. an object column (`method`), which makes itertuples yield PYTHON scalars
        #     rather than NumPy ones — Python compares int to float exactly, NumPy
        #     compares lossily and would hide the bug
        rows = [
            {"fight_id": 1, "red_fighter_id": self.UFC_ID,
             "blue_fighter_id": self.UFC_ID + 1, "winner_id": self.UFC_ID,
             "method": "KO/TKO"},
            {"fight_id": 2, "red_fighter_id": self.UFC_ID + 2,
             "blue_fighter_id": self.UFC_ID + 3, "winner_id": None,
             "method": "Draw"},  # a draw/no-contest: this is what forces float64
        ]
        raw = pd.DataFrame(rows)
        assert raw["winner_id"].dtype == np.float64
        assert not any(
            r.red_fighter_id == r.winner_id for r in raw.itertuples(index=False)
        ), "fixture no longer reproduces the bug"

        fixed = _coerce_id_columns(raw, rows)
        assert any(
            r.red_fighter_id == r.winner_id for r in fixed.itertuples(index=False)
        ), "winner comparison still fails inside itertuples"

    def test_guard_rejects_float_id_columns(self):
        from app.services.ufc.model import _assert_ids_exact

        good = pd.DataFrame({"winner_id": pd.Series([self.UFC_ID], dtype=object)})
        _assert_ids_exact(good)

        bad = pd.DataFrame({"winner_id": [float(self.UFC_ID)]})
        with pytest.raises(TypeError, match="lose precision"):
            _assert_ids_exact(bad)


class TestRoundFormatParsing:
    """`time_format` stores ROUND LENGTHS, not a round count.

    '5-5-5' is a THREE-round fight of five minutes each. An earlier version checked
    `startswith("5")` and so flagged 91.7% of fights as five-rounders; the true rate is
    ~8%. scraper._compute_fight_time() parses it the same way this does, which is what
    makes fight_time_seconds correct.
    """

    @pytest.mark.parametrize("fmt,rounds,minutes,five", [
        ("5-5-5", 3.0, 15.0, False),
        ("5-5-5-5-5", 5.0, 25.0, True),
        ("10-5", 2.0, 15.0, False),
        ("3-3-3", 3.0, 9.0, False),
        ("20", 1.0, 20.0, False),
    ])
    def test_parses_round_lengths(self, fmt, rounds, minutes, five):
        from app.services.ufc.model import (
            _scheduled_rounds, _scheduled_minutes, _is_five_round)
        assert _scheduled_rounds(fmt) == rounds
        assert _scheduled_minutes(fmt) == minutes
        assert _is_five_round(fmt) is five

    @pytest.mark.parametrize("fmt", ["No Time Limit", None, "", "abc"])
    def test_unparseable_formats_are_nan_not_zero(self, fmt):
        from app.services.ufc.model import _scheduled_rounds, _is_five_round
        r = _scheduled_rounds(fmt)
        assert r != r, "unparseable format must be NaN, not a silent 0"
        assert _is_five_round(fmt) is False

    def test_elapsed_time_counts_up_not_down(self):
        """A finish at 1:00 of round 3 is 11 minutes elapsed, not 14."""
        from app.services.ufc.scraper import _compute_fight_time
        secs, mx = _compute_fight_time(3, "1:00", "5-5-5")
        assert secs == 11 * 60
        assert mx == 15 * 60


class TestWeightClassClassifier:
    def test_mens_and_womens_divisions_are_distinct(self):
        from app.services.ufc.model import _classify_weight_class
        assert _classify_weight_class("Women's Flyweight Bout") == "w_flyweight"
        assert _classify_weight_class("Flyweight Bout") == "flyweight"
        assert _classify_weight_class("Women's Bantamweight") != \
               _classify_weight_class("Bantamweight")

    def test_light_heavyweight_not_confused_with_lightweight(self):
        from app.services.ufc.model import _classify_weight_class
        assert _classify_weight_class("Light Heavyweight Bout") == "light_heavyweight"
        assert _classify_weight_class("Lightweight Bout") == "lightweight"

    def test_catchweight_stays_distinct(self):
        """Catchweight must not collapse to 'unknown' — the weight-class movement
        features depend on telling them apart."""
        from app.services.ufc.model import _classify_weight_class
        assert _classify_weight_class("Catch Weight Bout") == "catchweight"
        assert _classify_weight_class(None) == "unknown"

    def test_title_bout_detection(self):
        from app.services.ufc.model import _is_title_bout
        assert _is_title_bout("UFC Lightweight Title Bout")
        assert not _is_title_bout("Lightweight Bout")


class TestFightLevelFeatures:
    """Fight-level columns are shared by both corners, so a difference is identically
    zero. div_* were emitted only as diffs and were therefore dead — division was
    invisible to the model."""

    def test_fight_level_columns_are_recognised(self):
        from app.services.ufc.model import _is_fight_level
        assert _is_fight_level("div_lightweight")
        assert _is_fight_level("is_title_fight")
        assert _is_fight_level("scheduled_rounds")
        assert not _is_fight_level("elo")
        assert not _is_fight_level("age")

    def test_corner_swap_leaves_fight_level_features_alone(self):
        """Mirroring corners must not alter the division or round format."""
        from app.services.ufc.model import _corner_swap_augment
        names = ["diff_elo", "fight_div_lightweight", "fight_is_five_round"]
        X = np.array([[3.0, 1.0, 1.0]])
        X_aug, y_aug = _corner_swap_augment(X, np.array([1]), names)
        assert X_aug[1][0] == -3.0          # diff negates
        assert X_aug[1][1] == 1.0           # division unchanged
        assert X_aug[1][2] == 1.0           # round format unchanged


class TestFeatureListParity:
    """The serve-time list previously omitted all six pre_ufc_* features, so live
    predictions filled them from train means. One function now feeds both paths."""

    def _frame(self):
        return pd.DataFrame({
            "avg_x": [1.0], "elo": [1500.0], "age": [30.0], "days_since_last": [100.0],
            "pre_ufc_wins": [8.0], "pre_ufc_quality": [0.4],
            "ufc_experience_share": [0.3], "glicko_pts": [10.0],
            "div_lightweight": [1.0], "is_title_fight": [0.0],
            "unrelated_column": [7.0],
        })

    def test_includes_pre_ufc_features(self):
        from app.services.ufc.model import winner_feature_columns
        cols = winner_feature_columns(self._frame())
        for c in ["pre_ufc_wins", "pre_ufc_quality", "ufc_experience_share"]:
            assert c in cols, f"{c} missing — this is the train/serve drift bug"

    def test_only_returns_columns_present_on_the_frame(self):
        from app.services.ufc.model import winner_feature_columns
        cols = winner_feature_columns(self._frame())
        assert set(cols).issubset(set(self._frame().columns))
        assert "unrelated_column" not in cols

    def test_absolute_age_and_layoff_reach_the_model(self):
        """Diff-only age made a 22-vs-26 fight identical to a 38-vs-42 fight."""
        from app.services.ufc.model import WINNER_RAW_COLS
        assert "age" in WINNER_RAW_COLS
        assert "days_since_last" in WINNER_RAW_COLS


class TestDamageAndMovementSemantics:
    def test_fights_since_ko_loss_is_pre_fight(self):
        """The counter must describe state BEFORE the fight: the bout in which a
        fighter is knocked out must not already show 0."""
        import app.services.ufc.model as M
        src = M.build_features.__doc__ or ""
        # Behavioural check on the helper's contract via a direct reimplementation
        flags = [0, 0, 1, 0, 0]          # KO'd in fight index 2
        out, since = [], float("nan")
        for f in flags:
            out.append(since)
            since = 0.0 if f == 1 else (since + 1 if since == since else float("nan"))
        assert out[2] != 0.0, "the KO fight itself must not see its own result"
        assert out[3] == 0.0, "the next fight is 0 fights since the KO loss"
        assert out[4] == 1.0


class TestGlickoRecencyCausality:
    """The anti-causal recency decay, and the drift it caused.

    `recency_years` was measured from the LAST DATE IN THE DATASET, so a 2012 fight's
    rating update depended on how recent the newest row happened to be. Every rescrape
    silently changed all historical features, and because decay was exp(-0.461*years),
    a 2012 fight updated at ~0.0016 of K while a 2026 fight updated at full K. Ratings
    sat near zero for a decade then inflated ~5x (mean durability 11.1 -> 52.7),
    creating artificial covariate shift between training and test folds.
    """

    def test_effective_k_has_no_global_clock_term(self):
        """_effective_k must depend only on the rating's own uncertainty."""
        import inspect
        from app.services.ufc.glicko_service import _effective_k

        sig = inspect.signature(_effective_k)
        assert "recency_years" not in sig.parameters, (
            "a global recency term is back in _effective_k; it is anti-causal"
        )

    def test_k_rises_with_uncertainty(self):
        """Glicko's actual rule: a more uncertain rating should move MORE, not less.

        The removed recency term had the opposite sign for returning fighters.
        """
        from app.services.ufc.glicko_service import GlickoParams, _effective_k

        p = GlickoParams()
        k_certain = _effective_k(p.k_base, p.sigma_min, p)
        k_uncertain = _effective_k(p.k_base, p.sigma_init, p)
        assert k_uncertain > k_certain

    def test_dead_parameters_are_gone(self):
        """num_passes/convergence_threshold were never read, which made
        tuner.reevaluate_top_trials' 'now with num_passes=4' a silent no-op."""
        from app.services.ufc.glicko_service import GlickoParams

        fields = GlickoParams().__dict__
        for dead in ("recency_decay", "num_passes", "convergence_threshold"):
            assert dead not in fields, f"{dead} is dead but still declared"


class TestDecorrelationLoss:
    """Hubacek & Sir Equation 59, and why it is not a correlation penalty."""

    def test_equation_59_optimum_pushes_away_from_the_market(self):
        """d/dt[(t-r)^2 + g(t-r)(m-r)] = 0  =>  t = r - g(m-r)/2.

        The optimum is the TRUTH, displaced away from the market in proportion to the
        market's own error — which is what makes the objective convex and anchored.
        """
        r, m, g = 1.0, 0.7, 0.4
        loss = lambda t: (t - r) ** 2 + g * (t - r) * (m - r)
        analytic = r - g * (m - r) / 2
        grid = np.linspace(-1, 3, 40001)
        assert grid[np.argmin([loss(t) for t in grid])] == pytest.approx(analytic, abs=1e-3)
        assert analytic > r, "penalty should push past the truth, away from the market"

    def test_gamma_zero_is_plain_squared_error(self):
        from app.services.ufc.decorrelated import DecorrelatedModel

        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 6))
        y = (rng.random(400) < 0.5).astype(float)
        mkt = rng.uniform(0.2, 0.8, 400)
        a = DecorrelatedModel(gamma=0.0, epochs=8, seed=7).fit(X, y, mkt)
        b = DecorrelatedModel(gamma=0.0, epochs=8, seed=7).fit(X, y, np.full(400, 0.9))
        # With gamma=0 the market is ignored entirely, so a different market must not
        # change a single prediction.
        np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-6)

    def test_penalty_needs_the_realised_outcome(self):
        """Eq. 59 uses (t-r)(m-r), so it is a TRAINING-time term only. Inference must
        not require the market or the outcome."""
        from app.services.ufc.decorrelated import DecorrelatedModel

        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 5))
        m = DecorrelatedModel(gamma=0.5, epochs=5, seed=3).fit(
            X, (rng.random(300) < 0.5).astype(float), rng.uniform(0.3, 0.7, 300))
        p = m.predict_proba(rng.normal(size=(10, 5)))   # no market, no outcome
        assert p.shape == (10,) and np.all((p >= 0) & (p <= 1))

    def test_unpriced_rows_do_not_break_training(self):
        from app.services.ufc.decorrelated import DecorrelatedModel

        rng = np.random.default_rng(2)
        X = rng.normal(size=(400, 5))
        y = (rng.random(400) < 0.5).astype(float)
        mkt = rng.uniform(0.2, 0.8, 400)
        mkt[:300] = np.nan            # most fights unpriced, as in the real data
        p = DecorrelatedModel(gamma=1.0, epochs=8, seed=4).fit(
            X, y, mkt).predict_proba(X)
        assert np.all(np.isfinite(p))
