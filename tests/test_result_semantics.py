"""Regression tests for how draws and no-contests reach ratings and the model.

A NULL winner used to be read as "blue won" in three places: the winner label
(`red_wins`), the streak/win-rate features (`won` was a 0/1 int), and Glicko's
finish-round override, which scored the red corner as losing the final round of every
NC and draw. That is how an eye-poke NC (Aspinall vs Gane) dented a fighter's rating.

Run:  ./venv/bin/python -m pytest tests/test_result_semantics.py -v
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.services.ufc.fighter_registry import is_decided, is_draw
from app.services.ufc.glicko_service import _compute_baselines, _run_glicko
from app.services.ufc.model import _decided_mask, _draw_mask


class TestResultClassification:
    @pytest.mark.parametrize("method,winner", [
        ("Decision - Majority", None),
        ("Decision - Split", None),
        ("Decision - Unanimous", None),
    ])
    def test_draws(self, method, winner):
        assert is_draw(method, winner)
        assert not is_decided(method, winner)

    @pytest.mark.parametrize("method,winner", [
        ("Could Not Continue", None),
        ("Overturned", None),
        ("DQ", 7),
    ])
    def test_voids_are_neither_draws_nor_decided(self, method, winner):
        assert not is_draw(method, winner)
        assert not is_decided(method, winner)

    @pytest.mark.parametrize("method", ["KO/TKO", "Submission", "Decision - Split",
                                        "TKO - Doctor's Stoppage"])
    def test_wins_are_decided(self, method):
        assert is_decided(method, 7)
        assert not is_draw(method, 7)

    def test_masks_handle_nan_winner(self):
        method = pd.Series(["KO/TKO", "Could Not Continue", "Decision - Majority"])
        winner = pd.Series([7.0, float("nan"), float("nan")])
        assert _decided_mask(method, winner).tolist() == [True, False, False]
        assert _draw_mask(method, winner).tolist() == [False, False, True]


def _round(sig: int) -> dict:
    return {"kd": 0, "sig_str_landed": sig, "sig_str_attempted": sig * 2,
            "total_str_landed": sig, "td_landed": 0, "td_attempted": 0, "sub_att": 0,
            "rev": 0, "ctrl_seconds": 0, "head_landed": sig, "body_landed": 0,
            "leg_landed": 0, "distance_landed": sig, "clinch_landed": 0,
            "ground_landed": 0}


def _fight(fid, d, red, blue, winner, method, finish_round, n_rounds=3):
    return {"id": fid, "date": d, "red_id": red, "blue_id": blue, "winner_id": winner,
            "method": method, "weight_class": "heavyweight",
            "finish_round": finish_round, "finish_time_seconds": 300,
            "round_minutes": 5, "max_rounds": 3, "is_title": False, "is_5rd": False}


def _rounds(red, blue, n, red_sig=10, blue_sig=10):
    return {r: {red: _round(red_sig), blue: _round(blue_sig)} for r in range(1, n + 1)}


def _run(fight_map, rounds_by_fight, baselines):
    ratings, *_ = _run_glicko(fight_map, rounds_by_fight, {}, baselines)
    return ratings


class TestGlickoVoidsAndDraws:
    def test_no_contest_does_not_move_ratings(self):
        fights = {
            1: _fight(1, date(2024, 1, 1), 1, 2, 1, "Decision - Unanimous", 3),
            2: _fight(2, date(2024, 6, 1), 1, 3, None, "Could Not Continue", 1),
        }
        rounds = {1: _rounds(1, 2, 3, 15, 10), 2: _rounds(1, 3, 1, 2, 2)}
        baselines = _compute_baselines(fights, rounds)

        with_nc = _run(fights, rounds, baselines)
        without = _run({1: fights[1]}, {1: rounds[1]}, baselines)
        for dim, (mu, sigma) in with_nc[1].items():
            assert mu == pytest.approx(without[1][dim][0]), dim
            assert sigma == pytest.approx(without[1][dim][1]), dim

    def test_draw_is_scored_symmetrically(self):
        """Identical output in a draw must leave both corners with the same pts rating.
        Before the fix red was scored as losing the final round."""
        fights = {1: _fight(1, date(2024, 1, 1), 1, 2, None, "Decision - Majority", 3)}
        rounds = {1: _rounds(1, 2, 3)}
        ratings = _run(fights, rounds, _compute_baselines(fights, rounds))
        assert ratings[1]["pts"][0] == pytest.approx(ratings[2]["pts"][0])


class TestConsensusOdds:
    def test_straddling_even_money_is_not_near_zero(self):
        """The old mean of American odds turned -110/+105 into -2.5."""
        from app.services.ufc.market_anchor import consensus_american
        c = consensus_american([-110, 105])
        assert -110 <= c <= 105 and abs(c) >= 100

    def test_single_book_round_trips(self):
        from app.services.ufc.market_anchor import consensus_american
        from app.services.ufc.market_anchor import american_to_prob
        for o in (-250, -110, 100, 150, 400):  # +100 and -100 are the same price
            assert american_to_prob(consensus_american([o])) == pytest.approx(american_to_prob(o))

    def test_attach_odds_averages_books(self):
        from types import SimpleNamespace
        from app.services.ufc.model import _attach_odds
        m = pd.DataFrame(index=pd.Index([1, 2], name="fight_id"))
        rows = [
            SimpleNamespace(fight_id=1, red_implied_prob=0.6, blue_implied_prob=0.4,
                            red_odds=-150, blue_odds=130),
            SimpleNamespace(fight_id=1, red_implied_prob=0.7, blue_implied_prob=0.3,
                            red_odds=-233, blue_odds=190),
        ]
        assert _attach_odds(m, rows) == 1
        # Raw prices average to red 0.65 / blue 0.37; goto shades the longshot a touch.
        assert 0.63 < m.loc[1, "odds_red_prob"] < 0.66
        assert m.loc[1, "odds_red_prob"] + m.loc[1, "odds_blue_prob"] == pytest.approx(1.0)
        assert pd.isna(m.loc[2, "odds_red_prob"])


class TestUpcomingSnapshots:
    def test_upcoming_bout_snapshot_reflects_latest_fight(self):
        """Serving reads the snapshot for the upcoming fight id. It must equal the rating
        AFTER the fighter's last bout, not before it (the old one-fight staleness)."""
        from app.services.ufc.glicko_service import _run_glicko
        fights = {
            1: _fight(1, date(2024, 1, 1), 1, 2, 1, "KO/TKO", 2),
            2: _fight(2, date(2024, 3, 1), 1, 3, None, "", None),  # not yet fought
        }
        rounds = {1: _rounds(1, 2, 2, 20, 5)}
        baselines = _compute_baselines(fights, rounds)
        ratings, *_, snaps = _run_glicko(fights, rounds, {}, baselines)
        up = snaps[(2, 1)]
        for d, (mu, _) in ratings[1].items():
            assert up[d] == pytest.approx(mu), d
        assert up["_meta_days_since"] == 60
        assert (2, 3) in snaps and snaps[(2, 3)]["_meta_days_since"] == -1


class TestGotoDevig:
    def test_sums_to_one_and_shades_longshot(self):
        from app.services.ufc.market_anchor import american_to_prob, goto_devig
        r, b = american_to_prob(-400), american_to_prob(300)
        p = goto_devig(r, b)
        assert p > r / (r + b)  # favourite gets more than plain normalisation gives

    def test_no_margin_is_identity(self):
        from app.services.ufc.market_anchor import goto_devig
        assert goto_devig(0.6, 0.4) == pytest.approx(0.6)


class TestCancelledBoutsAreRecorded:
    def test_record_before_delete(self, tmp_path, monkeypatch):
        """reconcile keeps a UFCCancelledBout row for every bout it removes."""
        import datetime as dt
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.models.ufc import UFCCancelledBout
        from app.services.ufc import reconcile_service as rs

        from sqlalchemy import BigInteger
        from sqlalchemy.ext.compiler import compiles

        @compiles(BigInteger, "sqlite")
        def _bigint_sqlite(type_, compiler, **kw):  # SQLite only auto-numbers INTEGER PKs
            return "INTEGER"

        eng = create_engine(f"sqlite:///{tmp_path}/t.db").execution_options(
            schema_translate_map={"ufc": None})
        UFCCancelledBout.__table__.create(eng)
        db = sessionmaker(bind=eng)()
        from types import SimpleNamespace
        fight = SimpleNamespace(ufcstats_id="abc123", red_fighter_id=1, blue_fighter_id=2,
                                weight_class="Lightweight Bout", created_at=dt.datetime(2026, 8, 1))
        event = SimpleNamespace(id=9, date=dt.date(2026, 9, 12))
        rs._record_cancellation(db, fight, event)
        rs._record_cancellation(db, fight, event)  # idempotent
        rows = db.query(UFCCancelledBout).all()
        assert len(rows) == 1 and rows[0].red_fighter_id == 1 and rows[0].event_date == dt.date(2026, 9, 12)


class TestShortNoticeFlags:
    def test_a_b_c_replacement(self):
        """A booked vs B, B pulled, A fights C: A opp_changed, C replacement."""
        import datetime as dt
        from app.services.ufc.short_notice import features_from_rows
        A, B, C, D, E = 1, 2, 3, 4, 5
        card = dt.date(2026, 9, 12)
        df = pd.DataFrame({
            "date": [card] * 4,
            "stats_fighter_id": [A, C, D, E],
            "red_fighter_id": [A, A, D, D],
            "blue_fighter_id": [C, C, E, E],
        })
        pulled = [(card - dt.timedelta(days=1), A, B)]  # listed a day off, still counts
        f = features_from_rows(df, pulled)
        assert f["sn_opp_changed"].tolist() == [1.0, 0.0, 0.0, 0.0]
        assert f["sn_replacement"].tolist() == [0.0, 1.0, 0.0, 0.0]

    def test_pulled_bout_months_away_is_ignored(self):
        import datetime as dt
        from app.services.ufc.short_notice import features_from_rows
        df = pd.DataFrame({"date": [dt.date(2026, 9, 12)] * 2, "stats_fighter_id": [1, 3],
                           "red_fighter_id": [1, 1], "blue_fighter_id": [3, 3]})
        f = features_from_rows(df, [(dt.date(2026, 5, 1), 1, 2)])
        assert f.sum().sum() == 0


class TestWithdrawalsAndMatchmaking:
    def test_withdrawer_is_the_fighter_who_did_not_fight(self):
        import datetime as dt
        from app.services.ufc.withdrawals import features_from_rows
        card = dt.date(2025, 3, 1)
        # A was booked vs B; B pulled out; A fought C that night. Later A and B fight others.
        df = pd.DataFrame({
            "date": [card, card, dt.date(2025, 9, 1), dt.date(2025, 9, 1)],
            "stats_fighter_id": [1, 3, 2, 4],
        })
        f = features_from_rows(df, [(card, 1, 2)])
        # B (id 2) withdrew: counted on B's later fight, not on the card itself.
        assert f.loc[2, "wd_withdrawals_3y"] == 1
        assert f.loc[2, "wd_days_since"] == (dt.date(2025, 9, 1) - card).days
        assert f.loc[0, "wd_withdrawals_3y"] == 0  # A fought; not a withdrawal

    def test_scrapped_bout_credits_both_as_scrapped(self):
        import datetime as dt
        from app.services.ufc.withdrawals import features_from_rows
        df = pd.DataFrame({"date": [dt.date(2025, 9, 1)] * 2, "stats_fighter_id": [1, 2]})
        f = features_from_rows(df, [(dt.date(2025, 3, 1), 1, 2)])
        assert f["wd_scrapped_3y"].tolist() == [1.0, 1.0]
        assert f["wd_withdrawals_3y"].tolist() == [0.0, 0.0]

    def test_card_depth_and_trajectory(self):
        import datetime as dt
        from app.services.ufc.matchmaking import compute
        # Fighter 9 was the last bout (prelim) on a 3-fight card, then headlined the next.
        df = pd.DataFrame({
            "fight_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "event_id": [10, 10, 10, 10, 10, 10, 11, 11],
            "card_position": [0, 0, 1, 1, 2, 2, 0, 0],
            "date": [dt.date(2025, 1, 1)] * 6 + [dt.date(2025, 6, 1)] * 2,
            "stats_fighter_id": [5, 6, 7, 8, 9, 10, 9, 11],
        })
        f = compute(df)
        assert f.loc[4, "mm_card_depth"] == 1.0 and f.loc[0, "mm_main_event"] == 1.0
        assert f.loc[6, "mm_prev_depth"] == 1.0 and f.loc[6, "mm_depth_change"] == 1.0


class TestScorecards:
    def test_parse_loser_first(self):
        from app.services.ufc.scorecards import parse, winner_margin
        d = "Mike Bell 28 - 29. Chris Lee 29 - 28. Sal D'amato 28 - 29."
        assert parse(d) == [("Mike Bell", 29, 28), ("Chris Lee", 28, 29), ("Sal D'amato", 29, 28)]
        assert abs(winner_margin(d) - 1 / 3) < 1e-9
        assert winner_margin("Derek Cleary 27 - 30. Sal D'amato 27 - 30. Junichiro Kamijo 27 - 30.") == 3.0
        assert winner_margin(None) is None and winner_margin("Punch to the face at 4:31") is None

    def test_margin_score_monotone(self):
        from app.services.ufc.scorecards import margin_score
        assert margin_score(0, 0.3) == 0.5
        assert margin_score(-1, 0.3) == 0.5
        assert 0.5 < margin_score(1 / 3, 0.3) < margin_score(1, 0.3) < margin_score(3, 0.3) < 1


class TestExpectedTimeFormat:
    def test_main_event_is_five_rounds_card_position_zero_based(self):
        from types import SimpleNamespace
        from app.services.ufc.model import _expected_time_format
        f = lambda wc, pos: _expected_time_format(SimpleNamespace(weight_class=wc, card_position=pos))
        assert f("Lightweight Bout", 0) == "5-5-5-5-5"          # main event
        assert f("Lightweight Bout", 1) == "5-5-5"              # co-main, non-title
        assert f("UFC Women's Flyweight Title Bout", 3) == "5-5-5-5-5"
        assert f("Lightweight Bout", None) == "5-5-5"
        assert f("Road to UFC 4 Flyweight Tournament Title Bout", 3) == "5-5-5"   # tournament "title": 3 rds
        dwcs = SimpleNamespace(weight_class="Middleweight Bout", card_position=0,
                               event=SimpleNamespace(name="DWCS 9.5"))
        assert _expected_time_format(dwcs) == "5-5-5"
        props = SimpleNamespace(weight_class="Lightweight Bout", card_position=4,
                                scheduled_format="5-5-5-5-5")
        assert _expected_time_format(props) == "5-5-5-5-5"                      # props win

    def test_scheduled_format_from_props(self):
        from app.services.ufc.line_watcher import scheduled_format_from_props
        assert scheduled_format_from_props({"ou_4.5_over": {}, "ou_1.5_over": {}}) == "5-5-5-5-5"
        assert scheduled_format_from_props({"sr_5": {}}) == "5-5-5-5-5"
        assert scheduled_format_from_props({"ou_2.5_over": {}, "red_ko": {}}) == "5-5-5"
        assert scheduled_format_from_props({"red_ko": {}, "dec_yes": {}}) is None
