"""Guards the boundary between prediction markets and everything that was here before.

Exchange prices are useful on the fight pages and in the picks view, and actively harmful
anywhere near the winner model or the pre-registered betting rule:

* `model.py` joins `UFCFightOdds` with no bookmaker filter (twice) to build the `odds_*` features.
  Exchange rows landing in that table would silently become model inputs, on ~1 year of coverage
  against a training set starting in 2020.
* `scripts/generate_picks.py` averages every `UFCFightOdds` row for a fight to get the market
  price the rule is evaluated against. PREREGISTRATION.md registers that book set as
  FanDuel/DraftKings/BetMGM/Bovada and lists changing it as a rule change -- so a new row there
  is not a data update, it invalidates the pre-registration.

The separation is structural: exchanges live in `ufc_prediction_market_*` and nothing writes them
to `ufc_fight_odds`. These tests assert that structure holds, because the failure is silent --
nothing errors, the numbers just quietly stop meaning what the docs say they mean.

Run:  ./venv/bin/python -m pytest tests/test_prediction_market_isolation.py -v
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from app.models.ufc import (
    UFCFightOdds, UFCMethodOdds,
    UFCPredictionMarket, UFCPredictionMarketHistory, UFCPredictionMarketQuote,
)

BACKEND = Path(__file__).resolve().parent.parent

#: Every module allowed to write to the sportsbook odds tables. Adding to this list means you are
#: changing what the model trains on and what the pre-registered rule is priced against.
SPORTSBOOK_WRITERS = {
    "app/services/ufc/odds_scraper.py",      # The Odds API: DraftKings, FanDuel, BetMGM
    "app/services/ufc/bovada_scraper.py",    # Bovada method markets
}


def _ingestion_sources() -> list[Path]:
    return sorted((BACKEND / "app/services/ufc/prediction_markets").glob("*.py"))


def test_ingestion_never_references_sportsbook_tables():
    """No module under prediction_markets/ may even name the sportsbook odds models.

    A name check rather than a write check on purpose: if the ingestion package cannot refer to
    `UFCFightOdds` at all, it cannot write to it by any route, including one added later.
    """
    forbidden = {"UFCFightOdds", "UFCFightOddsHistory", "UFCMethodOdds"}
    offenders = []
    for path in _ingestion_sources():
        names = {n.id for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Attribute)}
        hit = names & forbidden
        if hit:
            offenders.append(f"{path.relative_to(BACKEND)}: {sorted(hit)}")
    assert not offenders, (
        "Prediction-market ingestion must not touch the sportsbook odds tables — these feed the "
        "winner model and the pre-registered picks rule:\n  " + "\n  ".join(offenders)
    )


def test_only_known_scrapers_write_sportsbook_odds():
    """Nothing outside the two sportsbook scrapers may construct a sportsbook odds row."""
    offenders = []
    for path in (BACKEND / "app").rglob("*.py"):
        rel = str(path.relative_to(BACKEND))
        if rel in SPORTSBOOK_WRITERS or "/models/" in rel:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # A direct constructor call, e.g. UFCFightOdds(...), is the write we care about;
            # read-only `db.query(UFCFightOdds)` is fine and happens all over the routers.
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"UFCFightOdds", "UFCFightOddsHistory", "UFCMethodOdds"}:
                    offenders.append(f"{rel}:{node.lineno} constructs {node.func.id}")
    assert not offenders, (
        "Unexpected writer to the sportsbook odds tables:\n  " + "\n  ".join(offenders)
    )


def test_prediction_market_tables_are_distinct_from_odds_tables():
    """The two families must not share a table name on either dialect."""
    exchange = {m.__tablename__ for m in
                (UFCPredictionMarket, UFCPredictionMarketQuote, UFCPredictionMarketHistory)}
    sportsbook = {UFCFightOdds.__tablename__, UFCMethodOdds.__tablename__}
    assert not (exchange & sportsbook)


def test_generate_picks_reads_only_sportsbook_odds():
    """The pre-registered picks generator must not learn about exchanges.

    `scripts/generate_picks.py` computes the registered market price. If it ever grows a reference
    to the prediction-market tables, the price it produces is no longer the one the rule was
    registered against, and the audit log in picks_log.jsonl silently changes meaning mid-stream.
    """
    source = (BACKEND / "scripts/generate_picks.py").read_text()
    for name in ("UFCPredictionMarket", "prediction_markets", "kalshi", "polymarket"):
        assert name not in source, (
            f"scripts/generate_picks.py references {name!r}. That changes the registered market "
            "price — see PREREGISTRATION.md, which lists the book set as part of the rule."
        )


def test_model_feature_join_is_unfiltered_and_sportsbook_only():
    """Document-and-enforce: the model's odds join sees sportsbooks only.

    `build_matchup_df` deliberately queries every `UFCFightOdds` row with no bookmaker filter.
    That is only safe while exchanges are absent from the table, so this pins the invariant the
    absence relies on rather than the filter it does not have.
    """
    from app.services.ufc import model

    source = inspect.getsource(model)
    assert "UFCPredictionMarket" not in source, (
        "model.py references the prediction-market tables. Exchange coverage starts in 2025 "
        "against a training set starting in 2020, and would collide with the existing odds_* "
        "features — see the plan's 'what not to do' section."
    )


@pytest.mark.parametrize("price_pair,expected", [
    ((0.54, 0.47), (0.5347, 0.4653)),   # a normal exchange pair: sums slightly over 1
    ((0.5, 0.5), (0.5, 0.5)),
    ((0.54, None), (None, None)),        # one-sided: must not invent the other side
    ((None, None), (None, None)),
    ((0.0, 0.0), (None, None)),          # untraded: no probability to report
])
def test_normalise_pair(price_pair, expected):
    """The pair-to-probability rule, including the cases that must refuse to answer."""
    from app.services.ufc.prediction_markets.serving import normalise_pair

    got = normalise_pair(*price_pair)
    if expected[0] is None:
        assert got == (None, None)
    else:
        assert got[0] == pytest.approx(expected[0], abs=1e-3)
        assert got[1] == pytest.approx(expected[1], abs=1e-3)
        assert got[0] + got[1] == pytest.approx(1.0)


class _FakeMarket:
    def __init__(self, status="open", outcome_label=None):
        self.status = status
        self.outcome_label = outcome_label


class _FakeQuote:
    def __init__(self, price, volume=None, bid=None, ask=None):
        self.price, self.volume, self.bid, self.ask = price, volume, bid, ask
        self.open_interest = self.liquidity = None
        self.captured_at = None


@pytest.mark.parametrize("price,volume,expected", [
    # Polymarket seeds an untouched prop at 0.50 and reports no volume. Showing that as a market
    # probability invents a consensus that does not exist — every prop on an upcoming title fight
    # read 0.49–0.50 while the moneyline had $8.7k behind it.
    (0.50, None, False),
    (0.49, None, False),
    (0.495, None, False),
    # Real volume settles it regardless of price, including a genuine coin flip.
    (0.50, 8717.67, True),
    # No volume reported, but the price has clearly left the seed — this is the case that a
    # volume-only rule would wrongly hide (an observed settled prop at 0.37 reports no volume).
    (0.37, None, True),
    (0.995, None, True),
])
def test_traded_flag(price, volume, expected):
    from app.services.ufc.prediction_markets.serving import _quote_dict

    got = _quote_dict(_FakeMarket(), _FakeQuote(price, volume))
    assert got["traded"] is expected


def test_settled_market_serves_closing_line_not_settlement():
    """A settled quote is the result, not a price — the curve holds the closing line.

    Kalshi keeps trading after the outcome is known (an observed settled pair reads 0.99 / 0.42,
    summing to 1.41); Polymarket reports settled outcomes as exactly 0 or 1.
    """
    from app.services.ufc.prediction_markets.serving import _quote_dict

    settled = _quote_dict(_FakeMarket(status="settled"), _FakeQuote(0.0), closing=0.215)
    assert settled["price"] == 0.215
    assert settled["last_price"] == 0.0
    assert settled["is_closing_line"] is True

    # With no curve point before the fight there is nothing better to serve than the quote.
    fallback = _quote_dict(_FakeMarket(status="settled"), _FakeQuote(0.0), closing=None)
    assert fallback["price"] == 0.0
    assert fallback["is_closing_line"] is False

    # An open market is never rewritten.
    live = _quote_dict(_FakeMarket(status="open"), _FakeQuote(0.54), closing=0.9)
    assert live["price"] == 0.54
    assert live["is_closing_line"] is False
