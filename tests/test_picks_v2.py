"""Picks v2 pure logic: EV, pick side, no-pick rule, badges."""
from app.services.ufc.picks_v2 import american_from_decimal, choose_pick, evaluate_side, graded

TABLE = {"families": {"fam": {"buckets": [
    {"lo": 0.0, "hi": 0.03, "n": 100, "grade": "C", "roi": -0.01, "expected_roi": -0.01, "small_sample": False},
    {"lo": 0.03, "hi": 0.08, "n": 100, "grade": "B", "roi": 0.02, "expected_roi": 0.02, "small_sample": False},
    {"lo": 0.08, "hi": 0.15, "n": 100, "grade": "A-", "roi": 0.04, "expected_roi": 0.04, "small_sample": False},
    {"lo": 0.15, "hi": None, "n": 100, "grade": "A+", "roi": 0.09, "expected_roi": 0.09, "small_sample": False}]}}}


def test_ev_and_conversions():
    e = evaluate_side(0.5, 120, 110)
    assert abs(e["ev_best"] - 0.10) < 1e-9 and abs(e["ev_median"] - 0.05) < 1e-9
    assert american_from_decimal(2.5) == 150 and american_from_decimal(1.5) == -200


def test_pick_is_max_positive_ev_or_none():
    sides = [{"side": "over", "p": 0.58, "best": -110, "med": -115},
             {"side": "under", "p": 0.42, "best": 105, "med": 100}]
    assert choose_pick(sides)["side"] == "over"
    assert choose_pick([{"side": "x", "p": 0.4, "best": 120, "med": 110}]) is None
    # a positive but tiny edge (< 3% at the typical price) is not a pick
    assert choose_pick([{"side": "x", "p": 0.48, "best": 115, "med": 112}]) is None


def test_badges_never_change_the_grade_and_typical_price_drives_it():
    # best +200 but typical +110: graded on the typical price (EV +5%: the 3-8% band)
    g, _, badges = graded("fam", {"p": 0.5, "best": 200, "med": 110}, None, None, TABLE)
    assert "outlier price" in badges and g == "B"
    # market moved away: badge only, grade untouched (typical EV +9%: the 8-15% band)
    g2, _, badges2 = graded("fam", {"p": 0.5, "best": 120, "med": 118}, 0.45, 0.48, TABLE)
    assert "line moved away" in badges2 and g2 == "A-"
    _, _, b3 = graded("fam", {"p": 0.2, "best": 600, "med": 600}, None, None, TABLE)
    assert "high variance" in b3


def test_pick_uses_typical_price():
    # best price flatters "over", typical price favours "under"
    sides = [{"side": "over", "p": 0.48, "best": 150, "med": -105},
             {"side": "under", "p": 0.52, "best": 100, "med": 105}]
    assert choose_pick(sides)["side"] == "under"


def test_blend_prob_symmetric():
    from app.services.ufc.grading import blend_prob
    y, n = blend_prob(0.6, 0.4), blend_prob(0.4, 0.6)
    assert abs(y - 0.5) < 1e-9 and abs(y + n - 1) < 1e-9
    assert blend_prob(0.7, None) == 0.7
    assert 0.30 < blend_prob(0.44, 0.23) < 0.44            # pulled toward the market


def test_top_drivers_drop_odds_and_rank_by_size():
    from types import SimpleNamespace as R

    from app.services.ufc.picks_v2 import top_drivers
    rows = [R(feature_name="diff_age", shap_value=-0.2, feature_value=3.0),
            R(feature_name="diff_odds_implied", shap_value=0.9, feature_value=0.1),
            R(feature_name="diff_pro_elo", shap_value=0.3, feature_value=None),
            R(feature_name="diff_reach", shap_value=0.05, feature_value=2.0)]
    out = top_drivers(rows, n=2)
    assert [d["feature_name"] for d in out] == ["diff_pro_elo", "diff_age"]
    assert out[1]["shap_value"] == -0.2 and out[0]["feature_value"] is None


def test_exchange_prices_skip_untraded_and_include_fees():
    from app.services.ufc.picks_v2 import exchange_american, exchange_buy_price, merge_prices
    # seeded Polymarket prop: no book, no volume -> not a price
    assert exchange_buy_price(None, None, 0.265, None) is None
    # quoted book: buy at the ask; absurdly wide book -> skipped
    assert exchange_buy_price(0.34, 0.35, 0.35, 100) == 0.35
    assert exchange_buy_price(0.01, 0.52, 0.265, 0) is None
    # traded but no live book: not a price (a finished fight's last trade sat at 0.0005)
    assert exchange_buy_price(None, None, 0.655, 22191) is None
    assert exchange_buy_price(None, None, 0.0005, 2881) is None
    # settled / settling extremes are not bets
    assert exchange_buy_price(0.001, 0.005, 0.005, 100) is None
    # Kalshi fee makes the same price pay less than Polymarket's
    assert exchange_american("kalshi", 0.5) < exchange_american("polymarket", 0.5) == 100
    side = merge_prices({"best": -150, "book": "FanDuel", "med": -160}, {"Kalshi": -120})
    assert side["book"] == "Kalshi" and side["med"] == -160 and side["books"] == {"FanDuel": -150, "Kalshi": -120}
    assert merge_prices({"best": None, "book": None, "med": None}, {"Polymarket": 300})["book"] == "Polymarket"


def test_find_arb_and_stakes():
    from app.services.ufc.picks_v2 import bettable_best, find_arb
    # +110 at one book and +105 at another on a two-way market: 1/2.10 + 1/2.05 < 1
    arb = find_arb([("Over", "FanDuel", 110), ("Under", "BetMGM", 105)])
    assert arb and 0.03 < arb["margin"] < 0.04
    assert abs(sum(l["stake"] for l in arb["legs"]) - 100) < 0.02
    # every outcome pays the same
    pays = [l["stake"] * (1 + l["american"] / 100) for l in arb["legs"]]
    assert abs(pays[0] - pays[1]) < 0.05 and abs(pays[0] - arb["payout"]) < 0.05
    # a normal market (vig) is not an arb
    assert find_arb([("A", "x", -110), ("B", "y", -110)]) is None
    # Pinnacle is a reference, never a leg
    side = {"books": {"Pinnacle": 150, "FanDuel": 120}, "best": 150, "book": "Pinnacle"}
    assert bettable_best(side) == ("FanDuel", 120)


def test_exchange_contract_price_inverts_fee():
    from app.services.ufc.picks_v2 import exchange_american, exchange_contract_price
    for plat in ("kalshi", "polymarket"):
        for p in (0.12, 0.35, 0.5, 0.81):
            assert abs(exchange_contract_price(plat, exchange_american(plat, p)) - p) < 0.003
