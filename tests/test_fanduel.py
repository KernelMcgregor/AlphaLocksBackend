"""FanDuel parsing on saved API responses (UFC 332, captured 2026-10-03)."""
import json
from pathlib import Path

from app.services.ufc.fanduel import canonical_key, parse_markets, tab_slugs, ufc_events

FX = Path(__file__).parent / "fixtures" / "fanduel"
load = lambda n: json.loads((FX / n).read_text())


def test_listing_keeps_ufc_bouts_and_splits_names():
    ev = ufc_events(load("mma_listing.json"))
    assert len(ev) == 3
    mc = next(e for e in ev if e["event_id"] == "36123665")
    assert (mc["fighter_a"], mc["fighter_b"]) == ("Court McGee", "Eric Nolan")
    assert mc["open_date"].date().isoformat() == "2026-10-03"


def test_tabs_skip_popular_and_same_game_parlay():
    assert tab_slugs(load("mcgee_nolan_pages.json")[0]) == ["method-of-victory", "round-props", "time-props"]


def test_bout_markets_map_onto_our_keys_and_corners():
    # DB has Nolan in red, so FanDuel's fighter_a (McGee) is blue: swapped.
    rows = parse_markets(load("mcgee_nolan_pages.json"), "Court McGee", "Eric Nolan", swapped=True)
    by_key = {r["market_key"]: r for r in rows if r["market_key"]}
    assert by_key["moneyline_blue"]["american"] == 198 and by_key["moneyline_red"]["american"] == -240
    assert by_key["red_ko"]["american"] == 135 and by_key["blue_dec"]["american"] == 370
    assert by_key["ou_2.5_over"]["line"] == 2.5
    assert {"dec_yes", "dec_no", "sr_2", "sr_2_no", "sr_3", "er_1", "er_2", "er_3"} <= set(by_key)
    # the moneyline repeats on every tab; it is stored once
    assert sum(r["market_type"] == "MATCH_BETTING" for r in rows) == 2
    # markets nothing prices yet are kept, with their corner where they name one fighter
    combo = [r for r in rows if r["market_type"] == "ROUND_BETTING" and r["selection"].startswith("Court McGee")]
    assert combo and all(r["side"] == "blue" and r["market_key"] is None for r in combo)


def test_strike_ladders_read_as_lines():
    rows = parse_markets([load("silva_wang_strikes.json")], "Natalia Silva", "Wang Cong", swapped=False)
    sig = [r for r in rows if r["market_type"] == "FIGHTER_A_TOTAL_SIGNIFICANT_STRIKES"]
    assert sig and all(r["side"] == "red" for r in sig)
    assert {r["line"] for r in sig} >= {19.5, 29.5, 39.5}
    assert all(r["side"] == "blue" for r in rows if r["market_type"] == "FIGHTER_B_TOTAL_STRIKES")


def test_canonical_key_edge_cases():
    assert canonical_key("METHOD_OF_VICTORY", "Draw", None, None) is None
    assert canonical_key("TOTAL_ROUNDS", "Over", 1.5, None) == "ou_1.5_over"
    assert canonical_key("FIGHT_TO_START_ROUND_4", "No", None, None) == "sr_4_no"
    assert canonical_key("MATCH_BETTING", "Tie", None, None) is None
    assert canonical_key("WHAT_ROUND_WILL_FIGHT_END_(5_ROUNDS)", "Round 4", None, None) == "er_4"
