"""Bovada parsing on a saved coupon response (UFC 332, captured 2026-10-03)."""
import json
from pathlib import Path

from app.services.ufc.bovada import canonical_key, parse_event, ufc_bouts

LISTING = json.loads((Path(__file__).parent / "fixtures" / "bovada" / "ufc_listing.json").read_text())


def _mcgee():
    return next(b for b in ufc_bouts(LISTING) if "McGee" in b["fighter_a"] + b["fighter_b"])


def test_listing_skips_potential_fights():
    bouts = ufc_bouts(LISTING)
    assert len(bouts) == 2 and all("potential" not in b["event"]["link"] for b in bouts)
    assert _mcgee()["date"].isoformat() == "2026-10-03"


def test_main_markets_map_onto_our_keys():
    b = _mcgee()
    # DB has Nolan in red: Bovada's first-listed McGee is our blue corner
    rows = parse_event(b["event"], b["fighter_a"], b["fighter_b"], swapped=b["fighter_a"] == "Court McGee")
    k = {r["market_key"]: r for r in rows if r["market_key"]}
    assert k["moneyline_blue"]["american"] == 185 and k["moneyline_red"]["american"] == -225
    assert k["blue_ko"]["american"] == 1600 and k["red_ko"]["american"] == 135
    assert {"dec_yes", "dec_no", "ou_1.5_over", "ou_2.5_under", "sr_2", "sr_2_no", "er_1", "er_3",
            "itd_red_yes", "itd_blue_yes"} <= set(k)


def test_takedown_and_strike_props_keep_corner_and_line():
    b = _mcgee()
    rows = parse_event(b["event"], b["fighter_a"], b["fighter_b"], swapped=True)
    td = [r for r in rows if r["market_name"] == "Total Takedowns Landed O/U - Court McGee"]
    assert {r["selection"] for r in td} == {"Over 1.5", "Under 1.5"}
    assert all(r["side"] == "blue" and r["line"] == 1.5 and r["market_key"] is None for r in td)
    alt = [r for r in rows if r["market_name"] == "Alternate Total Significant Strikes - Eric Nolan"]
    assert alt and all(r["side"] == "red" for r in alt) and 54.5 in {r["line"] for r in alt}


def test_canonical_key_round_completion_is_next_round_start():
    assert canonical_key("Fight To Complete 2 Full Rounds", "Yes (Fight Completes 2 Full Rounds)", None, None) == "sr_3"
    assert canonical_key("Fight To Complete 1 Full Round", "No (Fight Does Not Complete 1 Full Round)", None, None) == "sr_2_no"
    assert canonical_key("Method of Victory", "Draw", None, None) is None


def test_method_ignores_ko_inside_a_fighter_name():
    assert canonical_key("Method of Victory", "Roman Kopylov Wins by Decision or Technical Decision", None, "red") == "red_dec"
    assert canonical_key("Method of Victory", "Roman Kopylov Wins by KO, TKO or DQ", None, "red") == "red_ko"
