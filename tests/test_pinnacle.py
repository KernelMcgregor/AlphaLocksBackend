"""Pinnacle parsing on saved guest-API responses (UFC 332, captured 2026-10-03)."""
import json
from pathlib import Path

from app.services.ufc.pinnacle import bouts_from, canonical_key, parse_bout

FX = Path(__file__).parent / "fixtures" / "pinnacle"
load = lambda n: json.loads((FX / n).read_text())


def _bout():
    (b,) = bouts_from(load("matchups.json"), load("markets.json"))
    return b


def test_bout_and_specials_are_grouped():
    b = _bout()
    assert {b["fighter_a"], b["fighter_b"]} == {"Court McGee", "Eric Nolan"}
    assert len(b["specials"]) == 11


def test_prices_map_onto_our_keys_with_limits():
    b = _bout()
    # put Nolan in our red corner whichever way Pinnacle lists the bout
    rows = parse_bout(b, swapped=b["fighter_a"] == "Court McGee")
    k = {r["market_key"]: r for r in rows if r["market_key"]}
    assert {"moneyline_red", "moneyline_blue", "itd_red_yes", "itd_red_no", "itd_blue_yes",
            "red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec",
            "dec_yes", "dec_no", "sr_2", "sr_2_no", "sr_3", "sr_3_no"} <= set(k)
    assert k["moneyline_red"]["american"] < 0 < k["moneyline_blue"]["american"]   # Nolan favoured
    assert all(r["max_stake"] for r in rows)
    # bout-scoped ids: two fights' moneylines can never collide in change detection
    assert k["moneyline_red"]["external_market_id"].startswith(b["id"] + ":")


def test_canonical_key_no_leg_of_a_method_prop_is_not_a_cell():
    assert canonical_key("Court McGee To Win By TKO/KO", "No", None, "blue") is None
    assert canonical_key("Fight Starts Round 3", "No", None, None) == "sr_3_no"
