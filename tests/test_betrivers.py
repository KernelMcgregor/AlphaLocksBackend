"""BetRivers (Kambi) parsing on saved responses (UFC 332, captured 2026-10-03)."""
import json
from pathlib import Path

from app.services.ufc.betrivers import parse_offers, ufc_events

FX = Path(__file__).parent / "fixtures" / "betrivers"
load = lambda n: json.loads((FX / n).read_text())


def test_listing_reads_ufc_bouts():
    ev = ufc_events(load("listing.json"))
    mc = next(e for e in ev if e["id"] == "1029162019")
    assert (mc["fighter_a"], mc["fighter_b"]) == ("Court McGee", "Eric Nolan")
    assert mc["date"].isoformat() == "2026-10-03"


def test_offers_map_onto_our_keys():
    offers = load("mcgee_nolan.json")["betOffers"]
    rows = parse_offers(offers, "Court McGee", "Eric Nolan", swapped=True)   # McGee is our blue
    k = {r["market_key"]: r for r in rows if r["market_key"]}
    assert k["moneyline_blue"]["american"] == 188 and k["moneyline_red"]["american"] == -240
    assert k["blue_ko"]["american"] == 1900 and k["blue_dec"]["american"] == 350
    assert k["ou_2.5_over"]["line"] == 2.5 and k["ou_1.5_under"]["american"] == 155
    assert k["itd_blue_yes"]["american"] == 575 and k["itd_blue_no"]["american"] == -1000
    assert {"dec_yes", "dec_no", "sr_2", "sr_2_no", "sr_3", "er_1", "er_2", "er_3"} <= set(k)
    # stored but not priced: winner x rounds keeps its corner
    combo = [r for r in rows if r["market_name"] == "Court McGee to Win & Over 1.5 Rounds"]
    assert combo and all(r["side"] == "blue" and r["market_key"] is None for r in combo)
