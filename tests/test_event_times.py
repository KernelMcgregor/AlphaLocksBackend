"""ufc.com listing parsing and card matching (app/services/ufc/event_times.py). No network,
no database."""
from __future__ import annotations

import datetime as dt

from app.services.ufc.event_times import ListedCard, match_card, parse_listing

LISTING = """
<h3 class="c-card-event--result__headline"><a href="/event/ufc-fight-night-november-21-2026">Prochazka vs Stirling</a></h3>
<div class="c-card-event--result__date tz-change-data" data-locale="en" data-format="D, M j / g:i A T"
 data-main-card="Sat, Nov 21 / 1:00 PM EST" data-main-card-timestamp="1795284000" data-prelims-card="" data-prelims-card-timestamp="1795273200" data-early-card="" data-early-card-timestamp="" data-card-event-title="Main Card" >
<h3 class="c-card-event--result__headline"><a href="/event/ufc-334">Gane vs Hokit</a></h3>
<div class="c-card-event--result__date tz-change-data" data-locale="en"
 data-main-card="Sat, Nov 14 / 9:00 PM EST" data-main-card-timestamp="1794708000" data-prelims-card="" data-prelims-card-timestamp="1794700800" data-early-card="" data-early-card-timestamp="1794693600" data-card-event-title="Main Card" >
<h3 class="c-card-event--result__headline"><a href="/event/ufc-fight-night-february-06-2027">TBD vs TBD</a></h3>
<div class="c-card-event--result__date tz-change-data" data-locale="en"
 data-main-card="" data-main-card-timestamp="" data-prelims-card="" data-prelims-card-timestamp="" data-early-card="" data-early-card-timestamp="" >
"""


def test_parse_listing_reads_timestamps_and_skips_untimed():
    cards = parse_listing(LISTING)
    assert [c.slug for c in cards] == ["ufc-fight-night-november-21-2026", "ufc-334"]
    fn, ufc334 = cards
    assert fn.headline == "Prochazka vs Stirling"
    assert fn.main_card_at == dt.datetime(2026, 11, 21, 18, 0)   # 1:00 PM EST
    assert fn.early_prelims_at is None
    # 9:00 PM Eastern on Saturday is 02:00 UTC on Sunday.
    assert ufc334.main_card_at == dt.datetime(2026, 11, 15, 2, 0)
    assert ufc334.early_prelims_at == dt.datetime(2026, 11, 14, 22, 0)


def card(slug, headline, main):
    return ListedCard(slug, headline, main, None, None)


def test_numbered_matches_by_number_including_sponsor_prefix():
    cards = [card("cryptocom-ufc-331", "Van vs Pantoja 2", dt.datetime(2026, 9, 20, 2)),
             card("ufc-334", "Gane vs Hokit", dt.datetime(2026, 11, 15, 2))]
    assert match_card("UFC 331: Van vs. Pantoja 2", dt.date(2026, 9, 19), cards).slug == "cryptocom-ufc-331"
    assert match_card("UFC 334: Gane vs. Hokit", dt.date(2026, 11, 14), cards).slug == "ufc-334"
    # UFC 33 must not match ufc-334 / ufc-331.
    assert match_card("UFC 33: Victory in Vegas", dt.date(2026, 11, 14), cards) is None


def test_fight_night_matches_by_date_and_headline():
    cards = [card("ufc-fight-night-november-21-2026", "Prochazka vs Stirling", dt.datetime(2026, 11, 21, 18)),
             card("ufc-fight-night-december-05-2026", "Adesanya vs Magomedov", dt.datetime(2026, 12, 5, 18))]
    hit = match_card("UFC Fight Night: Prochazka vs. Stirling", dt.date(2026, 11, 21), cards)
    assert hit.slug == "ufc-fight-night-november-21-2026"
    # Right names, wrong week: no match.
    assert match_card("UFC Fight Night: Prochazka vs. Stirling", dt.date(2026, 10, 1), cards) is None
    # A UTC date one day after ours (late-night card) still matches.
    late = [card("ufc-fight-night-x", "Silva vs Delgado", dt.datetime(2026, 9, 13, 2))]
    assert match_card("Noche UFC: Silva vs. Delgado", dt.date(2026, 9, 12), late).slug == "ufc-fight-night-x"


def test_accented_names_match():
    cards = [card("ufc-fight-night-y", "Procházka vs Stirling", dt.datetime(2026, 11, 21, 18))]
    assert match_card("UFC Fight Night: Prochazka vs. Stirling", dt.date(2026, 11, 21), cards) is not None
