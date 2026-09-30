"""Parser tests for app/services/ufc/sherdog_scraper.py on small saved Sherdog HTML fixtures.

No network, no database. Fixtures under tests/fixtures/sherdog/ are trimmed copies of real
Sherdog pages (fighter-info block + fight-history tables; search result table).
"""
from pathlib import Path

import pytest

from app.services.ufc import sherdog_scraper as sd

FIX = Path(__file__).parent / "fixtures" / "sherdog"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- fighter page


@pytest.fixture(scope="module")
def aspinall():
    return sd.parse_fighter_page(_read("fighter_tom_aspinall.html"),
                                 "https://www.sherdog.com/fighter/Tom-Aspinall-65231")


def test_profile_fields(aspinall):
    prof, _ = aspinall
    assert prof.sherdog_id == 65231
    assert prof.name == "Tom Aspinall"
    assert prof.nickname is None
    assert prof.birth_date == "1993-04-11"
    assert prof.nationality == "England"
    assert prof.locality == "Salford, Lancashire"
    assert prof.height == "6'5\""
    assert prof.weight == "255 lbs"
    assert prof.association == "Aspinall BJJ"
    assert prof.weight_class == "Heavyweight"
    assert prof.record == {"wins": 15, "losses": 3, "nc": 1}


def test_pro_rows_only_and_count_matches_record(aspinall):
    prof, bouts = aspinall
    assert all(b.section == "pro" for b in bouts)  # amateur table excluded by default
    assert len(bouts) == prof.n_pro_bouts == 19
    results = [b.result for b in bouts]
    assert results.count("W") == 15 and results.count("L") == 3 and results.count("NC") == 1


def test_newest_row_fields(aspinall):
    _, bouts = aspinall
    b = bouts[0]
    assert b.bout_index == 0
    assert b.result == "NC"
    assert b.opponent_name == "Ciryl Gane"
    assert b.opponent_sherdog_id == 293973
    assert b.opponent_url == "https://www.sherdog.com/fighter/Ciryl-Gane-293973"
    assert b.event_name == "UFC 321 - Aspinall vs. Gane"
    assert b.event_sherdog_id == 108759
    assert b.promotion == "UFC"
    assert b.date == "2025-10-25"
    assert b.method == "No Contest" and b.method_detail == "Accidental Eye Poke"
    assert b.method_class == "NC"
    assert b.referee == "Jason Herzog"
    assert b.round == 1 and b.time == "4:35"


def test_oldest_row_regional(aspinall):
    _, bouts = aspinall
    b = bouts[-1]
    assert b.opponent_name == "Michael Piszczek" and b.opponent_sherdog_id == 146813
    assert b.date == "2014-12-13"
    assert b.promotion == "MMA Versus UK"
    assert b.method == "TKO" and b.method_detail == "Submission to Punches"
    assert b.method_class == "KO/TKO"


def test_amateur_opt_in():
    _, bouts = sd.parse_fighter_page(_read("fighter_tom_aspinall.html"),
                                     "https://www.sherdog.com/fighter/Tom-Aspinall-65231",
                                     include_amateur=True)
    am = [b for b in bouts if b.section == "amateur"]
    assert len(am) == 3  # fixture keeps 2 newest + oldest amateur rows
    assert am[0].bout_index == 0  # index restarts per section
    # amateur row with a plain-text (unlinked) referee
    assert am[-1].referee == "Lee Hasdell"


def test_nickname_and_bellator_promotions():
    prof, bouts = sd.parse_fighter_page(_read("fighter_patricio_freire.html"),
                                        "https://www.sherdog.com/fighter/Patricio-Freire-9960")
    assert prof.name == "Patricio Freire"
    assert prof.nickname == "Pitbull"
    assert prof.birth_date == "1987-07-07"
    assert {b.promotion for b in bouts} >= {"UFC", "Bellator"}


def test_draw_row_and_ksw():
    _, bouts = sd.parse_fighter_page(_read("fighter_salahdine_parnasse.html"),
                                     "https://www.sherdog.com/fighter/Salahdine-Parnasse-172169")
    draws = [b for b in bouts if b.result == "D"]
    assert len(draws) == 1 and draws[0].method_class == "DRAW"
    assert sum(b.promotion == "KSW" for b in bouts) >= 10
    assert all(b.opponent_sherdog_id for b in bouts)
    assert all(b.date for b in bouts)


def test_empty_html_is_404():
    assert sd.parse_fighter_page("", "https://www.sherdog.com/fighter/X-1") == (None, [])


# ---------------------------------------------------------------- search page


def test_search_results():
    cands = sd.parse_search_results(_read("search_charles_oliveira.html"))
    assert [c.sherdog_id for c in cands] == [158789, 30300, 159485]
    c = cands[1]
    assert c.name == "Charles Oliveira"
    assert c.nickname == "do Bronxs"
    assert c.url == "https://www.sherdog.com/fighter/Charles-Oliveira-30300"
    assert c.weight == "155 lbs"


def test_search_no_table():
    assert sd.parse_search_results("<html><body>No results</body></html>") == []


# ---------------------------------------------------------------- helpers


@pytest.mark.parametrize("event,expected", [
    ("UFC Fight Night 282 - Ankalaev vs. Guskov", "UFC"),
    ("The Ultimate Fighter 34 Finale - x", "UFC"),
    ("Bellator MMA - Bellator 123", "Bellator"),
    ("BFC - Bellator Fighting Championships 45", "Bellator"),
    ("PFL 7 - 2024 Regular Season", "PFL"),
    ("CW 107 - Cage Warriors 107", "Cage Warriors"),
    ("LFA 150 - Smith vs. Jones", "LFA"),
    ("ONE Friday Fights 50 - x", "ONE"),
    ("ACB 11 - Vol. 1", "ACA"),
    ("Absolute Championship Berkut - Grand Prix Berkut 9", "ACA"),
    ("Rizin FF - Super Rizin 2", "Rizin"),
    ("KSW 77 - Khalidov vs. Sulecki", "KSW"),
    ("MMA Versus UK 1 - Empire Rises", "MMA Versus UK"),
    ("Eagle FC 44 - Spong vs. Kharitonov", "Eagle FC"),
    ("EFC - Eagle Fighting Championship", "EFC"),  # 2009 Brazilian show, not Khabib's Eagle FC
    ("", None),
])
def test_derive_promotion(event, expected):
    assert sd.derive_promotion(event) == expected


@pytest.mark.parametrize("method,cls", [
    ("KO (Punches)", "KO/TKO"), ("TKO (Doctor Stoppage)", "KO/TKO"),
    ("Submission (Rear-Naked Choke)", "SUB"), ("Technical Submission (Arm-Triangle)", "SUB"),
    ("Decision (Unanimous)", "DEC"), ("Technical Decision (Split)", "DEC"),
    ("Draw (Majority)", "DRAW"), ("No Contest (Accidental Eye Poke)", "NC"),
    ("DQ (Illegal Knee)", "DQ"), ("Disqualification", "DQ"),
])
def test_classify_method(method, cls):
    assert sd.classify_method(method) == cls


def test_dates():
    assert sd.parse_bout_date("Oct / 25 / 2025") == "2025-10-25"
    assert sd.parse_birth_date("Apr 11, 1993") == "1993-04-11"
    assert sd.parse_birth_date("N/A") is None


def test_ids_from_urls():
    assert sd.sherdog_id_from_url("/fighter/Ciryl-Gane-293973") == 293973
    assert sd.sherdog_id_from_url("https://www.sherdog.com/fighter/Jan-B%C5%82achowicz-25292") == 25292
    assert sd.sherdog_id_from_url("/events/UFC-321-108759") is None
    assert sd.event_id_from_url("/events/UFC-321-Aspinall-vs-Gane-108759") == 108759


def test_cache_names_are_id_keyed():
    assert sd._cache_name("https://www.sherdog.com/fighter/Tom-Aspinall-65231") == "fighter_65231.html"
    assert sd._cache_name("https://www.sherdog.com/fighter/Renamed-Slug-65231") == "fighter_65231.html"
    assert sd._cache_name(sd.BASE_URL + sd.SEARCH_PATH + "Tom+Aspinall") == "search_tom+aspinall.html"


def test_block_detection():
    assert sd.looks_blocked(403, "")
    assert sd.looks_blocked(429, "")
    assert sd.looks_blocked(200, "<title>Just a moment...</title>")
    assert not sd.looks_blocked(200, _read("search_charles_oliveira.html"))


# ---------------------------------------------------------------- name matching


def test_fold_accents_and_punctuation():
    assert sd.fold("José Aldo Jr.") == "jose aldo jr"
    assert sd.fold("Jan Błachowicz") == "jan blachowicz"
    assert sd.fold("Khalil Rountree-Jr") == "khalil rountree jr"


@pytest.mark.parametrize("first,last,nick,cand,cand_nick,expect_min", [
    ("Tom", "Aspinall", None, "Tom Aspinall", None, 1.0),
    ("Jan", "Blachowicz", None, "Jan Błachowicz", None, 1.0),          # accents
    ("Aspinall", "Tom", None, "Tom Aspinall", None, 1.0),              # order
    ("Patricio", "Pitbull", None, "Patricio Freire", "Pitbull", 0.8),  # ring name as surname
    ("Charles", "Oliveira", None, "Charles Oliveira da Silva", None, 0.85),  # extra surname
])
def test_name_score_matches(first, last, nick, cand, cand_nick, expect_min):
    assert sd.name_score(first, last, nick, cand, cand_nick) >= expect_min


def test_name_score_rejects_different_person():
    assert sd.name_score("Tom", "Aspinall", None, "Liz Carmouche", None) < 0.75


# ---------------------------------------------------------------- resolver (offline, fake fetcher)


class FakeFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str) -> str:
        url = sd.urljoin(sd.BASE_URL, url)
        self.calls.append(url)
        for key, html in self.pages.items():
            if url.endswith(key):
                return html
        return "<html></html>"


def _fighter(first, last, dob=None, dates=(), nick=None):
    return sd.OurFighter(id=1, ufcstats_id="x", first_name=first, last_name=last, nickname=nick,
                         dob=dob, ufc_fight_dates=list(dates))


def test_resolve_ring_name_by_dob_stops_early():
    ff = FakeFetcher({
        "SearchTxt=Patricio+Pitbull": _read("search_patricio_pitbull.html"),
        "Patricio-Freire-9960": _read("fighter_patricio_freire.html"),
    })
    res, parsed = sd.resolve_fighter(_fighter("Patricio", "Pitbull", dob="1987-07-07"), ff)
    assert res.status == "matched_dob"
    assert res.sherdog_id == 9960
    # Lazy verification: only search + candidate pages up to the DOB hit, no fallback query.
    assert not any("SearchTxt=Pitbull" in c for c in ff.calls)


def test_resolve_without_dob_uses_ufc_dates():
    ff = FakeFetcher({
        "SearchTxt=Tom+Aspinall": '<table class="new_table fightfinder_result"><tr class="table_head"></tr>'
        '<tr><td></td><td><a href="/fighter/Tom-Aspinall-65231">Tom Aspinall</a></td><td></td>'
        '<td></td><td></td><td></td></tr></table>',
        "Tom-Aspinall-65231": _read("fighter_tom_aspinall.html"),
    })
    res, _ = sd.resolve_fighter(_fighter("Tom", "Aspinall", dob=None, dates=["2024-07-27"]), ff)
    assert res.status == "matched_ufc_dates" and res.ufc_date_overlap == 1


def test_resolve_dob_mismatch_flagged():
    ff = FakeFetcher({
        "SearchTxt=Tom+Aspinall": '<table class="new_table fightfinder_result"><tr class="table_head"></tr>'
        '<tr><td></td><td><a href="/fighter/Tom-Aspinall-65231">Tom Aspinall</a></td><td></td>'
        '<td></td><td></td><td></td></tr></table>',
        "Tom-Aspinall-65231": _read("fighter_tom_aspinall.html"),
    })
    res, _ = sd.resolve_fighter(_fighter("Tom", "Aspinall", dob="2001-01-01"), ff)
    assert res.status == "dob_mismatch"


def test_resolve_not_found():
    res, parsed = sd.resolve_fighter(_fighter("Nobody", "Atall"), FakeFetcher({}))
    assert res.status == "not_found" and res.sherdog_id is None and parsed == []


def test_db_guard_refuses_remote_host():
    with pytest.raises(SystemExit):
        sd.load_our_fighters("postgresql://user:pw@prod.example.com:5432/db")
