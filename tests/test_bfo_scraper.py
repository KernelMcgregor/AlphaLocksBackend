"""Parser / matcher tests for app.services.ufc.bfo_scraper (offline, saved fixtures)."""

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

from app.services.ufc import bfo_scraper as bfo

FIX = Path(__file__).parent / "fixtures" / "bfo"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- odds helpers


@pytest.mark.parametrize("dec,am", [(2.22, 122), (2.0, 100), (1.5, -200), (1.16667, -600), (5.5, 450)])
def test_decimal_to_american(dec, am):
    assert bfo.decimal_to_american(dec) == am


def test_devig_pair_sums_to_one():
    a, b = bfo.devig_pair(122, -144)
    assert a + b == pytest.approx(1.0)
    assert a == pytest.approx(0.4329, abs=1e-4)
    assert bfo.devig_pair(None, -144) == (None, None)


def test_parse_american():
    assert bfo.parse_american("+424") == 424
    assert bfo.parse_american("−566") == -566
    assert bfo.parse_american("EV") == 100
    assert bfo.parse_american("") is None


# ------------------------------------------------------------------- decoding


def test_decode_ggd_single_book_series():
    series = bfo.decode_ggd(_read("ggd_fanduel_44968_p1.txt"))
    assert len(series) == 1 and series[0]["name"] == "FanDuel"
    pts = bfo.series_points(series[0])
    assert len(pts) == 11
    assert pts[0].decimal == 2.36 and pts[0].american == 136          # opener
    assert pts[-1].decimal == 2.22 and pts[-1].american == 122        # closer == page cell
    assert pts[0].ts == dt.datetime(2026, 9, 22, 14, 50, 15, tzinfo=dt.timezone.utc)
    assert all(a.ts <= b.ts for a, b in zip(pts, pts[1:]))


def test_decode_ggd_empty():
    assert bfo.decode_ggd("[]") == []
    assert bfo.decode_ggd("") == []


def test_clean_series_drops_isolated_spike():
    pts = bfo.series_points(bfo.decode_ggd(_read("ggd_5dimes_5065_p1_spike.txt"))[0])
    assert any(p.decimal == 5.5 for p in pts)  # raw data has a bogus 5.5 tick among ~1.17
    clean, dropped = bfo.clean_series(pts)
    assert dropped >= 1
    assert all(p.decimal < 2 for p in clean)
    o, c, ots, cts, n, d = bfo._summarise(pts)
    assert (o, c) == (-500, -510)


def test_clean_series_keeps_genuine_move():
    t0 = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    pts = [bfo.SeriesPoint(t0 + dt.timedelta(hours=i), d) for i, d in enumerate([1.5, 1.6, 3.4, 3.5, 3.3])]
    clean, dropped = bfo.clean_series(pts)
    assert dropped == 0 and len(clean) == 5


# -------------------------------------------------------------------- sitemap


def test_parse_sitemap_and_ufc_filter():
    evs = bfo.parse_sitemap_events(_read("sitemap_events.xml"))
    by_slug = {e.slug: e for e in evs}
    assert by_slug["ufc-145-jones-vs-evans-490"].date == dt.date(2012, 4, 21)
    assert by_slug["ufc-145-jones-vs-evans-490"].bfo_event_id == 490
    ufc = {e.slug for e in evs if bfo.is_ufc_event_slug(e.slug)}
    assert {"ufc-vegas-121-4368", "ufc-3159", "ufc-300-3205", "noche-ufc-4318",
            "ufc-145-jones-vs-evans-490"} <= ufc
    assert not ufc & {"ufc-bjj-11-4375", "road-to-ufc-4332", "oktagon-94-4379", "future-events-197"}


# ----------------------------------------------------------------- event page


def test_parse_event_page_modern():
    page = bfo.parse_event_page(_read("event_2026_ufc_vegas_121.html"))
    assert page.name == "UFC Vegas 121"
    assert page.date == dt.date(2026, 9, 26)
    assert page.books[21] == "FanDuel" and page.books[29] == "Kalshi" and page.books[26] == "Unibet"
    assert len(page.matchups) == 2
    mu = page.matchups[0]
    assert (mu.matchup_id, mu.fighter_a, mu.fighter_b) == (44968, "Raoni Barcelos", "Raul Rosas Jr")
    assert mu.fighter_b_slug == "raul-rosas-jr-14405"
    assert mu.page_odds[21] == [122, -144]
    assert mu.page_odds[20] == [130, -163]
    assert mu.page_odds[29] == [-566, 424]   # Kalshi: in-play/post-fight price
    assert 22 not in mu.page_odds            # DraftKings had no line
    assert page.matchups[1].fighter_a == "Ailin Perez"


def test_parse_event_page_2012_has_matchups_but_no_book_cells():
    page = bfo.parse_event_page(_read("event_2012_ufc_145.html"))
    assert page.name == "UFC 145: Jones vs. Evans"
    assert page.date == dt.date(2012, 4, 21)
    assert [(m.matchup_id, m.fighter_a, m.fighter_b) for m in page.matchups] == [
        (5065, "Jon Jones", "Rashad Evans"), (5050, "Michael McDonald", "Miguel Torres")]
    assert all(m.page_odds == {} for m in page.matchups)


class _FakeClient:
    """Serves the FanDuel fixture for (b=21, side 1) and a mirrored side 2."""

    def __init__(self):
        self.calls = []
        self.fd = bfo.decode_ggd(_read("ggd_fanduel_44968_p1.txt"))

    def chart(self, matchup_id, side, book_id=None, referer=None, recent=False):
        self.calls.append((matchup_id, side, book_id))
        if matchup_id != 44968 or book_id not in (None, 21):
            return []
        s = self.fd[0]
        if side == 2:  # fabricate the other side at roughly fair complement
            s = {"name": s["name"], "data": [{"x": p["x"], "y": round(p["y"] / (p["y"] - 1) * 0.95, 3)}
                                             for p in s["data"]]}
        return [{"name": "Mean" if book_id is None else "FanDuel", "data": s["data"]}]


def test_build_rows_history_modes():
    html = _read("event_2026_ufc_vegas_121.html")
    page = bfo.parse_event_page(html)
    ref = bfo.BFOEventRef("ufc-vegas-121-4368", 4368, bfo.BASE_URL + "/events/ufc-vegas-121-4368",
                          dt.date(2026, 9, 26))

    rows, moves = bfo.build_rows(ref, page, None, history="none")
    books = {r["bookmaker"] for r in rows if r["bfo_matchup_id"] == 44968}
    assert "Kalshi" not in books and "Polymarket" not in books  # exchanges excluded by default
    fd = next(r for r in rows if r["bfo_matchup_id"] == 44968 and r["bookmaker"] == "FanDuel")
    assert (fd["close_a"], fd["close_b"], fd["close_source"]) == (122, -144, "event_page")
    assert fd["close_prob_a"] == pytest.approx(0.4329, abs=1e-4)
    assert moves == []

    client = _FakeClient()
    rows, moves = bfo.build_rows(ref, page, client, history="books", legacy_books={1: "5Dimes"},
                                 max_matchups=1)
    fd = next(r for r in rows if r["bookmaker"] == "FanDuel")
    assert fd["open_a"] == 136 and fd["close_a"] == 122 and fd["n_points_a"] == 11
    assert fd["open_ts_a"] == "2026-09-22T14:50:15Z"
    assert not any(r["bookmaker"] == "5Dimes" for r in rows)  # empty legacy series -> no row
    cons = next(r for r in rows if r["bookmaker"] == bfo.CONSENSUS_BOOK_NAME)
    assert cons["open_a"] == 136
    assert cons["close_source"].startswith("median_of_")
    assert cons["close_ts_a"] == ""
    assert {m["bookmaker"] for m in moves} == {"FanDuel", bfo.MEAN_BOOK_NAME}
    assert (44968, 1, 29) not in client.calls  # never fetch exchange charts by default


# ------------------------------------------------------------------- matching


@pytest.mark.parametrize("a,b,min_score", [
    ("Raul Rosas Jr", "Raul Rosas Jr.", 1.0),
    ("Jose Aldo", "José Aldo", 1.0),
    ("Song Yadong", "Yadong Song", 0.99),
    ("Alatengheili", " Alatengheili", 1.0),
    ("TJ Dillashaw", "T.J. Dillashaw", 1.0),
    ("Zach Reese", "Zachary Reese", 1.0),
    ("Sean O'Malley", "Sean OMalley", 1.0),
    ("Ilimbek Akylbek Uulu", "Ilimbek Akylbek uulu", 1.0),
    ("Rory Macdonald", "Rory MacDonald", 1.0),
])
def test_name_similarity_positive(a, b, min_score):
    assert bfo.name_similarity(a, b) >= min_score


@pytest.mark.parametrize("a,b", [("Ian Machado Garry", "Valesca Machado"),
                                 ("Tom Breese", "Zach Reese"),
                                 ("Jon Jones", "Rashad Evans")])
def test_name_similarity_negative(a, b):
    assert bfo.name_similarity(a, b) < 0.82


def test_match_matchups_either_corner_and_date_window():
    db = [
        {"id": 1, "fight_date": dt.date(2026, 9, 26), "event_name": "FN", "red": "Raul Rosas Jr.",
         "blue": "Raoni Barcelos"},
        {"id": 2, "fight_date": dt.date(2026, 9, 26), "event_name": "FN", "red": "Norma Dumont",
         "blue": "Ailin Perez"},
        {"id": 3, "fight_date": dt.date(2026, 9, 20), "event_name": "other", "red": "Raoni Barcelos",
         "blue": "Someone Else"},
    ]
    mus = [
        {"bfo_matchup_id": 10, "fighter_a": "Raoni Barcelos", "fighter_b": "Raul Rosas Jr",
         "event_date": "2026-09-27"},  # BFO split bucket, one day later
        {"bfo_matchup_id": 11, "fighter_a": "Ailin Perez", "fighter_b": "Norma Dumont",
         "event_date": dt.date(2026, 9, 26)},
        {"bfo_matchup_id": 12, "fighter_a": "Mickey Gall", "fighter_b": "Sedriques Dumas",
         "event_date": dt.date(2026, 9, 26)},  # cancelled bout
    ]
    res = bfo.match_matchups(mus, db)
    assert res[10]["fight_id"] == 1 and res[10]["swapped"] is True
    assert res[11]["fight_id"] == 2 and res[11]["swapped"] is True
    assert 12 not in res
    # Outside the +-2 day window -> no match.
    far = [{"bfo_matchup_id": 13, "fighter_a": "Raoni Barcelos", "fighter_b": "Someone Else",
            "event_date": dt.date(2026, 9, 26)}]
    assert bfo.match_matchups(far, db) == {}


def test_safe_db_url_refuses_remote():
    with pytest.raises(SystemExit):
        bfo._safe_db_url("postgresql://user:pw@prod.example.com:5432/db")
    assert bfo._safe_db_url("postgresql://localhost/alocks_local")
    assert bfo._safe_db_url("sqlite:///x.db")


def test_load_db_fights_sqlite(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE ufc_fighters (id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, nickname TEXT);
        CREATE TABLE ufc_events (id INTEGER PRIMARY KEY, name TEXT, date DATE);
        CREATE TABLE ufc_fights (id INTEGER PRIMARY KEY, date DATE, event_id INT,
                                 red_fighter_id INT, blue_fighter_id INT);
        INSERT INTO ufc_fighters VALUES (1,'Jon','Jones',NULL),(2,'Rashad','Evans','Suga');
        INSERT INTO ufc_events VALUES (1,'UFC 145: Jones vs Evans','2012-04-21');
        INSERT INTO ufc_fights VALUES (7,NULL,1,1,2);
    """)
    con.commit()
    con.close()
    fights = bfo.load_db_fights(f"sqlite:///{path}", dt.date(2012, 4, 19), dt.date(2012, 4, 23))
    assert len(fights) == 1 and fights[0]["red"] == "Jon Jones"
    assert fights[0]["fight_date"] == dt.date(2012, 4, 21)
    res = bfo.match_matchups([{"bfo_matchup_id": 5065, "fighter_a": "Jon Jones",
                               "fighter_b": "Rashad Evans", "event_date": "2012-04-21"}], fights)
    assert res[5065]["fight_id"] == 7 and res[5065]["swapped"] is False
