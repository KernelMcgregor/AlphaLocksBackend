"""BMF belt lineage: pure-function tests. No database.

Run:  DATABASE_URL=sqlite:///<scratch>/t.db venv/bin/python -m pytest -q tests/test_bmf_belt.py
"""
from __future__ import annotations

from datetime import date

from app.services.ufc.champions import (
    bmf_champion_as_of, champions_as_of, is_bmf_title_bout, is_undisputed_title_bout,
)

HOLLOWAY, OLIVEIRA, LOPES = 1, 2, 3
BOUTS = [
    (date(2025, 7, 19), "Decision - Unanimous", HOLLOWAY, HOLLOWAY, 9),
    (date(2026, 3, 7), "Decision - Unanimous", OLIVEIRA, HOLLOWAY, OLIVEIRA),
]
DEC_12 = date(2026, 12, 12)


def test_holder_by_date():
    assert bmf_champion_as_of(BOUTS, date(2026, 1, 1)) == (HOLLOWAY, date(2025, 7, 19))
    assert bmf_champion_as_of(BOUTS, date(2026, 11, 23)) == (OLIVEIRA, date(2026, 3, 7))
    assert bmf_champion_as_of(BOUTS, date(2019, 1, 1)) is None


def test_belt_moves_on_a_win_and_stays_on_a_no_result():
    scheduled = BOUTS + [(DEC_12, None, None, OLIVEIRA, LOPES)]
    assert bmf_champion_as_of(scheduled, DEC_12)[0] == OLIVEIRA
    won = BOUTS + [(DEC_12, "KO/TKO", LOPES, OLIVEIRA, LOPES)]
    assert bmf_champion_as_of(won, DEC_12) == (LOPES, DEC_12)
    nc = BOUTS + [(DEC_12, "Overturned", LOPES, OLIVEIRA, LOPES)]
    assert bmf_champion_as_of(nc, DEC_12)[0] == OLIVEIRA
    draw = BOUTS + [(DEC_12, "Decision - Split", None, OLIVEIRA, LOPES)]
    assert bmf_champion_as_of(draw, DEC_12)[0] == OLIVEIRA


def test_bmf_bout_never_crowns_a_division_champion():
    assert is_bmf_title_bout("UFC BMF Title Bout")
    assert not is_undisputed_title_bout("UFC BMF Title Bout")
    rows = [(DEC_12, "UFC BMF Title Bout", "KO/TKO", LOPES, OLIVEIRA, LOPES)]
    assert champions_as_of(rows, DEC_12) == {}
