"""parse_window: the date windows behind the best-of lists. No database."""
import datetime as dt

import pytest

from app.services.ufc.time_window import parse_window

T = dt.date(2026, 11, 18)


def test_windows():
    assert parse_window("all", T) == (None, None)
    assert parse_window(None, T) == (None, None)
    assert parse_window("year:2025", T) == (dt.date(2025, 1, 1), dt.date(2025, 12, 31))
    assert parse_window("last:6m", T) == (dt.date(2026, 5, 18), None)
    assert parse_window("last:12m", T) == (dt.date(2025, 11, 18), None)
    # Clamped to the end of a shorter month.
    assert parse_window("last:1m", dt.date(2026, 3, 31)) == (dt.date(2026, 2, 28), None)


@pytest.mark.parametrize("w", ["year:25", "last:0m", "last:6", "forever"])
def test_bad_windows(w):
    with pytest.raises(ValueError):
        parse_window(w, T)
