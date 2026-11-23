"""Date windows for the "best of" lists (robberies, most exciting fights and cards).

A window is ``all``, ``year:YYYY`` or ``last:Nm`` (the last N months up to today).
"""
from __future__ import annotations

import datetime as dt
import re

_LAST = re.compile(r"^last:(\d{1,3})m$")
_YEAR = re.compile(r"^year:(\d{4})$")


def parse_window(window: str | None, today: dt.date | None = None) -> tuple[dt.date | None, dt.date | None]:
    """(start, end) dates, inclusive, either open-ended. Raises ValueError on a bad window."""
    w = (window or "all").strip().lower()
    if w == "all":
        return None, None
    if m := _YEAR.match(w):
        y = int(m.group(1))
        return dt.date(y, 1, 1), dt.date(y, 12, 31)
    if m := _LAST.match(w):
        n = int(m.group(1))
        if n < 1:
            raise ValueError(f"bad window {window!r}")
        today = today or dt.date.today()
        y, mo = divmod(today.year * 12 + today.month - 1 - n, 12)
        # Same day n months back, clamped to the end of a shorter month.
        day = min(today.day, _month_days(y, mo + 1))
        return dt.date(y, mo + 1, day), None
    raise ValueError(f"bad window {window!r}")


def _month_days(y: int, m: int) -> int:
    nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
    return (nxt - dt.timedelta(days=1)).day
