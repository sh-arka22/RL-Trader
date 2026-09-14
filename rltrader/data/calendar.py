"""NYSE session calendar. No synthetic bars, ever.

A missing bar is either a holiday (expected, must not appear) or a data gap (a defect,
must be reported). Without an authoritative calendar those two are indistinguishable,
which is how "0% missing" gets silently claimed on incomplete data.
"""
from __future__ import annotations
import datetime as dt
import functools
import pandas as pd


@functools.lru_cache(maxsize=8)
def _cal(name: str = "XNYS"):
    import exchange_calendars as xcals
    return xcals.get_calendar(name)


def sessions(start: dt.date, end: dt.date, name: str = "XNYS") -> list[dt.date]:
    idx = _cal(name).sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [d.date() for d in idx]


def missing_sessions(dates, start: dt.date, end: dt.date, name: str = "XNYS") -> list[dt.date]:
    have = set(dates)
    return [d for d in sessions(start, end, name) if d not in have]


def unexpected_sessions(dates, start: dt.date, end: dt.date, name: str = "XNYS") -> list[dt.date]:
    """Bars on days the exchange was shut — a provider defect, not a bonus."""
    valid = set(sessions(start, end, name))
    return [d for d in sorted(set(dates)) if start <= d <= end and d not in valid]
