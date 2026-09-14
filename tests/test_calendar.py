import datetime as dt
from rltrader.data import calendar as cal


def test_known_holidays_are_not_sessions():
    s = set(cal.sessions(dt.date(2024, 1, 1), dt.date(2024, 12, 31)))
    for holiday in [dt.date(2024, 1, 1), dt.date(2024, 7, 4), dt.date(2024, 11, 28),
                    dt.date(2024, 12, 25), dt.date(2024, 6, 19)]:
        assert holiday not in s, f"{holiday} must not be a session"


def test_known_sessions_present():
    s = set(cal.sessions(dt.date(2024, 1, 1), dt.date(2024, 12, 31)))
    assert dt.date(2024, 1, 2) in s and dt.date(2024, 12, 24) in s
    assert 250 <= len(s) <= 253, len(s)


def test_missing_and_unexpected_detected():
    sess = cal.sessions(dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert cal.missing_sessions(sess, dt.date(2024, 1, 1), dt.date(2024, 1, 31)) == []
    assert cal.missing_sessions(sess[:-1], dt.date(2024, 1, 1), dt.date(2024, 1, 31)) == [sess[-1]]
    phantom = sess + [dt.date(2024, 1, 1)]          # New Year's Day
    assert cal.unexpected_sessions(phantom, dt.date(2024, 1, 1), dt.date(2024, 1, 31)) == [dt.date(2024, 1, 1)]
