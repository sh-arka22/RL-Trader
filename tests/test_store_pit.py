"""The store must never reveal a fact before it was knowable."""
import datetime as dt
import pandas as pd
import pytest
from rltrader.data.store import Store, bar_public_at


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    days = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
    df = pd.DataFrame({"date": days, "open": [1.0, 2, 3], "high": [1.0, 2, 3],
                       "low": [1.0, 2, 3], "close": [1.0, 2, 3], "volume": [1e6] * 3})
    s.write_bars(df, "TEST", "yahoo", "i1", recorded_at=pd.Timestamp("2024-01-05", tz="UTC"))
    return s


def test_bar_is_public_at_1600_new_york():
    ts = bar_public_at(pd.Series([dt.date(2024, 1, 2)]))[0]
    assert ts == pd.Timestamp("2024-01-02 21:00", tz="UTC")     # 16:00 EST


def test_no_future_bar_visible_at_0929(store):
    as_of = pd.Timestamp("2024-01-03 09:29", tz="America/New_York").tz_convert("UTC")
    v = store.bars("TEST", as_of=as_of)
    assert list(v["date"]) == [dt.date(2024, 1, 2)], "a bar leaked from the future"


def test_valid_time_and_transaction_time_are_independent(store):
    """A bar public on 2024-01-02 but fetched on 2024-01-05 is visible to a BACKTEST
    standing on 2024-01-03, and invisible to an AUDIT reconstructing the store as it
    stood on 2024-01-04. Two questions, two axes."""
    as_of = pd.Timestamp("2024-01-03 09:29", tz="America/New_York").tz_convert("UTC")
    assert len(store.bars("TEST", as_of=as_of)) == 1
    assert store.bars("TEST", recorded_before=pd.Timestamp("2024-01-04 23:00", tz="UTC")).empty


def test_reingest_appends_and_latest_wins(store, tmp_path):
    days = [dt.date(2024, 1, 2)]
    fixed = pd.DataFrame({"date": days, "open": [9.0], "high": [9.0], "low": [9.0],
                          "close": [9.0], "volume": [1e6]})
    store.write_bars(fixed, "TEST", "yahoo", "i2", recorded_at=pd.Timestamp("2024-02-01", tz="UTC"))
    latest = store.bars("TEST")
    assert float(latest.loc[latest["date"] == days[0], "close"].iloc[0]) == 9.0
    old = store.bars("TEST", recorded_before=pd.Timestamp("2024-01-10", tz="UTC"))
    assert float(old.loc[old["date"] == days[0], "close"].iloc[0]) == 1.0, "history was rewritten"
