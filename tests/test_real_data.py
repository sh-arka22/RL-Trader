"""Anchors against the real tape. Skipped when the dataset has not been built."""
import datetime as dt
import pathlib
import pandas as pd
import pytest

from rltrader.data.build import PRINTED_CLOSES
from rltrader.data.adjust import as_traded
from rltrader.data.store import Store

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (ROOT / "bars").exists(),
                                reason="dataset not built — run scripts/build_dataset.py")


@pytest.mark.parametrize("key", list(PRINTED_CLOSES))
def test_unadjusted_close_matches_the_tape(key):
    ticker, date = key
    s = Store(ROOT)
    bars = s.bars(ticker, provider="yahoo")
    tr = as_traded(bars, s.actions(ticker), as_of=pd.Timestamp.now(tz="UTC"))
    got = float(tr.loc[tr["date"] == date, "traded_close"].iloc[0])
    truth = PRINTED_CLOSES[key]
    assert abs(got - truth) / truth * 1e4 < 25, f"{ticker} {date}: {got:.2f} vs {truth:.2f}"


def test_every_ticker_has_full_history():
    s = Store(ROOT)
    for t in ("NVDA", "TSLA", "AAPL", "META", "XOM"):
        b = s.bars(t)
        assert not b.empty and b["date"].min() <= dt.date(2015, 1, 5), t
        assert set(b["price_basis"]) == {"split_adjusted_as_of_recorded_at"}, t
