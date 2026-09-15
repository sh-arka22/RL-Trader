"""The bar must be reproducible from the store on demand, or it is not a bar."""
import datetime as dt
import pathlib
import numpy as np
import pytest

from rltrader.eval.bar import PASSIVE_BAR, SINGLE_NAME, TOLERANCE, WINDOW
from rltrader.eval.baselines import buy_and_hold, equal_weight, total_return_panel
from rltrader.eval.metrics import sharpe
from rltrader.data.store import Store

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (ROOT / "bars").exists(), reason="dataset not built")
FIVE = ["NVDA", "TSLA", "AAPL", "META", "XOM"]


@pytest.fixture(scope="module")
def px():
    return total_return_panel(Store(ROOT), FIVE, *WINDOW)


def test_session_count_is_pinned(px):
    assert len(px) == 2940


def test_spy_bar_reproduces():
    spy = total_return_panel(Store(ROOT), ["SPY"], *WINDOW)
    assert abs(sharpe(buy_and_hold(spy, "SPY")) - PASSIVE_BAR["SPY_buy_and_hold"]) < TOLERANCE


@pytest.mark.parametrize("key,rebal", [("EW5_buy_and_hold", None),
                                       ("EW5_daily_rebalanced", "D"),
                                       ("EW5_monthly_rebalanced", "ME")])
def test_equal_weight_bars_reproduce(px, key, rebal):
    got = sharpe(equal_weight(px, rebal))
    assert abs(got - PASSIVE_BAR[key]) < TOLERANCE, f"{key}: {got:.4f} vs {PASSIVE_BAR[key]}"


@pytest.mark.parametrize("ticker", FIVE)
def test_single_name_bars_reproduce(px, ticker):
    assert abs(sharpe(buy_and_hold(px, ticker)) - SINGLE_NAME[ticker]) < TOLERANCE


def test_rebalancing_is_not_buy_and_hold(px):
    """Quoting a rebalanced portfolio as 'buy-and-hold' overstates the passive bar."""
    assert sharpe(equal_weight(px, "D")) > sharpe(equal_weight(px, None))


def test_the_unreproducible_research_value_is_not_any_convention(px):
    """RESEARCH.md recorded 1.07. Guard against it quietly reappearing."""
    conventions = [sharpe(equal_weight(px, r)) for r in (None, "D", "ME")]
    conventions.append(sharpe(np.log1p(px.pct_change().dropna()).mean(axis=1)))
    assert not any(abs(c - 1.07) < 0.02 for c in conventions), conventions
