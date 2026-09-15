"""PPOStrategy must feed a trained model the SAME observation it trained on.

Two independent fill/feature implementations that happen to agree is not evidence that
train and eval match; this test pins them together directly.
"""
import numpy as np
import pandas as pd
import pytest

from rltrader.envs.trading_env import TradingEnv, build_features
from rltrader.eval import costs as C
from rltrader.eval.agent import PPOStrategy


class _StubModel:
    """Records every observation it is asked to predict from."""
    def __init__(self):
        self.seen = []

    def predict(self, obs, deterministic=True):
        self.seen.append(obs.copy())
        return np.zeros(obs.shape[0] - 1 - (len(self.seen[-1]) - 1) or 1), None


@pytest.fixture
def panels():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2019-01-01", periods=100)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (100, 3)), axis=0),
                         index=idx, columns=["A", "B", "C"])
    return close


def test_adapter_reconstructs_the_env_feature_vector_exactly(panels):
    close = panels
    lookback = 10
    returns = close.pct_change().fillna(0.0)
    held_risky = np.array([0.2, 0.1, 0.0])          # what the harness tracks (no cash)
    held_with_cash = np.concatenate([held_risky, [0.7]])

    window = returns.iloc[lookback:2 * lookback].to_numpy()
    expected = build_features(window, held_with_cash)

    class Recorder:
        def predict(self, obs, deterministic=True):
            Recorder.last = obs
            return np.zeros(3), None

    strat = PPOStrategy(Recorder(), lookback=lookback)
    strat.weights(close.iloc[:2 * lookback], held_risky)
    assert np.array_equal(Recorder.last, expected)


def test_adapter_output_is_long_only_and_excludes_cash(panels):
    close = panels

    class ZeroModel:
        def predict(self, obs, deterministic=True):
            return np.zeros(3), None

    strat = PPOStrategy(ZeroModel(), lookback=10)
    w = strat.weights(close.iloc[:50], np.zeros(3))
    assert w.shape == (3,)
    assert (w >= 0).all() and w.sum() < 1.0 + 1e-9   # cash is the implicit remainder
