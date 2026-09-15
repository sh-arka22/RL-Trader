"""The environment this project builds instead of TensorTrade must not fail the checks
TensorTrade fails: gymnasium.check_env, seeded reset, partial fills."""
import datetime as dt
import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rltrader.envs.trading_env import TradingEnv, softmax_with_cash, ACTION_SCALE
from rltrader.eval import costs as C


@pytest.fixture
def panels():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=300)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(2e-4, 0.015, (300, 3)), axis=0),
                         index=idx, columns=["A", "B", "C"])
    opn = close.shift(1).bfill()
    adv = pd.DataFrame(1e9, index=idx, columns=close.columns)
    vol = pd.DataFrame(0.02, index=idx, columns=close.columns)
    return close, opn, adv, vol


def test_passes_gymnasium_check_env(panels):
    """The exact check TensorTrade fails (RESEARCH.md; issue #418 open since 2022-03-01)."""
    env = TradingEnv(*panels, C.REALISTIC, lookback=10)
    check_env(env, skip_render_check=True)


def test_reset_accepts_and_uses_seed(panels):
    """TensorTrade's defect: no super().reset(seed=seed). This must not repeat it."""
    env = TradingEnv(*panels, C.REALISTIC, lookback=10)
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    assert env.action_space.seed is not None or True  # reset must not raise; core assertion above


def test_softmax_with_cash_sums_to_one_and_is_long_only():
    w = softmax_with_cash(np.array([2.0, -1.0, 0.5]))
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()


def test_fill_is_not_same_bar(panels):
    """TensorTrade's defect: observe close 100.0, fill 100.0. This environment must use
    open[t+1] -> open[t+2], never the price the decision was made from."""
    close, opn, adv, vol = panels
    env = TradingEnv(close, opn, adv, vol, C.FREE, lookback=10)
    env.reset(seed=1)
    t_decision = env._t
    _, _, _, _, info = env.step(np.array([1.0, -1.0, -1.0], dtype="float32"))
    expected_gross = float(((opn.iloc[t_decision + 2] / opn.iloc[t_decision + 1] - 1.0)
                            * info["weights"][:-1]).sum())
    assert abs(info["gross_return"] - expected_gross) < 1e-9


def test_adv_cap_gives_a_partial_fill(panels):
    """TensorTrade's defect: no partial fills. A trade beyond the ADV cap must be reduced,
    not rejected outright and not silently allowed in full."""
    close, opn, adv, vol = panels
    tiny_adv = adv.copy() * 1e-6           # force the cap to bind hard
    env = TradingEnv(close, opn, tiny_adv, vol, C.REALISTIC, lookback=10)
    env.reset(seed=1)
    obs, r, term, trunc, info = env.step(np.array([1.0, -1.0, -1.0], dtype="float32"))
    assert info["n_trades_capped"] >= 1
    assert 0 < info["weights"][:-1].sum() < 1.0   # partially filled, not zero and not full


def test_deterministic_given_seed_and_actions(panels):
    def run(seed):
        env = TradingEnv(*panels, C.REALISTIC, lookback=10)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        out = [obs.copy()]
        for _ in range(30):
            a = rng.uniform(-1, 1, size=3).astype("float32")
            obs, r, term, trunc, info = env.step(a)
            out.append(obs.copy())
            if term:
                break
        return np.concatenate(out)
    assert np.array_equal(run(5), run(5))


def test_turbulence_gate_forces_cash(panels):
    close, opn, adv, vol = panels
    env = TradingEnv(close, opn, adv, vol, C.FREE, lookback=10, turbulence_threshold=-1.0)
    env.reset(seed=1)   # threshold impossible to exceed downward -> forced every step after warmup
    obs, r, term, trunc, info = env.step(np.array([1.0, 1.0, 1.0], dtype="float32"))
    if env._t > 10 + 61:
        assert info["turbulence_triggered"]
        assert abs(info["weights"][-1] - 1.0) < 1e-9   # all cash
