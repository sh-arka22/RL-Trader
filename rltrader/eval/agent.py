"""Adapter: a trained SB3 policy becomes a harness.Strategy.

This is what makes the comparison in S3_BASELINE.md apples-to-apples. The agent is
trained inside TradingEnv (its own step/fill/cost loop, for SB3's sake), but it is SCORED
by walking rltrader.eval.harness.run_strategy exactly like every baseline — same fill
timing, same cost model application, same metrics. Two different fill implementations
that happen to agree is not evidence; one fill implementation used for both is.

The one piece that MUST be identical between training and evaluation is the feature
function, and it is: both call rltrader.envs.trading_env.build_features. If that function
changes, the model and the adapter go stale together, which is the point of sharing it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..envs.trading_env import ACTION_SCALE, build_features, softmax_with_cash


class PPOStrategy:
    """Deterministic inference wrapper. name encodes the checkpoint for the trial log."""

    def __init__(self, model, lookback: int = 20, tag: str = ""):
        self.model = model
        self.lookback = lookback
        self.name = f"ppo{('_' + tag) if tag else ''}"
        self._returns_cache: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> None:
        pass   # training happens offline via scripts/train_agent.py, not inside the harness

    def weights(self, history: pd.DataFrame, held: np.ndarray) -> np.ndarray:
        if len(history) <= self.lookback:
            n = history.shape[1]
            return np.zeros(n)
        returns = history.pct_change().fillna(0.0)
        window = returns.iloc[-self.lookback:].to_numpy()
        # TradingEnv's held vector is (n_assets + 1): risky weights THEN cash. The harness's
        # `held` is risky-only (n_assets); cash is the implicit remainder there. Reconstruct
        # the n+1 vector so build_features sees numerically identical input to training —
        # a size mismatch here is a silent train/eval skew, not just a shape error.
        held_with_cash = np.concatenate([held, [max(0.0, 1.0 - held.sum())]])
        obs = build_features(window, held_with_cash).astype("float32")
        action, _ = self.model.predict(obs, deterministic=True)
        w_all = softmax_with_cash(np.asarray(action) * ACTION_SCALE)
        return w_all[:-1]     # harness weights() excludes cash; it is the sum-to-<=1 remainder
