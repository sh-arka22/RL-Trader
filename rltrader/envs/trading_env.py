"""Custom Gymnasium trading environment. ~430 LOC, replacing TensorTrade.

RESEARCH.md rejected TensorTrade for concrete, cited reasons and this environment exists
to not repeat any of them:

  - dead slippage code, 30 bps hidden default commission          -> costs.py, explicit levels
  - no partial fills                                              -> ADV-cap partial fills, §2
  - same-bar fills (observe close 100.0, fill 100.0)               -> t open[t+1], §3
  - fails gymnasium.check_env (no super().reset(seed=seed))        -> reset() calls it, tested

FILL CONVENTION (matches rltrader.eval.harness exactly, on purpose)
---------------------------------------------------------------------
A target weight vector decided from data ending at bar t is filled at the OPEN of t+1 and
earns the open-to-open return to t+2. This environment and the evaluation harness must
agree bit-for-bit on this, or a policy trained in the environment would be scored by a
harness that fills differently — a silent train/eval mismatch. `PPOStrategy` (agent.py)
is the adapter that runs a trained policy through the harness for a fair, apples-to-apples
comparison against every baseline in S3.

COST IS TRAINED-IN, NOT OVERLAID (ARCHITECTURE.md 3.6, added 2026-09-15 after the FinRL
replication)
---------------------------------------------------------------------
A single ``cost_model`` is fixed for the lifetime of an environment instance. A cost-aware
policy trained at 0 bps and one trained at 10 bps are different policies; there is no
single trained policy that can be re-costed after the fact. Training three cost levels
means training three environments and three policies.

ACTION SPACE
------------
``Box(-1, 1, shape=(n_assets,))`` — SB3's own guidance is to keep Box action spaces
symmetric and normalised (`gymnasium.check_env` warns otherwise); internally this is
scaled by ``ACTION_SCALE`` to logit range before the softmax. Cash is an IMPLICIT logit
of 0, so the policy expresses "flee to cash" by pushing every risky logit negative;
softmax over ``[logits..., 0]`` always returns a long-only vector summing to 1 by
construction, with no invalid-action case for the policy to discover.

OBSERVATION SPACE
-----------------
``Box(-10, 10, shape=(n_assets * lookback + n_assets + 1,))``: a flattened window of the
last ``lookback`` daily returns per asset, followed by the CURRENTLY HELD weights
(including cash). The held weights are in the observation because the reward depends on
the trade from here, not from the last target — the same distinction that fixed the
harness turnover bug in S3.

TURBULENCE LIQUIDATION (FinRL, Apache-2.0 concept, reimplemented)
---------------------------------------------------------------------
A simplified Mahalanobis-distance trigger on the current return vector vs. its trailing
covariance. Above ``turbulence_threshold`` the environment forces the FILL toward cash
regardless of the action, and reports it in ``info["turbulence_triggered"]`` — the
override is visible, not silent.
"""
from __future__ import annotations
import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from ..eval import costs as C


ACTION_SCALE = 5.0   # maps the normalised [-1, 1] action to a useful logit range


def softmax_with_cash(logits: np.ndarray) -> np.ndarray:
    """[assets..., cash=0-logit] -> a long-only vector summing to exactly 1."""
    z = np.concatenate([np.asarray(logits, dtype="float64"), [0.0]])
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def build_features(returns_window: np.ndarray, held: np.ndarray) -> np.ndarray:
    """The ONE feature function used by both training (env) and evaluation (agent.py),
    so a trained policy sees numerically identical inputs in both places."""
    return np.concatenate([returns_window.ravel(), held]).astype("float32")


@dataclasses.dataclass
class TurbulenceGate:
    threshold: float | None
    window: int = 60

    def triggered(self, returns_history: pd.DataFrame, t: int) -> bool:
        if self.threshold is None or t < self.window + 1:
            return False
        hist = returns_history.iloc[t - self.window:t].to_numpy()
        cov = np.cov(hist, rowvar=False)
        try:
            inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            return False
        mu = hist.mean(axis=0)
        x = returns_history.iloc[t].to_numpy() - mu
        turb = float(x @ inv @ x.T)
        return turb > self.threshold


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, close: pd.DataFrame, open_: pd.DataFrame, adv: pd.DataFrame,
                vol: pd.DataFrame, cost_model: C.CostModel, lookback: int = 20,
                l1_penalty: float = 0.0, turbulence_threshold: float | None = None,
                portfolio_usd: float = 1_000_000.0):
        super().__init__()
        assert close.shape == open_.shape == adv.shape == vol.shape
        self.close, self.open_, self.adv, self.vol = close, open_, adv, vol
        self.returns = close.pct_change().fillna(0.0)
        self.n = close.shape[1]
        self.lookback = lookback
        self.cost_model = cost_model
        self.l1_penalty = l1_penalty
        self.portfolio_usd = portfolio_usd
        self.turbulence = TurbulenceGate(turbulence_threshold)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n,), dtype="float32")
        obs_dim = self.n * lookback + self.n + 1
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype="float32")

        self._t = 0
        self._w = np.zeros(self.n + 1)
        self._value = portfolio_usd
        self._np_random = None  # set by super().reset(seed=...)

    # ---- the two methods TensorTrade got wrong (RESEARCH.md, issue #418) ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)          # <- the exact line TensorTrade omits
        self._t = self.lookback
        self._w = np.zeros(self.n + 1)
        self._w[-1] = 1.0                  # start fully in cash
        self._value = self.portfolio_usd
        return self._obs(), {"t": self._t}

    def _obs(self) -> np.ndarray:
        window = self.returns.iloc[self._t - self.lookback:self._t].to_numpy()
        return build_features(window, self._w)

    def step(self, action: np.ndarray):
        if self._t + 2 >= len(self.close):
            return self._obs(), 0.0, True, False, {"reason": "end_of_data"}

        target_risky = softmax_with_cash(np.asarray(action) * ACTION_SCALE)[:-1]
        target_cash = 1.0 - target_risky.sum()
        w_target = np.concatenate([target_risky, [target_cash]])

        turbulent = self.turbulence.triggered(self.returns, self._t)
        if turbulent:
            w_target = np.zeros(self.n + 1)
            w_target[-1] = 1.0

        # Partial fills via the SAME ADV cap the harness's cost model enforces —
        # TensorTrade's defect this env exists to fix.
        dw_intended = (w_target - self._w)[:-1]
        adv_row = self.adv.iloc[self._t].to_numpy()
        vol_row = self.vol.iloc[self._t].to_numpy()
        c = C.trade_costs(dw_intended.reshape(1, -1), np.array([self._value]),
                          adv_row.reshape(1, -1), vol_row.reshape(1, -1), self.cost_model)
        notional = np.abs(dw_intended) * self._value
        cap = self.cost_model.adv_cap * adv_row
        filled_notional = np.minimum(notional, cap)
        dw_filled = np.sign(dw_intended) * np.divide(
            filled_notional, self._value, out=np.zeros_like(filled_notional), where=self._value > 0)
        w_filled_risky = self._w[:-1] + dw_filled
        w_filled = np.concatenate([w_filled_risky, [1.0 - w_filled_risky.sum()]])

        r_next = (self.open_.iloc[self._t + 2].to_numpy() / self.open_.iloc[self._t + 1].to_numpy() - 1.0)
        gross = float((w_filled_risky * r_next).sum())
        cost_frac = float(c["cost_fraction"][0])
        net = gross - cost_frac
        turnover_penalty = self.l1_penalty * float(np.abs(dw_filled).sum())
        reward = float(np.log1p(net) - turnover_penalty)

        self._value *= (1.0 + net)
        grown = w_filled_risky * (1.0 + r_next)
        total = grown.sum() + w_filled[-1]
        self._w = (np.concatenate([grown, [w_filled[-1]]]) / total if total > 0
                  else np.concatenate([np.zeros(self.n), [1.0]]))
        self._t += 1
        terminated = self._t + 2 >= len(self.close)

        info = {"gross_return": gross, "net_return": net, "cost_fraction": cost_frac,
                "turbulence_triggered": turbulent, "weights": w_filled.copy(),
                "portfolio_value": self._value, "n_trades_capped": c["n_trades_capped"]}
        return self._obs(), reward, terminated, False, info

    def render(self):
        pass
