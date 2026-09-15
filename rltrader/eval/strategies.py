"""Reference strategies, including one that cheats on purpose."""
from __future__ import annotations
import numpy as np
import pandas as pd


class EqualWeight:
    """The control. Holds every name in equal proportion, rebalanced each bar."""
    name = "equal_weight"

    def __init__(self, n: int):
        self.n = n

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def weights(self, history: pd.DataFrame, held: np.ndarray) -> np.ndarray:
        return np.full(self.n, 1.0 / self.n)


class BuyAndHoldFirst:
    """Buys once on the first decision bar and never trades again — zero turnover,
    so the gap between its FREE and PESSIMISTIC columns measures the cost model itself."""
    name = "buy_and_hold"

    def __init__(self, n: int):
        self.n = n
        self._w = None

    def fit(self, train: pd.DataFrame) -> None:
        self._w = None

    def weights(self, history: pd.DataFrame, held: np.ndarray) -> np.ndarray:
        if held.sum() <= 1e-12:                 # first bar: buy the basket
            return np.full(self.n, 1.0 / self.n)
        return held                              # afterwards: hold, and let it drift


class Momentum:
    """Long the top-k names by trailing return. A real, cheap, honest baseline.

    ``name`` encodes the parameters. A class-level constant name (the original bug: every
    lookback/top_k combination shared "momentum_60d_top2") makes six distinct trials
    collide into one key wherever results are indexed by name — including in the trial
    log and in any dict keyed on strategy.name — silently discarding 4 of 6 trials from
    a PBO or DSR count that must be exact to mean anything.
    """

    def __init__(self, n: int, lookback: int = 60, top_k: int = 2):
        self.n, self.lookback, self.top_k = n, lookback, top_k
        self.name = f"momentum_{lookback}d_top{top_k}"

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def weights(self, history: pd.DataFrame, held: np.ndarray) -> np.ndarray:
        if len(history) <= self.lookback:
            return np.full(self.n, 1.0 / self.n)
        past = history.iloc[-self.lookback - 1]
        now = history.iloc[-1]
        score = (now / past - 1.0).to_numpy()
        w = np.zeros(self.n)
        for i in np.argsort(score)[-self.top_k:]:
            if score[i] > 0:
                w[i] = 1.0 / self.top_k
        return w


class PlantedOracle:
    """DELIBERATE LOOK-AHEAD. Not a strategy — an instrument for testing the harness.

    RESEARCH.md / arXiv:2608.27734: a planted look-ahead oracle produced a design Sharpe
    of 34.7 and an evaluation Sharpe of 51.5, and the Deflated Sharpe Ratio still returned
    **1.00**. Statistical deflation does not detect leakage.

    So this exists to answer a different question: *is the harness even capable of showing
    a leak?* A harness that reports a modest Sharpe for a perfect oracle is broken, and
    would hide a real leak too. It must be fed the future explicitly, via the constructor,
    because run_strategy() only ever hands a strategy the past.
    """
    name = "planted_oracle_LOOKAHEAD"

    def __init__(self, future_open: pd.DataFrame, top_k: int = 1):
        self.future = future_open
        self.n = future_open.shape[1]
        self.top_k = top_k
        self._pos = {d: i for i, d in enumerate(future_open.index)}

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def weights(self, history: pd.DataFrame, held: np.ndarray) -> np.ndarray:
        t = self._pos[history.index[-1]]
        o = self.future.to_numpy(dtype="float64")
        if t + 2 >= len(o):
            return np.zeros(self.n)
        r_next = o[t + 2] / o[t + 1] - 1.0        # the return it is about to be scored on
        w = np.zeros(self.n)
        for i in np.argsort(r_next)[-self.top_k:]:
            if r_next[i] > 0:
                w[i] = 1.0 / self.top_k
        return w
