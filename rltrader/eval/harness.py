"""Walk-forward evaluation. The measurement apparatus, which RESEARCH.md concluded
matters more than the agent.

THREE STRUCTURAL DEFENCES
-------------------------
1. The strategy is called bar by bar and handed ONLY ``panel.iloc[:t+1]``. It cannot index
   into the future because the future is not in the object it receives. Leakage becomes a
   deliberate act (see oracle.py) rather than an accident.
2. A decision made from the close of session t is filled at the OPEN of t+1 and earns the
   open-to-open return from t+1 to t+2. Same-bar fills are the most common silent
   look-ahead in retail backtests.
3. Every run is appended to a hash-chained TrialLog before its metrics are read, so the
   denominator of a Deflated Sharpe Ratio is recorded as it happens rather than recalled
   afterwards.

WHAT THIS HARNESS DOES NOT DO
-----------------------------
It does not certify that a good result is real. RESEARCH.md: a planted look-ahead oracle
scored a Deflated Sharpe Ratio of 1.00 (arXiv:2608.27734). Statistics do not detect
leakage. Only the structure above does, and only for leaks that come through the data
feed. A leak through a feature computed elsewhere is invisible here and must be caught by
ablation.
"""
from __future__ import annotations
import dataclasses
import datetime as dt
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from . import costs as C
from .metrics import summary
from .trial_log import TrialLog


class Strategy(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> None:
        """Called once per fold with the training slice. May be a no-op."""

    def weights(self, history: pd.DataFrame, held: np.ndarray) -> np.ndarray:
        """History ends at the decision bar. Return target weights for the NEXT open.

        ``held`` is the book as it actually stands after drift. A strategy needs it to
        express "do nothing": without it, returning constant targets silently rebalances
        every bar, and buy-and-hold becomes indistinguishable from daily rebalancing.

        Must sum to <= 1 (the remainder is cash) and be non-negative (long-only).
        """


@dataclasses.dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self) -> str:
        return (f"fold{self.index}: train {self.train_start.date()}..{self.train_end.date()} "
                f"test {self.test_start.date()}..{self.test_end.date()}")


def walk_forward_folds(index: pd.DatetimeIndex, train_years: float = 4.0,
                       test_years: float = 1.0, step_years: float = 1.0) -> list[Fold]:
    """Anchored-start rolling folds. No fold's test window overlaps another's."""
    idx = pd.DatetimeIndex(index).sort_values()
    start, end = idx[0], idx[-1]
    folds, i, cursor = [], 0, start
    while True:
        tr_end = cursor + pd.DateOffset(days=int(train_years * 365.25))
        te_end = tr_end + pd.DateOffset(days=int(test_years * 365.25))
        if tr_end >= end:
            break
        folds.append(Fold(i, cursor, tr_end, tr_end, min(te_end, end)))
        cursor = cursor + pd.DateOffset(days=int(step_years * 365.25))
        i += 1
        if cursor + pd.DateOffset(days=int(train_years * 365.25)) >= end:
            break
    return folds


def liquidity_inputs(store, tickers, index: pd.DatetimeIndex, window: int = 20):
    """Dollar ADV and volatility, both computed from PAST bars only."""
    adv, vol = {}, {}
    for t in tickers:
        tp = store.tape(t).set_index("date")
        tp.index = pd.DatetimeIndex(tp.index)
        tp = tp.reindex(index)
        dollar = (tp["traded_close"] * tp["traded_volume"])
        adv[t] = dollar.rolling(window, min_periods=5).median().shift(1)
        vol[t] = tp["adj_close"].pct_change().rolling(window, min_periods=5).std().shift(1)
    return (pd.DataFrame(adv).bfill(), pd.DataFrame(vol).bfill())


def run_strategy(strategy: Strategy, panel_close: pd.DataFrame, panel_open: pd.DataFrame,
                 adv: pd.DataFrame, vol: pd.DataFrame, warmup: int = 20) -> dict:
    """Bar-by-bar walk. Returns the gross open-to-open stream and the weight path."""
    n = panel_close.shape[1]
    o = panel_open.to_numpy(dtype="float64")
    dates = panel_close.index
    w_held = np.zeros(n)          # weights actually held, AFTER drift
    W, DW, gross, used = [], [], [], []
    for t in range(warmup, len(dates) - 2):
        history = panel_close.iloc[:t + 1]              # <- the only thing the strategy sees
        w = np.asarray(strategy.weights(history, w_held.copy()), dtype="float64")
        if w.shape != (n,):
            raise ValueError(f"{strategy.name}: expected {n} weights, got {w.shape}")
        if (w < -1e-9).any() or w.sum() > 1.0 + 1e-9:
            raise ValueError(f"{strategy.name}: weights must be long-only and sum<=1, got {w}")
        # Turnover is measured against the DRIFTED book, not against the last target.
        # Comparing target to target makes a daily-rebalanced equal-weight portfolio look
        # like it never trades — rebalancing becomes free, which flatters every
        # high-turnover strategy and is exactly the kind of silent subsidy this project
        # exists to avoid. Holding still is the zero-turnover act; restoring weights is not.
        DW.append(w - w_held)
        r_next = o[t + 2] / o[t + 1] - 1.0               # filled at t+1 open, held to t+2 open
        gross.append(float((w * r_next).sum()))
        W.append(w)
        used.append(dates[t + 2])
        grown = w * (1.0 + r_next)
        total = grown.sum() + (1.0 - w.sum())            # cash earns 0
        w_held = grown / total if total > 0 else grown
    return {"gross_returns": np.array(gross), "weights": np.array(W), "dw": np.array(DW),
            "dates": pd.DatetimeIndex(used),
            "adv": adv.iloc[warmup + 2:warmup + 2 + len(gross)].to_numpy(),
            "vol": vol.iloc[warmup + 2:warmup + 2 + len(gross)].to_numpy()}


# Impact and the ADV cap only bite at size. At $1M on five mega-caps ($85bn combined ADV)
# participation is ~1e-5 and impact is genuinely negligible — that is a real finding, not a
# modelling shortcut, and it is only true because of the universe. Stated, not assumed.
DEFAULT_PORTFOLIO_USD = 1_000_000.0


def evaluate(strategy: Strategy, panel_close: pd.DataFrame, panel_open: pd.DataFrame,
             adv: pd.DataFrame, vol: pd.DataFrame, log: TrialLog | None = None,
             levels=C.LEVELS, warmup: int = 20, note: str = "",
             portfolio_usd: float = DEFAULT_PORTFOLIO_USD) -> dict:
    """Run once, score at every cost level. A result is the TABLE, never one number."""
    raw = run_strategy(strategy, panel_close, panel_open, adv, vol, warmup)
    out = {"strategy": strategy.name, "n_periods": len(raw["gross_returns"]),
           "portfolio_usd": portfolio_usd, "by_cost": {}}
    for model in levels:
        applied = C.apply(raw["gross_returns"], raw["dw"], raw["adv"], raw["vol"], model,
                          start_value=portfolio_usd)
        m = summary(applied["net_returns"])
        m["turnover"] = float(np.abs(raw["dw"]).sum(axis=1).mean() / 2)
        m["total_cost_bps"] = applied["total_cost_bps"]
        m["cost_model"] = str(model)
        out["by_cost"][model.name] = m
    if log is not None:
        rec = log.append("evaluation", {"strategy": strategy.name, "warmup": warmup,
                                        "levels": [l.name for l in levels]},
                         {k: v["sharpe"] for k, v in out["by_cost"].items()}, note)
        out["trial_hash"] = rec["hash"]
        out["n_trials_so_far"] = log.n_trials("evaluation")
    return out


def format_table(results: list[dict]) -> str:
    head = f"{'strategy':28s} {'free':>8} {'realistic':>10} {'pessimistic':>12} {'turnover':>9}"
    lines = [head, "-" * len(head)]
    for r in results:
        bc = r["by_cost"]
        lines.append(f"{r['strategy']:28s} {bc['free']['sharpe']:8.3f} "
                     f"{bc['realistic']['sharpe']:10.3f} {bc['pessimistic']['sharpe']:12.3f} "
                     f"{bc['realistic'].get('turnover', float('nan')):9.4f}")
    return "\n".join(lines)
