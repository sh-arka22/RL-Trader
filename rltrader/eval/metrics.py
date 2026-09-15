"""Performance metrics. Every one states its convention, because the conventions are
where the inflated numbers come from.

Rules enforced here:
  * Returns are SIMPLE returns of a total-return series, not log returns, unless the
    caller asks. Mixing the two silently changes a Sharpe by a few percent.
  * The risk-free rate defaults to 0 and is always reported, because "Sharpe 2.8" with an
    undisclosed rf is not a number.
  * Annualisation is 252 and is a parameter, never a constant baked into a formula.
  * Nothing here knows about transaction costs. Costs are applied to the return stream
    upstream (see costs.py) so that a cost-free number can never be produced by accident.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_returns(prices: pd.Series | np.ndarray, log: bool = False) -> np.ndarray:
    p = np.asarray(prices, dtype="float64")
    if log:
        return np.diff(np.log(p))
    return p[1:] / p[:-1] - 1.0


def sharpe(returns, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[~np.isnan(r)]
    if r.size < 2:
        return float("nan")
    excess = r - rf / periods
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino(returns, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[~np.isnan(r)]
    excess = r - rf / periods
    downside = excess[excess < 0]
    if downside.size < 2:
        return float("nan")
    dd = np.sqrt((downside ** 2).mean())
    return float(excess.mean() / dd * np.sqrt(periods)) if dd > 0 else float("nan")


def max_drawdown(equity) -> float:
    e = np.asarray(equity, dtype="float64")
    peak = np.maximum.accumulate(e)
    return float((e / peak - 1.0).min())


def cagr(equity, periods: int = TRADING_DAYS) -> float:
    e = np.asarray(equity, dtype="float64")
    years = (len(e) - 1) / periods
    return float((e[-1] / e[0]) ** (1 / years) - 1) if years > 0 and e[0] > 0 else float("nan")


def calmar(equity, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(equity)
    return float(cagr(equity, periods) / abs(mdd)) if mdd < 0 else float("nan")


def turnover(weights: np.ndarray) -> float:
    """Mean one-sided turnover per period. weights: (T, n_assets)."""
    w = np.asarray(weights, dtype="float64")
    if w.ndim != 2 or len(w) < 2:
        return float("nan")
    return float(np.abs(np.diff(w, axis=0)).sum(axis=1).mean() / 2)


def equity_curve(returns, start: float = 1.0) -> np.ndarray:
    r = np.asarray(returns, dtype="float64")
    return start * np.cumprod(1.0 + r)


def summary(returns, rf: float = 0.0, periods: int = TRADING_DAYS,
            weights: np.ndarray | None = None) -> dict:
    r = np.asarray(returns, dtype="float64")
    eq = equity_curve(r)
    out = {"n_periods": int(r.size), "sharpe": sharpe(r, rf, periods),
           "sortino": sortino(r, rf, periods), "cagr": cagr(eq, periods),
           "max_drawdown": max_drawdown(eq), "calmar": calmar(eq, periods),
           "total_return": float(eq[-1] - 1.0) if r.size else float("nan"),
           "vol_annual": float(r.std(ddof=1) * np.sqrt(periods)) if r.size > 1 else float("nan"),
           "rf": rf, "periods_per_year": periods}
    if weights is not None:
        out["turnover"] = turnover(weights)
    return out
