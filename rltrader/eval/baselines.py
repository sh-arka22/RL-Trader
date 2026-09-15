"""The controls. These are not decoration — RESEARCH.md concluded that buy-and-hold is
the thing to beat and that most published RL results do not beat it once costs are real.

Every baseline returns a daily NET return stream, so a baseline and an agent are always
compared on the same footing.
"""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd

from ..data.store import Store


def total_return_panel(store: Store, tickers, start: dt.date, end: dt.date,
                       provider: str = "yahoo", as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Dividend-adjusted close per ticker, aligned on the exchange calendar.

    Uses Store.tape(), which is basis-correct: `adj_close` is the total-return series and
    `traded_close` is what the tape printed. Never use the stored `close` for levels.
    """
    cols = {}
    for t in tickers:
        tp = store.tape(t, as_of=as_of, provider=provider)
        if tp.empty:
            raise ValueError(f"no data for {t}")
        s = tp.set_index("date")["adj_close"]
        cols[t] = s[(s.index >= start) & (s.index <= end)]
    px = pd.DataFrame(cols).sort_index()
    px.index = pd.DatetimeIndex(px.index, name="date")   # resample() needs a real DatetimeIndex
    if px.isna().any().any():
        bad = px.isna().sum()
        raise ValueError(f"misaligned calendar, NaNs per ticker: {bad[bad > 0].to_dict()}")
    return px


def buy_and_hold(px: pd.DataFrame, ticker: str) -> pd.Series:
    return px[ticker].pct_change().dropna()


def equal_weight(px: pd.DataFrame, rebalance: str | None = None) -> pd.Series:
    """Equal-weight portfolio.

    rebalance=None  -> buy once at the start and never trade again (true buy-and-hold).
    rebalance='D'/'ME'/'QE' -> reset to equal weights at that frequency. A daily-rebalanced
    equal-weight portfolio is NOT buy-and-hold: it harvests a rebalancing premium and
    incurs turnover, and quoting it as "buy-and-hold" overstates the passive bar.
    """
    rets = px.pct_change().dropna()
    if rebalance is None:
        norm = px / px.iloc[0]
        return (norm.mean(axis=1)).pct_change().dropna()
    if rebalance == "D":
        return rets.mean(axis=1)
    marks = rets.resample(rebalance).last().index
    w = np.full(px.shape[1], 1.0 / px.shape[1])
    out, idx = [], []
    for day, r in rets.iterrows():
        out.append(float((w * r.to_numpy()).sum()))
        idx.append(day)
        w = w * (1.0 + r.to_numpy())
        w = w / w.sum()
        if day in marks:
            w = np.full(px.shape[1], 1.0 / px.shape[1])
    return pd.Series(out, index=idx)


def weights_of(px: pd.DataFrame, rebalance: str | None) -> np.ndarray:
    """Realised weight path, for turnover accounting."""
    rets = px.pct_change().dropna()
    n = px.shape[1]
    w = np.full(n, 1.0 / n)
    path = []
    marks = set(rets.resample(rebalance).last().index) if rebalance not in (None, "D") else set()
    for day, r in rets.iterrows():
        path.append(w.copy())
        if rebalance == "D":
            w = np.full(n, 1.0 / n)
        else:
            w = w * (1.0 + r.to_numpy())
            w = w / w.sum()
            if day in marks:
                w = np.full(n, 1.0 / n)
    return np.array(path)
