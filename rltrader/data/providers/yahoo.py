"""Yahoo Finance via yfinance. Primary provider and the source of corporate actions.

``auto_adjust=False`` and ``actions=False`` are both deliberate: we want the RAW print.
yfinance 1.x defaults to auto_adjust=True, which silently returns a back-adjusted series —
exactly the thing this store refuses to persist.
"""
from __future__ import annotations
import datetime as dt
import pandas as pd
from .base import ProviderInfo, ProviderError


class Yahoo:
    info = ProviderInfo("yahoo", False, None,
                        "yfinance; SPLIT-ADJUSTED OHLCV (auto_adjust=False only withholds the "
                        "dividend adjustment) + splits/dividends",
                        price_basis="split_adjusted_as_of_recorded_at")

    def available(self) -> bool:
        try:
            import yfinance  # noqa: F401
            return True
        except Exception:
            return False

    def _t(self, ticker: str):
        import yfinance as yf
        return yf.Ticker(ticker)

    def bars(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        h = self._t(ticker).history(start=start.isoformat(),
                                    end=(end + dt.timedelta(days=1)).isoformat(),
                                    interval="1d", auto_adjust=False, actions=False,
                                    raise_errors=True)
        if h is None or h.empty:
            raise ProviderError(f"yahoo {ticker}: empty frame")
        h = h.reset_index()
        h["date"] = pd.to_datetime(h["Date"]).dt.date
        h = h.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close", "Volume": "volume"})
        return h[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)

    def actions(self, ticker: str) -> pd.DataFrame:
        a = self._t(ticker).actions
        if a is None or a.empty:
            return pd.DataFrame(columns=["ex_date", "kind", "value"])
        a = a.reset_index()
        a["ex_date"] = pd.to_datetime(a["Date"]).dt.date
        out = []
        for _, row in a.iterrows():
            if row.get("Dividends", 0):
                out.append({"ex_date": row["ex_date"], "kind": "dividend", "value": float(row["Dividends"])})
            if row.get("Stock Splits", 0):
                out.append({"ex_date": row["ex_date"], "kind": "split", "value": float(row["Stock Splits"])})
        return pd.DataFrame(out, columns=["ex_date", "kind", "value"])
