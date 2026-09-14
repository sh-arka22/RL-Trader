"""stockanalysis.com — keyless daily OHLCV + adjusted close.

Discovered and verified live on 2026-09-14 while building S2, after Tiingo/Alpaca were
found to need keys this environment does not have. The undocumented endpoint is::

    GET /api/symbol/s/<ticker>/history?range=Max&period=Daily

Verified depth on 2026-09-14: AAPL 11,074 rows to 1982-10-05, XOM 14,771 to 1968-01-03,
NVDA 6,952 to 1999-01-25, TSLA 4,076 to 2010-06-30, META 3,599 to 2012-05-21.

IMPORTANT — it returns ``c`` (raw close) and ``a`` (adjusted close). We keep ``c`` only.
The adjusted series is deliberately discarded: adjustment is our job, from dated actions.

Independence from Yahoo is PROBABLE, NOT PROVEN. It is treated as an independent source
only because the reconciliation report shows non-zero divergence; a mirror would be
bit-identical. That evidence lives in reports/data_reconciliation.md.
"""
from __future__ import annotations
import datetime as dt
import time
import pandas as pd
import requests
from .base import ProviderInfo, ProviderError

_URL = "https://stockanalysis.com/api/symbol/s/{t}/history"
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}


class StockAnalysis:
    info = ProviderInfo("stockanalysis", False, None,
                        "keyless; undocumented endpoint; SPLIT-ADJUSTED OHLC + "
                        "split+dividend adjusted close + volume",
                        price_basis="split_adjusted_as_of_recorded_at")

    def __init__(self, timeout: int = 60, retries: int = 4, pause: float = 1.5):
        self.timeout, self.retries, self.pause = timeout, retries, pause
        self._cache: dict[str, list[dict]] = {}

    def available(self) -> bool:
        return True

    def _fetch(self, ticker: str) -> list[dict]:
        """One HTTP call per ticker per process. The endpoint rate-limits at roughly
        ten rapid requests and then returns HTML instead of JSON, which surfaces as a
        JSONDecodeError — observed 2026-09-14 during the first full S2 ingest."""
        key = ticker.lower()
        if key in self._cache:
            return self._cache[key]
        last = None
        for attempt in range(self.retries):
            try:
                r = requests.get(_URL.format(t=key), params={"range": "Max", "period": "Daily"},
                                 headers=_UA, timeout=self.timeout)
                if r.status_code != 200:
                    raise ProviderError(f"stockanalysis {ticker}: HTTP {r.status_code}")
                payload = r.json()
                rows = payload.get("data") if isinstance(payload, dict) else None
                if not rows:
                    raise ProviderError(f"stockanalysis {ticker}: empty payload")
                self._cache[key] = rows
                return rows
            except Exception as e:      # includes JSONDecodeError on an HTML rate-limit page
                last = e
                time.sleep(self.pause * (2 ** attempt))
        raise ProviderError(f"stockanalysis {ticker}: {type(last).__name__}: {last}")

    def adjusted_reference(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """This provider's OWN adjusted close — used only to score our recomputation (G5)."""
        df = pd.DataFrame(self._fetch(ticker)).rename(columns={"t": "date", "a": "close"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df[["date", "close"]].sort_values("date").reset_index(drop=True)

    def bars(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        df = pd.DataFrame(self._fetch(ticker))
        df = df.rename(columns={"t": "date", "o": "open", "h": "high",
                                "l": "low", "c": "close", "v": "volume"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df[["date", "open", "high", "low", "close", "volume"]]
        return df.sort_values("date").reset_index(drop=True)

    def actions(self, ticker: str) -> pd.DataFrame:
        # No keyless corporate-action endpoint found. Yahoo is the action source.
        return pd.DataFrame(columns=["ex_date", "kind", "value"])
