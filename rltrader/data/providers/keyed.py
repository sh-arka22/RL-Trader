"""Key-gated providers: Tiingo and Alpaca.

Both were the plan of record in ARCHITECTURE.md. Neither has credentials in this
environment (verified 2026-09-14: no TIINGO_API_KEY / APCA_API_KEY_ID in env; Tiingo
without a token returns HTTP 403 {"detail":"Please supply a token"}).

They are implemented anyway and register automatically the moment a key appears, so
adding a key upgrades the reconciliation from 2 to 3 sources with no code change.
"""
from __future__ import annotations
import datetime as dt
import os
import pandas as pd
import requests
from .base import ProviderInfo, ProviderError


class Tiingo:
    info = ProviderInfo("tiingo", True, "TIINGO_API_KEY",
                        "free tier; TRUE raw daily prices + adjusted", price_basis="raw")

    def available(self) -> bool:
        return bool(os.environ.get(self.info.key_env))

    def bars(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        key = os.environ.get(self.info.key_env)
        if not key:
            raise ProviderError("tiingo: no TIINGO_API_KEY")
        r = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices",
                         params={"startDate": start.isoformat(), "endDate": end.isoformat(),
                                 "format": "json", "token": key}, timeout=60)
        if r.status_code != 200:
            raise ProviderError(f"tiingo {ticker}: HTTP {r.status_code} {r.text[:120]}")
        df = pd.DataFrame(r.json())
        if df.empty:
            raise ProviderError(f"tiingo {ticker}: empty")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)

    def actions(self, ticker: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["ex_date", "kind", "value"])


class Alpaca:
    info = ProviderInfo("alpaca", True, "APCA_API_KEY_ID",
                        "free IEX tier; TRUE raw daily bars (adjustment=raw)", price_basis="raw")

    def available(self) -> bool:
        return bool(os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY"))

    def bars(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        kid, sec = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not (kid and sec):
            raise ProviderError("alpaca: no credentials")
        rows, page = [], None
        while True:
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars",
                             params={"symbols": ticker, "timeframe": "1Day",
                                     "start": start.isoformat(), "end": end.isoformat(),
                                     "adjustment": "raw", "limit": 10000,
                                     **({"page_token": page} if page else {})},
                             headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec},
                             timeout=60)
            if r.status_code != 200:
                raise ProviderError(f"alpaca {ticker}: HTTP {r.status_code} {r.text[:120]}")
            j = r.json()
            rows.extend(j.get("bars", {}).get(ticker, []))
            page = j.get("next_page_token")
            if not page:
                break
        if not rows:
            raise ProviderError(f"alpaca {ticker}: empty")
        df = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low",
                                                "c": "close", "v": "volume", "t": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)

    def actions(self, ticker: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["ex_date", "kind", "value"])
