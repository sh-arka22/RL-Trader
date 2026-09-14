"""Provider protocol. Every source implements the same narrow contract."""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Protocol
import pandas as pd


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    needs_key: bool
    key_env: str | None
    note: str
    price_basis: str = "split_adjusted_as_of_recorded_at"


class Provider(Protocol):
    info: ProviderInfo

    def available(self) -> bool:
        """True if this provider can actually be used right now (keys present, etc.)."""

    def bars(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """RAW UNADJUSTED daily OHLCV.

        Returns columns: date, open, high, low, close, volume.
        Must raise on partial/ambiguous responses rather than returning a short frame
        silently — a silent short frame is indistinguishable from a delisting.
        """

    def actions(self, ticker: str) -> pd.DataFrame:
        """Corporate actions. Columns: ex_date, kind, value. May be empty."""


class ProviderError(RuntimeError):
    pass
