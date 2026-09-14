from .base import Provider, ProviderError, ProviderInfo
from .yahoo import Yahoo
from .stockanalysis import StockAnalysis
from .keyed import Tiingo, Alpaca

ALL = [Yahoo(), StockAnalysis(), Tiingo(), Alpaca()]


def available_providers():
    """Providers usable right now. Order matters: index 0 is primary."""
    return [p for p in ALL if p.available()]


def get(name: str):
    for p in ALL:
        if p.info.name == name:
            return p
    raise KeyError(name)


__all__ = ["Provider", "ProviderError", "ProviderInfo", "Yahoo", "StockAnalysis",
           "Tiingo", "Alpaca", "ALL", "available_providers", "get"]
