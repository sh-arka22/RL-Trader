"""Split/dividend handling, and the un-adjustment that free data forces on you.

READ THIS BEFORE CHANGING ANYTHING HERE.

The first version of this module was wrong, and the way it was wrong is the single most
expensive mistake in retail quant data work, so it is documented rather than deleted.

It assumed the providers return RAW prices and that splits must be applied going back.
They do not. Verified on real data, 2026-09-14:

    NVDA close 2024-06-07   stored by BOTH providers: 120.888   actually printed: ~1208
    TSLA close 2022-08-24   stored by BOTH providers: 297.097   actually printed:  ~891
    AAPL close 2020-08-28   stored by BOTH providers: 124.808   actually printed:  ~499

Yahoo's ``Close`` is split-adjusted even with ``auto_adjust=False`` (only the DIVIDEND
adjustment is withheld), and stockanalysis behaves the same way. So the stored series is
already split-adjusted as of ``recorded_at``. Re-applying splits double-counts them. The
symptom was a ~90,000 bp return error on exactly the split dates, and nowhere else —
five bad rows hiding in 14,705 good ones.

Consequences, all enforced below:

1. ``adjusted()`` applies DIVIDENDS ONLY. Splits are already in the series.
2. The stored series is NOT immutable. The next split silently rewrites every past price.
   That is why ``recorded_at`` exists and why ``price_basis`` is stored per row.
3. The price a trader actually saw on day t is recovered by UN-adjusting: multiply by every
   split ratio with ex_date > t. Use ``as_traded()`` for share counts, round lots and any
   claim about price levels. Use the stored series for returns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SPLIT_ADJUSTED = "split_adjusted_as_of_recorded_at"
RAW = "raw"


def _dedupe(actions: pd.DataFrame) -> pd.DataFrame:
    """One row per (ex_date, kind). Two providers reporting the same 10:1 split would
    otherwise multiply to 100:1 — a +90,000 bp error that appears the moment a second
    action source is added. Finding L1/M3 of the S2 review."""
    if actions is None or actions.empty:
        return actions
    return actions.sort_values(["ex_date", "kind", "value"]).drop_duplicates(
        subset=["ex_date", "kind"], keep="first")


def _known(actions: pd.DataFrame, as_of: pd.Timestamp, kind: str | None = None) -> pd.DataFrame:
    if actions is None or actions.empty:
        return pd.DataFrame(columns=["ex_date", "kind", "value"])
    a = actions[pd.to_datetime(actions["first_public_at"], utc=True) <= pd.Timestamp(as_of)]
    if kind:
        a = a[a["kind"] == kind]
    return _dedupe(a).sort_values("ex_date")


def _vintage(actions: pd.DataFrame, kind: str | None = None) -> pd.DataFrame:
    """Every action baked into the stored series, regardless of the backtest clock.

    The basis of the stored prices is a property of ``recorded_at`` — of the VINTAGE —
    not of the simulated present. Filtering these by ``as_of`` leaves prices that are
    neither as-traded nor as-stored. That was finding H1.
    """
    if actions is None or actions.empty:
        return pd.DataFrame(columns=["ex_date", "kind", "value"])
    a = actions[actions["kind"] == kind] if kind else actions
    return _dedupe(a).sort_values("ex_date")


def dividend_factors(bars: pd.DataFrame, actions: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Back-adjustment multiplier for cash dividends only, 1.0 at the last bar."""
    b = bars.sort_values("date").reset_index(drop=True)
    mult = pd.Series(1.0, index=b["date"].values, dtype="float64")
    close = pd.Series(b["close"].values, index=b["date"].values, dtype="float64")
    for _, a in _known(actions, as_of, "dividend").iterrows():
        prior = close.index[close.index < a["ex_date"]]
        if len(prior) == 0:
            continue
        prev = float(close.loc[prior[-1]])
        if prev <= 0:
            continue
        mult.loc[mult.index < a["ex_date"]] *= (1.0 - float(a["value"]) / prev)
    # Contract: exactly 1.0 at the last bar. An action dated after the final bar would
    # otherwise rescale the whole series (finding M4: -68.4 bp under truncated bars).
    # A uniform rescale is invisible to any returns-based check, so it is normalised here.
    last = float(mult.iloc[-1]) if len(mult) else 1.0
    return mult / last if last not in (0.0, 1.0) else mult


def unadjust_factors(bars: pd.DataFrame, actions: pd.DataFrame,
                     as_of: pd.Timestamp | None = None) -> pd.Series:
    """Multiplier that turns the stored split-adjusted series back into printed prices.

    price_as_traded(t) = price_stored(t) * prod(ratio for every split with ex_date > t)

    ``as_of`` is accepted and IGNORED, deliberately. Un-adjustment reverses what the
    vendor baked in, and the vendor bakes in every split up to ``recorded_at``. Gating it
    on the backtest clock (the original behaviour) returned 109.63 for NVDA 2024-05-31 at
    as_of=2024-06-01 when the tape printed 1096.33 — the exact 10x position-sizing error
    this module exists to prevent. Finding H1.
    """
    b = bars.sort_values("date").reset_index(drop=True)
    mult = pd.Series(1.0, index=b["date"].values, dtype="float64")
    for _, a in _vintage(actions, "split").iterrows():
        r = float(a["value"])
        if r > 0:
            mult.loc[mult.index < a["ex_date"]] *= r
    return mult


def adjusted(bars: pd.DataFrame, actions: pd.DataFrame, as_of: pd.Timestamp,
             include_dividends: bool = True, price_basis: str = SPLIT_ADJUSTED) -> pd.DataFrame:
    """Total-return series for a given knowledge state. Splits are NOT re-applied."""
    b = bars.sort_values("date").reset_index(drop=True).copy()
    # Trust the data over the argument: a raw-stamped row must not be dividend-adjusted as
    # if it were split-adjusted just because the caller left the default (finding M3).
    if "price_basis" in b.columns and len(b):
        stamped = set(b["price_basis"].dropna().unique())
        if stamped and stamped != {SPLIT_ADJUSTED}:
            raise NotImplementedError(f"unsupported price_basis in data: {sorted(stamped)}")
    if price_basis == RAW:
        raise NotImplementedError(
            "no raw-price provider is configured; add Alpaca (adjustment='raw') or Tiingo")
    f = (dividend_factors(b, actions, as_of) if include_dividends
         else pd.Series(1.0, index=b["date"].values))
    fv = f.reindex(b["date"].values).to_numpy()
    for c in ("open", "high", "low", "close"):
        b[f"adj_{c}"] = b[c].to_numpy() * fv
    b["adj_volume"] = b["volume"].to_numpy()      # already split-consistent
    b["adj_factor"] = fv
    return b


def as_traded(bars: pd.DataFrame, actions: pd.DataFrame,
              as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Prices as actually printed on the day. For share counts and round lots, not returns."""
    b = bars.sort_values("date").reset_index(drop=True).copy()
    u = unadjust_factors(b, actions, as_of).reindex(b["date"].values).to_numpy()
    for c in ("open", "high", "low", "close"):
        b[f"traded_{c}"] = b[c].to_numpy() * u
    b["traded_volume"] = np.where(u > 0, b["volume"].to_numpy() / u, np.nan)
    b["unadjust_factor"] = u
    return b
