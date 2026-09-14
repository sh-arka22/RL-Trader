"""Column contracts for the point-in-time store.

Two immutable fact tables:

``bars``     raw, UNADJUSTED OHLCV exactly as the exchange printed it.
``actions``  corporate actions (splits, dividends) as separate dated facts.

Adjusted prices are never stored. They are RECOMPUTED from these two tables for a
given knowledge timestamp (see ``rltrader.data.adjust``). This is the whole point:
an adjusted close is not a fact about a day, it is a fact about a day *and* about
everything the market learned afterwards. Storing it back-patched is how look-ahead
bias enters a dataset.
"""
from __future__ import annotations

# What the stored prices ACTUALLY are. Verified empirically on 2026-09-14, not assumed:
# both free providers return split-adjusted closes even when asked for unadjusted data.
# See rltrader/data/adjust.py for the evidence and the consequences.
PRICE_BASIS_SPLIT_ADJUSTED = "split_adjusted_as_of_recorded_at"
PRICE_BASIS_RAW = "raw"

BAR_COLUMNS: dict[str, str] = {
    "ticker": "string",
    "price_basis": "string",
    "date": "date32[day][pyarrow]",      # trading session date (exchange local)
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",                  # float: some providers return fractional composite volume
    "provider": "string",
    "first_public_at": "timestamp[us, tz=UTC][pyarrow]",   # when a trader could have known this bar
    "recorded_at": "timestamp[us, tz=UTC][pyarrow]",       # when WE fetched it
    "ingest_id": "string",
}

ACTION_COLUMNS: dict[str, str] = {
    "ticker": "string",
    "ex_date": "date32[day][pyarrow]",
    "kind": "string",                     # "split" | "dividend"
    "value": "float64",                   # split ratio (e.g. 10.0 for 10:1) or cash dividend per share
    "provider": "string",
    "first_public_at": "timestamp[us, tz=UTC][pyarrow]",
    "recorded_at": "timestamp[us, tz=UTC][pyarrow]",
    "ingest_id": "string",
}

PRICE_COLS = ("open", "high", "low", "close")

# A daily bar for session D is knowable at the closing auction print. We use 16:00
# America/New_York. Late consolidated-tape corrections are ignored, which makes this
# marginally optimistic; the 1-bar execution delay in the environment absorbs it.
SESSION_CLOSE_LOCAL = "16:00"
EXCHANGE_TZ = "America/New_York"

# Corporate actions are announced BEFORE the ex-date, but announcement dates are not
# available on free tiers. We therefore set first_public_at to the ex-date open, which
# means the system learns about an action LATER than a real trader would. That is the
# conservative direction: it can never create look-ahead.
ACTION_PUBLIC_LOCAL = "09:30"
