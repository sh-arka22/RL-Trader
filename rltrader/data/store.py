"""Point-in-time Parquet store with DuckDB views.

Layout::

    data/bars/ticker=<T>/year=<Y>/part-<ingest_id>.parquet
    data/actions/ticker=<T>/part-<ingest_id>.parquet

Append-only. A re-ingest writes a NEW part with a new ``recorded_at``; nothing is ever
overwritten in place.

WHAT ``as_of`` DOES AND DOES NOT GIVE YOU
-----------------------------------------
``as_of`` filters ``first_public_at`` only. It answers "which bars existed by then".
It does NOT rewind the price basis, and it cannot: this store holds a single vintage,
recorded 2026-09-14, and free providers do not serve historical vintages.

So ``bars('NVDA', as_of='2016-01-01')`` returns rows recorded ten years after that
instant, carrying closes that already embed the 2021 4:1 and 2024 10:1 splits. The
2015-01-02 close reads 0.50325; the tape printed 20.13. Returns survive that uniform
rescale, which is why every returns-based gate stayed green while the levels were 40x
wrong — finding C1 of the S2 review.

The rule:
  * returns, features, reconciliation  -> ``bars()`` is correct.
  * share counts, round lots, notional, anything quoted in dollars per share
                                       -> use ``tape()``, never ``bars()``.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import pathlib
import uuid
import pandas as pd

from .schema import EXCHANGE_TZ, SESSION_CLOSE_LOCAL, ACTION_PUBLIC_LOCAL


def new_ingest_id() -> str:
    return uuid.uuid4().hex[:12]


def _localise(dates: pd.Series, local_time: str) -> pd.Series:
    ts = pd.to_datetime(dates.astype("string") + " " + local_time)
    return ts.dt.tz_localize(EXCHANGE_TZ, nonexistent="shift_forward",
                             ambiguous=True).dt.tz_convert("UTC")


def bar_public_at(dates: pd.Series) -> pd.Series:
    """A daily bar is knowable at that session's 16:00 ET close."""
    return _localise(dates, SESSION_CLOSE_LOCAL)


def action_public_at(ex_dates: pd.Series) -> pd.Series:
    """Conservative: pretend we learn of an action at the ex-date open, not on announcement."""
    return _localise(ex_dates, ACTION_PUBLIC_LOCAL)


class Store:
    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root)
        (self.root / "bars").mkdir(parents=True, exist_ok=True)
        (self.root / "actions").mkdir(parents=True, exist_ok=True)

    # ---------------- write ----------------

    def write_bars(self, df: pd.DataFrame, ticker: str, provider: str, ingest_id: str,
                   recorded_at: pd.Timestamp | None = None,
                   price_basis: str = "split_adjusted_as_of_recorded_at") -> list[pathlib.Path]:
        if df.empty:
            return []
        recorded_at = recorded_at or pd.Timestamp.now(tz="UTC")
        out = df.copy()
        out["ticker"] = ticker
        out["provider"] = provider
        out["price_basis"] = price_basis
        out["ingest_id"] = ingest_id
        out["first_public_at"] = bar_public_at(out["date"])
        out["recorded_at"] = recorded_at
        out["year"] = pd.to_datetime(out["date"]).dt.year
        paths = []
        for year, chunk in out.groupby("year"):
            d = self.root / "bars" / f"ticker={ticker}" / f"year={int(year)}"
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"part-{provider}-{ingest_id}.parquet"
            chunk.drop(columns=["year"]).to_parquet(p, index=False)
            paths.append(p)
        return paths

    def write_actions(self, df: pd.DataFrame, ticker: str, provider: str, ingest_id: str,
                      recorded_at: pd.Timestamp | None = None) -> pathlib.Path | None:
        recorded_at = recorded_at or pd.Timestamp.now(tz="UTC")
        out = df.copy()
        if out.empty:
            out = pd.DataFrame(columns=["ex_date", "kind", "value"])
        out["ticker"] = ticker
        out["provider"] = provider
        out["ingest_id"] = ingest_id
        out["first_public_at"] = (action_public_at(out["ex_date"]) if len(out)
                                  else pd.Series(dtype="datetime64[ns, UTC]"))
        out["recorded_at"] = recorded_at
        d = self.root / "actions" / f"ticker={ticker}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"part-{provider}-{ingest_id}.parquet"
        out.to_parquet(p, index=False)
        return p

    # ---------------- read ----------------

    def _read(self, kind: str, ticker: str | None = None) -> pd.DataFrame:
        base = self.root / kind
        pat = f"ticker={ticker}/**/*.parquet" if ticker else "**/*.parquet"
        files = sorted(base.glob(pat))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def bars(self, ticker: str | None = None, provider: str | None = None,
             as_of: pd.Timestamp | None = None,
             recorded_before: pd.Timestamp | None = None) -> pd.DataFrame:
        """Two independent time axes — conflating them is a bug, not a simplification.

        ``as_of``          VALID time. What the market knew at that instant. This is the
                           axis a backtest walks. Filters ``first_public_at``.
        ``recorded_before`` TRANSACTION time. What OUR store contained at that instant.
                           This is the axis an audit walks, to reproduce a past run even
                           after the data was revised. Filters ``recorded_at``.
        """
        df = self._read("bars", ticker)
        if df.empty:
            return df
        if provider:
            df = df[df["provider"] == provider]
        if as_of is not None:
            as_of = pd.Timestamp(as_of)
            if as_of.tz is None:
                as_of = as_of.tz_localize("UTC")
            df = df[df["first_public_at"] <= as_of]
        if recorded_before is not None:
            rb = pd.Timestamp(recorded_before)
            if rb.tz is None:
                rb = rb.tz_localize("UTC")
            df = df[df["recorded_at"] <= rb]
        # ingest_id breaks ties on equal recorded_at. Without it the winner is decided by
        # glob order over part-<provider>-<uuid>.parquet, i.e. a coin flip — and --repro
        # pins recorded_at for both runs, manufacturing exactly those ties (finding L4).
        df = (df.sort_values(["ticker", "date", "provider", "recorded_at", "ingest_id"])
                .drop_duplicates(["ticker", "date", "provider"], keep="last"))
        return df.reset_index(drop=True)

    def actions(self, ticker: str | None = None, as_of: pd.Timestamp | None = None,
                recorded_before: pd.Timestamp | None = None,
                provider: str | None = None) -> pd.DataFrame:
        df = self._read("actions", ticker)
        if provider is not None and not df.empty:
            df = df[df["provider"] == provider]
        if df.empty:
            return pd.DataFrame(columns=["ticker", "ex_date", "kind", "value",
                                         "provider", "first_public_at", "recorded_at"])
        if as_of is not None:
            as_of = pd.Timestamp(as_of)
            if as_of.tz is None:
                as_of = as_of.tz_localize("UTC")
            df = df[df["first_public_at"] <= as_of]
        if recorded_before is not None:
            rb = pd.Timestamp(recorded_before)
            if rb.tz is None:
                rb = rb.tz_localize("UTC")
            df = df[df["recorded_at"] <= rb]
        return (df.sort_values(["ticker", "ex_date", "kind", "recorded_at", "ingest_id"])
                  .drop_duplicates(["ticker", "ex_date", "kind", "provider"], keep="last")
                  .reset_index(drop=True))

    def tape(self, ticker: str, as_of: pd.Timestamp | None = None,
             provider: str = "yahoo") -> pd.DataFrame:
        """Bars visible at ``as_of``, with price LEVELS as actually printed.

        This is the accessor a backtest must use for anything denominated in dollars per
        share. ``traded_*`` columns are the tape; ``adj_*`` are total-return adjusted for
        the knowledge state ``as_of``; ``close`` remains the stored split-adjusted series.
        """
        from .adjust import adjusted, as_traded
        b = self.bars(ticker, provider=provider, as_of=as_of)
        if b.empty:
            return b
        acts_all = self.actions(ticker, provider=provider)
        acts_known = self.actions(ticker, as_of=as_of, provider=provider)
        out = as_traded(b, acts_all)
        adj = adjusted(b, acts_known, as_of=as_of or pd.Timestamp.now(tz="UTC"))
        for c in ("adj_open", "adj_high", "adj_low", "adj_close", "adj_factor"):
            out[c] = adj[c].to_numpy()
        return out

    def duckdb(self):
        """A DuckDB connection with bars/actions registered as views over the Parquet tree."""
        import duckdb
        con = duckdb.connect()
        for kind in ("bars", "actions"):
            files = list((self.root / kind).glob("**/*.parquet"))
            if files:
                con.execute(
                    f"CREATE VIEW {kind} AS SELECT * FROM read_parquet('{self.root}/{kind}/**/*.parquet', "
                    f"union_by_name=true)")
        return con


def checksum(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
