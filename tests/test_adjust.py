"""Splits are ALREADY in the stored series. This suite exists to stop anyone
re-introducing the double-adjustment bug found on 2026-09-14. See adjust.py."""
import datetime as dt
import pandas as pd
from rltrader.data.adjust import adjusted, as_traded, dividend_factors, unadjust_factors

EX = dt.date(2020, 6, 15)


def _bars():
    """A 2:1 split as the PROVIDER delivers it: the whole history already halved,
    so the series is continuous and shows no jump at the ex-date."""
    days = [dt.date(2020, 6, 12), dt.date(2020, 6, 13), EX, dt.date(2020, 6, 16)]
    return pd.DataFrame({"date": days, "open": [50.0] * 4, "high": [50.0] * 4,
                         "low": [50.0] * 4, "close": [50.0] * 4, "volume": [1e6] * 4})


def _split(public_at="2020-06-15"):
    return pd.DataFrame([{"ex_date": EX, "kind": "split", "value": 2.0,
                          "first_public_at": pd.Timestamp(public_at, tz="UTC")}])


def _div(public_at="2020-06-15", value=1.0):
    return pd.DataFrame([{"ex_date": EX, "kind": "dividend", "value": value,
                          "first_public_at": pd.Timestamp(public_at, tz="UTC")}])


NOW = pd.Timestamp("2026-01-01", tz="UTC")


def test_splits_are_not_reapplied():
    """The regression test for the original bug: adjusting must not touch a split."""
    adj = adjusted(_bars(), _split(), as_of=NOW, include_dividends=True)
    r = adj["adj_close"].pct_change().dropna()
    assert r.abs().max() < 1e-12, f"a split was re-applied to an already-adjusted series: {r.tolist()}"


def test_unadjust_recovers_the_printed_price():
    tr = as_traded(_bars(), _split(), as_of=NOW)
    assert tr.loc[0, "traded_close"] == 100.0      # pre-split tape print
    assert tr.loc[2, "traded_close"] == 50.0       # on/after ex-date, no change
    assert tr.loc[0, "traded_volume"] == 5e5


def test_unadjustment_ignores_the_backtest_clock():
    """Un-adjustment reverses what the VENDOR baked in, so it follows the data vintage,
    not the simulated present. Gating it on as_of (the original behaviour) returned
    109.63 for NVDA 2024-05-31 at as_of=2024-06-01 when the tape printed 1096.33 —
    the exact 10x position-sizing error this module exists to prevent. Finding H1.

    This is NOT a look-ahead leak: the split ratio is used only to restate a price that
    was already public, never to reveal a fact about the future. What a backtest may SEE
    is controlled by Store.bars(as_of=...); what a price MEANS is controlled here."""
    early = unadjust_factors(_bars(), _split("2020-06-15"), as_of=pd.Timestamp("2020-06-14", tz="UTC"))
    late = unadjust_factors(_bars(), _split("2020-06-15"), as_of=NOW)
    assert early.equals(late)
    assert early.loc[dt.date(2020, 6, 12)] == 2.0


def test_two_providers_reporting_one_split_do_not_double_count():
    """Finding L1/M3: without dedupe this gives 4.0 and a +90,000 bp price error."""
    both = pd.concat([_split(), _split()], ignore_index=True)
    assert unadjust_factors(_bars(), both).loc[dt.date(2020, 6, 12)] == 2.0


def test_dividend_factor_is_exactly_one_at_the_last_bar():
    """Finding M4: an action dated past the final bar rescaled the whole series."""
    late_div = pd.DataFrame([{"ex_date": dt.date(2020, 7, 1), "kind": "dividend", "value": 1.0,
                              "first_public_at": pd.Timestamp("2020-07-01", tz="UTC")}])
    f = dividend_factors(_bars(), late_div, as_of=NOW)
    assert f.iloc[-1] == 1.0


def test_dividend_adjustment_scales_prior_prices_down():
    f = dividend_factors(_bars(), _div(value=0.5), as_of=NOW)
    assert abs(f.loc[dt.date(2020, 6, 12)] - 0.99) < 1e-12     # 0.5 on a 50.0 close
    assert f.loc[EX] == 1.0


def test_dividend_is_invisible_before_it_is_public():
    f = dividend_factors(_bars(), _div("2020-06-15"), as_of=pd.Timestamp("2020-06-14", tz="UTC"))
    assert (f == 1.0).all()


def test_raw_basis_is_refused_until_a_raw_provider_exists():
    import pytest
    with pytest.raises(NotImplementedError):
        adjusted(_bars(), _split(), as_of=NOW, price_basis="raw")
