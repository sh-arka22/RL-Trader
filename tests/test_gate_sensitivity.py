"""Mutation tests: plant the defect each gate is named after, assert the gate FAILS.

A gate that cannot fail is not a gate. Written against the pre-fix code by an
independent agent, so every test here failed once for the right reason. The xfail
markers it carried were removed when the fixes landed (commit bbdd77b); the
S2_REVIEW finding each test guards is named in a comment on its first line.
"""
from __future__ import annotations

import datetime as dt
import random
import zoneinfo

import pandas as pd
import pytest

from rltrader.data import calendar as cal
from rltrader.data.adjust import as_traded, adjusted
from rltrader.data.build import (PRINTED_CLOSES, gate_corporate_actions, gate_coverage,
                                 gate_no_phantom_sessions, gate_point_in_time,
                                 gate_printed_prices, gate_reconcile, run_gates)
from rltrader.data.store import Store

NY = zoneinfo.ZoneInfo("America/New_York")
NOW = pd.Timestamp("2026-09-14 23:49", tz="UTC")          # the review's "now"
REC = pd.Timestamp("2026-09-14 23:49", tz="UTC")          # recorded_at for every write

# The three split anchors G7 checks, with the ratio and ex-date behind each printed close.
ANCHORS = {
    "NVDA": (dt.date(2024, 6, 7), 1208.88, dt.date(2024, 6, 10), 10.0),
    "TSLA": (dt.date(2022, 8, 24), 891.29, dt.date(2022, 8, 25), 3.0),
    "AAPL": (dt.date(2020, 8, 28), 499.23, dt.date(2020, 8, 31), 4.0),
}


# --------------------------------------------------------------------------- helpers

def bar_frame(dates, first: float = 100.0, drift: float = 0.1) -> pd.DataFrame:
    """A clean OHLCV frame on real NYSE sessions. Monotone drift, so a one-bar shift
    or a 10x corruption is arithmetically visible rather than hidden by noise."""
    dates = list(dates)
    close = [first + drift * i for i in range(len(dates))]
    return pd.DataFrame({"date": dates,
                         "open": [c * 0.99 for c in close],
                         "high": [c * 1.01 for c in close],
                         "low": [c * 0.98 for c in close],
                         "close": close,
                         "volume": [1e6 + i for i in range(len(dates))]})


def near_copy(df: pd.DataFrame) -> pd.DataFrame:
    """A second provider that rounds differently — divergence well under the 1 bp gate."""
    out = df.copy()
    out["close"] = [round(c * (1 + ((i % 5) - 2) * 1e-8), 6)
                    for i, c in enumerate(out["close"])]
    return out


def two_provider_store(tmp_path, ticker="X", start=dt.date(2024, 1, 2),
                       end=dt.date(2024, 12, 31)) -> tuple[Store, list[dt.date]]:
    s = Store(tmp_path)
    sess = cal.sessions(start, end)
    a = bar_frame(sess)
    s.write_bars(a, ticker, "yahoo", "i1", recorded_at=REC)
    s.write_bars(near_copy(a), ticker, "stockanalysis", "i1", recorded_at=REC)
    return s, sess


def anchor_store(tmp_path, tickers=("NVDA", "TSLA", "AAPL"),
                 ex_date_override: dict | None = None) -> Store:
    """A store that reproduces every G7 anchor exactly: stored close = printed / ratio,
    plus the split that explains the difference. Un-adjustment recovers the tape to ~0 bp."""
    s = Store(tmp_path)
    for t in tickers:
        anchor_d, printed, ex, ratio = ANCHORS[t]
        sess = cal.sessions(ex - dt.timedelta(days=20), ex + dt.timedelta(days=20))
        stored = printed / ratio
        b = pd.DataFrame({"date": sess, "open": [stored] * len(sess),
                          "high": [stored] * len(sess), "low": [stored] * len(sess),
                          "close": [stored] * len(sess), "volume": [1e6] * len(sess)})
        s.write_bars(b, t, "yahoo", "i1", recorded_at=REC)
        ex_written = (ex_date_override or {}).get(t, ex)
        acts = pd.DataFrame([{"ex_date": ex_written, "kind": "split", "value": ratio}])
        s.write_actions(acts, t, "yahoo", "i1", recorded_at=REC)
    return s


def rewrite_parts(store: Store, ticker: str, fn, kind: str = "bars") -> int:
    """Plant a defect directly in the stored parts — the only way to forge a column that
    ``write_bars`` computes for itself (``first_public_at``)."""
    n = 0
    for p in sorted((store.root / kind).glob(f"ticker={ticker}/**/*.parquet")):
        df = pd.read_parquet(p)
        out = fn(df)
        if out is not None:
            out.to_parquet(p, index=False)
            n += 1
    return n


def named(gates, name):
    return next((g for g in gates if g.name == name), None)


# ------------------------------------------------------------------ G1 coverage

def test_g1_fails_when_every_series_is_truncated():
    # Guards S2_REVIEW H2: a stale/truncated feed must not read as full coverage.
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    s = Store(tmp)
    start, end = dt.date(2019, 1, 2), dt.date(2021, 12, 31)
    cut = dt.date(2020, 1, 2)
    kept = [d for d in cal.sessions(start, end) if d <= cut]
    missing_after_cut = len([d for d in cal.sessions(start, end) if d > cut])
    s.write_bars(bar_frame(kept), "X", "yahoo", "i1", recorded_at=REC)

    assert missing_after_cut > 400, "the mutation must remove a material slice of history"
    g = gate_coverage(s, ["X"], start, end)
    assert g.passed is False, (
        f"{missing_after_cut} sessions after {cut} are absent and G1 still says: {g.detail}")


def test_g1_fails_when_five_interior_sessions_are_deleted(tmp_path):
    # Guards S2_REVIEW H2 (control side): interior gaps ARE caught — the hole is the tail.
    start, end = dt.date(2024, 1, 2), dt.date(2024, 12, 31)
    sess = cal.sessions(start, end)
    dropped = sorted(random.Random(0).sample(sess[5:-5], 5))
    kept = [d for d in sess if d not in set(dropped)]
    s = Store(tmp_path)
    s.write_bars(bar_frame(kept), "X", "yahoo", "i1", recorded_at=REC)

    g = gate_coverage(s, ["X"], start, end)
    assert g.passed is False, g.detail
    assert g.data["bad"]["X/yahoo"] == 5, g.data


# ------------------------------------------------------- G2 phantom sessions

def test_g2_fails_on_a_bar_stamped_on_independence_day(tmp_path):
    # Guards build.py:78 (G2): a bar on a day the exchange was shut is a provider defect.
    start, end = dt.date(2024, 1, 2), dt.date(2024, 12, 31)
    holiday = dt.date(2024, 7, 4)
    sess = cal.sessions(start, end)
    assert holiday not in sess, "2024-07-04 must not be an NYSE session"
    s = Store(tmp_path)
    s.write_bars(bar_frame(sorted(sess + [holiday])), "X", "yahoo", "i1", recorded_at=REC)

    g = gate_no_phantom_sessions(s, ["X"], start, end)
    assert g.passed is False, g.detail
    assert str(holiday) in g.data["bad"]["X/yahoo"], g.data


# ------------------------------------------------------------- G3 reconcile

def test_g3_fails_when_one_bar_of_the_second_provider_is_10x(tmp_path):
    # Guards S2_REVIEW C2: five bad rows in 14,705 good ones is the bug that started this.
    s, sess = two_provider_store(tmp_path)
    bad_day = sess[len(sess) // 2]

    def corrupt(df):
        if df["provider"].iloc[0] != "stockanalysis":
            return None
        m = df["date"] == bad_day
        if not m.any():
            return None
        df.loc[m, "close"] = df.loc[m, "close"] * 10.0
        return df

    assert rewrite_parts(s, "X", corrupt) == 1

    g = gate_reconcile(s, ["X"], "yahoo", "stockanalysis")
    row = g.data["rows"][0]
    assert row["max_bps"] > 1000 and row["n_over_1bp"] == 1, row   # the evidence exists
    assert g.passed is False, (
        f"a {row['max_bps']:.0f} bp break passed because the median was "
        f"{row['median_bps']:.4f} bp: {g.detail}")


def test_g3_fails_when_the_second_provider_is_a_bit_identical_mirror(tmp_path):
    # Guards reconcile.py:6-8: two sources that agree to the bit are one source.
    s = Store(tmp_path)
    sess = cal.sessions(dt.date(2024, 1, 2), dt.date(2024, 12, 31))
    a = bar_frame(sess)
    s.write_bars(a, "X", "yahoo", "i1", recorded_at=REC)
    s.write_bars(a.copy(), "X", "stockanalysis", "i1", recorded_at=REC)   # bit-identical

    g = gate_reconcile(s, ["X"], "yahoo", "stockanalysis")
    assert g.data["independence"]["verdict"] == "MIRROR_SUSPECTED", g.data["independence"]
    assert g.data["rows"][0]["frac_bit_identical"] == 1.0
    assert g.passed is False, (
        "the reconciliation proves nothing when one source mirrors the other: " + g.detail)


# ---------------------------------------------------------- G4 point-in-time

def test_g4_fails_on_a_bar_public_at_its_own_0929(tmp_path):
    # Guards S2_REVIEW H3: a bar knowable before its own close is look-ahead, always.
    start, end = dt.date(2019, 1, 2), dt.date(2019, 12, 31)
    sess = cal.sessions(start, end)
    leak_day = dt.date(2019, 6, 3)
    s = Store(tmp_path)
    s.write_bars(bar_frame(sess), "X", "yahoo", "i1", recorded_at=REC)

    early = pd.Timestamp(dt.datetime.combine(leak_day, dt.time(9, 29), tzinfo=NY)).tz_convert("UTC")

    def mis_stamp(df):
        m = df["date"] == leak_day
        if not m.any():
            return None
        df.loc[m, "first_public_at"] = early
        return df

    assert rewrite_parts(s, "X", mis_stamp) == 1
    # Control: the defect is real and the gate DOES see it when it happens to look there.
    assert gate_point_in_time(s, ["X"], [leak_day]).passed is False

    g = named(run_gates(s, ["X"], start, end), "G4_point_in_time")
    assert g.passed is False, (
        f"{leak_day} is not one of the sampled probe dates, so the leak is invisible: {g.detail}")


def test_g4_cannot_catch_content_look_ahead_and_that_is_the_honest_answer(tmp_path):
    # Guards S2_REVIEW H3: G4 re-derives bar_public_at() arithmetic; it never reads content.
    s, sess = two_provider_store(tmp_path)
    i = len(sess) // 2
    day, nxt = sess[i], sess[i + 1]

    def paste_next_session(df):
        if df["provider"].iloc[0] != "yahoo":
            return None
        if not (df["date"] == day).any() or not (df["date"] == nxt).any():
            return None
        src = df.loc[df["date"] == nxt, ["open", "high", "low", "close"]].iloc[0]
        for c in ("open", "high", "low", "close"):
            df.loc[df["date"] == day, c] = float(src[c])
        return df

    assert rewrite_parts(s, "X", paste_next_session) == 1

    # The honest answer: NO. Maximal look-ahead in the content, every timestamp correct,
    # so G4 is green by construction — at stride 1 and at any stride.
    g4 = gate_point_in_time(s, ["X"], sess)
    assert g4.passed is True, "G4 unexpectedly grew a content check — update this test"

    # The defect is only visible cross-provider, in the statistic C2 says G3 must be
    # scored on. That is where a fix belongs; it can never live in G4.
    g3 = gate_reconcile(s, ["X"], "yahoo", "stockanalysis")
    row = g3.data["rows"][0]
    assert row["n_over_1bp"] >= 1 and row["max_bps"] > 1, row


# ------------------------------------------------------ G5 corporate actions

def _with_dividend(tmp_path):
    s = Store(tmp_path)
    sess = cal.sessions(dt.date(2024, 1, 2), dt.date(2024, 12, 31))
    s.write_bars(bar_frame(sess), "NVDA", "yahoo", "i1", recorded_at=REC)
    ex = sess[len(sess) // 2]
    s.write_actions(pd.DataFrame([{"ex_date": ex, "kind": "dividend", "value": 0.10}]),
                    "NVDA", "yahoo", "i1", recorded_at=REC)
    adj = adjusted(s.bars("NVDA", provider="yahoo"), s.actions("NVDA"), as_of=NOW)
    ref = adj[["date", "adj_close"]].rename(columns={"adj_close": "close"}).copy()
    return s, sess, ref


def test_g5_fails_when_one_reference_bar_is_10x(tmp_path):
    # Guards S2_REVIEW C2: a 90,000 bp break on one date must not pass as "0.0004 bp".
    s, sess, ref = _with_dividend(tmp_path)
    k = len(ref) // 3
    ref.loc[k, "close"] = float(ref.loc[k, "close"]) * 10.0

    g = gate_corporate_actions(s, ["NVDA"], {"NVDA": ref})
    row = g.data["rows"][0]
    assert row["max_bps"] > 1000 and row["n_over_1bp"] >= 1, row      # the evidence exists
    assert g.passed is False, (
        f"a {row['max_bps']:.0f} bp divergence passed on a median of "
        f"{row['median_bps']:.4f} bp: {g.detail}")


def test_g5_never_disappears_when_the_reference_is_unavailable(tmp_path):
    # Guards S2_REVIEW C3: a gate that is absent must never read as a gate that passed.
    s, sess, _ = _with_dividend(tmp_path)
    gates = run_gates(s, ["NVDA"], sess[0], sess[-1], ref_adj={})
    names = [g.name for g in gates]
    failed = [g.name for g in gates if not g.passed]

    assert "G5_corporate_actions" in names, (
        f"G5 vanished on an unavailable reference; 'failed' cannot see it: {names} / {failed}")
    assert named(gates, "G5_corporate_actions").passed is False


# --------------------------------------------------------- G7 printed prices

def test_g7_fails_when_the_nvda_anchor_is_missing(tmp_path):
    # Guards S2_REVIEW M1: the 10:1 anchor that exposed the original bug must not opt out.
    s = anchor_store(tmp_path, tickers=("TSLA", "AAPL"))          # NVDA removed entirely
    g = gate_printed_prices(s)
    assert len(g.data["rows"]) < len(PRINTED_CLOSES)              # the anchor silently left
    assert g.passed is False, (f"only {len(g.data['rows'])} of {len(PRINTED_CLOSES)} "
                               f"anchors checked: {g.detail}")


def test_g7_fails_on_an_ex_date_off_by_one(tmp_path):
    # Guards S2_REVIEW M2: '<' vs '<=' at the ex-date is a 10x error on every ex-date.
    # An action dated one session late makes the shipped '<' behave exactly like '<='.
    ex = ANCHORS["NVDA"][2]
    later = cal.sessions(ex, ex + dt.timedelta(days=10))[1]       # 2024-06-11
    s = anchor_store(tmp_path, ex_date_override={"NVDA": later})

    tr = as_traded(s.bars("NVDA", provider="yahoo"), s.actions("NVDA"), as_of=NOW)
    on_ex = float(tr.loc[tr["date"] == ex, "traded_close"].iloc[0])
    stored = float(tr.loc[tr["date"] == ex, "close"].iloc[0])
    assert on_ex == pytest.approx(stored * 10.0)   # the ex-date bar is already post-split

    g = gate_printed_prices(s)
    assert g.passed is False, (
        f"the ex-date bar is un-adjusted by 10x ({stored:.2f} -> {on_ex:.2f}) and the "
        f"ground-truth gate still says: {g.detail}")


# ------------------------------------------- as_traded / store-level invariants

def test_as_traded_returns_the_tape_price_at_a_point_in_time_as_of(tmp_path):
    # Guards S2_REVIEW H1: recovering the price printed in the past is not look-ahead.
    s = anchor_store(tmp_path, tickers=("NVDA",))
    anchor_d, printed, ex, ratio = ANCHORS["NVDA"]
    pit = pd.Timestamp("2024-06-01", tz="UTC")        # before the split's own ex-date

    tr = as_traded(s.bars("NVDA", provider="yahoo"), s.actions("NVDA"), as_of=pit)
    got = float(tr.loc[tr["date"] == anchor_d, "traded_close"].iloc[0])
    assert got == pytest.approx(printed, rel=1e-9), (
        f"at as_of={pit.date()} as_traded returned {got:.2f}; the tape printed "
        f"{printed:.2f} — a backtest sizing on this buys {ratio:.0f}x too many shares")


def test_a_split_reported_by_two_providers_is_not_double_counted(tmp_path):
    # Guards S2_REVIEW L1: the second action source is the documented upgrade path.
    s = anchor_store(tmp_path, tickers=("NVDA",))
    anchor_d, printed, ex, ratio = ANCHORS["NVDA"]
    s.write_actions(pd.DataFrame([{"ex_date": ex, "kind": "split", "value": ratio}]),
                    "NVDA", "stockanalysis", "i2", recorded_at=REC)   # same split, 2nd source

    acts = s.actions("NVDA")
    assert len(acts) == 2, acts          # one row per provider, by design (store.py:148)

    tr = as_traded(s.bars("NVDA", provider="yahoo"), acts, as_of=NOW)
    got = float(tr.loc[tr["date"] == anchor_d, "traded_close"].iloc[0])
    assert got == pytest.approx(printed, rel=1e-9), (
        f"the same {ratio:.0f}:1 split reported twice un-adjusted to {got:.2f} instead of "
        f"{printed:.2f} — the ratio was applied {ratio ** 0:.0f}+1 times")
