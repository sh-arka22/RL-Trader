"""S2 orchestration: fetch -> store -> validate -> report.

Gates (PLAN.md S2), all hard failures:
  G1 coverage      0% missing NYSE sessions per ticker
  G2 no phantoms   no bars on non-session days
  G3 reconcile     median |close divergence| <= 1 bp between two providers
  G4 point-in-time as_of(D, 09:29) exposes no row with first_public_at > that instant
  G5 actions       recomputed adjusted closes match a provider's own adjusted series <= 1 bp
  G6 reproducible  two runs produce an identical payload digest
"""
from __future__ import annotations
import dataclasses
import datetime as dt
import pathlib
import pandas as pd

from . import calendar as cal
from . import reconcile as rec
from .adjust import adjusted, as_traded
from .manifest import payload_digest, write_manifest
from .providers import available_providers, get
from .store import Store, new_ingest_id

TICKERS = ("NVDA", "TSLA", "AAPL", "META", "XOM")
START = dt.date(2015, 1, 1)


# Divergence gates are scored on FOUR statistics, not one. The S2 review proved that a
# single 10x-corrupted bar gives median 0.0004 bp and PASSES a median-only gate while
# max_bps reads 93,138 — the very shape of the split bug these gates exist to prevent
# (finding C2). Ceilings are set just above the worst clean value observed on real data
# (G3 max 17.5 bp on META 2016-07-05; G5 max 17.9 bp on META).
TOLERANCE = {"median_bps": 1.0, "p95_bps": 5.0, "max_bps": 25.0, "frac_over_1bp": 0.15}


def _score(rows: list[dict], tol: dict | None = None) -> tuple[bool, str]:
    """Judge a divergence table on every statistic it computed."""
    tol = tol or TOLERANCE
    usable = [r for r in rows if r.get("n")]
    if not usable:
        return False, "no usable comparison rows — the gate could not be evaluated"
    breaches = []
    for r in usable:
        frac = r.get("n_over_1bp", 0) / r["n"]
        for k in ("median_bps", "p95_bps", "max_bps"):
            if r.get(k, 0) > tol[k]:
                breaches.append(f"{r['ticker']} {k}={r[k]:.3f}>{tol[k]}")
        if frac > tol["frac_over_1bp"]:
            breaches.append(f"{r['ticker']} frac_over_1bp={frac:.3f}>{tol['frac_over_1bp']}")
    worst = {k: max(r.get(k, 0) for r in usable) for k in ("median_bps", "p95_bps", "max_bps")}
    detail = (f"n={len(usable)} worst median {worst['median_bps']:.4f} / p95 {worst['p95_bps']:.3f} / "
              f"max {worst['max_bps']:.3f} bp")
    return (not breaches), (detail if not breaches else detail + " — BREACH: " + "; ".join(breaches[:6]))


@dataclasses.dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    data: dict = dataclasses.field(default_factory=dict)


def ingest(store: Store, tickers=TICKERS, start: dt.date = START, end: dt.date | None = None,
           providers: list[str] | None = None, ingest_id: str | None = None,
           recorded_at: pd.Timestamp | None = None, log=print) -> str:
    end = end or dt.date.today()
    ingest_id = ingest_id or new_ingest_id()
    names = providers or [p.info.name for p in available_providers()]
    log(f"ingest {ingest_id}: {len(tickers)} tickers x {len(names)} providers "
        f"({', '.join(names)}) {start} -> {end}")
    for name in names:
        prov = get(name)
        for t in tickers:
            try:
                bars = prov.bars(t, start, end)
                store.write_bars(bars, t, name, ingest_id, recorded_at,
                                 price_basis=prov.info.price_basis)
                acts = prov.actions(t)
                if not acts.empty:
                    acts = acts[(acts["ex_date"] >= start) & (acts["ex_date"] <= end)]
                store.write_actions(acts, t, name, ingest_id, recorded_at)
                log(f"  {name:14s} {t:5s} bars={len(bars):5d} actions={len(acts):4d} "
                    f"{bars['date'].min()} -> {bars['date'].max()}")
            except Exception as e:
                log(f"  {name:14s} {t:5s} FAILED: {type(e).__name__}: {e}")
    return ingest_id


def gate_coverage(store: Store, tickers, start, end) -> GateResult:
    """Full expected window. NOT truncated to the end of the data — see calendar.py."""
    horizon = min(end, cal.last_complete_session())
    expected = len(cal.sessions(start, horizon))
    bad, detail = {}, []
    for t in tickers:
        b = store.bars(t)
        if b.empty:
            bad[t] = "no data"
            detail.append(f"{t}: no data at all")
            continue
        for prov, g in b.groupby("provider"):
            miss = cal.missing_sessions(list(g["date"]), start, horizon)
            if miss:
                bad[f"{t}/{prov}"] = len(miss)
                detail.append(f"{t}/{prov}: {len(miss)}/{expected} missing, "
                              f"first {miss[0]}, last {miss[-1]}")
    return GateResult("G1_coverage", not bad,
                      "; ".join(detail) or
                      f"0 missing of {expected} expected sessions ({start} -> {horizon}) "
                      f"for every ticker/provider", {"bad": bad, "expected_sessions": expected})


def gate_no_phantom_sessions(store: Store, tickers, start, end) -> GateResult:
    bad = {}
    for t in tickers:
        b = store.bars(t)
        if b.empty:
            continue
        for prov, g in b.groupby("provider"):
            extra = cal.unexpected_sessions(list(g["date"]), start, end)
            if extra:
                bad[f"{t}/{prov}"] = [str(d) for d in extra[:5]]
    return GateResult("G2_no_phantom_sessions", not bad,
                      "no bars on non-session days" if not bad else f"phantom bars: {bad}", {"bad": bad})


def gate_reconcile(store: Store, tickers, a: str, b: str, tol_bps: float = 1.0) -> GateResult:
    rows = []
    for t in tickers:
        da, db = store.bars(t, provider=a), store.bars(t, provider=b)
        if da.empty or db.empty:
            rows.append({"ticker": t, "field": "close", "n": 0, "status": "MISSING_PROVIDER"})
            continue
        rows.append(rec.summarise(rec.compare(da, db, "close", a, b), t, "close"))
    ok, detail = _score(rows)
    ind = rec.independence_evidence([r for r in rows if r.get("n")])
    # A source agreeing to the last decimal on every bar is not a second source.
    if ind["verdict"] in ("MIRROR_SUSPECTED", "UNKNOWN"):
        ok = False
    return GateResult("G3_reconcile", ok,
                      f"{detail}; independence: {ind['verdict']} — {ind['detail']}",
                      {"rows": rows, "independence": ind, "tolerance": TOLERANCE})


def gate_point_in_time(store: Store, tickers, probe_dates=None) -> GateResult:
    """Audit the TIMESTAMPS, not the filter.

    The original version asked whether ``bars(as_of=D 09:29)`` hides bars dated >= D.
    That can never fail: ``write_bars`` stamps ``first_public_at = date + 16:00 ET``, so
    the filter it is testing is the filter that produced the data. A store whose every
    bar held the NEXT session's OHLC still passed (finding, review answer 3).

    A leak can only enter through a WRONG STAMP, so the stamps are what get checked —
    every row, not every 250th. ``_check_stamps`` is a pure function on a frame so a
    planted defect can be tested directly.
    """
    import zoneinfo
    ny = zoneinfo.ZoneInfo("America/New_York")
    problems, n_rows = [], 0
    for t in tickers:
        b = store.bars(t)
        if b.empty:
            problems.append(f"{t}: no bars")
            continue
        n_rows += len(b)
        problems += _check_stamps(b, t, ny)
        a = store.actions(t)
        if not a.empty:
            expect = pd.to_datetime(a["ex_date"].astype("string") + " 09:30").dt.tz_localize(
                ny, nonexistent="shift_forward", ambiguous=True).dt.tz_convert("UTC")
            bad = a[pd.to_datetime(a["first_public_at"], utc=True).to_numpy() != expect.to_numpy()]
            for _, r in bad.head(3).iterrows():
                problems.append(f"{t} action {r['ex_date']} {r['kind']}: stamp "
                                f"{r['first_public_at']} != ex-date 09:30 ET")
    return GateResult("G4_point_in_time", not problems,
                      (f"{n_rows} bar stamps and every action stamp verified against the "
                       f"exchange calendar; none is knowable before its own session close")
                      if not problems else "; ".join(problems[:6]),
                      {"problems": problems, "n_rows_checked": n_rows})


def _check_stamps(b: pd.DataFrame, ticker: str, ny) -> list[str]:
    """Every bar must become public exactly at its own session close, and never earlier."""
    out = []
    fpa = pd.to_datetime(b["first_public_at"], utc=True)
    expect = pd.to_datetime(b["date"].astype("string") + " 16:00").dt.tz_localize(
        ny, nonexistent="shift_forward", ambiguous=True).dt.tz_convert("UTC")
    wrong = b[fpa.to_numpy() != expect.to_numpy()]
    for _, r in wrong.head(3).iterrows():
        out.append(f"{ticker} {r['date']}: first_public_at {r['first_public_at']} "
                   f"!= session close 16:00 ET")
    # independent of the equality test: nothing may be public before its own 09:29 probe
    open_probe = pd.to_datetime(b["date"].astype("string") + " 09:29").dt.tz_localize(
        ny, nonexistent="shift_forward", ambiguous=True).dt.tz_convert("UTC")
    early = b[fpa.to_numpy() <= open_probe.to_numpy()]
    for _, r in early.head(3).iterrows():
        out.append(f"{ticker} {r['date']}: bar is public at or before its own 09:29 ET open")
    return out


def gate_corporate_actions(store: Store, tickers, ref_adj: dict[str, pd.DataFrame],
                           tol_bps: float = 1.0) -> GateResult:
    """Recomputed adjusted close vs an independent provider's own adjusted series."""
    rows = []
    now = pd.Timestamp.now(tz="UTC")
    for t in tickers:
        bars = store.bars(t, provider="yahoo")
        acts = store.actions(t)
        if bars.empty or t not in ref_adj or ref_adj[t].empty:
            rows.append({"ticker": t, "n": 0, "status": "NO_REF"})
            continue
        # dividends only — the stored series is already split-adjusted (see adjust.py)
        adj = adjusted(bars, acts, as_of=now, include_dividends=True)
        ours = adj[["date", "adj_close"]].rename(columns={"adj_close": "close"})
        m = rec.compare(ours, ref_adj[t], "close", "ours", "ref")
        # compare RETURNS, not levels: providers normalise the series differently
        if len(m) > 2:
            m = m.sort_values("date")
            r1 = pd.Series(m["close_ours"].to_numpy()).pct_change()
            r2 = pd.Series(m["close_ref"].to_numpy()).pct_change()
            d_bps = ((r1 - r2).abs() * 1e4).dropna()
            rows.append({"ticker": t, "n": int(len(d_bps)),
                         "median_bps": float(d_bps.median()),
                         "p95_bps": float(d_bps.quantile(0.95)),
                         "max_bps": float(d_bps.max()),
                         "n_over_1bp": int((d_bps > 1).sum())})
        else:
            rows.append({"ticker": t, "n": 0, "status": "NO_OVERLAP"})
    ok, detail = _score(rows)
    missing = [r["ticker"] for r in rows if not r.get("n")]
    if missing:      # a rate-limited reference must FAIL the gate, never shrink it (C3/M1)
        ok = False
        detail += f" — no reference for {missing}"
    return GateResult("G5_corporate_actions", ok,
                      f"recomputed total return vs vendor adjusted close: {detail}",
                      {"rows": rows, "tolerance": TOLERANCE})


# Closes the market actually printed, on the last session before a known split.
# Independent ground truth: if un-adjustment cannot reproduce these, the split handling
# is wrong regardless of what any internal consistency check says.
PRINTED_CLOSES = {
    # last session BEFORE the split — catches a missing un-adjustment
    ("NVDA", dt.date(2024, 6, 7)): 1208.88,   # 10:1 split ex 2024-06-10
    ("TSLA", dt.date(2022, 8, 24)): 891.29,   #  3:1 split ex 2022-08-25
    ("AAPL", dt.date(2020, 8, 28)): 499.23,   #  4:1 split ex 2020-08-31
    # first session ON/AFTER the split — catches an off-by-one at the ex-date boundary.
    # Without these, flipping `<` to `<=` in adjust.unadjust_factors leaves G7 green
    # while every post-split price is wrong by the split ratio (finding M2).
    ("NVDA", dt.date(2024, 6, 10)): 121.79,
    ("TSLA", dt.date(2022, 8, 25)): 296.07,
    ("AAPL", dt.date(2020, 8, 31)): 129.04,
}


def gate_printed_prices(store: Store, tol_bps: float = 25.0) -> GateResult:
    """Un-adjusting the stored series must recover the price that was really quoted."""
    now = pd.Timestamp.now(tz="UTC")
    rows, bad = [], []
    for (t, d), truth in PRINTED_CLOSES.items():
        bars = store.bars(t, provider="yahoo")
        if bars.empty:
            # a vanished ticker must fail the gate, not quietly shrink it (finding M1)
            bad.append(f"{t}: no bars — anchor {d} could not be checked")
            continue
        tr = as_traded(bars, store.actions(t), as_of=now)
        hit = tr[tr["date"] == d]
        if hit.empty:
            bad.append(f"{t} {d}: no bar")
            continue
        got = float(hit["traded_close"].iloc[0])
        bps = abs(got - truth) / truth * 1e4
        rows.append({"ticker": t, "date": str(d), "recovered": got, "printed": truth,
                     "diff_bps": bps, "stored": float(hit["close"].iloc[0])})
        if bps > tol_bps:
            bad.append(f"{t} {d}: recovered {got:.2f} vs printed {truth:.2f} ({bps:.0f} bp)")
    return GateResult("G7_printed_prices", bool(rows) and not bad,
                      (f"un-adjusted closes match the tape on {len(rows)} split anchors"
                       if not bad else "; ".join(bad)), {"rows": rows})


def run_gates(store: Store, tickers=TICKERS, start: dt.date = START,
              end: dt.date | None = None, ref_adj: dict | None = None) -> list[GateResult]:
    end = end or dt.date.today()
    probes = [d for d in cal.sessions(dt.date(2018, 1, 1), min(end, dt.date(2026, 1, 1)))][::250]
    provs = sorted({p for t in tickers for p in store.bars(t)["provider"].unique()}) if tickers else []
    gates = [gate_coverage(store, tickers, start, end),
             gate_no_phantom_sessions(store, tickers, start, end)]
    if len(provs) >= 2:
        gates.append(gate_reconcile(store, tickers, provs[0], provs[1]))
    else:
        gates.append(GateResult("G3_reconcile", False,
                                f"only {len(provs)} provider(s) present: {provs}", {}))
    gates.append(gate_point_in_time(store, tickers, probes))
    # G5 is appended unconditionally. Making it conditional on ref_adj meant a rate-limit
    # (documented as routine in stockanalysis.py) silently deleted a hard gate and the run
    # still printed "all gates passed" with exit 0 — finding C3.
    gates.append(gate_corporate_actions(store, tickers, ref_adj or {}))
    gates.append(gate_printed_prices(store))
    return gates
