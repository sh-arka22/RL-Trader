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
    bad, detail = {}, []
    for t in tickers:
        b = store.bars(t)
        if b.empty:
            bad[t] = "no data"
            continue
        for prov, g in b.groupby("provider"):
            miss = cal.missing_sessions(list(g["date"]), start, min(end, max(g["date"])))
            if miss:
                bad[f"{t}/{prov}"] = len(miss)
                detail.append(f"{t}/{prov}: {len(miss)} missing, first {miss[0]}, last {miss[-1]}")
    return GateResult("G1_coverage", not bad,
                      "; ".join(detail) or "0 missing sessions for every ticker/provider", {"bad": bad})


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
    usable = [r for r in rows if r.get("n")]
    worst = max((r["median_bps"] for r in usable), default=float("inf"))
    ind = rec.independence_evidence(usable)
    ok = bool(usable) and worst <= tol_bps
    return GateResult("G3_reconcile", ok,
                      f"worst median divergence {worst:.4f} bp (tol {tol_bps}); "
                      f"independence: {ind['verdict']} — {ind['detail']}",
                      {"rows": rows, "independence": ind})


def gate_point_in_time(store: Store, tickers, probe_dates: list[dt.date]) -> GateResult:
    """Standing on date D at 09:29 ET, no bar for D (or later) may be visible."""
    import zoneinfo
    ny = zoneinfo.ZoneInfo("America/New_York")
    leaks = []
    for t in tickers:
        for d in probe_dates:
            as_of = pd.Timestamp(dt.datetime.combine(d, dt.time(9, 29), tzinfo=ny)).tz_convert("UTC")
            visible = store.bars(t, as_of=as_of)
            if visible.empty:
                continue
            future = visible[visible["date"] >= d]
            if len(future):
                leaks.append(f"{t} @ {d}: {len(future)} future bars visible "
                             f"(max date {future['date'].max()})")
    return GateResult("G4_point_in_time", not leaks,
                      "no future bar visible at any probe instant" if not leaks
                      else "; ".join(leaks[:5]), {"leaks": leaks})


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
    usable = [r for r in rows if r.get("n")]
    worst = max((r["median_bps"] for r in usable), default=float("inf"))
    return GateResult("G5_corporate_actions", bool(usable) and worst <= tol_bps,
                      f"worst median split-adjusted return divergence {worst:.4f} bp (tol {tol_bps})",
                      {"rows": rows})


# Closes the market actually printed, on the last session before a known split.
# Independent ground truth: if un-adjustment cannot reproduce these, the split handling
# is wrong regardless of what any internal consistency check says.
PRINTED_CLOSES = {
    ("NVDA", dt.date(2024, 6, 7)): 1208.88,   # 10:1 split ex 2024-06-10
    ("TSLA", dt.date(2022, 8, 24)): 891.29,   #  3:1 split ex 2022-08-25
    ("AAPL", dt.date(2020, 8, 28)): 499.23,   #  4:1 split ex 2020-08-31
}


def gate_printed_prices(store: Store, tol_bps: float = 25.0) -> GateResult:
    """Un-adjusting the stored series must recover the price that was really quoted."""
    now = pd.Timestamp.now(tz="UTC")
    rows, bad = [], []
    for (t, d), truth in PRINTED_CLOSES.items():
        bars = store.bars(t, provider="yahoo")
        if bars.empty:
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
    if ref_adj:
        gates.append(gate_corporate_actions(store, tickers, ref_adj))
    gates.append(gate_printed_prices(store))
    return gates
