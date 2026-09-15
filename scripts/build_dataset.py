#!/usr/bin/env python3
"""S2 entrypoint: build the point-in-time dataset and run every gate.

    python scripts/build_dataset.py --root data --start 2015-01-01
    python scripts/build_dataset.py --check-only          # gates on the existing store
    python scripts/build_dataset.py --repro               # two ingests, compare digests

Exit code is non-zero if any gate fails. No notebook is in this path.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rltrader.data.build import TICKERS, ingest, run_gates          # noqa: E402
from rltrader.data.manifest import payload_digest, write_manifest   # noqa: E402
from rltrader.data.providers import available_providers, get             # noqa: E402
from rltrader.data.store import Store                               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--tickers", nargs="*", default=list(TICKERS))
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--repro", action="store_true", help="ingest twice into temp roots and compare digests")
    a = ap.parse_args()

    start = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today()

    if a.repro:
        import tempfile
        digests = []
        pinned = pd.Timestamp("2000-01-01", tz="UTC")
        for i in range(2):
            with tempfile.TemporaryDirectory() as td:
                s = Store(td)
                ingest(s, a.tickers[:1], start, end, ingest_id="repro", recorded_at=pinned)
                digests.append(payload_digest(td))
        ok = digests[0] == digests[1]
        print(f"G6_reproducible: {'PASS' if ok else 'FAIL'}  {digests[0][:16]} vs {digests[1][:16]}")
        return 0 if ok else 1

    store = Store(a.root)
    print(f"providers available: {[p.info.name for p in available_providers()]}")
    if not a.check_only:
        ingest(store, a.tickers, start, end)

    # G5 reference: the second provider's own adjusted series, fetched through the
    # provider object so the per-ticker cache and backoff apply (no bare requests here).
    ref_adj = {}
    sa = get("stockanalysis")
    for t in a.tickers:
        try:
            ref_adj[t] = sa.adjusted_reference(t, start, end)
        except Exception as e:      # a nice-to-have reference, never a dependency
            print(f"  ref adj {t}: {type(e).__name__}: {e}")

    gates = run_gates(store, a.tickers, start, end, ref_adj=ref_adj)
    print("\n" + "=" * 78)
    for g in gates:
        print(f"{'PASS' if g.passed else 'FAIL'}  {g.name:24s} {g.detail}")
    print("=" * 78)

    # A full build writes the manifest (the receipt). A check-only run writes a separate
    # report, so `make check` does not dirty the tree on every invocation (finding L7).
    out_name = "gate_report.json" if a.check_only else "manifest.json"
    write_manifest(a.root, pathlib.Path(a.root) / out_name,
                   extra={"tickers": a.tickers, "start": str(start), "end": str(end),
                          "payload_digest": payload_digest(a.root),
                          "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail,
                                     "data": g.data} for g in gates]})
    failed = [g.name for g in gates if not g.passed]
    print(("all gates passed" if not failed else f"FAILED: {failed}"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
