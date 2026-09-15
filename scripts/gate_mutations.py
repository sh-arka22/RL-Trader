#!/usr/bin/env python3
"""Prove every data gate can fail. A gate that cannot fail is not a gate.

Run: .venv-core/bin/python scripts/gate_mutations.py  (requires a built store in data/)
"""

import sys, datetime as dt, tempfile, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from rltrader.data.store import Store
from rltrader.data.build import (gate_coverage, gate_reconcile, gate_corporate_actions,
                                 gate_printed_prices, gate_point_in_time, TICKERS)
import rltrader.data.calendar as cal

s = Store("data")
END = dt.date(2026, 9, 14); START = dt.date(2015, 1, 1)

def mk(mutate):
    td = tempfile.mkdtemp(); t2 = Store(td)
    for t in TICKERS:
        for prov in ("yahoo", "stockanalysis"):
            b = s.bars(t, provider=prov)
            if b.empty: continue
            b = mutate(b.copy(), t, prov)
            if b is None or b.empty: continue
            t2.write_bars(b[["date","open","high","low","close","volume"]], t, prov, "m1",
                          recorded_at=pd.Timestamp("2026-09-14", tz="UTC"))
        a = s.actions(t)
        if not a.empty:
            t2.write_actions(a[["ex_date","kind","value"]], t, "yahoo", "m1",
                             recorded_at=pd.Timestamp("2026-09-14", tz="UTC"))
    return t2

print("MUTATION                                   GATE                      RESULT")
# H2: truncate every series at 2020-01-02
m = mk(lambda b,t,p: b[b["date"] <= dt.date(2020,1,2)])
g = gate_coverage(m, TICKERS, START, END)
print(f"{'truncate all series at 2020-01-02':42s} {'G1_coverage':25s} {'FAIL (good)' if not g.passed else 'PASS (WEAK GATE)'}")

# G1: delete 5 interior sessions
def drop5(b,t,p):
    return b.drop(b.index[[500,501,900,1200,1500]]) if p=="yahoo" else b
g = gate_coverage(mk(drop5), TICKERS, START, END)
print(f"{'delete 5 interior sessions (yahoo)':42s} {'G1_coverage':25s} {'FAIL (good)' if not g.passed else 'PASS (WEAK GATE)'}")

# C2: corrupt ONE bar of provider 2 by 10x
def corrupt(b,t,p):
    if p=="stockanalysis" and t=="NVDA":
        b.loc[b.index[1000], "close"] *= 10
    return b
g = gate_reconcile(mk(corrupt), TICKERS, "yahoo", "stockanalysis")
print(f"{'one NVDA bar of provider 2 x10':42s} {'G3_reconcile':25s} {'FAIL (good)' if not g.passed else 'PASS (WEAK GATE)'}")

# C2b: provider 2 becomes a bit-identical mirror
def mirror(b,t,p):
    return s.bars(t, provider="yahoo").copy() if p=="stockanalysis" else b
g = gate_reconcile(mk(mirror), TICKERS, "yahoo", "stockanalysis")
print(f"{'provider 2 = exact mirror of provider 1':42s} {'G3_reconcile':25s} {'FAIL (good)' if not g.passed else 'PASS (WEAK GATE)'}")

# C3: reference unavailable (rate limit)
g = gate_corporate_actions(s, TICKERS, {})
print(f"{'G5 reference fetch rate-limited -> {}':42s} {'G5_corporate_actions':25s} {'FAIL (good)' if not g.passed else 'PASS (WEAK GATE)'}")

# M1: NVDA vanishes
m = mk(lambda b,t,p: None if t=="NVDA" else b)
g = gate_printed_prices(m)
print(f"{'NVDA removed entirely':42s} {'G7_printed_prices':25s} {'FAIL (good)' if not g.passed else 'PASS (WEAK GATE)'}")

# G4: one mis-stamped bar
td = tempfile.mkdtemp(); m = Store(td)
b = s.bars("AAPL", provider="yahoo")
m.write_bars(b[["date","open","high","low","close","volume"]], "AAPL","yahoo","m1",
             recorded_at=pd.Timestamp("2026-09-14", tz="UTC"))
f = sorted(pathlib.Path(td).glob("bars/**/*.parquet"))[10]
d = pd.read_parquet(f)
d.loc[d.index[0], "first_public_at"] = d.loc[d.index[0], "first_public_at"] - pd.Timedelta(hours=8)
d.to_parquet(f, index=False)
g = gate_point_in_time(m, ["AAPL"])
print(f"{'one bar stamped 8h early (pre-open)':42s} {'G4_point_in_time':25s} {'FAIL (good)' if not g.passed else 'PASS (WEAK GATE)'}")
