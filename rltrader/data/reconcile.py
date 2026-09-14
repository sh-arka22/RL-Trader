"""Two-provider reconciliation. Divergence is measured in basis points.

PLAN.md S2 gate: median absolute close divergence <= 1 bp, else hard fail.

A second purpose, just as important: if two providers agree to the BIT on every bar,
they are not two providers — one is mirroring the other, and the reconciliation proves
nothing. ``independence_evidence`` reports that case explicitly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

BPS = 1e4


def compare(a: pd.DataFrame, b: pd.DataFrame, col: str = "close",
            name_a: str = "a", name_b: str = "b") -> pd.DataFrame:
    left = a[["date", col]].rename(columns={col: f"{col}_{name_a}"})
    right = b[["date", col]].rename(columns={col: f"{col}_{name_b}"})
    m = left.merge(right, on="date", how="inner")
    if m.empty:
        return m
    x, y = m[f"{col}_{name_a}"].to_numpy(float), m[f"{col}_{name_b}"].to_numpy(float)
    denom = np.where(np.abs(y) > 0, np.abs(y), np.nan)
    m["diff_bps"] = (x - y) / denom * BPS
    m["abs_bps"] = m["diff_bps"].abs()
    return m


def summarise(m: pd.DataFrame, ticker: str, col: str) -> dict:
    if m.empty:
        return {"ticker": ticker, "field": col, "n": 0, "status": "NO_OVERLAP"}
    exact = int((m["abs_bps"] == 0).sum())
    return {
        "ticker": ticker, "field": col, "n": int(len(m)),
        "median_bps": float(m["abs_bps"].median()),
        "p95_bps": float(m["abs_bps"].quantile(0.95)),
        "max_bps": float(m["abs_bps"].max()),
        "n_over_1bp": int((m["abs_bps"] > 1).sum()),
        "frac_bit_identical": exact / len(m),
        "worst_date": str(m.loc[m["abs_bps"].idxmax(), "date"]),
    }


def independence_evidence(rows: list[dict]) -> dict:
    """Are the two sources actually independent, or is one a mirror?"""
    fracs = [r["frac_bit_identical"] for r in rows if r.get("n")]
    if not fracs:
        return {"verdict": "UNKNOWN", "detail": "no overlapping rows"}
    mean_identical = float(np.mean(fracs))
    if mean_identical > 0.999:
        v = "MIRROR_SUSPECTED"
        d = (f"{mean_identical:.4%} of bars are bit-identical — the second source is probably "
             "a mirror and the reconciliation gate proves nothing")
    elif mean_identical > 0.9:
        v = "PARTIALLY_DEPENDENT"
        d = f"{mean_identical:.4%} bit-identical — likely a shared upstream vendor"
    else:
        v = "INDEPENDENT_LIKELY"
        d = f"only {mean_identical:.4%} of bars are bit-identical — sources round/clean differently"
    return {"verdict": v, "detail": d, "mean_frac_bit_identical": mean_identical}
