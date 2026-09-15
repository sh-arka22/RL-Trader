#!/usr/bin/env python3
"""S3 baseline sweep: every reference strategy, three cost levels, full stats wiring.

    .venv-core/bin/python scripts/run_baselines.py

Writes data/baseline_table.json and appends every run to the hash-chained trial log
(data/trials.jsonl), which is what makes the DSR n_trials honest.
"""
from __future__ import annotations
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rltrader.data.store import Store                                    # noqa: E402
from rltrader.eval import costs as C                                     # noqa: E402
from rltrader.eval import stats                                          # noqa: E402
from rltrader.eval.baselines import total_return_panel                   # noqa: E402
from rltrader.eval.harness import evaluate, liquidity_inputs, format_table, run_strategy  # noqa: E402
from rltrader.eval.strategies import BuyAndHoldFirst, EqualWeight, Momentum  # noqa: E402
from rltrader.eval.trial_log import TrialLog                             # noqa: E402

FIVE = ["NVDA", "TSLA", "AAPL", "META", "XOM"]
START, END = dt.date(2015, 1, 1), dt.date(2026, 9, 11)


def build_panels(store: Store):
    close = total_return_panel(store, FIVE, START, END)
    op = {}
    for t in FIVE:
        tp = store.tape(t).set_index("date")
        tp.index = pd.DatetimeIndex(tp.index)
        tp = tp[(tp.index >= close.index[0]) & (tp.index <= close.index[-1])]
        op[t] = tp["open"] * tp["adj_factor"]
    opn = pd.DataFrame(op).sort_index()
    adv, vol = liquidity_inputs(store, FIVE, close.index)
    return close, opn, adv, vol


def main() -> int:
    store = Store("data")
    close, opn, adv, vol = build_panels(store)
    log = TrialLog("data/trials.jsonl")

    strategies = [EqualWeight(5), BuyAndHoldFirst(5)] + [
        Momentum(5, lookback=lb, top_k=k)
        for lb, k in [(20, 1), (20, 2), (60, 1), (60, 2), (120, 2), (252, 2)]
    ]
    results, raws = [], {}
    for strat in strategies:
        results.append(evaluate(strat, close, opn, adv, vol, log=log, note="S3 baseline sweep"))
        raws[strat.name] = run_strategy(strat, close, opn, adv, vol)

    print(format_table(results))
    ok, msg = log.verify()
    n_trials = log.n_trials("evaluation")
    print(f"\ntrial log: {msg}")

    bh, best_mom_name = raws["buy_and_hold"], max(
        (r for r in results if "momentum" in r["strategy"]),
        key=lambda r: r["by_cost"]["realistic"]["sharpe"])["strategy"]
    mom = raws[best_mom_name]
    n = min(len(bh["gross_returns"]), len(mom["gross_returns"]))
    bh_net = C.apply(bh["gross_returns"], bh["dw"], bh["adv"], bh["vol"], C.REALISTIC)["net_returns"]
    mom_net = C.apply(mom["gross_returns"], mom["dw"], mom["adv"], mom["vol"], C.REALISTIC)["net_returns"]
    dm = stats.diebold_mariano(-bh_net[:n], -mom_net[:n])
    print(f"\nDiebold-Mariano {best_mom_name} vs buy_and_hold (realistic costs): "
          f"stat={dm.statistic:.3f} p={dm.p_value:.4f} better={dm.better}")

    mom_names = [nm for nm in raws if nm.startswith("momentum")]
    min_len = min(len(raws[nm]["gross_returns"]) for nm in mom_names)
    mom_returns = np.array([raws[nm]["gross_returns"][:min_len] for nm in mom_names])
    pbo = stats.probability_of_backtest_overfitting(mom_returns.T, n_splits=8)
    print(f"PBO across {mom_returns.shape[0]} momentum variants: {pbo.pbo:.3f} "
          f"(prob_oos_loss={pbo.prob_oos_loss:.3f}) — variants are highly correlated "
          f"(all momentum), which is exactly the regime CSCV is least informative in; "
          f"read this as a caveat, not a clean pass or fail")

    best = max(results, key=lambda r: r["by_cost"]["realistic"]["sharpe"])
    r = raws[best["strategy"]]["gross_returns"]
    sh = best["by_cost"]["realistic"]["sharpe"]
    dsr = stats.deflated_sharpe_ratio(sh, n_trials=n_trials, skew=float(pd.Series(r).skew()),
                                      kurtosis=float(pd.Series(r).kurtosis()), n_obs=len(r), periods=252)
    print(f"\nBest single trial: {best['strategy']} realistic Sharpe {sh:.3f}, "
          f"DSR (n_trials={n_trials}, honest count, periods=252) = {dsr:.4f}")

    out = {"generated_at": dt.datetime.now(dt.UTC).isoformat(), "window": [str(START), str(END)],
           "n_trials_in_log": n_trials, "by_strategy": {r["strategy"]: r["by_cost"] for r in results},
           "diebold_mariano_best_momentum_vs_bh": {"statistic": dm.statistic, "p_value": dm.p_value,
                                                    "better": dm.better},
           "pbo_momentum_family": {"pbo": pbo.pbo, "prob_oos_loss": pbo.prob_oos_loss,
                                   "n_variants": mom_returns.shape[0]},
           "dsr_best_trial": {"strategy": best["strategy"], "sharpe": sh, "dsr": dsr, "n_trials": n_trials}}
    pathlib.Path("data/baseline_table.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote data/baseline_table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
