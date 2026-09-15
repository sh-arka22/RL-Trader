#!/usr/bin/env python3
"""S4: train PPO in TradingEnv, evaluate out-of-sample through the S3 harness.

    .venv-core/bin/python scripts/train_agent.py --cost-level realistic --seeds 0 1 2 3 4 \
        --timesteps 100000

COST IS TRAINED-IN (ARCHITECTURE.md 3.6): one process = one cost level. Do not average
models across --cost-level runs; retrain per level.

Train/test split matches docs/reports/S3_FINRL.md exactly (2015-01-02..2021-12-31 /
2022-01-03..2026-09-11), so the PPO-vs-FinRL-PPO-vs-baselines comparison is apples-to-apples.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rltrader.data.store import Store                                     # noqa: E402
from rltrader.eval import costs as C                                      # noqa: E402
from rltrader.eval.agent import PPOStrategy                                # noqa: E402
from rltrader.eval.baselines import total_return_panel                    # noqa: E402
from rltrader.eval.harness import evaluate, liquidity_inputs              # noqa: E402
from rltrader.eval.trial_log import TrialLog                              # noqa: E402
from rltrader.envs.trading_env import TradingEnv                          # noqa: E402

FIVE = ["NVDA", "TSLA", "AAPL", "META", "XOM"]
FULL_START, FULL_END = dt.date(2015, 1, 1), dt.date(2026, 9, 11)
TRAIN_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-03")
COST_BY_NAME = {m.name: m for m in C.LEVELS}


def build_panels(store: Store):
    close = total_return_panel(store, FIVE, FULL_START, FULL_END)
    op = {}
    for t in FIVE:
        tp = store.tape(t).set_index("date")
        tp.index = pd.DatetimeIndex(tp.index)
        tp = tp[(tp.index >= close.index[0]) & (tp.index <= close.index[-1])]
        op[t] = tp["open"] * tp["adj_factor"]
    opn = pd.DataFrame(op).sort_index()
    adv, vol = liquidity_inputs(store, FIVE, close.index)
    return close, opn, adv, vol


def split(df: pd.DataFrame):
    return df[df.index <= TRAIN_END], df[df.index >= TEST_START]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-level", choices=list(COST_BY_NAME), default="realistic")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--l1-penalty", type=float, default=0.0)
    ap.add_argument("--turbulence-threshold", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from stable_baselines3 import PPO
    from rltrader.eval.metrics import summary

    cost_model = COST_BY_NAME[a.cost_level]
    store = Store("data")
    close, opn, adv, vol = build_panels(store)
    close_tr, close_te = split(close)
    opn_tr, opn_te = split(opn)
    adv_tr, adv_te = split(adv)
    vol_tr, vol_te = split(vol)
    print(f"cost_level={a.cost_level} {cost_model} train={len(close_tr)} test={len(close_te)}")

    log = TrialLog("data/trials.jsonl")
    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)
    out_records = []

    for seed in a.seeds:
        t0 = time.time()
        train_env = TradingEnv(close_tr, opn_tr, adv_tr, vol_tr, cost_model,
                               lookback=a.lookback, l1_penalty=a.l1_penalty,
                               turbulence_threshold=a.turbulence_threshold)
        train_env.reset(seed=seed)
        model = PPO("MlpPolicy", train_env, seed=seed, verbose=0,
                    n_steps=2048, batch_size=64, gamma=0.99, learning_rate=3e-4)
        model.learn(total_timesteps=a.timesteps)
        train_secs = time.time() - t0

        tag = f"{a.cost_level}_seed{seed}_{a.timesteps}"
        model_path = models_dir / f"ppo_{tag}.zip"
        model.save(model_path)

        strat = PPOStrategy(model, lookback=a.lookback, tag=tag)
        adv_te2, vol_te2 = liquidity_inputs(store, FIVE, close_te.index)  # rolling stats reset at test start
        r = evaluate(strat, close_te, opn_te, adv_te2, vol_te2, log=log,
                    levels=[cost_model], warmup=a.lookback,
                    note=f"S4 PPO seed={seed} cost={a.cost_level} timesteps={a.timesteps}")
        m = r["by_cost"][cost_model.name]
        print(f"seed={seed} train_secs={train_secs:.0f} OOS[{a.cost_level}] "
              f"sharpe={m['sharpe']:.3f} cagr={m['cagr']:.2%} maxdd={m['max_drawdown']:.2%} "
              f"turnover={m['turnover']:.4f}")
        out_records.append({"seed": seed, "cost_level": a.cost_level, "timesteps": a.timesteps,
                            "train_secs": train_secs, "model_path": str(model_path),
                            "metrics": m, "trial_hash": r.get("trial_hash")})

    out_path = a.out or f"data/ppo_{a.cost_level}_{a.timesteps}.json"
    pathlib.Path(out_path).write_text(json.dumps(out_records, indent=2, default=str))
    sharpes = [r["metrics"]["sharpe"] for r in out_records]
    print(f"\n{a.cost_level} @ {a.timesteps} steps, {len(a.seeds)} seeds: "
          f"Sharpe {np.mean(sharpes):.3f} +/- {np.std(sharpes):.3f} "
          f"(range {min(sharpes):.3f}-{max(sharpes):.3f})")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
