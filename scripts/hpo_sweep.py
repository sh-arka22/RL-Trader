#!/usr/bin/env python3
"""S4/S5 HPO: Optuna sweep over PPO/TD3 hyperparameters.

    .venv-core/bin/python scripts/hpo_sweep.py --algo ppo --cost-level realistic \
        --n-trials 50 --n-jobs 8 --timesteps 50000

Every trial is a full train + out-of-sample evaluation through the S3 harness -- identical
fill/cost path to every other number in this project (rltrader/eval/harness.py). Every
trial is appended to the SAME hash-chained trial log used everywhere else
(data/trials.jsonl), so a Deflated Sharpe Ratio computed after this sweep counts these
trials honestly alongside the S3 baseline sweep and the S4 seed sweep -- there is exactly
one trial ledger for the whole project, not a separate untracked one for HPO.

Search space (ARCHITECTURE.md S4: "Optuna study on the validation fold only"):
  learning_rate   log-uniform  1e-5 .. 1e-3
  n_steps         categorical  {512, 1024, 2048, 4096}          (PPO only)
  batch_size      categorical  {32, 64, 128, 256}
  l1_penalty      log-uniform  1e-5 .. 1e-2  (0 included as a special case via a coin flip)
  gamma           uniform      0.95 .. 0.999

VALIDATION-FOLD DISCIPLINE: this script trains on 2015-01-02..2019-12-31 and scores each
trial on a 2020-01-02..2021-12-31 VALIDATION fold -- never the 2022-2026 test fold every
other report in this project is scored on. That fold is touched exactly once, by a single
final run with the winning hyperparameters, after this sweep is over. Mixing HPO and the
final held-out score on the same window is the single most common way a "held-out" result
quietly stops being held out.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rltrader.data.store import Store                          # noqa: E402
from rltrader.eval import costs as C                            # noqa: E402
from rltrader.eval.agent import PPOStrategy                     # noqa: E402
from rltrader.eval.baselines import total_return_panel          # noqa: E402
from rltrader.eval.harness import evaluate, liquidity_inputs    # noqa: E402
from rltrader.eval.trial_log import TrialLog                    # noqa: E402
from rltrader.envs.trading_env import TradingEnv                # noqa: E402

FIVE = ["NVDA", "TSLA", "AAPL", "META", "XOM"]
HPO_TRAIN_END = pd.Timestamp("2019-12-31")
HPO_VAL_START = pd.Timestamp("2020-01-02")
HPO_VAL_END = pd.Timestamp("2021-12-31")
COST_BY_NAME = {m.name: m for m in C.LEVELS}


def build_panels(store: Store):
    close = total_return_panel(store, FIVE, dt.date(2015, 1, 1), HPO_VAL_END.date())
    op = {}
    for t in FIVE:
        tp = store.tape(t).set_index("date")
        tp.index = pd.DatetimeIndex(tp.index)
        tp = tp[(tp.index >= close.index[0]) & (tp.index <= close.index[-1])]
        op[t] = tp["open"] * tp["adj_factor"]
    opn = pd.DataFrame(op).sort_index()
    adv, vol = liquidity_inputs(store, FIVE, close.index)
    return close, opn, adv, vol


def objective_factory(a, store, close, opn, adv, vol, log):
    from stable_baselines3 import PPO, TD3

    close_tr = close[close.index <= HPO_TRAIN_END]
    opn_tr = opn[opn.index <= HPO_TRAIN_END]
    adv_tr, vol_tr = adv[adv.index <= HPO_TRAIN_END], vol[vol.index <= HPO_TRAIN_END]
    close_val = close[(close.index >= HPO_VAL_START) & (close.index <= HPO_VAL_END)]
    opn_val = opn[(opn.index >= HPO_VAL_START) & (opn.index <= HPO_VAL_END)]
    adv_val, vol_val = liquidity_inputs(store, FIVE, close_val.index)
    cost_model = COST_BY_NAME[a.cost_level]

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        l1_penalty = trial.suggest_float("l1_penalty", 1e-5, 1e-2, log=True) \
            if trial.suggest_categorical("use_l1_penalty", [True, False]) else 0.0
        gamma = trial.suggest_float("gamma", 0.95, 0.999)

        env = TradingEnv(close_tr, opn_tr, adv_tr, vol_tr, cost_model,
                         lookback=a.lookback, l1_penalty=l1_penalty)
        env.reset(seed=trial.number)

        if a.algo == "td3":
            from stable_baselines3.common.noise import NormalActionNoise
            n_actions = env.action_space.shape[0]
            noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            buffer_size = min(200_000, a.timesteps * 4)   # see scripts/train_agent.py note
            model = TD3("MlpPolicy", env, seed=trial.number, verbose=0, batch_size=batch_size,
                       gamma=gamma, learning_rate=lr, action_noise=noise,
                       buffer_size=buffer_size, learning_starts=200)
        else:
            n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
            model = PPO("MlpPolicy", env, seed=trial.number, verbose=0, n_steps=n_steps,
                       batch_size=batch_size, gamma=gamma, learning_rate=lr)

        model.learn(total_timesteps=a.timesteps)

        strat = PPOStrategy(model, lookback=a.lookback, tag=f"hpo_{a.algo}_trial{trial.number}")
        r = evaluate(strat, close_val, opn_val, adv_val, vol_val, log=log,
                    levels=[cost_model], warmup=a.lookback,
                    note=f"S4 HPO trial {trial.number} algo={a.algo} lr={lr:.2e} "
                         f"batch={batch_size} l1={l1_penalty:.2e} gamma={gamma:.4f}")
        sharpe = r["by_cost"][cost_model.name]["sharpe"]
        trial.set_user_attr("model_path", None)   # only the winner is saved, in main()
        return sharpe if np.isfinite(sharpe) else -10.0

    return objective, (close_val, opn_val, adv_val, vol_val)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "td3"], default="ppo")
    ap.add_argument("--cost-level", choices=list(COST_BY_NAME), default="realistic")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel trials -- set to available vCPUs on the cloud VM")
    ap.add_argument("--timesteps", type=int, default=50_000,
                    help="kept modest per trial; this is a SEARCH, not a final run")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--out", default="data/hpo_result.json")
    a = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    store = Store("data")
    close, opn, adv, vol = build_panels(store)
    log = TrialLog("data/trials.jsonl")

    objective, val_panels = objective_factory(a, store, close, opn, adv, vol, log)
    study = optuna.create_study(direction="maximize",
                                study_name=f"rltrader_{a.algo}_{a.cost_level}",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=a.n_trials, n_jobs=a.n_jobs, show_progress_bar=False)

    print(f"best trial: #{study.best_trial.number}  validation Sharpe={study.best_value:.3f}")
    print(f"best params: {study.best_params}")
    print(f"n_trials logged this sweep: {a.n_trials}  "
          f"(honest cumulative log total: {log.n_trials('evaluation')})")

    out = {"algo": a.algo, "cost_level": a.cost_level, "n_trials": a.n_trials,
          "timesteps_per_trial": a.timesteps, "best_trial": study.best_trial.number,
          "best_validation_sharpe": study.best_value, "best_params": study.best_params,
          "all_trials": [{"number": t.number, "value": t.value, "params": t.params}
                         for t in study.trials]}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {a.out}")
    print("\nNEXT STEP (not run here): retrain with these params on the FULL 2015-2021 train "
          "window, evaluate ONCE on the untouched 2022-2026 test fold via scripts/train_agent.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
