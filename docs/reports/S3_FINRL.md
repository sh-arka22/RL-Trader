# S3_FINRL.md — FinRL PPO baseline, reproduced from a pinned SHA

**Stage:** S3 (baseline + evaluation harness) · **Date:** 2026-09-15 · **Status:** complete
**Isolation:** all work done in `/tmp/finrl_repro`. Nothing was installed into `.venv-core`.
No file in this repo was touched except this report. No `git` command was run in this repo.

---

## 1. What this answers

`PLAN.md` S3 requires a *pinned-SHA FinRL PPO replication*, reported "even if it loses to DJIA".
`RESEARCH.md` §1.1 records two claims to test: that FinRL's headline numbers omit transaction
costs, and that FinRL-Meta's own reproduction of the FinRL PPO baseline scored Sharpe **0.99 vs
DJIA 1.32**.

This report measures what FinRL PPO actually produces on **our** universe, and audits FinRL's
cost model **by reading and instrumenting its source**, not its README.

Headline: **FinRL PPO does not beat equal-weight buy-and-hold on our universe, and the
seed-to-seed spread is wider than the gap between any two of our passive baselines.**

---

## 2. The pin

| Item | Value |
|---|---|
| Repo | `https://github.com/AI4Finance-Foundation/FinRL` |
| **Pinned commit SHA** | **`2334a5fe6d30629157f13c3b0319e1637e15e123`** |
| Commit date | 2026-07-12T14:22:09+08:00 |
| Commit subject | `Merge pull request #1427 from Jaco-Ren/fix-single-stock-colab-link` |
| `git describe --tags` | `v0.3.8-29-g2334a5f` (29 commits past release v0.3.8) |
| Working tree at run time | **clean — `git status --porcelain` returned 0 lines** |
| Transitive git pin | ElegantRL `24228304867bdc80165de435a598ef90b1893598` (FinRL depends on ElegantRL by git URL, not by version) |
| Interpreter | CPython **3.12.4** (`uv venv --python 3.12`) |
| Host | macOS 26.6.2, arm64 (Apple silicon) |
| Installer | `uv 0.12.13`, `uv pip install -e .` |

> **Reproducibility warning found in the pin itself.** FinRL's `pyproject.toml` declares
> `elegantrl = {git = "https://github.com/AI4Finance-Foundation/ElegantRL.git"}` with **no rev**.
> Pinning the FinRL SHA therefore does **not** pin the build. The ElegantRL SHA above is what
> resolved on 2026-09-15 and must be recorded separately. Anyone re-running this next month can get
> a different dependency tree from the same FinRL SHA.

---

## 3. Resolved dependency versions

175 packages resolved. Key versions:

| Package | Version |
|---|---|
| stable-baselines3 | 2.9.0 |
| gymnasium | 1.3.0 |
| torch | 2.14.0 |
| numpy | 2.5.3 |
| pandas | 2.2.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| stockstats | 0.5.4 |
| yfinance | 0.2.58 |
| pyfolio-reloaded | 0.9.9 |
| pyportfolioopt | 1.6.0 |
| ray | 2.58.0 |
| matplotlib | 3.11.2 |
| tensorboard | 2.21.0 |
| alpaca-py | 0.37.0 |
| alpaca-trade-api | 3.2.0 |
| ccxt | 3.1.60 |
| pandas-market-calendars | 5.4.0 |
| wrds | 3.5.0 |
| jqdatasdk | 1.9.8 |
| selenium | 4.32.0 |
| elegantrl | git @ `24228304867bdc80165de435a598ef90b1893598` |

**FinRL ships two dependency manifests that disagree, and one of them is dead code.**
`setup.py` parses `requirements.txt` (which lists `TA-lib`, needing a C library, plus `pytest`,
`sphinx`, `swig`, `webdriver-manager`). `pyproject.toml` declares a poetry-core build backend, so
`pip`/`uv` uses **`pyproject.toml` and ignores `requirements.txt` entirely**. Practical
consequence: **TA-Lib is never installed and is never needed** — the widely repeated "you must
conda-install ta-lib before FinRL" advice is stale for this SHA. FinRL uses `stockstats` for
indicators.

Full `uv pip freeze` is in appendix A.

---

## 4. Does it run unmodified in 2026? — **Yes.**

This was the result I least expected. **Zero patches were required.** All three files of FinRL's
own stock-trading example ran to completion, unmodified, on Python 3.12.4:

| Step | File (unmodified) | Exit | Notes |
|---|---|---|---|
| 1. Data | `examples/FinRL_StockTrading_2026_1_data.py` | **0** | Downloaded DOW-30 2014-01-06 → 2026-03-20 from yfinance, added indicators / VIX / turbulence, wrote `train_data.csv` (90,450 rows), `trade_data.csv` (1,560 rows) |
| 2. Train | `examples/FinRL_StockTrading_2026_2_train.py` | **0** | Trained A2C, DDPG, PPO, TD3, SAC at 20,000 steps each; all 5 `.zip` models saved |
| 3. Backtest | `examples/FinRL_StockTrading_2026_3_Backtest.py` | **0** | Loaded all 5 agents, ran MVO baseline via `pypfopt`, pulled `^DJI`, produced result table and `backtest_result.png` |

`git status --porcelain` in the FinRL clone was empty after all three runs.

**Python 3.13:** `uv pip install -e . --dry-run` against a 3.13.8 interpreter **also resolves
cleanly** (175 packages, exit 0). Full execution was not attempted on 3.13; the resolution is the
verified part.

Non-fatal noise observed, not patched:

- `Logging Error: 'rollout_buffer'` printed on every A2C/DDPG/TD3/SAC rollout end. FinRL's
  `TensorboardCallback._on_rollout_end` assumes a PPO/A2C-shaped `locals` dict and swallows the
  `KeyError` for off-policy algorithms. Cosmetic.
- `YF deprecation warning: set proxy via new config function` from `YahooDownloader`, which still
  passes the removed `proxy=` argument.

**This is a genuine negative result for the "FinRL is bit-rotted" hypothesis.** The maintenance
signal in `docs/research/05_oss_landscape.md` (100/100 recent commits) is real. FinRL's problem is
not that it fails to run. Its problem is what it computes when it does run — §6 and §8.

---

## 5. The cost assumption, read from the source

### 5.1 Where the number lives

`finrl/meta/env_stock_trading/env_stocktrading.py` — the env used by every stock-trading example —
has **no default cost**. `buy_cost_pct` and `sell_cost_pct` are required positional-style
arguments (`list[float]`, one per asset). The number therefore lives in the callers:

| Source file (at pinned SHA) | Line | Value |
|---|---|---|
| `examples/FinRL_StockTrading_2026_2_train.py` | 36 | `buy_cost_list = sell_cost_list = [0.001] * stock_dimension` |
| `examples/FinRL_StockTrading_2026_3_Backtest.py` | 56 | `buy_cost_list = sell_cost_list = [0.001] * stock_dimension` |
| `finrl/applications/stock_trading/stock_trading.py` | 85 | `buy_cost_list = sell_cost_list = [0.001] * stock_dimension` |
| `finrl/applications/stock_trading/stock_trading_rolling_window.py` | 89 | `buy_cost_list = sell_cost_list = [0.001] * stock_dimension` |
| `finrl/applications/stock_trading/ensemble_stock_trading.py` | 81 | `"buy_cost_pct": 0.001` |
| `finrl/applications/stock_trading/fundamental_stock_trading.py` | 854 | `"buy_cost_pct": 0.001` |
| `finrl/meta/env_stock_trading/env_stocktrading_np.py` | 18–19 | `buy_cost_pct=1e-3, sell_cost_pct=1e-3` (defaults) |
| `finrl/meta/env_stock_trading/env_nas100_wrds.py` | 26–27 | `buy_cost_pct=1e-3, sell_cost_pct=1e-3` (defaults) |
| `finrl/meta/env_stock_trading/env_stocktrading_cashpenalty.py` | 55–56 | `buy_cost_pct=3e-3, sell_cost_pct=3e-3` (defaults) |
| `finrl/meta/env_stock_trading/env_stocktrading_stoploss.py` | 67–68 | `buy_cost_pct=3e-3, sell_cost_pct=3e-3` (defaults) |

**Answer: FinRL's default stock-trading cost assumption is `0.001` = 10 bps per side, applied as a
proportional fee on traded notional.** Two of its less-used envs default to 30 bps instead. There
is no single constant; the value is copy-pasted across ten call sites and is inconsistent
between envs.

### 5.2 What the cost model contains, and what it omits

From `_buy_stock` / `_sell_stock`:

```python
sell_amount = price * sell_num_shares * (1 - self.sell_cost_pct[index])
buy_amount  = price * buy_num_shares  * (1 + self.buy_cost_pct[index])
```

The model is **a flat proportional fee and nothing else**. It has **no bid/ask spread, no
slippage, no market impact, no partial fills, no borrow cost, no short selling, no financing on
cash, and no minimum commission.** Fill price is always the observed close, regardless of size.

### 5.3 Direct falsification test: *is the fee actually charged?*

An earlier instrumented run reported `env.cost == $0.00` and `env.trades == 0` on every
evaluation, which looks like the env silently ignoring its configured cost. **That reading was
wrong, and the cause is worth recording.** `StockTradingEnv.reset()` sets `self.cost = 0` and
`self.trades = 0`; SB3's `DummyVecEnv` **auto-resets on `done`**, so any counter read *after* a
vectorised rollout is post-reset and always zero.

I falsified the "costs are not deducted" hypothesis directly. Same fixed random action tape, same
data, same env, three cost levels, counters read **before** any reset, no `DummyVecEnv`:

| `buy/sell_cost_pct` | Final asset value | `env.cost` | `env.trades` | Δ final vs 0 bps |
|---|---:|---:|---:|---:|
| `0.0` (0 bps) | $1,654,604.58 | $0.00 | 5,609 | — |
| `0.001` (10 bps) | $1,588,519.62 | $59,624.20 | 5,609 | **−$66,084.96** |
| `0.01` (100 bps) | $959,461.99 | $570,586.69 | 5,569 | **−$695,142.59** |

A control confirms the artefact: after the identical rollout run *through* `DummyVecEnv`,
`env.cost = 0.00` and `env.trades = 0`.

**Conclusion: FinRL's env does charge the fee it is configured with, and the charge scales
correctly.** The `RESEARCH.md` §1.1 claim should be read precisely — the problem is not that the
code omits costs, it is that **published FinRL results are reported at a 10 bps fee-only model
with zero spread, zero slippage and zero impact**, which is an optimistic cost model rather than
an absent one.

So why are the PPO results at 10 bps and 0 bps nearly identical? Not because the fee is ignored —
because **FinRL's default action scale makes the agent barely trade**. See §8.1.

---

## 6. The replication on our universe

### 6.1 Setup

| Item | Value |
|---|---|
| Universe | NVDA, TSLA, AAPL, META, XOM |
| Bars | daily, yfinance, back-adjusted by `YahooDownloader._adjust_prices` |
| Download span | 2014-01-02 → 2026-09-15 (2014 is warm-up for the 60-day SMA and the 252-day turbulence window) |
| Train split | **2015-01-02 → 2021-12-31** (1,763 trading days, 8,815 rows) |
| Test split | **2022-01-03 → 2026-09-11** (1,177 trading days, 5,885 rows) |
| Agent | `DRLAgent.get_model("ppo")`, SB3 `MlpPolicy` |
| PPO params | `n_steps=2048, ent_coef=0.01, learning_rate=0.00025, batch_size=128` — verbatim from `examples/FinRL_StockTrading_2026_2_train.py` |
| Env kwargs | `hmax=100, initial_amount=1_000_000, num_stock_shares=[0]*5, reward_scaling=1e-4` — verbatim from the same file |
| Test-env risk gate | `turbulence_threshold=70, risk_indicator_col="vix"` — verbatim from `..._3_Backtest.py` |
| Seeds | 0, 1, 2, 3, 4 |
| Components | FinRL's own `YahooDownloader`, `FeatureEngineer`, `data_split`, `StockTradingEnv`, `DRLAgent` — no re-implementation |

Requested test end was 2026-09-14; yfinance returned data through **2026-09-11**, so the realised
test end is 2026-09-11. Stated as measured, not as requested.

**Determinism check (S3 exit criterion).** Arm A was run twice, in separate processes, from the
same seeds. Every metric was **bit-identical** to the printed precision across all 20 cells.

### 6.2 Per-seed results

Each row is one trained policy. "eval@10bps" and "eval@0bps" are the **same policy** rolled out
through two cost settings, so the pair isolates the cost effect. "train bps" is the cost the
policy was *trained* under.

Arms: **A** = FinRL example defaults (`hmax=100`, 20k steps) · **B** = `hmax=100`, 100k steps ·
**C** = `hmax=1000`, 20k steps (action-scale sensitivity).

| Arm | train bps | seed | Sharpe @10bps | Sharpe @0bps | CumRet @10bps | CumRet @0bps | MaxDD @10bps | MaxDD @0bps | Turnover (×/yr) | AnnVol | Mean invested |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 10 | 0 | 0.760 | 0.759 | 168.3% | 167.2% | -41.6% | -41.5% | 0.76 | 36.6% | 95% |
| A | 10 | 1 | 0.916 | 0.917 | 90.5% | 90.7% | -26.1% | -26.1% | 0.20 | 16.6% | 70% |
| A | 10 | 2 | 1.113 | 1.114 | 239.7% | 240.1% | -34.0% | -34.0% | 0.40 | 26.7% | 92% |
| A | 10 | 3 | 0.747 | 0.747 | 113.4% | 113.8% | -37.2% | -37.2% | 0.46 | 26.4% | 94% |
| A | 10 | 4 | 0.756 | 0.759 | 168.2% | 170.1% | -56.0% | -55.9% | 0.67 | 37.0% | 95% |
| A | 0 | 0 | 0.479 | 0.474 | 71.1% | 69.1% | -52.0% | -52.4% | 0.61 | 50.4% | 96% |
| A | 0 | 1 | 0.638 | 0.640 | 62.1% | 62.4% | -29.6% | -29.6% | 0.49 | 19.1% | 68% |
| A | 0 | 2 | 1.120 | 1.123 | 268.2% | 269.5% | -30.4% | -30.6% | 0.61 | 28.6% | 91% |
| A | 0 | 3 | 0.741 | 0.744 | 112.3% | 112.6% | -37.8% | -37.7% | 0.29 | 26.5% | 93% |
| A | 0 | 4 | 0.839 | 0.839 | 171.3% | 171.4% | -39.6% | -39.5% | 0.24 | 31.3% | 93% |
| B | 10 | 0 | 0.810 | 0.822 | 215.6% | 223.9% | -57.0% | -57.7% | 0.94 | 40.5% | 99% |
| B | 10 | 1 | 0.638 | 0.638 | 105.1% | 105.2% | -50.8% | -50.7% | 0.72 | 32.3% | 98% |
| B | 10 | 2 | 0.811 | 0.810 | 186.9% | 186.7% | -49.7% | -49.9% | 0.43 | 35.7% | 98% |
| B | 10 | 3 | 0.323 | 0.324 | 17.1% | 17.5% | -64.7% | -64.7% | 0.46 | 51.8% | 98% |
| B | 10 | 4 | 0.631 | 0.634 | 119.9% | 121.0% | -59.5% | -59.4% | 0.68 | 38.3% | 99% |
| B | 0 | 0 | 0.305 | 0.307 | 20.1% | 20.5% | -57.4% | -57.2% | 0.43 | 42.8% | 99% |
| B | 0 | 1 | 0.882 | 0.868 | 226.8% | 221.3% | -59.7% | -60.2% | 0.78 | 36.2% | 98% |
| B | 0 | 2 | 1.188 | 1.189 | 341.7% | 340.8% | -29.6% | -29.6% | 0.53 | 30.8% | 97% |
| B | 0 | 3 | 0.656 | 0.652 | 130.8% | 129.3% | -60.7% | -60.9% | 0.66 | 38.8% | 99% |
| B | 0 | 4 | 0.697 | 0.698 | 140.4% | 140.6% | -52.8% | -52.4% | 0.41 | 36.5% | 98% |
| C | 10 | 0 | 0.522 | 0.568 | 70.2% | 82.0% | -47.4% | -46.4% | 10.94 | 31.0% | 98% |
| C | 10 | 1 | 0.636 | 0.642 | 94.0% | 95.9% | -40.1% | -39.8% | 0.73 | 28.9% | 99% |
| C | 10 | 2 | 0.627 | 0.627 | 89.1% | 89.1% | -41.6% | -41.6% | 0.35 | 28.1% | 100% |
| C | 10 | 3 | 0.864 | 0.865 | 157.6% | 158.2% | -32.8% | -32.9% | 0.55 | 28.0% | 98% |
| C | 10 | 4 | 0.510 | 0.469 | 79.5% | 65.9% | -67.7% | -67.7% | 0.49 | 41.3% | 99% |
| C | 0 | 0 | 0.341 | 0.376 | 26.7% | 36.4% | -66.8% | -67.0% | 20.42 | 46.3% | 96% |
| C | 0 | 1 | 0.567 | 0.579 | 88.4% | 90.2% | -57.4% | -54.0% | 0.42 | 34.5% | 99% |
| C | 0 | 2 | 0.756 | 0.753 | 103.9% | 103.4% | -27.7% | -27.6% | 0.44 | 24.0% | 100% |
| C | 0 | 3 | 0.358 | 0.374 | 34.3% | 38.4% | -53.3% | -53.1% | 1.69 | 40.7% | 89% |
| C | 0 | 4 | 0.581 | 0.594 | 96.9% | 100.1% | -60.0% | -59.9% | 0.31 | 36.4% | 99% |

### 6.3 Seed spread — the headline

| Arm | train bps | Sharpe mean | Sharpe sd | min | median | max | **spread** |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — hmax=100, 20k steps (FinRL example defaults) | 10 | 0.858 | 0.159 | 0.747 | 0.760 | 1.113 | 0.366 |
| A — hmax=100, 20k steps, trained cost-free | 0 | 0.764 | 0.240 | 0.479 | 0.741 | 1.120 | 0.641 |
| B — hmax=100, 100k steps | 10 | 0.642 | 0.199 | 0.323 | 0.638 | 0.811 | 0.488 |
| B — hmax=100, 100k steps, trained cost-free | 0 | 0.746 | 0.324 | 0.305 | 0.697 | 1.188 | 0.883 |
| C — hmax=1000, 20k steps | 10 | 0.632 | 0.142 | 0.510 | 0.627 | 0.864 | 0.354 |
| C — hmax=1000, 20k steps, trained cost-free | 0 | 0.521 | 0.173 | 0.341 | 0.567 | 0.756 | 0.415 |

**Across all 30 trained policies evaluated at FinRL's own 10 bps assumption, Sharpe ranges from
0.305 to 1.188 — a spread of 0.88 Sharpe points driven by seed and by two defaults that FinRL
never presents as tunable.** Within the single most faithful arm (A, train@10 bps) the spread is
still 0.747 → 1.113.

Reporting any one of these as "the FinRL PPO result" is a choice, not a measurement.

---

## 7. The bar it had to clear

Same test window (2022-01-03 → 2026-09-11), same price data, rf = 0:

| Benchmark | Sharpe | CumRet | CAGR | MaxDD | AnnVol |
|---|---:|---:|---:|---:|---:|
| Equal-weight 5, buy once, never rebalance | 0.976 | 201.3% | 26.7% | -29.0% | 28.3% |
| Equal-weight 5, daily rebalanced | 0.959 | 205.7% | 27.1% | -39.9% | 29.5% |
| SPY buy-and-hold | 0.735 | 69.3% | 11.9% | -24.5% | 17.4% |
| DJIA (^DJI) buy-and-hold | 0.589 | 43.3% | 8.0% | -21.9% | 15.0% |
| NVDA buy-and-hold | 1.077 | 627.8% | 53.0% | -62.7% | 51.9% |
| XOM buy-and-hold | 1.037 | 207.2% | 27.2% | -20.5% | 26.6% |
| META buy-and-hold | 0.540 | 93.1% | 15.1% | -73.7% | 45.7% |
| AAPL buy-and-hold | 0.614 | 86.9% | 14.3% | -33.4% | 28.3% |
| TSLA buy-and-hold | 0.266 | -8.6% | -1.9% | -73.0% | 59.8% |

**Verdict.**

- FinRL PPO at its own defaults (arm A, train@10 bps): mean Sharpe **0.858**, best seed **1.113**.
- Equal-weight buy-and-hold of the same 5 names: Sharpe **0.976**, cumulative **+201.3%**.
- **PPO's mean Sharpe loses to equal-weight buy-and-hold. Its mean cumulative return (+156.0%)
  loses by 45 percentage points.** Only 1 of 5 seeds beats equal-weight on Sharpe, and that seed
  still returns +239.7% vs +201.3% while taking a worse drawdown.
- PPO does beat **DJIA** (0.589) and **SPY** (0.735) — but so does simply holding the five stocks.
  Beating the index while losing to the naive portfolio of your own universe is exactly the
  pattern `RESEARCH.md` §2 predicted.
- This reproduces the **FinRL-Meta pattern** (`RESEARCH.md` §1.1: PPO 0.99 vs DJIA 1.32) in
  structure though not in sign: on our window and universe the index is weak (DJIA 0.589), so PPO
  clears the index and loses to the naive alternative instead.

### 7.1 It is mostly beta, not a strategy

Daily-return correlation between the PPO equity curve and equal-weight buy-and-hold of the same
5 names, arm A, all 20 rollouts: **0.48 to 0.88**. Mean invested fraction **70–96%**, rising to
**≈100% from 2023 onward and staying there**. Annual turnover **0.20–0.76×**.

FinRL PPO at defaults converges to a **near-static long-only allocation**. Most of its reported
Sharpe is the market, not the policy.

---

## 8. Every place FinRL's defaults would flatter a result

Ordered by how much damage each one does.

### 8.1 🔴 `hmax=100` suppresses turnover, which hides the cost model

`hmax` is a cap in **shares per asset per day**. With `initial_amount=1_000_000` and 5 assets, the
agent can move at most a few tens of thousands of dollars a day — annual turnover of **0.2–0.9×**.
At 10 bps per side that is roughly **13 bps of drag per year**, invisible against 16–52% annual
volatility. **This is why eval@10bps ≈ eval@0bps in arm A, and it is not evidence that costs do
not matter.**

Raise the action scale to `hmax=1000` (arm C) and turnover reaches **10.9–20.4×/yr** for seed 0,
and the cost term becomes visible immediately:

| seed 0, arm C | eval @10 bps | eval @0 bps | Δ |
|---|---:|---:|---:|
| Sharpe (train@10 bps) | 0.522 | 0.568 | **−0.046** |
| CumRet (train@10 bps) | +70.2% | +82.0% | **−11.8 pp** |
| Sharpe (train@0 bps) | 0.341 | 0.376 | **−0.034** |
| CumRet (train@0 bps) | +26.7% | +36.4% | **−9.7 pp** |

**A default that makes the agent nearly static is a default that makes the cost model look
harmless.** Any FinRL result quoted with `hmax=100` on a small universe has effectively
sidestepped the transaction-cost question.

One further honest detail: costs are **path-dependent**, not a scalar drag. A fee changes how many
shares `cash // (price * (1 + fee))` affords, which changes every later position. In arm C seed 4
the cost-free rollout scored **worse** (0.469 vs 0.510) for exactly this reason. Our harness must
therefore treat "same policy, different cost level" as **different trajectories**, and must never
approximate a cost level by subtracting `bps × turnover` from a zero-cost equity curve.

### 8.2 🔴 `DOW_30_TICKER` is the **2026** index membership, backtested from 2014

`finrl/config_tickers.py` hard-codes today's Dow constituents — including **AMZN, NVDA, SHW and
CRM**, all added to the index long after 2014 — and the flagship example runs them from
2014-01-06. NVDA alone returned over 100× across that span. This is textbook
survivorship / index-reconstitution bias in FinRL's headline demo. The file even admits the
practice for the sibling list: `# Nasdaq 100 constituents at 2026/03/20`.
*(Not applicable to our 5-name run, which fixes the universe up front — but it invalidates
straight comparison against any published FinRL Dow-30 number.)*

### 8.3 🔴 Same-bar fill at the observed close

`step()` executes at `self.state[index + 1]`, which is **the close of the same day the agent
observed**, and all technical indicators in that observation are computed through that same close.
Observe close *t* → fill at close *t*, at zero slippage, for any size. This is precisely the flaw
`RESEARCH.md` §3 item 1 used to reject TensorTrade. **FinRL has it too.**

### 8.4 🟠 `clean_data` filters the universe over the full sample

`FeatureEngineer.clean_data` pivots closes and calls `merged_closes.dropna(axis=1)` over the
**entire** train+test span, dropping any ticker with a gap anywhere — including in the future
relative to the training period. `preprocess_data` is applied to train and test jointly before
`data_split`, and finishes with `df.ffill().bfill()`, where the `bfill` pulls later values
backwards at the start of the series.

### 8.5 🟠 The backtest starts 100% in cash

`num_stock_shares=[0]*n` with `initial_amount=1_000_000` means the test episode opens fully in
cash and must build exposure at `hmax` shares/day. Our arm A seed 0 was only **28% invested over
its first 60 test days** and **78% invested across 2022**. When a test window opens with a
drawdown — as 2022 did — that is free downside protection that no live account starting from an
existing book would receive. *(It did not save this agent: seed 0 still lost **−38.4%** in 2022
against equal-weight's **−26.6%**.)*

### 8.6 🟠 Train and test envs are not the same env

Training uses no risk gate. The backtest example adds `turbulence_threshold=70,
risk_indicator_col="vix"`. A train/test env mismatch is a silent evaluation confound. On our
window it is inert — daily VIX closes never reached 70 between 2022 and 2026 — but on a window
containing March 2020 it fires and liquidates the whole book, which reads as skill and is a
hard-coded constant.

### 8.7 🟡 Back-adjusted prices are traded as if they were live prices

`YahooDownloader._adjust_prices` rescales O/H/L/C by `Adj Close / Close`, so the whole history is
restated using **later** dividend and split information. The series the agent trades was never
quoted.

### 8.8 🟡 Action semantics are unprincipled

`actions = (actions * hmax).astype(int)` truncates toward zero, so with `hmax=100` any
|action| < 0.01 becomes a dead zone of exactly 0 shares. The cap is in **shares**, so a $500 name
and a $30 name get the same share budget and wildly different dollar budgets. Sells run before
buys and `available_amount = cash // price` lets the **first** buy in the loop consume all cash —
allocation is order-dependent, driven by `argsort` of the raw action vector.

### 8.9 🟡 20,000 training steps is ~11 episodes — and training longer makes it worse

FinRL's example trains for 20,000 steps on a 1,763-day episode. Arm B raised this to 100,000 (~57
episodes) and mean Sharpe at train@10bps **fell from 0.858 to 0.642**, with the spread widening to
0.323–0.811. There is no held-out model selection anywhere in the pipeline; the example trains
once and reports.

### 8.10 🟡 Reported Sharpe is raw and unadjusted

The env prints `252**0.5 * mean / std` with **rf = 0**, no bootstrap CI, no multiple-testing
adjustment, single seed. Our own §6.3 spread shows why that is not a measurement: seed choice
alone moves Sharpe by more than the gap between SPY (0.735) and equal-weight (0.976).

### 8.11 ⚪ Instrumentation trap worth carrying into `rltrader/eval/`

`StockTradingEnv.reset()` zeroes `self.cost` and `self.trades`, and SB3's `DummyVecEnv`
auto-resets on `done`. **Any cost or trade counter read after a vectorised rollout is silently
zero.** This produced a false "FinRL does not charge its fees" reading in this very
investigation. Our harness must capture accounting state from `info` at the terminal step, never
by reading env attributes afterwards.

---

## 9. What S3 should take from this

1. **The number to carry forward.** FinRL PPO on NVDA/TSLA/AAPL/META/XOM, train 2015–2021, test
   2022–2026, at FinRL's own 10 bps assumption and its own defaults:
   **Sharpe 0.858 ± 0.159 (n = 5), range 0.747–1.113; cumulative +156.0%; max drawdown −39.0%;
   turnover 0.50×/yr.** Pinned to FinRL `2334a5fe6d30`.
2. **It loses to the naive baseline.** Equal-weight buy-and-hold: Sharpe 0.976, +201.3%. Record
   this as the FinRL comparison point and do not treat it as a target to beat by tuning.
3. **Report spreads, never best seeds.** 5 seeds moved Sharpe by 0.37 in the tightest arm and 0.88
   across arms. `rliable` bootstrap CIs in `rltrader/eval/` are not optional.
4. **Do not inherit FinRL's env.** Same-bar close fills (§8.3), share-denominated actions (§8.8),
   order-dependent allocation (§8.8) and a fee-only cost model (§5.2) each independently disqualify
   it as our execution layer. The ~400-LOC Gymnasium env decided in `ARCHITECTURE.md` stands.
5. **Cost realism is about the *model*, not the *switch*.** FinRL charges its fee correctly (§5.3).
   The optimism is in what the fee omits — spread, slippage, impact, partial fills — and in an
   action scale that keeps turnover near zero. Our three-level cost model must vary turnover as
   well as bps, or it will reproduce FinRL's blind spot.
6. **The "FinRL is bit-rotted" prior is wrong.** It installs and runs unmodified on Python 3.12 in
   2026, and resolves on 3.13. Reject it on measurement design, not on maintenance.

---

## 10. Reproduce

Everything lives in `/tmp/finrl_repro` (outside this repo, by design):

```
/tmp/finrl_repro/FinRL/                     clone @ 2334a5fe6d30629157f13c3b0319e1637e15e123
/tmp/finrl_repro/.venv-finrl/               Python 3.12.4 venv (uv)
/tmp/finrl_repro/run_unmodified/            FinRL's 3 example files, run verbatim
/tmp/finrl_repro/work/run_s3_finrl.py       our-universe PPO runner (FinRL components only)
/tmp/finrl_repro/work/run_s3_finrl_v2.py    + exposure metrics + per-run series dump
/tmp/finrl_repro/work/cost_test.py          §5.3 cost falsification test
/tmp/finrl_repro/work/baselines.py          §7 passive baselines
/tmp/finrl_repro/work/results_*.json        all metrics
/tmp/finrl_repro/work/series/               per-run account-value and action tapes
/tmp/finrl_repro/logs/                      install, example, and run logs
```

```bash
mkdir -p /tmp/finrl_repro && cd /tmp/finrl_repro
git clone https://github.com/AI4Finance-Foundation/FinRL.git FinRL
cd FinRL && git checkout 2334a5fe6d30629157f13c3b0319e1637e15e123
cd /tmp/finrl_repro && uv venv --python 3.12 .venv-finrl
cd FinRL && VIRTUAL_ENV=/tmp/finrl_repro/.venv-finrl uv pip install -e .
```

---

## Appendix A — full `uv pip freeze` (175 packages, Python 3.12.4)

```
absl-py==2.5.0
aiodns==4.0.4
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiohttp-cors==0.8.1
aiosignal==1.4.0
ale-py==0.12.1
alpaca-py==0.37.0
alpaca-trade-api==3.2.0
annotated-types==0.8.0
asttokens==3.0.2
attrs==26.1.0
beautifulsoup4==4.15.0
bottleneck==1.6.0
ccxt==3.1.60
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
clarabel==0.11.1
click==8.5.0
cloudpickle==3.1.2
colorful==0.5.8
contourpy==1.4.0
cryptography==50.0.1
curl-cffi==0.16.3
cvxpy==1.9.2
cycler==0.12.1
decorator==5.3.1
deprecation==2.1.0
distlib==0.4.3
elegantrl @ git+https://github.com/AI4Finance-Foundation/ElegantRL.git@24228304867bdc80165de435a598ef90b1893598
empyrical-reloaded==0.5.12
exchange-calendars==4.13.2
executing==2.2.1
farama-notifications==0.0.6
filelock==3.32.6
-e file:///private/tmp/finrl_repro/FinRL
fonttools==4.65.0
frozendict==2.4.7
frozenlist==1.8.0
fsspec==2026.7.0
google-api-core==2.37.0
google-auth==2.58.0
googleapis-common-protos==1.75.3
grpcio==1.84.0
gymnasium==1.3.0
h11==0.16.0
highspy==1.15.1
idna==3.19
ijson==3.5.1
ipython==9.8.0
ipython-pygments-lexers==1.1.1
jedi==0.20.0
jinja2==3.1.6
joblib==1.6.0
jqdatasdk==1.9.8
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
korean-lunar-calendar==0.4.0
markdown==3.10.3
markdown-it-py==4.2.0
markupsafe==3.0.3
matplotlib==3.11.2
matplotlib-inline==0.2.2
mdurl==0.1.2
mpmath==1.3.0
msgpack==1.0.3
multidict==6.8.0
multitasking==0.0.13
narwhals==2.26.0
networkx==3.6.1
niltype==1.0.2
numpy==2.5.3
opencensus==0.11.4
opencensus-context==0.1.3
opencv-python==5.0.0.93
opentelemetry-api==1.44.0
opentelemetry-exporter-prometheus==0.65b0
opentelemetry-proto==1.44.0
opentelemetry-sdk==1.44.0
opentelemetry-semantic-conventions==0.65b0
osqp==1.1.3
outcome==1.3.0.post0
packaging==26.3
pandas==2.2.3
pandas-market-calendars==5.4.0
parso==0.8.7
peewee==3.17.3
pexpect==4.9.0
pillow==12.3.0
platformdirs==4.11.8
ply==3.11
prometheus-client==0.26.0
prompt-toolkit==3.0.53
propcache==0.5.2
proto-plus==1.28.4
protobuf==7.36.1
psutil==7.2.2
psycopg2-binary==2.9.13
ptyprocess==0.7.0
pure-eval==0.2.4
py-spy==0.4.2
pyarrow==25.0.1
pyasn1==0.6.4
pyasn1-modules==0.4.2
pycares==5.0.1
pycparser==3.0
pydantic==2.13.5
pydantic-core==2.46.5
pyfolio-reloaded==0.9.9
pygame-ce==2.5.8
pygments==2.21.0
pyluach==2.3.0
pymysql==1.2.0
pyparsing==3.3.2
pyportfolioopt==1.6.0
pysocks==1.7.1
python-dateutil==2.9.0.post0
python-discovery==1.6.0
python-dotenv==1.2.3
pytz==2026.3.post1
pyyaml==6.0.1
qdldl==0.1.9.post1
ray==2.58.0
referencing==0.37.0
requests==2.34.2
rich==15.0.0
rpds-py==2026.6.3
scikit-base==0.13.2
scikit-learn==1.9.1
scipy==1.18.1
scs==3.3.1
seaborn==0.13.2
selenium==4.32.0
setuptools==84.0.0
six==1.17.0
smart-open==8.0.1
sniffio==1.3.1
sortedcontainers==2.4.0
soupsieve==2.9.2
sparsediffpy==0.3.0
sqlalchemy==2.0.53
sseclient-py==1.9.0
stable-baselines3==2.9.0
stack-data==0.6.3
stockstats==0.5.4
sympy==1.14.0
tensorboard==2.21.0
tensorboard-data-server==0.7.2
tensorboardx==2.6.5
th==0.4.1
threadpoolctl==3.6.0
thriftpy2==0.7.1
toolz==1.1.0
torch==2.14.0
tqdm==4.70.1
traitlets==5.16.1
trio==0.34.0
trio-websocket==0.12.2
typing-extensions==4.16.0
typing-inspection==0.4.4
tzdata==2026.4
urllib3==1.26.20
virtualenv==21.7.9
wcwidth==0.8.3
webdriver-manager==4.1.2
websocket-client==1.9.2
websockets==10.4
werkzeug==3.1.8
wrapt==2.4.1
wrds==3.5.0
wsproto==1.3.2
yarl==1.24.5
yfinance==0.2.58
```
