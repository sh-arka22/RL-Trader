# S3 — Baseline & Evaluation Harness

Built 2026-09-14/15. This is the project's kill-gate stage (PLAN.md): everything the
trading agent is ever compared against is defined here, before the agent exists, so the
comparison cannot be tuned after the fact.

## 1. The bar had to be corrected before anything else could be measured

`RESEARCH.md` recorded the equal-weight-5 passive Sharpe as **1.07**. That number is
correct for the basket it was computed on — **AAPL/MSFT/JNJ/JPM/XOM** — which is not the
basket this project trades. It reproduces exactly, at **1.073**, daily-rebalanced. The
universe was later changed to **NVDA/TSLA/AAPL/META/XOM** and nobody moved the bar.

| Control | Sharpe |
|---|---|
| SPY buy-and-hold | 0.823 (matches RESEARCH.md's 0.82) |
| EW-5 buy-and-hold — **the real control** | **1.201** |
| EW-5 daily rebalanced | 1.234 |
| *Retired basket* AAPL/MSFT/JNJ/JPM/XOM, daily rebalanced | 1.073 ← where "1.07" came from |

The hurdle is **~0.13 Sharpe higher** than planned. An agent scoring 1.10 would have
looked like a win against the stale number while losing to doing nothing. The bar is now
computed from the store by `tests/test_baselines.py`, never copied from prose
(`rltrader/eval/bar.py`). Separately: the choice of return convention alone (log vs
simple, dividend-adjusted vs not, rebalanced vs not) moves the same-window Sharpe across
a **0.35-point range** (0.885–1.238) — wider than most improvements claimed in the
literature this project surveyed.

## 2. Harness design

`rltrader/eval/harness.py`. Three structural defences against look-ahead, because
`RESEARCH.md` found that statistics do not catch it (arXiv:2608.27734: a planted oracle
scored Deflated Sharpe Ratio **1.00**):

1. A strategy is called with only `panel.iloc[:t+1]` — the future is not in the object it
   receives, so leakage requires a deliberate act, not an accidental one.
2. A decision from the close of session `t` fills at the **open of `t+1`** and earns the
   open-to-open return to `t+2`. Same-bar fills are the most common silent look-ahead in
   retail backtests (this is exactly the TensorTrade defect `RESEARCH.md` §1 rejected it
   for).
3. Every evaluation is appended to a **hash-chained trial log**
   (`rltrader/eval/trial_log.py`) before its metrics are read. Editing or deleting a past
   record breaks the chain — the motive for deleting one is to shrink the Deflated Sharpe
   denominator, so the log makes that motive self-defeating.

**Proof the harness can show a leak**: `PlantedOracle` in `strategies.py` is handed the
answer key explicitly (never through the normal `weights(history, held)` interface) and
scores Sharpe **15.49** against a passive bar near 1.2 —
`tests/test_harness.py::test_planted_oracle_scores_absurdly`.

**Determinism**: two independent runs of the same strategy over the same window produce a
byte-identical result digest (`scripts/gate_mutations.py`-style check, ad hoc script, not
yet a committed gate — tracked as a follow-up).

### Three bugs found bringing it up

| Bug | Symptom | Fix |
|---|---|---|
| Turnover measured target-to-target | Daily rebalancing was free; equal-weight showed 0.0000 turnover | Measured against the drifted book |
| A strategy could not express "do nothing" | Buy-and-hold and daily-rebalanced equal-weight were the *same* strategy (both 1.217) | `weights()` receives the currently-held book |
| Costs computed on a $1 portfolio | ADV cap and impact never bound | Portfolio size explicit ($1M); impact-is-negligible-at-this-size is now a stated finding, not a silent assumption |
| `Momentum.name` was a class constant | Six distinct (lookback, top_k) trials collided under one key, silently discarding 4 of 6 from any dict keyed on strategy name (found while wiring PBO) | `name` now encodes the parameters |

## 3. Cost model

`rltrader/eval/costs.py`. Three declared levels, because `RESEARCH.md` (arXiv:2603.29086)
found that algorithm *rankings* flip under different cost models on the same data — a
single cost assumption does not produce a weaker number, it produces an arbitrary one.

| Level | Commission | Half-spread | Impact (Almgren-Chriss η) | ADV cap |
|---|---|---|---|---|
| free | 0 bp | 0 bp | 0 | none |
| **realistic** | 0 bp | 1 bp | 0.1 | 1% of ADV |
| pessimistic | 0 bp | 2 bp | 0.3 | 0.5% of ADV |

Fills clip at the ADV cap rather than reject, and the clipped/unfilled notional is
reported rather than silently dropped.

## 4. Statistics module

`rltrader/eval/stats.py` (written by an independent worker, 34 known-answer tests, all
pass): Sharpe with Lo (2002) standard errors, Probabilistic and Deflated Sharpe Ratio
(Bailey & Lopez de Prado 2014 — validated against the paper's own worked example, p.9-10),
Probability of Backtest Overfitting via CSCV (Bailey et al. 2015), Diebold-Mariano with
the Harvey-Leybourne-Newbold small-sample correction, stationary bootstrap (Politis &
Romano 1994), and Benjamini-Hochberg FDR control. `deflated_sharpe_ratio`'s docstring
states plainly, and a test enforces, that **DSR does not detect look-ahead bias** — the
same arXiv:2608.27734 finding as §2.

**Usage note (found wiring it in): DSR inputs are per-observation, not annualised, unless
`periods=` is passed.** Passing an annualised Sharpe without `periods=252` silently
saturates the statistic near 1.0 for any input — an easy way to manufacture false
confidence. `scripts/run_baselines.py` passes `periods=252` explicitly.

## 5. Baseline sweep results

`scripts/run_baselines.py`. Window 2015-01-02 → 2026-09-11 (2,940 sessions), $1M
portfolio, all three cost levels, every run logged to `data/trials.jsonl`
(hash-chain intact, `8` trials).

| Strategy | Sharpe (free) | Sharpe (realistic) | Sharpe (pessimistic) | Turnover | Max DD |
|---|---|---|---|---|---|
| equal_weight | 1.217 | 1.216 | 1.214 | 0.0070 | -39.9% |
| buy_and_hold | 1.181 | 1.181 | 1.181 | 0.0002 | -58.6% |
| momentum_20d_top1 | 1.031 | 0.991 | 0.940 | 0.1722 | -53.8% |
| momentum_20d_top2 | 1.265 | 1.232 | 1.191 | 0.1242 | -42.5% |
| momentum_60d_top1 | 0.917 | 0.897 | 0.872 | 0.0913 | -65.4% |
| momentum_60d_top2 | 1.166 | 1.150 | 1.129 | 0.0704 | -46.0% |
| momentum_120d_top2 | 1.171 | 1.159 | 1.143 | 0.0552 | -49.4% |
| momentum_252d_top2 | 1.173 | 1.165 | 1.153 | 0.0404 | -49.1% |

**Momentum beats buy-and-hold, and the significance survives honest multiple-testing
correction** — the headline result of this stage:

- Diebold-Mariano, `momentum_20d_top2` vs `buy_and_hold`, realistic costs: statistic
  8.471, **p = 0.0000**.
- Deflated Sharpe Ratio for the best-of-8 trial
  (`momentum_20d_top2`, realistic Sharpe 1.232):
  **0.9970**, above the project's 0.95 gate, counting every trial
  in the immutable log — not a cherry-picked denominator.
- **Caveat, stated rather than hidden**: PBO across the 6 momentum variants is
  **0.729** — elevated. The six variants are all momentum
  on the same universe and are highly correlated with each other, which is exactly the
  regime CSCV is least informative in (few effectively-independent trials to resample).
  `prob_oos_loss` reads 0.000, which is hard to reconcile with a 0.73 PBO and is flagged
  here as an open question for S4 rather than resolved by assertion.
- This is a **baseline-vs-baseline** result — no RL agent exists yet. It sets the standard
  the agent must clear: beating buy-and-hold is not enough, since simple momentum already
  does that here.

## 6. FinRL PPO comparison point

See `docs/reports/S3_FINRL.md` (654 lines, independent worker, isolated venv, pinned SHA
`2334a5fe6d30629157f13c3b0319e1637e15e123`, clean tree, zero patches).

**Headline: FinRL PPO does not beat equal-weight buy-and-hold on our universe.**

| | Sharpe | Cum. return |
|---|---|---|
| FinRL PPO, defaults, 10 bps, 5 seeds (mean ± std) | **0.858 ± 0.159** (range 0.747–1.113) | +156.0% |
| Equal-weight buy-and-hold, same 5 names | **0.976** | +201.3% |
| SPY | 0.735 | — |
| DJIA | 0.589 | — |

Only 1 of 5 seeds beats equal-weight. Return correlation to EW-5 is 0.48–0.88 and mean
invested fraction reaches ~100% from 2023 on — the trained policy is close to a static
long-only book, i.e. mostly beta, not evidence of a control edge.

**A claim I made in an earlier status update was wrong and is corrected here.** I stated
that FinRL's environment "does not deduct the transaction cost it is configured with"
based on `env_total_cost_usd=0` / `env_total_trades=0` in a raw results file. The worker
falsified this directly: those counters are zeroed by `reset()`, and SB3's `DummyVecEnv`
auto-resets on episode end, so reading them after rollout always shows zero regardless of
whether costs were charged. A fixed action tape read *before* reset shows the fee scaling
correctly (0 bps → $0 cost; 10 bps → $59,624; 100 bps → $570,587, on a $1.65M book).
**FinRL does deduct its configured cost.** The reason 10 bps looked nearly free in the
default run is real but different: `hmax=100` (a per-asset daily share cap) holds turnover
to 0.2–0.9×/yr regardless of cost, so 10 bps costs only ~13 bps/yr against 16–52% annual
volatility — the cost is genuinely small at that turnover, not absent. Raising `hmax` to
1000 lifts turnover to 10.9–20.4×/yr and the cost gap appears immediately (Sharpe 0.522
vs 0.568). This is a lesson about verifying an instrumentation claim before publishing it,
not about FinRL.

**Confirmed cost assumption, read from source, not the README**: 0.001 (10 bps) **per
side**, proportional to notional, hard-coded across ten call sites in FinRL's own
examples; `env_stocktrading.py` itself has no default. Two sibling environments in the
same repo default to 30 bps. The model has no spread, slippage, market impact, partial
fills, shorting, or borrow cost, and fills at the **observed close** — the same same-bar
fill defect `RESEARCH.md` §1 rejected TensorTrade for.

**Other flags from the report, most consequential first**: `DOW_30_TICKER` is the *2026*
index membership (AMZN, NVDA, SHW, CRM) backtested from 2014 — survivorship bias baked
into FinRL's own flagship demo; `clean_data` drops tickers over the full train+test span
before the split, which can leak a delisting/relisting pattern; the test environment
starts 100% cash while train does not, and a VIX turbulence gate is applied only at test
time — a train/test asymmetry; 100k training steps scored *worse* than 20k (0.642 vs
0.858), i.e. more compute made the result worse, not better.

**Design consequence for S4, recorded now**: FinRL's PPO is cost-aware during training —
its reward includes the fee, so training at 0 bps and training at 10 bps produce
*different policies*, not the same policy priced two ways. Costs are therefore
**path-dependent for any strategy that can react to them** (the worker found one seed
that scored worse at 0 bps than at 10 bps). Our current baselines are cost-*agnostic* —
`weights()` depends only on price history, never on realised fees — so overlaying three
cost levels onto one recorded trajectory (`harness.evaluate`) is exact for them. It will
stop being exact the moment an RL agent is introduced: **the agent must be retrained
separately at each cost level**, never trained once and cost-overlaid after the fact.
This is now recorded as an explicit constraint for S4/S5 (see `ARCHITECTURE.md`).

## 7. Honest gaps

- Determinism and the mutation-style "can this gate fail" proof exist as ad hoc scripts
  for the harness, not yet committed as tests the way S2's gates are. Follow-up for S4.
- PBO's disagreement with `prob_oos_loss` on the momentum family (§5) is unresolved.
- The planted-oracle test proves the harness CAN show a leak that enters through the price
  panel. A leak entering through a feature computed elsewhere (e.g. a knowledge-graph
  embedding built with future information) is not covered here and must be caught by the
  ablation protocol in `ARCHITECTURE.md`.
- Walk-forward folds (`walk_forward_folds`) are implemented and tested for
  non-overlap but not yet run end-to-end on the real universe; the sweep above uses the
  full window as a single evaluation, which is appropriate for a baseline comparison but
  not for the agent, which needs held-out folds.
