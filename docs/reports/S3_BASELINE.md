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

See `docs/reports/S3_FINRL.md` (companion report). Independent worker, isolated venv.

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
