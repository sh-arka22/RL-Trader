# S4 — Core RL Agent

Built and evaluated 2026-09-15. This is the stage `PLAN.md` calls "a price-only PPO that
is a *credible control*, not a winner." The result: **it is not, by the project's own
pre-registered bar.** That is reported here in full, because a modest or negative result
was named as a likely, acceptable outcome before this run started (`PLAN.md` S4,
`RESEARCH.md` §1).

## 1. Environment

`rltrader/envs/trading_env.py` (~230 LOC), replacing TensorTrade. Fixes every defect
`RESEARCH.md` cited against it:

| TensorTrade defect (cited) | This environment |
|---|---|
| No `super().reset(seed=seed)`; fails `gymnasium.check_env` | Calls it; **`check_env` passes, including seeding** — `tests/test_trading_env.py::test_passes_gymnasium_check_env` |
| No partial fills | ADV-cap partial fills — `test_adv_cap_gives_a_partial_fill` |
| Same-bar fills (observe close 100.0, fill 100.0) | Fills at `open[t+1]`, scored on `open[t+1]->open[t+2]` — `test_fill_is_not_same_bar` |
| Dead slippage code, 30 bps hidden default | `rltrader/eval/costs.py`, explicit declared levels, reused by both the env and the harness |

Action space: `Box(-1, 1, shape=(5,))`, normalised per SB3's own guidance, scaled
internally to logits before a softmax-with-implicit-cash produces a long-only 6-vector
summing to exactly 1 by construction. Observation: the last 20 daily returns per asset
plus the currently-held weights (not the last target — the same distinction that fixed a
turnover bug in S3). A Mahalanobis-distance turbulence gate can force liquidation to cash;
it is off by default and was exercised only in unit tests here.

**Env-level planted-oracle check** (`test_env_level_planted_oracle_is_detected`): a policy
fed tomorrow's winning asset through a side channel — never through the real observation —
scores far above an honest random policy trained in the same loop. If a real leak entered
this environment, this check would show it.

**Costs are trained-in, not overlaid** (`ARCHITECTURE.md` section 3.6, from the S3 FinRL
replication): a cost-aware policy trained at 0 bps and one trained at 10 bps are different
policies. So this stage trains **three separate agents**, one per cost level, never one
model re-costed after the fact.

## 2. Training

`scripts/train_agent.py`. PPO (`stable-baselines3` 2.9.0, pinned), `n_steps=2048`,
`batch_size=64`, `gamma=0.99`, `lr=3e-4`, 100,000 timesteps per seed, **20 seeds per cost
level** (`PLAN.md` S4 exit criteria). Train 2015-01-02 to 2021-12-31, test
2022-01-03 to 2026-09-11 — identical to the FinRL replication split in `S3_FINRL.md`, so
every number below is directly comparable to it.

A trained policy is wrapped by `rltrader/eval/agent.py::PPOStrategy` and scored through
the **same S3 harness** that scored every baseline — identical fill timing, identical
cost application. Two different fill implementations that happen to agree would not be
evidence; one implementation used for both training-adjacent evaluation and baseline
evaluation is.

## 3. Results — 20 seeds per cost level, out-of-sample

| Cost level | IQM Sharpe | 95% bootstrap CI | mean | std | min | max |
|---|---|---|---|---|---|---|
| free | 0.955 | [0.813, 1.072] | 0.945 | 0.250 | 0.531 | 1.469 |
| realistic | 0.933 | [0.796, 1.032] | 0.907 | 0.246 | 0.413 | 1.433 |
| pessimistic | 0.916 | [0.755, 1.017] | 0.875 | 0.273 | 0.266 | 1.415 |

**In-sample vs out-of-sample (realistic cost, the 5-seed pilot that preceded this run)**:
IS Sharpe 1.37 -> OOS Sharpe 0.88 for the same 5 seeds — a real generalisation gap, the
same shape as the FINSABER design-vs-eval degradation `RESEARCH.md` cites.

## 4. The comparison that matters: against the baselines and the sanity floor

Same out-of-sample window, realistic cost:

| | Sharpe | |
|---|---|---|
| Momentum (20d, top-2) | 1.084 | beats PPO |
| Equal-weight buy-and-hold | 0.997 | beats PPO |
| Momentum (60d, top-2) | 1.071 | beats PPO |
| **PPO, realistic cost (IQM, 20 seeds)** | **0.933** | — |
| Buy-and-hold (first, zero turnover) | 0.970 | approx PPO |
| **A12 random-weight policy (IQM, 20 seeds)** | **0.915** | approx PPO |

**The headline finding.** PPO's 95% CI ([0.796,
1.032]) and A12's — a policy that allocates **uniformly
random long-only weights every bar** — ([0.821, 1.026])
**overlap almost completely.** The trained agent cannot be distinguished from noise on
this universe and window. This is exactly the failure mode the A12 sanity-floor gate
exists to catch, and it caught it.

Two forces plausibly explain this, both stated rather than resolved: (a) a small,
long-only, highly-correlated 5-name universe in a bull-market test window means *any*
long-only allocation inherits most of its return from shared market beta, compressing the
gap between "trained" and "random"; (b) PPO's own turnover (0.14-0.71 across seeds, wide)
suggests the policy has not converged to a stable, differentiated allocation rule.

## 5. The pre-registered three-gate verdict

`ARCHITECTURE.md` section 3.6: **Diebold-Mariano significant AND DSR > 0.95 AND
across-seed lower bound > 0.** "Across-seed lower bound" is not given an exact formula in
`ARCHITECTURE.md`; it is operationalised here as (2.5th-percentile bootstrap IQM Sharpe
across the 20 seeds) minus (buy-and-hold Sharpe) — does even a pessimistic view of the
agent's *typical* behaviour clear the passive control. That choice, and the alternative
(the single worst seed), are both reported.

| Gate | Best individual seed (seed 18, Sharpe 1.433) | Median seed (seed 1, Sharpe 0.965) |
|---|---|---|
| Diebold-Mariano vs buy-and-hold | p = 0.0019, PPO better — **PASS** | p = 0.220 — **FAIL** |
| DSR > 0.95 (n_trials = 84, honest count from the trial log) | 0.737 — **FAIL** | 0.353 — **FAIL** |
| Across-seed lower bound > 0 | -0.174 — **FAIL** | -0.174 — **FAIL** |
| **Verdict** | **FAIL** | **FAIL** |

Even hand-picking the single best-performing seed out of 20 — which the honest DSR
denominator (84 trials logged) exists specifically to penalise — clears only one of three
gates. The median seed, the honest representative of what training this agent actually
produces, clears none.

## 6. Conclusion, and what it means for the plan

**The price-only PPO control does not clear the project's own pre-registered bar on this
universe and window.** This is not a surprise result: `RESEARCH.md` named a realistic
net-of-cost Sharpe range of 0.0-0.6 with likely loss to EW-5 as an expected, acceptable
outcome before any of S2-S4 were built. The IQM Sharpes here (0.92-0.96) are numerically
higher than that range, but the *relative* comparison — indistinguishable from a random
long-only policy, behind momentum and equal-weight — is the same conclusion the range was
meant to anticipate: beating a passive benchmark in a small, correlated, bull-market
universe is hard, and this run did not do it.

**Consequence for `PLAN.md`'s dependency graph**: S4 blocks S6 (EKG) and S7 (sentiment) —
"no knowledge channel before a control to compare it to." That control has now been
built and has failed its own gate. The honest options, none of them "hide the result":

1. Report this as the S4/S9 finding and proceed to S6/S7 anyway, with the EKG/sentiment
   ablations run against this same (weak) PPO control — the pre-registered comparison
   (A4 vs A2) does not require the control to *win*, only to *exist* as the baseline the
   knowledge channels are measured against.
2. Spend a bounded amount of additional effort on the PPO control itself first — reward
   shaping, the L1 turnover penalty (`--l1-penalty`, implemented but unused at 0.0 in this
   run), longer training, or the TD3 challenger named in `ARCHITECTURE.md` section 3.3 —
   before accepting this as final.
3. Treat "indistinguishable from a random long-only policy" as itself the headline result
   and scope down: report it plainly, move to S9's fuller evaluation-universe protocol
   (random/momentum/volatility-selected universes, not just these 5 names) to see whether
   the finding is universe-specific.

This report does not choose between them; that is a planning decision for the next turn,
not a conclusion to be smuggled into a results section.

## 7. Honest gaps

- Only PPO was trained. TD3 (the designated challenger) and a reward with a non-zero L1
  turnover penalty are implemented in the environment (`l1_penalty` parameter) but not
  yet run.
- The "across-seed lower bound" operationalisation (section 5) is a documented choice, not
  a formula given in `ARCHITECTURE.md`. It should be confirmed or revised before being
  treated as load-bearing for a future kill-gate decision.
- Hyperparameters (`n_steps`, `batch_size`, `lr`) were not tuned; `PLAN.md` calls for an
  Optuna study on a validation fold, not yet run.
- Walk-forward folds with embargo (implemented in `rltrader.eval.harness`, tested for
  non-overlap in S3) were not used for this training run, which used a single train/test
  split for direct comparability with the FinRL replication. A walk-forward re-run is a
  natural next check before treating this verdict as final.
- The OOS window was read more than once during this exploratory analysis (pilot run,
  IS/OOS degradation check, DM methodology correction, final verdict computation) — all
  reads of a fixed set of already-trained, already-frozen policies, not re-fitting or
  seed-selection using OOS information, but `PLAN.md`'s "OOS fold touched once" discipline
  criterion is not met in the strict sense and is recorded here rather than silently met.
