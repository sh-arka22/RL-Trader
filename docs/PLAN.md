# PLAN.md — Staged build plan

**Companion docs:** [`RESEARCH.md`](RESEARCH.md) (evidence) · [`ARCHITECTURE.md`](ARCHITECTURE.md) (spec)
**Ordering rule fixed by the evidence:** *measurement apparatus first, agent second, knowledge layers
third, UI last.* Every later stage is judged by the harness built in Stage 3.

## Dependency graph

```
S0 research ──► S1 architecture ──► S2 data ──► S3 baselines+harness ──┬──► S4 RL core ──┬──► S6 EKG ──┐
                                                    │                  │                 │            ├──► S8 UI ──► S9 full eval ──► S10 iterate
                                                    └──► S3b cost model┘                 └──► S7 sentiment ──┘
```

Hard dependencies: **S3 blocks S4** (no agent before a harness). **S4 blocks S6 and S7** (no
knowledge channel before a control to compare it to). **S6 and S7 are parallel** and independent.
**S9 requires all arms** of the ablation matrix.

---

## S0 — Research ✅ COMPLETE (2026-09-14)

**Goal** Decide the architecture on verified evidence, not intuition.
**Exit criteria (all met)**
- Six topic dossiers, **508 citations verified in-session**, every failed check recorded.
- Architecture comparison table with metrics *and* their context (universe, window, costs, baseline).
- The RL / LLM / RLHF / hybrid question answered with evidence.
- 14 contradictions between the brief and the evidence documented.

**Committed** `docs/research/01..06`, `docs/RESEARCH.md`, `docs/RESEARCH_PROTOCOL.md`, `tools/research_tools.py`
**Supermemory** kickoff note, execution-plan note, 6 per-dossier milestone notes.

---

## S1 — Architecture spec ✅ COMPLETE (2026-09-14)

**Goal** Turn the evidence into a buildable spec.
**Exit criteria (all met)** component decisions with fallbacks; EKG schema and coupling decided;
ablation matrix A0–A13 pre-registered; three-venv isolation plan; non-goals stated.
**Committed** `docs/ARCHITECTURE.md`, `docs/PLAN.md`

---

## S2 — Data pipeline

**Goal** A point-in-time-correct store for NVDA, TSLA, AAPL, META, XOM.
**Entry** S1 complete.
**Work** loaders for yfinance / alpaca-py / Tiingo; Parquet store partitioned by ticker-year;
DuckDB views; NYSE calendar; two-provider reconciliation; `first_public_at` on every row.

**Exit criteria — concrete**
| Check | Threshold |
|---|---|
| Coverage 2015-01-01 → present, 5 tickers | **0 % missing** trading days |
| Cross-provider close reconciliation | ≤ **1 bp** divergence, else hard fail |
| Point-in-time query test | `as_of(D, 09:29)` returns **no** row with `first_public_at > D 09:29` |
| Corporate actions | Splits/dividends reproduce adjusted closes to ≤ 1 bp |
| Reproducibility | `make data` twice → identical checksums |

**Commit** `rltrader/data/`, `scripts/build_dataset.py`, `data/manifest.json` (checksums), tests.
**Supermemory** dataset manifest + provider quirks found.
**Risk** a free provider breaks (Stooq already did) → two providers wired from day one.

---

## S3 — Baseline + evaluation harness  ⛔ blocks everything downstream

**Goal** Build the referee before the players. This is the project's most valuable artefact.
**Entry** S2 exit met.
**Work** baselines (per-asset B&H, equal-weight, SPY, 60/40, pinned-SHA FinRL PPO replication);
metrics module; `rliable` bootstrap CIs; DSR + PBO over the trial log; Diebold-Mariano;
immutable trial log; alternative evaluation universes (random / momentum / volatility, **including
delisted names**); the three-level cost model.

**Exit criteria — concrete**
| Check | Threshold |
|---|---|
| Reproduce the computed passive bar | SPY Sharpe **0.82**, EW-5 **1.07** (2015-01-01 → 2026-09-11, rf=0) ± 0.02 |
| FinRL PPO replication | Runs from a pinned SHA; result reported even if it loses to DJIA (expected: SR ≈ 0.99 vs 1.32) |
| **Planted-oracle test** | A look-ahead oracle scores near-perfect Sharpe **and** the harness flags it |
| Trial log | Every evaluated config appended; DSR recomputes when the count changes |
| Cost levels | All metrics emit at low / base / stress |
| Determinism | Same seed + same data → bit-identical metrics |

**Commit** `rltrader/eval/`, `scripts/run_baselines.py`, `reports/baselines/`.
**Supermemory** the measured baseline numbers — these are the bar for the whole project.

---

## S3b — Execution realism cross-check *(parallel with S3)*

**Goal** Prove the custom env's fills and costs are not optimistic.
**Work** `nautilus_trader` in an isolated pinned venv; replay the same order sequence; compare.
**Exit** per-trade fill-price divergence ≤ **2 bps** median, ≤ 10 bps p95, on a 250-day replay;
documented explanation for every outlier.
**Commit** `rltrader/envs/validate_nautilus.py`, `reports/execution_realism.md`.

---

## S4 — Core RL agent ⚠️ DONE, VERDICT: FAIL (2026-09-15)

**Goal** A price-only PPO that is a *credible control*, not a winner.

**Result** 20 seeds x 3 cost levels trained and evaluated OOS (2022-2026). IQM Sharpe
0.92-0.96, but **statistically indistinguishable from the A12 random-weight sanity floor**
(0.915, overlapping CIs) and behind momentum/equal-weight. The pre-registered three-gate
verdict (DM + DSR>0.95 + seed lower bound>0) **FAILS even for the single best-of-20 seed**
(2 of 3 gates) and fails all three for the median seed. See `docs/reports/S4_AGENT.md`
for the full result, the A12 comparison, and the options for how to proceed (S6/S7 anyway
with this as the control; TD3/reward-shaping first; or scope to S9's fuller universe
protocol). This is a legitimate, plan-consistent, honestly-reported outcome, not a defect
in the pipeline — `RESEARCH.md` named this range of result as likely before S2 began.
**Entry** S3 exit met; S3b green.
**Work** custom ~400-LOC Gymnasium env (6-dim simplex action, t+1 open fills, ADV cap, turnover
penalty); `check_env` with seeding; PPO primary, TD3 challenger; walk-forward with embargo;
20 seeds; Optuna study on the validation fold only; W&B logging.

**Exit criteria — concrete**
| Check | Threshold |
|---|---|
| `gymnasium.check_env` | Passes **including seeding** (`reset(seed)` reproducible) |
| Planted-oracle test on our env | Detected |
| Seeds | **20** per config; IQM Sharpe with bootstrap CI reported |
| Sanity floor | Beats the random-weight policy (A12) with CI excluding 0 |
| Honest reporting | Result vs EW-5 and SPY published **whether or not it wins** |
| Cost sensitivity | Metrics at 3 cost levels; rank stability commented on |
| OOS discipline | The OOS fold touched **once** |

**Commit** `rltrader/envs/`, `rltrader/agents/`, `sweeps/`, `reports/s4_ppo.md`.
**Supermemory** A2 control numbers + the hyper-parameters that mattered.
**Realistic expectation** net-of-cost Sharpe **0.0–0.6**; losing to EW-5 (1.07) is a likely and
acceptable outcome, and must be reported as such.

---

## S5 — Decision gate: is there anything to build on?

**Goal** Stop cheap if the foundation is unsound.
**Entry** S4 complete.
**Pass** A2 beats A12 significantly **and** the harness passed A13 **and** S3b divergence is within
tolerance → continue to S6/S7.
**Fail** → do **not** add an EKG on top of a broken base. Publish the negative result, fix the base,
or re-scope to a pure benchmarking study. *This gate exists because adding knowledge layers to an
unsound controller is the single most common way these projects produce fiction.*

---

## S6 — EKG self-evolution loop *(parallel with S7)*

**Goal** Test the hypothesis that accumulated experience improves decisions.
**Entry** S5 passed.
**Work** DuckDB + Parquet bitemporal edge table (four timestamps); Episode/FeatureState/Lesson/
Regime/Claim schema; Bayesian shrinkage k=20, support floor n≥10; decay half-lives 90 d / 20 d / 3 d;
invalidate-never-delete with weekly index pruning (50k cap); full rebuild at each fold boundary;
graph embedding + **learned gate** with graph dropout and a null-graph pass.

**Exit criteria — concrete**
| Check | Threshold |
|---|---|
| Point-in-time integrity | No edge with `first_public_at` after the decision timestamp ever reaches the policy |
| Rebuild determinism | Fold-boundary rebuild from raw Episodes is byte-reproducible |
| **A4 vs A2** | Pre-registered, parameter-matched, 20 seeds, IQM Sharpe + probability of improvement |
| **Placebos A5/A6** | Shuffled edges and permuted identities must **not** reproduce the A4 gain. If they do, the gain is dimensionality, not knowledge |
| Gate behaviour | Learned gate value logged; if it collapses to ~0 the EKG is being ignored — report that |
| Growth | Node/edge counts and prune effects tracked over the run |

**Commit** `rltrader/ekg/`, `reports/s6_ekg_ablation.md`.
**Supermemory** the A4-vs-A2 verdict, including a null result.
**Explicitly acceptable outcome** "the EKG did not improve net-of-cost Sharpe" — with the placebo
evidence that proves the test was fair.

---

## S7 — Sentiment / ontology agent *(parallel with S6)*

**Goal** Test whether public chatter adds anything after costs.
**Entry** S5 passed.
**Work** StockTwits collector (200 req/hr, respect it); Reddit via PRAW **inference-only**;
X **counts-only** purchase (~\$5 budget cap); FinBERT-class encoder; cached, version-pinned LLM
extraction into `Claim` nodes; features = mention-volume z-score, attention spikes, **disagreement**,
event flags, novelty — all with `first_public_at`.

**Exit criteria — concrete**
| Check | Threshold |
|---|---|
| Licence compliance | Reddit content never enters a training set; documented in `reports/` |
| Budget | X spend ≤ **\$5**, counts only, receipt logged |
| Collector robustness | 30-day continuous run, gaps logged, no silent backfill |
| PIT correctness | No claim with `first_public_at` after the decision bar reaches the policy |
| **A7 vs A2** | Pre-registered, 20 seeds |
| **A8 XOM control** | If sentiment "helps" XOM (~0.5 msg/h) as much as NVDA (~173 msg/h), declare noise-fitting |
| Effect size honesty | Report Rank IC with Newey-West + FDR correction; prior art's best is **0.0143** and fails correction |

**Commit** `rltrader/sentiment/`, `reports/s7_sentiment_ablation.md`.
**Supermemory** the verdict + the live API state observed.
**Explicitly acceptable outcome** a null result. It is pre-registered as a deliverable.

---

## S8 — Live UI

**Goal** Make the system legible: the EKG growing, learning curves, P&L vs baseline — live.
**Entry** S6 and S7 complete (there must be a real graph and real curves to show).
**Work** FastAPI + one WebSocket per session + **20 Hz coalescing pump** with bounded per-client
queues; React + Vite; `@cosmos.gl/graph` (MIT) for the force graph; lightweight-charts for equity;
uPlot for learning curves; graph deltas not full snapshots; reconnect + resync.

**Exit criteria — concrete**
| Check | Threshold |
|---|---|
| Graph scale | **≥ 50k nodes** animating at ≥ 30 fps on the dev machine; degradation path documented |
| Backpressure | A slow client is dropped/coalesced, never bufferbloats the server; proven with a throttled-client test |
| Latency | Trade → on-screen update **< 500 ms** |
| Correctness | UI equity curve matches `eval/` metrics exactly — no separate maths in the frontend |
| Licences | `@cosmograph/*` (CC-BY-NC) **absent** from the lockfile; lightweight-charts attribution notice present |

**Commit** `rltrader/serve/`, `ui/`, `reports/s8_ui.md`, a recorded demo.

---

## S9 — Full evaluation vs baseline

**Goal** The honest verdict.
**Entry** all arms A0–A13 runnable.
**Work** run the full matrix, 20 seeds, 3 cost levels, on the traded universe **and** the
random / momentum / volatility universes **including delisted names**; DSR + PBO over the complete
trial log; Diebold-Mariano vs each baseline; across-seed lower bounds.

**Exit criteria — concrete**
| Check | Threshold |
|---|---|
| Three-gate verdict | DM significant **and** DSR > 0.95 **and** across-seed lower bound > 0 — otherwise reported as "did not beat baseline" |
| Trial count | DSR computed over the **true** number of configurations ever evaluated |
| Universe robustness | Result reported on all evaluation universes, not just the traded 5 |
| Reproducibility | A clean clone reproduces every headline number from `make reproduce` |
| Publication | `reports/FINAL_EVALUATION.md` states the outcome plainly, including failure |

**Commit** `reports/FINAL_EVALUATION.md`, frozen configs, seeds, trial log.
**Supermemory** final verdict + what would be tried next.

---

## S10 — Iteration

**Goal** One focused improvement cycle, chosen by evidence from S9.
Candidates, in the order the evidence favours: better cost/impact modelling → richer features →
TD3/recurrent policies → regime-conditioned retraining cadence → position sizing from EKG
confidence. **Not** "add another agent".
**Exit** a documented A/B against the S9 result using the same pre-registered gates.

---

## Budget and schedule

| Stage | Effort | Cash |
|---|---|---|
| S2 data | 2–3 d | \$0 |
| S3 + S3b harness | 4–6 d | \$0 |
| S4 RL core | 4–6 d | ~\$10–40 (CPU sandbox + sweeps) |
| S6 EKG | 5–7 d | ~\$10–30 |
| S7 sentiment | 4–6 d | ≤ \$5 (X counts) + ~\$10 compute |
| S8 UI | 4–6 d | \$0 |
| S9 evaluation | 2–3 d | ~\$20–60 (full matrix) |
| **Total** | **~5–6 weeks** | **≈ \$60–150** |

Compute is **CPU sandboxes** (\$0.05/core/hr), not GPUs. A GPU pod is justified only for a parallel
sweep burst. `prime-rl` and `verifiers` are LLM-token RL and cannot train this environment.

---

## Standing rules

1. **No result without its context** — universe, window, cost level, baseline, seed count.
2. **Every knowledge layer is ablatable and must win its gate**, with placebos.
3. **The OOS fold is touched once per stage.** Tuning happens on validation.
4. **Null results ship.** They are pre-registered deliverables, not failures.
5. **No notebooks in the critical path.** Every number comes from a scripted CLI entrypoint.
6. Commit at every stage exit; write a Supermemory note at every stage exit.
