# ARCHITECTURE.md — System specification

**Status:** decided, 2026-09-14 · **Evidence:** [`RESEARCH.md`](RESEARCH.md) and the six dossiers in
[`research/`](research/) · **Rule:** every claim here traces to a verified citation in a dossier.

---

## 1. The decision, stated plainly

The brief's starting hypothesis was *"an RL policy is the fast execution core; the EKG + sentiment
ontology form a slower memory/reasoning layer that reshapes the policy's inputs and reward."*

**Confirmed in part, amended in three places:**

| Brief | Decision | Driver |
|---|---|---|
| RL policy as fast core | **Kept.** PPO primary, TD3 challenger | No architecture family has a portable edge; RL is the cheapest, most reproducible, most ablatable controller |
| EKG reshapes inputs **and reward** | **Inputs, retrieval context and position sizing only.** No reward shaping | Only potential-based shaping is policy-invariant; a wrong KG fact would become the objective |
| EKG as an always-on memory layer | **Ablatable channel behind an A/B gate** | No prior art shows self-evolving memory improving trading outcomes; HMM regime features took PPO from Sharpe 1.03 → −0.26 |
| LLM as reasoning layer | **Offline, cached, outside the decision loop** | FINSABER: buy-and-hold beat LLM agents and PPO in 4/4 bias-mitigated universes. FinRL-DeepSeek: LLM infusion degrades plain PPO at every strength |

The controller is therefore **RL**, the knowledge layer is **optional and measured**, and the
system's primary output is **an honest comparison against buy-and-hold**, not a promise of alpha.

---

## 2. System overview

```
                    ┌──────────────── OFFLINE / SLOW LOOP (daily, batch) ───────────────┐
                    │                                                                   │
  free data ──►  data layer  ──►  feature builder  ──┐                                  │
  (yfinance,     (Parquet,        (returns, vol,     │                                  │
   alpaca,        point-in-time    turbulence,       │                                  │
   Tiingo)        bitemporal)      technicals)       │                                  │
                                                     │                                  │
  StockTwits ─┐                                      │                                  │
  Reddit     ─┼─► sentiment/ontology agent ──► claims/events ──► EKG (DuckDB+Parquet)    │
  X counts   ─┘   (FinBERT + LLM extractor,        (Episodes, FeatureStates,             │
                   cached, versioned, PIT)          Lessons, Regimes, Claims)            │
                                                     │         │                        │
                    └────────────────────────────────┼─────────┼────────────────────────┘
                                                     │         │ graph embedding (gated)
                                                     ▼         ▼
                       ┌──────────────── FAST LOOP (per step) ─────────────────┐
                       │  Gymnasium env  ◄──►  PPO / TD3 policy (SB3)          │
                       │  obs = prices+features [+ EKG vector · gate]          │
                       │  action = 6-dim simplex (5 weights + cash)            │
                       │  fill at t+1 open, cost model, ADV cap                │
                       └───────────────────────────────────────────────────────┘
                                                     │
                       outcomes ─────────────────────┘  (write-back to EKG: Episode → Lesson)
                                                     │
                       ┌─────────────────────────────▼─────────────────────────┐
                       │ baseline harness: B&H, EW, SPY, FinRL-PPO replication │
                       │ 3 cost levels · 20 seeds · IQM Sharpe · DSR/PBO · DM  │
                       └───────────────────────────────────────────────────────┘
                                                     │
                       W&B (logging/lineage) · Optuna (search) · React+FastAPI UI
```

Two loops, deliberately decoupled: the **fast loop** is a plain Gymnasium/SB3 RL system that runs
without any of the knowledge machinery; the **slow loop** writes into the EKG and may — if it
passes its gate — enrich the fast loop's observation.

---

## 3. Components

### 3.1 Data layer

| Item | Decision |
|---|---|
| Providers | **yfinance** (primary OHLCV), **alpaca-py** free tier (cross-check + intraday if needed), **Tiingo** free (replaces the dead Stooq) |
| Universe | **NVDA, TSLA, AAPL, META, XOM** |
| Storage | Parquet, partitioned by ticker/year; DuckDB as the query engine |
| Point-in-time | Every row carries `recorded_at`; adjusted prices are recomputed, never back-patched in place |
| Integrity | Two independent providers reconciled; checksum + row-count manifest per pull; a divergence > 1 bp is a hard failure |
| Calendar | `exchange_calendars` NYSE; no synthetic bars, holidays dropped explicitly |

**Non-negotiable:** the loader must be able to answer *"what was knowable at 09:29 on date D"* for
any D. Every look-ahead defect in the literature traces back to a loader that could not.

### 3.2 Market environment (custom, ~400 LOC)

Replaces TensorTrade (see `RESEARCH.md` §3, item 1).

| Element | Specification |
|---|---|
| API | Gymnasium 1.3 — `reset(seed) -> (obs, info)`, `step() -> 5-tuple`, passes `check_env` **with seeding** |
| Observation | Window of per-asset returns, realised vol, volume z-score, technicals, turbulence index, current weights, cash — plus optional `ekg_vec` (see §3.4) |
| Action | **6-dim simplex**: target weights for 5 assets + cash, long-only, softmax-projected |
| Execution | Order at close of `t`, **filled at the open of `t+1`**. Never same-bar |
| Costs | commission 0 bps (retail 2026), half-spread **1–2 bps**, square-root market impact, trade size capped at **1 % of ADV** |
| Cost levels | Every experiment runs at **low / base / stress** cost settings; a result that exists only at "low" is not a result |
| Reward | `net log return − λ · L1(Δweights)` (turnover penalty). λ tuned once, then frozen |
| Risk rails | Turbulence-triggered de-risking (FinRL concept), max position 40 %, cash floor |
| Safety test | A **planted look-ahead oracle** must score a near-perfect Sharpe; if it does not, the env is wired wrong |

### 3.3 Policy layer

- **PPO** (SB3 2.9) as primary; **TD3** (SB3) as challenger. **SAC dropped** — least cost-robust.
- MLP policy, modest width. This is a small problem; capacity is not the constraint.
- **Walk-forward** protocol: expanding train window, embargoed gap, fixed OOS fold; purged CV
  where folds could touch.
- **20 seeds per configuration.** A single-seed result is not reported.
- Retrain-from-scratch at each fold boundary stays a **live competitor** to continual updating —
  PPO has been observed to collapse to zero return after 20M steps even on a stationary task.

### 3.4 EKG — the self-evolving knowledge graph

Scoped to the **experience** side. With only 5 instruments an entity graph is too small to carry
signal, so the graph learns from what the agent did, not from who owns whom.

**Node types:** `Episode` (a trade or holding period), `FeatureState` (discretised market context),
`Regime`, `Lesson` (a consolidated statistical claim), `Claim` (sentiment/event assertion),
`Instrument`.

**Edge table is bitemporal — four timestamps:**

| Field | Meaning |
|---|---|
| `valid_from` / `valid_to` | when the fact was true in the world |
| `first_public_at` | when it became knowable — the look-ahead guard |
| `recorded_at` / `superseded_at` | when the system believed it |

**Update rule**
- Write-back after every episode: `Episode → FeatureState → outcome`.
- Consolidation into `Lesson` uses **Bayesian shrinkage (k = 20)** with a **support floor of n ≥ 10**.
- **Decay half-lives:** Lesson 90 d, correlation 20 d, sentiment Claim 3 d.
- **Invalidate, never delete.** Weekly pruning touches only the retrieval index (50k cap); the full
  history stays in Parquet as an audit log.
- **Full rebuild from raw Episodes at every fold boundary**, to stop error accumulation.

**Storage:** DuckDB + Parquet edge table with a NetworkX point-in-time view.
Kuzu is **out** — `kuzudb/kuzu` is archived read-only since 2025-10-10. Neo4j/Memgraph are fallbacks.

**Coupling to the policy — the decisive design choice**

| Option | Decision |
|---|---|
| **Gated state augmentation** — graph embedding enters the observation through a learned gate, with graph dropout and a null-graph pass | **PRIMARY** |
| Retrieval-conditioned policy | **FALLBACK** |
| Reward shaping | **REJECTED** — no policy-invariance guarantee |
| Action masking | **Hard constraints only** (cash, position limits, market hours, borrow). Graph beliefs may scale position size; they may never veto a trade |

**Self-improvement pattern:** copy *verify → consolidate → rollback* (the Prime Agent `/refine`
loop, which edits prompts/memories/skills with rollback snapshots), **not** online weight learning
and **not** decentralised training. Prime Intellect's own published loop is verifiable environments
feeding async batch RL; INTELLECT-3 trained on a centralised 512×H200 cluster.

### 3.5 Sentiment / ontology agent

Re-scoped from "sentiment predicts returns" to "attention and disagreement are measurable".

| Source | Status (verified 2026-09-14) | Use |
|---|---|---|
| **StockTwits** public ticker stream | Open, no key, 200 req/hr/IP; only ~30 % of messages self-labelled | Primary live signal |
| **Reddit** (PRAW) | Free, rate-limited; **Terms §2.4 forbid training an AI model on user content** | **Inference only**, never a training corpus |
| **X / Twitter** | Pay-per-use; **"Counts: All" bills \$0.010 per _request_** | Buy **counts only** (≈ \$2 for a full-archive 5-ticker daily volume series). Never post text |
| Kaggle labelled tweets | Covers only 2021-09 → 2022-09 | Encoder validation only — **a labelled backtest through 2026 is impossible on free data** |

**Features produced** (not raw "sentiment score"): mention volume z-score, attention spikes,
**disagreement** (bull/bear label dispersion), event flags, novelty. All timestamped with
`first_public_at`.

**Models:** FinBERT-class encoder for labels; an LLM used **offline and cached** for triple/event
extraction into `Claim` nodes. Pinned model version, cached outputs, deterministic replay.

**Communication:** shared graph store (the EKG) — **not** a message bus. At 5 tickers and daily
frequency the bus adds operational cost and no throughput benefit, and the graph already provides
the audit trail.

**Built-in control:** XOM has ~0.5 messages/hour against NVDA's ~173. If the sentiment channel
"helps" XOM as much as NVDA, the channel is fitting noise.

### 3.6 Baseline harness — the project's real product

Built **first**, before any agent.

- Baselines: **buy-and-hold per asset**, **equal-weight 5**, **SPY**, **60/40**, and a
  **pinned-SHA FinRL PPO replication**.
- Metrics: annualised return, **IQM net-of-cost Sharpe** (pre-registered primary), Sortino, max
  drawdown, Calmar, turnover, hit rate, exposure.
- Statistics: stratified bootstrap CIs (`rliable`), probability of improvement, stationary
  bootstrap, **Deflated Sharpe Ratio** and **PBO** computed over the *full* logged trial set.
- **Three-gate verdict:** Diebold-Mariano significant **and** DSR > 0.95 **and** across-seed lower
  bound > 0. Anything else is reported as "did not beat baseline".
- **Trial log:** every configuration ever evaluated is appended to an immutable trial table. DSR is
  meaningless without the true trial count.
- Evaluation universes: the traded 5 **plus** random / momentum / volatility-selected universes
  **including delisted names**, because hand-picked survivor baskets inflate results.

### 3.7 UI

| Layer | Choice | Why |
|---|---|---|
| Graph | **`@cosmos.gl/graph`** (MIT), GPU force layout | Graphology FA2 manages 1.61 layout it/s at 100k nodes; cosmos.gl does 39.7. **Avoid `@cosmograph/cosmos` — CC-BY-NC-4.0** |
| Graph fallback | Sigma.js 3 + Graphology, frozen layout, ≤ 5k display subgraph | Works when WebGL2 is unavailable |
| Transport | One **WebSocket** per session + server-side **20 Hz coalescing frame pump** + bounded per-client queue | Broadcast without backpressure bufferbloats. `encode/broadcaster` is **archived** — do not use |
| Equity curves | TradingView **lightweight-charts 5.2.1** (Apache-2.0 **with mandatory attribution**) | Purpose-built; attribution notice is required |
| Learning curves | **uPlot** (MIT) | Many dense series, tiny bundle |
| Shell | **React + Vite + FastAPI** | No Python dashboard framework can own a continuously animating 10k+ node WebGL scene without writing the same JS component anyway |

### 3.8 Experiment tracking, HPO, compute

- **Optuna 5** for search and pruning (exposed samplers, resumable RDB study) + **W&B** for logging
  and lineage. W&B Sweeps only if zero-infra matters. Apply for the free **academic Pro** tier —
  the free tier is 5 GB/month.
- HPO discipline: tune on the **validation** fold only; the OOS fold is touched once per stage.
  Every trial lands in the trial log for DSR/PBO.
- **Compute: CPU, not GPU.** A 5-asset daily-bar MLP PPO is tiny; kernel-launch overhead dominates.
  Prime Intellect **sandbox** at \$0.05/core/hr + \$0.01/GB-RAM/hr; a cheap GPU pod only for a
  parallel sweep burst. **`prime-rl` and `verifiers` are LLM-token RL and cannot train this env.**
- Reproducibility: pinned lockfiles per venv, pinned data snapshots, seeds recorded, and no hosted
  LLM inside any loop that must replay bit-exactly.

---

## 4. Environment isolation

Dependency conflicts are a real constraint here, so the project runs **three venvs** with Parquet
as the only interface between them:

| venv | Contents | Reason |
|---|---|---|
| `core` | Gymnasium 1.3, SB3 2.9, torch, duckdb, pandas 2.x | The training stack |
| `metrics` | vectorbt / quantstats | Requires numpy ≥ 2.4 / pandas ≥ 3 — cannot share with `core` |
| `validate` | `nautilus_trader` (pinned rc) | Mid v1 → v2 Rust rewrite; must stay isolated |

---

## 5. Ablation matrix — what gets measured

| Arm | Description | Question answered |
|---|---|---|
| A0 | Buy-and-hold / equal-weight / SPY | The bar |
| A1 | FinRL PPO replication (pinned SHA) | Did we reproduce the published baseline? |
| A2 | **Price-only PPO** (our env) | The control everything is compared to |
| A3 | A2 + TD3 | Algorithm sensitivity |
| A4 | A2 + EKG channel, gate open | Does the EKG earn its place? |
| A5 | A4 with **shuffled edges** | Placebo — is it the graph or just extra dimensions? |
| A6 | A4 with **permuted identities** | Placebo — is it learning identity leakage? |
| A7 | A2 + sentiment features | Does chatter add anything? |
| A8 | A7 restricted to **XOM** | Sentiment-null control |
| A9 | Full system | The headline |
| A10 | Full system, **stress costs** | Does the edge survive friction? |
| A11 | Retrain-from-scratch per fold | Is continual updating even better than restarting? |
| A12 | Random-weight policy | Sanity floor |
| A13 | Planted look-ahead oracle | Does the harness detect cheating? |

20 seeds per arm. Pre-registered primary comparison: **A4 vs A2**, parameter-matched.

---

## 6. Repository layout (target)

```
rltrader/
  data/        loaders, point-in-time store, integrity checks
  envs/        gymnasium env, cost model, fill model
  agents/      ppo.py, td3.py, policies, wrappers
  ekg/         schema.py, store.py (duckdb), consolidate.py, embed.py, gate.py
  sentiment/   collectors (stocktwits, reddit, x_counts), encoder, extractor, features
  eval/        baselines.py, metrics.py, stats.py (DSR/PBO/DM), trial_log.py, universes.py
  serve/       fastapi app, websocket pump, state broadcaster
  ui/          react + vite (cosmos.gl, lightweight-charts, uplot)
  sweeps/      optuna studies, wandb config
scripts/       reproducible CLI entrypoints (no notebooks in the critical path)
docs/          RESEARCH.md, ARCHITECTURE.md, PLAN.md, research/
```

---

## 7. Non-goals

- Intraday or HFT execution. Daily bars only.
- Live capital. Paper trading at most.
- Beating the market as a success condition. **Measuring honestly is the success condition.**
- LLMs inside the decision loop.
- Decentralised or distributed training. One CPU box is the right size.
