# RESEARCH.md — Evidence synthesis and the architecture decision

**Phase:** research (complete) · **Date:** 2026-09-14 · **Window surveyed:** 2022 → 2026-09
**Protocol:** [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) — no citation enters this repo unless it was
fetched and checked live in the session that wrote it.

| Dossier | Topic | Size | Checks verified in-session |
|---|---|---|---|
| [`research/01_rl_trading.md`](research/01_rl_trading.md) | Deep RL for trading + the critique literature | 66 KB | 76 (36 arXiv, 28 URL, 12 repo) |
| [`research/02_llm_agents.md`](research/02_llm_agents.md) | LLM trading agents, RLHF/DPO, LLM+RL hybrids | 76 KB | 108 (58 arXiv, 27 repo, 17 URL, 6 pricing) |
| [`research/03_ekg_continual.md`](research/03_ekg_continual.md) | Financial KGs, self-evolving memory, continual RL, Prime Intellect | 85 KB | 124 (77 arXiv, 15 repo, 32 URL) |
| [`research/04_data_sentiment.md`](research/04_data_sentiment.md) | Free-data audit, sentiment, ontology, the 5 stocks | 63 KB | ~86 rows (10 arXiv, 12 repo, 10 HF, 6 Kaggle, ~120 probes) |
| [`research/05_oss_landscape.md`](research/05_oss_landscape.md) | Code-level OSS audit + install tests | 51 KB | 41 verified, 3 rejected |
| [`research/06_ui_mlops.md`](research/06_ui_mlops.md) | Graph UI, transport, charts, W&B/HPO, compute | 74 KB | 73 verified, 6 marked UNVERIFIED |

Raw generator output is preserved under [`research/_raw/`](research/_raw/) and is **not** evidence —
only the verification logs in each dossier are.

---

## 1. The decision

> **A hybrid, but not the hybrid the brief proposed.**
> A **price-only PPO policy is the execution core and the thing to beat**. The EKG and the
> sentiment/ontology agent are **optional, ablatable input channels** that must each win an A/B
> gate against that core before they are switched on. The LLM sits **outside the decision loop**,
> offline, as a cached feature/knowledge extractor.

Confidence: **moderate** for "RL core, LLM out of the loop" (multiple independent replications
agree). **Low** for "EKG improves trading" — that is an open hypothesis this project must
falsify, not an assumption it may build on.

### 1.1 Why, in one table

| Option | Best verified evidence *for* | Best verified evidence *against* | Verdict |
|---|---|---|---|
| **A. Pure deep RL** (PPO/SAC/A2C/TD3) | FinRL Dow-30 ensemble Sharpe **2.81** vs DJIA **2.02** | …but that result **omits transaction costs**; FinRL's own PPO baseline scores SR **0.99 vs DJIA 1.32** in the FinRL-Meta reproduction; FINSABER finds buy-and-hold beats PPO on Sharpe in **4/4** bias-mitigated universes | **Core, as the control** — not because it is proven to win |
| **B. LLM agents** (prompted / multi-agent debate) | TradingAgents reports Sharpe **8.21** | …on a **3-month, 3-ticker** window with **zero costs**. StockBench: on a contamination-free window most frontier LLMs fail to beat equal-weight buy-and-hold (GPT-5 below it). Profit Mirage: **51–62 %** Sharpe decay past the backbone knowledge cutoff | **Rejected for the decision loop** |
| **C. RLHF / DPO / GRPO finance models** | FinDPO: **458.97 %** cumulative, Sharpe **2.03** at 5 bps vs S&P 500 Sharpe **0.62** | It is a *sentiment classifier* feeding a separate portfolio constructor — not a control architecture. No paper compares DPO against SFT on P&L. Trading-R1's own ablation: SFT-only beats RL-only on 4 of 6 tickers | **Not applicable as an architecture** |
| **D. LLM + RL hybrid** | The one direct head-to-head found: **Sharpe 1.10 (LLM+RL) vs 0.64 (RL-only) vs 1.03 (LLM-only)** | …6 stocks, monthly generation, **no costs, no buy-and-hold**. FinRL-DeepSeek — the most relevant ablation — shows LLM infusion **degrades plain PPO at every strength tested**, helping only risk-sensitive CPPO. In FinRL Contest 2025's own table, PPO-DeepSeek returned **94.43 %** vs plain PPO **204.51 %** | **Adopted only in weak form**: offline, cached, ablated |

### 1.2 The single most important finding

**No published study matches this project's constraints** — 5 liquid US large caps, daily bars,
free data, modelled costs, benchmarked against buy-and-hold *and* the index. Every headline number
above fails at least one of those conditions. The project therefore cannot inherit a result; it
must **measure its own**, and the measurement apparatus matters more than the agent.

---

## 2. The bar to beat (computed, not quoted)

`01_rl_trading.md` computed the passive benchmark first-party from yfinance, 2015-01-01 → 2026-09-11, rf = 0:

| Benchmark | Sharpe |
|---|---|
| SPY buy-and-hold | **0.82** |
| Equal-weight 5 large caps (AAPL/MSFT/JNJ/JPM/XOM) | **1.07** |

> **CORRECTION (S3, 2026-09-14).** The 1.07 above is for the *retired* basket
> AAPL/MSFT/JNJ/JPM/XOM, daily-rebalanced (reproduced at 1.073). The universe actually
> traded is NVDA/TSLA/AAPL/META/XOM, whose passive bar is **1.201** buy-and-hold /
> **1.234** daily-rebalanced over the same window. The hurdle is ~0.13 Sharpe higher than
> planned. The bar is now computed from the store by `tests/test_baselines.py` and pinned
> in `rltrader/eval/bar.py`; do not copy it from prose. See that module for the full
> convention table — the spread between conventions is 0.35 Sharpe, larger than most
> improvements claimed in the literature surveyed here.
| Realistic net-of-cost range for an RL agent on 5 daily-bar large caps | **0.0 – 0.6** |

Read that honestly: **the passive benchmark is expected to win.** A credible outcome for this
project is a well-measured agent that loses to equal-weight buy-and-hold, reported as such. The
FinRL-Meta reproduction shows the standard pattern — you can beat the *published RL baseline* and
still lose to the *index*.

---

## 3. Corrections to the master brief

Fourteen items where verified evidence contradicts `PRIME_INTELLECT_PROMPT.md`. Each is carried
into [`ARCHITECTURE.md`](ARCHITECTURE.md).

| # | Brief says | Evidence says | Action |
|---|---|---|---|
| 1 | Build on **TensorTrade** | v1.0.4 (2026-02-06) is alive but: slippage model is **dead code**, **no partial fills**, **same-bar fills** (observe close 100.0 → fill 100.0), fails `gymnasium.check_env` (seeds do nothing, issue open since 2022), and pins Python 3.12 + numpy 1.26.4 + ~600 MB unused TensorFlow. Install on 3.13 is unsatisfiable | **Drop it.** Write a ~400-LOC Gymnasium env (skeleton from gym-anytrading MIT, OMS concepts from TensorTrade Apache-2.0) |
| 2 | **PPO/SAC** execution core | SAC is the *least* cost-robust agent found (margin Sharpe −0.5 → −1.2 under impact modelling); algorithm rankings flip when only the cost model changes | **PPO primary, TD3 challenger.** Drop SAC |
| 3 | EKG reshapes policy **inputs and reward** | Only potential-based shaping is policy-invariant; an arbitrary KG bonus makes a wrong fact the objective | Amend to **inputs, retrieval context and position sizing**. No reward shaping |
| 4 | EKG makes decisions better "the more it trades" | No source shows self-evolving agent memory improving *trading* outcomes — all gains are QA benchmarks. HMM regime features moved PPO from Sharpe **1.03 → −0.26**. A no-KG LSTM beat 6 of 7 GCN variants | Treat as a **hypothesis to falsify** behind an A/B gate |
| 5 | Copy **Prime Intellect's** continual self-improvement | They do **not** do online weight learning. Published loop = verifiable environments → async batch RL. Protocol repo is **archived**; INTELLECT-3 trained on a **centralised** 512×H200 cluster | Copy **verify → consolidate → rollback** (the Prime Agent `/refine` pattern), not decentralised training |
| 6 | Sentiment agent informs trading decisions | 90M-message StockTwits study: no unconditional next-day predictability. WSB: no risk-adjusted alpha even at \$0 commission. Best verified 1-day Rank IC **0.0143**, surviving neither Newey-West nor FDR | Re-scope to **attention / event / disagreement** features; a **null result is a valid deliverable** |
| 7 | Use **Stooq** for free data | CSV endpoint now behind a JS proof-of-work; solved the PoW and still got "Access denied". `pandas-datareader` raises `NotImplementedError` | Replace with **Tiingo** free tier |
| 8 | X/Twitter is unaffordable | Directionally right, materially wrong: cap is **3M/month**, full-archive search back to 2006 is now pay-per-use, and **"Counts: All" bills \$0.010 per _request_** — a full-archive daily cashtag volume series for 5 tickers costs **≈ \$2** | **Buy counts (~\$5). Never buy post text** (\$1,250/yr for 5 tickers) |
| 9 | Reddit via PRAW as a data source | Data API Terms §2.4 forbid using user content to train an ML/AI model without rightsholder permission | Reddit is **inference-only**, never a fine-tuning corpus |
| 10 | Kaggle tweet set for historical training | The referenced corpus covers only **2021-09 → 2022-09**; no free labelled corpus overlaps 2024–2026 | A "historical sentiment backtest ending 2026" is **not possible on free data**. Say so |
| 11 | Evaluate on 5 hand-picked stocks | FINSABER shows hand-picked large-cap universes are exactly the selection pattern that inflates results | **Trade 5, evaluate on random / momentum / volatility universes including delisted names** |
| 12 | **D3-force or Three.js** for the EKG UI | d3-force is CPU velocity-Verlet; last npm release **3.0.0 (2021-06-05)**. Graphology FA2 does **1.61 layout it/s at 100k nodes** vs **39.7 it/s** for GPU cosmos.gl | **`@cosmos.gl/graph` (MIT)**. Avoid `@cosmograph/cosmos` — **CC-BY-NC-4.0**, non-commercial |
| 13 | Train on **cloud GPUs** | A 5-asset daily-bar MLP PPO is a tiny workload; kernel-launch overhead dominates. `prime-rl`/`verifiers` are LLM-token RL and do not accept a Gymnasium continuous-control env | **CPU sandbox** (\$0.05/core/hr); a cheap GPU pod only for parallel sweep bursts |
| 14 | **W&B** for all hyper-parameter tuning | W&B does not document its Bayesian surrogate; free tier is 5 GB/month | **Optuna for search/pruning + W&B for logging and lineage**. Apply for the free academic Pro tier |

---

## 4. Why measurement is the hard part

Three independent findings converge on the same warning.

1. **Deflation does not catch leakage.** A planted look-ahead oracle scored design Sharpe 34.7,
   eval Sharpe 51.5, and a **Deflated Sharpe Ratio of 1.00** — the statistic certified a cheat.
2. **On a clean panel, almost nothing survives.** On a 453-stock point-in-time US panel with full
   costs, *every* active strategy was rejected (classic factors DSR ≤ 0.32, PBO 0.83; best of 102
   searched strategies: design SR 1.69 → eval SR 0.18). Only equal-weight buy-and-hold was certified.
3. **Costs decide the winner.** Changing *only* the cost model (flat 10 bps → Almgren-Chriss)
   flips the algorithm ranking on NASDAQ-100.

Consequences adopted as project rules:

- Fills at **t+1 open**, never same-bar. 1–2 bps half-spread, square-root impact, 1 %-of-ADV cap.
- **Every result reported at three cost levels.**
- Pre-registered metric (**IQM net-of-cost Sharpe**), pre-registered comparison, 20 seeds,
  stratified-bootstrap CIs, shuffled-edge and permuted-identity placebos.
- A trial log of every configuration ever evaluated, so DSR/PBO are computed over the **true**
  candidate count.
- Three-gate verdict: Diebold-Mariano **and** DSR > 0.95 **and** across-seed lower bound > 0.

---

## 5. Component decisions carried forward

| Layer | Decision | Fallback |
|---|---|---|
| Environment | Custom ~400-LOC Gymnasium env; 6-dim simplex action (5 weights + cash), long-only, volume-clipped | gym-anytrading skeleton |
| Algorithms | **PPO** primary, **TD3** challenger, SB3 2.9 / sb3-contrib 2.9, Gymnasium 1.3 | CleanRL reference impls |
| Reward | Net log return − L1 turnover penalty | Differential Sharpe |
| Execution realism | `nautilus_trader` pinned in an isolated venv as the **validation** engine (fill + latency + fee + queue models) | Cost-model unit tests only |
| Metrics | vectorbt / quantstats in a separate venv, Parquet interface | Hand-rolled |
| EKG | DuckDB + Parquet **bitemporal** edge table + NetworkX point-in-time view; gated state augmentation | Retrieval-conditioned policy; Neo4j/Memgraph store |
| Sentiment | StockTwits (open, 200 req/hr) + Reddit inference-only + X *counts* only; FinBERT-class encoder | Attention/volume features with no text model |
| Data | yfinance + alpaca-py + Tiingo, qlib-style point-in-time discipline | EODHD free |
| Universe | **NVDA, TSLA, AAPL, META, XOM** (mean ρ 0.219, ADV \$85.2 bn/day, 0 % missing 2018-2026; XOM = diversifier **and** sentiment-null control) | — |
| UI | React + Vite + FastAPI, one WebSocket + 20 Hz coalescing pump, `@cosmos.gl/graph`, lightweight-charts (equity) + uPlot (learning curves) | Sigma.js 3 frozen layout; Panel shell |
| HPO | Optuna 5 search/pruning + W&B logging | W&B Sweeps bayes+hyperband |
| Compute | Prime Intellect **CPU sandbox**; GPU pod only for sweep bursts | Local CPU |

---

## 6. Open risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| The agent loses to buy-and-hold | The base rate says it will | Pre-register it as an acceptable, reportable outcome |
| The EKG adds nothing | No prior art shows a trading gain | A/B gate; ship the ablation, not the belief |
| Sentiment adds nothing | Effect sizes fail significance correction | Null result is a deliverable; XOM is the built-in control |
| Custom env has a subtle look-ahead bug | It is the single most common failure in this field | Planted-oracle test + nautilus cross-check + t+1 fills |
| Free data breaks mid-project | Stooq already did | Two independent providers, cached Parquet, checksums |
| Sweep cost blowout | An LLM-in-the-loop sweep costs \$3.5k–\$248k | LLM offline and cached; Optuna pruning; CPU only |

---

## 7. What the next phase does

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the system spec and [`PLAN.md`](PLAN.md) for the staged
build. The ordering rule is fixed by the evidence above: **baseline harness and cost model first,
agent second, EKG third, sentiment fourth, UI last** — because every later component is judged by
the apparatus built in the first stage.
