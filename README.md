# RL-Trader

Multi-agent automated trading system — **research phase**.

Goal: trade 5 US equities profitably, benchmarked against industry-standard baselines
(buy-and-hold, S&P 500, and a published RL baseline such as FinRL's PPO agent).

## Planned components

| # | Component | Summary |
|---|---|---|
| 1 | Trading agents | Execution/environment layer on [tensortrade-org/tensortrade](https://github.com/tensortrade-org/tensortrade) (Gymnasium-compatible) |
| 2 | EKG | Self-evolving knowledge graph: trades, regimes, features, outcomes as nodes/edges; updates continually |
| 3 | Sentiment/ontology agent | Ontological KG from social/financial chatter (Kaggle + Reddit PRAW + StockTwits), talks to the trading agents |
| 4 | Baseline harness | Every agent reported against baselines, every run |
| 5 | W&B | All hyper-parameter tuning via sweeps |
| 6 | Live UI | Animated force-directed EKG + learning curves + P&L vs baseline |

## Repository layout

```
docs/
  RESEARCH.md          # synthesis: architecture comparison table + evidence
  ARCHITECTURE.md      # the RL / LLM / RLHF / hybrid decision + system spec
  PLAN.md              # staged build plan with entry/exit criteria
  RESEARCH_PROTOCOL.md # citation-verification rules for this repo
  research/            # per-topic research dossiers (raw material for RESEARCH.md)
tools/
  research_tools.py    # Parallel.ai search + deep research, arXiv/URL/GitHub verification, Supermemory
```

## Status

**Research phase complete (2026-09-14).** 508 citations verified live; 415 KB of dossiers.

Start here: **[`docs/RESEARCH.md`](docs/RESEARCH.md)** (evidence + the decision) →
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** (system spec) →
**[`docs/PLAN.md`](docs/PLAN.md)** (staged build).

### The decision

A hybrid — but not the one the brief proposed. **A price-only PPO policy is the execution core and
the control to beat.** The EKG and the sentiment agent are **ablatable input channels** that must
each win an A/B gate against that control. The **LLM stays outside the decision loop**, offline, as
a cached feature extractor.

### What the evidence changed

| Brief said | Evidence said | Result |
|---|---|---|
| Build on TensorTrade | Dead slippage code, no partial fills, same-bar fills, `check_env` seeding failure, Python 3.12 + numpy 1.26.4 cage | Custom ~400-LOC Gymnasium env |
| PPO/SAC core | SAC is the least cost-robust agent tested | PPO primary, **TD3** challenger |
| EKG reshapes inputs **and reward** | Only potential-based shaping is policy-invariant | Inputs, retrieval and position sizing only |
| Stooq for free data | CSV endpoint now behind a proof-of-work wall | Tiingo free tier |
| X/Twitter unaffordable | Full-archive **counts** bill $0.010/request | ~$2 for 5 tickers — counts only |
| Train on cloud GPUs | 5 assets on daily bars is a tiny workload | CPU sandboxes |

Full list: 14 corrections in [`docs/RESEARCH.md`](docs/RESEARCH.md) §3.

### The bar to beat (computed first-party, 2015-01-01 → 2026-09-11, rf=0)

| Benchmark | Sharpe |
|---|---|
| SPY buy-and-hold | 0.82 |
| Equal-weight 5 large caps | 1.07 |
| Realistic net-of-cost range for the agent | 0.0 – 0.6 |

Losing to buy-and-hold is a likely outcome and a **reportable** one. Null results ship.

### Universe

**NVDA, TSLA, AAPL, META, XOM** — mean pairwise correlation 0.219, combined ADV $85.2 bn/day, 0 %
missing data 2018–2026. XOM is both the sector diversifier and the sentiment-null control.
