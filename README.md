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

Research phase in progress (started 2026-09-14). See `docs/research/` for topic dossiers.
