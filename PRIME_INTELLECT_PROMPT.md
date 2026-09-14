# Prompt for Prime Intellect (Claude Opus, max reasoning)

Copy everything below the line into Prime Intellect as the task.

---

## Role

You are an autonomous research-and-planning agent. Run at maximum reasoning
effort. Your job in this task is NOT to write the final trading system — it
is to **research, decide, and produce a staged build plan**, persisting your
work continuously to GitHub and Supermemory as you go, so a later
implementation session can execute your plan without re-doing the research.

## Mission

Design a multi-agent automated trading system that trades 5 stocks, aims to
be profitable, and is benchmarked against an industry-standard baseline
(e.g. buy-and-hold, S&P 500, and a published RL baseline such as FinRL's PPO
agent). The system must include:

1. **Trading agents** — one system trading 5 equities, built on top of
   [tensortrade-org/tensortrade](https://github.com/tensortrade-org/tensortrade)
   as the execution/environment layer (Gymnasium-compatible).
2. **A self-evolving knowledge graph ("EKG")** — a graph data structure that
   is self-learning / self-training: it accumulates trades, market regimes,
   features, and outcomes as nodes/edges, and updates itself continually so
   agent decision quality improves the more it trades. Take direct
   inspiration from Prime Intellect's own approach to continual,
   decentralized self-improvement — research how they structure ongoing
   learning loops and adapt the idea to a financial knowledge graph.
3. **A sentiment/ontology agent** — a separate agent that builds an
   ontological knowledge graph from social/financial-chatter data (entities,
   relations, sentiment, events) and communicates with the trading agents
   (shared graph store, message bus, or embedding exchange — your call,
   justify it) to inform their decisions. **Data source note (already
   verified, do not re-litigate)**: as of Feb 2026 X/Twitter killed its free
   API tier — new developers get pay-per-use ($0.005/read, capped 2M/month)
   or a legacy $200+/mo Basic tier open only to grandfathered accounts; free
   scraping (snscrape and similar) is effectively dead because X locked down
   anonymous access and rotates internals. Build the ontology agent on: (a)
   Kaggle's static labeled tweet-sentiment datasets (~80k tweets, e.g.
   "Stock Tweets for Sentiment Analysis and Prediction") for historical
   training/backtesting, plus (b) Reddit via PRAW (free, rate-limited —
   r/wallstreetbets, r/stocks) and (c) StockTwits' public ticker-stream API
   (free tier, self-labeled bullish/bearish) for live signal. Only budget for
   paid X API access if research turns up evidence it's decisively better
   than this combination — don't default to it.
4. **Baseline comparison harness** — every trading agent's performance must
   be reported against the baseline(s); the design must explain how you'll
   make agents match-or-beat baseline, not just how you'll train them.
5. **Weights & Biases** for all hyperparameter tuning (sweeps).
6. **A visually striking live UI**: an animated force-directed / "spider
   web" graph rendering the EKG growing and rewiring itself in real time,
   plus live agent learning-curve plots and live P&L/equity-curve charts per
   agent vs. baseline. Propose a concrete stack (e.g. a Python backend
   streaming over WebSocket + a D3-force or Three.js frontend) and justify
   it over alternatives.

## The one open decision you must resolve via research

RL, LLM-driven, RLHF/preference-tuned, or a hybrid? Don't guess — survey the
literature and pick, with evidence. Starting hypothesis to validate or
overturn: a hybrid where an RL policy (PPO/SAC, trained in the TensorTrade
Gymnasium env) is the fast execution core, and the EKG + sentiment ontology
agent form a slower-moving memory/reasoning layer that reshapes the RL
policy's inputs/reward over time (closer to how Prime Intellect frames
continual multi-agent self-improvement than to a pure single-shot RLHF
setup). Confirm, refute, or replace this hypothesis in your final design —
show the comparative evidence either way.

## Research phase (do this first, thoroughly)

- Search papers from **2022 to present**: arXiv, published venues, and
  engineering write-ups (Medium, company research blogs). Use Parallel.ai's
  deep-research capability whenever your own knowledge might be stale or
  incomplete — do not rely on memory for anything time-sensitive (paper
  recency, current dataset/API availability, current best-reported
  Sharpe/return numbers). Never cite a paper you have not actually verified
  exists.
- Cover: deep RL for portfolio/single-asset trading (DQN/PPO/SAC/A2C
  variants and results), LLM-based trading/finance agents, RLHF and
  preference-tuned finance models, graph neural networks / knowledge graphs
  for financial reasoning, continual and self-improving multi-agent
  systems, and social-sentiment/ontology-graph approaches to trading
  signals from Twitter/X.
- Also study these existing open-source systems (some already known from
  prior research in this project) for architecture and pitfalls, not just
  to copy: tensortrade-org/tensortrade (base), TradeMaster-NTU/TradeMaster,
  AI4Finance-Foundation/FinRL-Trading, freqtrade/freqtrade,
  nautechsystems/nautilus_trader, ChuaCheowHuan/gym-continuousDoubleAuction,
  davebyrd/minabides.
- Deliverable of this phase: a comparison table of architectures with
  reported metrics vs. baselines, and your reasoned RL/LLM/RLHF/hybrid
  decision.
- **Whenever this research surfaces a new GitHub repo worth keeping**,
  follow the standing global rule: add it to Supermemory as a joined
  PROJECT memory + SKILLS/TECH STACK memory (with domain routing to
  `Ai Ml`/`Quantative` buckets as applicable), cross-linking the repo name
  and URL verbatim in both.

## Constraints (non-negotiable)

- **Compute**: training happens on cloud GPUs (Prime Intellect compute or
  equivalent) — do not design for local/laptop training.
- **Workflow**: terminal/CLI-first. Scripts and reproducible commands, not
  notebook-only workflows.
- **Data**: only free or easily-accessible datasets/APIs (e.g. yfinance,
  Stooq, Alpaca's free tier for equities). For the sentiment/ontology agent
  use the Kaggle + Reddit(PRAW) + StockTwits combination specified above
  instead of live X/Twitter API access (already confirmed unaffordable —
  see section 3 above). Flag anything else that turns out to require a paid
  tier, with a free fallback.
- **5 stocks**: you choose them during research — pick liquid, well-covered
  large-caps with easy free data access, and justify the picks (sector
  diversity, data quality, liquidity).

## Continuous tracking (do this throughout, not just at the end)

- **GitHub**: push all research notes, the architecture decision, and the
  staged plan to `https://github.com/sh-arka22/RL-Trader`. Use commits/PRs
  per stage (e.g. `docs/RESEARCH.md`, `docs/ARCHITECTURE.md`,
  `docs/PLAN.md`), so progress is visible incrementally, not as one final
  dump.
- **Supermemory**: after each research milestone and each planning
  decision, write a chunked memory update to container tag
  `repo_tradeai__6191323c4dd2d8a7` summarizing what was decided and why, so
  a future session can resume without re-reading everything.

## Required output: a staged plan

Produce a concrete, numbered stage plan (research → architecture spec →
data pipeline → baseline replication → core RL agent training → EKG
self-evolution loop → sentiment/ontology agent + integration → UI → full
evaluation vs. baseline → iteration). For each stage give: goal, entry
criteria, exit criteria (what "done" looks like, incl. concrete metrics),
and what gets committed to GitHub / written to Supermemory at the end of
it. Flag dependencies between stages explicitly.

## Guardrails

- Ground every architecture claim in something you actually found during
  research (cite it), not intuition.
- If evidence contradicts the hybrid hypothesis above, say so and propose
  the alternative — don't force-fit the starting hypothesis.
- If a component (e.g. free Twitter/X data) turns out to be infeasible,
  say so explicitly and propose the nearest feasible substitute rather than
  silently dropping the requirement.
