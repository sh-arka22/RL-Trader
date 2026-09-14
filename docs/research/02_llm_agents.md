# 02 — LLM Trading Agents, Multi-Agent LLM Finance, and RLHF/Preference Tuning (2023 → Sept 2026)

**Research date:** 2026-09-14 · **Author:** deep-research subagent · **Protocol:** `docs/RESEARCH_PROTOCOL.md`
**Citations verified in this session:** see [Verification log](#verification-log) at the end. Every arXiv ID was
confirmed live against `arxiv.org/abs/<id>`; every repo against the GitHub REST API; every price against the
vendor's own pricing page fetched on 2026-09-14.

---

## 0. Bottom line

| Question | Answer from the evidence |
|---|---|
| Do LLM trading agents beat buy-and-hold? | **Not reliably.** On a contamination-free window StockBench found "most models struggle to outperform the simple buy-and-hold baseline" (arXiv:2510.02209). FINSABER re-ran FinMem and FinAgent over 2004–2024 with real commissions and Buy-and-Hold beat both on Sharpe in **all four** bias-mitigated universes (arXiv:2505.07078). |
| Are the famous headline numbers trustworthy? | **No.** TradingAgents reports Sharpe 8.21 on a **3-month, 3-ticker** test with **no transaction costs** (arXiv:2412.20138). Profit Mirage measures Sharpe decay of **51–62 %** for five named agents once the test period passes the backbone's knowledge cutoff (arXiv:2510.07920). |
| Does an LLM improve an RL trading policy? | **Sometimes, weakly, and not monotonically.** FinRL-DeepSeek: LLM infusion *degrades* PPO at every strength tested and *helps* CPPO (arXiv:2502.07393). SAPPO: +0.35 Sharpe over PPO, but the edge dies above ~22 bps of cost (my computation, §3.5). |
| Does RLHF/DPO/GRPO beat prompting for trading? | **Not shown.** Fin-R1 (GRPO) reports QA scores, not P&L. FinDPO reports a strong portfolio result but **never compares DPO against SFT on P&L**. Trading-R1's own ablation shows SFT-only beats RL-only on 4 of 6 tickers. |
| Where should the LLM sit in RL-Trader? | **Out of the decision loop.** Use it offline as a *feature/knowledge* layer (news → structured sentiment/risk/event tags, EKG node extraction), with a point-in-time freeze, a cached and pinned model version, and a mandatory `--no-llm` ablation arm. Do **not** put an LLM-debate committee on the daily decision path. Full justification in §7. |

---
## 1. Verified system inventory

All arXiv IDs below were confirmed live in this session. `—` means the paper genuinely does not report the field
(and that is itself a finding). "Costs" = the transaction-cost assumption stated by the paper.

| System | arXiv / venue | Architecture (what is actually new) | Universe | Test window | Costs | Headline metrics | Baselines & outcome | Code (verified) | Backbone |
|---|---|---|---|---|---|---|---|---|---|
| **FinMem** | 2311.13743 (2023-11-23) | Profiling + 3-layer memory (news/10-Q/10-K) + reflection; FAISS retrieval; self-adaptive risk character | TSLA, NFLX, AMZN, MSFT, COIN (5 stocks, single-asset trading) | **2022-10-06 → 2023-04-10** (~6 mo) | **NOT REPORTED (implicitly zero)** | TSLA CR 61.78 %, Sharpe 2.68, MDD 10.80 %; AMZN CR 4.89 %, Sharpe 0.23 | B&H, A2C, PPO, DQN, Generative Agents, FinGPT. Beat all on the 5 tickers | `pipiku915/FinMem-LLM-StockTrading` 959★, MIT, last push 2024-08-18 | GPT-4-Turbo, temp **0.7** |
| **FinAgent** | 2402.18485 (2024-02-28) | Multimodal (K-line charts + news + filings), tool-augmented, market-intelligence + low/high-level reflection | AAPL, AMZN, GOOGL, MSFT, TSLA, ETHUSD | **2023-06-01 → 2024-01-01** (398-day span, latter half = test) | commission fee in reward; **rate NOT REPORTED** | TSLA ARR 92.27 %, SR 2.01; AAPL ARR 31.90 %, SR 1.43 | B&H, MACD, KDJ&RSI, ZMR, LGBM, LSTM, Transformer, DQN, SAC, PPO, FinGPT, FinMem. Beat all on 5/6 assets | `effective-p/FinAgent` 4★ (unofficial-looking mirror; no license) | GPT-4 / GPT-4V preview |
| **TradingAgents** | 2412.20138 (2024-12-28) | Trading-firm role split: 4 analysts → bull/bear researcher **debate** → trader → 3-way risk team → fund manager; LangGraph | AAPL, GOOGL, AMZN (3 tickers) | **2024-01-01 → 2024-03-29** (~60 trading days) | **NOT REPORTED** | AAPL CR 26.62 %, ARR 30.5 %, **SR 8.21**, MDD 0.91 % | B&H, MACD, KDJ+RSI, ZMR, SMA. Beat all | `TauricResearch/TradingAgents` **105,755★**, Apache-2.0, push 2026-09-07 | gpt-4o-mini + gpt-4o (fast) + o1-preview (deep) |
| **FinCon** | 2407.06567 (2024-07-09) | Manager–analyst hierarchy + **conceptual verbal reinforcement** (CVRF) risk-control belief updates | 8 assets (per FINSABER's replication of its selection); 2022-10-05 → 2023-06-10 | see left | **NOT REPORTED** | Survey-reported CR > 57 %, Sharpe 0.825 | Not extractable from the abstract | `The-FinAI/FinCon` 68★, no license, push 2026-02-27 | GPT-4-Turbo |
| **TradingGPT** | 2309.03736 (2023-09-07) | First layered-memory + distinct-character multi-agent design; inter-agent debate | stocks/funds (narrative) | **NOT REPORTED** | **NOT REPORTED** | **NOT REPORTED** | — | none verified | — |
| **StockAgent** | 2407.18957 (2024-07-15) | Multi-agent LLM **simulation** of investor behaviour with external market shocks — not a real-market backtest | simulated market | simulated | n/a | behavioural findings only | n/a (simulation) | `MingyuJ666/Stockagent` 701★, no license, push 2026-06-16 | — |
| **FinRobot** | 2405.14767 (2024-05-23) | Open agent *platform* (Financial-CoT, multi-source LLM layer), not a single trading policy | — | — | — | no closed-loop trading result | — | `AI4Finance-Foundation/FinRobot` 7,984★, Apache-2.0, push 2026-09-11 | pluggable |
| **FinGPT** | 2306.06031 (2023-06-09) | Open finance-LLM stack; includes **RLSP** (RL on Stock Prices) for sentiment labelling | — | — | — | sentiment Macro-F1 77.3 (LoRA SFT) → **80.9** (SFT+RLSP) | SFT-only. No P&L test | `AI4Finance-Foundation/FinGPT` 21,250★, MIT, push 2026-09-14 | ChatGLM / Llama |
| **PIXIU / FinMA** | 2306.05443 (2023-06-08) | Instruction-tuned finance LLM + FLARE benchmark | — | — | — | benchmark scores only | — | `The-FinAI/PIXIU` (= `chancefocus/PIXIU`) 887★, MIT, push 2025-03-04 | LLaMA 7B/30B |
| **LLMFactor** | 2406.10811 (2024-06-16) | Sequential Knowledge-Guided Prompting to extract *interpretable* factors for movement prediction | — | — | — | classification metrics, not P&L | — | none verified | — |
| **Alpha-GPT** | 2308.00016 (2023-07-31) | LLM ↔ genetic-programming alpha search, human in the loop | CN + US equities | 2024 WorldQuant IQC (live competition) | **NOT REPORTED** | HF strategy: return 14 %, Sharpe 5.47, MDD 2.36 % vs human Top-1 21 %/6.88/1.61 — **lost to the human** | human researchers | none verified | — |
| **Alpha-GPT 2.0** | 2402.09746 (2024-02-15) | Human-in-the-loop alpha mining, modelling and analysis loop | — | — | — | — | — | none verified | — |
| **QuantAgent** (2024) | 2402.03755 (2024-02-06) | Self-improving writer/judge loop generating signal code | 500 CN A-shares | data range 2023, split **NOT REPORTED** | **NOT REPORTED** | IC / signal Sharpe (values not extractable) | conceptual | `pillar/quantagent` 3★, `remlih/QuantAgent` 1★ — **neither looks official** | GPT-4-0125-preview |
| **QuantHarness / "QuantAgent" (2025)** | 2509.09995 (2025-09-12) | Price-driven multi-agent HFT: 4 upstream agents → short-horizon direction | BTC, CL, DJI, ES, VIX, NQ, QQQ, SPX | 1h/4h candles, dates **NOT REPORTED** | **NOT REPORTED** | direction metrics only | random, linear regression, XGBoost | none verified | — |
| **TradExpert** | 2411.00782 (2024-10-16) | Mixture-of-4-expert LLMs (news / market / alpha / fundamentals) + general router | — | 1 year (per FINSABER's survey table), 30 symbols | **NOT REPORTED** | — | — | none verified | — |
| **HedgeAgents** | 2502.13165 (2025-02-17) | Central fund manager + per-asset hedging specialists; experience sharing, "extreme-market conferences" | AAPL, Bitcoin, Yuan | **NOT REPORTED** | **NOT REPORTED** | survey-reported ARR 72 %, total return 405 % | **NOT REPORTED** | `JansenAnalytics/hedge-agents` 3★ (third-party replication) | GPT-4-1106-preview, temp 0.7 |
| **FinRL-DeepSeek** | 2502.07393 (2025-02-11) | **CPPO** + LLM news-derived *recommendation* score (scales actions) and *risk* score (scales CVaR trajectory return) | Nasdaq-100 | train 2013–2018 / trade 2019–2023 (also 2019–2022/2023) | **NOT REPORTED** | Information Ratio: PPO 0.0100, CPPO −0.0148, **PPO-DeepSeek −0.0093**, **CPPO-DeepSeek +0.0078** | plain PPO, plain CPPO, Nasdaq-100 index | `benstaf/FinRL_DeepSeek` 331★, MIT, push 2025-04-08 | DeepSeek-V3, Qwen2.5-72B, Llama-3.3-70B |
| **Trading-R1** | 2509.11420 (2025-09-14) | SFT warm-start → **RFT** with volatility-adjusted market-outcome reward; 3-stage curriculum (structure → evidence → decision) | NVDA, AAPL, MSFT, AMZN, META, SPY (train set = 14 tickers) | **2024-06-01 → 2024-08-31** (held out) | **NOT REPORTED** | NVDA CR 8.08 %/SR 2.72; AAPL 5.82 %/1.80; SPY 3.34 %/1.60 | 10 LLM baselines + its own SFT-only and RL-only arms | `TauricResearch/Trading-R1` 486★, no license | Qwen-based |
| **FinDPO** | 2507.18417 (2025-07-24) | First finance **DPO** sentiment model + logit-to-score portfolio mapping | 417 S&P 500 firms with text | **2015-02 → 2021-06** | **0 bps and 5 bps** variants reported | @5 bps: CR 458.97 %, ARR 66.64 %, **Sharpe 2.03**; @0 bps: CR 747.10 %, Sharpe 3.41 | HIV-4, VADER, LMD, FinBERT, FinLlama, S&P 500 (0.62 Sharpe). Beat all sentiment baselines | — | Llama-3-8B-Instruct, 41.9 M LoRA params, 1×A100-40GB × 4.5 h |
| **Fin-R1** | 2503.16252 (2025-03-20) | SFT + **GRPO** (format reward + answer-accuracy reward judged by Qwen2.5-Max) | — | — | — | avg benchmark score 75.2 (2nd place) | other reasoning LLMs on FinQA/ConvFinQA/etc. | — | Qwen2.5-7B-Instruct, 7 B |
| **HARLF** | 2507.18560 (2025-07-24) | 3-tier base → meta → super RL hierarchy; FinBERT monthly sentiment as a feature | 14 assets (equities + commodities) | 2018–2024 | **costs excluded by the authors** | ARR 26 %, Sharpe 1.2 | equal-weight, S&P 500 | `Sidharth-bhat/HARLF-portfolio-optimization` 1★ | FinBERT (not a generative LLM) |
| **GIFT** | 2606.08450 (2026-06-07) | LLM writes *executable* state-enhancement + reward-shaping code; PPO still learns the action | 6 panels × 5 S&P 500 equities + cash | 6 rolling windows, data 2000-01 → 2024-06 | **0.1 % per unit turnover** | component-removal deltas only (see §4.2) | Pure PPO, same architecture/split/costs | none verified | — |
| **LM-Guided RL** | 2508.02366 (2025-08-04) | "Strategist" LLM emits a horizon-level policy that augments the RL observation space | AAPL, AMZN, GOOGL, META, MSFT, TSLA | **NOT REPORTED** | **NOT REPORTED** | mean Sharpe: hybrid **1.10**, RL-only 0.64, LLM-only 1.03 | RL-only and LLM-only in identical envs | none verified | — |
| **FLAG-Trader** | 2502.11433 (2025-02-17) | A 135 M LLM *is* the policy net, post-trained with PPO | MSFT, JNJ, UVV, HON, TSLA, BTC | stocks 2020-10-01 → 2021-05-06; BTC 2023-04-05 → 2023-11-05 | **NOT REPORTED** | CR/Sharpe/AV/MDD reported | B&H + InvestorBench LLM agents | ACL Findings 2025 | small fine-tuned LLM |
| **AlphaAgent** | 2502.16789 (2025-02-24) | Idea/factor/eval agent loop with originality + complexity + hypothesis-alignment regularisation to fight alpha decay | CSI 500, S&P 500 | train 2015-01→2019-12, val 2020, **test 2021-01→2024-12** | **CSI 500 buy 5 bps / sell 15 bps; S&P 500 sell 5 bps** | CSI 500 ARR 11.00 %, IR 1.488, MDD −9.36 %; S&P 500 ARR 8.74 %, IR 1.05 | LSTM, Transformer, LightGBM, StockMixer, TRA, AlphaForge, RD-Agent, o1, DeepSeek-R1 | `RndmVariableQ/AlphaAgent` 410★, no license | GPT-3.5-turbo |
| **InvestorBench** | 2412.18174 (2024-12-24) | POMDP-style benchmark for financial decision agents; 13 LLMs | crypto + equities + ETFs | — | — | benchmark scores | — | `felis33/INVESTOR-BENCH` 30★, MIT | 13 LLMs |
| **StockBench** | 2510.02209 (2025-10-02) | **Contamination-free** multi-month trading benchmark | top-20 DJIA | **2025-03-03 → 2025-06-30** (82 trading days) | **explicitly none** (authors state this as a limitation) | best = Kimi-K2 +1.9 %, Sortino 0.0420; **passive baseline +0.4 %, Sortino 0.0155**; GPT-5 +0.3 %, Sortino 0.0132 | equal-weight B&H. **Most models fail to beat it** | `ChenYXxxx/stockbench` 179★, Apache-2.0 | GPT-5, Claude-4-Sonnet, o3, Qwen3, Kimi-K2, GLM-4.5, DeepSeek |
| **Agent Market Arena (AMA)** | 2510.11695 (2025-10-13) | **Live**, lifelong, multi-market arena; 4 agent designs × 5 backbones | TSLA, BMRN, ETH, BTC | init 2025-05-01→07-31; **live 2025-08-01 → 2025-09-30** | **NOT REPORTED** | InvestorAgent/GPT-4.1 TSLA CR 40.83 %, SR 6.47; **TradeAgent TSLA ranges −38.72 % (GPT-4.1) to +21.91 % (Gemini-2.0-Flash)** | B&H | live board only | GPT-4o, GPT-4.1, Gemini-2.0-flash, Claude-3.5-haiku, Claude-sonnet-4 |
| **PortBench** | 2605.27887 (2026-05-27) | 5-stage full-pipeline portfolio benchmark, correlation-aware | 183 instruments, 6 classes | stress windows 2015-16, 2020, 2022, 2024 | **10 bps slippage + 5 bps commission** | benchmark scores (QA 0.819, CEPS 0.329–0.495) | equal-weight, 60/40, risk parity, min-var, Black-Litterman | `AgenticFinLab/portbench` 9★, Apache-2.0 | GLM-5.1, Kimi-K2.6, DS-V4-Flash |
| **Mint-Agent** | 2608.16386 (2026-08-17) | Finance-native agentic foundation model | — | — | — | reasoning benchmarks only | — | none verified | — |

### 1.1 Code reality check (GitHub API, fetched 2026-09-14)

| Repo | Stars | Last push | License | Note |
|---|---:|---|---|---|
| `TauricResearch/TradingAgents` | 105,755 | 2026-09-07 | Apache-2.0 | By far the most adopted; the paper's own site warns performance varies with backbone, temperature, period, data quality |
| `virattt/ai-hedge-fund` | 63,382 | 2026-09-03 | MIT | The system Look-Ahead-Bench (2601.13770) used as its trading harness |
| `AI4Finance-Foundation/FinGPT` | 21,250 | 2026-09-14 | MIT | Active |
| `AI4Finance-Foundation/FinRL` | 16,282 | 2026-07-13 | MIT | RL baseline library |
| `AI4Finance-Foundation/FinRobot` | 7,984 | 2026-09-11 | Apache-2.0 | Active |
| `tensortrade-org/tensortrade` | 7,117 | 2026-02-19 | Apache-2.0 | RL-Trader's planned execution layer — **last push 7 months ago** |
| `pipiku915/FinMem-LLM-StockTrading` | 959 | **2024-08-18** | MIT | Stale for 2 years |
| `The-FinAI/PIXIU` | 887 | 2025-03-04 | MIT | |
| `MingyuJ666/Stockagent` | 701 | 2026-06-16 | none | No license = not legally reusable |
| `TauricResearch/Trading-R1` | 486 | 2025-09-15 | none | No license |
| `RndmVariableQ/AlphaAgent` | 410 | 2026-07-03 | none | |
| `benstaf/FinRL_DeepSeek` | 331 | 2025-04-08 | MIT | The key LLM+RL ablation code |
| `ChenYXxxx/stockbench` | 179 | 2025-10-28 | Apache-2.0 | Contamination-free benchmark |
| `waylonli/FINSABER` | 145 | 2026-08-05 | Apache-2.0 | **The bias-mitigated backtest harness — most useful repo in this dossier** |
| `The-FinAI/FinCon` | 68 | 2026-02-27 | none | |
| `Open-Finance-Lab/FinRL_Contest_2025` | 66 | 2025-10-20 | none | |
| `felis33/INVESTOR-BENCH` | 30 | 2025-09-13 | MIT | |
| `AgenticFinLab/portbench` | 9 | 2026-09-12 | Apache-2.0 | |
| `effective-p/FinAgent` | 4 | 2026-06-06 | none | **Not credible as the official FinAgent release** |
| `pillar/quantagent` | 3 | 2025-11-17 | MIT | **Not credible as the official QuantAgent release** |

> **Honest negative:** several widely-cited systems (TradingGPT, Alpha-GPT 1.0/2.0, TradExpert, LLMFactor, FinVerse,
> HedgeAgents, FinCon's full pipeline) have **no verifiable official code**. The repos that do exist for FinAgent and
> QuantAgent (4★ and 3★) are almost certainly third-party mirrors, not author releases.

---
## 2. Multi-agent LLM patterns in finance — which ones actually earned their keep?

### 2.1 The pattern catalogue

| Pattern | Canonical system | What it adds | Measured gain? |
|---|---|---|---|
| **Layered memory + reflection** | FinMem (2311.13743), TradingGPT (2309.03736) | shallow (news) / intermediate (10-Q) / deep (10-K) memory with decay + self-reflection on past trades | FinMem's own ablation shows memory layers help; **but** the same memory design is the single worst offender for leakage (Profit Mirage: FinMem prediction-consistency 0.8213, the highest of 5 systems) |
| **Analyst → researcher-debate → trader → risk → fund-manager** | TradingAgents (2412.20138) | bull/bear debate, configurable `max_debate_rounds`, hierarchical sign-off | Profit Mirage: multi-agent systems show the **lowest** leakage (TradingAgents PC 0.6903, FinCON 0.7136 vs FinMem 0.8213), i.e. debate makes the system *more input-driven*. But the ACM reproducibility study found TradingAgents returns **15.8 % ± 4.2 %** (GPT-4o) vs **GOOGL buy-and-hold 19.1 %** on GOOGL, May–Jul 2025, 0.05 % round-trip spread — it **lost to buy-and-hold** |
| **Manager–worker hierarchy + verbal reinforcement** | FinCon (2407.06567) | a risk manager updates "investment beliefs" in natural language between episodes (CVRF) | Profit Mirage shows FinCon is second-lowest leakage, but its own reported CR/Sharpe cannot be reproduced from the paper's abstract, and FINSABER shows its *stock-selection* universe underperforms B&H |
| **Specialist hedging agents + conferences** | HedgeAgents (2502.13165) | per-asset specialists, budget allocation, "extreme-market conference" | headline ARR 72 % / total return 405 % with **no reported window, costs, or baseline** — uninterpretable |
| **Tool augmentation (charts, indicators, APIs)** | FinAgent (2402.18485) | K-line vision, 60 indicators, expert guidance | **Negative result in the paper's own ablation** (Table 6): on AAPL, *removing* the tool module (`w/o-T`) gives ARR **33.75 % / SR 1.52** vs the full agent's **31.90 % / 1.43**; on ETHUSD `w/o-T` gives **54.80 % / 1.40** vs full **43.08 % / 1.18**. Tools helped on TSLA/AMZN/GOOGL and hurt on AAPL/ETHUSD |
| **Mixture-of-experts routing** | TradExpert (2411.00782) | 4 domain-expert LLMs + router | no extractable protocol |

### 2.2 Does the debate itself help? General ML evidence says: be sceptical

| Source | Finding |
|---|---|
| **Why Do Multi-Agent LLM Systems Fail?** arXiv:2503.13657 | "Despite enthusiasm for Multi-Agent LLM Systems (MAS), their performance gains on popular benchmarks are often minimal." 14 failure modes across 7 frameworks and 200+ tasks (MAST), Cohen's κ = 0.88 |
| **Stop Overvaluing Multi-Agent Debate** arXiv:2502.08788 | Argues MAD's reported gains come from weak evaluation; homogeneous debate is the problem |
| **The Cost of Consensus** arXiv:2605.00914 | Isolated self-correction beats *unguided homogeneous* multi-agent debate; debate can decrease accuracy via peer pressure and inter-agent sycophancy |
| **Agent Market Arena** arXiv:2510.11695 | In **live** trading, "modifying the underlying LLM within a fixed agent framework produces only modest changes in profitability, whereas varying the agent design while holding the LLM constant leads to far greater performance divergence" — architecture > backbone |

### 2.3 The finding that matters most for RL-Trader

AMA's live board shows the **same agent** swinging from **−38.72 % to +21.91 % on TSLA** over the same two months
purely by changing the backbone (TradeAgent: GPT-4.1 vs Gemini-2.0-Flash). FINSABER shows the **same effect for a
single system**: FinMem on TSLA over its own reported window scores **Sharpe 2.679 as published (GPT-4-Turbo)**,
**0.927 when re-run with GPT-4o-mini**, and **0.404 with GPT-4o** — a 85 % Sharpe collapse from a backbone swap alone.

> **Conclusion for §2.** Role-split + debate is the *only* multi-agent pattern with a measurable, direction-consistent
> benefit, and the benefit is **reduced leakage / higher input-sensitivity**, not higher returns. Memory-heavy
> single-agent designs (FinMem style) maximise leakage. Tool augmentation is ticker-dependent and can hurt.

---

## 3. LLM + RL hybrids — the master brief's core hypothesis

This is the section the project's architecture decision hangs on. I split it by pattern and demand a **same-policy
ablation** (the identical RL policy with and without the LLM channel). Papers without one are marked **NO ABLATION**.

### 3.1 Pattern A — LLM as feature / signal generator into an RL state

| Paper | LLM channel | Universe | Window | Costs | Same-policy ablation result |
|---|---|---|---|---|---|
| **FinRL-DeepSeek** 2502.07393 | DeepSeek/Qwen/Llama produce a 1–5 *recommendation* score that scales the action, and a 1–5 *risk* score that scales the CVaR trajectory return | Nasdaq-100 (FNSPID news) | train 2013–2018, trade **2019–2023** | **NOT REPORTED** | **MIXED / MOSTLY NEGATIVE.** Information Ratio: PPO **0.0100** → PPO-DeepSeek 10 % **−0.0093**, 1 % **−0.0252**, 0.1 % **−0.0011**. CPPO **−0.0148** → CPPO-DeepSeek 10 % **+0.0078**, 1 % −0.0032, 0.1 % −0.0060. Paper's own words: *"For PPO-DeepSeek, stronger LLM infusion mostly degrades performance, even for tiny (0.1 %) perturbations"* and, in the 6-year-training run, *"The use of LLM always worsens performance in this test."* |
| **SAPPO** ACL REALM 2025 (`2025.realm-1.12`) | LLaMA 3.3 scores Refinitiv news → daily sentiment in [−1,1] added to the state **and** to the PPO advantage via `A' = A + λ·wₙ·mₙ`, λ = 0.1 | GOOG, MSFT, META | train 2013-01→2019-12, **test 2020-01→2020-12** (251 days) | **0.05 % per turnover**, VWAP first 10 min | **POSITIVE.** SAPPO Sharpe **1.90** vs PPO (λ=0) **1.55**; ARR 30.2 % vs 26.5 %; MDD −13.8 % vs −17.5 %. Swapping LLaMA-3.3 for FinBERT gives 1.72 — i.e. **most of the gain is available from a 110 M encoder, not a 70 B LLM** |
| **LLM-augmented PPO-CVaR multiseed** (`Shafiya0101/signal-density-llm-trading`, 0★) | signal-coverage sweep | Nasdaq-100 | 2019–2023 | **NOT REPORTED** | full coverage mean return 166.34 % (3 seeds) vs no-signals 121.27 % (3 seeds). Non-monotonic; unpublished repo, treat as weak evidence |
| **HARLF** 2507.18560 | FinBERT monthly sentiment feature | 14 assets | 2018–2024 | **costs excluded** | **NO ABLATION** |

### 3.2 Pattern B — LLM as reward model / reward shaper

| Paper | Mechanism | Ablation |
|---|---|---|
| **GIFT** 2606.08450 | The LLM never trades. It **writes executable code** for (i) factor-guided state enhancement, (ii) risk-rule reward shaping, (iii) diagnostic-guided refinement. PPO learns the actions. Costs 0.1 %/turnover; 6 rolling windows on 6 × 5-equity S&P 500 panels | **Partial.** Removing factor-state: Sharpe −0.129; removing reward shaping: Sharpe −0.129, Sortino −0.213, MDD +3.05 pp; removing diagnostic refinement: Sharpe −0.393. Absolute Pure-PPO numbers not published, so we cannot confirm the full system beats Pure PPO |
| **FinRL-DeepSeek (risk channel)** 2502.07393 | LLM risk score multiplies the CVaR trajectory return | The only place LLM infusion **helped**: CPPO −0.0148 → CPPO-DeepSeek +0.0078 IR |

There is **no published paper** that trains a proper LLM-as-reward-model (Bradley-Terry preference head) for trading
P&L and shows it beating an identical RL policy with a hand-written reward. That is a genuine gap, not an oversight in
my search.

### 3.3 Pattern C — LLM planner over an RL executor

| Paper | Result |
|---|---|
| **LM-Guided RL** 2508.02366 | Mean Sharpe: **hybrid 1.10, RL-only 0.64, LLM-only 1.03**; per-ticker hybrid vs RL-only: AAPL 1.70/1.42, AMZN 1.21/0.42, GOOGL 1.16/0.23, META 0.46/0.15, MSFT 1.16/0.99, TSLA 0.92/0.62. Mean MDD hybrid 0.31 vs RL-only 0.36. **But the paper reports neither the test window nor transaction costs**, so the numbers are not economically interpretable. Note also that LLM-only (1.03) ≈ hybrid (1.10) — the RL executor adds ~0.07 Sharpe |
| **HARLF** 2507.18560 | Hierarchical, but the "LLM" is FinBERT used as a sentiment feature. Not a language planner. **NO ABLATION** |

### 3.4 Pattern D — RL-tuned LLM traders

| Paper | Ablation from the paper's own table |
|---|---|
| **Trading-R1** 2509.11420, test **2024-06-01 → 2024-08-31**, costs **NOT REPORTED** | CR %/SR by arm — NVDA: SFT-only **7.42/2.72**, RL-only 3.27/1.25, **Trading-R1 8.08/2.72**. AAPL: SFT −2.37/−1.27, RL 4.04/1.14, R1 **5.82/1.80**. MSFT: SFT −0.24/−0.64, RL −0.18/−0.81, R1 **2.38/0.87**. AMZN: SFT 1.93/0.36, RL −0.05/−0.29, R1 **5.39/1.72**. META: SFT 2.52/0.54, RL −0.18/−0.36, R1 **5.12/0.86**. SPY: SFT 1.78/0.86, RL 1.85/1.00, R1 **3.34/1.60**. **RL-only is worse than SFT-only on 4/6 tickers**; the gain comes from the *combination*, over a 63-day window, with no cost model and no B&H row |
| **FLAG-Trader** 2502.11433 | LLM-as-policy trained with PPO; **NO same-policy no-LLM ablation** |

### 3.5 The cost-fragility test the papers do not run (my computation)

SAPPO is the cleanest positive hybrid result in the literature. It is also **cost-fragile**. From the paper's own
Table 1: SAPPO daily turnover **12.0 %** vs PPO **3.5 %**, at an assumed **5 bps** cost. Extra turnover is
8.5 pp/day → **21.4× portfolio value per year**. Re-pricing that extra turnover:

| Round-trip cost | Extra annual cost from SAPPO's higher turnover | SAPPO's return edge over PPO (starts at +3.70 pp) |
|---:|---:|---:|
| 5 bps (as published) | 1.07 % | **+3.70 pp** |
| 10 bps | 2.14 % | +2.63 pp |
| 15 bps | 3.21 % | +1.56 pp |
| 20 bps | 4.28 % | +0.49 pp |
| **25 bps** | 5.36 % | **−0.58 pp** |
| 30 bps | 6.43 % | −1.66 pp |

*Method: extra annual cost = (0.12 − 0.035) × 252 × bps/10,000; edge = 3.70 pp − (extra cost − 1.07 %). This is my
arithmetic on the paper's published turnover and return figures, not a number from the paper.*

**The entire published advantage of LLM sentiment over plain PPO disappears at ~22 bps of round-trip cost.**
For reference, FINSABER used Moomoo's real retail schedule ($0.0049/share, $0.99 minimum per order).

### 3.6 Verdict on the hybrid hypothesis

| Claim in the master brief | Evidence status |
|---|---|
| "RL policy = fast execution core" | **Supported.** In every bias-mitigated comparison found, RL (PPO/SAC/TD3) beats the LLM agents: FINSABER bias-mitigated Sharpe — Random-5 universe: B&H 0.315, **PPO 0.179**, FinAgent 0.094, FinMem −0.253. Momentum: B&H 0.384, PPO 0.185, FinAgent 0.104, FinMem 0.025. Volatility: B&H 0.703, **PPO 0.514**, FinAgent 0.241, FinMem −0.228. FinCon-selection: B&H 0.389, PPO 0.132, FinAgent −0.076, FinMem −0.292 |
| "LLM/EKG as a slower memory-reasoning layer that reshapes RL inputs/reward" | **Partially supported, conditionally.** SAPPO (+0.35 Sharpe, cost-fragile) and GIFT (component deltas) support it. FinRL-DeepSeek **refutes it for plain PPO** and supports it only for risk-sensitive CPPO |
| Implicit assumption that "more LLM = better" | **Refuted.** FinRL-DeepSeek's infusion sweep is monotonically bad for PPO down to 0.1 % strength; StockBench's news ablation buys only +0.5 pp over 4 months for the best model and **0.0 pp** for GPT-OSS-120B |
| **Also note** | FINSABER's headline: **Buy-and-Hold beats both the LLM agents AND PPO in 4 of 4 bias-mitigated universes**, with B&H-vs-FinMem p-values of 3.0e-6 / 4.0e-5 / 4.0e-6. Any RL-Trader claim of "beating the baseline" must survive this kind of test |

---
## 4. RLHF / DPO / GRPO / preference tuning in finance

### 4.1 What exists (all verified)

| System | Method | Reward / preference signal | Reported result | Universe / window / costs / baseline | Train cost |
|---|---|---|---|---|---|
| **FinGPT RLSP** 2306.06031 | RL on Stock Prices (LoRA SFT → RLSP) | **realised post-news price move** as the label/reward | sentiment Macro-F1 **77.3 → 80.9** | 620k+ headlines, 2016–2024. **No trading P&L at all** — universe/window/costs/baseline all NOT REPORTED | — |
| **Fin-R1** 2503.16252 | SFT + **GRPO** | format reward + answer-accuracy reward (judged by Qwen2.5-Max) | avg benchmark 75.2, 2nd place | FinQA, ConvFinQA, Ant-Finance, TFNS, Finance-Instruct-500K. **No trading backtest** | Qwen2.5-7B base; GPU hours NOT REPORTED |
| **FinDPO** 2507.18417 | **DPO** on synthetic preference pairs from FPB/TFNS/NWGI labels | ground-truth sentiment label = chosen; reference-model output = rejected. **Not human preference, not realised return** | @0 bps: CR 747.10 %, ARR 111.78 %, Sharpe 3.41. **@5 bps: CR 458.97 %, ARR 66.64 %, Sharpe 2.03, Sortino 3.75, Calmar 2.21** | **417 S&P 500 firms with text; 2015-02 → 2021-06; 0 and 5 bps; vs HIV-4, VADER, LMD, FinBERT, FinLlama and the S&P 500 (CR 83.12 %, Sharpe 0.62)**. FinDPO is the only method with significant positive returns at 5 bps | **1 × A100-40GB for 4.5 h**, 41.9 M LoRA params |
| **LLMs Meet Finance** 2504.13125 | SFT, DPO and synthetic-data RL on the Open FinLLM Leaderboard | accepted DB answers vs overlong SFT outputs | benchmark deltas only; **DPO improved output length and one task but degraded NER F1** | no trading evaluation | 3–5 h on 4 × 2080Ti per 1.5 B model |
| **Trading-R1** 2509.11420 | SFT warm-start → RFT with market-grounded outcome reward | **volatility-adjusted, percentile-based, asset-specific realised return** | see §3.4 | 6 tickers, 2024-06-01 → 2024-08-31, **costs NOT REPORTED**, no B&H row | GPU hours NOT REPORTED |
| **FinRankGRPO** (OpenReview `KBC0xQJ4uz`) | listwise GRPO for asset ranking | ranking objective | **no accessible result table** — treat as UNQUANTIFIED | — | — |

### 4.2 Does preference tuning beat plain RL or plain prompting for *trading decisions*?

**No direct, clean comparison exists.** Specifically:

| Comparison you would want | Does any paper run it? |
|---|---|
| DPO-tuned trader vs SFT-tuned trader, same data, on P&L | **No.** FinDPO compares DPO vs SFT on sentiment **F1**, but its portfolio table contains no FinSFT row |
| GRPO-tuned trader vs prompted frontier model, on P&L | **No.** Fin-R1 never runs a backtest |
| RL-on-LLM vs RL-on-a-small-policy-net, same env | **No.** FLAG-Trader has no no-LLM arm |
| SFT vs RL vs SFT+RL on P&L | **Yes — Trading-R1 only.** And **RL-only loses to SFT-only on 4 of 6 tickers** |

### 4.3 Honest negatives on preference tuning

1. **Benchmark gains ≠ P&L gains.** FinGPT's RLSP improves Macro-F1 by 3.6 points and is never converted into a
   cost-adjusted portfolio result. Fin-R1's GRPO gains are entirely on QA benchmarks.
2. **DPO can degrade other capabilities.** 2504.13125 reports an NER F1 decline after DPO.
3. **Market-outcome rewards invite backtest overfitting.** Trading-R1's reward *is* the realised forward return of the
   training window; the held-out window is 63 trading days with no cost model. This is exactly the setting where the
   Deflated Sharpe Ratio / PBO literature says a selected Sharpe is likely a false positive.
4. **FinDPO's preferences are synthetic label preferences,** not human preferences and not returns. Calling it
   "RLHF for trading" would be wrong.

---

## 5. The critical side — leakage, replication failure, and irreproducibility

This section is deliberately given equal weight. It is the strongest, most decision-relevant evidence in the dossier.

### 5.1 Parametric look-ahead: the model already knows the answer

| Study | Design | Result |
|---|---|---|
| **Profit Mirage** 2510.07920 (2025-10-09) | Re-evaluates FinMem, FinAgent, QuantAgent, FinCon, TradingAgents on AAPL, NVDA, TSLA, BYD, Tencent, BTC, **2020-01-01 → 2025-06-30**, "standard transaction costs and realistic slippage models" (values not stated), GPT-4o backbone temp 0.7 | (a) Rolling the calendar past the backbone's cutoff: **Sharpe decay 51.48 % (QuantAgent) → 62.23 % (FinCon)**; TradingAgents −55.68 %; **total-return decay 50.18 % (TradingAgents) → 71.85 % (FinMem)**. "Almost every published LLM-based agent fails to beat a random baseline once its knowledge cutoff is passed." (b) Counterfactual perturbation — **prediction consistency** (lower=better): FinMem **0.8213**, QuantAgent 0.7789, FinAgent 0.7245, FinCon 0.7136, TradingAgents 0.6903. Every system keeps ≥69 % of its predictions unchanged after *materially altering the market inputs*. (c) **FinLeak-Bench**: 2,000 historical QA pairs; GPT-4o and peers answer **>85 %** correctly |
| **Look-Ahead-Bench** 2601.13770 (2026-01-20) | Dual-period design using the `virattt/ai-hedge-fund` harness (63,382★). P1 in-sample **2021-04-01 → 2021-09-30**; P2 out-of-sample **2024-07-01 → 2024-12-31**; alpha measured vs B&H | **P1 → P2 alpha decay: Llama-3.1-8B −17.23 pp, Llama-3.1-70B −15.25 pp, DeepSeek-3.2 −21.77 pp.** Raw: Llama-8B +39.13 % (α +13.81 pp) in-sample → +21.33 % (α **−3.42 pp**) out-of-sample, against B&H +25.32 % → +24.75 %. Purpose-built point-in-time models (`Pitinf-Small`, PiT-Inference) show ≈0 decay: +25.07 % (α −0.25 pp) → +24.81 % (α +0.06 pp) |
| **The Memorization Problem** 2504.14765 (2025-04-20) | Probes LLM recall of economic indicators, news headlines, stock returns, conference calls | Establishes that when the model has seen realised values, forecast accuracy cannot distinguish skill from recall |
| **DatedGPT** 2603.11838 (2026-03-12) | Time-aware pretraining as a structural fix | Confirms the failure mode is considered severe enough to warrant retraining from scratch |
| **FinCAD / "Summoning the Oracle"** 2605.24564 (2026-05-23) | Inference-time context-aware decoding to suppress memorised outcomes | Reports cutting in-sample backtest returns by up to **−67.1 %** on memorised dates — i.e. that much of the reported return *was* memorisation |
| **KTD-Fin / From Knowing to Doing** 2605.28359 (2026-05-27) | Masks tickers, dates and tool timestamps; CSI300 A-shares **2024-01-01 → 2026-04-10**, **5 bps buy / 15 bps sell, CNY 5 minimum, next-day open** | Memory-only condition: anchor model returns **−0.16 %** bright vs **0.00 %** blinded; returns are "largely explained by passive market/style exposure rather than stock-selection alpha" |

**Concrete implication for the famous results:** FinMem's test window is **2022-10-06 → 2023-04-10** and its backbone is
GPT-4-Turbo. That window sits inside the period the backbone read about. SAPPO scores **2020** news with **LLaMA 3.3**
(released Dec 2024) — the sentiment model already knows how COVID-2020 resolved. These are not hypothetical risks.

### 5.2 Replication and audit results

| Audit | Scope | Headline |
|---|---|---|
| **FINSABER** 2505.07078 (2025-05-11), code `waylonli/FINSABER` 145★ | Re-runs FinMem and FinAgent over **2004–2024, 100+ symbols incl. delisted names**, risk-free 3 %, **Moomoo commissions $0.0049/share, $0.99 min/order** | **On FinMem's own window/tickers**: FinMem TSLA Sharpe **2.679 published** → **0.927 (GPT-4o-mini)** → **0.404 (GPT-4o)**; AMZN 0.233 → 0.297 → **−0.968**. **Bias-mitigated universes (Sharpe)**: Random-5 (91 syms) B&H **0.315** vs FinMem −0.253, FinAgent 0.094, PPO 0.179; Momentum (84) B&H **0.384** vs 0.025 / 0.104 / 0.185; Volatility (63) B&H **0.703** vs −0.228 / 0.241 / 0.514; FinCon-selection (80) B&H **0.389** vs −0.292 / −0.076 / 0.132. B&H-vs-FinMem p = 3.0e-6, 4.0e-5, 4.0e-6. FinMem's commission ratio is **5–9× FinAgent's** (overtrading), and FinMem shows negative alpha in all scenarios |
| **Agentic Trading (audit)** 2605.19337 (2026-05-19) | 77 studies screened; **n = 19** met "action output + closed-loop evaluation" | **2/19** report an extractable time-consistent split; **1/19** specifies a transaction-cost model; **1/19** documents universe/survivorship handling; 11/19 report execution timing; **15/19 are R0** and **0/19 reach R3 reproducibility**. Also: few papers report failed prompts, abandoned reward schemes, or degraded live performance |
| **Beyond Agent Architecture** 2606.08285 (2026-06-06) | Coded evidence matrix over **30 trade-relevant primary studies** | "Architecture reporting is generally clearer than the evaluation assumptions needed to judge whether a trading result is economically interpretable or reproducible." Four recurring failure modes, led by timing information present only at narrative level |
| **Reproducibility in the TradingAgents Framework**, ACM `10.1145/3800973.3801029` (Proc. 2026 Int. Conf. on AI and Fintech) — **DOI page returns HTTP 403 to automated fetch; content taken from indexed excerpts, flag as partially verified** | Varies temperature, seed, top_k, top_p; 5 runs (GPT-4o), 10 runs (Qwen3:30B) per config | **GOOGL, May–Jul 2025, 0.05 % round-trip spread (~0.85 % total over ~17 trades)**: GPT-4o **15.8 % ± 4.2 % SEM**, Qwen3:30B **18.1 % ± 2.8 %**, vs **GOOGL B&H 19.1 %** and **QQQ B&H 17.4 %** (perfect foresight = 97 %). Deterministic Qwen3:30B (T=0, seed=42, top_k=1, top_p=0) gave **28.2 %** with zero variance. Conclusion: stochastic configurations do not justify their complexity once cherry-picking is removed |
| **Survey** 2408.06361 | 27 LLM trading papers | Median testing period **1.3 years**; start/end dates "chosen rather arbitrarily"; single-stock studies pick the tickers with the **most available news**; few studies consider trading costs; slippage not modelled |

### 5.3 The specific methodological failures, scored

| Failure | Who is implicated | Evidence |
|---|---|---|
| Test window inside the backbone's knowledge window | FinMem (2022-10 → 2023-04 with GPT-4-Turbo), SAPPO (2020 news scored by LLaMA-3.3) | Profit Mirage §2.1; Look-Ahead-Bench Table 1 |
| Tiny test windows | TradingAgents **60 trading days**; Trading-R1 **63 days**; AMA **2 months**; FinMem ~6 months; StockBench 82 days | all from the papers themselves |
| Cherry-picked tickers | TradingAgents uses 3; FinMem 5; survey 2408.06361: single-stock studies pick highest-news-volume tickers | FINSABER §4; survey |
| No transaction costs | TradingAgents, FinMem, FinRL-DeepSeek, Trading-R1, AMA, StockBench (explicitly), HARLF (explicitly) | papers |
| Single run / no seeds | TradingAgents (paper), most systems | ACM reproducibility study; StockBench per-model variance ×10⁻⁴ ranges **0.074 (DeepSeek-V3) → 10.190 (GPT-OSS-120B)**, a 138× spread |
| Absurd Sharpe not flagged | TradingAgents **SR 8.21** — the authors themselves note it "exceeds our expected empirical range"; AMA InvestorAgent **SR 6.47** over 2 months | papers |
| No multiple-testing correction | none of the surveyed systems | Deflated Sharpe Ratio (Bailey & López de Prado, *J. Portfolio Management* 40(5):94); backtest-overfitting critique arXiv:2209.05559 |

---
## 6. Operational reality — cost, latency, determinism

### 6.1 Live API pricing (fetched by me from each vendor's own page on **2026-09-14**)

| Vendor | Model | $/1M input | $/1M output | Cached input | Batch |
|---|---|---:|---:|---:|---|
| OpenAI | `gpt-6-astra` (flagship) | 10.00 | 50.00 | 1.00 | $5.00 / $25.00 |
| OpenAI | `gpt-5.6-sol` | 4.00 | 20.00 | 0.40 | $2.00 / $10.00 |
| OpenAI | `gpt-5.6-terra` | 2.00 | 12.00 | 0.20 | $1.00 / $6.00 |
| OpenAI | `gpt-5.6-luna` (cheap) | **0.20** | **1.20** | 0.02 | **$0.10 / $0.60** |
| Anthropic | Claude Fable 5.1 | 10.00 | 50.00 | 0.25 (hit) | 50 % off |
| Anthropic | Claude Opus 5 | 5.00 | 25.00 | 0.50 | 50 % off |
| Anthropic | Claude Sonnet 5 | 2.00 | 10.00 | 0.20 | 50 % off |
| Anthropic | Claude Haiku 4.5 | 1.00 | 5.00 | 0.10 | 50 % off |
| Google | Gemini 3.8 Flash | **0.75** (promo to 2026-12-31; $1.50 after) | **3.75** ($7.50 after) | 0.075 | 50 % off |
| Google | Gemini 3.5 Flash-Lite | 0.30 | 2.50 | 0.03 | $0.15 / $1.25 |
| DeepSeek | `deepseek-flash` (V4.1-Flash) | **0.15** off-peak / 0.30 peak | **0.60** / 1.20 | 0.003 / 0.006 | — |
| DeepSeek | `deepseek-v4-pro` | 0.66 / 1.32 | 1.98 / 3.96 | 0.022 / 0.044 | — |

*DeepSeek peak = 01:00–04:00 and 06:00–10:00 UTC Mon–Fri; off-peak = half price. Note that both windows overlap
European market hours, so a live EU-hours trading loop pays peak rates.*

**GPU rental (RunPod pricing page, "Updated September 13, 2026", fetched 2026-09-14, Secure Cloud on-demand):**
A100 PCIe/SXM 80 GB **$1.59/hr**, H100 PCIe **$2.89/hr**, H100 SXM **$3.49/hr**, H100 NVL **$3.19/hr**,
H200 **$4.59/hr**, B200 **$6.79/hr**, B300 **$7.89/hr**.

### 6.2 Published token/call counts per trading decision (the only two hard numbers in the literature)

| Source | Measurement |
|---|---|
| **TradingAgents** 2412.20138 | **11 LLM calls + 20+ tool calls per prediction.** The authors state this is why they could only backtest 3 months. No token counts, no wall-clock, no dollars |
| **Trading-R1** 2509.11420 Table 8 | **Input tokens per ticker-day**: NVDA mean 18,169 · MSFT 22,684 · AAPL 20,197 · META 19,030 · AMZN 20,349 · TSLA 18,343 · JNJ 34,624 · CVX 24,799 · SPY 4,830 · QQQ 5,285. Raw unprocessed inputs "can easily exceed 80K tokens" before their filtering |

**No paper in this survey publishes seconds-per-decision or dollars-per-decision.** That is a reporting gap, not a
search failure.

### 6.3 Cost of an LLM layer for RL-Trader: 5 tickers, daily, 252 trading days/year (my computation)

Scenario A = one analysis call per ticker per day (20k in / 1.5k out, from Trading-R1's measured distribution).
Scenario B = TradingAgents' published 11-call pipeline (modelled as 132k in / 13k out per ticker-day).
Scenario C = 11 calls with 2 debate rounds (264k in / 26k out).

| Model | A: $/yr | B: $/yr | C: $/yr |
|---|---:|---:|---:|
| DeepSeek `deepseek-flash` (off-peak) | **$5** | **$35** | **$70** |
| OpenAI `gpt-5.6-luna` (batch) | $4 | $26 | $53 |
| OpenAI `gpt-5.6-luna` (standard) | $7 | $53 | $106 |
| Google Gemini 3.5 Flash-Lite | $12 | $91 | $182 |
| DeepSeek `deepseek-v4-pro` (off-peak) | $20 | $142 | $284 |
| Google Gemini 3.8 Flash (promo) | $26 | $186 | $372 |
| Anthropic Claude Haiku 4.5 | $35 | $248 | $496 |
| Anthropic Claude Sonnet 5 | $69 | $496 | $993 |
| OpenAI `gpt-5.6-terra` | $73 | $529 | $1,058 |
| OpenAI `gpt-6-astra` / Claude Fable 5.1 | $346 | $2,482 | $4,964 |

**Live inference is cheap. Research is not.** The real bill is the backtest sweep, because every hyper-parameter
config replays the whole history through the LLM:

| Workload (5 tickers, 11-call pipeline) | deepseek-flash | gpt-5.6-luna | Claude Sonnet 5 | gpt-6-astra |
|---|---:|---:|---:|---:|
| 1-year backtest × 1 config | $35 | $53 | $496 | $2,482 |
| 5-year backtest × 1 config | $174 | $265 | $2,482 | $12,411 |
| **5-year backtest × 20-config W&B sweep** | **$3,478** | **$5,292** | **$49,644** | **$248,220** |
| 20-year backtest (FINSABER-style) × 1 config | $696 | $1,058 | $9,929 | $49,644 |

**This is the decisive operational number for this project.** A FINSABER-grade, bias-mitigated evaluation over
20 years and 100+ symbols with an LLM in the decision loop is financially out of reach at anything above the
cheapest tier — and FINSABER itself had to downgrade to `gpt-4o-mini` for exactly this reason, which is precisely
what broke reproducibility (FinMem Sharpe 2.679 → 0.927). *Caching mitigates but does not solve this: news and price
context change daily, so the cacheable prefix is the system prompt, not the payload.*

### 6.4 Latency

| Source | Number |
|---|---|
| TradingAgents | 11 LLM calls + 20 tool calls serially per ticker-day; wall-clock **NOT REPORTED** |
| Artificial Analysis (third-party benchmark, fetched via deep research) | Reasoning-heavy flagship TTFT can reach **~135 s**; a fast small model shows **~0.87 s TTFT / 287 tok/s** |

For a **daily-close** decision, latency is irrelevant (the whole pipeline fits in the overnight window). For anything
**intraday** it is fatal: 11 sequential calls × 5 tickers at even 5 s/call is ~5 minutes per decision cycle, before
retries. This alone rules out an LLM in an intraday RL-Trader loop.

### 6.5 Determinism and reproducibility

| Provider / tool | Position |
|---|---|
| **OpenAI** (cookbook, verified live) | Chat Completions are "non-deterministic by default"; `seed` + matching parameters gives **"(mostly) consistent"** output; `system_fingerprint` flags backend changes. **No bit-exactness guarantee** |
| **vLLM** `docs.vllm.ai/en/latest/features/batch_invariance/` (verified live) | Batch invariance **can be enabled**: output becomes independent of batch size and request order |
| **Thinking Machines, "Defeating Nondeterminism in LLM Inference"** (verified live) | All kernels in an LLM forward pass are deterministic; the nondeterminism comes from **batch-size-dependent reduction order**. Batch-invariant kernels fix it |
| **StockBench** 2510.02209 | Quantifies the damage: per-model return variance ×10⁻⁴ from **0.074 (DeepSeek-V3)** to **10.190 (GPT-OSS-120B)** on the same 82-day task |
| **ACM TradingAgents reproducibility study** | GPT-4o ±4.2 % SEM across 5 runs on a 3-month GOOGL backtest — the error bar is larger than the alpha being claimed |

**Practical conclusion:** bit-exact reproducibility is achievable **only** by self-hosting an open-weight model on
vLLM with batch invariance enabled and a pinned model hash. Any hosted-API LLM inside the decision loop makes the
backtest un-reproducible by construction, which violates `docs/RESEARCH_PROTOCOL.md`.

---
## 7. Recommendation for RL-Trader

### 7.1 The decision

> **Put the LLM OUT of the decision loop. Use it as an offline feature/knowledge layer only.
> Do not build a TradingAgents-style debate committee on the daily decision path. Do not use an LLM as the policy.**

### 7.2 Why — the evidence chain

| # | Evidence | Source |
|---|---|---|
| 1 | In every bias-mitigated comparison that exists, **PPO beats both FinMem and FinAgent**, and **buy-and-hold beats all three** | FINSABER 2505.07078, 4/4 universes, p ≤ 4e-5 |
| 2 | On a contamination-free window, most frontier LLMs **cannot beat equal-weight B&H**; GPT-5 scored below the passive baseline on both return and Sortino | StockBench 2510.02209 |
| 3 | Published LLM-agent alpha **decays 51–62 % in Sharpe** once the test period crosses the backbone cutoff | Profit Mirage 2510.07920 |
| 4 | Standard LLMs show **−15 to −22 pp alpha decay** from in-sample to out-of-sample periods; the effect is a property of the model, not of the harness | Look-Ahead-Bench 2601.13770 |
| 5 | Swapping the backbone changes the result more than any architectural choice: FinMem Sharpe **2.679 → 0.404**; TradeAgent TSLA **−38.7 % → +21.9 %** | FINSABER; AMA 2510.11695 |
| 6 | LLM infusion into **plain PPO degrades it at every strength tested**, down to 0.1 % | FinRL-DeepSeek 2502.07393 |
| 7 | The one clean positive (SAPPO, +0.35 Sharpe) **dies at ~22 bps of cost** and is 80 %-replicable with FinBERT (1.72 of the 1.90) | §3.5, my computation on `2025.realm-1.12` |
| 8 | An LLM in the loop makes a 20-year FINSABER-grade sweep cost **$3.5k–$248k** and makes the backtest non-reproducible | §6.3, §6.5 |
| 9 | 0 of 19 audited LLM-trading studies reach R3 reproducibility; 1 of 19 states a transaction-cost model | Agentic Trading 2605.19337 |

### 7.3 What the LLM *should* do in RL-Trader

| Use | Justification | Guardrail |
|---|---|---|
| **News/social → structured tags** (sentiment ∈ [−1,1], risk 1–5, event type, entity links) written once into the EKG | FinRL-DeepSeek's channel design; SAPPO's state augmentation; StockBench's news ablation (+0.5 pp for the best model) | Run **offline, batched, once per (date, ticker)**; cache to disk keyed by `(model_id, prompt_hash, date)`; never call at decision time |
| **EKG node/edge extraction** from filings and chatter | This is entity/relation extraction, where LLMs are genuinely strong and where no forward-return leakage path exists if the input is point-in-time | Freeze the input to `publish_timestamp ≤ decision_timestamp` |
| **Research assistant / code generator for reward shaping**, GIFT-style | GIFT 2606.08450 keeps PPO as the action learner and only lets the LLM propose executable state/reward code | Every proposed reward must pass the same `--no-llm` ablation gate |
| **Post-hoc explanation of RL trades** | The one capability the audit (2605.19337) credits without reservation | Explanations must never feed back into the policy |

### 7.4 What the LLM must NOT do

- Not emit BUY/SELL/HOLD at decision time.
- Not be the policy network (FLAG-Trader has no ablation; Trading-R1's RL-only arm loses to SFT-only on 4/6 tickers).
- Not run multi-round debate on the critical path (cost §6.3, latency §6.4, and 2503.13657 / 2502.08788 / 2605.00914
  all say debate gains are small or negative).
- Not be a hosted API in any run that must be reproducible.

### 7.5 Mandatory experimental protocol (adopt these or the result is not evidence)

1. **`--no-llm` arm is compulsory.** Every reported LLM-augmented result ships next to the identical policy with the
   LLM channel zeroed. FinRL-DeepSeek is the template; it is also why we know the honest answer.
2. **Point-in-time freeze + a post-cutoff holdout.** Reserve a test window strictly after the sentiment model's
   release date. Report it separately. (Look-Ahead-Bench's dual-period design is the cheapest version of this.)
3. **Costs always on.** Use FINSABER's schedule ($0.0049/share, $0.99 minimum) or a flat 10 bps, and publish a
   cost-sensitivity curve as in §3.5. A Sharpe without a cost curve is not a result.
4. **≥5 seeds, report mean ± SEM.** StockBench's 138× cross-model variance spread and the ACM study's ±4.2 % SEM
   show single runs are noise.
5. **Universe discipline.** Follow FINSABER: random-N, momentum, and volatility-selected universes including delisted
   names — not 5 hand-picked mega-caps. The master brief's "5 stocks" is fine as a *product*, but the *evidence* must
   come from a wider, survivorship-clean universe.
6. **Pin and log** model id, prompt hash, seed, temperature, top_p, tool outputs, and provider fingerprint. Prefer a
   self-hosted open-weight model on vLLM with batch invariance on.
7. **Deflate the Sharpe.** Report the number of configs searched and apply DSR/PBO before claiming an edge.

### 7.6 Cheapest defensible configuration

| Component | Choice | Annual cost (5 tickers, daily) |
|---|---|---|
| Sentiment/risk/event tagging | `deepseek-flash` off-peak batch, 1 call per ticker-day, cached | **~$5/yr** live; ~$174 for a 5-year backtest pass |
| Fallback if reproducibility is mandatory | self-hosted Qwen/Llama-class model on vLLM batch-invariant, A100 @ $1.59/hr | a 5-year, 5-ticker tagging pass is a few GPU-hours ≈ **<$20** |
| Decision policy | PPO/SAC in TensorTrade, **no LLM** | GPU training only |
| Comparator that must be beaten | Buy-and-hold **and** plain PPO, both cost-adjusted | — |

---

## 8. Where this contradicts the master brief

| Brief assumption | Status | What to change |
|---|---|---|
| "hybrid where … the EKG + sentiment ontology agent form a slower-moving memory/reasoning layer that reshapes the RL policy's inputs/reward" | **Survives, but only in the weak form.** The literature supports an *offline, cached, ablated* feature channel. It does **not** support an online reasoning layer | Keep the hybrid, but specify it as offline + ablated, and budget for it to add **nothing** |
| Implicit "LLM agents are the state of the art to beat" | **Refuted.** B&H and plain PPO beat FinMem/FinAgent in 4/4 bias-mitigated universes | Set the bar at cost-adjusted buy-and-hold and plain PPO, not at FinMem's published Sharpe |
| "benchmarked against … a published RL baseline such as FinRL's PPO agent" | **Correct and important** — PPO is the strongest honest comparator in this literature | Keep. Add B&H as the primary bar |
| Layered-memory design (FinMem-style) as inspiration for the EKG | **Caution.** FinMem's layered memory has the *highest* measured leakage (PC 0.8213) and 5–9× FinAgent's commission ratio from overtrading | If the EKG mimics FinMem's memory, add an explicit point-in-time guard and a turnover penalty |
| 5 hand-picked liquid large-caps as the evaluation universe | **Insufficient as evidence.** This is exactly the selection pattern FINSABER shows inflates results | Trade 5 tickers in production; *evaluate* on FINSABER-style random/momentum/volatility universes |
| Unstated assumption that adding an LLM is cheap | **Partly false.** Live inference is $5–$500/yr, but a 20-config × 5-year sweep is $3.5k–$248k | Force the LLM layer to be offline and cached, or the W&B sweep plan is unaffordable |
| Sentiment from Reddit/StockTwits/Kaggle instead of X | **Consistent with the evidence** — StockBench's ablation says news is worth ~0.5 pp/4 months at best, so paying for premium text is not justified | No change |

---
## 9. Verification log

Protocol: every arXiv paper was confirmed by fetching `https://arxiv.org/abs/<id>` and reading the
`citation_title` / `citation_date` meta tags (the `export.arxiv.org` API returned HTTP 429 for this IP, so the abs
page was used — it is equally authoritative). Every repo was confirmed with the authenticated GitHub REST API.
Every price was read from the vendor's own page, fetched on 2026-09-14.

### 9.1 arXiv papers (all verified in this session)

| Title | URL | Check | Status | Paper date | Verified |
|---|---|---|---|---|---|
| Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to  | https://arxiv.org/abs/2209.05559 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2022/09/12 | 2026-09-14 |
| BloombergGPT: A Large Language Model for Finance | https://arxiv.org/abs/2303.17564 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2023/03/30 | 2026-09-14 |
| PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for F | https://arxiv.org/abs/2306.05443 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2023/06/08 | 2026-09-14 |
| FinGPT: Open-Source Financial Large Language Models | https://arxiv.org/abs/2306.06031 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2023/06/09 | 2026-09-14 |
| Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General | https://arxiv.org/abs/2306.12659 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2023/06/22 | 2026-09-14 |
| Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment | https://arxiv.org/abs/2308.00016 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2023/07/31 | 2026-09-14 |
| TradingGPT: Multi-Agent System with Layered Memory and Distinct Characters for | https://arxiv.org/abs/2309.03736 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2023/09/07 | 2026-09-14 |
| FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Chara | https://arxiv.org/abs/2311.13743 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2023/11/23 | 2026-09-14 |
| QuantAgent: Seeking Holy Grail in Trading by Self-Improving Large Language Mod | https://arxiv.org/abs/2402.03755 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/02/06 | 2026-09-14 |
| Alpha-GPT 2.0: Human-in-the-Loop AI for Quantitative Investment | https://arxiv.org/abs/2402.09746 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/02/15 | 2026-09-14 |
| A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversifi | https://arxiv.org/abs/2402.18485 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/02/28 | 2026-09-14 |
| FinRobot: An Open-Source AI Agent Platform for Financial Applications using La | https://arxiv.org/abs/2405.14767 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/05/23 | 2026-09-14 |
| Efficient Optimal Control of Open Quantum Systems | https://arxiv.org/abs/2405.19245 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/05/29 | 2026-09-14 |
| FinVerse: An Autonomous Agent System for Versatile Financial Analysis | https://arxiv.org/abs/2406.06379 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/06/10 | 2026-09-14 |
| LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock | https://arxiv.org/abs/2406.10811 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/06/16 | 2026-09-14 |
| A Survey of Large Language Models for Financial Applications: Progress, Prospe | https://arxiv.org/abs/2406.11903 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/06/15 | 2026-09-14 |
| FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcem | https://arxiv.org/abs/2407.06567 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/07/09 | 2026-09-14 |
| When AI Meets Finance (StockAgent): Large Language Model-based Stock Trading i | https://arxiv.org/abs/2407.18957 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/07/15 | 2026-09-14 |
| Large Language Model Agent in Financial Trading: A Survey | https://arxiv.org/abs/2408.06361 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/07/26 | 2026-09-14 |
| TradExpert: Revolutionizing Trading with Mixture of Expert LLMs | https://arxiv.org/abs/2411.00782 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/10/16 | 2026-09-14 |
| Financial News-Driven LLM Reinforcement Learning for Portfolio Management | https://arxiv.org/abs/2411.11059 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/11/17 | 2026-09-14 |
| INVESTORBENCH: A Benchmark for Financial Decision-Making Tasks with LLM-based  | https://arxiv.org/abs/2412.18174 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/12/24 | 2026-09-14 |
| TradingAgents: Multi-Agents LLM Financial Trading Framework | https://arxiv.org/abs/2412.20138 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2024/12/28 | 2026-09-14 |
| FinRL-DeepSeek: LLM-Infused Risk-Sensitive Reinforcement Learning for Trading  | https://arxiv.org/abs/2502.07393 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/02/11 | 2026-09-14 |
| Stop Overvaluing Multi-Agent Debate -- We Must Rethink Evaluation and Embrace  | https://arxiv.org/abs/2502.08788 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/02/12 | 2026-09-14 |
| FLAG-Trader: Fusion LLM-Agent with Gradient-based Reinforcement Learning for F | https://arxiv.org/abs/2502.11433 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/02/17 | 2026-09-14 |
| HedgeAgents: A Balanced-aware Multi-agent Financial Trading System | https://arxiv.org/abs/2502.13165 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/02/17 | 2026-09-14 |
| AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract | https://arxiv.org/abs/2502.16789 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/02/24 | 2026-09-14 |
| Agent Trading Arena: A Study on Numerical Understanding in LLM-Based Agents | https://arxiv.org/abs/2502.17967 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/02/25 | 2026-09-14 |
| Why Do Multi-Agent LLM Systems Fail? | https://arxiv.org/abs/2503.13657 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/03/17 | 2026-09-14 |
| Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement L | https://arxiv.org/abs/2503.16252 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/03/20 | 2026-09-14 |
| FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agen | https://arxiv.org/abs/2504.02281 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/04/03 | 2026-09-14 |
| LLMs Meet Finance: Fine-Tuning Foundation Models for the Open FinLLM Leaderboa | https://arxiv.org/abs/2504.13125 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/04/17 | 2026-09-14 |
| The Memorization Problem: Can We Trust LLMs&#39; Economic Forecasts? | https://arxiv.org/abs/2504.14765 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/04/20 | 2026-09-14 |
| Can LLM-based Financial Investing Strategies Outperform the Market in Long Run | https://arxiv.org/abs/2505.07078 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/05/11 | 2026-09-14 |
| FinDPO: Financial Sentiment Analysis for Algorithmic Trading through Preferenc | https://arxiv.org/abs/2507.18417 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/07/24 | 2026-09-14 |
| HARLF: Hierarchical Reinforcement Learning and Lightweight LLM-Driven Sentimen | https://arxiv.org/abs/2507.18560 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/07/24 | 2026-09-14 |
| Language Model Guided Reinforcement Learning in Quantitative Trading | https://arxiv.org/abs/2508.02366 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/08/04 | 2026-09-14 |
| QuantHarness: Price-Driven Multi-Agent LLMs for High-Frequency Trading | https://arxiv.org/abs/2509.09995 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/09/12 | 2026-09-14 |
| Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning | https://arxiv.org/abs/2509.11420 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/09/14 | 2026-09-14 |
| StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets? | https://arxiv.org/abs/2510.02209 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/10/02 | 2026-09-14 |
| Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents | https://arxiv.org/abs/2510.07920 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/10/09 | 2026-09-14 |
| When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents | https://arxiv.org/abs/2510.11695 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/10/13 | 2026-09-14 |
| Detecting Lookahead Bias in LLM Forecasts | https://arxiv.org/abs/2512.23847 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2025/12/29 | 2026-09-14 |
| Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time | https://arxiv.org/abs/2601.13770 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/01/20 | 2026-09-14 |
| Toward Expert Investment Teams:A Multi-Agent LLM System with Fine-Grained Trad | https://arxiv.org/abs/2602.23330 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/02/26 | 2026-09-14 |
| DatedGPT: Preventing Lookahead Bias in Large Language Models with Time-Aware P | https://arxiv.org/abs/2603.11838 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/03/12 | 2026-09-14 |
| The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogen | https://arxiv.org/abs/2605.00914 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/04/29 | 2026-09-14 |
| Agentic Trading: When LLM Agents Meet Financial Markets | https://arxiv.org/abs/2605.19337 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/05/19 | 2026-09-14 |
| Summoning the Oracle to Slay It: Mitigating Look-Ahead Bias in Financial Backt | https://arxiv.org/abs/2605.24564 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/05/23 | 2026-09-14 |
| PortBench: A Correlation-Aware, Full-Pipeline Benchmark for LLM-Driven Portfol | https://arxiv.org/abs/2605.27887 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/05/27 | 2026-09-14 |
| From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on | https://arxiv.org/abs/2605.28359 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/05/27 | 2026-09-14 |
| Representation Signatures and Risk-Feedback Alignment in LLM Trading Agents | https://arxiv.org/abs/2605.28850 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/05/16 | 2026-09-14 |
| Beyond Agent Architecture: Execution Assumptions and Reproducibility in LLM-Ba | https://arxiv.org/abs/2606.08285 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/06/06 | 2026-09-14 |
| GIFT: LLM-Guided State-Reward Interface for Financial Reinforcement Learning | https://arxiv.org/abs/2606.08450 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/06/07 | 2026-09-14 |
| Fin-Analyst at FinMMEval 2026 Task 3: A Live Hybrid Trading Agent with LLM Spe | https://arxiv.org/abs/2607.12233 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/07/14 | 2026-09-14 |
| Second Thought: Reasoning in Parallel as LLM Agents Act and Observe | https://arxiv.org/abs/2608.13667 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/08/13 | 2026-09-14 |
| Mint-Agent: Introducing Finance-Native Agentic Foundation Models | https://arxiv.org/abs/2608.16386 | `verify_abs` (arXiv abs page, citation_title meta) | OK 200 | 2026/08/17 | 2026-09-14 |

**Deleted / rejected during verification:** `2405.19245` was fetched while chasing a candidate ID and resolved to
*"Efficient Optimal Control of Open Quantum Systems"* — unrelated to finance, so it is **not cited anywhere in this
dossier**. `pillar/quantagent` (3★) and `effective-p/FinAgent` (4★) exist but are **not credible official releases**
and are labelled as such rather than presented as the systems' code.

### 9.2 GitHub repositories

| Repo | URL | Check | Status | Verified |
|---|---|---|---|---|
| AI4Finance-Foundation/FinGPT | https://github.com/AI4Finance-Foundation/FinGPT | `gh api repos/AI4Finance-Foundation/FinGPT` | OK — 21250★, push 2026-09-14, lic MIT | 2026-09-14 |
| AI4Finance-Foundation/FinRL | https://github.com/AI4Finance-Foundation/FinRL | `gh api repos/AI4Finance-Foundation/FinRL` | OK — 16282★, push 2026-07-13, lic MIT | 2026-09-14 |
| AI4Finance-Foundation/FinRobot | https://github.com/AI4Finance-Foundation/FinRobot | `gh api repos/AI4Finance-Foundation/FinRobot` | OK — 7984★, push 2026-09-11, lic Apache-2.0 | 2026-09-14 |
| AgenticFinLab/portbench | https://github.com/AgenticFinLab/portbench | `gh api repos/AgenticFinLab/portbench` | OK — 9★, push 2026-09-12, lic Apache-2.0 | 2026-09-14 |
| ChenYXxxx/stockbench | https://github.com/ChenYXxxx/stockbench | `gh api repos/ChenYXxxx/stockbench` | OK — 179★, push 2025-10-28, lic Apache-2.0 | 2026-09-14 |
| JansenAnalytics/hedge-agents | https://github.com/JansenAnalytics/hedge-agents | `gh api repos/JansenAnalytics/hedge-agents` | OK — 3★, push 2026-04-21, lic None | 2026-09-14 |
| LLMQuant/awesome-trading-agents | https://github.com/LLMQuant/awesome-trading-agents | `gh api repos/LLMQuant/awesome-trading-agents` | OK — 792★, push 2026-08-13, lic CC0-1.0 | 2026-09-14 |
| MingyuJ666/Stockagent | https://github.com/MingyuJ666/Stockagent | `gh api repos/MingyuJ666/Stockagent` | OK — 701★, push 2026-06-16, lic None | 2026-09-14 |
| Open-Finance-Lab/FinRL_Contest_2025 | https://github.com/Open-Finance-Lab/FinRL_Contest_2025 | `gh api repos/Open-Finance-Lab/FinRL_Contest_2025` | OK — 66★, push 2025-10-20, lic None | 2026-09-14 |
| RndmVariableQ/AlphaAgent | https://github.com/RndmVariableQ/AlphaAgent | `gh api repos/RndmVariableQ/AlphaAgent` | OK — 410★, push 2026-07-03, lic None | 2026-09-14 |
| Shafiya0101/signal-density-llm-trading | https://github.com/Shafiya0101/signal-density-llm-trading | `gh api repos/Shafiya0101/signal-density-llm-trading` | OK — 0★, push 2026-06-18, lic MIT | 2026-09-14 |
| Sidharth-bhat/HARLF-portfolio-optimization | https://github.com/Sidharth-bhat/HARLF-portfolio-optimization | `gh api repos/Sidharth-bhat/HARLF-portfolio-optimization` | OK — 1★, push 2025-11-30, lic MIT | 2026-09-14 |
| TanTaf/Finmem | https://github.com/TanTaf/Finmem | `gh api repos/TanTaf/Finmem` | OK — 0★, push 2026-09-11, lic None | 2026-09-14 |
| TauricResearch/TradingAgents | https://github.com/TauricResearch/TradingAgents | `gh api repos/TauricResearch/TradingAgents` | OK — 105755★, push 2026-09-07, lic Apache-2.0 | 2026-09-14 |
| The-FinAI/FinCon | https://github.com/The-FinAI/FinCon | `gh api repos/The-FinAI/FinCon` | OK — 68★, push 2026-02-27, lic None | 2026-09-14 |
| The-FinAI/PIXIU | https://github.com/The-FinAI/PIXIU | `gh api repos/The-FinAI/PIXIU` | OK — 887★, push 2025-03-04, lic MIT | 2026-09-14 |
| TheFinAI/PIXIU | https://github.com/TheFinAI/PIXIU | `gh api repos/TheFinAI/PIXIU` | **FAIL — gh: Not Found (HTTP 404)
** | 2026-09-14 |
| benstaf/FinRL_DeepSeek | https://github.com/benstaf/FinRL_DeepSeek | `gh api repos/benstaf/FinRL_DeepSeek` | OK — 331★, push 2025-04-08, lic MIT | 2026-09-14 |
| chancefocus/PIXIU | https://github.com/chancefocus/PIXIU | `gh api repos/chancefocus/PIXIU` | OK — 887★, push 2025-03-04, lic MIT | 2026-09-14 |
| effective-p/FinAgent | https://github.com/effective-p/FinAgent | `gh api repos/effective-p/FinAgent` | OK — 4★, push 2026-06-06, lic None | 2026-09-14 |
| felis33/INVESTOR-BENCH | https://github.com/felis33/INVESTOR-BENCH | `gh api repos/felis33/INVESTOR-BENCH` | OK — 30★, push 2025-09-13, lic MIT | 2026-09-14 |
| hongha5192-bit/AlphaAgent | https://github.com/hongha5192-bit/AlphaAgent | `gh api repos/hongha5192-bit/AlphaAgent` | OK — 6★, push 2026-04-06, lic MIT | 2026-09-14 |
| pillar/quantagent | https://github.com/pillar/quantagent | `gh api repos/pillar/quantagent` | OK — 3★, push 2025-11-17, lic MIT | 2026-09-14 |
| pipiku915/FinMem-LLM-StockTrading | https://github.com/pipiku915/FinMem-LLM-StockTrading | `gh api repos/pipiku915/FinMem-LLM-StockTrading` | OK — 959★, push 2024-08-18, lic MIT | 2026-09-14 |
| remlih/QuantAgent | https://github.com/remlih/QuantAgent | `gh api repos/remlih/QuantAgent` | OK — 1★, push 2026-04-18, lic MIT | 2026-09-14 |
| tensortrade-org/tensortrade | https://github.com/tensortrade-org/tensortrade | `gh api repos/tensortrade-org/tensortrade` | OK — 7117★, push 2026-02-19, lic Apache-2.0 | 2026-09-14 |
| waylonli/FINSABER | https://github.com/waylonli/FINSABER | `gh api repos/waylonli/FINSABER` | OK — 145★, push 2026-08-05, lic Apache-2.0 | 2026-09-14 |
| x9yang/Stockagent | https://github.com/x9yang/Stockagent | `gh api repos/x9yang/Stockagent` | OK — 0★, push 2026-05-06, lic None | 2026-09-14 |

### 9.3 Non-arXiv URLs

| Title | URL | Check | Status | Verified |
|---|---|---|---|---|
|  | https://aclanthology.org/2025.findings-acl.716.pdf | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| Large Language Model Agents in Finance: A Survey Bridging Research, Pr | https://aclanthology.org/2025.findings-emnlp.972 | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| Leveraging LLM-based sentiment analysis for portfolio optimization wit | https://aclanthology.org/2025.realm-1.12/ | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| Gemini Developer API pricing &nbsp;/&nbsp; Gemini API &nbsp;/&nbsp; Go | https://ai.google.dev/gemini-api/docs/pricing | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| Models &amp; Pricing / DeepSeek API Docs | https://api-docs.deepseek.com/quick_start/pricing | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| [2605.19337] Agentic Trading: When LLM Agents Meet Financial Markets | https://arxiv.org/abs/2605.19337 | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| [2606.08285] Beyond Agent Architecture: Execution Assumptions and Repr | https://arxiv.org/abs/2606.08285 | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| Pricing / OpenAI API | https://developers.openai.com/api/docs/pricing | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| How to make your completions outputs consistent with the new seed para | https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| Just a moment... | https://dl.acm.org/doi/10.1145/3800973.3801029 | `verify_url` HTTP GET | **FAIL 403** | 2026-09-14 |
| Batch Invariance - vLLM | https://docs.vllm.ai/en/latest/features/batch_invariance/ | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
|  | https://github.com/TauricResearch/Trading-R1 | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| Overview / FinRL Contest 2025 | https://open-finance-lab.github.io/FinRL_Contest_2025/ | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| Verifying your browser / OpenReview | https://openreview.net/forum?id=KBC0xQJ4uz | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| Pricing - Claude Platform Docs | https://platform.claude.com/docs/en/about-claude/pricing | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| Defeating Nondeterminism in LLM Inference - Thinking Machines Lab | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| PiT Inference — Point-in-Time LLMs for Finance | https://www.pitinference.com/ | `verify_url` HTTP GET | OK 200 | 2026-09-14 |
| GPU Cloud Pricing / Per-Second H100, A100, RTX / Runpod | https://www.runpod.io/pricing | `verify_url` HTTP GET | OK 200 | 2026-09-14 |

### 9.4 Pricing / documentation pages parsed in-session

| Page | URL | Check | Status | Fetched |
|---|---|---|---|---|
| openai pricing page | https://developers.openai.com/api/docs/pricing | direct HTTP GET + price table parsed in-session | OK 200, 20596 chars | 2026-09-14 |
| anthropic pricing page | https://platform.claude.com/docs/en/about-claude/pricing | direct HTTP GET + price table parsed in-session | OK 200, 29638 chars | 2026-09-14 |
| gemini pricing page | https://ai.google.dev/gemini-api/docs/pricing | direct HTTP GET + price table parsed in-session | OK 200, 57634 chars | 2026-09-14 |
| deepseek pricing page | https://api-docs.deepseek.com/quick_start/pricing | direct HTTP GET + price table parsed in-session | OK 200, 3057 chars | 2026-09-14 |
| together pricing page | https://www.together.ai/pricing | direct HTTP GET + price table parsed in-session | OK 200, 13352 chars | 2026-09-14 |
| runpod pricing page | https://www.runpod.io/pricing | direct HTTP GET + price table parsed in-session | OK 200, 11328 chars | 2026-09-14 |

### 9.5 Known verification gaps (stated plainly)

| Item | Problem | How it is handled in this dossier |
|---|---|---|
| ACM DOI `10.1145/3800973.3801029` ("Reproducibility in the TradingAgents Framework") | The ACM page returns **HTTP 403** to automated fetch (Cloudflare interstitial). Its numbers come from indexed excerpts, not a direct read | Cited with an explicit **"partially verified"** marker in §5.2. Its headline claim (TradingAgents loses to GOOGL B&H) is *directionally corroborated* by FINSABER and StockBench, which are fully verified |
| OpenReview `KBC0xQJ4uz` (FinRankGRPO) | Page loads a browser-check interstitial; no result table accessible | Listed in §4.1 as **UNQUANTIFIED**; no number from it is used |
| Deflated Sharpe Ratio (Bailey & López de Prado, *J. Portfolio Management* 40(5):94, 2014) | Publisher page is paywalled | Cited by bibliographic reference only, no number claimed |
| `Shafiya0101/signal-density-llm-trading` | Repo verified (0★, MIT, push 2026-06-18) but it is **unpublished, unreviewed** work | Labelled "weak evidence" in §3.1; no conclusion rests on it |
| "PiT-Inference" / `Pitinf-*` models in Look-Ahead-Bench | Vendor site `pitinference.com` verified live (HTTP 200), but the models are **commercial and unbenchmarked by third parties** | Reported only as what the paper says, with the vendor relationship flagged |
| Per-decision **latency** for any LLM trading pipeline | No paper publishes it | Reported as **NOT REPORTED** in §6.4; my latency comment is an explicit arithmetic estimate, labelled as such |

### 9.6 Numbers in this dossier that are MY computation, not published

| Number | Where | Method |
|---|---|---|
| SAPPO cost-fragility table (edge dies at ~22 bps) | §3.5 | `(0.12 − 0.035) × 252 × bps/10⁴` applied to the paper's own turnover and return figures |
| $/year LLM layer for 5 tickers, scenarios A/B/C | §6.3 | Trading-R1's measured ~20k input tokens/ticker-day × TradingAgents' published 11 calls × vendor prices fetched 2026-09-14 × 252 days × 5 tickers |
| Backtest-sweep cost table | §6.3 | same unit cost × (years × 252 × 5 tickers × configs) |
| "11 sequential calls × 5 tickers at 5 s/call ≈ 5 min/cycle" | §6.4 | arithmetic on the published 11-call count |

---

## 10. Source files

Raw deep-research outputs and citation dumps are kept under `docs/research/_raw/`:
`llm_P1_systems.md`, `llm_P2_critiques.md`, `llm_P3_hybrid.md`, `llm_P4_rlhf.md`, `llm_P5_ops.md`
(+ `*_cites.json`). These are **unverified generator output**; only claims re-checked above appear in this dossier.
