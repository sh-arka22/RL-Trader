# 03 — The Self-Evolving Knowledge Graph (EKG): graphs, agent memory, continual RL, and Prime Intellect

**Dossier owner:** `ekg-continual` research subagent
**Date of all live fetches:** 2026-09-14
**Protocol:** `docs/RESEARCH_PROTOCOL.md`. Every arXiv ID, repo and URL named below was verified in
this session (`rt.verify_arxiv`, `rt.gh_repo`, `rt.verify_url`). Two publisher pages returned HTTP 403
to an automated client and are marked **BLOCKED** rather than silently kept.
**Counts:** 76 arXiv IDs verified, 15 GitHub repos verified, 30 non-arXiv URLs checked (28 OK, 2 blocked).

---

## 0. Executive decisions (read this if you read nothing else)

| # | Decision | Evidence basis | Confidence |
|---|---|---|---|
| D1 | **Build the EKG. Do NOT let it touch the reward.** Primary coupling = **gated state augmentation** (graph embedding concatenated to the PPO/SAC observation through a learned gate). Fallback = **retrieval-conditioned policy** (top-k analogue episodes). | Potential-based shaping is the only shaping form with a policy-invariance guarantee (Ng/Harada/Russell); an arbitrary KG bonus has no such guarantee, and a wrong fact then *becomes the objective*. State features can be down-weighted by the policy; rewards cannot. | High |
| D2 | **Never let the EKG veto a trade.** Action masking is reserved for *deterministic* constraints (cash, position limits, market hours, borrow). KG beliefs may only shrink position size. | Invalid-action masking is a valid policy gradient and hugely sample-efficient (arXiv:2006.14171), but a wrong mask deletes profitable legal actions permanently. | High |
| D3 | **Storage = DuckDB + Parquet (bitemporal edge table) + NetworkX in-memory view.** Neo4j/Memgraph only if we ever need concurrent multi-writer online serving. **Kuzu is dead** — `kuzudb/kuzu` is archived read-only since 2025-10-10. | Verified: `kuzudb/kuzu` archived=True, last commit 2025-10-10, 4,025 stars. Graphiti explicitly deprecated its Kuzu backend. | High |
| D4 | **Assume the EKG gives a *small* edge, and design the test to detect a small edge.** GNN/KG gains in finance are real but mostly reported gross of costs, with undisclosed splits. | RSR and THGNN explicitly ignore transaction costs; MDGNN's stated backtest interval `01/01/2020–02/31/2023` is a malformed date and its split is undisclosed. | High |
| D5 | **Self-evolving agent memory has NO demonstrated transfer to trading.** All headline numbers (LoCoMo, LongMemEval, DMR, HotpotQA) are QA benchmarks. Treat A-MEM / Zep / Mem0 as *engineering patterns*, not as evidence of alpha. | Part B, section B.6. This directly qualifies the master brief's EKG premise. | High |
| D6 | **Prime Intellect's "continual self-improvement" is not online weight learning.** Their published loop is: verifiable **environments** → async **RL** runs → new model. The only documented *self-improving agent* loop (Prime Agent `/refine`) edits **prompts, memories, skills and subagent specs**, not weights. That is exactly the right template for the EKG. | Part D. Primary sources only. | High |
| D7 | **Retrain cadence must be an experiment, not an assumption.** No source supports "retrain every N days". Continuous gradient updating is *not* a safe default: PPO collapsed to zero episodes completed after 20M steps on a **stationary** Ant-v3 task. | Nature 632 (2024), *Loss of plasticity in deep continual learning*. | High |
| D8 | **Ablation is the deliverable, not the EKG.** Same seeds, same window, same costs, ≥10 (target 20) seeds, IQM + stratified bootstrap CI + probability-of-improvement (rliable), plus deflated Sharpe and stationary bootstrap on the return series. | arXiv:2108.13264; Bailey & López de Prado DSR; Politis & Romano. | High |

---

## Part A — Financial knowledge graphs and graph learning (2022–2026)

### A.1 What a "financial KG" actually is — six distinct constructions

| KG family | Concrete instance | Node / edge semantics | Time handling | Verified source |
|---|---|---|---|---|
| Dynamic news KG | **FinDKG** (2024) | ~400,000 Wall Street Journal articles 1999–2023; **13,645 entities, 15 relations**; splits 119,549 / 11,444 / 13,069 | Article timestamps; rolling 1-month graphs assembled weekly | arXiv:2407.10909 |
| Value-chain graph | **FS-GCLSTM** (2023, rev. 2025) using LSEG supplier–customer links | Firms as nodes, directed supplier→customer edges with confidence + timestamp | Rolling window, most-recent relations | arXiv:2303.09406 |
| Relation ranking graph | **RSR / Temporal Graph Convolution** | Sector + Wiki company relations over NASDAQ/NYSE | "Time-sensitive" relation strength | arXiv:1809.09441 |
| Correlation/heterogeneous graph | **THGNN** | Daily regenerated company-relation graph from price correlation (positive/negative edges) | Graph rebuilt **each trading day** from trailing 20 days | arXiv:2305.08740; repo `TongjiFinLab/THGNN` |
| Multi-relational dynamic graph | **MDGNN** | Industry, investment-bank, stock-pair, stock–industry, industry–industry meta-paths | Dynamic, but split boundaries undisclosed | arXiv:2402.06633 |
| LLM-constructed stock KG | **KG construction for stock markets w/ LLM reasoning** (2026) | Schema + LLM multi-hop reasoning over it | Not a trading backtest | arXiv:2601.11528 |
| Momentum-spillover network | **Network Momentum across Asset Classes** | 64 continuous futures; learned linear graph of momentum spillover | Linear, interpretable graph learning | arXiv:2308.11294 |

### A.2 GNNs for stock prediction — what was actually measured

**Every row states the cost assumption. "Ignored" is a finding, not a gap.**

| Method | Universe | Test window | Costs | Headline metric vs baseline |
|---|---|---|---|---|
| **RSR / TGC** (arXiv:1809.09441) | NASDAQ 3,274 stocks; NYSE 3,163 stocks | Transaction records 2013-01-02 → 2017-12-08; **exact train/test boundary not reported** | **Explicitly ignored** | NYSE cumulative investment return ratio: Rank_LSTM **0.68**, GCN **0.97**, RSR_E **1.00**, RSR_I **1.06**. On NASDAQ the relation-aware variants fall **below** Rank_LSTM. |
| **HATS** (arXiv:1908.07999) | 431 firms (mostly S&P 500 with relational data) | Prices 2013-02-08 → 2019-06-17, 1,174 trading days | **Not reported** (no commission, spread or slippage stated) | Sharpe **+19.8%** and F1 **+3%** vs existing baselines; per-baseline deltas not reported |
| **THGNN** (arXiv:2305.08740) | S&P 500 and CSI 300 | Trailing 20 days; **exact calendar split not reported** | **Explicitly ignored** | Full model ACC **0.579**, ARR **0.665**, ASR **1.421**, CR **1.804**, IR **1.340** |
| **THGNN ablation** (same) | S&P 500 | same | ignored | Removing the temporal module costs **−0.040 ACC, −0.179 ARR, −0.457 ASR**; removing heterogeneity costs **−0.026 ACC, −0.065 ARR, −0.142 ASR** |
| **MDGNN** (arXiv:2402.06633) | CSI100, CSI300 | Stated `01/01/2020–02/31/2023` — **a malformed date**; split undisclosed | **Not reported** | CSI300 full: IC **0.0322**, IR **0.2488**, CR **0.9828**, Prec@30 **0.5232**. Remove meta-path → IC **0.0216**, IR **0.1723**, CR **0.7502**, Prec@30 **0.5076** |
| **FS-GCLSTM** (arXiv:2303.09406) | Eurostoxx 600, S&P 500 | Rolling window 3,000 trading days advanced by 300; exact calendar dates not reported | **1 basis point**, daily-rebalanced long-only equal weight | Highest annualised return / Sharpe / Sortino vs ARIMA, FCL, LSTM, GConvGRU; **numeric deltas not exposed** |
| **GNN + rolling window** (arXiv:1909.10660) | Nikkei 225, ~20 years | Rolling window analysis | Not stated in retrieved text | +29.5% return ratio and 2.2× Sharpe vs market benchmark; **+6.32% return and 1.3× Sharpe vs LSTM baseline** |
| **DGDNN** (arXiv:2401.01846) | next-day movement classification | not fully disclosed | not reported | +9.06% classification accuracy, +0.09 MCC, +0.06 F1 vs SOTA |
| **DeepPocket / graph-conv RL** (arXiv:2105.08664) | portfolio management | see paper | see paper | Graph-conv RL portfolio framework — architecture evidence, not an audited net-alpha claim |
| **EarnMore** (arXiv:2311.10801) | Customisable stock pools | see paper | see paper | Maskable stock representation for changing universes — directly relevant to our 5-stock pool |

**"ESTIMATE" was not found as a verifiable finance paper in this session — the citation is dropped rather than guessed.**

### A.3 GNN + RL for portfolios — what the graph actually contributed

| Study | Graph role | Universe / window / cost | Ablation result | Verified |
|---|---|---|---|---|
| Graph-attention heterogeneous multi-agent DRL (Sci Rep 2025) | GAT over asset relations + heterogeneous multi-agent allocation | S&P 500 train 2000–2016 / val 2017–2020 / **test 2021–2023**; NASDAQ100 train 2005–2017 / val 2018–2020 / test 2021–2023; Russell 2000 train 2010–2018 / val 2019–2021 / test 2022–2023. **Transaction-cost component 0.02–0.03** | Removing graph attention degrades performance, but **no complete numeric ablation table**; authors report **8–12% out-of-sample degradation** from in-sample and >500,000 parameters | `nature.com/articles/s41598-025-32408-w` (via deep-research fetch) |
| GraphSAGE + PPO (Expert Syst. Appl. 2024) | GraphSAGE feature extractor feeding PPO; nodes = stocks, bonds, indices | Three datasets; **exact out-of-sample dates and costs not stated** | Share-Extractor and Separate-Extractor GRL significantly beat **PPO with no feature extractor**, and beat Equal Weight and the index | ScienceDirect S0957417423025290 |
| GPM (R-GCN portfolio RL) | Heterogeneous company-relation graph → R-GCN state | NASDAQ + NYSE, sector-industry + Wiki relations | Higher accumulated profit than iCNN, EIIE, EI3, SARL, AlphaStock; **no numeric margin, no graph-removal causal estimate** | ScienceDirect S0925231222005021 — **BLOCKED (HTTP 403)**, cite as secondary |

**Reading:** the defensible claim is *representational* — removing the graph encoder degrades the authors' own policy. The claim that cannot be defended from published evidence is **robust net alpha after costs across markets**.

### A.4 GraphRAG / RAG-over-KG for finance — the evidence is thinner than the hype

| System / benchmark | What is verified | Vector-RAG delta | Cost / latency |
|---|---|---|---|
| **FinanceBench** (arXiv:2311.11944) | 10,231 open-book questions on public companies, with evidence strings | Benchmark itself does **not** evaluate GraphRAG vs vector RAG | not reported |
| **HybridRAG** (arXiv:2408.04948) | 50 Nifty-50 earnings-call transcripts, Q1 FY2024, ~16 questions each | **vector-only / KG-only / hybrid scores not exposed in retrieved text** | not reported |
| **FinSage** (arXiv:2504.14493) | Multi-aspect RAG for financial filings | **+24.06% accuracy over best baseline on FinanceBench**; 92.51% recall on 75 expert questions — note this is *multi-aspect RAG*, **not** a KG ablation | not reported |
| **RAG vs GraphRAG systematic evaluation** (arXiv:2502.11371) | Head-to-head on standard QA + query-focused summarisation | Finds task-dependent wins; no universal GraphRAG superiority | discussed |
| **FinReflectKG** (arXiv:2508.17906) | Agentic financial KG construction from SEC filings; reflection agent reaches **64.8% CheckRules compliance**, beating single-pass and multi-pass | KG *construction* quality, not retrieval accuracy delta | not reported |
| **FinReflectKG–HalluBench** (arXiv:2603.20252) | 2026 GraphRAG hallucination benchmark over SEC 10-Ks | retrieval-accuracy delta not exposed | not reported |
| **FinRAGBench-V** (arXiv:2505.17471) | Multimodal financial RAG with visual citation | not a KG-vs-vector experiment | not reported |
| **Microsoft GraphRAG** (`microsoft.github.io/graphrag/`, HTTP 200) | Architecture: KG + community summaries at query time | no finance-specific controlled delta published | indexing cost is the known drawback |

**Conclusion for RL-Trader:** a GraphRAG layer is justifiable for **entity resolution, event context and auditability**. It is *not* justifiable as a claimed alpha source. Do not spend Stage-N budget on GraphRAG retrieval quality before the price-only baseline is beaten.

### A.5 HONEST NEGATIVES (Part A)

| Negative | Evidence | Consequence for us |
|---|---|---|
| **Knowledge graph made it worse in 6 of 7 cases** | *Modeling Momentum Spillover with Economic Links Discovered from Financial Documents* (ACM, DOI 10.1145/3604237.3626862) states: "the LSTM model without the knowledge graph information outperforms **6 out of 7** graph convolutional network solutions with a single economic link." | This is the single most important finding in Part A. A no-graph LSTM is a *hard* baseline, not a straw man. |
| **Simple models win on average** | Comparative study over **55 stock markets and 65 two-month periods, Jan 2011 → Jul 2022**: SVM best at mean accuracy **0.5164**; average accuracy across all algorithms ≈ **51%**; LSTM among the worst. (Int. J. Data Sci. Anal., 2025) | 51% accuracy is the population mean. Any EKG gain must be measured against that, after costs. |
| **A naive persistence model beat LSTM** | 12 Tehran Stock Exchange stocks, chronological 70/30 split: day-to-day LSTM performed **worse than constant-price prediction**; proposed models only marginally beat persistence. (Humanit. Soc. Sci. Commun. 2025, s41599-025-04761-8) | Add a **persistence (no-change) control** to the baseline harness. If the EKG agent cannot beat persistence net of costs, it is noise. |
| **Costs are routinely ignored** | RSR and THGNN explicitly ignore transaction costs; HATS, MDGNN and the GraphSAGE-PPO study disclose no cost model. | Never compare our net numbers to their gross numbers. |
| **Look-ahead bias is systemic and now benchmarked** | *Look-Ahead-Bench* (arXiv:2601.13770) is a 2026 standardized benchmark of look-ahead bias in point-in-time LLMs for finance; *Assessing Look-Ahead Bias in GPT Sentiment* (arXiv:2309.17322 — **UNVERIFIED, not checked this session**) | Every EKG edge needs `first_public_at`, not just `valid_from`. See E.2. |
| **Fixed sector graph vs learned graph: no universal winner** | RSR's relation-aware variants beat Rank_LSTM on NYSE but lose on NASDAQ (arXiv:1809.09441); MDGNN's meta-path ablation is positive on CSI300 (arXiv:2402.06633). | Test **sector-only**, **correlation-only**, **learned**, and **hybrid** as separate arms. Do not assume the rich graph wins. |
| **No peer-reviewed GNN-for-stocks replication-failure paper was found** | Searched; not found. | Reported as an evidence gap. Do **not** cite a replication failure that does not exist. |

---

## Part B — Self-improving / self-evolving agent memory

### B.1 Per-system comparison (write rule → retrieval → measured benefit → cost → failure mode)

| System | How memory WRITES/updates itself | Retrieval | Measured benefit (benchmark, metric, baseline) | Cost / latency | Failure modes |
|---|---|---|---|---|---|
| **Generative Agents** (arXiv:2304.03442) | Natural-language memory stream + periodic **reflection** triggered by accumulated importance; reflections re-enter memory. **No deletion policy.** | Score = relevance + recency + importance | 25-agent Smallville simulation; behavioural evaluation only — **no QA score** | not reported | Reflection can amplify an early wrong observation; unbounded stream |
| **MemGPT / Letta** (arXiv:2310.08560; `letta-ai/letta` 24,733★, Apache-2.0, last commit 2026-09-10) | LLM self-edits a virtual memory hierarchy via function calls; context-pressure triggers writes; **FIFO eviction → recursive summary**; evicted text stays in recall storage **indefinitely** | Paginated function-call retrieval over recall + archival storage | **Deep Memory Retrieval**: MemGPT **66.9%** vs GPT-3.5-Turbo 38.7%; **92.5%** vs GPT-4 32.1%; **93.4%** vs GPT-4-Turbo 35.3% | not reported | Weak model → bad write/retrieve decisions; summary loss; unbounded recall storage |
| **Zep + Graphiti** (arXiv:2501.13956; `getzep/graphiti` 30,866★, Apache-2.0, last commit 2026-09-11) | Episodes → **bitemporal** KG. A contradicting edge **invalidates** the old edge but preserves it (valid-time + transaction-time) | Graph node/edge search + entity summaries, cross-encoder rerank, top-20 facts | **DMR** 94.8% (GPT-4-Turbo) vs MemGPT 93.4%; 98.2% w/ GPT-4o-mini. **LongMemEval** 63.8% vs 55.4% full-context GPT-4o-mini; 71.2% vs 60.2% full-context GPT-4o | LongMemEval **3.20 s / 1.6K tokens** vs **31.3 s / ~115K tokens** full-context (GPT-4o-mini); 2.58 s vs 28.9 s (GPT-4o) | **Loses on single-session-assistant questions by 17.7% (GPT-4o) and 9.06% (GPT-4o-mini)**; extraction + temporal-contradiction pipeline can fail before retrieval |
| **HippoRAG** (arXiv:2405.14831; `OSU-NLP-Group/HippoRAG` 4,003★, MIT, last commit 2026-09-03) | Offline: NER + OpenIE build an open KG; similarity edges between noun phrases. Not an autonomous conversational writer | Query entities → **Personalized PageRank** → passage scores | Single-step w/ Contriever: MuSiQue R@2/R@5 **41.0/52.1**, 2WikiMultiHop **71.5/89.5**, HotpotQA **59.0/76.2**; IRCoT+HippoRAG (ColBERTv2) avg **62.7/78.2** vs BM25/Contriever/GTR/ColBERTv2/RAPTOR/IRCoT | 1,000 queries: **$0.10 / 3 min** vs IRCoT **$1–3 / 20–40 min**; 6–13× faster online. Offline indexing ~10× slower, +$15 per 10K passages | ~**48%** of errors from ignoring contextual cues; OpenIE degrades on long passages; graph inherits extraction errors |
| **HippoRAG 2** (same repo) | Dual passage+phrase representation, denser/sparser hybrid memory | Hybrid embedding + sparse + PPR | ~**+7 mean F1** over NV-Embed-v2 on associative benchmarks (NQ, PopQA, MuSiQue, 2Wiki, HotpotQA, LV-Eval, NarrativeQA) | reported as more efficient than v1; no single comparable $/ms | More structure = more extraction sensitivity |
| **Mem0 / Mem0^g** (arXiv:2504.19413; `mem0ai/mem0` 65,270★, Apache-2.0, last commit 2026-09-11) | Per extracted fact a memory manager picks **ADD / UPDATE / DELETE / NOOP** — the clearest explicit forgetting policy in the field | Selective retrieval of salient NL memories; Mem0^g adds graph memory | LoCoMo: single-hop F1 **38.72**, BLEU-1 27.13, judge **67.13**; multi-hop F1 **28.64**, judge 51.15. Mem0^g highest judge **68.44%**. Abstract: **+26% relative LLM-as-judge over OpenAI memory**, graph variant ~**+2%** overall | "materially lower overhead than full-context"; full latency table not exposed | **Mem0^g can be WORSE than plain Mem0 on single-turn answers** — graph adds little there; a faulty extractor can DELETE a true fact |
| **A-MEM** (arXiv:2502.12110; `agiresearch/A-mem` 1,178★, MIT, last commit 2025-12-12) | Each interaction → atomic Zettelkasten note (content, timestamp, keywords, tags, context, embedding, links). New note retrieves neighbours, LLM **generates links** then **evolves** the retrieved notes' context/keywords/tags. **No pruning step.** | Dense embedding + cosine top-k, then link expansion | LoCoMo F1 **3.45** vs LoCoMo baseline 2.55 and MemGPT 1.18 (note: this paper's F1 scale is not comparable to Mem0's) | **~1,200 tokens/op, <$0.0003/op, 5.4 s** (GPT-4o-mini); 1.1 s local Llama-3.2-1B | Multi-step linking → latency **and error accumulation** (identified explicitly by MemoryOS); no forgetting → unbounded growth |
| **MemoryOS** (arXiv:2506.06326) | Short/mid/long-term tiers; STM overflow → FIFO page transfer; mid-term segments deleted or promoted by **heat score** | Semantic segmentation + top-m segments, top-k pages | LoCoMo w/ GPT-4o-mini: avg **+49.11% F1**, **+46.18% BLEU-1**; GVD **+3.2% acc over A-MEM**; Qwen2.5-7B 91.8/82.3/90.5 vs A-MEM 87.2/79.5/87.8 | **4.9 LLM calls** vs 13 for A-MEM*; **3,874 tokens** vs 16,977 for MemGPT | Heat ≠ truth — a frequently retrieved wrong memory survives; FIFO mishandles topic boundaries |
| **ExpeL** (arXiv:2308.10144) | Autonomously gathers training-task experiences and **extracts NL insights**; non-parametric | Retrieves insights + past experiences for the new task | Consistent improvement as experience accumulates across HotpotQA, ALFWorld, WebShop vs ReAct; **no single exact score table exposed** | not reported | Wrong lesson can be generalised and preserved; transfer fails under task shift |
| **Reflexion** (arXiv:2303.11366) | After each trial, self-reflection over trajectory + reward → verbal summary appended to a **bounded** episodic buffer (typically 1–3) | Next trial conditions on recent reflections (ALFWorld truncates to last 3) | **+22% ALFWorld, +20% HotpotQA, +11% HumanEval**; HumanEval pass@1 **91% vs 80% GPT-4**; ReAct+Reflexion solves **130/134** ALFWorld tasks | not reported | Depends on a reliable verifier. Flaky tests → confidently wrong reflections |
| **Voyager** (arXiv:2305.16291) | GPT-4 writes and **verifies** executable skills; verified skill stored in a vector DB keyed by an LLM-written description | Top-5 skill retrieval from env feedback + suggestion | **3.3× more unique items, 2.3× travel distance, up to 15.3× faster tech-tree** vs prior SOTA; top-5 retrieval accuracy **96.5% ± 0.3** over 309 samples | GPT-4 ≈ 15× the cost of GPT-3.5 | Skills brittle outside the verified environment |
| **LangMem / LangGraph, CrewAI memory, OpenAI Agents SDK sessions** | Hot-path tools + background consolidation (LangMem); save-time LLM importance analysis (CrewAI); session replay + trimming/compression (OpenAI) | pluggable vector / composite recency-importance / session replay | **No independent published benchmark in the official docs** for any of the three | deployment-dependent | Cross-tenant leakage, stale procedural instructions, unbounded stores. Session persistence ≠ durable semantic memory |

### B.2 Benchmarks these numbers come from

| Benchmark | Size / shape | What it measures | Verified |
|---|---|---|---|
| **LongMemEval** | 500 manually created questions; 5 abilities — extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention | Long-term interactive memory in chat assistants | arXiv:2410.10813; `xiaowu0162/LongMemEval` 1,085★ MIT, last commit 2026-05-11; site HTTP 200 |
| **LoCoMo** | up to ~35 sessions, ~9K tokens/dialogue (MemoryOS reports ~300 turns) | Very long-term conversational memory | referenced by arXiv:2504.19413, 2502.12110, 2506.06326 |
| **BEAM** | 1M and 10M token scales | Memory behaviour when context ≫ typical benchmarks | `mem0.ai/research` and `mem0.ai/blog/state-of-ai-agent-memory-2026` (HTTP 200, published 2026-09-03): **92.5 LoCoMo, 94.4 LongMemEval at ~6,900 tokens/query**; biggest gains **+29.6 pts temporal reasoning, +23.1 multi-hop**. Vendor-reported. |
| **MemoryArena** (arXiv:2602.16313) | unified gym for multi-session agentic memory | interdependent multi-session tasks | verified 2026-02 |
| **Evo-Memory** (arXiv:2511.20857) | sequential task streams requiring memory evolution after each interaction | test-time learning w/ self-evolving memory | verified — memory helps but is "fragile in stability and procedural reuse" |
| **WhenLoss** (arXiv:2605.24579) | diagnostic | separates **write** bottlenecks from **retrieval** bottlenecks | verified 2026-05 |
| Surveys | arXiv:2508.07407 (self-evolving AI agents), arXiv:2507.21046 (what/when/how/where to evolve), arXiv:2601.22628 (TTCS, ICLR 2026 Lifelong Agent workshop) | taxonomy | all verified |

### B.3 Failure modes — evidence, not speculation

| Failure mode | Hard evidence | Mitigation we will adopt |
|---|---|---|
| **Memory poisoning (direct)** | **AgentPoison** (arXiv:2407.12784): average retrieval-attack success **62.0%** when only **one** database instance is poisoned; **79.0%** under a larger trigger. Requires **no model retraining**. | EKG writes are privileged transactions: every node/edge carries `source_id`, `writer`, `confidence`. Only the trade-outcome writer and the market-data writer may create high-trust edges. |
| **Memory injection (query-only)** | **MINJA** (arXiv:2503.03704): injects malicious records **without any direct memory-bank access**, **98.2% average injection success, ~76% attack success**. A 2026 follow-up (arXiv:2601.05504) reports >95% injection / 70% attack success "under idealized conditions" and notes realistic-deployment robustness is understudied. | No untrusted free text (Reddit/StockTwits) may write directly to the EKG. Sentiment agent output enters as a *low-confidence, quarantined* node type. |
| **Drift / error accumulation** | MemoryOS (arXiv:2506.06326) explicitly attributes error accumulation to A-MEM's multi-step LLM linking. Reflexion and Generative Agents both feed self-generated text back into memory — a positive feedback loop. arXiv:2601.11653 (2026): agent behaviour degrades from "loss of constraint focus, error accumulation, and memory-induced drift". | Provenance chains + periodic **recomputation from raw outcome logs**, not from derived summaries. Derived nodes are never inputs to other derived nodes more than one hop deep. |
| **Unbounded growth** | MemGPT keeps evicted items in recall storage indefinitely; A-MEM has no pruning; Generative Agents has no deletion mechanism. | Explicit decay + pruning rule (see E.3). Append-only audit log on disk, but a **bounded retrieval index**. |
| **Retrieval ≠ correct use** | LongMemEval: Chain-of-Note / structured output adds **up to 10 absolute points** with the *same* retrieved content. HippoRAG: **~48%** of errors are "ignored contextual cues". | Measure retrieval quality and decision quality separately. |
| **Long context can match or beat memory systems** | arXiv:2501.01880 (*Long Context vs RAG*) reports discrepant results across setups. Zep itself loses 17.7% / 9.06% on single-session questions. | Include a **"dump the last N trades into the prompt/state"** control arm. |

### B.4 The honest negative that matters most for this project

> **No source found in this session demonstrates that a self-evolving LLM-agent memory improves out-of-sample trading returns, Sharpe, drawdown, turnover, or robustness to regime change.**

Every headline number in B.1 is a QA, coding, or game benchmark. Reflexion and Voyager are genuine *sequential* evidence (ALFWorld, Minecraft) but not *financial* evidence. Evo-Memory (arXiv:2511.20857) is the closest thing to a sequential test-time-learning benchmark and concludes that memory "substantially improves performance but remains fragile in stability and procedural reuse".

**Consequence:** the EKG must be justified inside this project by our own ablation (Part E.6), not by citation.

---

## Part C — Continual learning under non-stationarity

### C.1 Catastrophic forgetting in RL — what actually works

| Family | Best RL evidence | Measured result | Trade-off |
|---|---|---|---|
| **EWC / online EWC** | arXiv:1612.00796 (Atari, one network, fixed resources) | Learns multiple Atari games where plain SGD learns one; total human-normalised score stays **below 1** | Importance estimates protect stale weights → **plasticity loss** |
| **Replay / distillation (CLEAR)** | **CORA** (arXiv:2110.10067) | Forgetting summary **1.1 / 0.7 / 0.9** (Atari / Procgen / MiniHack) vs Online IMPALA **5.6 / 2.2 / 2.8** | Memory + compute; CLEAR later **struggles to maintain plasticity on MiniHack** |
| **Replay-enhanced CRL (RECALL)** | arXiv:2311.11557 | Beats other Continual World baselines on average performance, forgetting and forward transfer (FT comparable to ClonEx) | needs stored transitions |
| **Parameter isolation (PackNet, ClonEx-SAC)** | **COOM** (NeurIPS 2023 D&B, HTTP 200) | PackNet best overall, ClonEx-SAC second, L2 third. **AGEM and VCL perform WORSE than naive fine-tuning.** "Perfect Memory" needs ~2× runtime and **16× memory** | capacity fragmentation; needs task identity |
| **Generative replay** | discussed by RECALL; **no separate RL numeric result found** | — | under-validated in RL. Reported as a gap. |
| **EWC for KG continual learning** | arXiv:2512.01890 (empirical evaluation) | verified as existing; treat as the KG-specific reference | — |
| **Safe continual RL** | arXiv:2604.19737 — Safe EWC and Cost-Fisher EWC for non-stationary environments | authors present them as initial solutions "and their limitations" | 2026, early |

### C.2 Loss of plasticity and primacy bias — the finding that changes our training loop

| Intervention | Benchmark | Exact result |
|---|---|---|
| **Baseline degradation** | Ant-v3, **stationary** | Standard PPO improves for ~**3M steps**, then collapses; by **20M steps it fails every episode**. Units become dormant; stable rank falls. (Nature 632, 2024, HTTP 200) |
| **Continual backpropagation** | Ant-v3 stationary + changing-friction | Selective reinitialisation of **<1 unit per update** keeps PPO improving; much better than standard PPO under changing friction |
| **ReDo** | Ant-v3; thresholds 0.01/0.03/0.1, periods 10…10⁵ | Best: threshold **0.1**, period **10²**. ReDo+L2 beat standard PPO **but still degraded and was worse than L2 alone** in that test |
| **Dormant neuron phenomenon** | Atari, DQN | ReDo recycles dormant neurons; compared against Reset and weight decay (PMLR v202 sokar23a, HTTP 200; arXiv:2302.12902) |
| **Resets / primacy bias** | Atari 100k, DMC | SPR+resets IQM **0.478** vs SPR **0.380**; SAC+resets **656** vs 501; DrQ+resets **762** vs 569. Schedules: SPR final linear layer every **2×10⁴** steps; SAC all networks every **2×10⁵**; DrQ last 3 of 7 layers every **2×10⁵** (arXiv:2205.07802; PMLR v162 nikishin22a; `evgenii-nikishin/rl_with_resets` 107★ MIT) |
| **L2 / Shrink-and-Perturb** | Ant-v3 changing friction | L2 limits weight growth but small weights prevent commitment; **Shrink-and-Perturb gave no marked improvement over L2** in the Ant experiment |
| **Regenerative regularisation (L2 Init)** | 5 continual classification problems incl. **500** Permuted-MNIST tasks | Top-3 on all five; beats all on 5+1 CIFAR; ≥ continual backprop on 4 of 5. **Supervised evidence — do not promote to RL/finance** (arXiv:2308.11958) |
| **Third facet: co-observation** | arXiv:2608.18803 (2026) | Even a continual learner that never forgets **still underperforms joint training**, because features requiring data partitions to be observed together are undiscoverable from separate training |

### C.3 Continual RL benchmarks

| Benchmark | Scale | Key result |
|---|---|---|
| **Continual World / CW20** (arXiv:2105.10919) | Meta-World, 20 tasks, 1M steps each, 20M total | **Only fine-tuning (0.20) and PackNet (0.19) achieved positive forward transfer, both below the single-task reference 0.46. Backward transfer negligible.** |
| **CORA** (arXiv:2110.10067) | Atari 6, Procgen 6, MiniHack 15, CHORES | CLEAR forgetting numbers above; exposes late plasticity failure |
| **COOM** (NeurIPS 2023 D&B; `TTomilin/COOM`) | Doom, 7 sequences, 100K steps/task | PackNet + ClonEx-SAC lead; AGEM, VCL < fine-tuning |
| **Avalanche-RL** | framework | infrastructure, not a score table |
| 2025–2026 | no mature, widely replicated CRL leaderboard found with CORA/COOM-comparable scores | **evidence gap, not absence of work** |

### C.4 Meta-RL / fast adaptation — adaptation is measured in EPISODES

| Approach | Mechanism | Measured adaptation speed |
|---|---|---|
| **RL²** | recurrent state over interaction | adapted **within one episode** in the cited MuJoCo comparison |
| **VariBAD** (arXiv:1910.08348) | Bayesian task inference + strategic exploration | adapted within **one episode** in MuJoCo; matched Bayes-optimal behaviour from **episode 2** in a 6-episode toy task (posterior sampling needed **6** rollouts). Beat RL² except tied on HalfCheetahVel |
| **PEARL** | latent context + posterior sampling | trained 3–5 episodes; **begins performing well only from episode 3** |
| **Algorithm Distillation** (arXiv:2210.14215) | causal transformer imitates an RL algorithm's improvement process | in-context RL appears at ~**1 episode** of context; **2–4 episodes** for near-optimal; 300 episodes ≈ 15K env steps. Generalised better OOD than RL². **No market/non-stationarity score.** |
| **In-context model-based planning** (arXiv:2502.19009) | distilled RL algorithms for in-context planning | 2025, verified |

**Why this matters for markets:** context-based adaptation lets a policy respond to a volatility/liquidity regime **without rewriting weights**. That is structurally the same idea as the EKG — and it is cheaper and better-evidenced. It is the fallback if the EKG coupling fails (see E.5).

### C.5 Non-stationarity in markets — the finance evidence, with its context

**HMM regimes + RL (Dow 30, 2010–2025, 80/20 split, transaction costs NOT reported):**

| Algorithm | Annual return **with** HMM | **without** | Sharpe **with** | **without** | MaxDD **with** | **without** |
|---|---:|---:|---:|---:|---:|---:|
| A2C | 0.14 | 0.11 | 0.75 | 0.71 | −0.13 | −0.07 |
| DDPG | 0.10 | 0.08 | 0.72 | 0.48 | −0.07 | −0.10 |
| **PPO** | **−0.02** | **0.19** | **−0.26** | **1.03** | −0.18 | −0.10 |
| TD3 | 0.11 | 0.07 | 0.62 | 0.44 | −0.10 | −0.08 |
| SAC | 0.12 | 0.16 | 0.69 | 0.95 | −0.09 | −0.09 |

Source: HMM-RL study, `cloud-conf.net/datasec/2025/...966100a067.pdf` (fetched by deep research). **Reading: regime features helped A2C, DDPG and TD3 and HURT PPO and SAC.** PPO — our intended core algorithm — went from Sharpe **1.03 → −0.26**. No costs, no seeds, no significance test reported.

| Other finance evidence | Universe / window / cost | Result |
|---|---|---|
| **Adaptive & Regime-Aware RL for Portfolio Optimization** (arXiv:2509.14385) | asset universe and calendar split **not specified**; turnover penalty λ=0.002, reward clipped to [−0.03, 0.03], capital reset every 30 steps, shock every 25 steps | PPO Sharpe **1.0677**, MaxDD **−72.58%** vs equal-weight Sharpe **0.4152**, MaxDD **−28.91%**. Authors themselves warn these are stylised, not exact financial analogues |
| **FinRL Contests** (arXiv:2504.02281) | **Dow 30, daily OHLCV 2021-01-01 → 2023-12-01, 734 trading days**; rolling 30 train / 5 val / 5 test days (alt: 6/2/1 with 8-day retrain); cost held constant across models but **numeric value not given** | PPO cumulative return **63.37%**, Sharpe **1.55** vs DJIA / S&P 500 / mean-variance. **No repeated-run count reported**; authors describe policies as sensitive to hyperparameters, noise and random seeds; ensemble and individual agents converge to near-identical actions |
| **When to Retrain** (arXiv:2608.19488) | streaming ML under budget/latency constraints | compares periodic, error-triggered, ADWIN-triggered and no-retrain policies. **The retrieved evidence exposes no dataset, cadence or numeric outcome, and does not establish the data are financial.** |
| **Review of RL in Financial Applications** (arXiv:2411.12746) | survey | context |

### C.6 HONEST NEGATIVES (Part C)

1. **Continual-learning machinery often fails to beat plain fine-tuning.** Continual World: only fine-tuning and PackNet achieved positive forward transfer; AGEM and VCL were below fine-tuning in COOM.
2. **Regime detection is not a guaranteed improvement.** The HMM-RL table above contains its own counterexample (PPO 1.03 → −0.26 Sharpe).
3. **Deep RL in finance is unstable.** FinRL Contests report seed/noise/hyperparameter sensitivity with no seed count; arXiv:1709.06560 (*Deep RL that Matters*) established the general variance problem.
4. **No evidence supports any specific retrain cadence.** Any "retrain weekly" claim in this project must be our own measured result.
5. **The statistical null is strong.** The Deflated Sharpe Ratio exists because analysts can search millions of strategies; selection bias, backtest overfitting and non-normality inflate ordinary Sharpe (Bailey & López de Prado, `davidhbailey.com/dhbpapers/deflated-sharpe.pdf`; SSRN 2460551 **BLOCKED HTTP 403**).
6. **Never-forgetting is not enough** — arXiv:2608.18803 shows a perfect continual learner still underperforms joint training on co-observation-dependent features. Periodic **full retrain from scratch** must stay in the baseline set.

---

## Part D — Prime Intellect's actual approach (primary sources only)

All URLs in this section returned **HTTP 200** on 2026-09-14; all repo stats are live `gh_repo` reads on 2026-09-14.

### D.1 Model releases

| Model | Release | Size / base | Method | Compute | Headline benchmarks vs baselines | Licence | Paper |
|---|---|---|---|---|---|---|---|
| **INTELLECT-1** | `primeintellect.ai/blog/intellect-1-release`, **2024-11-29** | 10B, Llama-3 architecture | Globally distributed **pretraining**, 1T tokens; post-training = SFT + DPO + model merging (with Arcee AI) | Up to **112 H100s**, 5 countries, 3 continents, **42 days**, up to 14 concurrent nodes, **30 independent compute providers**; ~400× less communication than standard data parallelism | Base: MMLU 37.5, HellaSwag 72.26, ARC-C 52.13, GPQA 26.12, GSM8K 8.1, TruthfulQA 35.47, Winogrande 65.82, BBH 32.97 vs MPT-7B, Falcon-7B, Pythia-12B, LLM360-Amber, LLaMA-7B/13B, LLaMA2-7B/13B | report CC BY 4.0 | arXiv:**2412.01152** |
| **INTELLECT-2** | `.../blog/intellect-2-release`, **2025-05-11** | 32B, **QwQ-32B** base | Fully **asynchronous, permissionless, globally distributed RL** post-training (GRPO) | ~1:4 training:inference compute ratio; **total GPU/node/contributor counts not stated in the paper** | AIME24 **78.8**, AIME25 **64.9**, LCB-v5 **67.8**, GPQA-D **66.8**, IFEval **81.5** vs QwQ-32B 76.6/64.8/66.1/66.3/83.4; R1-Distill-32B 69.9/58.4/55.1/65.2/72.0; DeepSeek-R1 78.6/65.1/64.1/71.6/82.7 | Apache-2.0 (HF card) | arXiv:**2505.07291** |
| **INTELLECT-3** | `.../blog/intellect-3`, **2025-11-26** | **106B MoE, 12B active**, GLM-4.5-Air base | **SFT then large-scale RL** via `prime-rl` + `verifiers` + public environments | **512 NVIDIA H200 GPUs across 64 nodes**, ~2 months — **centralised, not permissionless** | AIME24 **90.8**, AIME25 **88.0**, LCB-v6 **69.3**, GPQA **74.4**, HLE **14.6**, MMLU-Pro **81.9** vs GLM-4.5-Air / GLM-4.5 / GLM-4.6 / DeepSeek-R1-0528 / DeepSeek-v3.2 / GPT-OSS-120B | MIT (HF card) | arXiv:**2512.16144** |
| Newer base model through 2026-09-14 | **Not found.** The 2026 releases are tooling and agents (`prime-agent`, `verifiers` v1, prime-rl algorithms layer), not a new INTELLECT base model. | — | — | — | — | — | — |

### D.2 The stack, verified live on 2026-09-14

| Component | Repo / URL | Live status | What it actually does |
|---|---|---|---|
| **prime-rl** | `PrimeIntellect-ai/prime-rl` | **2,038★, Apache-2.0, last commit 2026-09-13**, created 2025-02-18 | Asynchronous RL training framework (SFT + RL, multi-node Slurm/K8s). Docs claim capability for **1T+ param MoE on 1,000+ GPU clusters** — a stated capability, not a reported completed run |
| **verifiers** | `PrimeIntellect-ai/verifiers` | **4,614★, MIT, last commit 2026-09-14** | Library defining RL environments + evals; integrated with the Hub and prime-rl. Hub is migrating to **verifiers v1**; v0 environments are tagged Legacy |
| **Environments Hub** | `primeintellect.ai/blog/environments`, **2025-08-27** | HTTP 200 | "Environments define the world, rules and feedback loop of **state, action and reward**." Private beta had **30+ researchers/companies**. Homepage claims **2,500+ environments**; an independently enumerated dated Hub count is **not found** |
| **prime-environments** | `PrimeIntellect-ai/prime-environments` | **257★, Apache-2.0, last commit 2026-05-07** | Exists (the deep-research run wrongly reported "not found" — corrected here by direct `gh_repo` check) |
| **protocol** | `PrimeIntellect-ai/protocol` | **139★, Apache-2.0, ARCHIVED, last commit 2025-11-10** | P2P compute/intelligence network. **Archived — the standalone decentralised protocol repo is no longer maintained.** No separate published protocol spec or token/economic paper was found |
| **OpenDiLoCo** | `PrimeIntellect-ai/OpenDiloco`, arXiv:**2407.07852** | **592★, Apache-2.0, last commit 2025-01-13** (dormant) | Open DiLoCo implementation. **500×** less communication in an 8-replica experiment; **125×** at 1.1B params with 125 local steps; **90–95%** compute utilisation across 2 continents / 3 countries; 4 workers × 8 H100; ~67.5 min between syncs; all-reduce = **6.9%** of training time |
| **TOPLOC** | arXiv:**2501.16007**, `.../blog/toploc` (2025-01-28) | HTTP 200 | LSH over salient intermediate activations to detect unauthorised **model / prompt / precision** changes. **258 bytes per 32 new tokens** vs 262 KB to store embeddings (Llama-3.1-8B-Instruct). Reported **100% TPR and 100% TNR** in its experiments; SYNTHETIC-2 reports TOPLOC-v2 **false-positive rate 0.000925%** over a 4M-sample run |
| **SYNTHETIC-1** | `.../blog/synthetic-1-release`, **2025-02-20** | HTTP 200 | **2M** verified reasoning traces from DeepSeek-R1: math 777k, algorithmic coding 144k, real SWE 70k, STEM QA 313k, code understanding 61k. Verification = symbolic checks, containerised unit tests, LLM judge, exact match |
| **SYNTHETIC-2** | `.../blog/synthetic-2-release`, **2025-07-10** | HTTP 200 | **4M** verified traces via planetary-scale **pipeline-parallel** distributed inference on DeepSeek-R1-0528, **1,253 GPUs**; TOPLOC-v2 proofs |
| **genesys** | `PrimeIntellect-ai/genesys` | **139★, Apache-2.0, last commit 2025-02-28** (dormant) | Synthetic-data generation + verification framework: sample from a teacher model, assign rewards via pluggable `verify` functions, verification runs asynchronously |
| **Prime Agent** | `.../blog/prime-agent` (**2026-08-05**), `PrimeIntellect-ai/prime-agent` (created 2026-05-08) | HTTP 200 | "A self-improving RLM agent." `/refine` reviews a trajectory and makes **small evidence-backed updates to supplemental prompts, memories, skills and subagent specs**. Snapshots support rollback. **The immutable base system prompt is not rewritten. No gradient updates.** |
| 2026 blog trail | `true-agents-model-the-world` (2026-06-05), `algorithms-layer` (2026-07-05), `series-a` (2026-07-08, **$130M Series A**) | all HTTP 200 | Direction: RL environments as the scaling substrate; six algorithms shipped in prime-rl; "companies can own their model optimization loop… build agents that improve continuously in production" |

### D.3 INTELLECT-2's async RL loop — the mechanism worth copying

1. **Inference workers** generate rollouts asynchronously on heterogeneous, mostly consumer-grade GPUs.
2. **TOPLOC validators** confirm the provider used the authorised model/precision.
3. **SHARDCAST** distributes model weights over an HTTP tree topology.
4. **GRPO training workers** filter (offline + online) and train on the incoming rollout stream.
5. Inference workers run on weights **≥2 RL steps old**; asynchrony up to **4 steps matched the synchronous baseline** with no observed degradation.

**PCCL is named in the task brief but is NOT documented in the INTELLECT-2 paper evidence retrieved here — marked not found.**

### D.4 What Prime Intellect does NOT claim (the honest reading)

| Claim that sounds plausible | Actual primary-source status |
|---|---|
| "They run continually self-improving models" | **Not documented.** All three INTELLECT models are bounded training runs. No online weight-update loop at deployment is published. |
| "They have a continual-learning manifesto" | **Not found.** The strongest dated statements are the Environments Hub rationale (open rails vs walled gardens, 2025-08-27) and the homepage line "Train, deploy, and **continuously improve** your own models". |
| "The decentralised protocol is the product" | `PrimeIntellect-ai/protocol` is **archived**; INTELLECT-3 used a centralised 64-node H200 cluster. The decentralised story peaked with INTELLECT-2. |
| "Self-improvement means the model rewrites itself" | The only documented self-improvement loop (Prime Agent `/refine`) rewrites **harness state** — prompts, memories, skills, subagent specs — with rollback snapshots. |

### D.5 The transferable lesson for RL-Trader's EKG

Prime Intellect's actual, documented pattern is:

> **verifiable environment + explicit verifier → reward-labelled experience → periodic batch policy update; and separately, a durable, versioned, rollback-able memory/skill layer that is refined by evidence, never silently.**

That maps onto the EKG exactly:

| Prime Intellect primitive | RL-Trader EKG analogue |
|---|---|
| `verifiers` environment (state, action, reward) | TensorTrade Gymnasium env with explicit net-of-cost PnL reward |
| Verifier / TOPLOC proof | **Outcome verification**: an EKG edge is only promoted to high confidence after the trade closes and realised PnL is known |
| Async off-policy rollouts, ≤4 steps stale | EKG reads are **allowed to be stale**; the policy must tolerate a graph that is hours old |
| `/refine` edits prompts/memories/skills with snapshots + rollback | EKG consolidation job edits nodes/edges with **versioned snapshots + rollback**, never in place |
| Immutable base system prompt | **Immutable core observation and reward**. The EKG may only add gated auxiliary features |
| Offline + online rollout filtering | Admission control on EKG writes (confidence, provenance, dedup) |

---

## Part E — SYNTHESIS: the concrete EKG design for RL-Trader

### E.1 What the EKG is, and what it is NOT

**It IS:** a bitemporal, provenance-carrying store of *(market context → action → outcome)* experience plus *(entity → relation → entity)* structure, that (a) supplies a gated auxiliary feature vector to the RL policy, (b) supplies retrievable analogue episodes, and (c) supplies human-readable explanations and the live UI graph.

**It is NOT:** a reward source, a trade veto, an oracle, or a claimed alpha generator. Sections A.5, B.4 and C.6 are the reasons.

### E.2 Node and edge schema

**Every edge is a row in one bitemporal table.** Four timestamps, not one — this is the anti-look-ahead mechanism.

```
Edge(
  edge_id, src_id, rel_type, dst_id,
  valid_from, valid_to,          -- when the FACT was true in the world
  first_public_at,               -- when we could LEGALLY have known it  <-- leakage guard
  recorded_at, superseded_at,    -- transaction time (bitemporal, Graphiti-style)
  confidence  float,             -- [0,1]
  support_n   int,               -- how many observations back it
  writer      enum,              -- market|execution|feature|sentiment|consolidator
  source_id, parent_edge_ids[],  -- provenance chain
  decay_lambda float
)
```

**Retrieval rule (hard invariant):** at decision time `t`, only edges with `first_public_at <= t AND valid_from <= t AND (valid_to IS NULL OR valid_to > t) AND (superseded_at IS NULL OR superseded_at > t)` are visible. This is enforced in one SQL predicate in DuckDB and unit-tested with a deliberate future-edge injection test.

| Node type | Key attributes | Written by |
|---|---|---|
| `Instrument` | ticker, sector, exchange, ADV, listing/delisting dates | market (static, point-in-time) |
| `Bar` (not stored in graph; Parquet) | OHLCV | market |
| `FeatureState` | discretised feature fingerprint (vol bucket, trend bucket, RSI bucket, spread bucket, position state) — the **join key** between experience and context | feature |
| `Regime` | id, method (HMM/BOCPD/vol-tercile), start, end, posterior | feature |
| `Episode` | trade/holding episode: entry_t, exit_t, instrument, side, size, **realised net PnL**, MAE/MFE, turnover, costs paid | execution |
| `Decision` | one policy step: obs hash, action, logprob, value estimate, EKG features used, edge_ids read | execution |
| `Event` | earnings, guidance, macro print, news cluster | sentiment/market |
| `Entity` | company, person, product, supplier, competitor | sentiment (**low trust**) |
| `Claim` | sentiment/ontology assertion from Reddit/StockTwits/Kaggle | sentiment (**quarantined**) |
| `Lesson` | consolidated pattern: "in regime R, action A on feature-state F had mean net PnL μ, n observations, IQR …" | consolidator |

| Edge type | Meaning | Trust |
|---|---|---|
| `OBSERVED_IN(Episode → Regime)` | episode occurred in regime | high (mechanical) |
| `TOOK_ACTION(Decision → Instrument)` | action record | high |
| `RESULTED_IN(Decision → Episode)` | causal link decision→outcome | high |
| `SIMILAR_TO(FeatureState ↔ FeatureState)` | embedding/κ-NN link | medium, recomputed |
| `SUPPORTS / CONTRADICTS(Episode → Lesson)` | evidence link with sign | derived, depth ≤1 |
| `CO_MOVES_WITH(Instrument ↔ Instrument)` | rolling correlation above threshold, computed **only from data before `first_public_at`** | medium |
| `SUPPLIES / COMPETES_WITH(Entity ↔ Entity)` | economic link | medium; must carry `first_public_at` |
| `MENTIONS(Claim → Instrument)` | sentiment mention | **low, quarantined** |

**Only 5 instruments.** The instrument graph is tiny. Therefore **the EKG's value must come from the experience side (Episode/FeatureState/Lesson), not from the entity side.** This is an important scoping decision: a 5-node company graph cannot carry much relational signal, and A.5 shows entity graphs often lose to a no-graph LSTM anyway.

### E.3 Update rule — what writes, when, with what decay/consolidation/pruning

| Phase | Trigger | Writer | Action |
|---|---|---|---|
| **1. Ingest (hot path, cheap)** | every env step | execution | Append `Decision`; **no LLM call**. Pure append to Parquet. Hot path must add **< 1 ms**. |
| **2. Seal** | position closes | execution | Create `Episode` with realised **net** PnL after costs; link `RESULTED_IN`. **This is our TOPLOC analogue: an experience only becomes evidence after it is verified by realised outcome.** |
| **3. Consolidate (batch)** | end of each trading day / each walk-forward fold | consolidator | Group sealed Episodes by `(Regime, FeatureState, action)`. Update/insert `Lesson` with running mean, variance, `support_n`. **Bayesian shrinkage to the prior**: `μ̂ = (n·μ_obs + k·μ_prior)/(n+k)`, `k = 20`. Never write a Lesson with `support_n < 10`. |
| **4. Decay** | daily | consolidator | `confidence ← confidence · exp(−λ·Δt_days)`, half-life by edge class: `Lesson` **90 trading days**, `CO_MOVES_WITH` **20 days**, `Claim` **3 days**, `SUPPLIES` **250 days**, `Episode` never decays (it is raw evidence). |
| **5. Contradiction** | new Lesson contradicts an existing one | consolidator | **Invalidate, never delete** — set `superseded_at`, keep the old row (Graphiti's bitemporal pattern). |
| **6. Prune (retrieval index only)** | weekly | consolidator | Drop from the *retrieval index* any edge with `confidence < 0.15` or `support_n < 10` or unread for 60 days. **The Parquet audit log keeps everything.** Cap the index at **50k edges**; evict by `confidence × recency × read_count`. |
| **7. Rebuild** | each walk-forward fold boundary | consolidator | Recompute all derived nodes **from raw Episodes**, not from previous derived nodes. Prevents the A-MEM/MemoryOS error-accumulation loop (B.3). |
| **8. Quarantine** | sentiment writes | sentiment agent | `Claim` nodes enter with `confidence ≤ 0.3`, cannot create `Lesson`s, and cannot exceed 20% of the retrieved feature mass. Defence against AgentPoison/MINJA-class injection (B.3). |

**Growth bound:** 5 instruments × ~250 trading days/yr × ~10 folds ≈ O(10⁴) Episodes; Lessons are capped by `|Regime| × |FeatureState| × |action|` ≈ 3 × 256 × 3 ≈ 2,300. The graph is deliberately **small enough to fit in RAM and to audit by hand**.

### E.4 Storage choice

| Option | 2026 status (verified) | Verdict |
|---|---|---|
| **DuckDB + Parquet (+ DuckPGQ)** | DuckDB MIT; **DuckPGQ is a DuckDB community extension** supporting SQL/PGQ and graph algorithms (`duckdb.org/community_extensions/extensions/duckpgq`, HTTP 200) | ✅ **CHOSEN.** Embedded, zero infra, columnar, perfect for bitemporal predicate filtering, trivially reproducible on cloud GPU boxes, and the audit log *is* the database. Matches the CLI-first constraint. |
| **NetworkX in-memory view** | BSD-3 | ✅ **CHOSEN as the companion.** The graph is ≲50k edges. Materialise a point-in-time subgraph per fold, compute embeddings/centrality there. Also the natural feed for the D3/Three.js UI. |
| **Kuzu** | `kuzudb/kuzu` **ARCHIVED 2025-10-10**, read-only, 4,025★, MIT. Graphiti deprecated its Kuzu backend. Community forks exist (e.g. `Vela-Engineering/kuzu`, created 2026-02, 43★) | ❌ **REJECTED.** Do not take a new dependency on an archived engine. |
| **Neo4j** | `neo4j.com/licensing/` HTTP 200; AGPLv3 / commercial split; server or Aura | ⚠️ Fallback only. Best if we ever need concurrent multi-writer online serving + Cypher + GDS. Adds a server to operate. |
| **Memgraph** | `github.com/memgraph/memgraph` HTTP 200; BSL community / MEL enterprise; in-memory C++; Benchgraph published (`memgraph.com/benchgraph`, HTTP 200) | ⚠️ Fallback. Lowest latency server option; **check BSL obligations before any commercial use**. |
| **Graphiti** | `getzep/graphiti` **30,866★, Apache-2.0, last commit 2026-09-11** | 📐 **Copy the schema, not the dependency.** Graphiti is a temporal-provenance *framework* over a backend (Neo4j/FalkorDB/Neptune), and it runs LLM extraction per episode. We do not need LLM extraction for mechanical trade outcomes, and per-episode LLM calls would dominate cost. **Adopt its bitemporal invalidate-don't-delete pattern.** |

**Do not quote any vendor throughput number as a guarantee.** Our write volume (≲10² edges/day) makes throughput irrelevant; correctness, auditability and reproducibility dominate.

### E.5 THE COUPLING QUESTION — how the graph changes the RL policy

| # | Mechanism | Published evidence | Robustness to a **wrong** EKG fact | Verdict |
|---|---|---|---|---|
| (i) | **State augmentation** `π(a | o_t, g_t·f(G_t))` | SR-DRL (arXiv:2009.12462): trained on 5 objects → solves **78%** of 20-object BlockWorld; trained 10×10/4 boxes → **89%** of 15×15/5-box Sokoban; **96%** of test levels at 10⁹ steps vs I2A 90%, DRC 99%. GraphSAGE+PPO beats PPO-without-extractor. GPM/R-GCN beats iCNN/EIIE/EI3/SARL/AlphaStock (margin not exposed) | **Highest.** The policy can learn to ignore a noisy channel; a gate + graph-dropout makes ignoring the default | ✅ **PRIMARY** |
| (ii) | **Reward shaping** `r' = r + F(·)` | Potential-based `F = γΦ(s′) − Φ(s)` preserves optimal policy (Ng/Harada/Russell 1999). Arbitrary KG/LLM bonuses have **no such guarantee**. Eureka (arXiv:2310.12931) shows LLM reward *discovery* can beat expert rewards, but reward hacking is documented (arXiv:2502.18770); Bootstrapped Reward Shaping (arXiv:2501.00989) | **Lowest.** A wrong fact becomes the optimisation target | ❌ **REJECTED** for EKG facts. Permitted only as a clipped, annealed **potential** over a validated abstraction, reported with and without |
| (iii) | **Action masking / veto** | Invalid-action masking is a valid policy gradient (arXiv:2006.14171); in μRTS first reward appeared in **~0.06%** of training steps with masking vs **3.43%** with an invalid-action penalty of −0.01; penalty −1 was consistently worst. Shielding (arXiv:1708.08611) guarantees safety but needs a faithful abstraction and becomes intractable for complex dynamics | **Dangerous.** A wrong "cannot trade" edge permanently deletes a profitable legal action | ⚠️ **Deterministic constraints only** (cash, position limit, market hours, borrow). **KG beliefs may only scale position size, never veto.** |
| (iv) | **Curriculum / environment reshaping** | PAIRED (arXiv:2012.02096) keeps transfer as difficulty rises; pure minimax produces unsolvable tasks and fails. PLR (PMLR v139 jiang21b, HTTP 200) improves Procgen sample efficiency; combined with the prior leading method reports **>76%** test-return improvement over standard baselines. Replay-guided AED: arXiv:2110.02439 | **Medium.** Biases the training distribution, not the live action | 🔁 **Stage-2 add-on.** Regime curriculum (calm → volatile → drawdown → correlation-break). **Never score levels using test-period returns** — that is leakage |
| (v) | **Retrieval-conditioned policy** | Retrieval-Augmented RL (PMLR v162 goyal22a, HTTP 200): retrieval-augmented DQN avoids task interference and learns faster; retrieval-augmented R2D2 learns significantly faster and scores higher on Atari. RA-DT (arXiv:2410.07071; PMLR v330 schmied26a) outperforms baselines on grid worlds **with a fraction of the context length**. NEC (PMLR v70 pritzel17a, HTTP 200) is the episodic-control ancestor | **High.** Retrieval can abstain; a null-memory token lets the policy ignore it | ✅ **FALLBACK / second arm** |

**DECISION.**

- **Primary: (i) gated state augmentation.**
  `z_t = GNN_or_pool(G_t^{≤t})`; `c_t = mean confidence of read edges`; `g_t = σ(W[o_t, z_t, c_t])`; observation becomes `[o_t ‖ g_t ⊙ z_t]`.
  Train with **graph dropout p=0.2**, **edge-time jitter**, and an explicit **null-graph** pass so the policy never becomes graph-dependent. Log `edge_ids` read per decision (this also drives the UI and the audit trail).
- **Fallback: (v) retrieval-conditioned policy.** Retrieve top-k analogue `Episode`s by hybrid key `(regime, feature-state bucket, instrument, recency)`; pool their net-PnL distribution into a small fixed vector `[μ, σ, n, win-rate, IQR]`; append a **null-memory token**. Train with memory dropout and adversarially corrupted retrievals.
- Both arms read the **same** EKG through the **same** point-in-time predicate, so the ablation isolates the coupling, not the data.

**Contradiction with the master brief, stated plainly:** the brief proposes that the EKG "reshapes the RL policy's **inputs/reward** over time". The **inputs** half is supported. The **reward** half is not: no source found in this session supports shaping an RL policy with an unvalidated knowledge source, and the only shaping form with a correctness guarantee is potential-based. Recommend amending the architecture doc to "reshapes the policy's **inputs, retrieval context, and position sizing**".

### E.6 Anti-overfitting: stopping the EKG from becoming an elaborate curve fit

| Risk | Control |
|---|---|
| **Look-ahead through graph construction** | `first_public_at` predicate (E.2), enforced in a single query path. **Unit test:** inject an edge with `first_public_at` in the future and assert the retrieved feature vector is unchanged. |
| **The EKG memorises the training window** | Bayesian shrinkage with `k=20` and a hard `support_n ≥ 10` floor before a Lesson can be read (E.3). Decay half-lives (E.3) force old lessons to fade. |
| **Graph rebuilt with hindsight at fold boundaries** | Rebuild (phase 7) uses only Episodes with `exit_t <= fold_train_end`. Each fold gets its **own** EKG snapshot, hashed and archived. |
| **Silent leakage via sentiment** | `Claim` quarantine + 20% feature-mass cap + independent timestamp audit on the Kaggle/Reddit/StockTwits ingest. |
| **Selection over EKG variants** | Every EKG hyperparameter (decay λ, k, support floor, index cap, top-k) is chosen on **validation folds only** and logged. The full candidate set is retained for the data-snooping correction (E.7). |
| **Policy becomes graph-dependent and fails when the graph is stale/wrong** | Graph dropout + null-graph passes during training; **explicit corruption test** at 0 / 5 / 10 / 25 / 50% edge error at evaluation time. |
| **Plasticity loss from continuous updating** | Track dormant-unit fraction, feature effective rank, and weight norm every fold (Part C.2 diagnostics). Apply layer-norm; schedule resets of the final layers if dormancy exceeds threshold; **always keep a periodic full-retrain-from-scratch arm** (C.6 #6). |
| **Regime features hurting PPO** | C.5 shows HMM features moved PPO from Sharpe 1.03 → −0.26 in one study. Therefore the regime channel is a **separately ablated arm**, not a free addition. |

### E.7 THE ABLATION DESIGN (this is what proves the EKG earns its place)

**Fixed across every arm:** same 5 instruments, same walk-forward folds, same seeds `{0…19}`, same TensorTrade env, same cost model (commission + spread + slippage, stated in basis points), same training budget, same network parameter count (parameter-match the no-graph arm with a widened MLP).

| Arm | Description | Purpose |
|---|---|---|
| A0 | Buy-and-hold (per stock + equal-weight) | industry baseline |
| A1 | **Persistence / no-change control** | the "does anything work at all" test (A.5) |
| A2 | Price-only PPO, parameter-matched | **the arm the EKG must beat** |
| A3 | A2 + **last-N-trades flattened into the observation** (no graph) | controls for "memory" without "graph" — the long-context control from B.3 |
| A4 | A2 + **gated EKG state augmentation** (PRIMARY) | the hypothesis |
| A5 | A2 + **retrieval-conditioned** (FALLBACK) | alternative coupling |
| A6 | A4 + A5 | interaction |
| A7 | A4 with **shuffled edges** | graph-structure placebo |
| A8 | A4 with **node-identity permuted** | identity placebo |
| A9 | A4 with **stale graph** (frozen at fold start) | freshness value |
| A10 | A4 with **sector-only static graph** | is the learned structure worth anything over ex-ante structure? (A.5) |
| A11 | A4 + regime channel | isolates the C.5 risk |
| A12 | A4 with EKG-derived **position-size scaling** | soft-constraint arm |
| A13 | Periodic **full retrain from scratch**, no continual updating | C.6 #6 |

**Primary metric (pre-registered, one only):** *net-of-cost annualised Sharpe on the untouched final test folds*, aggregated across seeds as **IQM**.
**Primary comparison (pre-registered, one only):** **A4 vs A2**.

**Statistics:**

| Test | Purpose | Source |
|---|---|---|
| **IQM + stratified bootstrap 95% CI + performance profiles + probability of improvement** (rliable) | correct RL comparison across seeds; percentile CIs work from N=10 runs; IQM = 25%-trimmed mean; 50,000 bootstrap resamples for aggregates, 2,000 for pointwise bands | arXiv:2108.13264; `google-research/rliable` (HTTP 200) |
| **Stationary / moving-block bootstrap** on the daily net return series | daily returns are **not** i.i.d. — never bootstrap them independently | Politis & Romano |
| **Deflated Sharpe Ratio** over the **full logged candidate set** (all 14 arms × all EKG hyperparameter configs) | corrects for selection bias, backtest overfitting and non-normality | Bailey & López de Prado, `davidhbailey.com/dhbpapers/deflated-sharpe.pdf` (SSRN 2460551 **BLOCKED 403**) |
| **White's Reality Check / Hansen's SPA** over the candidate family | data-snooping correction across the whole arm family | Hansen SPA / White RC (secondary source; **primary papers not fetched this session — marked UNVERIFIED**) |

**Decision rule (write this into `docs/ARCHITECTURE.md` before running):**

> Ship the EKG **only if** A4 beats A2 on IQM net Sharpe with a stratified-bootstrap 95% CI excluding zero **and** probability-of-improvement > 0.75, **and** A4 beats A7 and A8 (the placebos) by more than it beats A2 **and** the DSR over the full candidate set exceeds 0.95 **and** A4 does not increase max drawdown or turnover versus A2. If A4 wins but A7/A8 win equally, the gain is capacity, not structure — widen A2 instead and drop the graph.

### E.8 Staged build order for the EKG (dependencies explicit)

| Stage | Entry criteria | Exit criteria |
|---|---|---|
| E-1 Schema + store | Baseline harness exists (A0–A2 runnable) | DuckDB bitemporal edge table + point-in-time query + **future-edge leakage unit test passing** |
| E-2 Mechanical writers | E-1 | Decision/Episode ingest at <1 ms hot path; sealed episodes carry net PnL |
| E-3 Consolidation + decay + prune | E-2, ≥1 full fold of episodes | Lessons with shrinkage + decay + invalidate-don't-delete; weekly prune job; graph snapshot hashed per fold |
| E-4 Coupling (A4) | E-3 | Gated encoder trained; graph dropout + null-graph verified; `edge_ids` logged per decision |
| E-5 Ablation | E-4, 20 seeds affordable on cloud GPU | Full A0–A13 table with rliable CIs + DSR; go/no-go decision recorded |
| E-6 Sentiment quarantine ingest | E-5 = GO, sentiment agent exists | `Claim` nodes ≤20% feature mass; injection red-team test (AgentPoison-style trigger) fails to move the policy |
| E-7 UI feed | E-3 | NetworkX point-in-time subgraph streamed over WebSocket; nodes/edges animate as consolidation runs |

---

## Contradictions with the master brief (required by the protocol)

| Brief says | Evidence says | Recommended amendment |
|---|---|---|
| EKG "reshapes the RL policy's inputs/**reward** over time" | Only potential-based shaping preserves the optimum; an arbitrary KG reward makes a wrong fact the objective (E.5) | Change to "inputs, retrieval context and position sizing". Keep reward = net PnL only. |
| "Take direct inspiration from Prime Intellect's approach to continual, decentralized self-improvement" | Prime Intellect's published continual loop is **environments + verifiers + batch RL runs**, plus a **harness-level** `/refine` that edits prompts/memories/skills with rollback — **not** online weight self-improvement. Their `protocol` repo is **archived**; INTELLECT-3 trained on a **centralised** 64-node H200 cluster | Inspiration is still valid, but the copied mechanism is **verify-then-consolidate with versioned rollback**, not decentralised training. Stated in D.5. |
| "agent decision quality improves the more it trades" | No source demonstrates self-evolving agent memory improving trading outcomes (B.4). Continual World shows CL methods often fail to beat plain fine-tuning (C.6 #1). Nature 2024 shows uninterrupted updating can *collapse* a PPO agent on a stationary task (C.2) | Treat as a **hypothesis to be falsified by E.7**, not a design assumption. Keep A13 (periodic retrain from scratch) as a live competitor. |
| Implicit: a richer graph is better | A no-KG LSTM beat **6 of 7** GCN variants with a single economic link (A.5); Mem0^g can be worse than Mem0; Zep loses 17.7% on single-session questions | Scope the EKG to the **experience side**. With 5 instruments the entity graph is too small to carry signal. |
| Implicit: graph DB choice is open | `kuzudb/kuzu` archived 2025-10-10 | Use DuckDB+Parquet+NetworkX; Neo4j/Memgraph only as fallbacks. |

---

## Verification log

All checks executed **2026-09-14** in the session that wrote this file.

**Totals:** 77 arXiv IDs verified (75 via `rt.verify_arxiv` + 2 via `rt.verify_url` on the abs page), 15 GitHub repos via `rt.gh_repo`, 30 non-arXiv URLs via `rt.verify_url` (28 OK, 2 blocked).


### A. arXiv papers (`rt.verify_arxiv` → abs page)

| arXiv ID | Title (as returned) | URL | Status |
|---|---|---|---|
| 1809.09441 | Temporal Relational Ranking for Stock Prediction | https://arxiv.org/abs/1809.09441 | OK |
| 1908.07999 | HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction | https://arxiv.org/abs/1908.07999 | OK |
| 2407.10909 | FinDKG: Dynamic Knowledge Graphs with Large Language Models for Detecting Global Trends in Fina | https://arxiv.org/abs/2407.10909 | OK |
| 1909.10660 | Exploring Graph Neural Networks for Stock Market Predictions with Rolling Window Analysis | https://arxiv.org/abs/1909.10660 | OK |
| 2401.01846 | DGDNN: Decoupled Graph Diffusion Neural Network for Stock Movement Prediction | https://arxiv.org/abs/2401.01846 | OK |
| 2402.06633 | MDGNN: Multi-Relational Dynamic Graph Neural Network for Comprehensive and Dynamic Stock Invest | https://arxiv.org/abs/2402.06633 | OK |
| 2306.03763 | ChatGPT Informed Graph Neural Network for Stock Movement Prediction | https://arxiv.org/abs/2306.03763 | OK |
| 2502.11371 | RAG vs. GraphRAG: A Systematic Evaluation and Key Insights | https://arxiv.org/abs/2502.11371 | OK |
| 2504.14493 | FinSage: A Multi-aspect RAG System for Financial Filings Question Answering | https://arxiv.org/abs/2504.14493 | OK |
| 2508.17906 | FinReflectKG: Agentic Construction and Evaluation of Financial Knowledge Graphs | https://arxiv.org/abs/2508.17906 | OK |
| 2311.10801 | Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customiza | https://arxiv.org/abs/2311.10801 | OK |
| 2308.11294 | Network Momentum across Asset Classes | https://arxiv.org/abs/2308.11294 | OK |
| 2603.20252 | FinReflectKG -- HalluBench: GraphRAG Hallucination Benchmark for Financial Question Answering S | https://arxiv.org/abs/2603.20252 | OK |
| 2601.11528 | Knowledge Graph Construction for Stock Markets with LLM-Based Explainable Reasoning | https://arxiv.org/abs/2601.11528 | OK |
| 2601.13770 | Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance | https://arxiv.org/abs/2601.13770 | OK |
| 2501.13956 | Zep: A Temporal Knowledge Graph Architecture for Agent Memory | https://arxiv.org/abs/2501.13956 | OK |
| 2502.12110 | A-MEM: Agentic Memory for LLM Agents | https://arxiv.org/abs/2502.12110 | OK |
| 2405.14831 | HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models | https://arxiv.org/abs/2405.14831 | OK |
| 2310.08560 | MemGPT: Towards LLMs as Operating Systems | https://arxiv.org/abs/2310.08560 | OK |
| 2303.11366 | Reflexion: Language Agents with Verbal Reinforcement Learning | https://arxiv.org/abs/2303.11366 | OK |
| 2305.16291 | Voyager: An Open-Ended Embodied Agent with Large Language Models | https://arxiv.org/abs/2305.16291 | OK |
| 2308.10144 | ExpeL: LLM Agents Are Experiential Learners | https://arxiv.org/abs/2308.10144 | OK |
| 2304.03442 | Generative Agents: Interactive Simulacra of Human Behavior | https://arxiv.org/abs/2304.03442 | OK |
| 2407.12784 | AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases | https://arxiv.org/abs/2407.12784 | OK |
| 2503.03704 | Memory Injection Attacks on LLM Agents via Query-Only Interaction | https://arxiv.org/abs/2503.03704 | OK |
| 2508.07407 | A Comprehensive Survey of Self-Evolving AI Agents: A New Paradigm Bridging Foundation Models an | https://arxiv.org/abs/2508.07407 | OK |
| 2410.10813 | LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory | https://arxiv.org/abs/2410.10813 | OK |
| 2601.05504 | Memory Poisoning Attack and Defense on Memory Based LLM-Agents | https://arxiv.org/abs/2601.05504 | OK |
| 2601.11653 | AI Agents Need Memory Control Over More Context | https://arxiv.org/abs/2601.11653 | OK |
| 1612.00796 | Overcoming catastrophic forgetting in neural networks | https://arxiv.org/abs/1612.00796 | OK |
| 2105.10919 | Continual World: A Robotic Benchmark For Continual Reinforcement Learning | https://arxiv.org/abs/2105.10919 | OK |
| 2302.12902 | The Dormant Neuron Phenomenon in Deep Reinforcement Learning | https://arxiv.org/abs/2302.12902 | OK |
| 2205.07802 | The Primacy Bias in Deep Reinforcement Learning | https://arxiv.org/abs/2205.07802 | OK |
| 2502.19009 | Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning | https://arxiv.org/abs/2502.19009 | OK |
| 2604.19737 | Safe Continual Reinforcement Learning in Non-stationary Environments | https://arxiv.org/abs/2604.19737 | OK |
| 2608.18803 | Forgetting, plasticity, and co-observation: a third facet of continual learning | https://arxiv.org/abs/2608.18803 | OK |
| 2505.07291 | INTELLECT-2: A Reasoning Model Trained Through Globally Decentralized Reinforcement Learning | https://arxiv.org/abs/2505.07291 | OK |
| 2501.16007 | TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference | https://arxiv.org/abs/2501.16007 | OK |
| 2407.07852 | OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training | https://arxiv.org/abs/2407.07852 | OK |
| 2412.01152 | INTELLECT-1 Technical Report | https://arxiv.org/abs/2412.01152 | OK |
| 2006.14171 | A Closer Look at Invalid Action Masking in Policy Gradient Algorithms | https://arxiv.org/abs/2006.14171 | OK |
| 1708.08611 | Safe Reinforcement Learning via Shielding | https://arxiv.org/abs/1708.08611 | OK |
| 2410.07071 | Retrieval-Augmented Decision Transformer: External Memory for In-context RL | https://arxiv.org/abs/2410.07071 | OK |
| 2012.02096 | Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design | https://arxiv.org/abs/2012.02096 | OK |
| 2110.02439 | Replay-Guided Adversarial Environment Design | https://arxiv.org/abs/2110.02439 | OK |
| 2108.13264 | Deep Reinforcement Learning at the Edge of the Statistical Precipice | https://arxiv.org/abs/2108.13264 | OK |
| 2501.00989 | Bootstrapped Reward Shaping | https://arxiv.org/abs/2501.00989 | OK |
| 2512.16144 | INTELLECT-3: Technical Report | https://arxiv.org/abs/2512.16144 | OK |
| 2504.19413 | Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory | https://arxiv.org/abs/2504.19413 | OK |
| 2506.06326 | Memory OS of AI Agent | https://arxiv.org/abs/2506.06326 | OK |
| 2511.20857 | Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory | https://arxiv.org/abs/2511.20857 | OK |
| 2507.21046 | A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve (confirmed via abs page) | https://arxiv.org/abs/2507.21046 | OK |
| 2501.01880 | Long Context vs. RAG for LLMs: An Evaluation and Revisits | https://arxiv.org/abs/2501.01880 | OK |
| 2110.10067 | CORA: Benchmarks, Baselines, and Metrics as a Platform for Continual Reinforcement Learning Age | https://arxiv.org/abs/2110.10067 | OK |
| 2308.11958 | Maintaining Plasticity in Continual Learning via Regenerative Regularization | https://arxiv.org/abs/2308.11958 | OK |
| 2311.11557 | Replay-enhanced Continual Reinforcement Learning | https://arxiv.org/abs/2311.11557 | OK |
| 2210.14215 | In-context Reinforcement Learning with Algorithm Distillation | https://arxiv.org/abs/2210.14215 | OK |
| 1910.08348 | VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning | https://arxiv.org/abs/1910.08348 | OK |
| 2504.02281 | FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agents | https://arxiv.org/abs/2504.02281 | OK |
| 1709.06560 | Deep Reinforcement Learning that Matters | https://arxiv.org/abs/1709.06560 | OK |
| 2509.14385 | Adaptive and Regime-Aware RL for Portfolio Optimization | https://arxiv.org/abs/2509.14385 | OK |
| 2608.19488 | When to Retrain: An Empirical Study of Retraining Policies for Streaming ML Under Concept Drift | https://arxiv.org/abs/2608.19488 | OK |
| 2009.12462 | Symbolic Relational Deep Reinforcement Learning based on Graph Neural Networks and Autoregressi | https://arxiv.org/abs/2009.12462 | OK |
| 2310.12931 | Eureka: Human-Level Reward Design via Coding Large Language Models | https://arxiv.org/abs/2310.12931 | OK |
| 2502.18770 | Reward Shaping to Mitigate Reward Hacking in RLHF | https://arxiv.org/abs/2502.18770 | OK |
| 2105.08664 | Deep Graph Convolutional Reinforcement Learning for Financial Portfolio Management -- DeepPocke | https://arxiv.org/abs/2105.08664 | OK |
| 2303.09406 | Exploiting Supply Chain Interdependencies for Stock Return Prediction: A Full-State Graph Convo | https://arxiv.org/abs/2303.09406 | OK |
| 2305.08740 | Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction | https://arxiv.org/abs/2305.08740 | OK |
| 2408.04948 | HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient | https://arxiv.org/abs/2408.04948 | OK |
| 2311.11944 | FinanceBench: A New Benchmark for Financial Question Answering | https://arxiv.org/abs/2311.11944 | OK |
| 2505.17471 | FinRAGBench-V: A Benchmark for Multimodal RAG with Visual Citation in the Financial Domain | https://arxiv.org/abs/2505.17471 | OK |
| 2411.12746 | A Review of Reinforcement Learning in Financial Applications | https://arxiv.org/abs/2411.12746 | OK |
| 2602.16313 | MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks | https://arxiv.org/abs/2602.16313 | OK |
| 2605.24579 | WhenLoss: Diagnosing Write and Retrieval Bottlenecks in Long-Context Memory Systems | https://arxiv.org/abs/2605.24579 | OK |
| 2512.01890 | Elastic Weight Consolidation for Knowledge Graph Continual Learning: An Empirical Evaluation | https://arxiv.org/abs/2512.01890 | OK |
| 2507.21046 | A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve… | https://arxiv.org/abs/2507.21046 | OK (abs page, HTTP 200) |
| 2601.22628 | TTCS: Test-Time Curriculum Synthesis for Self-Evolving | https://arxiv.org/abs/2601.22628 | OK (abs page, HTTP 200) |

### B. GitHub repositories (`rt.gh_repo`)

| Repo | Stars | Archived | Licence | Last commit | Status |
|---|---:|---|---|---|---|
| https://github.com/kuzudb/kuzu | 4025 | **True** | MIT | 2025-10-10 | OK |
| https://github.com/getzep/graphiti | 30866 | **False** | Apache-2.0 | 2026-09-11 | OK |
| https://github.com/letta-ai/letta | 24733 | **False** | Apache-2.0 | 2026-09-10 | OK |
| https://github.com/OSU-NLP-Group/HippoRAG | 4003 | **False** | MIT | 2026-09-03 | OK |
| https://github.com/mem0ai/mem0 | 65270 | **False** | Apache-2.0 | 2026-09-11 | OK |
| https://github.com/PrimeIntellect-ai/prime-rl | 2038 | **False** | Apache-2.0 | 2026-09-13 | OK |
| https://github.com/PrimeIntellect-ai/verifiers | 4614 | **False** | MIT | 2026-09-14 | OK |
| https://github.com/PrimeIntellect-ai/protocol | 139 | **True** | Apache-2.0 | 2025-11-10 | OK |
| https://github.com/PrimeIntellect-ai/OpenDiloco | 592 | **False** | Apache-2.0 | 2025-01-13 | OK |
| https://github.com/PrimeIntellect-ai/genesys | 139 | **False** | Apache-2.0 | 2025-02-28 | OK |
| https://github.com/PrimeIntellect-ai/prime-environments | 257 | **False** | Apache-2.0 | 2026-05-07 | OK |
| https://github.com/agiresearch/A-mem | 1178 | **False** | MIT | 2025-12-12 | OK |
| https://github.com/TongjiFinLab/THGNN | 127 | **False** | GPL-3.0 | 2023-08-31 | OK |
| https://github.com/evgenii-nikishin/rl_with_resets | 107 | **False** | MIT | 2022-05-17 | OK |
| https://github.com/xiaowu0162/LongMemEval | 1085 | **False** | MIT | 2026-05-11 | OK |

### C. Non-arXiv URLs (`rt.verify_url`)

| URL | HTTP | Page title | Status |
|---|---:|---|---|
| https://www.primeintellect.ai/blog/intellect-1-release | 200 | INTELLECT-1 Release The First Globally Trained 10B Parameter Model | OK |
| https://www.primeintellect.ai/blog/intellect-2-release | 200 | INTELLECT-2 Release: The First 32B Parameter Model Trained Through Globally Distributed Re | OK |
| https://www.primeintellect.ai/blog/intellect-3 | 200 | INTELLECT-3: A 100B+ MoE trained with large-scale RL | OK |
| https://www.primeintellect.ai/blog/environments | 200 | Environments Hub: A Community Hub To Scale RL To Open AGI | OK |
| https://www.primeintellect.ai/blog/synthetic-1-release | 200 | SYNTHETIC-1 Release: Two Million Collaboratively Generated Reasoning Traces from Deepseek- | OK |
| https://www.primeintellect.ai/blog/synthetic-2-release | 200 | SYNTHETIC-2 Release: Four Million Collaboratively Generated Reasoning Traces | OK |
| https://www.primeintellect.ai/blog/opendiloco | 200 | OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training | OK |
| https://www.primeintellect.ai/blog/toploc | 200 | TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference | OK |
| https://www.primeintellect.ai/blog/true-agents-model-the-world | 200 | True Agents Model the World | OK |
| https://www.primeintellect.ai/blog/algorithms-layer | 200 | prime-rl gets an Algorithms layer | OK |
| https://www.primeintellect.ai/blog/series-a | 200 | $130M Series A to Build the Open Superintelligence Stack | OK |
| https://www.primeintellect.ai/blog/prime-agent | 200 | Prime Agent: A self-improving RLM agent | OK |
| https://docs.primeintellect.ai/prime-rl/overview | 200 | Overview - Prime Intellect Docs | OK |
| https://www.nature.com/articles/s41586-024-07711-7 | 200 | Loss of plasticity in deep continual learning / Nature | OK |
| https://proceedings.mlr.press/v162/nikishin22a.html | 200 | The Primacy Bias in Deep Reinforcement Learning | OK |
| https://proceedings.mlr.press/v202/sokar23a.html | 200 | The Dormant Neuron Phenomenon in Deep Reinforcement Learning | OK |
| https://proceedings.mlr.press/v139/jiang21b.html | 200 | Prioritized Level Replay | OK |
| https://proceedings.mlr.press/v162/goyal22a.html | 200 | Retrieval-Augmented Reinforcement Learning | OK |
| https://proceedings.mlr.press/v70/pritzel17a.html | 200 | Neural Episodic Control | OK |
| https://proceedings.neurips.cc/paper_files/paper/2023/file/d61d9f4fe4357296cb658795fd7999f0-Paper-Datasets_and_Benchmarks.pdf | 200 |  | OK |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | 403 | Content Blocked | **BLOCKED** |
| https://memgraph.com/benchgraph | 200 | Benchgraph | OK |
| https://github.com/memgraph/memgraph | 200 |  | OK |
| https://duckdb.org/community_extensions/extensions/duckpgq | 200 | duckpgq – DuckDB Community Extensions | OK |
| https://neo4j.com/licensing/ | 200 | Legal terms overview | OK |
| https://mem0.ai/research | 200 | Mem0 Research: LoCoMo, LongMemEval &amp; BEAM Benchmarks | OK |
| https://xiaowu0162.github.io/long-mem-eval/ | 200 | LongMemEval | OK |
| https://github.com/google-research/rliable | 200 |  | OK |
| https://microsoft.github.io/graphrag/ | 200 | Welcome - GraphRAG | OK |
| https://www.sciencedirect.com/science/article/pii/S0925231222005021 | 403 | ScienceDirect | **BLOCKED** |

### D. Sources cited but NOT independently verified in this session — read with caution

| Claim / source | Why it is here | Status |
|---|---|---|
| ACM DOI 10.1145/3604237.3626862 — *Modeling Momentum Spillover with Economic Links…* ("LSTM without KG beat 6 of 7 GCN solutions") | Retrieved as full text by Parallel search on 2026-09-14; the ACM DOI page itself was not re-fetched with `rt.verify_url` | **UNVERIFIED URL — quote is from a live 2026-09-14 fetch of `dl.acm.org/doi/fullHtml/10.1145/3604237.3626862`** |
| `nature.com/articles/s41598-025-32408-w` (graph-attention multi-agent portfolio RL) | Live fetch by deep research 2026-09-14; not re-checked with `rt.verify_url` | UNVERIFIED URL |
| `sciencedirect.com/.../S0957417423025290` (GraphSAGE+PPO) | Live fetch by deep research; ScienceDirect blocks automated clients | UNVERIFIED URL |
| `sciencedirect.com/.../S0925231222005021` (GPM) | `rt.verify_url` → **HTTP 403 BLOCKED** | BLOCKED |
| `papers.ssrn.com/…abstract_id=2460551` (Deflated Sharpe Ratio) | `rt.verify_url` → **HTTP 403 BLOCKED**. Use `davidhbailey.com/dhbpapers/deflated-sharpe.pdf` instead | BLOCKED |
| `link.springer.com/article/10.1007/s41060-025-00854-4` (55 markets, 51% mean accuracy) | Live fetch by deep research 2026-09-14 | UNVERIFIED URL |
| `nature.com/articles/s41599-025-04761-8` (LSTM loses to constant-price on 12 TSE stocks) | Live fetch by deep research 2026-09-14 | UNVERIFIED URL |
| `cloud-conf.net/datasec/2025/…/966100a067.pdf` (HMM+RL Dow 30 table) | Live fetch by deep research 2026-09-14 | UNVERIFIED URL |
| White's Reality Check (1) and Hansen's SPA (2005) primary papers | Named as methodology; primary papers not fetched | **UNVERIFIED — cite only after fetching** |
| Politis & Romano stationary bootstrap | Named as methodology; `users.ssc.wisc.edu` PDF not re-checked | UNVERIFIED URL |
| arXiv:2309.17322 (look-ahead bias in GPT sentiment) | Surfaced in search; **not** passed through `rt.verify_arxiv` | **UNVERIFIED** |
| "PCCL" as an INTELLECT-2 component | Named in the task brief; **not found** in the INTELLECT-2 paper evidence | **NOT FOUND** |
| "ESTIMATE" as a finance GNN method | Named in the task brief; no matching verifiable paper found | **NOT FOUND — citation dropped** |
| arXiv:2512.16144 INTELLECT-3 technical report | Verified to exist; the *benchmark table values* quoted in D.1 come from the deep-research fetch of `arxiv.org/html/2512.16144v1`, not from a manual read | ID OK; numbers second-hand |

### E. Budget used

| Resource | Used | Cap |
|---|---|---|
| Parallel.ai deep-research runs (`ptask_create`, processor=`pro`) | 5 | 5 |
| Parallel.ai searches (`psearch`) | 24 | ~40 |
| `rt.verify_arxiv` calls | 77 | — |
| `rt.gh_repo` calls | 15 | — |
| `rt.verify_url` calls | 32 | — |

---

*End of dossier. Raw deep-research reports retained at `/tmp/ekg_deep/{A_kg_gnn,B_memory,C_continual,D_primeintellect,E_coupling}.md` with their citation JSON.*
