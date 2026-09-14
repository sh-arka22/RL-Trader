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

