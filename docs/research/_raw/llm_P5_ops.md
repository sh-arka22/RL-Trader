# LLM Trading Decision Loops: Cost, Latency, and Reproducibility as of 2026-09-14

## Scope and reporting rules

| Rule | Treatment |
|---|---|
| Price freshness | API and GPU pages were fetched on **2026-09-14**. A provider's own page-update date is shown separately when available. |
| Missing trading-reproducibility fields | Every reported trading-performance number includes asset universe, exact test window, transaction-cost assumption, and baseline. Missing source fields are shown as **NOT REPORTED**. |
| Latency benchmark fields | Latency is an inference benchmark, not a trading backtest. The four trading fields are therefore shown as **NOT REPORTED**, rather than inferred. |
| Currency and units | USD; API prices are per 1M tokens; GPU prices are per GPU-hour; latency is seconds and output tokens/sec. |

## 1. Current published API pricing

| Provider | Tier / model | Input / 1M | Output / 1M | Prompt-cache terms | Batch terms | Fetch date | Exact source URL |
|---|---|---:|---:|---|---|---|---|
| OpenAI | Current flagship: `gpt-6-astra` | $10.00 | $50.00 | Cached input $1.00; cache write $12.50 [8] | Explicit percentage NOT REPORTED; a separate Batch pricing section is present [8] | 2026-09-14 | https://developers.openai.com/api/docs/pricing |
| OpenAI | Cheap: `gpt-5-nano` | $0.05 | $0.40 | Cached input $0.005; cache write NOT REPORTED [8] | NOT REPORTED | 2026-09-14 | https://developers.openai.com/api/docs/pricing |
| OpenAI | Mini: `gpt-5-mini` | $0.25 | $2.00 | Cached input $0.025; cache write NOT REPORTED [8] | NOT REPORTED | 2026-09-14 | https://developers.openai.com/api/docs/pricing |
| Anthropic | Highest-priced listed model; flagship designation NOT REPORTED: Claude Fable 5.1 | $10.00 | $50.00 | 5m write $12.50; 1h write $20.00; cache hit $0.25 [3] | 50% off input and output [3] | 2026-09-14 | https://platform.claude.com/docs/en/about-claude/pricing |
| Anthropic | Current cheap: Claude Haiku 4.5 | $1.00 | $5.00 | 5m write $1.25; 1h write $2.00; cache hit $0.10 [3] | 50% off input and output [3] | 2026-09-14 | https://platform.claude.com/docs/en/about-claude/pricing |
| Anthropic | Older listed model, retired except Bedrock and Google Cloud: Claude Haiku 3.5 | $0.80 | $4.00 | 5m write $1.00; 1h write $1.60; cache hit $0.08 [3] | 50% off input and output [3] | 2026-09-14 | https://platform.claude.com/docs/en/about-claude/pricing |
| Google Gemini | Current announced model: Gemini 3.8 Flash; exact price NOT REPORTED on the extracted current pricing table [5] | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | 2026-09-14 | https://ai.google.dev/gemini-api/docs/pricing |
| Google Gemini | Flash tier: Gemini 3.5 Flash | $1.50 | $9.00 | Cached input $0.15; storage $1.00 per 1M tokens/hour [5] | $0.75 input; $4.50 output [5] | 2026-09-14 | https://ai.google.dev/gemini-api/docs/pricing |
| Google Gemini | Cheap tier: Gemini 3.5 Flash-Lite | $0.30 | $2.50 | Cached input $0.03; storage $1.00 per 1M tokens/hour [5] | $0.15 input; $1.25 output [5] | 2026-09-14 | https://ai.google.dev/gemini-api/docs/pricing |
| DeepSeek | Largest listed: `deepseek-v4-pro` / DeepSeek-V4-Pro-0813; flagship designation NOT REPORTED | $0.66 off-peak / $1.32 peak | $1.98 off-peak / $3.96 peak | Cache hit $0.022 off-peak / $0.044 peak [4] | NOT REPORTED | 2026-09-14 | https://api-docs.deepseek.com/quick_start/pricing/?article_id=article_1779470751466_8 |
| DeepSeek | Cheap: `deepseek-flash` / DeepSeek-V4.1-Flash | $0.15 off-peak / $0.30 peak | $0.60 off-peak / $1.20 peak | Cache hit $0.003 off-peak / $0.006 peak [4] | NOT REPORTED | 2026-09-14 | https://api-docs.deepseek.com/quick_start/pricing/?article_id=article_1779470751466_8 |
| xAI | Current flagship: `grok-4.6`, short context | $2.00 | $6.00 | Cached input $0.50 [6] | NOT REPORTED for this row | 2026-09-14 | https://docs.x.ai/developers/pricing |
| xAI | Current flagship: `grok-4.6`, long context | $4.00 | $12.00 | Cached input $1.00 [6] | NOT REPORTED for this row | 2026-09-14 | https://docs.x.ai/developers/pricing |
| xAI | Cheap listed: `grok-4.20-multi-agent-0309`, short context | $1.25 | $2.50 | Cached input $0.20 [6] | 20% off input, output, cached, and reasoning tokens [6] | 2026-09-14 | https://docs.x.ai/developers/pricing |
| xAI | Cheap listed: `grok-4.20-0309-reasoning`, short context | $1.25 | $2.50 | Cached input $0.20 [6] | 20% off input, output, cached, and reasoning tokens [6] | 2026-09-14 | https://docs.x.ai/developers/pricing |
| Mistral | Large tier: Mistral Large 3 | $0.50 | $1.50 | Cached input $0.05 [13] | 50% batch reduction; cached input up to 90% lower [2] | 2026-09-14 | https://docs.mistral.ai/inference/pricing |
| Mistral | Flagship-priced tier: Mistral Medium 3.5 | $1.50 | $7.50 | Cached input $0.15 [13] | 50% batch reduction; cached input up to 90% lower [2] | 2026-09-14 | https://docs.mistral.ai/inference/pricing |
| Mistral | Cheap: Mistral Small 4 | $0.15 | $0.60 | Cached input $0.015 [13] | 50% batch reduction; cached input up to 90% lower [2] | 2026-09-14 | https://docs.mistral.ai/inference/pricing |
| Mistral | Cheapest listed: Ministral 3 3B | $0.10 | $0.10 | Cached input $0.01 [13] | 50% batch reduction; cached input up to 90% lower [2] | 2026-09-14 | https://docs.mistral.ai/inference/pricing |
| Together AI | DeepSeek V4 Pro 0813 | $1.32 | $3.96 | Cached input $0.13 [7] | Model-specific discount NOT REPORTED [7] | 2026-09-14 | https://www.together.ai/pricing |
| Together AI | DeepSeek V4 Flash 0731 | $0.14 | $0.28 | Cached input $0.03 [7] | Model-specific discount NOT REPORTED | 2026-09-14 | https://www.together.ai/pricing |
| Together AI | Qwen3.8-2.4T-A95B | $2.00 | $6.00 | Cached input $0.25 [7] | Model-specific discount NOT REPORTED | 2026-09-14 | https://www.together.ai/pricing |
| Together AI | Qwen3.8 Flash | $0.15 | $0.47 | NOT REPORTED [7] | Model-specific discount NOT REPORTED | 2026-09-14 | https://www.together.ai/pricing |
| Together AI | Llama serverless rows | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED [7] | 2026-09-14 | https://www.together.ai/pricing |
| Fireworks | Llama, Qwen, and DeepSeek serverless rows | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED; the retrieved official text gives no exact model prices [9] | 2026-09-14 | https://fireworks.ai/pricing; https://docs.fireworks.ai/guides/querying-text-models |

| Provider-wide pricing note | Exact source-stated term | Source URL |
|---|---|---|
| DeepSeek peak schedule | Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday; off-peak rates are half of peak rates [4] | https://api-docs.deepseek.com/quick_start/pricing/?article_id=article_1779470751466_8 |
| xAI page freshness | Pricing page last updated September 7, 2026 [6] | https://docs.x.ai/developers/pricing |
| Google Batch pricing | The pricing page states a 50% Batch reduction [5] | https://ai.google.dev/gemini-api/docs/pricing |
| Anthropic cache plus Batch | Batch and prompt-cache discounts can be combined; the table above reports the page's exact cache prices and 50% Batch discount [3] | https://platform.claude.com/docs/en/about-claude/pricing |

## 2. Self-hosting GPU rental cost

| Provider / source type | GPU | Price per GPU-hour | Price type / condition | Page update or verification date | Exact source URL |
|---|---|---:|---|---|---|
| Prime Intellect marketplace tracker | A100 | NOT REPORTED | Offer and type NOT REPORTED; tracker says data refreshes every 15 minutes [20] | NOT REPORTED | https://computecomparison.com/provider/prime-intellect |
| Prime Intellect marketplace tracker | H100 | $0.94 | Spot / interruptible; unspecified H100; verified September 12, 2026 [25] | Verified 2026-09-12; fetched 2026-09-14 | https://gpurentalprices.com/providers/primeintellect |
| Prime Intellect marketplace tracker | H200 | NOT REPORTED | Offer and type NOT REPORTED [20] | NOT REPORTED | https://computecomparison.com/provider/prime-intellect |
| RunPod official | A100 PCIe | $1.19 Community; $1.59 Secure | On-demand [10] | Page updated 2026-09-13; fetched 2026-09-14 | https://www.runpod.io/pricing |
| RunPod official | A100 SXM | $1.39 Community; $1.59 Secure | On-demand [10] | Page updated 2026-09-13; fetched 2026-09-14 | https://www.runpod.io/pricing |
| RunPod official | H100 PCIe | $1.99 Community; $2.89 Secure | On-demand [10] | Page updated 2026-09-13; fetched 2026-09-14 | https://www.runpod.io/pricing |
| RunPod official | H100 SXM | $2.69 Community; $3.49 Secure | On-demand [10] | Page updated 2026-09-13; fetched 2026-09-14 | https://www.runpod.io/pricing |
| RunPod official | H100 NVL | $2.59 Community; $3.19 Secure | On-demand [10] | Page updated 2026-09-13; fetched 2026-09-14 | https://www.runpod.io/pricing |
| RunPod official | H200 | $3.59 Community; $4.59 Secure | On-demand [10] | Page updated 2026-09-13; fetched 2026-09-14 | https://www.runpod.io/pricing |
| Lambda official | A100 SXM 80GB | $2.79 | Instance price per GPU-hour [12] | NOT REPORTED | https://lambda.ai/pricing |
| Lambda official | A100 SXM 40GB / PCIe 40GB | $1.99 | Instance price per GPU-hour [12] | NOT REPORTED | https://lambda.ai/pricing |
| Lambda official | H100 SXM 80GB | $3.99-$4.19 | Instance offers [12] | NOT REPORTED | https://lambda.ai/pricing |
| Lambda official | H100 PCIe 80GB | $3.29 | Instance price per GPU-hour [12] | NOT REPORTED | https://lambda.ai/pricing |
| Lambda official | H200 | NOT REPORTED | Not listed in extracted instance prices [12] | NOT REPORTED | https://lambda.ai/pricing |
| Vast.ai official | A100/H100/H200 | NOT REPORTED | Live official page states prices are supply-and-demand driven but exposes no exact value in the retrieved page [11] | Fetched 2026-09-14 | https://vast.ai/pricing |
| Vast.ai marketplace comparison | A100 | $0.80 | Indicative on-demand list price | August 2026 timestamp; fetched 2026-09-14 [16] | https://gpu.consulting/gpu-rental-cost |
| Vast.ai marketplace comparison | H100 | $1.65 | Indicative on-demand list price | August 2026 timestamp; fetched 2026-09-14 [16] | https://gpu.consulting/gpu-rental-cost |
| Vast.ai marketplace comparison | H200 | $2.20 | Indicative on-demand list price | August 2026 timestamp; fetched 2026-09-14 [16] | https://gpu.consulting/gpu-rental-cost |

## 3. Measured latency

| Model / provider | TTFT | Output speed | Measurement conditions explicitly stated | Asset universe | Exact trading test window | Transaction cost | Baseline | Exact source URL |
|---|---:|---:|---|---|---|---|---|---|
| GPT-5.6 Sol max / OpenAI API | 134.67 sec [23] | 57.6 tokens/sec [23] | Artificial Analysis defines output speed as tokens/sec after the first streamed chunk and TTFT as seconds to first answer token including reasoning time; measurement date/window, p50/p95, and average status are NOT REPORTED [23] | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | https://artificialanalysis.ai/models/gpt-5-6-sol |
| Nemotron 3.5 Lightning / NVIDIA, median across serving providers | 0.87 sec [24] | 286.6 tokens/sec [24] | Text input/text output; 1M-token context; values are medians across providers; measurement date/window and p50/p95 are NOT REPORTED [24] | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | https://artificialanalysis.ai/models/nemotron-3-5-lightning |
| Current OpenAI, Anthropic, Gemini, DeepSeek, xAI, Mistral, Together, and Fireworks models not covered by the two cited measurements | NOT REPORTED | NOT REPORTED | A provider-comparable, current, published TTFT/tokens/sec table was NOT REPORTED in the retrieved official pricing/model pages | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | Provider URLs listed in Section 1 |

## 4. Multi-agent trading-loop latency and reproducibility evidence

| System / source | Reported operational measurement | Asset universe | Exact test window | Transaction cost | Baseline | What is and is not reproducible | Exact source URL |
|---|---|---|---|---|---|---|---|
| TradingAgents paper | 11 LLM calls and 20+ tool calls per prediction [15] | Apple, Nvidia, Microsoft, Meta, Google, and more; reported comparison columns include AAPL, GOOGL, AMZN [15] | January 1-March 29, 2024 [15] | NOT REPORTED [15] | Buy and Hold, MACD, KDJ+RSI, ZMR, SMA [15] | Per-decision latency, runtime, token count, and dollar cost are NOT REPORTED [15] | https://arxiv.org/abs/2412.20138 |
| TradingAgents repository | Debate rounds are configurable; example sets `max_debate_rounds = 2` [14] | NOT REPORTED for repository configuration | NOT REPORTED [14] | NOT REPORTED | NOT REPORTED | Agent roles and workflow are documented, but model calls/decision, tokens/decision, dollars/decision, and latency/runtime are NOT REPORTED [14] | https://github.com/TauricResearch/TradingAgents |
| Five-agent debate latency | Published end-to-end latency per trading decision: NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | NOT REPORTED | The cited TradingAgents source reports call counts, not elapsed time; therefore no fabricated seconds-per-decision estimate is supplied [15] | https://arxiv.org/abs/2412.20138 |

## 5. Trading performance numbers with the required reproducibility fields

| Asset | Strategy | Exact paper metrics in source order: CR, AR, SR, MDD | Asset universe | Exact test window | Transaction cost | Baseline comparison | Exact source URL |
|---|---|---|---|---|---|---|---|
| AAPL | TradingAgents | 26.62, 30.5, 8.21, 0.91 [15] | Apple, Nvidia, Microsoft, Meta, Google; reported comparison table includes AAPL [15] | 2024-01-01 to 2024-03-29 [15] | NOT REPORTED [15] | Buy and Hold, MACD, KDJ+RSI, ZMR, SMA [15] | https://arxiv.org/abs/2412.20138 |
| GOOGL | TradingAgents | 24.36, 27.58, 6.39, 1.69 [15] | Apple, Nvidia, Microsoft, Meta, Google; reported comparison table includes GOOGL [15] | 2024-01-01 to 2024-03-29 [15] | NOT REPORTED [15] | Buy and Hold, MACD, KDJ+RSI, ZMR, SMA [15] | https://arxiv.org/abs/2412.20138 |
| AMZN | TradingAgents | 23.21, 24.90, 5.60, 2.11 [15] | Apple, Nvidia, Microsoft, Meta, Google, and more; reported comparison table includes AMZN [15] | 2024-01-01 to 2024-03-29 [15] | NOT REPORTED [15] | Buy and Hold, MACD, KDJ+RSI, ZMR, SMA; MACD result is NOT REPORTED in the cited row [15] | https://arxiv.org/abs/2412.20138 |

## 6. Published cost per trading decision and token-count evidence

| Source | Published dollar cost per decision | Published tokens per decision | Published calls per decision | Published latency per decision | Required backtest fields | Exact source URL |
|---|---:|---:|---:|---:|---|---|
| TradingAgents paper | NOT REPORTED [15] | NOT REPORTED [15] | 11 LLM calls; 20+ tool calls [15] | NOT REPORTED [15] | Universe: Apple, Nvidia, Microsoft, Meta, Google, and more; dates: 2024-01-01 to 2024-03-29; transaction costs: NOT REPORTED; baselines: Buy and Hold, MACD, KDJ+RSI, ZMR, SMA [15] | https://arxiv.org/abs/2412.20138 |
| TradingAgents GitHub repository | NOT REPORTED [14] | NOT REPORTED [14] | NOT REPORTED [14] | NOT REPORTED [14] | Backtest dates/configuration: NOT REPORTED [14] | https://github.com/TauricResearch/TradingAgents |
| Published cost analysis located in this review | A defensible published $/decision figure: NOT REPORTED | A defensible published tokens/decision figure: NOT REPORTED | The only exact published operational count located is 11 LLM calls and 20+ tool calls per prediction [15] | NOT REPORTED | No cost figure is reported that can be combined with a current price table without inventing token counts | https://arxiv.org/abs/2412.20138; https://github.com/TauricResearch/TradingAgents |

## 7. Provider reproducibility claims

| Provider | Seed / random control | Temperature statement | Fingerprint / backend control | Bit-exact reproducibility | Exact source URL |
|---|---|---|---|---|---|
| OpenAI | `seed` is supported; same seed and parameters should produce the same result on a best-effort basis [22] | Match temperature, top_p, max_tokens, and other request parameters across calls [22] | `system_fingerprint` identifies model weights, infrastructure, and configuration; matching seed, parameters, and fingerprint usually produces mostly identical output [22] | **No**: OpenAI says determinism is not guaranteed and even matching fingerprints can still yield different responses [22] | https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter |
| Mistral | `random_seed` initializes random sampling; the page says different calls generate deterministic results when it is set [21] | Recommended range 0.0-0.7; lower temperature is more focused and deterministic [21] | NOT REPORTED | Bit-exact guarantee NOT REPORTED [21] | https://docs.mistral.ai/api/endpoint/chat |
| Google Gemini | `seed` is listed in the generation-configuration representation, but the retrieved official documentation gives no reproducibility guarantee [17] | Controls randomness; range 0.0-2.0; default varies by model [17] | NOT REPORTED | NOT REPORTED; the official page does not guarantee bit-exact output [17] | https://ai.google.dev/api/generate-content |
| Anthropic | Seed parameter or provider fingerprint: NOT REPORTED in the retrieved official API material | Provider statement sufficient to guarantee temperature-0 bit identity: NOT REPORTED | NOT REPORTED | NOT REPORTED | https://platform.claude.com/docs/en/api/overview |
| xAI | Seed/reproducibility guarantee: NOT REPORTED in the retrieved official developer material | Temperature/reproducibility guarantee: NOT REPORTED | NOT REPORTED | NOT REPORTED | https://docs.x.ai/overview |
| DeepSeek | Seed/reproducibility guarantee: NOT REPORTED in the retrieved official API material | Temperature-0 bit-exact guarantee: NOT REPORTED | NOT REPORTED | NOT REPORTED | https://platform.deepseek.com/docs/api-reference |

## 8. Operational decision table

| Decision question | Evidence-backed answer | Action implication |
|---|---|---|
| Cheapest listed API input/output among rows with explicit prices | Mistral Ministral 3 3B: $0.10/$0.10 per 1M tokens [13]; DeepSeek Flash is $0.15/$0.60 off-peak [4] | Use the exact model and peak schedule in cost models; do not treat headline prices as time-invariant. |
| Cheapest explicit hosted GPU row in this review | Prime Intellect tracker: H100 spot $0.94/hr, verified 2026-09-12 [25]; Vast comparison: A100 $0.80/hr, August 2026 indicative [16] | Treat marketplace prices as interruptible/indicative, not equivalent to a guaranteed on-demand SLA. |
| Does an LLM trading loop have a published seconds-per-decision number? | NOT REPORTED for TradingAgents; the paper reports 11 LLM calls and 20+ tool calls, not elapsed latency [15] | Measure wall-clock latency in the deployed graph, including data tools, retries, queueing, and debate branches. |
| Is bit-exact provider reproducibility achievable? | OpenAI explicitly says no guarantee; Mistral claims deterministic results with `random_seed` but gives no bit-exact guarantee; Gemini lists `seed` without a guarantee [219.19, 219.23; 221.65-67; 223.268] | Store model ID, full request, seed, sampling parameters, tool outputs, response, provider fingerprint where available, and evaluation timestamp. |
| Can a published TradingAgents $/decision figure be reproduced from the paper? | No: token counts and dollars/decision are NOT REPORTED [15] | Do not multiply 11 calls by a nominal price; capture actual input/output token usage per call and price each call by model, cache status, and batch status. |

## References

1. *TradingAgents: Multi-Agents LLM Financial ...*. https://arxiv.org/abs/2412.20138
2. *Pricing | Mistral*. https://mistral.ai/pricing/
3. *Pricing*. https://platform.claude.com/docs/en/about-claude/pricing
4. *Models & Pricing | DeepSeek API Docs*. https://api-docs.deepseek.com/quick_start/pricing/?article_id=article_1779470751466_8
5. *Gemini Developer API pricing  |  Gemini API  |  Google AI for Developers*. https://ai.google.dev/gemini-api/docs/pricing
6. *API Pricing | SpaceXAI Docs*. https://docs.x.ai/developers/pricing
7. *Pricing | Together AI*. https://www.together.ai/pricing
8. *Pricing*. https://developers.openai.com/api/docs/pricing
9. *Text Models*. https://docs.fireworks.ai/guides/querying-text-models
10. *GPU Cloud Pricing | Per-Second H100, A100, RTX | Runpod*. https://www.runpod.io/pricing
11. *GPU Pricing — Live Platform Rates | Vast.ai*. https://vast.ai/pricing
12. *GPU cloud pricing: rent NVIDIA H100, H200, and B200 | Lambda*. https://lambda.ai/pricing
13. *Pricing | Mistral Docs*. https://docs.mistral.ai/inference/pricing
14. *GitHub - TauricResearch/TradingAgents: TradingAgents: Multi-Agents LLM Financial Trading Framework · GitHub*. https://github.com/TauricResearch/TradingAgents
15. *TradingAgents: Multi-Agents LLM Financial Trading Framework*. https://arxiv.org/html/2412.20138
16. *GPU Rental Cost 2026 | H100 Cost per Hour Compared*. https://gpu.consulting/gpu-rental-cost
17. *Generating content  |  Gemini API  |  Google AI for Developers*. https://ai.google.dev/api/generate-content
18. *http://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing*. http://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
19. *http://docs.z.ai/guides/overview/pricing*. http://docs.z.ai/guides/overview/pricing
20. *Prime Intellect GPU Pricing 2026 — H100, A100 & RTX Rates*. https://computecomparison.com/provider/prime-intellect
21. *Chat*. https://docs.mistral.ai/api/endpoint/chat
22. *How to make your completions outputs reproducible with the new seed parameter*. https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter
23. *GPT-5.6 Sol (max) - Intelligence, Performance & Price Analysis | Artificial Analysis*. https://artificialanalysis.ai/models/gpt-5-6-sol
24. *Nemotron 3.5 Lightning - Intelligence, Performance & Price Analysis | Artificial Analysis*. https://artificialanalysis.ai/models/nemotron-3-5-lightning
25. *Prime Intellect Pricing (September 2026): GPU Rental Rates + 5 Live Prices*. https://gpurentalprices.com/providers/primeintellect
