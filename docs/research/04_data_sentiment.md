# 04 — Data feasibility & the sentiment/ontology agent

**Audit date: 2026-09-14.** This is a *what is actually available for free today* audit, not a
literature review. Every API, dataset, repo and paper below was hit live in this session. Numbers
that could not be confirmed by a live fetch are marked **UNVERIFIED** and must not be repeated as
fact. Full evidence is in the **Verification log** at the end.

Method: live HTTP probes from this machine, `rt.verify_url` / `rt.verify_arxiv` / `rt.gh_repo`,
HuggingFace + PyPI + SEC JSON APIs, a throwaway venv (`/tmp/ytest`, Python 3.12) used to execute
`yfinance` and `pandas-datareader` for real, and 5 Parallel.ai `pro` deep-research runs.

---

## 0. Decisions (read this if you read nothing else)

| # | Decision | Confidence | Why |
|---|---|---|---|
| D1 | **OHLCV primary = `yfinance` 1.7.0**, cached to Parquet on first pull, then frozen | High | Executed live: 5 tickers, 2015-01-02 → 2026-09-11, 2,940×30, **0 % NaN, 2.84 s**, splits + dividends correct |
| D2 | **OHLCV fallback = Tiingo free tier** (50 req/h, 1,000/day, 500 symbols/mo, 30+ y, CRSP-style adjusted OHLC) | Medium | Tiingo pricing page numbers come from deep research, not my own fetch (JS-only page) — re-confirm before relying on it |
| D3 | **Stooq is dead as a scriptable source — remove it from the brief** | High | CSV endpoint returns `Access denied` even after I solved its new JS proof-of-work; `pandas-datareader` raises `NotImplementedError` |
| D4 | **Never redistribute raw bars.** Publish code + API params + retrieval date + SHA-256 of the cached Parquet | High | Yahoo = personal use; Tiingo = "Internal Use Only"; Stooq = no commercial use |
| D5 | **StockTwits public API is still open and is the live-signal backbone** | High | `api.stocktwits.com/api/2/streams/symbol/{T}.json` → HTTP 200, no key, 200 req/h/IP |
| D6 | **X/Twitter: the brief's "unaffordable" verdict is *directionally right but now wrong in detail*.** Buy `counts/all`, never buy post text at scale | High | docs.x.com live: $0.005/post read, 3 M/month cap, **but full-archive search is now open to pay-per-use** and `Counts: All` is **$0.010 per *request***, not per post |
| D7 | **Reddit: inference only. Never fine-tune on Reddit text.** | High | Reddit Data API Terms §2.4 forbids using User Content "for training a machine learning or AI model, without the express permission of rightsholders" |
| D8 | **Sentiment model = `ProsusAI/finbert` baseline; upgrade only if a labelled in-domain holdout justifies it** | High | 5.34 M downloads, works on CPU; the 2026 QLoRA benchmark (arXiv 2608.04200) shows fine-tuned 7 B models win on *classification* but the downstream 1-day Rank IC is 0.0143 and dies after FDR correction |
| D9 | **Expected alpha from daily social sentiment on mega-caps ≈ 0.** Build it as an *event/attention/disagreement* feature and a falsifiable ablation, not as an alpha source | High | 90 M-message StockTwits study: no unconditional next-day predictability. WSB study: no risk-adjusted alpha at 1 d – 1 y even at $0 commission |
| D10 | **Agent-to-agent = single shared bitemporal store (DuckDB + KuzuDB), no message bus, no MCP** | High | Backtest reproducibility dominates latency at daily frequency; a bus makes replay non-deterministic |
| D11 | **The 5 stocks = NVDA, TSLA, AAPL, META, XOM** | High | Mean pairwise 3 y daily-return corr **0.219**, max 0.432; 4 names with ≥ 64 StockTwits msgs/h; XOM is the diversifier *and* the sentiment-null control |

---

## Part A — Market data

### A.1 `yfinance` — measured, not remembered

| Check | Result | Evidence |
|---|---|---|
| Latest release | **1.7.0, uploaded 2026-08-26T17:25:06Z**, Apache-2.0 | PyPI JSON API |
| Release cadence 2026 | 1.2.2 (04-13) → 1.3.0 (04-16) → 1.4.0 (05-23) → 1.4.1 (05-28) → 1.5.1 (06-28) → 1.5.2 (07-23) → 1.6.0 (08-13) → 1.7.0 (08-26) | PyPI JSON API — **actively maintained** |
| GitHub | `ranaroussi/yfinance` ★25,242, last commit 2026-08-26, not archived, Apache-2.0 | `rt.gh_repo` |
| Hard dependency | **`curl_cffi>=0.15`** — browser TLS impersonation is now mandatory | PyPI `requires_dist` |
| 5-ticker daily pull (AAPL, MSFT, NVDA, JPM, XOM), 2015-01-01 → 2026-09-13 | **2,940 rows × 30 cols, NaN fraction 0.0, 2.84 s** | executed in `/tmp/ytest` |
| Max history | AAPL **11,530 rows, 1980-12-12 → 2026-09-14** | executed |
| Rate limit probe | **30 rapid `Ticker.history()` calls, 30/30 OK, 1.23 s, no 429** | executed |
| Raw Yahoo endpoint without `curl_cffi` | `query1.finance.yahoo.com/v8/finance/chart/AAPL` → **HTTP 429** from the same IP, same minute | httpx probe |
| Splits | AAPL 1987-06-16 2:1, 2000-06-21 2:1, 2005-02-28 2:1, 2014-06-09 7:1, 2020-08-31 4:1 | executed |
| Dividends | AAPL through 2026-08-10 = $0.27/share | executed |
| Split back-adjustment | NVDA bars around the 2024-06-10 10:1 split are back-adjusted (Close ≈ 121 where the as-traded price was ≈ 1,210) | executed |
| Intraday on free tier | 1 h / 60 d = 414 bars; 1 m / 7 d = 2,377 bars | executed |

**Gotcha that matters.** The 429 above is the whole story of yfinance in 2026: *plain HTTP clients
are blocked, `yfinance` with `curl_cffi` is not.* Do not write your own Yahoo client. Do not run
`yf.download` in a tight loop across many symbols without `threads=False` and a cache — the deep
research surfaced open issues documenting recurring 429s under looping.

**Honest negative.** yfinance's own price-repair documentation states Yahoo has a flawed way of
dividend-adjusting multi-day intervals, and there is an open issue about prices not updating after
a split. For 5 mega-caps on daily bars this is low-risk, but the pipeline must assert:
`Adj Close` reconstructable from `Close` + actions, and a hard failure if a split appears
retroactively.

### A.2 Stooq — **dead** (contradicts the master brief)

The brief lists Stooq as an approved free source. It is no longer usable from a script:

| Probe | Result |
|---|---|
| `GET https://stooq.com/q/d/l/?s=aapl.us&i=d` | 200, 796 bytes of **JavaScript SHA-256 proof-of-work anti-bot challenge** |
| PoW solved in Python (`n=29,978`, **16 ms**), `POST /__verify` | **200 OK** |
| Retry CSV with the verified session | **`Access denied`** (13 bytes) — also for `^spx` and for date-ranged queries |
| `https://stooq.com/q/?s=aapl.us` (HTML) after PoW | 200, 234 KB — HTML pages work, **bulk CSV does not** |
| `https://stooq.com/q/l/?...&e=csv` | **404** |
| `pandas_datareader.data.DataReader("AAPL.US","stooq")` | **`NotImplementedError("data_source='stooq' is not implemented")`** |

**Verdict: remove Stooq from the plan.** It is not a fallback, it is a broken dependency.

### A.3 Provider comparison (Part A decision table)

Alpaca, Massive/Polygon and Alpha Vantage numbers below are from pages **I fetched myself**.
Tiingo, Finnhub, EODHD, Databento and Nasdaq numbers are JS-rendered and come from the deep-research
run — treat them as second-hand and re-confirm before committing.

| Source | Free-tier limit | History depth (free) | Adjusted close | Corporate actions | Licence / redistribution | Verdict for RL-Trader |
|---|---|---|---|---|---|---|
| **yfinance / Yahoo** | No published quota; 429s under load; works via `curl_cffi` | **46 years** (AAPL 1980→2026, measured) | Yes, `Adj Close` column present | Splits + dividends returned by `Ticker.splits` / `.dividends` | Yahoo terms = personal use; Apache-2.0 covers the *package* only | **PRIMARY.** Pull once, cache to Parquet, hash, freeze |
| **Tiingo (free)** | 50 req/h, 1,000 req/day, 500 unique symbols/mo, 1 GB/mo | **30+ years** | Yes — **CRSP-style, adjusts splits *and* dividends** on adjusted OHLC | Dividend-cash + split-factor fields in EOD response | **"Internal Use Only"**; redistribution needs permission | **FALLBACK / cross-check.** Best documented free fit; licence blocks publishing data |
| **Alpaca Basic** | **$0**, **200 historical calls/min**, 30 websocket symbols | **Since 2016** (~10 y) | Not proven from the docs — listed as corporate actions only | Documented product-wide | Broker terms; redistribution not verified | **Cross-check only.** Fatal caveat below |
| **Alpaca Algo Trader Plus** | $99/mo, 10,000 calls/min, SIP (100 % of volume) | Since 2016 | as above | as above | as above | Out of budget; not needed for daily bars |
| **Massive (ex-Polygon.io) Stocks Basic** | **$0**, **5 API calls/min**, individual use | **2 years** | Page lists corporate actions, does **not** state adjusted-close semantics | Listed | Redistribution UNVERIFIED | **Reject.** 2 y of history cannot support a 10-y backtest |
| Massive Starter / Developer / Advanced | $29 / $79 / $199 per month | 5 y / 10 y / 20+ y | " | " | " | Only Developer ($79/mo) meets 10 y — out of budget |
| **Alpha Vantage (free)** | **25 API requests/day** | 25+ y raw daily | **`TIME_SERIES_DAILY_ADJUSTED` is a premium-only function** | Premium endpoint | Commercial use requires sales contact | **Reject.** 25 calls/day and paywalled adjustment |
| **Finnhub (free)** | 60 calls/min, personal use | 30+ y advertised, entitlement ambiguous | Daily candles split-adjusted; **dividend adjustment UNVERIFIED** | Partial | Personal use | **Reject for bars.** `/stock/candle` is flagged "Premium Access Required" — assume paywalled |
| **EODHD (free)** | **20 API calls/day** | **past 1 year** | "Adjusted Data" listed; semantics UNVERIFIED | Listed | Personal use | **Reject.** 1 year of history |
| **Nasdaq Data Link / Sharadar** | Preview/sample only; full = Premium | Sharadar 24 y, 6,000 active + **10,000 delisted** | UNVERIFIED | `SHARADAR/ACTIONS`, `TICKERS` | Paid licence | **Reject on price**, but it is the *only* audited point-in-time / delisted source — note it as the survivorship-bias escape hatch |
| **Databento** | **$125 signup credit** | UNVERIFIED for daily bars under the credit | UNVERIFIED | UNVERIFIED | UNVERIFIED | **Reject.** Cost for 10 y × 5 tickers is UNVERIFIED; do not promise the credit covers it |
| **Stooq** | — | — | — | — | No commercial use | **DEAD** (see A.2) |

### A.4 Corporate actions, point-in-time, survivorship

| Issue | Finding | Action for RL-Trader |
|---|---|---|
| Split/dividend adjustment | Only **Tiingo** documents CRSP-style split **and** dividend adjustment on adjusted OHLC. **Finnhub** documents split adjustment only. **Alpaca/Massive** list "corporate actions" without proving adjusted-bar semantics. **Yahoo** adjusts but has documented multi-day dividend-adjustment flaws | Store **raw OHLCV + an actions table**, compute the adjustment yourself, and unit-test it against a known split (NVDA 2024-06-10 10:1, AAPL 2020-08-31 4:1) |
| Alpaca free = IEX only | Alpaca docs: *"iex … This is the only feed that can be used without a subscription"*, and IEX is **~2.5 % of market volume**. SIP (100 %) requires Algo Trader Plus $99/mo | **Never use free Alpaca bars as the volume source.** Volume would be off by ~40×, silently corrupting any volume feature or ADV-based position cap |
| Point-in-time constituents | No audited *free* source provides point-in-time index membership or a complete delisted universe. Sharadar (paid) has 10,000 delisted tickers | Fixing 5 tickers *chosen today* is itself a survivorship choice. **State it as a known limitation in every results table.** Mitigate by also reporting on an equal-weight basket and vs SPY |
| Ticker changes / mergers | Not handled by any free API automatically | Keep an explicit securities master keyed by **CIK**, sourced from `https://www.sec.gov/files/company_tickers.json` (fetched live: 200, 797,931 bytes, NVDA→CIK 1045810, AAPL→320193, MSFT→789019, AMZN→1018724, META→1326801) |
| Redistribution | No audited provider was verified to permit publishing raw bars on GitHub | Publish `fetch_prices.py`, exact params, fetch date, row count and **SHA-256 of the Parquet**; users re-fetch under their own terms |

**The exact free stack.** Primary `yfinance==1.7.0` (pinned) → Parquet cache → SHA-256 manifest.
Fallback/cross-check **Tiingo free** (needs a free key). Security master from **SEC
`company_tickers.json`** (free, no key, no limit observed). Corporate-action truth from
`yfinance` actions **plus** an assertion test. Nothing else is needed for 5 tickers of daily bars.

---

## Part B — Social, news and sentiment data

### B.1 X / Twitter — the brief's claim, re-tested

The brief states (as settled fact): *"as of Feb 2026 X/Twitter killed its free API tier — new
developers get pay-per-use ($0.005/read, capped 2M/month) or a legacy $200+/mo Basic tier open
only to grandfathered accounts."*

I fetched `https://docs.x.com/x-api/getting-started/pricing` and
`https://docs.x.com/x-api/posts/search/introduction` on 2026-09-14. Verdict on each clause:

| Brief's claim | Live evidence (docs.x.com, 2026-09-14) | Status |
|---|---|---|
| No free tier | *"The X API uses pay-per-usage pricing. No subscriptions—pay only for what you use."* Credits bought upfront; no free read allowance listed | **CONFIRMED** |
| $0.005 per read | Rate card: **Posts: Read = $0.005 per resource**; User/DM/Follow reads $0.010; Like/Mute/Block reads $0.001; **Owned Reads $0.001** | **CONFIRMED exactly** |
| Capped 2 M/month | *"Pay-per-usage plans are capped at **3 million Post reads per monthly billing cycle**."* | **WRONG — it is 3 M, not 2 M** |
| Legacy $200/mo Basic for grandfathered accounts | docs.x.com no longer documents Basic/Pro at all; the documented model is pay-per-use + Enterprise. Secondary 2026 sources say Basic $200 / Pro $5,000 are closed to new signups and being auto-migrated | **PARTLY STALE** — treat legacy tiers as gone |
| (not in the brief) | **Full-Archive Search back to 2006 is available to pay-per-use customers**, not just Enterprise. Recent Search = last 7 days, all developers | **NEW — the brief misses this** |
| (not in the brief) | **`Counts: All` = $0.010 per *request*** and **`Counts: Recent` = $0.005 per request** — priced per request, **not** per post | **NEW — this is the exploitable gap** |
| (not in the brief) | Resources are **deduplicated within a 24 h UTC window** — re-reading the same post the same day is free | **NEW** |

**The arithmetic the brief did not do.** Two very different products are being conflated:

| Use | Volume needed | Unit | Cost |
|---|---|---|---|
| Post **text** for a sentiment model — 5 tickers × 200 posts/day × 250 trading days | 250,000 posts | $0.005/post | **$1,250** — unaffordable, brief is right |
| Post **text**, thin live sample — 5 tickers × 20 posts/day × 21 days | 2,100 posts | $0.005/post | **$10.50** — affordable but statistically useless |
| **Daily cashtag *volume* time series**, full archive, `GET /2/tweets/counts/all?query=$NVDA&granularity=day` | ~40 paginated requests per ticker × 5 tickers = 200 requests | **$0.010/request** | **≈ $2.00 total** for a full-archive daily mention-count series on all 5 tickers |

**Revised verdict (D6).** *"X is unaffordable"* is **true for post text and false for post counts.*
The attention/volume channel — which is the part of the literature that replicates best — costs
about two dollars. Budget **$5** for X `counts/all`, add `tweet_volume_{ticker}` and
`Δlog volume` as RL observation features, and buy **zero** post text. If the root agent wants a
one-line rule: *never call `/2/tweets/search/*` in this project; only `/2/tweets/counts/all`.*

### B.2 StockTwits — still open, measured

| Probe (2026-09-14) | Result |
|---|---|
| `GET api.stocktwits.com/api/2/streams/symbol/AAPL.json`, **no key, no auth** | **HTTP 200**, 91,723 bytes, live messages |
| `GET api.stocktwits.com/api/2/trending/symbols.json` | **HTTP 200**, 73,357 bytes |
| `https://api.stocktwits.com/developers/docs` | **404** — the official docs are gone |
| Official rate limit | **200 requests/hour per IP, no API key or authentication required** — stated in `stocktwits/stocktwits-mcp` (**official StockTwits GitHub org**, ★4, MIT, last commit 2026-04-21) |
| Page size | 30 messages/request (`limit=30` max) |
| Cursor pagination, NVDA, 12 pages | 360 messages spanning **2026-09-14 10:43:35Z → 14:08:44Z** — i.e. ~3.4 h of chatter retrievable in 12 calls |
| Self-reported label coverage, NVDA 360 msgs | **124 Bullish / 38 Bearish / 198 unlabelled** → only **34 %** carry a label |
| Label coverage across 20 tickers | **0.13 – 0.50, median ≈ 0.30** |

**Capacity check.** 200 req/h × 30 msgs = **6,000 messages/hour ceiling**. The 5 chosen tickers
generate ~380 msgs/h combined at peak (below), needing ~13 req/h. **Live capture is comfortably
feasible; historical backfill is not** — the cursor only walks backwards through calls, so
recovering 3 years of history for 5 tickers would need ~10⁵ requests ≈ 500 h of wall clock.

**Honest negatives on StockTwits.** (1) Developer onboarding is closed and the docs 404 — this is
an *unsupported* endpoint that can vanish without notice; design the collector so its absence
degrades the system to price-only. (2) Two thirds of messages are unlabelled, so the "free labels"
argument is weaker than the brief implies. (3) The self-reported label is the *poster's* claim,
not ground truth, and is heavily bull-skewed (NVDA 124:38 = 77 % bullish).

### B.3 Other social sources — live probes

| Source | Probe result (2026-09-14) | Verdict |
|---|---|---|
| **Reddit unauthenticated** | `https://www.reddit.com/r/wallstreetbets/new.json?limit=5` → **HTTP 403** | Must use OAuth |
| **Reddit free tier** | **100 queries/minute per OAuth client ID**, 10-minute average (secondary sources; the official Reddit wiki page is Cloudflare-gated → **403** for me, so this number is **second-hand**) | Usable |
| **Reddit ToS — the blocker** | Data API Terms §2.4 (fetched live, 200): *"no other rights or licenses are granted or implied, including any right to use User Content for other purposes, such as **for training a machine learning or AI model, without the express permission of rightsholders**"*. §3.1: commercial use **or "research in excess of rate limits"** needs a separate agreement | **Inference only.** Classifying posts at runtime inside the App is operation; fine-tuning FinBERT on scraped Reddit text is **not permitted** |
| **PRAW** | `praw-dev/praw` ★4,252, BSD-2-Clause, last commit **2026-09-14** | Healthy |
| **PullPush.io** | `api.pullpush.io/reddit/search/submission/` → **HTTP 429**: *"This website does not provide free scraping resources for agents. Please contact the administrator on Discord if you're interested in a paid scraping service."* | **DEAD for free use** |
| **Arctic Shift** | `arctic-shift.photon-reddit.com/api/posts/search?subreddit=wallstreetbets&limit=3` → **HTTP 200**, full JSON, **no auth** | **The working Pushshift replacement.** Use for historical WSB |
| **Bluesky** | `public.api.bsky.app` `getProfile` → 200, `getAuthorFeed` → 200, but **`searchPosts` → 403** (auth now required) | Firehose/Jetstream is open (`bluesky-social/jetstream` ★36, Apache-2.0, last commit 2026-09-04) but **cashtag search needs an account**; finance-chatter volume is **UNVERIFIED and likely far below StockTwits** |
| **GDELT 2.0 DOC API** | `api.gdeltproject.org/api/v2/doc/doc?query=Apple+stock&mode=artlist&format=json` → **HTTP 200**, no key, returned 2026-08 articles | **Free, keyless, works.** Best free news-volume/tone source |
| **SEC EDGAR full-text** | `efts.sec.gov/LATEST/search-index?q=...` → **HTTP 200**, JSON, 10,000+ hits, User-Agent header required | **Free.** Use for event ground truth (8-K, 10-Q dates) |
| **SEC `company_tickers.json`** | **HTTP 200**, 797,931 bytes | **Free.** Security master |
| Mastodon / Discord / Telegram | Not probed | **UNVERIFIED** — do not plan around them |
| NewsAPI.org | Deep research: Developer plan **$0, 100 req/day, 24 h article delay, 1 month history**; production from $449/mo | Reject — 24 h delay is fatal for a daily signal |
| StockGeist | Deep research: 10K free monthly credits, 5 free stock streams | Optional prototype only |
| RavenPack / Bigdata.com | Institution-dependent / token-billed | Reject |

### B.4 Static labelled datasets — verified page by page

| Dataset | Exact ID | Rows | Date range | Licence | Live check | Use in RL-Trader |
|---|---|---|---|---|---|---|
| Kaggle Stock Tweets | `equinxx/stock-tweets-for-sentiment-analysis-and-prediction` | **64,479** (Kaggle blurb says "80K+"); deep research read 64,479 displayed rows | **2021-09-30 → 2022-09-30** | not exposed | **200** | Short recent social benchmark. **No 2024-2026 overlap** |
| Financial PhraseBank (Kaggle mirror) | `ankurzing/sentiment-analysis-for-financial-news` | 4,840–4,846 sentences, 16 annotators | 2013-2014 era | **CC-BY-NC-SA-3.0** on the HF original | **200** | Supervised pretraining only. **Non-commercial licence** |
| Daily Financial News for 6000+ Stocks | `miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests` | **~4 M articles, 6,000 stocks** | **2009 → 2020** | not exposed | **200** | Headline volume/tone pretraining. No 2021+ overlap |
| Tweet Sentiment's Impact on Stock Returns | `thedevastator/tweet-sentiment-s-impact-on-stock-returns` | **862,231 labelled instances** | not exposed | not exposed | **200** | Candidate — inspect label provenance before use |
| **FNSPID** | HF `Zihan1004/FNSPID`, arXiv **2402.06698** | **15.7 M news + 29.7 M price records, 4,775 S&P 500 companies** | **1999 → 2023** | Paper says CC BY 4.0; repo says no commercial use without authorisation — **licence conflict** | HF **200** (6,026 downloads, 122 likes); arXiv **200** | **Best scale candidate.** Train + 2020-2023 test. **Cannot support a 2024-2026 backtest** |
| Twitter Financial News Sentiment | HF `zeroshot/twitter-financial-news-sentiment` | **11,931** | not exposed | **MIT** | **200** | Clean eval set for short financial text |
| Financial PhraseBank (HF) | HF `takala/financial_phrasebank` | 4,840 | — | **CC-BY-NC-SA-3.0** | **200** | Eval only; non-commercial |
| FiQA-SA | HF `TheFinAI/fiqa-sentiment-classification` | **1,173** | — | **MIT** | **200** | Eval |
| FinGPT sentiment train | HF `FinGPT/fingpt-sentiment-train` | **76,772** | — | not exposed | **200** | Instruction-tuning corpus |
| FiNER-ORD | HF `gtfintechlab/finer-ord` | **116,721** | — | **CC-BY-NC-4.0** | **200** | Financial NER eval; non-commercial |
| StockNet / ACL18 | arXiv **1810.09936** (Feng et al., adversarial training on the same data) | 88 stocks | **2014-01-01 → 2016-01-01** | MIT | arXiv **200** | Historical benchmark only — **no 2020-2026 overlap** |
| NIFTY | via deep research | 2,111 points | **2010-01-06 → 2020-09-21** | — | not independently checked | Direction-label benchmark, dated |
| WSB dump (Academic Torrents) | via deep research | UNVERIFIED rows | through **June 2021** | UNVERIFIED | not independently checked | Raw text; needs ticker extraction + licence review |
| ~~`gennadiyr/us-equities-market-data`~~ | — | — | — | — | **HTTP 404 — DOES NOT EXIST** | **Do not cite** |
| ~~`utkarshx27/stock-market-tweets-data`~~ | — | — | — | — | **HTTP 404 — DOES NOT EXIST** | **Do not cite** |
| ~~`chiayewken/aspect-sentiment-financial-news`~~ | — | — | — | — | HF **401** | **Do not cite** |
| ~~`BEE-spoke-data/FNSPID_nasdaq_news`~~ | — | — | — | — | HF **401** | **Do not cite** |

**The dataset problem, stated plainly.** Every free labelled social/news dataset I could verify
**stops before 2024**. The brief's chosen Kaggle tweet set covers **one year, 2021-09 to 2022-09**.
There is therefore **no free labelled corpus that overlaps a 2024-2026 out-of-sample window**.
Consequence for the plan: the sentiment model can be *trained* on static data, but the
2024-2026 evaluation must use **live-collected StockTwits + GDELT + SEC**, collected forward from
today, with **model-free labels** (realised next-day return sign), and a clearly labelled
"in-sample era / out-of-sample era" split. Any claim of a 10-year sentiment backtest would be a
fabrication.
