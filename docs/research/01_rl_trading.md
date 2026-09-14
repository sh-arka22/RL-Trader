# 01 — Deep Reinforcement Learning for Trading & Portfolio Management (2022 → Sept 2026)

**Dossier owner:** research subagent `rl-lit` · **Date of all live fetches: 2026-09-14**
**Protocol:** `docs/RESEARCH_PROTOCOL.md`. Every source named below was verified in this session
(`rt.verify_arxiv` / `rt.verify_url` / `rt.gh_repo`). Sources that failed verification are marked
**BLOCKED** or **UNVERIFIED** and must not be cited elsewhere. See the Verification log at the end.

**Evidence budget used:** 5 Parallel.ai deep-research runs (`pro`), 26 `psearch` calls, 45 arXiv ID
checks, 31 URL checks, 12 GitHub repo checks, plus first-party extraction of 4 primary PDFs/HTML
papers and an independently computed baseline study on free daily data (yfinance 1.7.0).

---

## 0. Executive summary — the seven things that actually matter

| # | Finding | Strength of evidence |
|---|---|---|
| 1 | **No DRL algorithm family has a portable edge.** TD3 wins TradeMaster's Dow-30 table; A2C wins an independent hourly DJIA study; PPO wins FinRL's own benchmarks; the 2026 market-impact study shows the *winner changes when you change only the cost model*. The ranking is an artefact of the surrounding stack. | Strong — 4 independent benchmark tables, all verified |
| 2 | **Cost modelling flips signs, not just magnitudes.** In `arXiv:2603.29086` (NASDAQ-100, 2025 OOS), DDPG's margin Sharpe moves **−2.1 → +0.3** and SAC's **−0.5 → −1.2** purely by swapping a flat 10 bps fee for an Almgren–Chriss impact model. Non-optimised TD3 turnover falls 19% → 1%. | Strong — first-party read of the paper HTML |
| 3 | **Statistical deflation does not catch leakage.** `arXiv:2608.27734` plants a look-ahead oracle: design Sharpe **34.7**, eval Sharpe **51.5**, **Deflated Sharpe = 1.00**. DSR/PBO correct for *search*, not for a contaminated information set. Only a structural feature-registry guardrail removed it. | Strong — first-party read |
| 4 | **On a properly deflated, cost-aware, point-in-time US large-cap panel, passive wins.** Same paper: equal-weight buy-and-hold is the *only* arm certified (DSR 0.97); classic momentum / low-vol / reversal L/S all sit at DSR ≤ 0.32 with **PBO = 0.83**; the best of 102 LLM-discovered strategies posts design SR 1.69 → **eval SR 0.18 (+4.7%)** vs buy-and-hold **0.60 (+40.8%)**. | Strong |
| 5 | **The headline FinRL Contest numbers do not survive contact with their own footnotes.** The Sharpe 9.56 of FinRL Contest 2023 is over a **15-trading-day** window, the contest paper itself flags the Sharpe column as *"reported wrongly"*, and **all three winners underperformed the DJIA on cumulative return in both test windows**. | Strong — first-party table extraction from `arXiv:2504.02281v3` |
| 6 | **Realistic net-of-cost expectation for a 5-stock daily strategy: Sharpe 0.0–0.6**, i.e. at or below a frozen equal-weight basket of the same 5 names. My own computed baseline: equal-weight AAPL/MSFT/JNJ/JPM/XOM 2015-01-01→2026-09-11 = **Sharpe 1.07 (rf=0) / 0.97 (rf=2%)** at ~zero turnover. A daily agent must clear that *after* costs. | Strong — self-computed + 2 verified cost-aware papers |
| 7 | **Recommended execution core: PPO (primary) + TD3 (challenger), continuous simplex weights, net-log-return reward with explicit L1 turnover penalty**, inside a purged/embargoed walk-forward with a DSR gate. Rationale in §9. | Synthesis |

**Contradiction of the master brief:** the brief's hypothesis — *"an RL policy (PPO/SAC) is the fast
execution core"* — is **partially supported for PPO, not supported for SAC.** SAC is the single
worst-behaved agent across the two most cost-realistic studies found (see §1, §7). Substitute
**TD3** as the second arm. See §9.3 for the full list of brief assumptions that the evidence
contradicts.

---

## 1. Algorithm families: what was actually used, and what beat what

Algorithm-family claims in this literature are almost never clean. The table records the
*strongest verified* instance of each family and the *contradicting* instance.

| Family | Strongest verified positive instance | Contradicting / negative instance | Net verdict |
|---|---|---|---|
| **DQN / Double / Duelling / Rainbow** | FinRL Contest 2024–25 crypto task: Double-DQN & Duelling-DQN cum. ret. 0.48%, Sharpe 0.21 vs plain DQN 0.34%/0.15 (`2504.02281v3`, Table 7) | Same table: **BTC buy-and-hold returned 0.74%**, beating every DQN variant on return. Ensembles tied at 0.66% | Discrete value methods are *not* competitive for allocation; usable only for single-asset entry/exit |
| **PPO** | FinRL Contest ensemble study: PPO cum. 63.37%, ann. 18.41%, **Sharpe 1.55**, MDD −9.96% vs DJIA 18.95%/0.47 (Dow 30, 2021-01-01→2023-12-01, 0.1% cost) | `2603.29086`: optimised PPO 20% ret / SR 1.06 under flat 10 bps → **15% / 1.03** under Almgren–Chriss. In the margin env PPO falls 15% → 9% and **underperforms QQEW** | Best-supported *baseline*; degrades gracefully under realistic costs; never dominant |
| **A2C / A3C** | `2407.09557` (30 DJIA, **hourly**, train 2022-03-04→2023-12-01, test 2023-12-01→2024-03-01): A2C top on cumulative reward, ahead of PPO, TD3, DDPG, SAC. **Costs not stated** | TradeMaster (29 DJ30, 2012-2021): A2C TR 51.92%, SR 0.750 — mid-table, below TD3 | Cheap and stable; a legitimate third arm, weak evidence base |
| **DDPG** | TradeMaster: TR 56.95%, SR 0.800 — second best of 8 | `2603.29086` margin env: OOS Sharpe **−2.1** under flat 10 bps. FinRL-Meta reproduction: ann. ret **12.7%**, Sharpe **0.88** vs DJIA 19.7%/1.32 — *loses to the index* | Unstable; do not use unmodified |
| **TD3** | TradeMaster: **best overall** — TR 57.15%, SR 0.804, CR 1.616, SoR 1.564. `2603.29086` portfolio-opt env under AC impact: **32% return, best overall** | `2407.09557`: TD3 below A2C. `2603.29086`: non-optimised TD3 is the *only* config that fails to beat QQEW, with 19% daily turnover and $200k/day costs before HPO | Strongest *conditional* performer; requires hard turnover constraints |
| **SAC** | FinRL Contest ensemble study: cum. 50.62%, Sharpe 1.27 (beats DJIA 0.47) | `2603.29086`: margin Sharpe **−0.5 → −1.2** under impact model; needs HPO to cut turnover 5%→2% and costs by 82%. TradeMaster omits SAC from its main table | **Weakest risk/robustness profile of the continuous-control family.** Contradicts the master brief |
| **Ensembles / multi-agent** | FinRL-Meta reproduction (DJIA 30, 2020-07-01→2022-03-31, 0.1% cost, quarterly rolling): Ensemble ann. 25.9%, vol 15.9%, **Sharpe 1.53**, Calmar 2.27, MDD −11.4% vs DJIA 19.7%/14.4%/**1.32**/1.74/−11.3% | FinRL Contest ensemble study: **Ensemble-1 (1.48) < single PPO (1.55)**; Ensemble-3 with 30 agents is *worst* (1.11). Crypto: all 3 ensembles identical at 0.28 | Ensembles buy ~0.2 Sharpe over the index but **do not beat a well-tuned single PPO**. Diminishing/negative returns to ensemble size |
| **Offline RL — CQL / IQL / Decision Transformer** | `2411.17900` (DJIA + 29 constituents, train 2009-01-01→2020-07-01, test 2020-07-01→2021-10-29): GPT-2 LoRA Decision Transformer, A2C expert trajectories → 43.72 ± 2.04% cum., MDD −8.42 ± 0.57%, Sharpe 1.76 ± 0.08; beats CQL, IQL, BC and random-init DT | **Transaction costs not stated.** Result is conditional on the quality of the expert trajectories, which are themselves online DRL agents. Test window is a pure bull run | Promising for *data reuse*, not yet an independent source of edge |
| **Distributional RL — C51 / QR-DQN / IQN** | `2501.04421`: C51 >32% improvement over 5 classical/ML baselines on TTF natural-gas futures | **Zero transaction costs.** ~90-calendar-day evaluation. Not equities. QR-DQN shows unstable CVaR behaviour | No usable equity evidence. Do not adopt |
| **Transformer / sequence policies** | `2502.01992` (FinRLlama, RLMF on LLaMA); FinRL Contest 2024 Task 2: LLM-signal long-short top-3/bottom-3, 7 large caps, test 2022-10-09→2023-12-15 (229 days) → 134.05% cum vs buy-and-hold 72.71% | Cost assumption not stated for the 134.05% figure; 3-day forced holding period; 7-name universe; single run | Interesting, unaudited |
| **Hierarchical RL** | No verified instance met the four-field reporting standard (universe + window + costs + baseline) | — | **Evidence gap.** Do not build on it |

### 1.1 The single most decision-relevant result in the whole corpus

`arXiv:2603.29086` — *Realistic Market Impact Modeling for RL Trading Environments* (2026-03,
first-party read). NASDAQ-100 daily OHLCV **2010-01 → 2026-01**, 90/10 split, HPO on data to
**Jan 2025**, **2025 held out as true OOS**. Benchmark **QQEW** (equal-weight NASDAQ-100, +19% OOS).
Costs: flat **10 bps** baseline vs **Almgren–Chriss** (default half-spread 5 bps, square-root
impact against trailing 21-day dollar volume, permanent impact with exponential decay).

| Environment | Agent | Flat 10 bps | Almgren–Chriss | Beat QQEW (19%)? |
|---|---|---|---|---|
| MACE stock trading | PPO (HPO) | 20% ret, **SR 1.06** | 15% ret, SR 1.03; costs −55%, MDD −20%→−16%, turnover 1.3%→1.1%, vol 19%→15% | Yes, both |
| MACE stock trading | TD3 (HPO) | 15% ret, SR 0.9 | **18% ret, SR 1.1**; turnover 4%→3%, costs −13% | Yes, both |
| MACE stock trading | TD3 (no HPO) | POV 1%, turnover **19%/day**, **$200k/day cost** | turnover 1%, $8k/day | **No — only failing config** |
| Margin (long/short, leverage) | A2C | 5.5% OOS (IS 13%) | 6.5% OOS (IS 17.5%) | **No — all agents fail** |
| Margin | PPO | 15% OOS | 9% OOS | No |
| Margin | DDPG | **SR −2.1** | **SR +0.3** | No |
| Margin | SAC | SR −0.5 | **SR −1.2** | No |
| Portfolio optimisation | TD3 | 26% (worst of agents) | **32% (best overall)** | Yes under AC |

Four conclusions follow, and they should govern RL-Trader's design:
1. **Cost model is a first-class hyperparameter**, not a post-hoc adjustment. It changes the ranking.
2. **HPO's main job is turnover control**, not return. Optimised SAC cut costs 82% (POV 0.46 → 0.18).
3. **Leverage/short environments destroy the result.** Every margin agent lost to equal-weight.
4. Self-disclosed limitation: static NASDAQ-100 composition circa 2021 ⇒ mild survivorship bias.

---

## 2. The master comparison table

Every row carries universe → window → costs → metrics → baseline → beat-B&H verdict → verified URL.
`n/s` = not stated by the source. Rows marked **①** were extracted first-party from the paper in
this session; the rest come from deep-research extraction against a verified URL.

| # | Paper / benchmark | Method | Universe | Test window | Costs modelled | Reported metrics | Baseline | Beat B&H / index? | Verified URL |
|---|---|---|---|---|---|---|---|---|---|
| 1 ① | FinRL Contests, Table 4 (Contest 2023 Task 1) | PPO-Switch ensemble / indicator & alpha-101 feature eng. | Dow Jones 30 | **15 trading days** 2023-10-25→11-14; then **6 days** 11-15→11-22 | 0.1% per buy/sell (env default) | Nik-Elena cum 3.50%, "Sharpe 9.56", MDD −0.40%; WeCan 2.35%/8.07/−0.55%; SZU-FIN-621 1.05%/2.26/−1.36% | DJIA 5.42%, Sharpe 0.45, MDD −1.87% | **NO.** All 3 winners lost to DJIA on return in *both* windows. Paper footnote: *"The results of the Sharpe ratio were reported wrongly."* | https://arxiv.org/html/2504.02281v3 |
| 2 ① | FinRL Contests, Table 5 (ensemble study) | PPO / SAC / DDPG + 3 ensemble sizes | Dow Jones 30 | 2021-01-01→2023-12-01 (734 days), **rolling 30-day train / 5-day val / 5-day test** | 0.1% per buy/sell | **PPO** 63.37% cum, 18.41% ann, **SR 1.55**, SoR 2.44, MDD −9.96%, Calmar 1.85; Ens-1 62.60%/1.48; Ens-2 58.77%/1.33; Ens-3 46.89%/1.11; SAC 50.62%/1.27; DDPG 63.19%/1.47 | DJIA 18.95%/6.15%/**0.47**/−21.94%; Min-Variance 13.9%/7.34%/0.48/−14.9% | **Yes vs DJIA.** But a **30-day training window** is implausibly short and the ensembles *lose to single PPO* | https://arxiv.org/html/2504.02281v3 |
| 3 ① | FinRL Contests, Table 6 (Contest 2025, FinRL-DeepSeek) | PPO / CPPO / GRPO-DAPO / dynamic model selection, + LLM sentiment | Nasdaq-100 constituents | train 2013-01-01→2018-12-31 (1510 d); **test 2019-01-01→2023-12-31 (1258 d)** | 0.1% per buy/sell | Otago Alpha PPO 191.14% cum, SR 1.08, MDD −28.16%; GRPO-DAPO 335.57%, 0.95, **−50.24%**; PPO-100 204.51%; Queen's Gambit PPO+MIST 342.65% with MDD **−92.47%** | S&P 500 90.03% cum; Nasdaq-100 164.52% | **Yes on cumulative return** (PPO 191% vs NDX 165%), **but** Sharpe column mixes daily and annualised conventions, and the paper flags Queen's Gambit's Rachev ratio as *"reported wrongly"*. Winner used **OptionMetrics/WRDS + Bloomberg** put-call data — **not free** | https://arxiv.org/html/2504.02281v3 |
| 4 ① | FinRL Contests, Table 7 (crypto, second-level LOB) | DQN / DDQN / Duelling DQN + ensembles | Bitcoin | n/s | 0.1% per action | Ensembles 0.66% cum, SR 0.28, MDD −0.73%; DDQN 0.48%/0.21; DQN 0.34%/0.15 | **BTC price 0.74% cum, SR 0.20**; fixed-time exit −0.1% | **NO** on return | https://arxiv.org/html/2504.02281v3 |
| 5 ① | FinRL-Meta (NeurIPS 2022 D&B), Table 2 | Ensemble of A2C/PPO/DDPG, quarterly rolling model selection | DJIA 30 | train 2009-04-01→2019-06-30; **test 2020-07-01→2022-03-31** | 0.1% per buy/sell | **Ensemble 25.9% ann, 15.9% vol, SR 1.53, Calmar 2.27, MDD −11.4%**; A2C 23.3%/1.37; PPO 13.1%/0.99; DDPG 12.7%/0.88 | **DJIA 19.7% ann, 14.4% vol, SR 1.32, Calmar 1.74, MDD −11.3%** | **Marginally yes for the ensemble (1.53 vs 1.32).** PPO and DDPG **both lose to the index.** Edge = +0.21 Sharpe on a 21-month window | https://papers.neurips.cc/paper_files/paper/2022/file/0bf54b80686d2c4dc0808c2e98d430f7-Paper-Datasets_and_Benchmarks.pdf |
| 6 ① | TradeMaster (NeurIPS 2023 D&B), Table 2 | 8 algos: A2C, DDPG, TD3, PG, PPO, EIIE, IMIT, SARL | **29 of DJ30** (top unit price), Yahoo Finance daily | 2012–2021, rolling 3-phase split, **last year = test**; exact calendar dates n/s | **Not stated in main text** (platform supports costs/slippage) | TD3 TR 57.15%, **SR 0.804**, CR 1.616, SoR 1.564, MDD 32.36%; DDPG 56.95/0.800; EIIE 49.66/0.756/**MDD 30.86 (best)**; SARL 52.13/0.752; A2C 51.92/0.750; PG 51.17/0.742; PPO 50.99/0.742; **IMIT −4.95/0.102** | **None. Table 2 contains no passive benchmark at all** | **Cannot be determined** — no index row. Mean of 5 seeds; SR spread across 7 working algos is only **0.742–0.804** | https://proceedings.neurips.cc/paper_files/paper/2023/file/b8f6f7f2ba4137124ac976286eacb611-Paper-Datasets_and_Benchmarks.pdf |
| 7 ① | Realistic Market Impact Modeling | A2C/PPO/DDPG/SAC/TD3 × {10 bps, Almgren–Chriss} × {HPO, no HPO} | NASDAQ-100 (static ~2021 list) | data 2010-01→2026-01; **true OOS = 2025** | **10 bps flat vs Almgren–Chriss** (5 bps half-spread, √-impact vs 21-day ADV, decaying permanent impact) | PPO 20%/SR 1.06 → 15%/1.03; TD3 15%/0.9 → 18%/**1.1**; DDPG margin SR **−2.1 → +0.3**; SAC margin **−0.5 → −1.2**; TD3 PO 26% → **32%** | **QQEW +19% OOS** | Stock-trading env: **yes** for all but non-optimised TD3/AC. Margin env: **NO for every agent** | https://arxiv.org/html/2603.29086 |
| 8 ① | What survives honest evaluation? (E1–E3) | Registry-validated LLM strategy discovery + DSR/PBO deflation | **453 liquid US large caps** + SPY (Tiingo), PIT top-200 by 63-day $-volume, re-selected every 21 days | design 2017–2021, **eval 2022–2025** | **1 bp commission + 2 bp spread + √ market impact vs 21-day $-volume + 50 bp/yr borrow**; stressed ×0.5 and ×2 | Leaky oracle: design SR **34.7**, eval **51.5**, **DSR 1.00**. Classic factors: momentum L/S eval SR 0.01, low-vol −0.78, STR −0.20, **all DSR ≤ 0.32, PBO = 0.83**. Best of 102 agent strategies: design SR 1.69 → **eval SR 0.18, +4.7%, DSR 0.86** | **Equal-weight B&H: design 1.15, eval 0.60, +40.8%, DSR 0.97 (only certified arm).** SPY B&H 0.98 / 0.67 / +51.5% / DSR 0.94 | **NO.** Every active strategy, LLM-discovered or classic, was rejected. Passive was certified | https://arxiv.org/abs/2608.27734 |
| 9 | FinRL (library paper) | PPO / TD3 / DDPG | single stocks + ETFs; two DJIA constituent portfolios | 2019-01-01→2020-09-23 | **Cost per run not stated** (env supports 1/1000 or 2/1000) | PPO ann 14.89–44.94%, SR 1.10–1.49; PPO on S&P 500 SR 0.74; TD3 SR 1.38 / 1.03; DDPG 1.28 / 0.98 | Min-variance SR 0.44; DJIA SR 0.48 | Claimed yes — but cost attribution missing and the window is a 21-month bull run | https://ar5iv.labs.arxiv.org/html/2011.09607 |
| 10 | FinRL-Podracer | K parallel PPO pods + generational evolution | NASDAQ-100 | 2019-05-13→2021-05-26 | **0.2%** | daily/minute cum 149.553% / 362.408%; **SR 2.12 / 2.42** | QQQ, RLlib, SB3, FinRL, vanilla Podracer | Claimed yes. **No ensemble-only ablation** ⇒ effect not attributable to the ensemble | https://arxiv.org/abs/2111.05188 |
| 11 | Gort et al. — crypto backtest overfitting | PPO with walk-forward vs 5-fold CV selection, 2,700 HP configs, 50 trials | 10 cryptos (AAVE, AVAX, BTC, NEAR, LINK, ETH, LTC, MATIC, UNI, SOL), 5-min bars | train 2022-02-02→2022-04-30; **test 2022-05-01→2022-06-27** | Market-price execution; **slippage explicitly ignored** | Best PPO-CV **−34.96%** | S&P BDM index −50.78%; equal-weight −47.78%; TD3, SAC | Relative loss reduction only — **all three returns negative** | https://arxiv.org/abs/2209.05559 |
| 12 | Offline DT-LoRA-GPT2 | Decision Transformer (GPT-2 + LoRA) vs CQL, IQL, BC | DJIA index + 29 constituents | train 2009-01-01→2020-07-01; test 2020-07-01→2021-10-29 | **Not stated** | 43.72 ± 2.04% cum, MDD −8.42 ± 0.57%, **SR 1.76 ± 0.08** (A2C expert trajectories) | CQL, IQL, BC, random-init DT | vs offline baselines yes; **no passive baseline, no costs** | https://arxiv.org/abs/2411.17900 |
| 13 | HMM regime detection + FinRL agents | HMM regime posterior appended to state | Dow 30 | 2010–2025, **80/20 split** | **Not stated. No seeds reported** | Sharpe HMM vs no-HMM: A2C 0.75/0.71, DDPG 0.72/0.48, **PPO −0.26/1.03**, TD3 0.62/0.44, SAC 0.69/0.95 | Same algorithm without HMM | **Mixed — HMM *hurts* PPO and SAC.** No index baseline | https://www.cloud-conf.net/datasec/2025/proceedings/pdfs/IDS2025-3SVVEmiJ6JbFRviTl4Otnv/966100a067/966100a067.pdf |
| 14 | Independent DRL comparison | A2C, PPO, DDPG, TD3, SAC | 30 DJIA companies, **hourly** | train 2022-03-04→2023-12-01; test 2023-12-01→2024-03-01 | **Not stated** | A2C top on cumulative reward; PPO, TD3 lower; DDPG, SAC lag | No external baseline stated | Cannot determine | https://arxiv.org/abs/2407.09557 |
| 15 | Risk-averse distributional RL | C51, QR-DQN, IQN, CVaR policies | TTF natural-gas futures | ~90 calendar days | **Zero transaction costs** | C51 >32% over 5 classical/ML baselines | 5 ML baselines + 4 RL policies | Not equities; zero-cost ⇒ unusable as equity evidence | https://arxiv.org/abs/2501.04421 |
| 16 | `rl-trader` (independent open-source null) | PPO, purged/embargoed walk-forward, seed lottery, DSR gate | **Synthetic GBM-with-regime-switching** (n=2000, lookback 32, episode 252) | 4-fold purged/embargoed walk-forward | **5 bps cost + 1 bp slippage**, applied identically in train and eval | RL median-seed OOS **SR −0.637**; seed band **[−0.637, 0.000]**; DM vs B&H t=1.314, **p=0.189**; **DSR 0.000**; turnover 1.000; MDD −27.1% | Buy-and-hold **−0.965**; flat cash 0.000 | **`rl_beats_baseline = false`.** Honest null. *Caveat: metrics are on synthetic data, not real prices* | https://github.com/FatihHekim0glu/rl-trader |

### 2.1 What the table says when you read down the "Beat B&H?" column

Of the 16 rows, **4** show a defensible win over a passive benchmark under a stated cost model
(rows 2, 3, 5, 7-stock-env), **5** show an explicit loss or tie (1, 4, 7-margin, 8, 16), **4** cannot
be evaluated because no passive baseline or no cost model exists (6, 12, 13, 14), and the remaining
3 are unaudited. **The wins cluster in short windows, bull regimes, and 30-asset universes.** The
losses cluster in the studies that modelled costs most carefully (7-margin, 8, 16) — which is
exactly the wrong correlation if DRL alpha were real.

---

## 3. Reward design

### 3.1 Comparison of reward families

| Reward | Who ablated it against what | Result | Verdict for a 5-asset daily portfolio |
|---|---|---|---|
| **Raw log return / PnL** | RA-DRL (Log vs DSR vs MDD, same env, 4 markets, 0.05% cost); Regret-Optimized study (Return-only arm) | RA-DRL: **Log is strongest on return measures.** Regret study: return-only + cost scheduling + bootstrap augmentation **overfits training and fails to produce a profitable OOS strategy** | **Primary reward.** Use *net* log return (after the cost charge), not gross |
| **Differential Sharpe Ratio (Moody & Saffell)** | RA-DRL (vs Log, vs MDD); used as the primary reward in `2603.29086` MACE env (with drawdown penalty) | RA-DRL: DSR strongest on *risk-adjusted* measures, **not on return**; no complete pairwise table published. **No controlled study shows DSR beating plain log return overall** | **Folklore, not fact.** Run as a pre-registered second arm only |
| **Sortino-based** | No clean same-environment ablation located | Sortino is reported as a *metric*, essentially never optimised as a *reward* | Do not adopt; a high reported Sortino does not imply a Sortino reward |
| **Drawdown-penalised** | RA-DRL (MDD reward); Regret study (embedded-drawdown arm); `2603.29086` (DSR − η·ΔDD²) | RA-DRL: MDD reward best on *risk* measures. Regret study: drawdown arm has the **lowest return** while usually having the **best MDD** | A trade-off, not dominance. Useful as a small auxiliary penalty term, not the main objective |
| **CVaR / tail-risk** | No qualifying same-environment equity ablation found | — | **Evidence gap.** Do not adopt |
| **Mean-variance utility** | `2510.06466` includes a portfolio-variance penalty + costs in the reward, but publishes no component ablation | Design choice, unproven | Optional |
| **Transaction-cost / turnover penalty** | Regret study (**L1 penalty on Δweights** + cost scheduler, max cost 0.0025); RA-DRL (proportional cost × |Δw|, 0.05%); `2509.14385` (penalty 0.002) | **The best-supported reward component in the whole literature.** `2603.29086` shows the cost signal *changes training dynamics*, constraining over-trading and enabling OOS convergence | **Mandatory.** Add an explicit L1 turnover term on top of the net-return reward |

### 3.2 Verdict

**Use `r_t = log(V_t / V_{t-1}) − λ·Σ|Δw_i|` as the primary reward.** Costs must be charged inside
the environment *and* echoed as an explicit L1 penalty — the first makes the return honest, the
second gives the policy gradient a dense, non-delayed learning signal about turnover. Treat DSR,
drawdown penalty and CVaR as three separate pre-registered experimental arms, each counted in the
trial ledger for the Deflated Sharpe calculation (§6.3). *Do not* report a Sharpe that was
selected by trying all four rewards without deflating by 4 trials.

---

## 4. State and action design

### 4.1 Action space

| Design | Evidence | Recommendation |
|---|---|---|
| Discrete buy/hold/sell (3 actions per asset) | `2405.01604` implements it; **no controlled small-portfolio head-to-head showing it beats continuous weights** | Not recommended for allocation |
| **Continuous simplex weights (softmax / Dirichlet)** | `2510.06466` (Dirichlet policy), `2605.17307` (Dirichlet over assets + cash, long-only fully invested), `2603.29086` (softmax over N+1 slots = cash + N stocks, then share-level rebalancing trades **clipped to a max fraction of daily volume**) | **Adopt: a 6-dimensional simplex — 5 equity weights + explicit cash.** Feasibility is built into the action; no invalid states |
| Long-short normalised weights | `2405.01604` samples in [−1,1] and normalises by Σ|w| | Only with explicit borrow cost. `2603.29086` shows **every margin/leverage agent underperformed equal-weight** |
| Per-asset order quantities | **No qualifying comparison found** | Evidence gap. Avoid |

**Position sizing:** copy `2603.29086` — convert target weights to share-level trades, then **clip
each trade to a maximum fraction of that day's volume**. This is what stopped the pathological 19%
daily turnover. Also enforce a per-stock weight cap (their cap was 2%; for 5 names use ~40%).

### 4.2 State space and look-back

| Feature family | Verified evidence | Marginal value proven? |
|---|---|---|
| Daily returns (not prices) | Used universally; `2508.03910` warns **state normalisation can *degrade* performance** across IBOVESPA, NYSE and crypto | Returns yes; aggressive normalisation **no** |
| Technical indicators (MACD, RSI, CCI, Bollinger, DX, SMA, VIX, turbulence) | FinRL's 10-indicator set (`2504.02281v3` Table 2); `2603.29086` uses MACD, RSI, CCI, Bollinger, MAs | **No feature-level ablation exists in any verified source.** Treated as a fixed block |
| Moving averages + covariance/correlation matrix | `2405.01604`: 28 prices + 28 ten-day MAs + a 28×28 ten-day correlation matrix | **No removal ablation.** Unproven |
| Multi-horizon momentum / volatility | `2605.17307`: momentum over 1, 5, 20, 60 d; volatility over 5, 20 d; 120-day momentum selection filter | No causal ablation |
| Regime variables (HMM posterior) | HMM-RL Dow-30 study: **helps A2C/DDPG/TD3, hurts PPO (1.03 → −0.26) and SAC (0.95 → 0.69)** | **Actively harmful for the recommended PPO core** |
| Cross-sectional attention / embeddings | `2510.06466`, `2501.17992` (top-500 US stocks, autoencoder + online meta-learning) | Built for large universes; **excessive for 5 assets**, adds degrees of freedom |
| Sentiment / news | FinRL Contest 2025 FinRL-DeepSeek; Contest 2024 RLMF | PPO-DeepSeek 94.43% vs **PPO-100 204.51%** in the same table — the LLM signal *reduced* return in that team's own ablation |
| Order-book features | No qualifying daily-bar evidence | Out of scope for daily bars |

**Look-back window:** there is **no defensible literature answer**. Implementations use 10 days
(`2405.01604`), 32 bars (`rl-trader`), and multi-horizon 1/5/20/60 (`2605.17307`), and **not one
verified source publishes a look-back sweep with held-out selection.** Pre-register a grid of
{5, 10, 20, 60} trading days, select on a forward validation segment only, freeze before test, and
count those 4 configurations in the trial ledger.

**Normalisation:** fit any scaler on the **training segment only**. TradeMaster's published
protocol z-score-normalises each split *"based on the mean and standard deviation of train,
validation and test set, respectively"* — i.e. the test features are scaled using **test-period
statistics**. That is a look-ahead leak baked into a NeurIPS Datasets & Benchmarks paper. Do not
copy it.

---

## 5. THE CRITICAL SIDE — reproducibility and validity failures

This section carries equal weight to §1–§4 by design. **The headline conclusion: as of 2026-09-14,
the published literature supports DRL as a research framework and a conditional benchmark
competitor — not as a validated source of net trading alpha.**

### 5.1 Documented failures, by failure mode

| Failure mode | Verified instance | Why it matters here |
|---|---|---|
| **Statistically meaningless test windows** | FinRL Contest 2023 Task 1: **15 trading days**, then **6 trading days** (`2504.02281v3`, Table 4) | A 15-day Sharpe has ~0.24 years of data. Under the minimum-track-record arithmetic (§5.3) it cannot distinguish SR=9 from SR=0 |
| **Arithmetic errors in the published table** | Same paper, two explicit footnotes: *"The results of the Sharpe ratio were reported wrongly"* (Table 4) and *"The Rachev ratio of Queen's Gambit are reported wrongly and need further scrutinization"* (Table 6) | The single most-cited FinRL contest number (Sharpe 9.56) is disavowed by its own authors |
| **Inconsistent metric conventions in one table** | `2504.02281v3` Table 6 mixes annualised Sharpe (Otago Alpha 1.08) with daily Sharpe (PPO-100 0.0671, S&P 500 0.0448) in the same column | Cross-row comparison in that table is invalid |
| **Winner used paid, non-reproducible data** | Otago Alpha's put-call ratio came from **OptionMetrics via WRDS and a Bloomberg Terminal** | Directly contradicts the master brief's free-data constraint. The best 2025 contest result is not reproducible on free data |
| **No passive baseline at all** | TradeMaster NeurIPS 2023 Table 2 — 8 algorithms, **zero index rows** | You cannot tell whether TD3's TR 57.15% beat or lost to the Dow over the same years |
| **Test-set statistics used for normalisation** | TradeMaster: z-score per split, including the test split | Look-ahead leak in a benchmark paper |
| **Slippage explicitly ignored** | Gort et al. (`2209.05559`) — market-price execution | The most overfitting-aware DRL study found still has no slippage |
| **Zero transaction costs** | `2501.04421` (distributional RL, natural gas) | Result is untranslatable to equities |
| **Costs never attributed to reported numbers** | FinRL library paper (`2011.09607`) reports SR 1.10–1.49 without stating which cost setting produced them | Cannot be reproduced or audited |
| **Instant bar-price fills, no market impact, no order-book, survivorship bias** | **FinRL's own 2026 successor paper, FinRL-X, lists all of these as backtest-to-live distortions** | The maintainers agree the earlier results are distorted |
| **Single seed / seed count not stated** | `2504.02281v3` does not state seeds. HMM-RL study reports no seeds. Henderson et al. (`1709.06560`) document high across-seed variance in deep RL | A reported policy can be a lucky initialisation |
| **Evaluation confined to a bull market** | FinRL-Meta test 2020-07→2022-03; FinRL library test 2019-01→2020-09; Contest 2025 test 2019–2023 | A regime result, not an algorithm result |
| **Ensembles reported as the contribution without an ensemble-only ablation** | FinRL-Podracer (`2111.05188`) | Cannot attribute the 149%/362% return to the ensemble |

### 5.2 The two results that should change RL-Trader's design

**(a) Deflation does not catch leakage.** `arXiv:2608.27734`, experiment E1, 453-stock PIT US
large-cap universe, design 2017–2021, eval 2022–2025, costs 1 bp commission + 2 bp spread +
√-impact + 50 bp borrow:

| Arm | Design SR | Eval SR | **DSR** |
|---|---:|---:|---:|
| **Look-ahead oracle (tomorrow's return, available today)** | **34.7** | **51.5** | **1.00** |
| Buy-and-hold (equal-weight) | 1.15 | 0.60 | 0.99 |
| SPY buy-and-hold | 0.98 | 0.67 | 0.97 |
| Random floor | 0.06 | −0.09 | 0.46 |
| Momentum L/S (leakage-safe) | −0.07 | 0.01 | 0.35 |

A deliberately contaminated strategy passes Deflated Sharpe **and** PBO completely. **Deflation
corrects for *search*, not for a contaminated information set.** The only defence that worked was
structural: a tool/feature registry whose feature space excludes look-ahead **by construction**.
→ RL-Trader must build a causal feature registry and a vectorised-vs-stepwise parity oracle
(the `rl-trader` repo's approach: parity to 1e-10 catches any vectorised path that peeked).

**(b) On a fair, deflated, cost-aware field, nothing active survives.** Same paper, E2 and E3:

| Experiment | Arms | Result |
|---|---|---|
| E2 — classic price factors, 9-trial grid | 126-day momentum, low-vol, short-term reversal (market-neutral rank L/S) | **Every active arm DSR ≤ 0.32. PBO = 0.83** (the search itself is flagged overfit). Only equal-weight B&H clears 0.95 (DSR 0.97); SPY sits at 0.94 |
| E3 — 100 LLM-composed strategies, ledger N=102, PBO=0.01 | Best agent find: RSI × volume z-score, monthly re-rank | Design SR **1.69** → **eval SR 0.18, +4.7% return, DSR 0.86** (fails 0.95). Buy-and-hold eval SR **0.60, +40.8%**. SPY eval SR **0.67, +51.5%** |

The "evaporation curve": the agent's running-best in-sample Sharpe climbed from −0.07 to 1.69 while
the deflation threshold implied by its own trial count climbed to **1.21** — the search partly
outran its bar in-sample and still collapsed out of sample.

### 5.3 Multiple testing: the arithmetic RL-Trader must respect

Computed in this session from the Bailey–López de Prado "false strategy" expression
E[max SR] ≈ σ_SR·[(1−γ)·Z⁻¹(1−1/N) + γ·Z⁻¹(1−1/(Ne))], with γ = Euler–Mascheroni:

| Independent trials N | E[max annual Sharpe] under the **null** (σ_SR = 1) | Min. backtest years to claim a true SR = 1.0 | Min. backtest years to claim a true SR = 0.5 |
|---:|---:|---:|---:|
| 5 | 1.19 | 1.4 | 5.7 |
| 10 | 1.58 | 2.5 | 9.9 |
| 50 | 2.28 | 5.2 | 20.7 |
| **100** | **2.53** | **6.4** | **25.6** |
| 500 | 3.05 | 9.3 | 37.3 |
| 1 000 | 3.26 | 10.6 | 42.4 |
| 5 000 | 3.69 | 13.6 | 54.4 |

Read this next to the contest results: **a Sharpe of 2.5 from a 100-configuration search is exactly
what pure noise produces.** Bailey & López de Prado's own example: 5 years of data supports at most
**45** independent configurations (E[IS SR]=1, E[OOS SR]=0); 2 years supports **7**.

A realistic RL-Trader sweep (3 algorithms × 4 rewards × 4 look-backs × 5 seeds × ~10 W&B HPO
samples) is N ≈ 2 400 trials. That implies a null-expected max Sharpe above **3.4**, and it means a
credible claim of a *true* SR = 0.5 would need **~50 years** of daily data. **You do not have it.**
The only escapes are: (i) shrink the search and record every trial in a ledger, (ii) target a
*relative* claim (beat the frozen 5-stock equal-weight basket) rather than an absolute Sharpe, and
(iii) hold out a genuinely untouched final window.

### 5.4 What fraction of claimed outperformance survives realistic costs?

**There is no published, defensible percentage for DRL specifically.** Reporting one would be false
precision. What the verified evidence gives instead:

| Evidence | Quantity | Context |
|---|---|---|
| Hou, Xue & Zhang — replication of 452 published anomalies | **65% fail** the single-test hurdle with NYSE breakpoints + value weighting; **82% fail** under a multiple-testing hurdle | Published equity anomalies through Dec 2016; costs not modelled; baseline = the original claims |
| McLean & Pontiff — post-publication decay | **~35% average decay** after publication | Return-decay statistic, not a net-trading result |
| Capital Fund Management — systematic strategy decay | **25–50% haircut** after size adjustments | Practitioner note; universe/window/cost detail not fully stated |
| Harvey, Liu & Zhu | A new factor needs **t > 3.0**, not 2.0 | Cross-section of expected returns |
| `2608.27734` — direct measurement on a cost-aware PIT US large-cap panel | **100% of active strategies rejected** (3 classic factors + 102 agent-discovered); only passive certified | 2022–2025 eval, full cost stack |
| `2603.29086` — direct cost-model swap | PPO return **20% → 15%**; DDPG margin Sharpe **−2.1 → +0.3**; SAC **−0.5 → −1.2** | NASDAQ-100, 2025 OOS |
| `2606.00060` — ML Bitcoin, walk-forward, with costs | Formal Sharpe comparisons vs the passive benchmark **do not reject after bootstrap adjustment** | Explicit negative |

**Working assumption for RL-Trader: treat any backtest edge as ~60–80% illusory before you see the
held-out window, and assume the remainder is competing against a 0.8–1.1 Sharpe passive baseline.**

### 5.5 Reproducibility of RL in general

- Henderson et al., *Deep Reinforcement Learning that Matters* (`1709.06560`): across-trial and
  across-seed variance is large enough that few-seed comparisons are uninformative.
- Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* (`2108.13264`, `rliable`):
  point estimates from a handful of runs change conclusions; use stratified-bootstrap CIs,
  performance profiles, and the interquartile mean.
- **Finance makes this strictly worse**: the market sample is short relative to network capacity,
  rewards are non-stationary, returns are autocorrelated, and a different seed changes the *trade
  sequence* and therefore the *realised transaction cost*.
- **No verified financial source publishes a seed-to-seed Sharpe spread.** The one project that
  does is the independent `rl-trader` null: 5 seeds, band **[−0.637, 0.000]** — i.e. the spread
  straddles the entire decision boundary. **Report "not publicly established" rather than a number.**
- Caveat on tooling: `google-research/rliable` is **archived** (last commit 2024-08-12). Vendor it.

---

## 6. Sample efficiency, non-stationarity, and the validation protocol

### 6.1 How much data?

**No verified 2022–2026 paper varies daily training-history length while holding universe, test
window, costs and baseline fixed.** Any "you need N years" claim is unsupported. What exists:

| Source | Training history | Test window | Note |
|---|---|---|---|
| FinRL-Meta | 2009-04-01→2019-06-30 (~10.3 yr) | 2020-07-01→2022-03-31, quarterly rolling | Benchmark split, not an ablation |
| FinRL Contest 2023 | 2010-07-01→2023-10-24 (3 352 days) | 15 + 6 days | Absurd train:test ratio |
| FinRL Contest ensemble study | **30-day rolling train**, 5-day val, 5-day test | 2021-01-01→2023-12-01 | 30 days of training is far too short to be credible |
| `2603.29086` | 2010-01→~2023 (90% of 2010–2026) | **2025 only** | HPO strictly before 2025 |
| `2608.27734` | design 2017–2021 | **eval 2022–2025 (9 yr for the ETF arm)** | Explicit power argument: t ∝ SR·√years, so 9 years is needed to certify SR ≈ 0.7; **4 years cannot certify a moderate edge at all** |

**Design implication:** with ~12 years of free daily data on 5 names, you can certify a *large*
edge or a *relative* edge, but you cannot certify a moderate absolute Sharpe. Plan for a
relative claim.

### 6.2 Non-stationarity and regime handling — the honest negative

The HMM + FinRL Dow-30 study (2010–2025, 80/20 split, **costs not stated, no seeds**):

| Agent | Sharpe with HMM regime state | Sharpe without | Effect |
|---|---:|---:|---|
| A2C | 0.75 | 0.71 | small + |
| DDPG | 0.72 | 0.48 | + |
| TD3 | 0.62 | 0.44 | + |
| **PPO** | **−0.26** | **1.03** | **large −** |
| **SAC** | **0.69** | **0.95** | **−** |

**Adding a regime feature destroyed PPO.** Since PPO is the recommended core, regime augmentation
must be an isolated, falsifiable arm — not a default. This directly qualifies the master brief's
EKG hypothesis: *a slower-moving memory layer that reshapes the RL policy's inputs is exactly the
intervention that broke PPO here.* Build it so it can be ablated out.

### 6.3 Recommended validation protocol for RL-Trader

1. **Point-in-time features only**, enforced by a feature registry. No feature may reference
   data > t. (`2608.27734` E1 — statistics alone will not save you.)
2. **Parity oracle**: the vectorised backtester must match a step-by-step env rollout to 1e-10 for
   any action sequence. Any mismatch = the vectorised path peeked. (`rl-trader`.)
3. **Purged k-fold with an embargo sized to the label horizon**, then **walk-forward** for the
   final chronological evaluation. Purge removes training observations whose label interval
   overlaps the test interval; embargo removes the observations immediately after.
4. **Combinatorial Purged CV (CPCV)** for model *selection* only. Arian, Norouzi & Seco (2024)
   report CPCV with lower PBO and a better DSR statistic than K-fold, purged K-fold and
   walk-forward on synthetic Heston / Merton-jump / drift-burst / regime-switching data plus
   historical S&P 500 — **but the numeric bias values are behind a paywall and I could not verify
   them (BLOCKED: ScienceDirect 403, SSRN 403). Treat the CPCV ranking as UNVERIFIED and use CPCV
   for selection, walk-forward for the headline.**
5. **Trial ledger**: log every seed, reward, look-back, algorithm and HPO sample. Compute the DSR
   with `n_trials = #seeds × #HP configs × #rewards × #look-backs`. This is what the `rl-trader`
   repo does and what `2608.27734` formalises.
6. **Three-gate verdict, as a pure function** (adopt `rl-trader`'s design verbatim):
   `rl_beats_baseline = True` only if **all** hold, net of costs:
   (a) median-seed OOS Sharpe beats the baseline with a **Diebold–Mariano significant** margin;
   (b) **DSR > 0.95** with the honest trial count; (c) the **across-seed Sharpe lower bound > 0**.
7. **Costs applied identically in training and evaluation.** Never train cost-free and evaluate
   with costs.
8. **≥ 5 seeds**, report the full per-seed equity curves and a stratified-bootstrap CI.
9. **Retraining cadence is a hyperparameter, not a convention.** FreqAI's documented example is a
   30-day `train_period_days` / 7-day `backtest_period_days` window moved weekly, plus a
   `live_retrain_hours` example of 0.5 h. Qlib and TradeMaster publish **no** default cadence.
   Compare static / expanding / rolling retrain arms on identical forward windows.

---

## 7. What Sharpe is realistically achievable on 5 liquid US large caps, daily bars, free data?

### 7.1 First-party baseline study (computed in this session)

**Method:** `yfinance` 1.7.0, `auto_adjust=True` (split- and dividend-adjusted), downloaded
2026-09-14; data span 2013-12-02 → **2026-09-11**. Sharpe = annualised mean / annualised σ of daily
returns. Two baskets: **EW5_div** = AAPL, MSFT, JNJ, JPM, XOM (sector-diverse), daily-rebalanced
equal weight; **EW5_tech** = AAPL, MSFT, AMZN, GOOGL, NVDA (hindsight mega-tech). `BH5` = buy once,
never rebalance. Turnover ≈ 0 for BH5; EW5 daily rebalancing is a slight overstatement of a
zero-cost strategy.

**Window 2015-01-01 → 2026-09-11 (2 940 trading days), rf = 0:**

| Asset / basket | Ann. return | Ann. vol | **Sharpe** | Sortino | Max DD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| **SPY** | 13.76% | 17.56% | **0.82** | 1.01 | −33.7% | 0.41 |
| QQQ | 18.90% | 21.91% | 0.90 | 1.16 | −35.1% | 0.54 |
| **EW5_div** (AAPL/MSFT/JNJ/JPM/XOM) | 19.77% | 18.38% | **1.07** | 1.37 | −35.2% | 0.56 |
| BH5_div (no rebalance) | 19.63% | 20.19% | 0.99 | 1.29 | −32.4% | 0.61 |
| **EW5_tech** (AAPL/MSFT/AMZN/GOOGL/NVDA) | 35.94% | 26.88% | **1.28** | 1.72 | −41.8% | 0.86 |
| BH5_tech | 48.64% | 36.70% | 1.26 | 1.78 | −55.3% | 0.88 |
| AAPL | 25.08% | 28.75% | 0.92 | 1.27 | −38.5% | 0.65 |
| MSFT | 24.25% | 27.59% | 0.92 | 1.29 | −37.1% | 0.65 |
| JNJ | 11.35% | 18.35% | 0.68 | 0.91 | −27.4% | 0.41 |
| JPM | 19.21% | 26.90% | 0.79 | 1.07 | −43.6% | 0.44 |
| XOM | 9.67% | 27.48% | 0.47 | 0.67 | −61.3% | 0.16 |
| NVDA | 68.97% | 48.16% | 1.33 | 1.97 | −66.3% | 1.04 |

With **rf = 2%**: SPY **0.71**, EW5_div **0.97**, EW5_tech **1.20**.

**Sub-period Sharpe (rf = 0) — the regime dependence is the whole story:**

| Window | SPY | QQQ | EW5_div | EW5_tech |
|---|---:|---:|---:|---:|
| 2015–2019 | 0.88 | 0.98 | 1.11 | 1.56 |
| 2018–2019 | 0.83 | 0.93 | 1.07 | 1.02 |
| 2020 | 0.67 | 1.28 | 0.58 | 1.53 |
| **2022 (bear)** | **−0.71** | **−1.07** | **+0.15** | **−1.00** |
| 2023 → 2026-09 | 1.41 | 1.46 | 1.78 | 1.79 |

Two honest caveats: (i) **EW5_tech is hindsight-selected** — picking NVDA/AMZN/GOOGL in 2015 was not
obvious, so 1.28 is a selection-biased upper bound; (ii) **EW5_div's +0.15 in 2022 was carried
entirely by XOM (+87.9% that year)** — also partly hindsight. A pre-registered 5-name basket should
be expected nearer SPY, i.e. **0.7–1.0**.

### 7.2 Cost arithmetic — the constraint that decides everything

Annual return drag = 252 × (daily one-way turnover) × (cost in bps):

| Daily one-way turnover | @ 5 bps | @ 10 bps | @ 20 bps |
|---:|---:|---:|---:|
| 5% | 0.63%/yr | 1.26%/yr | 2.52%/yr |
| 10% | 1.26% | 2.52% | 5.04% |
| 25% | 3.15% | 6.30% | 12.60% |
| 50% | 6.30% | 12.60% | 25.20% |
| 100% | 12.60% | 25.20% | **50.40%** |

`2603.29086`'s *non-optimised* TD3 ran **19% daily turnover**. At 10 bps that is **~4.8%/yr** of
pure drag — and its measured cost was $200k/day on the paper's book. A daily-rebalanced 5-stock
agent must be held under ~10% daily one-way turnover or costs eat the entire equity risk premium.

### 7.3 Cost model to use (all components verified live, 2026-09-14)

| Component | Value | Source |
|---|---|---|
| Online commission, listed US stocks/ETFs | **$0** at Schwab (industry fees still apply) | schwab.com/pricing |
| SEC Section 31 fee | **$20.60 per $1M of covered sales = 0.206 bps**, effective 2026-04-04 | SEC FY2026 fee-rate advisory |
| FINRA TAF | Per-share, assessed to the member firm on covered sales; immaterial for liquid large-cap dollar trades | FINRA TAF FAQ |
| Quoted spread, mega-cap | BIS working paper 1229: aggregated large-cap spreads fell **60 bps (1996) → 13 bps (2023)** across US/Europe/Japan. **Not a 2026 AAPL/MSFT quote.** Model **1–5 bps** one-way crossing and disclose it | bis.org/publ/work1229.pdf |
| Slippage / impact | `2603.29086` default half-spread **5 bps** plus √-impact vs 21-day ADV; `2608.27734` uses **1 bp commission + 2 bp spread + √-impact + 50 bp/yr borrow** | both papers |
| **Recommended base case** | **5–12 bps per one-way dollar turnover**; stress at **10–20 bps** | synthesis |

### 7.4 The answer

| Scenario | Net annualised Sharpe | Basis |
|---|---|---|
| In-sample / heavily tuned backtest | **> 1 is routine and meaningless** | FinRL reports 1.10–1.49 without cost attribution; `2608.27734` E3 posts design SR 1.69 → eval 0.18 |
| Cost-aware research backtest, large universe | **0.9–1.1** | `2603.29086` TD3/PPO on NASDAQ-100 under Almgren–Chriss — but that is **100 stocks**, not 5 |
| **Honest OOS planning range, 5 US large caps, daily bars, free data, 5–12 bps one-way** | **0.0 – 0.6** | Cost sensitivity (`2603.29086`), full rejection under deflation (`2608.27734`), the independent null (`rl-trader`), and the 0.82–1.07 passive bar computed in §7.1 |
| Exceptional, replication-worthy | 0.6 – 1.0 | Requires a long untouched OOS spanning 2022-style drawdowns |
| > 1.0 OOS | **Not a reasonable base case** | No verified source demonstrates it for a 5-name daily strategy net of realistic costs |

**Blunt version: the most likely honest outcome is a Sharpe at or below a frozen equal-weight
basket of the same five stocks (0.8–1.1 over 2015–2026), i.e. the DRL agent adds turnover and model
risk without adding risk-adjusted return.** Design the project so that *demonstrating this cleanly*
is a successful outcome.

### 7.5 Free data — verified status, 2026-09-14

| Source | Status | Problem found **in this session** |
|---|---|---|
| `yfinance` 1.7.0 | Works | **4 of 11 tickers failed on the first bulk download** (rate-limited) and needed a staggered retry with 12–48 s sleeps. Docs state it is unaffiliated with Yahoo and "for research and education". Must distinguish split-adjusted `Close` from dividend-adjusted `Adj Close` |
| Stooq | Docs page live (200) | **The CSV download endpoint returned a JS/anti-bot page to an automated request** — not usable as an unattended primary feed |
| Alpaca free tier | Live | $0/mo, **IEX only**, 15-minute delayed API, 200 calls/min, **30-symbol limit**, 7+ years history. Fine for 5 symbols |
| Nasdaq Data Link | Live | Free access varies by dataset; not guaranteed for full US equity history |
| **Survivorship bias** | Unresolved by any free source | None of the above provides a point-in-time, delisted-inclusive US universe. **Freeze the 5 names before the test and disclose the selection bias**, as `2608.27734` does |

---

## 8. Open-source landscape — verified repo status (GitHub API, 2026-09-14)

| Repo | Stars | License | Last commit | Verdict |
|---|---:|---|---|---|
| `freqtrade/freqtrade` | 54 370 | GPL-3.0 | **2026-09-14** | Most actively maintained; FreqAI has the only documented retrain cadence |
| `microsoft/qlib` | 48 551 | MIT | 2026-07-23 | Active; no default cadence published |
| `nautechsystems/nautilus_trader` | 28 924 | LGPL-3.0 | **2026-09-14** | Active; execution-grade |
| `AI4Finance-Foundation/FinRL` | 16 282 | MIT | 2026-07-12 | Active; the baseline-replication target |
| **`tensortrade-org/tensortrade`** | 7 117 | Apache-2.0 | **2026-02-09** | **~7 months stale.** The master brief mandates it as the execution layer — flag this risk |
| **`TradeMaster-NTU/TradeMaster`** | 3 070 | Apache-2.0 | **2025-06-04** | **~15 months stale.** Use its *numbers*, not its code |
| `AI4Finance-Foundation/FinRL-Meta` | 1 938 | MIT | 2026-04-02 | Active enough; `2603.29086` extends it |
| `google-research/rliable` | 878 | Apache-2.0 | 2024-08-12 | **ARCHIVED.** Vendor it |
| `Open-Finance-Lab/FinRL_Contest_2025` | 66 | — | 2025-10-20 | Winner code + reports |
| `Open-Finance-Lab/FinRL_Contest_2023` | 51 | — | 2025-01-21 | Winner code + reports |
| **`TradeMaster-NTU/PRUDEX-Compass`** | 51 | MIT | **2023-06-21** | **Effectively dead.** Re-implement the 6 axes yourself |
| `FatihHekim0glu/rl-trader` | 0 | MIT | 2026-06-27 | **Best methodological template found.** Copy its parity oracle, seed-lottery, DM test and DSR verdict gates |

---

## 9. Recommendation for RL-Trader's execution core

### 9.1 The decision

| Slot | Choice | Why |
|---|---|---|
| **Primary policy** | **PPO** (Stable-Baselines3 / Gymnasium) | Only family with a positive result in *every* verified benchmark family: FinRL Contest ensemble study (SR 1.55 vs DJIA 0.47), FinRL-Meta, FinRL library, and — critically — the cost-realistic `2603.29086` (SR 1.06 → 1.03 under Almgren–Chriss, still beating QQEW). It degrades *gracefully* under cost realism. It is also the FinRL contest reference agent, so baseline replication is direct |
| **Challenger** | **TD3** | Best in TradeMaster (SR 0.804 of 8) and the **only** agent that *improved* under the Almgren–Chriss impact model (return 15%→18%, SR 0.9→1.1; portfolio-opt env 26%→32%). Must be paired with hard turnover clipping — un-tuned TD3 is the pathological-turnover case |
| **Third arm (cheap)** | A2C | Wins the one independent hourly DJIA comparison; cheap to run; use as a variance check |
| **Rejected** | **SAC** | Contradicts the master brief. Worst cost-robustness on record: margin Sharpe **−0.5 → −1.2** under impact modelling; needs HPO to cut turnover 5%→2% and costs 82%. Keep only as an ablation arm |
| **Rejected** | DDPG, DQN/Rainbow, distributional RL, hierarchical RL | DDPG: margin SR −2.1, loses to DJIA in FinRL-Meta (0.88 vs 1.32). DQN family: loses to BTC buy-and-hold in the one verified table. Distributional: zero-cost, non-equity evidence only. HRL: no qualifying evidence at all |
| **Ensembles** | **Defer to Stage 2** | FinRL-Meta shows +0.21 Sharpe over DJIA, but the FinRL Contest study shows **Ensemble-1 (1.48) < single PPO (1.55)** and monotone degradation with ensemble size (1.48 → 1.33 → 1.11). Every ensemble member multiplies the trial count in the DSR denominator |
| **Offline RL / Decision Transformer** | **Defer to Stage 3** | `2411.17900` is the strongest sequence-model result (SR 1.76 ± 0.08) but has **no cost model** and depends on online-DRL expert trajectories you must generate first. Natural fit *after* a working PPO core exists |

### 9.2 Concrete environment specification

- **Action**: 6-dim simplex (softmax over 5 stocks + cash), long-only, no leverage. Convert to
  share-level trades; **clip each trade to a max fraction of that day's volume**; per-name weight
  cap ≈ 40%.
- **Reward**: `log(V_t/V_{t-1}) − λ·Σ|Δw_i|`, costs charged inside the env at 10 bps one-way
  (stress arms at 5 and 20 bps), applied identically in train and eval.
- **State**: daily log returns + FinRL's 10-indicator block, look-back pre-registered over
  {5, 10, 20, 60} and selected on forward validation only. Scalers fitted on train split **only**.
- **Cost model**: start flat 10 bps; add the Almgren–Chriss/√-impact arm from `2603.29086` (it is
  released as a FinRL-Meta extension) before believing any result.
- **Baselines, frozen before any training**: (1) buy-and-hold SPY; (2) buy-and-hold the frozen
  5-name basket; (3) daily-rebalanced equal weight of the same 5; (4) FinRL's stock PPO agent.
  Targets to beat, net of identical costs: **SPY 0.82, EW5 ≈ 1.07** (2015-01-01→2026-09-11, rf=0).
- **Verdict**: the three-gate pure function of §6.3.6.

### 9.3 Where the evidence contradicts the master brief

| Brief assumption | Evidence | Action |
|---|---|---|
| "PPO/**SAC** as the fast execution core" | SAC is the least cost-robust agent found (`2603.29086` margin SR −0.5 → −1.2); TradeMaster omits it entirely | **Replace SAC with TD3** |
| "benchmark against a published RL baseline such as FinRL's PPO agent" | Sound — and FinRL's own PPO **loses to the DJIA** in the FinRL-Meta reproduction (ann. 13.1%, SR 0.99 vs DJIA 19.7%, SR 1.32) | Keep, but expect the baseline itself to lose to the index |
| "EKG reshapes the RL policy's inputs/reward over time" | The nearest verified analogue — HMM regime state appended to a FinRL agent — **destroyed PPO (1.03 → −0.26)** and hurt SAC | Build the EKG as an **ablatable** input channel with an explicit A/B gate, not a default-on layer |
| Sentiment signals improve trading | In FinRL Contest 2025's own table, **PPO-DeepSeek 94.43% < PPO-100 204.51%** on the same universe/window | Sentiment must earn its place by ablation |
| Free data is sufficient | The **winning** 2025 contest entry used OptionMetrics/WRDS + Bloomberg. yfinance rate-limited 4/11 tickers in this session; Stooq's CSV endpoint is bot-blocked | Freeze the 5 tickers, cache raw pulls to disk, add retry + completeness tests |
| TensorTrade as the execution layer | Last commit **2026-02-09**, ~7 months stale | Accept but pin the version and budget for maintenance; `nautilus_trader` and `freqtrade` are far more active |
| Implicit: a good backtest Sharpe proves the system works | A leaky oracle scored **DSR 1.00**; 102 searched strategies all failed deflation; PBO 0.83 on a 9-trial factor grid | Ship the **verdict function + trial ledger before the first agent** |

---

## 10. Verification log

All checks performed **2026-09-14**. `verify_arxiv` returns `None` if a paper does not exist;
every ID below returned a real title. `verify_url` requires HTTP < 400.

| Claim / source | URL | Check | Status |
|---|---|---|---|
| FinRL Contests benchmark paper (all contest tables) | https://arxiv.org/abs/2504.02281 · https://arxiv.org/html/2504.02281v3 | verify_arxiv + verify_url + first-party table extraction | **OK 200** |
| TradeMaster NeurIPS 2023 D&B (Table 2, split protocol) | https://proceedings.neurips.cc/paper_files/paper/2023/file/b8f6f7f2ba4137124ac976286eacb611-Paper-Datasets_and_Benchmarks.pdf | verify_url + first-party PDF extraction | **OK 200** |
| TradeMaster abstract page | https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8f6f7f2ba4137124ac976286eacb611-Abstract-Datasets_and_Benchmarks.html | verify_url | **OK 200** |
| FinRL-Meta NeurIPS 2022 D&B (ensemble Table 2) | https://papers.neurips.cc/paper_files/paper/2022/file/0bf54b80686d2c4dc0808c2e98d430f7-Paper-Datasets_and_Benchmarks.pdf | verify_url + first-party PDF extraction | **OK 200** |
| FinRL-Meta arXiv | https://arxiv.org/abs/2211.03107 | verify_arxiv | **OK** |
| FinRL-Meta (earlier arXiv) | https://arxiv.org/abs/2112.06753 | verify_arxiv | **OK** |
| FinRL library paper | https://arxiv.org/abs/2011.09607 · https://ar5iv.labs.arxiv.org/html/2011.09607 | verify_arxiv + verify_url | **OK 200** |
| FinRL framework paper | https://arxiv.org/abs/2111.09395 | verify_arxiv | **OK** |
| FinRL-Podracer | https://arxiv.org/abs/2111.05188 | verify_arxiv | **OK** |
| Realistic Market Impact Modeling (cost-model swap) | https://arxiv.org/abs/2603.29086 · https://arxiv.org/html/2603.29086v1 | verify_arxiv + first-party HTML extraction | **OK 200** |
| What survives honest evaluation? (leaky oracle, PBO 0.83, E3) | https://arxiv.org/abs/2608.27734 · https://arxiv.org/html/2608.27734v1 | verify_arxiv + first-party HTML extraction | **OK 200** |
| Gort et al., crypto backtest overfitting | https://arxiv.org/abs/2209.05559 | verify_arxiv | **OK** |
| Offline DT-LoRA-GPT2 for quantitative trading | https://arxiv.org/abs/2411.17900 | verify_arxiv | **OK** |
| Independent DRL comparison (hourly DJIA, A2C top) | https://arxiv.org/abs/2407.09557 | verify_arxiv | **OK** |
| Distributional RL, natural-gas futures (C51) | https://arxiv.org/abs/2501.04421 | verify_arxiv | **OK** |
| FinRLlama (Contest 2024 Task 2) | https://arxiv.org/abs/2502.01992 | verify_arxiv | **OK** |
| PRUDEX-Compass | https://arxiv.org/abs/2302.00586 | verify_arxiv | **OK** |
| FinRL-X (2026, backtest-to-live distortions) | https://arxiv.org/abs/2603.21330 · https://ai4finance.org/FinRL-Paper.pdf | verify_arxiv + verify_url | **OK 200** |
| Regret-Optimized Portfolio Enhancement (reward ablation) | https://arxiv.org/abs/2502.02619 | verify_arxiv | **OK** |
| Adaptive and Regime-Aware RL for Portfolio Optimization | https://arxiv.org/abs/2509.14385 | verify_arxiv | **OK** |
| Attention-Enhanced RL (Dirichlet policy) | https://arxiv.org/abs/2510.06466 | verify_arxiv | **OK** |
| Portfolio Management using DRL (10-day MA + correlation state) | https://arxiv.org/abs/2405.01604 | verify_arxiv | **OK** |
| DRL Framework for Diversified Portfolio Mgmt (multi-horizon, Dirichlet+cash) | https://arxiv.org/abs/2605.17307 | verify_arxiv | **OK** |
| RL Portfolio Allocation with Dynamic Embedding (top-500 US) | https://arxiv.org/abs/2501.17992 | verify_arxiv | **OK** |
| Comparing Normalization Methods (normalisation can degrade) | https://arxiv.org/abs/2508.03910 | verify_arxiv | **OK** |
| DRL for Asset Allocation in US Equities (24 stocks) | https://arxiv.org/abs/2010.04404 | verify_arxiv | **OK** |
| A Risk-Aware RL Reward for Financial Trading | https://arxiv.org/abs/2506.04358 | verify_arxiv | **OK** |
| Multimodal DRL for Portfolio Optimization | https://arxiv.org/abs/2412.17293 | verify_arxiv | **OK** |
| RL Framework for Quantitative Trading (cautionary case) | https://arxiv.org/abs/2411.07585 | verify_arxiv | **OK** |
| RL in Financial Decision Making: Systematic Review | https://arxiv.org/abs/2512.10913 | verify_arxiv | **OK** |
| A Review of RL in Financial Applications (meta-analysis) | https://arxiv.org/abs/2411.12746 | verify_arxiv | **OK** |
| ML Bitcoin trading under costs (walk-forward, does not reject) | https://arxiv.org/abs/2606.00060 | verify_arxiv | **OK** |
| Regime-Based Portfolio Allocation Using HMM and RL | https://arxiv.org/abs/2605.27848 | verify_arxiv | **OK** |
| Can LLM Investing Strategies Outperform the Market Long Run? | https://arxiv.org/abs/2505.07078 | verify_arxiv | **OK** |
| Self-Supervised Auxiliary Task Discovery for Stock Trading RL | https://arxiv.org/abs/2608.15841 | verify_arxiv | **OK** |
| FineFT ensemble RL for futures | https://arxiv.org/abs/2512.23773 | verify_arxiv | **OK** |
| Pretrained Time-Series Foundation Models for Return Forecasting | https://arxiv.org/abs/2606.27100 | verify_arxiv | **OK** |
| Deep RL that Matters (seed variance) | https://arxiv.org/abs/1709.06560 | verify_arxiv | **OK** |
| Deep RL at the Edge of the Statistical Precipice (rliable) | https://arxiv.org/abs/2108.13264 · https://agarwl.github.io/rliable/ | verify_arxiv + verify_url | **OK 200** |
| Deflated Sharpe Ratio (Bailey & López de Prado) | https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | verify_url | **OK 200** |
| The Probability of Backtest Overfitting (PBO/CSCV) | https://davidhbailey.com/dhbpapers/backtest-prob.pdf | verify_url | **OK 200** |
| Pseudo-Mathematics and Financial Charlatanism (MinBTL) | http://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf | verify_url | **OK 200** |
| Harvey, Liu & Zhu — t > 3.0 | https://www.nber.org/papers/w20592 | verify_url | **OK 200** |
| Hou, Xue & Zhang — 65% / 82% anomaly failure | http://global-q.org/uploads/1/2/2/6/122679606/houxuezhang2020rfs.pdf | verify_url | **OK 200** |
| McLean & Pontiff — ~35% post-publication decay | http://fmg.ac.uk/sites/default/files/2020-08/Jeffrey-Pontiff.pdf | verify_url | **OK 200** |
| CFM — 25–50% strategy-decay haircut | http://cfm.com/wp-content/uploads/2022/12/312-2021-05-Why-and-how-systematic-strategies-decay.pdf | verify_url | **OK 200** |
| Chen & Zimmermann — Open Source Asset Pricing | https://www.openassetpricing.com/ | verify_url | **OK 200** |
| Open Source Cross-Sectional Asset Pricing (FEDS 2021-037) | https://www.federalreserve.gov/econres/feds/files/2021-037pap.pdf | verify_url | **OK 200** |
| HMM + FinRL Dow-30 regime study (per-agent Sharpe table) | https://www.cloud-conf.net/datasec/2025/proceedings/pdfs/IDS2025-3SVVEmiJ6JbFRviTl4Otnv/966100a067/966100a067.pdf | verify_url + first-party PDF extraction | **OK 200** |
| FreqAI retrain cadence (30d/7d, live_retrain_hours) | https://www.freqtrade.io/en/stable/freqai-running/ | verify_url | **OK 200** |
| FinRL Contest 2025 official site | https://open-finance-lab.github.io/FinRL_Contest_2025/ | verify_url | **OK 200** |
| SecureFinAI Contest 2026 (NOT "FinRL Contest 2026") | https://jinboatus1.github.io/SecureFinAI_Contest_2026/ | verify_url | **OK 200** |
| Schwab $0 online commission | https://www.schwab.com/pricing | verify_url | **OK 200** |
| SEC Section 31 fee $20.60/$1M from 2026-04-04 | https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2 | verify_url | **OK 200** |
| FINRA Trading Activity Fee FAQ | https://www.finra.org/rules-guidance/guidance/faqs/trading-activity-fee | verify_url | **OK 200** |
| BIS WP 1229 — large-cap spreads 60 bps (1996) → 13 bps (2023) | https://www.bis.org/publ/work1229.pdf | verify_url | **OK 200** |
| Alpaca free data tier | https://alpaca.markets/data | verify_url | **OK 200** |
| yfinance documentation | https://ranaroussi.github.io/yfinance/index.html | verify_url | **OK 200** |
| Stooq free-data page | https://stooq.com/db/h/ | verify_url | **OK 200** (CSV endpoint bot-blocked in practice) |
| `AI4Finance-Foundation/FinRL` | https://github.com/AI4Finance-Foundation/FinRL | gh_repo | **OK** 16 282★, MIT, last commit 2026-07-12 |
| `AI4Finance-Foundation/FinRL-Meta` | https://github.com/AI4Finance-Foundation/FinRL-Meta | gh_repo | **OK** 1 938★, MIT, 2026-04-02 |
| `TradeMaster-NTU/TradeMaster` | https://github.com/TradeMaster-NTU/TradeMaster | gh_repo | **OK** 3 070★, Apache-2.0, **2025-06-04 (stale)** |
| `TradeMaster-NTU/PRUDEX-Compass` | https://github.com/TradeMaster-NTU/PRUDEX-Compass | gh_repo | **OK** 51★, MIT, **2023-06-21 (dead)** |
| `tensortrade-org/tensortrade` | https://github.com/tensortrade-org/tensortrade | gh_repo | **OK** 7 117★, Apache-2.0, **2026-02-09 (stale)** |
| `microsoft/qlib` | https://github.com/microsoft/qlib | gh_repo | **OK** 48 551★, MIT, 2026-07-23 |
| `freqtrade/freqtrade` | https://github.com/freqtrade/freqtrade | gh_repo | **OK** 54 370★, GPL-3.0, 2026-09-14 |
| `nautechsystems/nautilus_trader` | https://github.com/nautechsystems/nautilus_trader | gh_repo | **OK** 28 924★, LGPL-3.0, 2026-09-14 |
| `google-research/rliable` | https://github.com/google-research/rliable | gh_repo | **OK** 878★, Apache-2.0, **ARCHIVED 2024-08-12** |
| `Open-Finance-Lab/FinRL_Contest_2025` | https://github.com/Open-Finance-Lab/FinRL_Contest_2025 | gh_repo | **OK** 66★, 2025-10-20 |
| `Open-Finance-Lab/FinRL_Contest_2023` | https://github.com/Open-Finance-Lab/FinRL_Contest_2023 | gh_repo | **OK** 51★, 2025-01-21 |
| `FatihHekim0glu/rl-trader` (honest null + verdict gates) | https://github.com/FatihHekim0glu/rl-trader | gh_repo + README read | **OK** 0★, MIT, 2026-06-27 |
| SPY / mega-cap baseline Sharpe table (§7.1) | yfinance 1.7.0, `auto_adjust=True`, fetched 2026-09-14 | first-party computation | **OK** — raw prices + `baselines.json` saved to `rl_research_state/` |

### 10.1 Sources that FAILED verification — **do not cite these**

| Source | URL | Result |
|---|---|---|
| "DRL for Trading — A Critical Survey" (MDPI *Data* 6(11):119) | https://www.mdpi.com/2306-5729/6/11/119 | **BLOCKED — HTTP 403.** Claims about "very few papers model cost+slippage+spread jointly" are therefore **UNVERIFIED** |
| Arian, Norouzi & Seco — CPCV vs walk-forward comparison | https://www.sciencedirect.com/science/article/pii/S0950705124011110 | **BLOCKED — HTTP 403** |
| Same, SSRN preprint | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376 | **BLOCKED — HTTP 403.** The CPCV-beats-walk-forward ranking is **UNVERIFIED** |
| White — A Reality Check for Data Snooping | https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00152 | **BLOCKED — HTTP 403** |
| Benchmarking DRL trade execution (J. Pac.-Basin Finance) | https://www.sciencedirect.com/science/article/pii/S0927538X25002136 | **BLOCKED — HTTP 403** |
| RA-DRL multi-reward study (Int. J. Comput. Intell. Syst.) | http://link.springer.com/article/10.1007/s44196-025-00875-8 | **PARTIAL — HTTP 200 but served a Cloudflare "Client Challenge" page, not the article.** Its Log-vs-DSR-vs-MDD conclusion in §3.1 is **UNVERIFIED at the source** |
| Hansen — Test for Superior Predictive Ability | https://www.jstor.org/stable/27638834 | **NOT FETCHED** |
| "FinRL Contest 2026" | — | **DOES NOT EXIST.** The 2026 event is *SecureFinAI Contest 2026*. Do not relabel it |

---

## 11. Open questions for the next research pass

1. No verified source gives a **seed-to-seed Sharpe spread** for a DRL trading agent on real data.
   RL-Trader should measure and publish its own — it would be a genuine contribution.
2. No verified source publishes a **look-back-length sweep** with held-out selection.
3. No verified source publishes a **feature-level ablation** of FinRL's 10-indicator block.
4. The **CPCV-vs-walk-forward bias comparison is paywalled**; re-attempt through an institutional
   proxy or reproduce it on synthetic data.
5. No verified study runs a DRL agent on a **3–10 name** US equity universe with full cost
   reporting. This is the exact gap RL-Trader occupies — and the reason its result will be novel
   whether it is positive or null.
