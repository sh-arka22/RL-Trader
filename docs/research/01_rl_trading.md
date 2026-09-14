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
