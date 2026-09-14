# Daily-Bar RL Trading: A 2026 Reality Check

## Executive Summary

- **Passive benchmark**: Long samples put the S&P 500 excess-return Sharpe around **0.39-0.51**, while SPY's trailing one-year Sharpe reached **1.45 as of September 10, 2026**; use **0.35-0.60** as a multi-year net planning range, not the recent one-year number as a forecast. [24][8]
- **Technical edge**: A broad US momentum factor produced only **0.32 net Sharpe** in the full sample and a more tradable optimized version produced **0.13**; a 2023 technical-rule study found mostly insignificant or negative out-of-sample results after costs. A credible five-stock daily technical system belongs around **0.0-0.4 net Sharpe**, with **0.4-0.6** an outlier, not a base case. [23][21]
- **Deep RL headline versus reality**: A prominent 2020 ensemble reported **0.87-1.30** out-of-sample Sharpe with a 0.1% per-trade cost, but it traded a 30-stock Dow universe, used close-price execution, and selected models through rolling validation. A 2025 four-stock replication reported no Sharpe, no transaction-cost assumption, overlapping confidence intervals, and inconclusive convergence. [10][26]
- **Execution cost**: Commission can be **0 bps** at major brokers, but quoted spreads around **1 bp** for examples such as AAPL and MSFT do not describe realized retail cost. A 2025 study of 74,709 retail stock observations measured **7-46 bps round-trip** execution cost. Treat that range as combined spread, slippage, routing, and price impact rather than adding another arbitrary slippage number. [29][28][18][22]
- **Turnover dominates**: If one-way portfolio turnover is **10% per day**, a 7-46 bp round-trip cost implies approximately **1.76%-11.59% per year** before commissions and regulatory charges. Replacing all five names each day is **100% one-way turnover**, or roughly **17.64%-115.92% per year** under the same arithmetic. [22]
- **Statistical power**: Lo's standard approximation gives approximately **7.92 annual-equivalent IID observations**, or about **1,997 daily observations / 7.9 trading years**, for a measured Sharpe of 0.8 to exclude zero with a two-sided 95% normal approximation. Serial correlation, non-normality, and model selection make this a lower bound; a five-year backtest is not strong evidence of a 0.8 Sharpe. [25]
- **Regulatory reality**: FINRA's new intraday margin standards became effective **June 4, 2026** and replace the old PDT count and $25,000 minimum, but firms can phase in implementation through **October 20, 2027**. Cash accounts still must respect T+1 settlement and good-faith/freeride restrictions. [11][16][19]
- **Verdict**: For a single pre-specified, genuinely untouched multi-year holdout, my explicit operating prior is **10%-25%**, with **15% central**, that a well-built RL system on exactly five daily-bar US large caps beats both the comparable buy-and-hold benchmark and SPY net of retail execution costs. That is a decision prior, not a published frequency estimate. The highest-probability design is low-turnover, cost-aware, walk-forward, strongly regularized, and benchmarked against simple rules.

## 1. Realistic Sharpe Ranges, Not Backtest Headlines

The useful distinction is between a published backtest result and a planning range for a small retail implementation. The first is a historical statistic conditional on that paper's universe, data cleaning, model selection, execution assumptions, and costs. The second is what a new five-stock daily-bar project should regard as plausible before seeing its result.

| Strategy | Published anchor and test window | Published net or reported Sharpe | 2026 planning range after realistic costs |
|---|---|---:|---:|
| SPY buy-and-hold | S&P 500 proxy, 1928-2025; 1993-2025; 2000-2025; recent SPY trailing year to Sep. 10, 2026 | About **0.42**, **0.51**, and **0.39** for the long samples; **1.45** for the recent one-year SPY page | **0.35-0.60** for a multi-year expectation; a favorable one-year sample can be **1.3-1.5** |
| Classic momentum / technical | US UMD, 1926-2011; tradable optimized US sample, 1980-2011; technical-rule OOS windows formed from rolling 36-month selection and 36-month tests | **0.32** US UMD net; **0.13** optimized tradable net; mostly insignificant or negative OOS technical-rule outcomes at positive costs | **0.0-0.4**; **0.4-0.6** is an unusually strong result requiring robustness |
| Published deep RL | Ensemble Dow-30 study: train 2009-2015, validate Q4 2015, OOS Jan. 2016-May 2020; FinRL examples; four-stock replication using AAPL, AMZN, MSFT, GOOGL | Ensemble **1.30**, component agents **0.87-1.12**; FinRL single-stock PPO examples **1.10-1.49**; 2025 replication reported no Sharpe and was inconclusive | **0.0-0.6** for a new five-stock daily system; **0.6-0.8** is strong; a claimed **>1.0** requires independent replication |

The SPY row is the correct opportunity-cost anchor, not a promise that passive investing delivers the same Sharpe in every window. Damodaran's annual total-return and 3-month Treasury-bill series, updated January 5, 2026, imply approximately 8.4% mean excess return and 19.8% standard deviation for 1928-2025, about 9.66% and 18.9% for 1993-2025, and about 7.63% and 19.7% for 2000-2025. Those calculations produce the roughly 0.39-0.51 Sharpe range. [24] SPY's official June 30, 2026 fact sheet reports a **0.0945% expense ratio**, or **9.45 bps per year**, and annualized total returns of **13.26% over five years** and **15.35% over ten years**. [2]

The recent SPY observation illustrates why a single annualized Sharpe is a poor target. The public calculation page showed **1.45** over the trailing year as of September 10, 2026, using daily returns and excess return over a risk-free rate. [8] That number is real for that window, but it is not evidence that a five-stock learner can produce 1.45 going forward. The mechanism is simple: a strong market regime raises realized excess return faster than realized volatility, while a short sample has wide estimation error.

For the proposed project, report both the realized strategy Sharpe and the contemporaneous SPY Sharpe over exactly the same dates. The actionable conclusion is that beating a recent passive Sharpe of 1.45 is a much harder hurdle than beating the long-run passive range. A model that earns 0.5 net Sharpe may be useful in a weak or sideways market, yet still lose to SPY in a strong bull window.

## 2. Case Study: Momentum's Published Edge Shrinks Under Trading Costs

The strongest clean evidence for classic momentum is not a five-stock chart. Frazzini, Israel, and Moskowitz studied the UMD momentum factor and trading costs. Their US full-sample UMD result was **7.88% gross return, 5.43% net return, 0.48 gross Sharpe, and 0.32 net Sharpe**, with monthly turnover of **1.14%** and estimated market impact of **18.32 bps**. [23] This is a useful positive case: a broad, diversified factor can retain a statistically meaningful but modest net Sharpe after institutional trading costs.

The same paper's more constrained optimized tradable US sample is the more relevant warning for a small implementation. It reported **4.57% gross return, 2.00% net return, 2.56% total trading costs, and 0.13 net Sharpe** for 1980-2011. [23] The mechanism is turnover and capacity: concentrating the signal into a smaller opportunity set makes each decision more important, while execution costs consume a larger fraction of expected return. Five stocks do not create five independent bets; they create concentration in a few correlated exposures.

A separate 2023 study tested **6,406** technical rules across **23 developed and 18 emerging equity indices**. It selected rules in a 36-month test and evaluated them in the next 36 months, with one-way cost assumptions of **0, 10, 25, and 50 bps**. [21] In-sample Sharpe ratios were often above 1 even at the higher cost assumption, but out-of-sample results were mostly insignificant or negative. At 25 or 50 bps, **14 of 18** emerging markets showed significant OOS underperformance, and the best rules underperformed buy-and-hold at positive costs. [21]

That is the failure case a reinforcement-learning project must beat. An RL agent can learn a more flexible mapping than a moving average, but flexibility also increases the number of implicit trials. The practical conclusion is to treat **0.0-0.4 net Sharpe** as the credible technical/momentum range for five daily stocks, to penalize turnover directly, and to publish the result if it is negative. A positive result is more persuasive when it survives 25-50 bp stress costs, not when it wins only at zero cost.

## 3. Case Study: Deep RL Reported 1.30, Then Replication Became the Test

The prominent positive case is the 2020 "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy." It used Dow Jones 30 constituents, daily data from **January 1, 2009 through May 8, 2020**, trained through September 2015, validated in Q4 2015, and evaluated OOS from **January 1, 2016 through May 8, 2020**. [10] It used PPO, A2C, and DDPG, retrained every three months, and selected the highest validation-Sharpe model in a rolling process. [10]

The reported OOS ensemble Sharpe was **1.30**, with PPO **1.10**, A2C **1.12**, and DDPG **0.87**. The paper also reported DJIA Sharpe **0.47** and a minimum-variance benchmark **0.45**. [10] This is a valuable case study because it shows what a carefully engineered research pipeline can report, including walk-forward validation and a stated **0.1% cost on every buy or sell**. [10] It is not a retail execution guarantee: the implementation executed at the close under a market-unaffected assumption, and the universe was 30 stocks rather than five. [10]

The negative case is the 2025 study using **AAPL, AMZN, MSFT, and GOOGL** with an 80/10/10 split. It stated that no transaction-cost assumption was used, did not report Sharpe, found overlapping confidence intervals, and concluded that the agents failed to converge reliably and the results were inconclusive. [26] A 2021 survey explains why these results are difficult to compare: very few studies include transaction cost, slippage, and spread together, and papers use different datasets and preprocessing choices. [20]

The implication is not that RL cannot work. It is that published DRL Sharpe values are selected, design-dependent estimates rather than a stable population parameter. FinRL's 2020 examples reported single-stock PPO Sharpes from **1.10 to 1.49** across SPY, QQQ, GOOGL, AMZN, AAPL, and MSFT, with materially different return, volatility, and drawdown outcomes. [13] These headline numbers are useful replication targets, not reasonable default expectations for a new five-stock system. After replacing paper execution with retail all-in costs, using a genuinely untouched holdout, and correcting for search, **0.0-0.6** is the honest base range.

## 4. The 2026 Retail Cost Stack

| Cost item | Evidence-based number | How to use it |
|---|---:|---|
| Online commission | **0 bps** at Fidelity and Schwab for ordinary online US stocks/ETFs | Do not equate zero commission with zero execution cost; exceptions exist for special transactions and some account types. [29][28] |
| Quoted spread | About **1 bp** for AAPL and MSFT examples; about **2.5 bps** for a sampled portfolio; **3.7 bps** average and just over **4.5 bps** for the full S&P 500 in Nasdaq's 2024 analysis | Use quote spread for a liquid-limit-order scenario, not as the full retail cost. [18] |
| Realized retail round trip | **7-46 bps** across six brokerage accounts and **74,709** all-stock observations in a 2025 Journal of Finance study | Use as the combined execution-cost range for spread, price improvement/routing, timing, and price impact. [22] |
| SEC Section 31 fee | **$20.60 per $1M**, or **0.206 bps**, on covered sales from April 4, 2026 | Apply to sells, not buys. A $20,000 sale costs **$0.412**. [17] |
| FINRA TAF | **$0.000195 per share**, capped at **$9.79 per trade**, effective January 1, 2026 | At 100 shares of a $200 stock, $0.0195 is **0.00975 bps** on $20,000 notional. [15] |

The 7-46 bp retail number is the most important one. The 2025 study measured actual prices relative to the midpoint for same-stock buy/sell observations made about 30 minutes apart; it found retail price improvement relative to the NBBO in 31% of observations and showed that much retail trading occurred off-exchange. [22] It is not a pure slippage estimate. Therefore, adding 1 bp of quoted spread, 46 bp of slippage, and regulatory fees would double count. For a model, use an all-in round-trip parameter such as **7-15 bps base**, **25 bps stress**, and **46 bps severe stress**, explicitly labeling those as engineering scenarios bounded by the study rather than as three independently measured components.

The five-position cost is determined by turnover, not by the number five alone. Define one-way turnover as

`u_t = 0.5 * sum(abs(target_weight_i - current_weight_i))`.

Using a round-trip cost `c` of 7-46 bps, a simple annual cost approximation is `252 * u * c`. The resulting arithmetic is:

| One-way portfolio turnover per day | Gross traded notional per day | Annual cost at 7 bps | Annual cost at 46 bps |
|---:|---:|---:|---:|
| **10%** | 20% including both sides | **1.764%** | **11.592%** |
| **25%** | 50% including both sides | **4.410%** | **28.980%** |
| **50%** | 100% including both sides | **8.820%** | **57.960%** |
| **100%**: replace all five names | 200% including sells and buys | **17.640%** | **115.920%** |

A full five-for-five replacement each day therefore cannot be made economically plausible by saying that commissions are free. A low-turnover portfolio may still be viable, especially with passive limit execution, but the test must record actual turnover and costs each day. The mechanism is convex in the strategy's claimed edge: a 20 bp expected daily signal cannot survive a 46 bp round-trip assumption if the agent trades frequently.

## 5. Statistical Power: A 0.8 Sharpe Needs About Eight IID Years

Lo's standard Sharpe-ratio result gives the asymptotic standard error, under IID-style assumptions, as approximately

`SE(estimated Sharpe) = sqrt((1 + 0.5 * Sharpe^2) / T)`,

where `T` is the number of observations in the period used for the Sharpe. [25] For a true annual Sharpe of **0.8**, the variance factor is `1 + 0.5 * 0.8^2 = 1.32`. Requiring a two-sided 95% estimate to clear zero gives

`T = (1.96^2 * 1.32) / 0.8^2 = 7.92 annual-equivalent observations`.

At 252 trading days per year, that is approximately **1,997 daily observations**, or **7.9 years**. Ignoring the finite-Sharpe correction would give roughly six years, but that is the less conservative calculation. The answer to the question is therefore **about eight IID years at a minimum**, not two or three years.

The result is optimistic for daily trading. Lo shows that serial correlation changes the annualization multiplier and that ignoring autocorrelation can materially overstate annualized Sharpe; the standard square-root rule is valid only under restrictive conditions. [25] Daily bars also contain non-normal tails, regime changes, overlapping feature horizons, and a small effective number of independent market regimes. Five stocks add observations per day but do not turn one common market history into five independent decades.

The Deflated Sharpe Ratio literature addresses the problem that the highest backtest Sharpe is selected from many trials. Bailey and coauthors explicitly adjust for non-normality, sample length, the number of independent trials, and the variance of Sharpe estimates; the maximum observed Sharpe rises as the number of trials increases even when true Sharpe is zero. [9] The Probability of Backtest Overfitting framework uses combinatorial symmetric cross-validation to estimate how often the selected configuration is an overfit. [5]

For this project, report the number of architectures, seeds, reward functions, feature sets, rebalance frequencies, cost settings, and stopping rules tried before the final holdout. Apply DSR or an equivalent multiple-testing correction. A five-year result with a reported Sharpe of 0.8 can be interesting; it is not persuasive evidence of a genuine 0.8 edge unless the holdout was protected and the search burden was small.

## 6. Regulatory Operating Envelope for a Small US Account in 2026

| Constraint | 2026 rule or operational fact | Practical consequence for the system |
|---|---|---|
| PDT and intraday margin | FINRA's new Rule 4210 standards are effective **June 4, 2026**, replace the old day-trade count and **$25,000** PDT minimum, and can be phased in by firms until **October 20, 2027**. [11] | Do not hard-code the old PDT rule as the universal 2026 rule. Ask the broker which implementation applies during the phase-in; house restrictions can still bind. |
| Intraday deficit monitoring | The new framework defines intraday margin level, IML-reducing transactions, and intraday margin deficits; real-time monitoring is permitted but not required, and deficits must be addressed promptly. [11] | A strategy that opens and closes positions repeatedly can trigger broker controls even if a backtest shows adequate end-of-day equity. |
| Settlement | Most US broker-dealer securities moved from T+2 to **T+1 on May 28, 2024**; payment and delivery generally must occur no later than the first business day after trade date. [16] | A cash-account simulator must model settled cash, not just portfolio value. |
| Cash-account violations | Fidelity states that cash-account purchases must be paid in full by settlement, and identifies good-faith, freeride, and cash-liquidation violations; three good-faith violations in 12 months can restrict the account. [19] | Same-day signal turnover can be infeasible in a small cash account if unsettled proceeds are reused. |
| Broker and product scope | Margin eligibility, options, shorting, fractional shares, order types, and risk controls vary by broker and security | Backtest the exact allowed order set and declare whether shorting, leverage, fractional shares, and extended-hours trading are excluded. |

The key 2026 nuance is transition, not a free pass. FINRA's notice says the new standards replace the old PDT framework, but it also gives members an 18-month phase-in. [11] A retail system should therefore maintain two deployment modes: a cash-account mode that enforces T+1 settlement and settled-cash availability, and a margin mode that imports the actual broker's intraday limits. A backtest that assumes unlimited same-day round trips is not a valid description of a small account.

## 7. Benchmark Set That a Quant Audience Will Accept

| Benchmark or control | Why it is required | Minimum reporting |
|---|---|---|
| SPY total-return buy-and-hold | Measures the opportunity cost of not owning the broad US market | Net CAGR, Sharpe, volatility, drawdown, beta, and dates |
| Five-stock equal-weight buy-and-hold | Controls for the exact selected universe and concentration | Same metrics and each stock's contribution |
| Five-stock equal-weight monthly or quarterly rebalance | Tests whether periodic rebalancing, not RL, creates the result | Turnover and cost sensitivity |
| Each stock buy-and-hold | Detects a result driven by one winner | Per-name return, risk, and exposure |
| Moving-average / time-series momentum | Strong simple technical baseline | Same lag, same close-to-next-open or next-close execution |
| Cross-sectional momentum | Tests ranking without a neural network | Same universe and same rebalance schedule |
| Linear or logistic model | Tests whether nonlinearity is doing any work | Same features, costs, and holdout |
| No-skill, cash, and random-action controls | Detects leakage, accidental exposure, and reward bugs | Turnover, average exposure, and distribution of outcomes |
| Cost and turnover stress grid | Determines whether edge survives execution | **0, 10, 25, and 50 bps** one-way or round-trip convention stated explicitly |

The project should pre-register the universe, corporate-action treatment, signal timestamp, execution price, missing-data rules, train/validation/test dates, and cost convention. The 2023 technical-rule study is a useful protocol model because it separates rule selection from the subsequent 36-month OOS test and explicitly varies 0, 10, 25, and 50 bps costs. [21] The DSR and PBO literature then supplies the required correction for the number of alternatives tried. [9][5]

Every result table should include net CAGR, annualized volatility, Sharpe using a stated risk-free series, Sortino, maximum drawdown, worst calendar period, turnover, average gross exposure, beta and alpha versus SPY, number of trades, implementation cost as a percentage of gross PnL, and the number of independent holdout observations. Report both arithmetic and geometric results. A claim that the agent beats SPY only before costs, or only because it takes more beta, is not an alpha claim.

## Synthesis: What the Evidence Actually Supports

| Dimension | SPY buy-and-hold | Classic technical / momentum | Deep RL on five daily stocks |
|---|---|---|---|
| Mechanism | Broad equity risk premium and diversification | Persistent but crowded signal, usually requiring turnover | Flexible policy search over features, actions, and reward design |
| Best published evidence here | Long-run excess Sharpe about **0.39-0.51**; recent one-year **1.45** | US UMD **0.32 net**; optimized tradable **0.13 net** | Selected 2020 Dow ensemble **1.30** with stated paper cost; 2025 four-stock replication inconclusive |
| Execution sensitivity | Low; SPY fee **9.45 bps/year** | Moderate to high; costs can remove the edge | High; model selection and turnover multiply execution risk |
| Credible 2026 net range | **0.35-0.60** over a multi-year sample | **0.0-0.4** | **0.0-0.6** |
| Main failure mode | Underperforming during a weak or sideways market | OOS decay and trading cost drag | Overfit policy, unstable regime response, hidden turnover, and benchmark selection |
| Highest-probability improvement | Keep exposure diversified and cheap | Slow the signal and diversify the opportunity set | Reduce turnover, use cost-aware rewards, and protect the holdout |

The non-obvious tension is that RL's strongest published number is above both passive and classic-factor anchors, but the evidence base is weaker precisely where the number is most impressive. The 2020 ensemble had a defined cost and rolling validation, yet used 30 Dow stocks and close execution. The 2025 four-stock study was closer to the proposed universe, yet had no cost assumption, no Sharpe, and inconclusive results. [10][26]

My probability estimate is therefore **10%-25%**, central **15%**, that a single well-built RL system on five daily-bar US large caps beats both same-universe buy-and-hold and SPY over a genuinely untouched, meaningful multi-year OOS period net of retail costs. It is not a measured literature frequency because no cited paper reports that exact experiment as a probability distribution. It is a decision prior derived from the low net Sharpe of tradable momentum, the weak OOS technical-rule record, the selected nature of DRL headlines, the 7-46 bp retail cost range, and the DSR multiple-testing problem. [23][21][22][9]

To maximize that probability, do five things. First, use a slower policy or explicit turnover penalty so the agent cannot manufacture signal from frictionless daily churn. Second, train with a cost model sampled across the 7-46 bp empirical range and report the full stress grid. Third, make the final holdout untouchable and count every experiment that influenced the chosen architecture. Fourth, prefer simple, stable features and compare against moving average, momentum, linear, and buy-and-hold controls before adding model complexity. Fifth, enlarge the research opportunity set if the objective is a genuine ML edge: five correlated names provide too little cross-sectional diversity for a high-capacity learner.

The honest success criterion is not "the backtest Sharpe exceeded 1." It is: the strategy remains positive after 25-46 bp stress costs, beats same-universe buy-and-hold without simply taking more beta, survives a multi-year untouched holdout, retains performance across market regimes, and remains statistically credible after accounting for the number of trials. On the evidence available in 2026, that outcome is possible, but it is the exception rather than the base case.

## References

1. *New “T+1” Settlement Cycle – What Investors Need To Know ...*. https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/new-t1-settlement-cycle-what-investors-need-know-investor-bulletin
2. *Fact Sheet:State Street® SPDR® S&P 500® ETF Trust, Jun2026*. https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spy.pdf
3. *FinRL-X: An AI-Native Modular Infrastructure for Quantitative ...*. https://ai4finance.org/FinRL-Paper.pdf
4. *Trading Activity Fee - FINRA.org*. https://www.finra.org/rules-guidance/guidance/trading-activity-fee
5. *The Probability of Backtest Overfitting by David H. Bailey, Jonathan Borwein, Marcos Lopez de Prado, Qiji Jim Zhu :: SSRN*. http://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
6. *The Statistics of Sharpe Ratios - CFA Institute*. https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios
7. *Constant Leverage Covering Strategy for Equity Momentum ...*. https://www.mdpi.com/2227-7072/12/2/55
8. *State Street SPDR S&P 500 ETF (SPY) Sharpe Ratio: 1.37*. https://portfolioslab.com/symbol/SPY/sharpe-ratio
9. *THE DEFLATED SHARPE RATIO: CORRECTING FOR ... - David H. Bailey*. https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
10. *http://openfin.engineering.columbia.edu/sites/openfin.engineering.columbia.edu/files/content/publications/ensemble.pdf*. http://openfin.engineering.columbia.edu/sites/openfin.engineering.columbia.edu/files/content/publications/ensemble.pdf
11. *Regulatory Notice 26-10 | FINRA.org*. http://finra.org/rules-guidance/notices/26-10
12. *Time series momentum - docs.lhpedersen.com*. http://docs.lhpedersen.com/TimeSeriesMomentum.pdf
13. *FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance*. https://arxiv.org/html/2011.09607v1
14. *Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy*. https://arxiv.org/pdf/2511.12120
15. *FINRA Fee Adjustment Schedule | FINRA.org*. https://www.finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule
16. *Shortening the Securities Transaction Settlement Cycle*. https://www.sec.gov/files/risk-alert-tplus1-032724.pdf
17. *SEC.gov | Section 31 Transaction Fee Rate Advisory for Fiscal Year 2026*. https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2
18. *Sampling the S&P 500 to Minimize Spreads | Nasdaq*. https://www.nasdaq.com/articles/sampling-sp-500-minimize-spreads
19. *Avoiding Cash Account Trading Violations - Fidelity*. http://fidelity.com/learning-center/trading-investing/trading/avoiding-cash-trading-violations
20. *Deep Reinforcement Learning for Trading—A Critical Survey*. https://www.mdpi.com/2306-5729/6/11/119
21. *The predictive ability of technical trading rules: an empirical analysis of developed and emerging equity markets | Financial Markets and Portfolio Management | Springer Nature Link*. https://link.springer.com/article/10.1007/s11408-023-00433-2
22. *The “Actual Retail Price” of Equity Trades - SCHWARZ - 2025 - The Journal of Finance - Wiley Online Library*. https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13467
23. *Trading Costs of Asset Pricing Anomalies*. https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini,%20Israel%20and%20Moskowitz.pdf
24. *New Home Page*. https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html
25. *The Statistics Of Sharpe Ratios*. https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf
26. *Reinforcement Learning for Stock Transactions*. https://arxiv.org/html/2505.16099v2
27. *Understanding the New Intraday Margin Requirements - FINRA.org*. https://www.finra.org/investors/insights/intraday-margin-requirements
28. *Pricing: $0 commissions on online stock, ETF, and options trades | Charles Schwab*. https://www.schwab.com/node/14936
29. *Trading Commissions and Margin Rates | Fidelity*. https://www.fidelity.com/trading/commissions-margin-rates
