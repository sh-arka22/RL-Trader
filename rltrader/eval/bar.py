"""THE BAR. The number the agent has to beat, pinned to an exact convention.

Measured 2026-09-14 on the S2 store, window 2015-01-02 -> 2026-09-11 (2,940 sessions),
rf = 0, 252 trading days, simple returns on the dividend-adjusted (total-return) series.

WHY THIS FILE EXISTS
--------------------
The research phase recorded the equal-weight-5 passive Sharpe as **1.07**. That number is
correct — but for a basket this project no longer trades. It is AAPL/MSFT/JNJ/JPM/XOM,
daily-rebalanced, which reproduces at **1.073**. The universe was later changed to
NVDA/TSLA/AAPL/META/XOM, and the bar moved with it. Nobody updated the bar.

So the hurdle in PLAN.md was set against a basket that was replaced: the real passive
control is **1.201**, not 1.07 — about 0.13 Sharpe higher. An agent scoring 1.10 would
have looked like a win against the stale number and is in fact a loss to doing nothing.

The second lesson is the convention spread. On the CURRENT universe the same window gives
0.885 to 1.238 depending only on choices nobody usually states:

    total-return, simple returns, daily-rebalanced          1.234
    total-return, simple returns, true buy-and-hold         1.201
    total-return, LOG returns, daily-rebalanced             0.926
    price-only (no dividends), simple, daily-rebalanced     1.193
    price-only (no dividends), simple, buy-and-hold         1.188
    price-only (no dividends), LOG returns, daily           0.885
    mean of the five INDIVIDUAL buy-and-hold Sharpes        0.833   <- a different quantity

    (for reference, the retired basket AAPL/MSFT/JNJ/JPM/XOM:
     daily-rebalanced 1.073, true buy-and-hold 0.989)

The spread between conventions is **0.35 Sharpe**. That is larger than almost every
improvement claimed in the RL-trading literature this project surveyed. An unstated
convention is therefore not a detail; it is enough to manufacture a result.

The pipeline is not the source of the disagreement: on the same window it reproduces
yfinance's own `auto_adjust=True` series to three decimals (SPY 0.823 vs 0.823,
EW-5 1.201 vs 1.201, 1.234 vs 1.234). That is also an end-to-end check of the S2
dividend reconstruction at the portfolio level.

CONSEQUENCE: the hurdle is HIGHER than PLAN.md assumed, for two independent reasons — a
retired basket and an unstated convention. Planning against 1.07 would have declared
victory at a level the passive control already beats.

RULE ADOPTED: the bar is recomputed from the store by tests/test_baselines.py, never
copied from a document. A metric that lives only in prose goes stale silently.
"""
from __future__ import annotations
import datetime as dt

WINDOW = (dt.date(2015, 1, 1), dt.date(2026, 9, 11))
CONVENTION = {"returns": "simple", "series": "total_return_dividend_adjusted",
              "rf": 0.0, "periods_per_year": 252, "sessions": 2940}

# Primary bar: what a passive investor actually gets, with no rebalancing trades.
PASSIVE_BAR = {
    "SPY_buy_and_hold": 0.823,
    "EW5_buy_and_hold": 1.201,          # <- the primary control for the 5-name universe
    "EW5_daily_rebalanced": 1.234,      # not buy-and-hold: it harvests a rebalancing premium
    "EW5_monthly_rebalanced": 1.238,
}

SINGLE_NAME = {"NVDA": 1.328, "TSLA": 0.768, "AAPL": 0.925, "META": 0.671, "XOM": 0.472}

# Any agent Sharpe must be reported against EW5_buy_and_hold AND SPY_buy_and_hold, at all
# three cost levels. RESEARCH.md: losing to buy-and-hold is a likely, acceptable, reportable
# outcome. It is not an outcome to be hidden by choosing a friendlier convention.
TOLERANCE = 0.005
