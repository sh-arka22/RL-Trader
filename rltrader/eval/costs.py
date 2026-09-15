"""Transaction costs at three levels, because the level changes the ranking.

RESEARCH.md finding (arXiv:2603.29086): the ORDER of RL algorithms flips when only the
cost model changes — flat 10 bps vs Almgren-Chriss on the same NASDAQ-100 data. So a
single cost assumption does not produce a weaker result, it produces an arbitrary one.
Every number this project reports is therefore reported at all three levels, and a result
that survives only at FREE is reported as a failure.

FILL CONVENTION
---------------
A decision made from the close of session t is filled at the OPEN of session t+1. Filling
at the same close the decision was made from is the most common silent look-ahead in
retail backtests (TensorTrade does exactly this: observe close 100.0, fill 100.0).

COMPONENTS
----------
commission     0 bps. US retail equity commissions are zero at the major brokers (2026).
half_spread    paid on every unit of notional traded, both directions.
impact         Almgren-Chriss square-root law: eta * sigma * sqrt(participation), where
               participation = traded notional / ADV. Sub-linear, so it punishes large
               trades far more than many small ones.
adv_cap        a trade may not exceed this fraction of the day's dollar volume. Trades are
               CLIPPED, not rejected, and the clipping is reported — an uncapped backtest
               can "buy" more of a stock than traded that day and never notice.
"""
from __future__ import annotations
import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=True)
class CostModel:
    name: str
    commission_bps: float
    half_spread_bps: float
    impact_eta: float          # 0 disables impact
    adv_cap: float             # fraction of ADV; 1.0 effectively disables the cap
    note: str

    def __str__(self) -> str:
        return (f"{self.name}(commission={self.commission_bps}bp, "
                f"half_spread={self.half_spread_bps}bp, eta={self.impact_eta}, "
                f"adv_cap={self.adv_cap:.1%})")


# The three levels every result is reported at.
FREE = CostModel("free", 0.0, 0.0, 0.0, 1.0,
                 "frictionless. NOT a result — the control that shows how much of a "
                 "strategy is an artefact of ignoring costs.")
REALISTIC = CostModel("realistic", 0.0, 1.0, 0.1, 0.01,
                      "0 commission, 1 bp half-spread, sqrt impact, 1% ADV cap. "
                      "Large-cap US equities, daily bars, retail broker, 2026.")
PESSIMISTIC = CostModel("pessimistic", 0.0, 2.0, 0.3, 0.005,
                        "2 bp half-spread, 3x impact, 0.5% ADV cap. Stressed liquidity.")

LEVELS = (FREE, REALISTIC, PESSIMISTIC)
BPS = 1e-4


def trade_costs(dw: np.ndarray, portfolio_value: np.ndarray, adv: np.ndarray,
                vol: np.ndarray, model: CostModel) -> dict:
    """Cost of a weight change, as a fraction of portfolio value, per period.

    dw    (T, n) intended weight change per asset
    value (T,)   portfolio value before the trade
    adv   (T, n) dollar average daily volume per asset
    vol   (T, n) daily return volatility per asset (for the impact term)
    """
    dw = np.asarray(dw, dtype="float64")
    value = np.asarray(portfolio_value, dtype="float64").reshape(-1, 1)
    adv = np.asarray(adv, dtype="float64")
    vol = np.asarray(vol, dtype="float64")

    notional = np.abs(dw) * value
    cap = model.adv_cap * adv
    clipped = np.minimum(notional, cap)
    n_capped = int((notional > cap + 1e-9).sum())
    unfilled = float((notional - clipped).sum())

    spread = clipped * (model.half_spread_bps + model.commission_bps) * BPS

    if model.impact_eta > 0:
        participation = np.divide(clipped, adv, out=np.zeros_like(clipped), where=adv > 0)
        impact_frac = model.impact_eta * vol * np.sqrt(participation)
        impact = clipped * impact_frac
    else:
        impact = np.zeros_like(clipped)

    total = (spread + impact).sum(axis=1)
    return {"cost_fraction": np.divide(total, value.ravel(),
                                       out=np.zeros_like(total), where=value.ravel() > 0),
            "spread_cost": spread.sum(axis=1), "impact_cost": impact.sum(axis=1),
            "notional_traded": clipped.sum(axis=1), "n_trades_capped": n_capped,
            "unfilled_notional": unfilled}


def apply(gross_returns: np.ndarray, dw: np.ndarray, adv: np.ndarray, vol: np.ndarray,
          model: CostModel, start_value: float = 1.0) -> dict:
    """Net return stream after costs, with the cost path kept for auditing."""
    gross = np.asarray(gross_returns, dtype="float64")
    value, net, detail = start_value, [], []
    values = []
    for t in range(len(gross)):
        values.append(value)
        c = trade_costs(dw[t:t + 1], np.array([value]), adv[t:t + 1], vol[t:t + 1], model)
        cf = float(c["cost_fraction"][0])
        r = float(gross[t]) - cf
        detail.append(cf)
        net.append(r)
        value *= (1.0 + r)
    return {"net_returns": np.array(net), "cost_path": np.array(detail),
            "final_value": value, "values": np.array(values),
            "total_cost_bps": float(np.sum(detail) * 1e4), "model": str(model)}
