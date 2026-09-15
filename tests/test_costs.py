"""Costs must actually cost something, and the cap must actually cap."""
import numpy as np
from rltrader.eval import costs as C


def test_free_is_free():
    dw = np.array([[0.5, -0.5]])
    r = C.trade_costs(dw, np.array([1e6]), np.full((1, 2), 1e9), np.full((1, 2), 0.02), C.FREE)
    assert r["cost_fraction"][0] == 0.0


def test_half_spread_is_charged_on_both_sides():
    dw = np.array([[0.5, -0.5]])            # 100% of book turned over, two-sided
    r = C.trade_costs(dw, np.array([1e6]), np.full((1, 2), 1e12), np.full((1, 2), 0.0),
                      C.REALISTIC)
    assert abs(r["cost_fraction"][0] - 1e-4) < 1e-9     # 1.0 notional x 1 bp


def test_adv_cap_binds_and_is_reported():
    dw = np.array([[1.0, 0.0]])
    adv = np.array([[1e6, 1e6]])            # 1% cap = 10k, trade wants 1m
    r = C.trade_costs(dw, np.array([1e6]), adv, np.full((1, 2), 0.02), C.REALISTIC)
    assert r["n_trades_capped"] == 1
    assert abs(r["unfilled_notional"] - (1e6 - 1e4)) < 1.0


def test_impact_is_sublinear():
    """Square-root law: ten trades of size x cost less than one trade of size 10x."""
    adv, vol, value = np.full((1, 1), 1e9), np.full((1, 1), 0.02), np.array([1e9])
    big = C.trade_costs(np.array([[0.10]]), value, adv, vol, C.REALISTIC)["impact_cost"][0]
    small = C.trade_costs(np.array([[0.01]]), value, adv, vol, C.REALISTIC)["impact_cost"][0]
    assert big < 10 * small


def test_pessimistic_costs_more_than_realistic():
    dw, adv, vol, val = np.array([[0.3]]), np.full((1, 1), 1e9), np.full((1, 1), 0.02), np.array([1e7])
    a = C.trade_costs(dw, val, adv, vol, C.REALISTIC)["cost_fraction"][0]
    b = C.trade_costs(dw, val, adv, vol, C.PESSIMISTIC)["cost_fraction"][0]
    assert b > a


def test_three_levels_are_declared():
    assert [m.name for m in C.LEVELS] == ["free", "realistic", "pessimistic"]
    assert C.FREE.commission_bps == 0 and C.REALISTIC.adv_cap == 0.01
