"""The harness is the apparatus. If it cannot show a leak, it cannot certify anything."""
import datetime as dt
import numpy as np
import pandas as pd
import pytest

from rltrader.eval import costs as C
from rltrader.eval.harness import evaluate, run_strategy, walk_forward_folds
from rltrader.eval.strategies import BuyAndHoldFirst, EqualWeight, Momentum, PlantedOracle
from rltrader.eval.trial_log import TrialLog


@pytest.fixture
def panel():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2018-01-01", periods=400)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(3e-4, 0.012, (400, 3)), axis=0),
                      index=idx, columns=["A", "B", "C"])
    return px, px.shift(1).bfill()


@pytest.fixture
def liq(panel):
    px, _ = panel
    adv = pd.DataFrame(1e10, index=px.index, columns=px.columns)
    vol = pd.DataFrame(0.02, index=px.index, columns=px.columns)
    return adv, vol


def test_planted_oracle_scores_absurdly(panel, liq):
    """If a perfect look-ahead oracle does NOT stand out, the harness would hide a real
    leak too. arXiv:2608.27734: DSR gave 1.00 to exactly this — statistics will not help."""
    close, opn = panel
    r = evaluate(PlantedOracle(opn), close, opn, *liq)
    assert r["by_cost"]["free"]["sharpe"] > 5.0, r["by_cost"]["free"]["sharpe"]


def test_strategy_cannot_see_the_future(panel, liq):
    """Structural, not advisory: the object handed over does not contain future rows."""
    close, opn = panel
    seen = []

    class Peeker:
        name = "peeker"

        def fit(self, train): pass

        def weights(self, history, held):
            seen.append((history.index[-1], len(history)))
            return np.zeros(3)

    run_strategy(Peeker(), close, opn, *liq)
    for last_date, n in seen:
        assert close.index[n - 1] == last_date, "history extended past the decision bar"


def test_buy_and_hold_does_not_trade_but_rebalancing_does(panel, liq):
    """Turnover must be measured against the drifted book. Comparing target to target
    makes rebalancing free and flatters every high-turnover strategy."""
    close, opn = panel
    bh = evaluate(BuyAndHoldFirst(3), close, opn, *liq)
    ew = evaluate(EqualWeight(3), close, opn, *liq)
    n = bh["n_periods"]
    bh_to = bh["by_cost"]["realistic"]["turnover"]
    # Buy-and-hold is not zero-turnover: it buys the basket once. That single entry trade
    # is a real cost, spread over the whole run, so mean turnover ~ 0.5/n rather than 0.
    assert bh_to == pytest.approx(0.5 / n, rel=0.05), (bh_to, 0.5 / n)
    # The real distinction is not the ratio but the SHAPE: buy-and-hold's total traded
    # notional is a constant (one entry) however long the run; rebalancing's grows with it.
    ew_to = ew["by_cost"]["realistic"]["turnover"]
    assert bh_to * n == pytest.approx(0.5, rel=0.05), "entry trade should be the only trade"
    assert ew_to * n > 1.5, "daily rebalancing must trade far more than a single entry"
    assert ew_to > bh_to * 3


def test_costs_are_monotonic_across_levels(panel, liq):
    close, opn = panel
    r = evaluate(Momentum(3, lookback=20, top_k=1), close, opn, *liq)
    f, re, pe = (r["by_cost"][k]["sharpe"] for k in ("free", "realistic", "pessimistic"))
    assert f >= re >= pe, (f, re, pe)


def test_buy_and_hold_pays_only_its_entry_trade(panel, liq):
    """Its cost is one purchase, not an ongoing drag — the gap between FREE and
    PESSIMISTIC must stay tiny but must NOT be exactly zero."""
    close, opn = panel
    r = evaluate(BuyAndHoldFirst(3), close, opn, *liq)
    gap = abs(r["by_cost"]["free"]["sharpe"] - r["by_cost"]["pessimistic"]["sharpe"])
    assert 0 < gap < 0.01, gap


def test_weights_are_validated(panel, liq):
    close, opn = panel

    class Bad:
        name = "bad"

        def fit(self, train): pass

        def weights(self, history, held): return np.array([0.9, 0.9, 0.9])

    with pytest.raises(ValueError, match="long-only and sum"):
        run_strategy(Bad(), close, opn, *liq)


def test_folds_do_not_overlap(panel):
    close, _ = panel
    folds = walk_forward_folds(close.index, train_years=0.5, test_years=0.25, step_years=0.25)
    for a, b in zip(folds, folds[1:]):
        assert a.test_start <= b.test_start
        assert a.train_end <= a.test_start, "train window bleeds into test"


def test_every_evaluation_is_logged(tmp_path, panel, liq):
    """The DSR denominator must be recorded as it happens, not recalled afterwards."""
    close, opn = panel
    log = TrialLog(tmp_path / "t.jsonl")
    for k in (1, 2, 3):
        evaluate(Momentum(3, lookback=10, top_k=1), close, opn, *liq, log=log, note=f"run{k}")
    assert log.n_trials("evaluation") == 3
    assert log.verify()[0]
