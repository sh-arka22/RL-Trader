import datetime as dt
import numpy as np
import pandas as pd
from rltrader.data import reconcile as rec


def _frame(vals):
    days = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(len(vals))]
    return pd.DataFrame({"date": days, "close": vals})


def test_identical_sources_are_flagged_as_a_mirror():
    v = list(np.linspace(100, 110, 50))
    m = rec.compare(_frame(v), _frame(v), "close", "a", "b")
    s = rec.summarise(m, "T", "close")
    assert s["median_bps"] == 0 and s["frac_bit_identical"] == 1.0
    assert rec.independence_evidence([s])["verdict"] == "MIRROR_SUSPECTED"


def test_rounding_differences_read_as_independent():
    rng = np.random.default_rng(0)
    a = np.linspace(100, 110, 200)
    b = np.round(a + rng.normal(0, 0.0005, 200), 4)
    s = rec.summarise(rec.compare(_frame(list(a)), _frame(list(b)), "close", "a", "b"), "T", "close")
    assert s["frac_bit_identical"] < 0.9
    assert rec.independence_evidence([s])["verdict"] == "INDEPENDENT_LIKELY"


def test_divergence_is_measured_in_bps():
    s = rec.summarise(rec.compare(_frame([100.0]), _frame([100.01]), "close", "a", "b"), "T", "close")
    assert abs(s["median_bps"] - 1.0) < 0.01      # 1 cent on $100 == 1 bp
