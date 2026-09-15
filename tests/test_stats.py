"""Known-answer tests for ``rltrader.eval.stats``.

The standard here is the one ``docs/reports/S2_DATA.md`` section 8 arrived at the hard
way: *a check that cannot fail is worthless*.  A test that runs a function and asserts the
answer is a float proves nothing.  So every test below compares against something computed
elsewhere:

* **published numbers** — Lo (2002) Tables 1 and 2, the Bailey & Lopez de Prado (2014)
  worked example on pp. 9-10, the Benjamini & Hochberg (1995) example, and the
  combination counts in Bailey et al. (2017);
* **closed-form algebra** — a deterministic series whose Sharpe is exactly
  ``sqrt(252)/3``, the identity that the HLN-corrected DM statistic *is* the one-sample
  t statistic at ``h = 1``, the Politis-Romano bootstrap variance in its two equivalent
  forms;
* **independent implementations** — ``scipy`` and ``statsmodels``, dev-only dependencies
  the module itself never imports, so this file checks ``stats.py`` against foreign code
  rather than against itself;
* **simulation against the estimator's own definition** — coverage rates, false discovery
  rates, and the sampling distribution of the Sharpe estimator.

Each group also contains at least one test whose job is to *fail if the correction were
dropped*: the i.i.d. Sharpe standard error under-states uncertainty on autocorrelated
returns, the uncorrected DM statistic over-rejects in small samples, the i.i.d. bootstrap
under-covers on AR(1) data, and uncorrected multiple testing manufactures discoveries.

And one test is an honest negative: ``test_dsr_does_not_catch_lookahead`` plants a
look-ahead oracle, shows the Deflated Sharpe Ratio certifies it at 1.00, and asserts that
outcome — reproducing the arXiv:2608.27734 finding recorded in ``docs/RESEARCH.md``
section 4.  If someone ever "fixes" the DSR into a leak detector, this test tells them
they have changed the statistic, not improved it.

Reference values that could NOT be sourced are marked ``NO PUBLISHED REFERENCE`` in the
docstring of the test that needed them, and are checked against simulation instead.
"""
import math

import numpy as np
import pytest
from scipy import stats as sps
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from rltrader.eval import stats as S

SEED = 20260915


def ar1(n, phi, sigma=1.0, seed=0, mu=0.0):
    """An AR(1) series, used wherever serial dependence has to be real, not assumed."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1.0 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x + mu


def exact_sharpe_series(target_sr, n):
    """A series whose maximum-likelihood Sharpe is *exactly* ``target_sr``.

    Values ``(-sqrt(3), 0, 0, 0, 0, +sqrt(3))`` repeated have mean 0, ML variance 1,
    skewness exactly 0 and non-excess kurtosis exactly 3 — the Gaussian moments, hit
    exactly rather than approximately.  Shifting by ``target_sr`` therefore gives a sample
    with ML mean ``target_sr`` and ML standard deviation 1.
    """
    if n % 6:
        raise ValueError("n must be a multiple of 6")
    base = np.tile(np.array([-math.sqrt(3.0), 0.0, 0.0, 0.0, 0.0, math.sqrt(3.0)]), n // 6)
    return base + target_sr


# ======================================================================================
# 0. the stdlib special functions, against scipy
# ======================================================================================
def test_normal_cdf_and_inverse_match_scipy():
    """``stats.py`` ships its own erf-based CDF and AS 241 inverse so the harness needs no
    scipy at runtime.  They have to be the same functions scipy has."""
    x = np.linspace(-8.0, 8.0, 401)
    assert np.max(np.abs(S._norm_cdf(x) - sps.norm.cdf(x))) < 1e-15

    p = np.concatenate([[1e-12, 1e-9, 1e-6], np.linspace(0.001, 0.999, 499), [1 - 1e-9]])
    assert np.max(np.abs(S._norm_ppf(p) - sps.norm.ppf(p))) < 1e-12
    assert S._norm_ppf_scalar(0.0) == -math.inf and S._norm_ppf_scalar(1.0) == math.inf
    with pytest.raises(ValueError):
        S._norm_ppf_scalar(1.5)


def test_student_t_survival_matches_scipy():
    """The DM p-value is a Student-t tail; a wrong incomplete beta would silently shift
    every p-value the harness reports."""
    for df in (1, 2, 3, 7, 30, 199, 2941):
        t = np.linspace(-12.0, 12.0, 121)
        mine = np.array([S._t_sf(float(v), df) for v in t])
        assert np.max(np.abs(mine - sps.t.sf(t, df))) < 1e-12


# ======================================================================================
# 1. Sharpe ratio and the Lo (2002) standard error
# ======================================================================================
def test_sharpe_of_a_deterministic_series_is_the_analytic_value():
    """126 returns of +2% alternating with 126 of -1%: mean 0.005, ML sd 0.015, so the
    daily Sharpe is exactly 1/3 and the annualised Sharpe exactly ``sqrt(252)/3``."""
    r = np.array([0.02, -0.01] * 126)
    assert S.sharpe(r, periods=1) == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert S.sharpe(r) == pytest.approx(math.sqrt(252.0) / 3.0, rel=1e-12)
    # the two variance conventions differ by exactly sqrt(T/(T-1)) and nothing else
    assert S.sharpe(r, ddof=1) == pytest.approx(S.sharpe(r) * math.sqrt(251.0 / 252.0),
                                                rel=1e-12)


def test_sharpe_scales_with_the_risk_free_rate_and_the_horizon():
    """Sharpe is (mean - rf)/sd: subtracting rf from every return must move it by exactly
    rf/sd, and annualisation is exactly sqrt(periods) for i.i.d. returns."""
    r = np.array([0.02, -0.01] * 126)
    sd = float(np.std(r))
    assert S.sharpe(r, rf=0.002, periods=1) == pytest.approx((0.005 - 0.002) / sd, rel=1e-12)
    assert S.sharpe(r, periods=252) == pytest.approx(S.sharpe(r, periods=1) * math.sqrt(252))


def test_sharpe_refuses_a_constant_series_instead_of_returning_infinity():
    """A zero-variance series has an infinite Sharpe.  Reporting ``inf`` would let a
    degenerate backtest clear any threshold; the function raises instead."""
    with pytest.raises(ValueError, match="zero variance"):
        S.sharpe(np.full(50, 0.001))
    with pytest.raises(ValueError):
        S.sharpe([0.01])
    with pytest.raises(ValueError, match="non-finite"):
        S.sharpe([0.01, np.nan, 0.02])


def test_iid_standard_error_reproduces_lo_2002_table_1():
    """Lo (2002), *Financial Analysts Journal* 58(4), Table 1 (p. 39), "Asymptotic Standard
    Errors of Sharpe Ratio Estimators": published to three decimals for every combination
    of ``SR`` in 0.50..3.00 and ``T`` in {12, 24, 36, 48, 60, 125, 250, 500}.

    The columns testable with an exactly-constructed sample are the multiples of six.
    Lo's text restates two of these cells verbatim: "in a sample of 60 observations, the
    standard error of the Sharpe ratio estimator is 0.188 when the true Sharpe ratio is
    1.50 but is 0.303 when the true Sharpe ratio is 3.00" (p. 38).
    """
    table = {  # SR -> {T: published SE}
        0.50: {12: 0.306, 24: 0.217, 36: 0.177, 48: 0.153, 60: 0.137},
        1.00: {12: 0.354, 24: 0.250, 36: 0.204, 48: 0.177, 60: 0.158},
        1.50: {12: 0.421, 24: 0.298, 36: 0.243, 48: 0.210, 60: 0.188},
        2.00: {12: 0.500, 24: 0.354, 36: 0.289, 48: 0.250, 60: 0.224},
        2.50: {12: 0.586, 24: 0.415, 36: 0.339, 48: 0.293, 60: 0.262},
        3.00: {12: 0.677, 24: 0.479, 36: 0.391, 48: 0.339, 60: 0.303},
    }
    for sr, row in table.items():
        for n, published in row.items():
            x = exact_sharpe_series(sr, n)
            assert S.sharpe(x, periods=1) == pytest.approx(sr, rel=1e-12)
            se = S.sharpe_standard_error(x, periods=1, iid=True)
            assert round(se, 3) == published, (sr, n, se, published)


def test_hac_standard_error_reduces_to_equation_9_on_gaussian_moments():
    """Lo (2002) eqs. (A3)/(A8): with ``dg/dtheta = (1/sigma, -(mu-Rf)/(2 sigma^3))`` and
    moment conditions ``(r-mu, (r-mu)^2-sigma^2)``, the GMM variance collapses to eq. (9)'s
    ``(1 + SR^2/2)/T`` **exactly** when the sample skewness is 0 and the sample non-excess
    kurtosis is 3.  ``exact_sharpe_series`` hits those moments exactly, so this is an
    algebraic identity, not an approximation."""
    x = exact_sharpe_series(1.25, 600)
    assert sps.skew(x, bias=True) == pytest.approx(0.0, abs=1e-12)
    assert sps.kurtosis(x, fisher=False, bias=True) == pytest.approx(3.0, abs=1e-12)
    hac = S.sharpe_standard_error(x, periods=1, lag=0)
    eq9 = S.sharpe_standard_error(x, periods=1, iid=True)
    assert hac == pytest.approx(eq9, rel=1e-12)
    assert eq9 == pytest.approx(math.sqrt((1 + 0.5 * 1.25 ** 2) / 600), rel=1e-12)


def test_standard_error_matches_the_sampling_distribution_of_the_estimator():
    """The definitional check: over 4,000 independent Gaussian samples the spread of the
    Sharpe estimates must match the reported standard error.  This is what a standard
    error *means*, and it is checked by simulation because no published table covers it."""
    rng = np.random.default_rng(SEED)
    n, mu, sd = 250, 0.0008, 0.012
    draws = rng.normal(mu, sd, size=(4000, n))
    srs = np.array([S.sharpe(d, periods=1) for d in draws])
    empirical = float(srs.std(ddof=1))
    reported = float(np.mean([S.sharpe_standard_error(d, periods=1) for d in draws]))
    assert reported == pytest.approx(empirical, rel=0.05)


def test_iid_standard_error_understates_uncertainty_on_autocorrelated_returns():
    """The reason Lo (2002) exists, and the test that fails if the HAC correction is
    dropped.

    Measured over 1,500 AR(1) paths (``phi = 0.6``, ``T = 500``), the true spread of the
    Sharpe estimator is about **twice** what the i.i.d. formula reports: eq. (9) recovers
    ~50% of it, the Newey-West HAC estimator ~83% at the default bandwidth and ~92% at
    ``lag = 20``.  A harness using eq. (9) on a position-holding strategy would therefore
    report a t-statistic roughly twice as large as the data supports.

    The residual 8-17% shortfall of the HAC estimator is an honest limitation, not a bug:
    Bartlett-kernel HAC variances are downward-biased in finite samples.  The test pins
    the direction and the rough size of both effects rather than pretending the correction
    is exact.
    """
    rng = np.random.default_rng(SEED + 1)
    n, phi, reps = 500, 0.6, 1500
    srs, se_iid, se_hac, se_hac20 = [], [], [], []
    for _ in range(reps):
        seed = int(rng.integers(0, 2 ** 31))
        x = 0.0008 + 0.012 * ar1(n, phi, seed=seed)
        srs.append(S.sharpe(x, periods=1))
        se_iid.append(S.sharpe_standard_error(x, periods=1, iid=True))
        se_hac.append(S.sharpe_standard_error(x, periods=1))
        se_hac20.append(S.sharpe_standard_error(x, periods=1, lag=20))
    truth = float(np.std(srs, ddof=1))
    mean_iid, mean_hac, mean_hac20 = (float(np.mean(se_iid)), float(np.mean(se_hac)),
                                      float(np.mean(se_hac20)))
    assert mean_iid < 0.60 * truth, (mean_iid, truth)
    assert 0.75 * truth < mean_hac < 1.10 * truth, (mean_hac, truth)
    assert mean_hac20 > mean_hac  # a longer bandwidth captures more of the dependence
    assert 0.85 * truth < mean_hac20 < 1.10 * truth, (mean_hac20, truth)


def test_lo_annualisation_factor_reproduces_lo_2002_table_2():
    """Lo (2002) Table 2 (p. 41), "Scale Factors for Time-Aggregated Sharpe Ratios When
    Returns Follow an AR(1) Process".  Rows are the first-order autocorrelation in per
    cent, columns the aggregation value ``q``; every cell is ``eta(q)`` to two decimals.

    Lo's own sentence for the q=12 column (p. 41): "-20 percent is 4.17 times the monthly
    Sharpe ratio, whereas the scale factor is 3.46 in the IID case and 2.88 when the
    monthly first-order autocorrelation is 20 percent."
    """
    qs = [2, 3, 4, 6, 12, 24, 36, 48, 125, 250]
    table = {
        90: [1.03, 1.05, 1.07, 1.10, 1.21, 1.41, 1.60, 1.77, 2.67, 3.70],
        50: [1.15, 1.28, 1.39, 1.60, 2.12, 2.91, 3.53, 4.06, 6.49, 9.15],
        20: [1.29, 1.52, 1.73, 2.07, 2.88, 4.04, 4.93, 5.68, 9.14, 12.92],
        0: [1.41, 1.73, 2.00, 2.45, 3.46, 4.90, 6.00, 6.93, 11.18, 15.81],
        -20: [1.58, 1.99, 2.33, 2.90, 4.17, 5.95, 7.31, 8.45, 13.67, 19.35],
        -50: [2.00, 2.45, 3.02, 3.84, 5.69, 8.26, 10.21, 11.84, 19.26, 27.31],
    }
    for pct, row in table.items():
        rho = pct / 100.0
        for q, published in zip(qs, row):
            eta = S.lo_annualisation_factor(rho=[rho ** k for k in range(1, q)], q=q)
            assert round(eta, 2) == published, (pct, q, eta, published)
    # the IID row is exactly sqrt(q) -- the textbook factor is the rho = 0 special case
    for q in qs:
        assert S.lo_annualisation_factor(rho=np.zeros(q - 1), q=q) == pytest.approx(
            math.sqrt(q), rel=1e-12)


def test_lo_annualisation_factor_matches_the_closed_form_and_direct_aggregation():
    """Two independent confirmations of eq. (20).

    1.  Lo's eq. (22) gives a closed form for AR(1):
        ``eta(q) = sqrt(q) [1 + (2 rho/(1-rho))(1 - (1-rho^q)/(q(1-rho)))]^(-1/2)``.
    2.  The definition itself: ``eta(q) = q*mu / sd(sum of q consecutive returns)``.
        Simulated on a long AR(1) path, which is the only check that does not reuse the
        same summation the implementation performs.
    """
    for rho in (-0.5, -0.2, 0.2, 0.5, 0.8):
        for q in (2, 6, 12, 60):
            closed = math.sqrt(q) * (1 + (2 * rho / (1 - rho))
                                     * (1 - (1 - rho ** q) / (q * (1 - rho)))) ** -0.5
            got = S.lo_annualisation_factor(rho=[rho ** k for k in range(1, q)], q=q)
            assert got == pytest.approx(closed, rel=1e-10)

    phi, q, n = 0.5, 12, 600_000
    x = 0.001 + 0.01 * ar1(n, phi, seed=4)
    blocks = x[: (n // q) * q].reshape(-1, q).sum(axis=1)
    empirical = blocks.mean() / blocks.std()
    eta = S.lo_annualisation_factor(rho=[phi ** k for k in range(1, q)], q=q)
    assert eta * S.sharpe(x, periods=1) == pytest.approx(empirical, rel=0.05)
    assert eta < math.sqrt(q)  # positive autocorrelation: sqrt(q) overstates


def test_sharpe_confidence_interval_is_the_point_estimate_plus_minus_z_se():
    x = 0.0005 + 0.01 * ar1(400, 0.3, seed=9)
    est = S.sharpe_confidence_interval(x, alpha=0.05)
    z = sps.norm.ppf(0.975)
    assert est.ci_low == pytest.approx(est.sharpe - z * est.standard_error, rel=1e-12)
    assert est.ci_high == pytest.approx(est.sharpe + z * est.standard_error, rel=1e-12)
    assert est.n_obs == 400 and est.method == "lo2002-hac"
    assert est.lag == int(math.floor(4 * (400 / 100) ** (2 / 9)))  # Newey-West (1994)


# ======================================================================================
# 2. Deflated Sharpe ratio
# ======================================================================================
def test_dsr_reproduces_the_published_worked_example():
    """Bailey & Lopez de Prado (2014), *Journal of Portfolio Management* 40(5):94-107,
    the numerical example on pp. 9-10 — the only fully specified numeric case the paper
    publishes.

    Verbatim inputs: "N = 100, V[{SR_n}] = 1/2, T = 1250, gamma_3 = -3 and gamma_4 = 10"
    for a strategy with an annualised Sharpe of 2.5 over 5 years of daily data.
    Verbatim outputs: "SR_0 = ... ~ 0.1132, non-annualized (with 250 observations per
    year)" and "DSR ~ ... = 0.9004 < 0.95"; "Should the strategist have made his discovery
    after running only N=46 independent trials ... DSR would have been 0.9505"; "If the
    strategy had exhibited Normal returns (gamma_3 = 0, gamma_4 = 3), DSR = 0.9505 after
    N=88 independent trials."

    Three published values, four decimals each.  They also pin every convention at once:
    ``sqrt(T-1)`` and not ``sqrt(T)``, non-excess kurtosis, and per-observation Sharpe.
    """
    std_ann = math.sqrt(0.5)
    assert S.expected_max_sharpe(100, math.sqrt(0.5 / 250)) == pytest.approx(0.1132, abs=5e-5)
    assert S.deflated_sharpe_ratio(2.5, 100, -3.0, 10.0, 1250, trial_sharpe_std=std_ann,
                                   periods=250) == pytest.approx(0.9004, abs=5e-5)
    assert S.deflated_sharpe_ratio(2.5, 46, -3.0, 10.0, 1250, trial_sharpe_std=std_ann,
                                   periods=250) == pytest.approx(0.9505, abs=5e-5)
    assert S.deflated_sharpe_ratio(2.5, 88, 0.0, 3.0, 1250, trial_sharpe_std=std_ann,
                                   periods=250) == pytest.approx(0.9505, abs=5e-5)
    # the paper's point: skew -3 / kurtosis 10 costs this strategy 42 trials of credibility
    assert (S.deflated_sharpe_ratio(2.5, 88, -3.0, 10.0, 1250, trial_sharpe_std=std_ann,
                                    periods=250)
            < S.deflated_sharpe_ratio(2.5, 88, 0.0, 3.0, 1250, trial_sharpe_std=std_ann,
                                      periods=250))


def test_psr_reduces_to_the_normal_test_under_gaussian_moments():
    """With ``skew = 0``, ``kurtosis = 3`` and a zero benchmark, the PSR denominator is
    ``sqrt(1 + SR^2/2)`` — Lo's V_IID — so the PSR is exactly the normal CDF of the
    classical Sharpe t-statistic ``SR sqrt(T-1) / sqrt(1 + SR^2/2)``.  Checked against
    scipy rather than against the module's own normal CDF."""
    for sr, n in ((0.05, 500), (0.1, 1250), (-0.02, 300)):
        z = sr * math.sqrt(n - 1) / math.sqrt(1 + 0.5 * sr ** 2)
        assert S.probabilistic_sharpe_ratio(sr, 0.0, 0.0, 3.0, n) == pytest.approx(
            float(sps.norm.cdf(z)), rel=1e-12)


def test_expected_max_sharpe_matches_the_maximum_of_n_normal_draws():
    """``SR_0`` is an approximation to the expected maximum of ``N`` draws from
    ``N(0, V)`` (the false-strategy theorem).  Simulated directly with 40,000 replications
    per ``N``; the published closed form is accurate to a couple of per cent and is known
    to run slightly high at small ``N``."""
    rng = np.random.default_rng(SEED + 2)
    for n_trials in (10, 100, 1000, 5000):
        mc = float(rng.standard_normal((40_000, n_trials)).max(axis=1).mean())
        formula = S.expected_max_sharpe(n_trials, 1.0)
        assert formula == pytest.approx(mc, rel=0.03), (n_trials, formula, mc)
    assert S.expected_max_sharpe(1, 1.0) == 0.0          # one trial: no selection bias
    assert S.expected_max_sharpe(500, 0.0) == 0.0        # no dispersion: nothing to deflate
    # scaling is linear in sqrt(V), exactly as published
    assert S.expected_max_sharpe(250, 0.4) == pytest.approx(
        0.4 * S.expected_max_sharpe(250, 1.0), rel=1e-12)


def test_dsr_shrinks_as_the_trial_count_rises():
    """The whole purpose of deflation: the same backtest becomes less credible the more
    configurations were searched to find it.  Strictly decreasing in ``n_trials``, and it
    crosses the project's 0.95 gate as the honest trial count grows."""
    kwargs = dict(skew=0.0, kurtosis=3.0, n_obs=1250, trial_sharpe_std=math.sqrt(0.5),
                  periods=250)
    values = [S.deflated_sharpe_ratio(2.5, n, **kwargs) for n in (1, 10, 50, 88, 200, 1000)]
    assert all(a > b for a, b in zip(values, values[1:])), values
    assert values[0] > 0.999            # a single trial is barely deflated at all
    assert values[3] == pytest.approx(0.9505, abs=5e-4)   # N=88 sits on the gate
    assert values[-1] < 0.80            # 1,000 trials sink the same Sharpe below the gate


def test_dsr_penalises_negative_skew_and_fat_tails():
    """Non-normality is the third correction in the paper's title.  Same Sharpe, same
    sample, worse distribution -> lower DSR."""
    base = dict(sharpe=0.15, n_trials=50, n_obs=1000, trial_sharpe_std=0.03)
    normal = S.deflated_sharpe_ratio(skew=0.0, kurtosis=3.0, **base)
    left_tail = S.deflated_sharpe_ratio(skew=-2.0, kurtosis=3.0, **base)
    fat = S.deflated_sharpe_ratio(skew=0.0, kurtosis=12.0, **base)
    assert left_tail < normal and fat < normal


def test_dsr_rejects_excess_kurtosis_and_impossible_moments():
    """``kurtosis`` is non-excess.  Passing scipy's default (Fisher/excess) kurtosis would
    silently inflate the DSR, so values below the mathematical minimum of 1 are refused."""
    with pytest.raises(ValueError, match="NON-excess"):
        S.deflated_sharpe_ratio(0.1, 100, 0.0, 0.0, 1000)      # excess kurtosis of a normal
    with pytest.raises(ValueError):
        S.deflated_sharpe_ratio(0.1, 0, 0.0, 3.0, 1000)        # zero trials
    with pytest.raises(ValueError, match="non-positive"):
        S.probabilistic_sharpe_ratio(0.9, 0.0, 3.0, 3.0, 500)  # inconsistent moments


def test_dsr_does_not_catch_lookahead():
    """**An honest negative, encoded as a test.**

    ``docs/RESEARCH.md`` section 4 and ``docs/research/01_rl_trading.md`` finding 3 record
    the E1 experiment of arXiv:2608.27734 ("What survives honest evaluation?"): a planted
    look-ahead oracle scored **design Sharpe 34.7, evaluation Sharpe 51.5 and a Deflated
    Sharpe Ratio of 1.00**, and PBO cleared it as well.  The statistics certified a cheat.

    This test plants the same cheat — tomorrow's return, known today — and asserts the
    same result:

    * the oracle's annualised Sharpe is absurd (>20 here; the paper measured 34.7/51.5 on
      a real 453-stock panel);
    * the DSR is 1.0 to floating-point precision even at **one million** trials, because
      deflation subtracts an ``SR_0`` of order 0.02 from a Sharpe of order 1.3;
    * PBO is 0.0 for the leaked column, because a leak wins *both* in and out of sample;
    * an honest strategy with a perfectly respectable annualised Sharpe near 0.5 is
      correctly rejected by the same gate.

    The conclusion this test protects: **DSR > 0.95 is not evidence that a backtest is
    clean.**  Leakage is caught by point-in-time data (S2), t+1 fills, and the
    planted-oracle harness test in the S3 exit criteria — never by this module.  If this
    test ever starts failing because the DSR "detects" the oracle, the statistic has been
    changed into something that is no longer the Bailey & Lopez de Prado DSR.
    """
    rng = np.random.default_rng(SEED)
    n = 1000
    market = rng.normal(0.0004, 0.02, n + 1)
    oracle = np.abs(market[1:])          # long if tomorrow is up, short if down: perfect
    honest = rng.normal(0.0004, 0.02, n)

    sr_oracle_daily = S.sharpe(oracle, periods=1)
    assert S.sharpe(oracle) > 20.0
    skew = float(sps.skew(oracle, bias=True))
    kurt = float(sps.kurtosis(oracle, fisher=False, bias=True))

    for n_trials in (1, 100, 10_000, 1_000_000):
        dsr = S.deflated_sharpe_ratio(sr_oracle_daily, n_trials, skew, kurt, n)
        assert dsr == pytest.approx(1.0, abs=1e-12), (n_trials, dsr)
        assert dsr > 0.95                                   # clears the project gate

    # ... and PBO does not flag it either: the leak is consistent, not overfit
    trials = rng.normal(0.0, 0.02, size=(n, 50))
    trials[:, 13] = oracle
    pbo = S.probability_of_backtest_overfitting(trials, n_splits=16)
    assert pbo.pbo == 0.0
    assert int(np.bincount(pbo.selected_trial, minlength=50).argmax()) == 13

    # the same gate applied to an honest strategy does its job
    sr_h = S.sharpe(honest, periods=1)
    dsr_h = S.deflated_sharpe_ratio(sr_h, 100, float(sps.skew(honest, bias=True)),
                                    float(sps.kurtosis(honest, fisher=False, bias=True)), n)
    assert 0.0 < S.sharpe(honest) < 1.5
    assert dsr_h < 0.95


# ======================================================================================
# 3. Probability of backtest overfitting (CSCV)
# ======================================================================================
def test_cscv_combination_count_matches_the_published_values():
    """Bailey, Borwein, Lopez de Prado & Zhu (2017) eq. (2.3): the number of CSCV splits
    is ``comb(S, S/2)``.  The paper prints 6 for S=4 ("the six combinations of four
    subsamples A, B, C, D") and 924 for S=12, both correct — but prints **12,780** for
    S=16 on two separate pages, which is a digit transposition of the true 12,870.  This
    test asserts the arithmetic, not the typo."""
    rng = np.random.default_rng(SEED + 3)
    m = rng.standard_normal((480, 6))
    assert S.probability_of_backtest_overfitting(m, n_splits=4).n_combinations == 6
    assert S.probability_of_backtest_overfitting(m, n_splits=12).n_combinations == 924
    assert S.probability_of_backtest_overfitting(m, n_splits=16).n_combinations == 12870
    assert math.comb(16, 8) == 12870 != 12780


def test_pbo_is_one_half_on_pure_noise():
    """With 50 identically worthless trials the in-sample winner is a coin flip out of
    sample, so its relative rank is uniform and PBO sits at 1/2.  This is the honest null:
    a search over noise has exactly a 50% chance of picking a below-median strategy.

    NO PUBLISHED REFERENCE: Bailey et al. report PBO values only from figures based on
    unseeded Monte Carlo or proprietary data (74%, 55%, 13%, 0.04%), none of which is
    reproducible.  The 0.5 target is derived from the procedure's own definition and
    confirmed by simulation here.
    """
    rng = np.random.default_rng(SEED + 4)
    m = rng.standard_normal((1200, 50)) * 0.01
    res = S.probability_of_backtest_overfitting(m, n_splits=16)
    assert res.pbo == pytest.approx(0.5, abs=0.12), res.pbo
    assert res.n_obs_used == 1200 and res.n_trials == 50
    # the logit distribution is centred, not merely the fraction below zero
    assert abs(float(np.median(res.logits))) < 0.6


def test_pbo_is_low_when_one_trial_has_genuine_skill():
    """A column with a real edge wins in sample *and* out of sample, so the selection
    procedure is not overfitting and PBO collapses.  Contrasted against the noise case in
    the same test so the difference cannot be a coincidence of seeds."""
    rng = np.random.default_rng(SEED + 5)
    noise = rng.standard_normal((1200, 50)) * 0.01
    skilled = noise.copy()
    skilled[:, 7] += 0.0015              # a genuine per-period edge of 0.15 sd
    pbo_noise = S.probability_of_backtest_overfitting(noise, n_splits=16).pbo
    res = S.probability_of_backtest_overfitting(skilled, n_splits=16)
    assert res.pbo < 0.10, res.pbo
    assert pbo_noise > res.pbo + 0.3
    assert int(np.bincount(res.selected_trial, minlength=50).argmax()) == 7


def test_pbo_reaches_one_when_in_sample_ranking_reverses_out_of_sample():
    """The upper end of the statistic's range, and proof it can reject.

    Each trial is built to be strong in a random half of the blocks and weak in the other
    half.  Whichever trial looks best in sample is, by construction, the one whose weak
    blocks are all out of sample.  PBO must be ~1 and the in-sample/out-of-sample
    degradation slope must be strongly negative.  A statistic that returned ~0.5 here
    would be measuring nothing.
    """
    rng = np.random.default_rng(SEED + 6)
    n_splits, n_trials, block = 16, 40, 60
    m = rng.standard_normal((n_splits * block, n_trials)) * 0.01
    for j in range(n_trials):
        good = set(rng.permutation(n_splits)[: n_splits // 2].tolist())
        for s in range(n_splits):
            sign = 1.0 if s in good else -1.0
            m[s * block:(s + 1) * block, j] += sign * 0.02
    res = S.probability_of_backtest_overfitting(m, n_splits=n_splits)
    assert res.pbo > 0.95, res.pbo
    assert res.degradation_slope < -0.5


def test_pbo_matches_a_naive_reference_implementation():
    """The fast path computes each split's Sharpe from per-block sums.  This test rebuilds
    the whole procedure the slow, obvious way — explicit row slicing, explicit means, and
    ``scipy.stats.rankdata`` for the ranks — and requires the two to agree exactly."""
    rng = np.random.default_rng(SEED + 7)
    t_obs, n_trials, n_splits = 96, 5, 6
    m = rng.standard_normal((t_obs, n_trials)) * 0.01 + 0.0005
    res = S.probability_of_backtest_overfitting(m, n_splits=n_splits)

    import itertools
    block = t_obs // n_splits
    blocks = [m[s * block:(s + 1) * block] for s in range(n_splits)]
    logits = []
    for c in itertools.combinations(range(n_splits), n_splits // 2):
        ins = np.vstack([blocks[s] for s in c])
        oos = np.vstack([blocks[s] for s in range(n_splits) if s not in c])
        sr_is = ins.mean(axis=0) / ins.std(axis=0)
        sr_oos = oos.mean(axis=0) / oos.std(axis=0)
        star = int(np.argmax(sr_is))
        omega = float(sps.rankdata(sr_oos)[star]) / (n_trials + 1)
        logits.append(math.log(omega / (1 - omega)))
    assert res.logits == pytest.approx(np.array(logits), rel=1e-10)
    assert res.pbo == pytest.approx(float(np.mean(np.array(logits) < 0)))


def test_pbo_accepts_a_custom_performance_metric():
    """Step (ii) of the paper's conditions: any metric estimable on subsamples.  A custom
    callable must reproduce the default when it *is* the default metric."""
    rng = np.random.default_rng(SEED + 8)
    m = rng.standard_normal((240, 8)) * 0.01

    def sharpe_cols(block):
        return block.mean(axis=0) / block.std(axis=0)

    fast = S.probability_of_backtest_overfitting(m, n_splits=6)
    slow = S.probability_of_backtest_overfitting(m, n_splits=6, performance=sharpe_cols)
    assert fast.logits == pytest.approx(slow.logits, rel=1e-10)


def test_pbo_rejects_inputs_it_cannot_score():
    rng = np.random.default_rng(SEED + 9)
    m = rng.standard_normal((240, 8))
    with pytest.raises(ValueError, match="even integer"):
        S.probability_of_backtest_overfitting(m, n_splits=15)
    with pytest.raises(ValueError, match="at least 2 trials"):
        S.probability_of_backtest_overfitting(m[:, :1], n_splits=4)
    with pytest.raises(ValueError, match="cannot be split"):
        S.probability_of_backtest_overfitting(m[:6], n_splits=16)
    with pytest.raises(ValueError, match="2-D"):
        S.probability_of_backtest_overfitting(m[:, 0], n_splits=4)
    flat = m.copy()
    flat[:, 3] = 0.01                    # a constant column has no Sharpe
    with pytest.raises(ValueError, match="zero variance"):
        S.probability_of_backtest_overfitting(flat, n_splits=4)


# ======================================================================================
# 4. Diebold-Mariano
# ======================================================================================
def test_dm_does_not_reject_identical_forecasts():
    """Two identical forecast series have a loss differential that is identically zero.
    The test must return "no evidence of a difference" rather than dividing by a zero
    variance — this is the degenerate case the S3 harness hits whenever an ablation arm is
    accidentally wired to the same policy."""
    rng = np.random.default_rng(SEED + 10)
    e = rng.standard_normal(300)
    res = S.diebold_mariano(e, e.copy())
    assert res.statistic == 0.0
    assert res.p_value == 1.0
    assert res.better == "tie"
    assert res.long_run_variance == 0.0
    for h in (1, 5, 20):
        assert S.diebold_mariano(e, e.copy(), h=h).p_value == 1.0


def test_dm_rejects_when_one_forecast_is_clearly_better():
    """A model with half the error variance, over 500 periods, must be detected at
    p < 0.01 — and the sign convention must point at the right model."""
    rng = np.random.default_rng(SEED + 11)
    n = 500
    ea = rng.normal(0.0, 1.0, n)
    eb = rng.normal(0.0, 0.5, n)
    res = S.diebold_mariano(ea, eb)
    assert res.p_value < 0.01
    assert res.statistic > 0            # A carries the larger loss
    assert res.better == "b"
    flipped = S.diebold_mariano(eb, ea)
    assert flipped.statistic == pytest.approx(-res.statistic, rel=1e-12)
    assert flipped.better == "a"
    assert S.diebold_mariano(eb, ea, alternative="a-better").p_value < 0.005
    assert S.diebold_mariano(eb, ea, alternative="b-better").p_value > 0.995


def test_dm_at_h1_is_exactly_the_one_sample_t_statistic():
    """An exact algebraic identity, checked against scipy.

    At ``h = 1`` there are no lags, so ``V = gamma_0`` with the maximum-likelihood divisor
    ``T``.  The Harvey-Leybourne-Newbold factor is then ``sqrt((T-1)/T)``, which converts
    that divisor into the unbiased one — so ``DM* == t`` to machine precision, and both are
    referred to ``t_{T-1}``.  This is also how R's ``forecast::dm.test`` behaves.
    """
    rng = np.random.default_rng(SEED + 12)
    for n in (25, 200, 941):
        ea = rng.normal(0.0, 1.0, n)
        eb = rng.normal(0.0, 0.9, n)
        res = S.diebold_mariano(ea, eb, h=1)
        tt = sps.ttest_1samp(ea ** 2 - eb ** 2, 0.0)
        assert res.statistic == pytest.approx(float(tt.statistic), rel=1e-12)
        assert res.p_value == pytest.approx(float(tt.pvalue), rel=1e-10)
        assert res.df == n - 1
        assert res.harvey_correction == pytest.approx(math.sqrt((n - 1) / n), rel=1e-12)


def test_dm_hac_variance_matches_statsmodels():
    """For ``h > 1`` the long-run variance is a Newey-West HAC estimate with Bartlett
    weights and ``L = h-1`` lags.  statsmodels computes the same object as the HAC
    covariance of a regression of the loss differential on a constant, so the DM statistic
    must equal its t-value times the HLN factor."""
    rng = np.random.default_rng(SEED + 13)
    n = 400
    ea = rng.normal(0.0, 1.0, n)
    eb = rng.normal(0.0, 0.85, n)
    d = ea ** 2 - eb ** 2
    for h in (2, 3, 5, 12):
        res = S.diebold_mariano(ea, eb, h=h)
        ols = sm.OLS(d, np.ones((n, 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": h - 1, "use_correction": False})
        assert res.statistic == pytest.approx(float(ols.tvalues[0]) * res.harvey_correction,
                                              rel=1e-10), h
        assert res.lag == h - 1


def test_dm_uniform_weights_reproduce_the_published_truncation():
    """Diebold & Mariano (1995) truncate at ``h-1`` with uniform weights;
    ``weights="uniform"`` must reproduce ``gamma_0 + 2*sum(gamma_k)`` computed by hand,
    and must differ from the Bartlett default whenever ``h > 1``."""
    rng = np.random.default_rng(SEED + 14)
    n, h = 300, 4
    ea = rng.normal(0.0, 1.0, n)
    eb = rng.normal(0.0, 0.9, n)
    d = ea ** 2 - eb ** 2
    dev = d - d.mean()
    v = float(dev @ dev) / n + 2.0 * sum(float(dev[k:] @ dev[:-k]) / n for k in range(1, h))
    res = S.diebold_mariano(ea, eb, h=h, weights="uniform")
    assert res.long_run_variance == pytest.approx(v, rel=1e-12)
    assert res.long_run_variance != S.diebold_mariano(ea, eb, h=h).long_run_variance


def test_harvey_correction_reduces_small_sample_over_rejection():
    """Harvey, Leybourne & Newbold (1997) exists because the raw DM statistic, referred to
    a standard normal, rejects far too often in small samples with ``h > 1``.

    Measured over 3,000 replications at ``T = 32``, ``h = 4`` under a true null (an
    MA(3) loss differential with mean zero): the uncorrected normal test rejects about 17%
    of the time at a nominal 5%, the corrected t test about 12%.  Both over-reject — the
    correction shrinks the error, it does not remove it, and this test says so rather than
    claiming the corrected test is exact.
    """
    rng = np.random.default_rng(SEED + 15)
    n, h, reps = 32, 4, 3000
    zeros = np.zeros(n)
    rej_raw = rej_corrected = 0
    for _ in range(reps):
        z = rng.standard_normal(n + h)
        d = np.convolve(z, np.ones(h), mode="valid")[:n]     # MA(h-1), mean zero
        res = S.diebold_mariano(d, zeros, h=h, loss=lambda e: e)
        rej_raw += abs(res.statistic / res.harvey_correction) > 1.959963984540054
        rej_corrected += res.p_value < 0.05
    size_raw, size_corrected = rej_raw / reps, rej_corrected / reps
    assert size_raw > 0.13                       # the uncorrected test is badly oversized
    assert size_corrected < size_raw - 0.03      # the correction materially helps
    assert size_corrected > 0.05                 # ... and is still not exact. Say so.


def test_dm_supports_alternative_loss_functions():
    """``loss`` is part of the hypothesis: MSE and MAE can disagree, and DM (1995) is
    explicitly defined for an arbitrary loss."""
    rng = np.random.default_rng(SEED + 16)
    n = 400
    ea = rng.standard_t(3, n)            # fat tails: squared loss is dominated by outliers
    eb = rng.normal(0.0, 1.3, n)
    mse = S.diebold_mariano(ea, eb, loss="squared")
    mae = S.diebold_mariano(ea, eb, loss="absolute")
    assert mse.mean_loss_differential != mae.mean_loss_differential
    custom = S.diebold_mariano(ea, eb, loss=lambda e: np.abs(e))
    assert custom.statistic == pytest.approx(mae.statistic, rel=1e-12)


def test_dm_validates_its_inputs():
    rng = np.random.default_rng(SEED + 17)
    a, b = rng.standard_normal(100), rng.standard_normal(100)
    with pytest.raises(ValueError, match="differ in length"):
        S.diebold_mariano(a, b[:50])
    with pytest.raises(ValueError, match="h must be"):
        S.diebold_mariano(a, b, h=0)
    with pytest.raises(ValueError, match="exceeds the sample"):
        S.diebold_mariano(a, b, h=200)
    with pytest.raises(ValueError, match="unknown alternative"):
        S.diebold_mariano(a, b, alternative="greater")
    with pytest.raises(ValueError, match="unknown loss"):
        S.diebold_mariano(a, b, loss="huber")
    with pytest.raises(ValueError, match="at least 3"):
        S.diebold_mariano(a[:2], b[:2])
