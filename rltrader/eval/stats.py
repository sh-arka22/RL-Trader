"""Statistical significance for the S3 evaluation harness.

PLAN.md S3 builds *the referee before the players*. This module is the referee's
arithmetic: everything that turns a backtest number into a claim about whether an edge
exists. Six statistics, each cited to the paper it comes from:

===============  ==============================================================
``sharpe``       Sharpe (1966, 1994); standard error from Lo (2002), which
                 corrects for serial correlation in the return series.
``deflated_sharpe_ratio``
                 Bailey & Lopez de Prado (2014) — deflates a *selected* Sharpe
                 for the number of trials, the sample length, skewness and
                 kurtosis.  **It does not detect look-ahead leakage.**
``probability_of_backtest_overfitting``
                 Bailey, Borwein, Lopez de Prado & Zhu (2017), the CSCV
                 procedure.
``diebold_mariano``
                 Diebold & Mariano (1995) with the Harvey, Leybourne & Newbold
                 (1997) small-sample correction and a Newey & West (1987) HAC
                 long-run variance.
``stationary_bootstrap``
                 Politis & Romano (1994) — dependence-preserving resampling for
                 confidence intervals on Sharpe differences.
``benjamini_hochberg``
                 Benjamini & Hochberg (1995) FDR control across many strategy
                 tests.
===============  ==============================================================

Two warnings are load-bearing, and both are in the repo's own evidence base.

1.  **Deflation does not catch leakage.**  ``docs/RESEARCH.md`` section 4 and
    ``docs/research/01_rl_trading.md`` finding 3 record that arXiv:2608.27734 planted a
    look-ahead oracle and measured design Sharpe 34.7, eval Sharpe 51.5 and a **Deflated
    Sharpe Ratio of 1.00**.  The statistic certified the cheat.  DSR and PBO correct for
    *search*; nothing here corrects for a contaminated information set.  Point-in-time
    discipline and the planted-oracle test do that, not this module.
    ``tests/test_stats.py::test_dsr_does_not_catch_lookahead`` encodes that negative
    result as an assertion so it cannot quietly stop being true.

2.  **A statistic that cannot reject is worthless**, exactly as ``docs/reports/S2_DATA.md``
    section 8 found for gates that could not fail.  Every function here is tested against
    an analytically known answer or an independent third-party implementation
    (scipy / statsmodels, dev dependencies used *only* by the test suite), plus a paired
    case where the naive alternative provably gets it wrong.

Runtime dependencies: numpy and the standard library.  Nothing else — the special
functions below (inverse normal CDF, regularised incomplete beta) are implemented here so
the harness has no hidden scipy dependency, and are checked against scipy in the tests.

Conventions used throughout, because getting these wrong is the usual source of a
wrong number:

* ``returns`` are **per-period simple net returns** (after costs), not percentages.
* ``rf`` is quoted **per period, in the same units as ``returns``** (divide an annual
  rate by ``periods`` before passing it).
* Moment estimators use the **maximum-likelihood divisor (``ddof=0``)** by default.  That
  is the convention Lo (2002) and Bailey & Lopez de Prado use to derive their standard
  errors; ``empyrical``/``quantstats`` use ``ddof=1``.  The two differ by exactly
  ``sqrt(n / (n - 1))`` and ``ddof`` is exposed where it matters.
* ``kurtosis`` arguments are **non-excess** (3.0 for a Gaussian).  Passing excess
  kurtosis silently biases the DSR, so values below 1.0 are rejected.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "sharpe",
    "sharpe_standard_error",
    "sharpe_confidence_interval",
    "lo_annualisation_factor",
    "SharpeEstimate",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "PBOResult",
    "diebold_mariano",
    "DMResult",
    "stationary_bootstrap",
    "stationary_bootstrap_variance",
    "stationary_bootstrap_ci",
    "sharpe_difference_ci",
    "BootstrapCI",
    "benjamini_hochberg",
    "BHResult",
    "three_gate_verdict",
    "Verdict",
]

EULER_MASCHERONI = 0.5772156649015328606  # gamma, used by the DSR expected-maximum term


# --------------------------------------------------------------------------------------
# special functions (stdlib only; validated against scipy in tests/test_stats.py)
# --------------------------------------------------------------------------------------
def _norm_cdf(x):
    """Standard normal CDF via ``math.erf``.  Accurate to ~1e-16."""
    x = np.asarray(x, dtype=float)
    out = 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(x / math.sqrt(2.0)))
    return float(out) if out.ndim == 0 else out


# Wichura (1988), Algorithm AS 241 "The Percentage Points of the Normal Distribution",
# Applied Statistics 37(3):477-484 — the PPND16 branch, |relative error| < 1e-15.
_AS241_A = (3.3871328727963666080e0, 1.3314166789178437745e2, 1.9715909503065514427e3,
            1.3731693765509461125e4, 4.5921953931549871457e4, 6.7265770927008700853e4,
            3.3430575583588128105e4, 2.5090809287301226727e3)
_AS241_B = (1.0, 4.2313330701600911252e1, 6.8718700749205790830e2,
            5.3941960214247511077e3, 2.1213794301586595867e4, 3.9307895800092710610e4,
            2.8729085735721942674e4, 5.2264952788528545610e3)
_AS241_C = (1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
            3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
            2.27238449892691845833e-2, 7.74545014278341407640e-4)
_AS241_D = (1.0, 2.05319162663775882187e0, 1.67638483018380384940e0,
            6.89767334985100004550e-1, 1.48103976427480074590e-1, 1.51986665636164571966e-2,
            5.47593808499534494600e-4, 1.05075007164441684324e-9)
_AS241_E = (6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
            2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
            2.71155556874348757815e-5, 2.01033439929228813265e-7)
_AS241_F = (1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1,
            1.48753612908506148525e-2, 7.86869131145613259100e-4, 1.84631831751005468180e-5,
            1.42151175831644588870e-7, 2.04426310338993978564e-15)


def _poly(coefs, r):
    out = 0.0
    for c in reversed(coefs):
        out = out * r + c
    return out


def _norm_ppf_scalar(p: float) -> float:
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"norm_ppf needs 0 <= p <= 1, got {p}")
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * _poly(_AS241_A, r) / _poly(_AS241_B, r)
    r = p if q < 0.0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        val = _poly(_AS241_C, r) / _poly(_AS241_D, r)
    else:
        r -= 5.0
        val = _poly(_AS241_E, r) / _poly(_AS241_F, r)
    return -val if q < 0.0 else val


def _norm_ppf(p):
    """Inverse standard normal CDF (Wichura 1988, AS 241 / PPND16)."""
    arr = np.asarray(p, dtype=float)
    out = np.vectorize(_norm_ppf_scalar, otypes=[float])(arr)
    return float(out) if out.ndim == 0 else out


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny, eps, maxit = 1e-300, 3e-16, 500
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError("incomplete beta continued fraction did not converge")


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_sf(t: float, df: float) -> float:
    """Upper-tail survival function P(T > t) of a Student-t with ``df`` d.o.f."""
    if df <= 0:
        raise ValueError(f"degrees of freedom must be positive, got {df}")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    half = 0.5 * _betainc(0.5 * df, 0.5, df / (df + t * t))
    return half if t >= 0.0 else 1.0 - half


def _as_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values; clean or drop them first")
    return arr


# --------------------------------------------------------------------------------------
# 1. Sharpe ratio and its standard error
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SharpeEstimate:
    """A Sharpe ratio with the uncertainty attached.  Never report the first field alone."""

    sharpe: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_obs: int
    periods: int
    lag: int
    method: str

    def __str__(self) -> str:  # pragma: no cover - reporting convenience
        return (f"SR={self.sharpe:.3f} +/- {self.standard_error:.3f} "
                f"[{self.ci_low:.3f}, {self.ci_high:.3f}] "
                f"(n={self.n_obs}, {self.method}, lag={self.lag})")


def sharpe(returns, rf: float = 0.0, periods: int = 252, ddof: int = 0) -> float:
    r"""Annualised Sharpe ratio.

    Sharpe, W. F. (1966), "Mutual Fund Performance", *Journal of Business* 39(1):119-138;
    restated in Sharpe, W. F. (1994), "The Sharpe Ratio", *Journal of Portfolio
    Management* 21(1):49-58.

    .. math::

        \widehat{SR} = \sqrt{q}\;\frac{\hat\mu - R_f}{\hat\sigma},
        \qquad
        \hat\mu = \frac1T\sum_{t=1}^T r_t,
        \qquad
        \hat\sigma^2 = \frac{1}{T-\mathrm{ddof}}\sum_{t=1}^T (r_t-\hat\mu)^2

    with :math:`q` = ``periods`` (252 for daily bars).  The :math:`\sqrt{q}` scaling is
    only valid for i.i.d. returns; with serial correlation use
    :func:`lo_annualisation_factor` instead, which is Lo's (2002) point.

    Parameters
    ----------
    returns : array_like
        Per-period simple net returns.  In this project: net of the modelled costs, at
        whichever of the three cost levels is being reported.
    rf : float
        Risk-free rate **per period**, same units as ``returns``.  Divide an annual rate
        by ``periods``.  ``docs/RESEARCH.md`` section 2 computes the project's passive bar
        at ``rf = 0``: SPY 0.82, equal-weight-5 1.07.
    periods : int
        Periods per year.  Pass ``1`` to get a per-period (non-annualised) Sharpe, which
        is the unit :func:`deflated_sharpe_ratio` needs.
    ddof : int
        Divisor correction for the volatility estimate.  ``0`` (default) is the
        maximum-likelihood estimator used by Lo (2002) and by Bailey & Lopez de Prado;
        ``1`` is the ``empyrical``/``quantstats`` convention.  The two differ by exactly
        ``sqrt(T / (T - 1))``.

    Raises
    ------
    ValueError
        If fewer than two observations are supplied, or the returns have zero variance
        (an infinite Sharpe is a bug in the caller, not a result worth reporting).
    """
    r = _as_1d(returns, "returns")
    n = r.size
    if n < 2:
        raise ValueError(f"need at least 2 returns, got {n}")
    if n - ddof <= 0:
        raise ValueError(f"ddof={ddof} leaves no degrees of freedom for n={n}")
    excess = r - rf
    sd = float(np.sqrt(np.sum((r - r.mean()) ** 2) / (n - ddof)))
    if sd == 0.0:
        raise ValueError("returns have zero variance: the Sharpe ratio is undefined")
    return float(np.mean(excess) / sd * math.sqrt(periods))


def _newey_west_omega(h: np.ndarray, lag: int) -> np.ndarray:
    r"""Newey & West (1987) HAC covariance of the moment conditions ``h`` (T x k).

    Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
    *Econometrica* 55(3):703-708.

    .. math::

        \hat\Omega = \hat\Gamma_0
        + \sum_{k=1}^{L}\Big(1-\frac{k}{L+1}\Big)\big(\hat\Gamma_k+\hat\Gamma_k'\big),
        \qquad
        \hat\Gamma_k = \frac1T\sum_{t=k+1}^{T} h_t h_{t-k}'

    The Bartlett weights :math:`1-k/(L+1)` are what make the estimate positive
    semi-definite; a uniform truncation can return a negative variance.
    """
    t_obs = h.shape[0]
    omega = h.T @ h / t_obs
    for k in range(1, lag + 1):
        gamma = h[k:].T @ h[:-k] / t_obs
        omega = omega + (1.0 - k / (lag + 1.0)) * (gamma + gamma.T)
    return omega


def _default_lag(n: int) -> int:
    """Newey & West (1994) rule-of-thumb bandwidth ``floor(4 (T/100)^(2/9))``."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def sharpe_standard_error(returns, rf: float = 0.0, periods: int = 252,
                          lag: int | None = None, iid: bool = False) -> float:
    r"""Standard error of the annualised Sharpe ratio, Lo (2002).

    Lo, A. W. (2002), "The Statistics of Sharpe Ratios", *Financial Analysts Journal*
    58(4):36-52.

    **i.i.d. case** (Lo 2002 eq. 9; also Jobson & Korkie 1981 with Memmel's 2003
    correction):

    .. math::  \mathrm{Var}(\widehat{SR}) = \frac{1 + \tfrac12 SR^2}{T}

    **Non-i.i.d. case** (Lo 2002 eqs. 12-14), the default here.  With
    :math:`\theta=(\mu,\sigma^2)` and :math:`g(\theta)=(\mu-R_f)/\sigma`, GMM gives

    .. math::

        \mathrm{Var}(\widehat{SR}) = \frac{1}{T}\,
        \frac{\partial g}{\partial\theta}'\,\hat\Omega\,\frac{\partial g}{\partial\theta},
        \qquad
        \frac{\partial g}{\partial\theta} =
        \Big(\frac{1}{\sigma},\; -\frac{\mu-R_f}{2\sigma^3}\Big)'

    where :math:`\hat\Omega` is the Newey-West HAC covariance of the moment conditions
    :math:`h_t = \big(r_t-\mu,\;(r_t-\mu)^2-\sigma^2\big)'`.

    Why it matters here: a trading policy that holds a position for several days produces
    autocorrelated returns, and the i.i.d. standard error is then **too small** — it
    manufactures significance.  ``tests/test_stats.py`` checks exactly that on an AR(1)
    series, so this correction cannot silently stop working.

    Parameters
    ----------
    lag : int or None
        HAC truncation lag.  ``None`` uses Newey & West's (1994) rule of thumb
        ``floor(4 (T/100)^(2/9))``.  ``lag=0`` reduces the estimator to White's
        heteroskedasticity-only form, which for Gaussian returns equals eq. 9 exactly.
    iid : bool
        Use the closed-form eq. 9 instead.  Only honest when the returns are serially
        independent — and they usually are not.

    Returns
    -------
    float
        Standard error on the **annualised** scale (the per-period standard error times
        ``sqrt(periods)``, matching the way :func:`sharpe` annualises the point estimate).
    """
    r = _as_1d(returns, "returns")
    n = r.size
    if n < 2:
        raise ValueError(f"need at least 2 returns, got {n}")
    mu = float(r.mean())
    dev = r - mu
    sig2 = float(np.mean(dev ** 2))
    if sig2 == 0.0:
        raise ValueError("returns have zero variance: the Sharpe standard error is undefined")
    sig = math.sqrt(sig2)
    sr = (mu - rf) / sig
    if iid:
        return math.sqrt((1.0 + 0.5 * sr ** 2) / n) * math.sqrt(periods)
    if lag is None:
        lag = _default_lag(n)
    if lag < 0 or lag >= n:
        raise ValueError(f"lag must be in [0, {n - 1}], got {lag}")
    h = np.column_stack([dev, dev ** 2 - sig2])
    omega = _newey_west_omega(h, lag)
    grad = np.array([1.0 / sig, -(mu - rf) / (2.0 * sig ** 3)])
    var = float(grad @ omega @ grad) / n
    if var < 0.0:  # pragma: no cover - Bartlett weights make Omega PSD
        raise RuntimeError("HAC variance estimate is negative")
    return math.sqrt(var) * math.sqrt(periods)


def sharpe_confidence_interval(returns, rf: float = 0.0, periods: int = 252,
                               alpha: float = 0.05, lag: int | None = None,
                               iid: bool = False, ddof: int = 0) -> SharpeEstimate:
    """Sharpe ratio with a Lo (2002) asymptotic normal confidence interval.

    The interval is ``SR +/- z_{1-alpha/2} * SE``.  Asymptotic: at ``T`` below a few
    hundred it is optimistic, which is why the harness also reports the stationary
    bootstrap interval (:func:`sharpe_difference_ci`).
    """
    sr = sharpe(returns, rf=rf, periods=periods, ddof=ddof)
    se = sharpe_standard_error(returns, rf=rf, periods=periods, lag=lag, iid=iid)
    z = _norm_ppf_scalar(1.0 - alpha / 2.0)
    n = _as_1d(returns, "returns").size
    used_lag = 0 if iid else (_default_lag(n) if lag is None else lag)
    return SharpeEstimate(sharpe=sr, standard_error=se, ci_low=sr - z * se,
                          ci_high=sr + z * se, n_obs=n, periods=periods,
                          lag=used_lag, method="lo2002-iid" if iid else "lo2002-hac")


def lo_annualisation_factor(returns=None, q: int = 252, rho=None) -> float:
    r"""Lo's (2002) serial-correlation-aware annualisation factor :math:`\eta(q)`.

    Lo, A. W. (2002), *Financial Analysts Journal* 58(4):36-52, eqs. (19)-(20):

    .. math::

        \widehat{SR}(q) = \eta(q)\,\widehat{SR},
        \qquad
        \eta(q) = \frac{q}{\sqrt{q + 2\sum_{k=1}^{q-1}(q-k)\,\rho_k}}

    For i.i.d. returns all :math:`\rho_k=0` and :math:`\eta(q)=\sqrt{q}`, the textbook
    factor.  With positive autocorrelation :math:`\eta(q) < \sqrt{q}`, so the usual
    :math:`\sqrt{252}` **overstates** the annual Sharpe.

    This is a definition, not an approximation: :math:`\eta(q)` is exactly
    :math:`q\mu / \sigma(q)` where :math:`\sigma(q)` is the standard deviation of the
    ``q``-period compounded (log-additive) return.  The test suite checks it against a
    directly simulated aggregation of an AR(1) series.

    Parameters
    ----------
    returns : array_like or None
        Series from which to estimate :math:`\rho_1 \dots \rho_{q-1}`.
    rho : array_like or None
        Autocorrelations :math:`\rho_1 \dots \rho_{q-1}` supplied directly (takes
        precedence over ``returns``).  Shorter sequences are zero-padded.
    """
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}")
    if q == 1:
        return 1.0
    if rho is None:
        if returns is None:
            raise ValueError("supply either returns or rho")
        r = _as_1d(returns, "returns")
        dev = r - r.mean()
        denom = float(dev @ dev)
        if denom == 0.0:
            raise ValueError("returns have zero variance")
        rho_hat = np.array([float(dev[k:] @ dev[:-k]) / denom for k in range(1, q)])
    else:
        rho_hat = np.asarray(rho, dtype=float).ravel()[: q - 1]
        if rho_hat.size < q - 1:
            rho_hat = np.concatenate([rho_hat, np.zeros(q - 1 - rho_hat.size)])
    k = np.arange(1, q)
    denom_sum = q + 2.0 * float(np.sum((q - k) * rho_hat))
    if denom_sum <= 0.0:
        raise ValueError("implied variance of the q-period return is non-positive; "
                         "the autocorrelation estimates are inconsistent")
    return q / math.sqrt(denom_sum)


# --------------------------------------------------------------------------------------
# 2. Deflated Sharpe ratio
# --------------------------------------------------------------------------------------
def probabilistic_sharpe_ratio(sharpe: float, benchmark_sr: float, skew: float,
                               kurtosis: float, n_obs: int) -> float:
    r"""Probabilistic Sharpe Ratio: P(true SR > benchmark) given a non-normal sample.

    Bailey, D. H. & Lopez de Prado, M. (2012), "The Sharpe Ratio Efficient Frontier",
    *Journal of Risk* 15(2):3-44; restated as eq. (1) of the Deflated Sharpe Ratio paper
    (Bailey & Lopez de Prado 2014, *Journal of Portfolio Management* 40(5):94-107).

    .. math::

        \widehat{PSR}(SR^*) = Z\!\left[
            \frac{(\widehat{SR} - SR^*)\sqrt{T-1}}
                 {\sqrt{1 - \hat\gamma_3\widehat{SR} + \frac{\hat\gamma_4-1}{4}\widehat{SR}^2}}
        \right]

    where :math:`Z[\cdot]` is the standard normal CDF, :math:`\hat\gamma_3` is the skewness
    and :math:`\hat\gamma_4` the **non-excess** kurtosis (3 for a Gaussian).  All Sharpe
    quantities must be in the **same, per-observation** frequency as ``n_obs``: a daily
    Sharpe with ``n_obs`` daily observations.  Annualised inputs give a wrong answer, so
    de-annualise first (``sharpe(..., periods=1)``).

    Negative skewness and fat tails *reduce* the PSR: the same Sharpe is less trustworthy
    when the return distribution has a long left tail.
    """
    if n_obs < 2:
        raise ValueError(f"need at least 2 observations, got {n_obs}")
    if kurtosis < 1.0:
        raise ValueError(
            f"kurtosis={kurtosis} is below the mathematical minimum of 1; this argument "
            "is NON-excess kurtosis (3.0 for a Gaussian) -- you probably passed excess "
            "kurtosis, which biases the DSR upward")
    variance = 1.0 - skew * sharpe + 0.25 * (kurtosis - 1.0) * sharpe ** 2
    if variance <= 0.0:
        raise ValueError(
            "the estimated variance of the Sharpe estimator is non-positive "
            f"(1 - g3*SR + (g4-1)/4*SR^2 = {variance:.4g}); the moment inputs are "
            "mutually inconsistent")
    z = (sharpe - benchmark_sr) * math.sqrt(n_obs - 1) / math.sqrt(variance)
    return float(_norm_cdf(z))


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    r"""Expected maximum Sharpe over ``n_trials`` independent trials with **zero** true SR.

    Bailey & Lopez de Prado (2014), *Journal of Portfolio Management* 40(5):94-107,
    the :math:`SR_0` term of the Deflated Sharpe Ratio (their eq. 5, from the
    false-strategy theorem):

    .. math::

        SR_0 = \sqrt{V}\left[(1-\gamma)\,Z^{-1}\!\Big(1-\frac1N\Big)
               + \gamma\,Z^{-1}\!\Big(1-\frac{1}{N e}\Big)\right]

    with :math:`\gamma \approx 0.5772156649` the Euler-Mascheroni constant,
    :math:`V = \mathrm{Var}[\widehat{SR}_n]` the variance of the Sharpe ratios **across
    the trials**, and :math:`N` the number of trials.  It is the expected value of the
    maximum of ``N`` draws from :math:`N(0, V)` — i.e. the Sharpe you should expect to see
    from pure luck after searching ``N`` configurations.

    This grows without bound in ``N``: 100 trials of noise produce an expected best Sharpe
    of about :math:`2.5\sqrt{V}`, 10,000 trials about :math:`3.9\sqrt{V}`.  That is the
    whole argument for keeping an immutable trial log (PLAN.md S3).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if trial_sharpe_std < 0.0:
        raise ValueError(f"trial_sharpe_std must be >= 0, got {trial_sharpe_std}")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    term = ((1.0 - g) * _norm_ppf_scalar(1.0 - 1.0 / n_trials)
            + g * _norm_ppf_scalar(1.0 - 1.0 / (n_trials * math.e)))
    return float(trial_sharpe_std * term)


def deflated_sharpe_ratio(sharpe: float, n_trials: int, skew: float, kurtosis: float,
                          n_obs: int, *, trial_sharpe_std: float | None = None,
                          periods: int = 1) -> float:
    r"""Deflated Sharpe Ratio — Bailey & Lopez de Prado (2014).

    Bailey, D. H. & Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting, and Non-Normality", *Journal of Portfolio
    Management* 40(5):94-107.
    (https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

    .. math::

        \widehat{DSR} = \widehat{PSR}(SR_0) = Z\!\left[
            \frac{(\widehat{SR} - SR_0)\sqrt{T-1}}
                 {\sqrt{1 - \hat\gamma_3\widehat{SR} + \frac{\hat\gamma_4-1}{4}\widehat{SR}^2}}
        \right]

    i.e. the Probabilistic Sharpe Ratio benchmarked not against zero but against
    :math:`SR_0`, the Sharpe a *lucky* search of ``n_trials`` configurations would be
    expected to produce (:func:`expected_max_sharpe`).  It answers one question only:
    **is this Sharpe larger than the best of N coin flips?**  The project's gate is
    DSR > 0.95 (PLAN.md S3, RESEARCH.md section 4).

    .. warning::

        **The DSR does not detect look-ahead bias, and must never be used as evidence that
        a backtest is leak-free.**  ``docs/RESEARCH.md`` section 4 and
        ``docs/research/01_rl_trading.md`` finding 3 record the measurement from
        arXiv:2608.27734 ("What survives honest evaluation?"), experiment E1: a
        deliberately planted look-ahead oracle scored **design Sharpe 34.7, evaluation
        Sharpe 51.5 and DSR = 1.00**, and PBO cleared it too.  The statistic certified a
        cheat, because deflation corrects for the *number of trials*, not for a
        *contaminated information set*.  Only structural controls remove leakage:
        point-in-time data (S2), ``t+1`` fills, a feature registry, and the planted-oracle
        test in the S3 exit criteria.  ``tests/test_stats.py::test_dsr_does_not_catch_lookahead``
        reproduces the failure and asserts it, so this limitation stays visible.

    Parameters
    ----------
    sharpe : float
        The **selected** (best-of-search) Sharpe ratio.  Per observation unless
        ``periods`` is given.
    n_trials : int
        The honest number of configurations evaluated — every seed, reward, look-back,
        algorithm and HPO sample, from the immutable trial log.  Under-reporting this is
        how a DSR is faked.
    skew, kurtosis : float
        Sample skewness and **non-excess** kurtosis of the selected strategy's returns.
    n_obs : int
        Length ``T`` of the selected strategy's return series.
    trial_sharpe_std : float or None, keyword-only
        :math:`\sqrt{V}`, the standard deviation of the Sharpe ratios **across trials**,
        in the same per-observation frequency.  ``None`` falls back to the asymptotic
        null value :math:`1/\sqrt{T}` (the standard error of a Sharpe estimate when the
        true Sharpe is zero and returns are i.i.d. Gaussian).  That fallback is an
        approximation: when the trial log exists, pass the measured dispersion, which is
        normally larger and therefore deflates harder.
    periods : int
        If ``sharpe`` (and ``trial_sharpe_std``) are annualised with this factor, they are
        divided by ``sqrt(periods)`` first.  Default ``1`` = already per-observation.

    Returns
    -------
    float
        Probability in [0, 1] that the true Sharpe exceeds what selection bias alone
        would deliver.  Above 0.95 the result clears the project's deflation gate — which
        says nothing whatsoever about leakage.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    scale = math.sqrt(periods)
    sr = sharpe / scale
    if trial_sharpe_std is None:
        v_std = 1.0 / math.sqrt(n_obs)
    else:
        v_std = trial_sharpe_std / scale
    sr0 = expected_max_sharpe(n_trials, v_std)
    return probabilistic_sharpe_ratio(sr, sr0, skew, kurtosis, n_obs)


# --------------------------------------------------------------------------------------
# 3. Probability of Backtest Overfitting (CSCV)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PBOResult:
    """Output of :func:`probability_of_backtest_overfitting`."""

    pbo: float
    logits: np.ndarray = field(repr=False)
    oos_relative_ranks: np.ndarray = field(repr=False)
    selected_trial: np.ndarray = field(repr=False)
    is_performance: np.ndarray = field(repr=False)
    oos_performance: np.ndarray = field(repr=False)
    n_combinations: int = 0
    n_splits: int = 0
    n_trials: int = 0
    n_obs_used: int = 0
    degradation_slope: float = float("nan")
    prob_oos_loss: float = float("nan")

    def __str__(self) -> str:  # pragma: no cover - reporting convenience
        return (f"PBO={self.pbo:.3f} over {self.n_combinations} CSCV splits "
                f"(S={self.n_splits}, N={self.n_trials}); "
                f"OOS degradation slope {self.degradation_slope:.3f}, "
                f"P(OOS loss)={self.prob_oos_loss:.3f}")


def _sharpe_columns(mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(var > 0.0, mean / np.sqrt(np.where(var > 0.0, var, 1.0)), -np.inf)
    return out


def probability_of_backtest_overfitting(matrix_of_trial_returns, n_splits: int = 16,
                                        performance=None) -> PBOResult:
    r"""Probability of Backtest Overfitting by Combinatorially Symmetric Cross-Validation.

    Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2017), "The Probability
    of Backtest Overfitting", *Journal of Computational Finance* 20(4):39-69
    (SSRN 2326253; https://davidhbailey.com/dhbpapers/backtest-prob.pdf).

    The CSCV procedure, as published:

    1.  Form the ``T x N`` matrix ``M`` of per-period returns: one column per trial
        (configuration), all columns covering the same ``T`` observations.
    2.  Partition the rows into ``S`` disjoint contiguous submatrices of equal length
        (``S`` even; the paper uses ``S = 16``).
    3.  For each of the :math:`\binom{S}{S/2}` ways of choosing ``S/2`` submatrices as the
        in-sample set ``J``, the remaining ``S/2`` form the out-of-sample set
        :math:`\bar J`.
    4.  Pick :math:`n^* = \arg\max_n \mathrm{SR}_n(J)`, the in-sample winner.
    5.  Compute its **relative rank** among all ``N`` trials out of sample,
        :math:`\omega_c = \mathrm{rank}(n^*) / (N+1) \in (0,1)`, and the logit
        :math:`\lambda_c = \log\frac{\omega_c}{1-\omega_c}`.
    6.  :math:`PBO = P(\lambda \le 0)` — the fraction of splits in which the in-sample
        best strategy lands **below the out-of-sample median**.

    Interpretation.  ``PBO ~ 0.5`` means the selection procedure has no skill: the winner
    of the search is a coin flip out of sample.  That is the *expected* value when all
    trials are equally worthless, which makes it the honest null, not a failure of the
    statistic.  ``PBO`` near 0 means the search finds something real; near 1 means the
    search reliably picks the *worst* strategies out of sample.  RESEARCH.md section 4
    records PBO = 0.83 for a nine-trial classic-factor grid on a point-in-time US panel.

    Like the DSR, this measures **overfitting by search**.  It does not and cannot detect
    look-ahead leakage: a leaked column wins in sample *and* out of sample, so its CSCV
    rank is top and its PBO is ~0.

    Parameters
    ----------
    matrix_of_trial_returns : array_like, shape (T, N)
        Per-period returns, rows = time (chronological), columns = trials.  The trailing
        ``T mod n_splits`` rows are dropped so that the blocks are of equal length.
    n_splits : int
        ``S``, the number of submatrices.  Must be even and at least 4.  The number of
        CSCV combinations is ``comb(S, S/2)``: 252 at S=10, 12,870 at S=16, 184,756 at
        S=20.
    performance : callable or None
        ``f(block) -> array of length N`` scoring each column of a ``(rows, N)`` slice.
        ``None`` (default) uses the per-observation Sharpe ratio ``mean/std`` with the
        maximum-likelihood variance, which is what the paper uses.  A custom callable
        forces a slower explicit loop over all combinations.
    """
    m = np.asarray(matrix_of_trial_returns, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D (T, N) matrix, got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix_of_trial_returns contains non-finite values")
    t_all, n_trials = m.shape
    if n_trials < 2:
        raise ValueError(f"CSCV needs at least 2 trials to rank, got {n_trials}")
    if n_splits < 4 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be an even integer >= 4, got {n_splits}")
    block_len = t_all // n_splits
    if block_len < 2:
        raise ValueError(
            f"{t_all} observations cannot be split into {n_splits} blocks of >= 2 rows")
    used = block_len * n_splits
    m = m[:used]

    half = n_splits // 2
    combos = list(itertools.combinations(range(n_splits), half))
    n_comb = len(combos)
    mask = np.zeros((n_comb, n_splits), dtype=float)
    for i, c in enumerate(combos):
        mask[i, list(c)] = 1.0
    comp = 1.0 - mask

    if performance is None:
        # Exact Sharpe from block sums: SR = mean / sqrt(E[x^2] - mean^2).
        centre = m.mean(axis=0)
        cm = m - centre
        blocks = cm.reshape(n_splits, block_len, n_trials)
        b_sum = blocks.sum(axis=1)
        b_sumsq = (blocks ** 2).sum(axis=1)
        cnt = float(half * block_len)

        def _score(sel: np.ndarray) -> np.ndarray:
            s1 = sel @ b_sum
            s2 = sel @ b_sumsq
            mean_c = s1 / cnt
            var = s2 / cnt - mean_c ** 2
            var = np.maximum(var, 0.0)
            return _sharpe_columns(mean_c + centre, var)

        is_perf = _score(mask)
        oos_perf = _score(comp)
    else:
        blocks = m.reshape(n_splits, block_len, n_trials)
        is_perf = np.empty((n_comb, n_trials))
        oos_perf = np.empty((n_comb, n_trials))
        for i, c in enumerate(combos):
            other = [s for s in range(n_splits) if s not in c]
            is_perf[i] = np.asarray(performance(blocks[list(c)].reshape(-1, n_trials)))
            oos_perf[i] = np.asarray(performance(blocks[other].reshape(-1, n_trials)))

    if not np.all(np.isfinite(is_perf)) or not np.all(np.isfinite(oos_perf)):
        raise ValueError("a trial has zero variance in some split; its Sharpe is undefined")

    best = np.argmax(is_perf, axis=1)
    rows = np.arange(n_comb)
    oos_star = oos_perf[rows, best]
    # Ascending rank in 1..N with mid-ranks for ties, then the paper's omega = rank/(N+1).
    lower = np.sum(oos_perf < oos_star[:, None], axis=1)
    equal = np.sum(oos_perf == oos_star[:, None], axis=1)
    rank = lower + 0.5 * (equal + 1.0)
    omega = rank / (n_trials + 1.0)
    logits = np.log(omega / (1.0 - omega))
    pbo = float(np.mean(logits <= 0.0))

    is_star = is_perf[rows, best]
    slope = float("nan")
    if np.std(is_star) > 0:
        slope = float(np.polyfit(is_star, oos_star, 1)[0])
    return PBOResult(pbo=pbo, logits=logits, oos_relative_ranks=omega, selected_trial=best,
                     is_performance=is_perf, oos_performance=oos_perf,
                     n_combinations=n_comb, n_splits=n_splits, n_trials=n_trials,
                     n_obs_used=used, degradation_slope=slope,
                     prob_oos_loss=float(np.mean(oos_star <= 0.0)))


# --------------------------------------------------------------------------------------
# 4. Diebold-Mariano test of equal predictive accuracy
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DMResult:
    """Output of :func:`diebold_mariano`.  ``statistic > 0`` means model A loses *more*."""

    statistic: float
    p_value: float
    mean_loss_differential: float
    long_run_variance: float
    lag: int
    n_obs: int
    df: int
    harvey_correction: float
    alternative: str
    better: str

    def __str__(self) -> str:  # pragma: no cover - reporting convenience
        return (f"DM*={self.statistic:.3f}, p={self.p_value:.4g} ({self.alternative}, "
                f"df={self.df}, lag={self.lag}) -> better: {self.better}")


_LOSSES = {
    "squared": lambda e: e ** 2,
    "absolute": lambda e: np.abs(e),
}


def diebold_mariano(errors_a, errors_b, h: int = 1, loss="squared",
                    lag: int | None = None, weights: str = "bartlett",
                    alternative: str = "two-sided") -> DMResult:
    r"""Diebold-Mariano test of equal predictive accuracy, with the HLN correction.

    Diebold, F. X. & Mariano, R. S. (1995), "Comparing Predictive Accuracy", *Journal of
    Business & Economic Statistics* 13(3):253-263.
    Harvey, D., Leybourne, S. & Newbold, P. (1997), "Testing the Equality of Prediction
    Mean Squared Errors", *International Journal of Forecasting* 13(2):281-291.
    Newey, W. K. & West, K. D. (1987), *Econometrica* 55(3):703-708 (HAC variance).

    With loss differential :math:`d_t = L(e_{a,t}) - L(e_{b,t})` and
    :math:`\bar d = T^{-1}\sum_t d_t`, the DM statistic is

    .. math::

        DM = \frac{\bar d}{\sqrt{\hat V / T}},
        \qquad
        \hat V = \hat\gamma_0 + 2\sum_{k=1}^{L} w_k\,\hat\gamma_k,
        \qquad
        \hat\gamma_k = \frac1T\sum_{t=k+1}^{T}(d_t-\bar d)(d_{t-k}-\bar d)

    Diebold & Mariano truncate at :math:`L = h-1` with uniform weights
    (:math:`w_k = 1`), because an optimal :math:`h`-step-ahead forecast error is at most
    MA(h-1).  ``weights="bartlett"`` (the default) instead applies the Newey-West kernel
    :math:`w_k = 1 - k/(L+1)`, which guarantees a non-negative variance estimate; the two
    coincide when ``h = 1``.

    Harvey, Leybourne & Newbold's small-sample correction multiplies the statistic by

    .. math::

        \sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}}

    and compares the result to a Student-t distribution with ``T - 1`` degrees of freedom
    rather than to a standard normal.  Without it the test over-rejects badly at the
    sample sizes an equity backtest actually has; ``tests/test_stats.py`` measures that
    over-rejection instead of asserting it.

    A useful identity, used as a known-answer test: at ``h = 1`` the HLN-corrected
    statistic is **exactly** the one-sample t statistic of ``d``, because the
    :math:`\sqrt{(T-1)/T}` factor converts the maximum-likelihood variance into the
    unbiased one.

    Parameters
    ----------
    errors_a, errors_b : array_like
        Forecast errors (actual - forecast) of the two models over the **same** periods.
        For strategy comparison in this harness the natural inputs are the per-period
        deviations from the target the harness is scoring.
    h : int
        Forecast horizon in periods.  Drives both the truncation lag and the HLN factor.
    loss : {"squared", "absolute"} or callable
        Loss function applied element-wise to each error series.
    lag : int or None
        Override the truncation lag ``L``.  ``None`` uses the DM default ``h - 1``.
    weights : {"bartlett", "uniform"}
        HAC kernel; see above.
    alternative : {"two-sided", "a-better", "b-better"}
        ``"a-better"`` tests H1: model A has strictly lower expected loss.

    Notes
    -----
    Identical error series give :math:`d_t \equiv 0` and therefore a zero variance.  That
    is not an error condition: the models *are* equally accurate, so the function returns
    ``statistic = 0`` and ``p = 1``.  A non-degenerate series with a non-positive variance
    estimate raises instead of silently returning a number.
    """
    ea = _as_1d(errors_a, "errors_a")
    eb = _as_1d(errors_b, "errors_b")
    if ea.size != eb.size:
        raise ValueError(f"error series differ in length: {ea.size} vs {eb.size}")
    n = ea.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if h > n:
        raise ValueError(f"horizon h={h} exceeds the sample length {n}")
    if alternative not in ("two-sided", "a-better", "b-better"):
        raise ValueError(f"unknown alternative {alternative!r}")
    if callable(loss):
        lfun = loss
    elif loss in _LOSSES:
        lfun = _LOSSES[loss]
    else:
        raise ValueError(f"unknown loss {loss!r}; use {sorted(_LOSSES)} or a callable")

    d = np.asarray(lfun(ea), dtype=float) - np.asarray(lfun(eb), dtype=float)
    if d.shape != ea.shape:
        raise ValueError("the loss function must be element-wise")
    dbar = float(d.mean())
    lag = h - 1 if lag is None else int(lag)
    if lag < 0 or lag >= n:
        raise ValueError(f"lag must be in [0, {n - 1}], got {lag}")

    correction = math.sqrt((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n)
    df = n - 1

    if np.all(d == 0.0):
        return DMResult(statistic=0.0, p_value=1.0, mean_loss_differential=0.0,
                        long_run_variance=0.0, lag=lag, n_obs=n, df=df,
                        harvey_correction=correction, alternative=alternative,
                        better="tie")

    dev = d - dbar
    gamma0 = float(dev @ dev) / n
    v = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0) if weights == "bartlett" else 1.0
        if weights not in ("bartlett", "uniform"):
            raise ValueError(f"unknown weights {weights!r}")
        v += 2.0 * w * float(dev[k:] @ dev[:-k]) / n
    if v <= 0.0:
        raise ValueError(
            f"long-run variance estimate is non-positive ({v:.4g}); use "
            'weights="bartlett", which is positive semi-definite by construction')
    if n + 1.0 - 2.0 * h + h * (h - 1.0) / n <= 0.0:
        raise ValueError(f"the HLN correction is undefined for h={h} at T={n}")

    stat = dbar / math.sqrt(v / n) * correction
    if alternative == "two-sided":
        p = 2.0 * _t_sf(abs(stat), df)
    elif alternative == "a-better":
        p = _t_sf(-stat, df)
    else:
        p = _t_sf(stat, df)
    p = min(1.0, max(0.0, p))
    better = "a" if dbar < 0 else ("b" if dbar > 0 else "tie")
    return DMResult(statistic=float(stat), p_value=float(p), mean_loss_differential=dbar,
                    long_run_variance=float(v), lag=lag, n_obs=n, df=df,
                    harvey_correction=correction, alternative=alternative, better=better)


# --------------------------------------------------------------------------------------
# 5. Stationary bootstrap
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapCI:
    """A bootstrap point estimate with its interval and the settings that produced it."""

    estimate: float
    ci_low: float
    ci_high: float
    alpha: float
    n_boot: int
    block_size: float
    p_value: float = float("nan")
    replicates: np.ndarray = field(default=None, repr=False)

    def excludes_zero(self) -> bool:
        """The project's gate wording: does the interval exclude zero?"""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def __str__(self) -> str:  # pragma: no cover - reporting convenience
        return (f"{self.estimate:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}] "
                f"({100 * (1 - self.alpha):.0f}% stationary bootstrap, b={self.block_size}, "
                f"B={self.n_boot})")


def stationary_bootstrap(x, block_size: float, n_boot: int, rng=None) -> np.ndarray:
    r"""Politis & Romano (1994) stationary bootstrap resampling.

    Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *Journal of the
    American Statistical Association* 89(428):1303-1313.

    The resampled series is built from blocks of **geometric** length: starting from a
    uniformly drawn index :math:`I_1 \sim U\{0,\dots,n-1\}`, at each step

    .. math::

        I_{t+1} = \begin{cases}
            U\{0,\dots,n-1\} & \text{with probability } p \\
            (I_t + 1) \bmod n & \text{with probability } 1-p
        \end{cases}
        \qquad p = 1/b

    so block lengths are :math:`\mathrm{Geom}(p)` with mean ``b = block_size`` and the
    data are wrapped circularly.  Unlike the fixed-block bootstrap this makes the
    resampled series **stationary**, which is the whole point of the paper.

    Why the harness needs it: daily strategy returns are serially dependent (positions
    persist, volatility clusters).  An i.i.d. bootstrap destroys that dependence and
    produces confidence intervals that are too narrow — the test suite measures the
    under-coverage on an AR(1) series rather than asserting it.

    Parameters
    ----------
    x : array_like, shape (n,) or (n, k)
        The series to resample.  A 2-D array is resampled **jointly by row**, which is
        how you preserve the cross-correlation between two strategies' returns when
        bootstrapping a Sharpe *difference*.
    block_size : float
        Mean block length ``b >= 1``.  ``b = 1`` degenerates to the i.i.d. bootstrap.
        Politis & Romano's optimal ``b`` grows like :math:`n^{1/3}`; for daily equity
        returns the harness default is around 20 (one trading month).
    n_boot : int
        Number of bootstrap replicates ``B``.
    rng : numpy.random.Generator, int or None
        Seeded for reproducibility — PLAN.md S3 requires bit-identical metrics from the
        same seed.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_boot, n)`` for 1-D input, ``(n_boot, n, k)`` for 2-D input.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D or 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains non-finite values")
    n = arr.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}")
    if block_size < 1.0:
        raise ValueError(f"block_size (mean block length) must be >= 1, got {block_size}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    p = 1.0 / float(block_size)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = gen.integers(0, n, size=n_boot)
    if n > 1:
        jump = gen.random((n_boot, n - 1)) < p
        fresh = gen.integers(0, n, size=(n_boot, n - 1))
        for t in range(1, n):
            cont = (idx[:, t - 1] + 1) % n
            idx[:, t] = np.where(jump[:, t - 1], fresh[:, t - 1], cont)
    return arr[idx]


def stationary_bootstrap_variance(x, block_size: float) -> float:
    r"""Exact variance of the stationary-bootstrap sample mean, Politis & Romano (1994).

    For the circular scheme above, :math:`E^*[\bar x^*] = \bar x` exactly and

    .. math::

        \mathrm{Var}^*(\sqrt{n}\,\bar x^*) = \tilde R(0)
            + 2\sum_{k=1}^{n-1} b_n(k)\,\tilde R(k),
        \qquad
        b_n(k) = \Big(1-\frac{k}{n}\Big)(1-p)^k + \frac{k}{n}(1-p)^{n-k}

    where :math:`\tilde R(k) = n^{-1}\sum_{i=1}^{n}(x_i-\bar x)(x_{i+k}-\bar x)` is the
    **circular** autocovariance (indices wrap).  This is the population variance of the
    resampling distribution, so simulating :func:`stationary_bootstrap` must converge to
    it — the test suite uses that as a known-answer check on the sampler itself.

    Returns the variance of :math:`\bar x^*` (not of :math:`\sqrt n \bar x^*`).
    """
    arr = _as_1d(x, "x")
    n = arr.size
    if block_size < 1.0:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    p = 1.0 / float(block_size)
    dev = arr - arr.mean()
    # circular autocovariances via the wrapped product
    r = np.array([float(dev @ np.roll(dev, -k)) / n for k in range(n)])
    k = np.arange(1, n)
    w = (1.0 - k / n) * (1.0 - p) ** k + (k / n) * (1.0 - p) ** (n - k)
    total = r[0] + 2.0 * float(np.sum(w * r[1:]))
    return total / n


def stationary_bootstrap_ci(x, statistic, block_size: float, n_boot: int = 1000,
                            alpha: float = 0.05, rng=None,
                            method: str = "percentile") -> BootstrapCI:
    """Confidence interval for ``statistic`` under the stationary bootstrap.

    Politis & Romano (1994) for the resampling; Efron & Tibshirani (1993), *An
    Introduction to the Bootstrap*, chapter 13 for the percentile and basic (reverse
    percentile) intervals.

    Parameters
    ----------
    statistic : callable
        ``f(sample) -> float`` applied to the original data and to every replicate.
        ``sample`` has the same shape as ``x``.
    method : {"percentile", "basic"}
        ``"percentile"``: ``[q_{alpha/2}, q_{1-alpha/2}]`` of the replicates.
        ``"basic"``: ``[2*theta - q_{1-alpha/2}, 2*theta - q_{alpha/2}]``, which corrects
        the direction of the bias but can leave the parameter's natural range.

    The reported ``p_value`` is the two-sided bootstrap p-value for H0: statistic = 0,
    ``2 * min(P*(theta* <= 0), P*(theta* >= 0))``, clipped to 1.  It is an approximation
    (it inverts the percentile interval) and is reported alongside the interval, not
    instead of it.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if method not in ("percentile", "basic"):
        raise ValueError(f"unknown method {method!r}")
    arr = np.asarray(x, dtype=float)
    theta = float(statistic(arr))
    reps = stationary_bootstrap(arr, block_size, n_boot, rng=rng)
    vals = np.array([float(statistic(reps[i])) for i in range(n_boot)])
    if not np.all(np.isfinite(vals)):
        raise ValueError("the statistic returned a non-finite value on some replicate")
    lo_q, hi_q = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])
    if method == "percentile":
        lo, hi = float(lo_q), float(hi_q)
    else:
        lo, hi = float(2.0 * theta - hi_q), float(2.0 * theta - lo_q)
    p = 2.0 * min(float(np.mean(vals <= 0.0)), float(np.mean(vals >= 0.0)))
    return BootstrapCI(estimate=theta, ci_low=lo, ci_high=hi, alpha=alpha, n_boot=n_boot,
                       block_size=float(block_size), p_value=min(1.0, p), replicates=vals)


def sharpe_difference_ci(returns_a, returns_b, block_size: float = 20.0,
                         n_boot: int = 1000, alpha: float = 0.05, periods: int = 252,
                         rf: float = 0.0, rng=None, method: str = "percentile") -> BootstrapCI:
    """Confidence interval for ``sharpe(a) - sharpe(b)`` under the stationary bootstrap.

    The two return series are resampled **jointly** (the same index path is applied to
    both columns), so the cross-correlation between the strategies is preserved.  This is
    the interval the S3 harness reports for "agent minus buy-and-hold": the comparison is
    paired, since both arms see the same market days.

    Politis & Romano (1994) for the resampling; Ledoit & Wolf (2008), "Robust performance
    hypothesis testing with the Sharpe ratio", *Journal of Empirical Finance* 15(5):850-859,
    for the argument that a time-series bootstrap is the right tool for Sharpe differences
    under serial dependence.
    """
    a = _as_1d(returns_a, "returns_a")
    b = _as_1d(returns_b, "returns_b")
    if a.size != b.size:
        raise ValueError(f"paired series differ in length: {a.size} vs {b.size}")
    paired = np.column_stack([a, b])

    def _diff(sample: np.ndarray) -> float:
        return (sharpe(sample[:, 0], rf=rf, periods=periods)
                - sharpe(sample[:, 1], rf=rf, periods=periods))

    return stationary_bootstrap_ci(paired, _diff, block_size=block_size, n_boot=n_boot,
                                   alpha=alpha, rng=rng, method=method)


# --------------------------------------------------------------------------------------
# 6. Benjamini-Hochberg false discovery rate control
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BHResult:
    """Output of :func:`benjamini_hochberg`."""

    rejected: np.ndarray = field(repr=False)
    adjusted_pvalues: np.ndarray = field(repr=False)
    n_rejected: int = 0
    threshold: float = 0.0
    alpha: float = 0.05
    n_tests: int = 0
    method: str = "bh"

    def __str__(self) -> str:  # pragma: no cover - reporting convenience
        return (f"{self.n_rejected}/{self.n_tests} rejected at FDR<={self.alpha} "
                f"({self.method}); largest rejected p = {self.threshold:.4g}")


def benjamini_hochberg(pvalues, alpha: float = 0.05,
                       dependency: str = "independent") -> BHResult:
    r"""Benjamini-Hochberg step-up procedure for false discovery rate control.

    Benjamini, Y. & Hochberg, Y. (1995), "Controlling the False Discovery Rate: A
    Practical and Powerful Approach to Multiple Testing", *Journal of the Royal
    Statistical Society B* 57(1):289-300.

    Order the p-values :math:`p_{(1)} \le \dots \le p_{(m)}`, let

    .. math::

        k = \max\Big\{ i : p_{(i)} \le \frac{i}{m}\,q^* \Big\}

    and reject :math:`H_{(1)},\dots,H_{(k)}`.  The procedure controls the expected
    proportion of false discoveries among the rejections at :math:`\frac{m_0}{m}q^* \le q^*`
    for independent test statistics (BH 1995, Theorem 1).

    It is a **step-up** rule: a hypothesis with :math:`p > \alpha/m` (which Bonferroni
    would keep) is still rejected if enough smaller p-values sit below their own
    thresholds.  That is why it has more power than Bonferroni while still controlling a
    meaningful error rate.

    Adjusted p-values (BH "q-values") are the monotone step-up transform

    .. math::  \tilde p_{(i)} = \min_{j \ge i}\Big\{\min\Big(1, \frac{m}{j}p_{(j)}\Big)\Big\}

    so that ``adjusted_pvalues <= alpha`` reproduces the rejection set exactly.

    Why the harness needs it: RESEARCH.md correction 6 records a sentiment signal whose
    best 1-day Rank IC of 0.0143 survived neither Newey-West nor FDR correction.  When 5
    tickers x 3 cost levels x several feature sets are each tested, an uncorrected 5%
    threshold manufactures discoveries at exactly that rate — the test suite measures the
    uncorrected false-discovery count rather than assuming it.

    Parameters
    ----------
    pvalues : array_like
        p-values in [0, 1].  Shape is preserved in the outputs.
    alpha : float
        Target FDR level :math:`q^*`.
    dependency : {"independent", "arbitrary"}
        ``"arbitrary"`` divides the thresholds by :math:`c(m)=\sum_{i=1}^m 1/i`, giving the
        Benjamini & Yekutieli (2001) procedure (*Annals of Statistics* 29(4):1165-1188),
        which controls the FDR under **any** dependence structure.  Strategy tests on
        overlapping windows and correlated tickers are dependent, so this is the
        conservative option available when that matters.
    """
    p = np.asarray(pvalues, dtype=float)
    flat = p.ravel()
    if flat.size == 0:
        raise ValueError("pvalues is empty")
    if not np.all(np.isfinite(flat)):
        raise ValueError("pvalues contains non-finite values")
    if np.any(flat < 0.0) or np.any(flat > 1.0):
        raise ValueError("pvalues must lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if dependency not in ("independent", "arbitrary"):
        raise ValueError(f"unknown dependency {dependency!r}")

    m = flat.size
    cm = 1.0 if dependency == "independent" else float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(flat, kind="stable")
    sorted_p = flat[order]
    ranks = np.arange(1, m + 1)
    below = sorted_p <= ranks / m * alpha / cm
    k = int(np.max(np.nonzero(below)[0]) + 1) if np.any(below) else 0

    adjusted_sorted = np.minimum.accumulate((m * cm / ranks * sorted_p)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    rejected_sorted = np.zeros(m, dtype=bool)
    rejected_sorted[:k] = True
    rejected = np.empty(m, dtype=bool)
    rejected[order] = rejected_sorted

    return BHResult(rejected=rejected.reshape(p.shape),
                    adjusted_pvalues=adjusted.reshape(p.shape),
                    n_rejected=k, threshold=float(sorted_p[k - 1]) if k else 0.0,
                    alpha=alpha, n_tests=m,
                    method="bh" if dependency == "independent" else "by")


# --------------------------------------------------------------------------------------
# the S3 verdict: all three gates, or no claim
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Verdict:
    """The three-gate S3 verdict.  ``passed`` is the conjunction, never a single gate."""

    passed: bool
    dm_significant: bool
    dsr_above_threshold: bool
    lower_bound_positive: bool
    dm_p_value: float
    dsr: float
    seed_lower_bound: float
    detail: str

    def __str__(self) -> str:  # pragma: no cover - reporting convenience
        return f"{'PASS' if self.passed else 'FAIL'} - {self.detail}"


def three_gate_verdict(dm: DMResult, dsr: float, seed_lower_bound: float,
                       dm_alpha: float = 0.05, dsr_threshold: float = 0.95) -> Verdict:
    """The pre-registered S3 verdict rule, as written in PLAN.md / RESEARCH.md section 4.

    A result counts as an edge only if **all three** hold:

    1.  the Diebold-Mariano test rejects equal accuracy against the baseline
        (``dm.p_value < dm_alpha``) *and* the point estimate favours the challenger;
    2.  the Deflated Sharpe Ratio over the honest trial count exceeds ``dsr_threshold``
        (0.95);
    3.  the across-seed bootstrap lower bound on the metric is above zero.

    None of the three detects look-ahead leakage.  That is the planted-oracle test's job,
    not this function's.
    """
    gate_dm = bool(dm.p_value < dm_alpha and dm.better == "a")
    gate_dsr = bool(dsr > dsr_threshold)
    gate_lb = bool(seed_lower_bound > 0.0)
    passed = gate_dm and gate_dsr and gate_lb
    detail = (f"DM p={dm.p_value:.4g} (better={dm.better}, need p<{dm_alpha} and better=a) "
              f"{'OK' if gate_dm else 'FAIL'}; "
              f"DSR={dsr:.3f} (need >{dsr_threshold}) {'OK' if gate_dsr else 'FAIL'}; "
              f"seed lower bound={seed_lower_bound:.4g} (need >0) "
              f"{'OK' if gate_lb else 'FAIL'}")
    return Verdict(passed=passed, dm_significant=gate_dm, dsr_above_threshold=gate_dsr,
                   lower_bound_positive=gate_lb, dm_p_value=dm.p_value, dsr=dsr,
                   seed_lower_bound=seed_lower_bound, detail=detail)
