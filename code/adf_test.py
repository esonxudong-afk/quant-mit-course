"""
ADF & KPSS Stationarity Tests
==============================
Augmented Dickey-Fuller (ADF) test and KPSS test for determining whether
a time series is stationary or needs differencing.

The ADF test has H₀: unit root (non-stationary).
The KPSS test has H₀: stationary (trend-stationary or level-stationary).
Running both jointly gives a more robust stationarity assessment.

References:
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators
  for autoregressive time series with a unit root.
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992).
  Testing the null hypothesis of stationarity against the alternative
  of a unit root.
- MacKinnon, J. G. (1994). Approximate asymptotic distribution functions
  for unit-root and cointegration tests.

Usage:
    from time_series.adf_test import ADFTest, KPSSTest, check_stationarity

    adf = ADFTest(series)
    print(adf.report())

    result = check_stationarity(series)
    print(result["conclusion"])
"""

import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════
# MacKinnon p-value approximation for ADF test
# ═══════════════════════════════════════════════════════════════════════

def _mackinnon_p_value(stat: float, regression: str = "c", nobs: int = 100) -> float:
    """Approximate p-value for ADF test statistic using MacKinnon (1994) tables.

    Uses the response-surface coefficients for the 1%, 5%, 10% critical values
    and interpolates via a fitted cumulative distribution function.

    Args:
        stat: The ADF test statistic (t-statistic on γ in Δy_t = α + γ y_{t-1} + ...).
        regression: Type of regression — "c" (constant), "ct" (constant+trend), "nc" (none).
        nobs: Number of observations (used for finite-sample adjustment).

    Returns:
        Approximate two-sided p-value in [0, 1].
    """
    # MacKinnon (1994) response surface coefficients for ADF t-statistic
    # Format: {regression: {pct: (beta_inf, beta_1, beta_2)}}
    # Model: C(p) = beta_inf + beta_1/T + beta_2/T^2
    _mackinnon_coeffs = {
        "c": {
            0.01: (-3.4335, -5.999, -29.25),
            0.05: (-2.8621, -2.738, -8.36),
            0.10: (-2.5671, -1.438, -4.48),
        },
        "ct": {
            0.01: (-3.9638, -8.353, -47.44),
            0.05: (-3.4126, -4.039, -17.83),
            0.10: (-3.1279, -2.418, -7.58),
        },
        "nc": {
            0.01: (-2.5658, -1.960, -10.04),
            0.05: (-1.9393, -0.398, 0.0),
            0.10: (-1.6156, -0.181, 0.0),
        },
    }

    T = float(nobs)
    coeffs = _mackinnon_coeffs.get(regression, _mackinnon_coeffs["c"])

    # Compute critical values at 1%, 5%, 10%
    crit_01 = coeffs[0.01][0] + coeffs[0.01][1] / T + coeffs[0.01][2] / (T ** 2)
    crit_05 = coeffs[0.05][0] + coeffs[0.05][1] / T + coeffs[0.05][2] / (T ** 2)
    crit_10 = coeffs[0.10][0] + coeffs[0.10][1] / T + coeffs[0.10][2] / (T ** 2)

    # Fit a quadratic polynomial: stat ~ a + b*p + c*p^2 through the 3 points
    # p-values for critical values: 0.01, 0.05, 0.10
    p_points = np.array([0.01, 0.05, 0.10])
    s_points = np.array([crit_01, crit_05, crit_10])

    # Build a 3-point interpolation / extrapolation using polynomial fit
    coeffs_poly = np.polyfit(p_points, s_points, 2)  # s = c2*p^2 + c1*p + c0

    c2, c1, c0 = coeffs_poly

    # For a given stat, solve c2*p^2 + c1*p + c0 - stat = 0 for p
    # Use the inverse: find the p where fitted polynomial equals stat
    discriminant = c1 ** 2 - 4 * c2 * (c0 - stat)

    if discriminant < 0:
        # Statistic is more extreme than our fitted curve can handle
        if stat < min(s_points):
            return 0.0001  # very significant
        else:
            return 0.9999  # not significant at all

    sqrt_disc = np.sqrt(discriminant)
    p1 = (-c1 + sqrt_disc) / (2 * c2)
    p2 = (-c1 - sqrt_disc) / (2 * c2)

    # Pick the root in [0, 1]
    if 0 <= p1 <= 1:
        p_val = p1
    elif 0 <= p2 <= 1:
        p_val = p2
    else:
        # Fall back to linear interpolation between closest points
        if stat <= crit_01:
            p_val = 0.0
        elif stat <= crit_05:
            # Linear between p=0.01 and p=0.05
            p_val = 0.01 + 0.04 * (stat - crit_01) / (crit_05 - crit_01)
        elif stat <= crit_10:
            p_val = 0.05 + 0.05 * (stat - crit_05) / (crit_10 - crit_05)
        else:
            p_val = 0.10 + 0.90 * (stat - crit_10) / (crit_10 - crit_05)

    return float(np.clip(p_val, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════
# KPSS critical values (Kwiatkowski et al. 1992, Table 1)
# ═══════════════════════════════════════════════════════════════════════

_KPSS_CRITICAL_VALUES = {
    # (eta_mu, eta_tau) for level-stationary and trend-stationary
    "eta_mu":  {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739},
    "eta_tau": {0.10: 0.119, 0.05: 0.146, 0.025: 0.176, 0.01: 0.216},
}


# ═══════════════════════════════════════════════════════════════════════
# ADF Test
# ═══════════════════════════════════════════════════════════════════════

class ADFTest:
    """Augmented Dickey-Fuller test for unit root (non-stationarity).

    H₀: The series has a unit root (is non-stationary).
    H₁: The series is stationary.

    Regression: Δy_t = α + βt + γ y_{t-1} + Σ δ_i Δy_{t-i} + ε_t
    The test statistic is the t-statistic on γ.
    """

    def __init__(self, series: np.ndarray, max_lag: int | None = None):
        """
        Args:
            series: 1-D array of observations.
            max_lag: Maximum lag order for augmentation. If None, uses
                     BIC-based automatic selection up to 12*(n/100)^(1/4).
        """
        y = np.asarray(series, dtype=np.float64).flatten()
        if y.ndim != 1:
            raise ValueError("series must be a 1-D array")
        if len(y) < 10:
            raise ValueError("series must have at least 10 observations")

        self._y = y
        self._n = len(y)
        self._dy = np.diff(y)  # first differences

        # Determine max lag for BIC selection
        if max_lag is None:
            self._max_lag = int(np.floor(12 * (self._n / 100) ** 0.25))
            self._max_lag = max(1, min(self._max_lag, self._n // 3))
        else:
            self._max_lag = max(1, int(max_lag))

        # Compute optimal lag via BIC
        self._lag = self._select_lag_bic()
        self._adf_stat = self._compute_adf_statistic()
        self._p_val = _mackinnon_p_value(self._adf_stat, regression="c", nobs=self._n)

    # ── BIC lag selection ──────────────────────────────────────────────

    def _select_lag_bic(self) -> int:
        """Select optimal lag length by minimizing BIC."""
        best_lag = 0
        best_bic = np.inf

        for lag in range(0, self._max_lag + 1):
            bic = self._bic_for_lag(lag)
            if bic < best_bic:
                best_bic = bic
                best_lag = lag

        return best_lag

    def _bic_for_lag(self, lag: int) -> float:
        """Compute BIC for a given lag length."""
        # Construct regression: Δy_t = α + γ y_{t-1} + Σ δ_i Δy_{t-i}
        n_eff = self._n - lag - 1
        if n_eff <= lag + 2:
            return np.inf

        y_lag = self._y[lag:-1]  # y_{t-1}
        dy = self._dy[lag:]      # Δy_t

        # Build design matrix
        X_cols = []
        # Constant
        X_cols.append(np.ones(n_eff))
        # y_{t-1}
        X_cols.append(y_lag)
        # Δy_{t-i} for i=1..lag
        for i in range(1, lag + 1):
            X_cols.append(self._dy[lag - i:-i])

        X = np.column_stack(X_cols)

        try:
            # OLS estimation
            beta, residuals, rank, singular = np.linalg.lstsq(X, dy, rcond=None)
            rss = np.sum(residuals) if residuals.size > 0 else np.sum((dy - X @ beta) ** 2)
            if rss <= 0:
                return np.inf

            k_params = X.shape[1]
            log_lik = -0.5 * n_eff * (np.log(2 * np.pi * rss / n_eff) + 1)
            bic = -2 * log_lik + k_params * np.log(n_eff)
            return float(bic)
        except np.linalg.LinAlgError:
            return np.inf

    # ── ADF statistic computation ──────────────────────────────────────

    def _compute_adf_statistic(self) -> float:
        """Compute the ADF t-statistic on γ (coefficient of y_{t-1})."""
        lag = self._lag
        n_eff = self._n - lag - 1

        y_lag = self._y[lag:-1]
        dy = self._dy[lag:]

        X_cols = [np.ones(n_eff), y_lag]
        for i in range(1, lag + 1):
            X_cols.append(self._dy[lag - i:-i])

        X = np.column_stack(X_cols)

        if X.shape[0] <= X.shape[1]:
            return 0.0  # not enough data

        try:
            beta, residuals, rank, singular = np.linalg.lstsq(X, dy, rcond=None)
            residuals = dy - X @ beta
            rss = np.sum(residuals ** 2)
            df = n_eff - X.shape[1]
            if df <= 0:
                return 0.0
            sigma2 = rss / df

            # Standard error of γ (the second coefficient, index 1)
            XtX_inv = np.linalg.inv(X.T @ X)
            se_gamma = np.sqrt(sigma2 * XtX_inv[1, 1])

            if se_gamma <= 0:
                return 0.0

            t_stat = beta[1] / se_gamma
            return float(t_stat)
        except np.linalg.LinAlgError:
            return 0.0

    # ── public API ─────────────────────────────────────────────────────

    def adf_statistic(self) -> float:
        """Return the ADF test statistic."""
        return self._adf_stat

    def p_value(self) -> float:
        """Return the MacKinnon approximate p-value for the ADF statistic."""
        return self._p_val

    def optimal_lag(self) -> int:
        """Return the optimal lag length selected by BIC."""
        return self._lag

    def is_stationary(self, alpha: float = 0.05) -> bool:
        """Return True if the null of unit root is rejected at level α."""
        return self._p_val < alpha

    def report(self) -> dict:
        """Generate a diagnostic report.

        Returns:
            dict with keys: statistic, p_value, lag, stationary, alpha, conclusion.
        """
        stationary = self.is_stationary()
        return {
            "test": "ADF",
            "statistic": round(self._adf_stat, 6),
            "p_value": round(self._p_val, 6),
            "lag": self._lag,
            "n_obs": self._n,
            "stationary": stationary,
            "conclusion": (
                "Stationary (reject unit root)"
                if stationary
                else "Non-stationary (fail to reject unit root)"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# KPSS Test
# ═══════════════════════════════════════════════════════════════════════

class KPSSTest:
    """KPSS test for stationarity.

    H₀: The series is stationary (level or trend stationary).
    H₁: The series has a unit root (is non-stationary).

    Test statistic: η = T^{-2} Σ S_t^2 / s^2(l)
    where S_t is the partial sum of residuals from regressing y on constant
    (and optionally trend), and s^2(l) is the long-run variance estimator.
    """

    def __init__(self, series: np.ndarray, trend: str = "c", nlags: int | None = None):
        """
        Args:
            series: 1-D array of observations.
            trend: "c" for level-stationary (constant only),
                   "ct" for trend-stationary (constant + trend).
            nlags: Number of lags for Newey-West long-run variance.
                   If None, uses automatic lag selection: floor(4*(n/100)^(1/4)).
        """
        y = np.asarray(series, dtype=np.float64).flatten()
        if y.ndim != 1:
            raise ValueError("series must be a 1-D array")
        if len(y) < 10:
            raise ValueError("series must have at least 10 observations")
        if trend not in ("c", "ct"):
            raise ValueError("trend must be 'c' or 'ct'")

        self._y = y
        self._n = len(y)
        self._trend = trend

        if nlags is None:
            self._nlags = int(np.floor(4 * (self._n / 100) ** 0.25))
        else:
            self._nlags = int(nlags)

        self._kpss_stat = self._compute_kpss()

    def _compute_kpss(self) -> float:
        """Compute the KPSS test statistic."""
        y = self._y
        n = self._n

        # Regress y on constant (and trend if ct)
        if self._trend == "c":
            X = np.ones((n, 1))
        else:  # ct
            t = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
            X = np.column_stack([np.ones(n), t])

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        # Partial sum process
        S = np.cumsum(residuals)

        # Estimate long-run variance (Newey-West with Bartlett kernel)
        s2_lr = self._newey_west_variance(residuals, self._nlags)

        if s2_lr <= 0:
            return np.inf

        # KPSS statistic
        kpss = np.sum(S ** 2) / (n ** 2 * s2_lr)
        return float(kpss)

    @staticmethod
    def _newey_west_variance(residuals: np.ndarray, nlags: int) -> float:
        """Compute Newey-West long-run variance estimator with Bartlett kernel.

        s²(l) = γ₀ + 2 Σ_{j=1}^{l} w(j, l) γⱼ
        where w(j, l) = 1 - j/(l+1) is the Bartlett kernel weight.
        """
        n = len(residuals)
        if n <= 1:
            return 1e-10

        # Autocovariances
        gamma_0 = np.sum(residuals ** 2) / n

        lr_var = gamma_0
        max_lag = min(nlags, n - 1)

        for j in range(1, max_lag + 1):
            w = 1.0 - j / (max_lag + 1.0)  # Bartlett weight
            gamma_j = np.sum(residuals[j:] * residuals[:-j]) / n
            lr_var += 2.0 * w * gamma_j

        return max(lr_var, 1e-10)

    # ── public API ─────────────────────────────────────────────────────

    def kpss_statistic(self) -> float:
        """Return the KPSS test statistic."""
        return self._kpss_stat

    def is_stationary(self, alpha: float = 0.05) -> bool:
        """Return True if the null of stationarity is NOT rejected at level α.

        Note: KPSS has H₀ of stationarity, so we fail to reject → stationary.
        """
        table_key = "eta_mu" if self._trend == "c" else "eta_tau"
        crit_value = _KPSS_CRITICAL_VALUES[table_key].get(alpha)
        if crit_value is None:
            # Find nearest alpha
            alphas = sorted(_KPSS_CRITICAL_VALUES[table_key].keys())
            if alpha <= alphas[0]:
                crit_value = _KPSS_CRITICAL_VALUES[table_key][alphas[0]]
            elif alpha >= alphas[-1]:
                crit_value = _KPSS_CRITICAL_VALUES[table_key][alphas[-1]]
            else:
                # Linear interpolation
                for i in range(len(alphas) - 1):
                    if alphas[i] <= alpha <= alphas[i + 1]:
                        cv_low = _KPSS_CRITICAL_VALUES[table_key][alphas[i]]
                        cv_high = _KPSS_CRITICAL_VALUES[table_key][alphas[i + 1]]
                        frac = (alpha - alphas[i]) / (alphas[i + 1] - alphas[i])
                        crit_value = cv_low + frac * (cv_high - cv_low)
                        break

        return self._kpss_stat < crit_value

    def report(self) -> dict:
        """Generate a diagnostic report.

        Returns:
            dict with keys: statistic, stationary, trend, nlags, n_obs, conclusion.
        """
        stationary = self.is_stationary()
        table_key = "eta_mu" if self._trend == "c" else "eta_tau"
        crit_05 = _KPSS_CRITICAL_VALUES[table_key][0.05]

        return {
            "test": "KPSS",
            "statistic": round(self._kpss_stat, 6),
            "critical_value_5pct": crit_05,
            "trend": self._trend,
            "nlags": self._nlags,
            "n_obs": self._n,
            "stationary": stationary,
            "conclusion": (
                "Stationary (fail to reject stationarity)"
                if stationary
                else "Non-stationary (reject stationarity)"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════════

def check_stationarity(series: np.ndarray, alpha: float = 0.05) -> dict:
    """Run both ADF and KPSS tests for a comprehensive stationarity check.

    Decision matrix:
    ┌───────────────┬─────────────────────┬──────────────────────┐
    │               │ KPSS: stationary    │ KPSS: non-stationary │
    ├───────────────┼─────────────────────┼──────────────────────┤
    │ ADF: station. │ Stationary          │ Mixed (→ stationary) │
    │ ADF: non-stat.│ Mixed (→ non-stat.) │ Non-stationary       │
    └───────────────┴─────────────────────┴──────────────────────┘

    Args:
        series: 1-D array of observations.
        alpha: Significance level.

    Returns:
        dict with keys: adf_report, kpss_report, conclusion, stationary,
                        needs_differencing.
    """
    adf = ADFTest(series)
    kpss = KPSSTest(series)
    kpss_trend = KPSSTest(series, trend="ct")

    adf_stationary = adf.is_stationary(alpha)
    kpss_stationary = kpss.is_stationary(alpha)

    # Determine overall conclusion
    if adf_stationary and kpss_stationary:
        conclusion = "Stationary"
        stationary = True
    elif not adf_stationary and not kpss_stationary:
        conclusion = "Non-stationary (unit root present)"
        stationary = False
    elif adf_stationary and not kpss_stationary:
        conclusion = "Mixed evidence, leaning stationary (ADF rejects unit root)"
        stationary = True
    else:
        conclusion = "Mixed evidence, leaning non-stationary (KPSS rejects stationarity)"
        stationary = False

    return {
        "adf": adf.report(),
        "kpss": kpss.report(),
        "conclusion": conclusion,
        "stationary": stationary,
        "needs_differencing": not stationary,
    }
