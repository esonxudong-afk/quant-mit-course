"""
ACF & PACF Analyzer
====================
Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
analysis for time series identification.

The ACF captures the correlation between a series and its lagged values.
The PACF captures the direct correlation at each lag, removing intermediate effects.

Used for identifying ARMA model orders:
- PACF cutoff → AR order (p)
- ACF cutoff → MA order (q)

References:
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
  Time Series Analysis: Forecasting and Control (5th ed.).
- Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in
  time series models.

Usage:
    from time_series.acf_pacf import ACFPACFAnalyzer

    analyzer = ACFPACFAnalyzer(series, nlags=20)
    acf_vals, acf_ci = analyzer.acf()
    print(analyzer.report())
"""

import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════
# Helper: Bartlett's formula for ACF confidence intervals
# ═══════════════════════════════════════════════════════════════════════

def _bartlett_standard_error(acf_values: np.ndarray, n: int, lag: int) -> float:
    """Compute standard error of ACF at given lag using Bartlett's formula.

    SE(r_k) = sqrt((1 + 2 * Σ_{i=1}^{k-1} r_i^2) / n)

    Args:
        acf_values: Complete ACF array (length nlags+1).
        n: Number of observations.
        lag: The lag k at which to compute SE.

    Returns:
        Standard error of the ACF at this lag.
    """
    if lag == 0:
        return 0.0
    sum_sq = 1.0 + 2.0 * np.sum(acf_values[1:lag] ** 2)
    return np.sqrt(sum_sq / n)


# ═══════════════════════════════════════════════════════════════════════
# Helper: Levinson-Durbin recursion for PACF
# ═══════════════════════════════════════════════════════════════════════

def _levinson_durbin(acf_values: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute PACF via the Durbin-Levinson recursion.

    Args:
        acf_values: ACF values ρ₀, ρ₁, ..., ρ_{max_lag}.
        max_lag: Maximum lag.

    Returns:
        PACF values φ_{kk} for k = 1, ..., max_lag.
    """
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0  # convention

    if max_lag >= 1:
        phi = np.zeros(max_lag + 1)
        phi[1] = acf_values[1]
        pacf[1] = phi[1]

        for k in range(2, max_lag + 1):
            # Compute numerator: ρ_k - Σ φ_{k-1,j} * ρ_{k-j}
            numerator = acf_values[k]
            for j in range(1, k):
                numerator -= phi[j] * acf_values[k - j]

            # Compute denominator: 1 - Σ φ_{k-1,j} * ρ_j
            denominator = 1.0
            for j in range(1, k):
                denominator -= phi[j] * acf_values[j]

            if abs(denominator) < 1e-12:
                phi_kk = 0.0
            else:
                phi_kk = numerator / denominator

            # Update phi coefficients
            phi_new = np.zeros(max_lag + 1)
            for j in range(1, k):
                phi_new[j] = phi[j] - phi_kk * phi[k - j]
            phi_new[k] = phi_kk

            phi = phi_new.copy()
            pacf[k] = phi_kk

    return pacf


# ═══════════════════════════════════════════════════════════════════════
# ACF/PACF Analyzer
# ═══════════════════════════════════════════════════════════════════════

class ACFPACFAnalyzer:
    """Autocorrelation and partial autocorrelation analysis.

    Computes ACF and PACF with confidence intervals, suggests ARMA orders,
    and performs Ljung-Box white noise tests.
    """

    def __init__(self, series: np.ndarray, nlags: int = 40):
        """
        Args:
            series: 1-D array of observations.
            nlags: Number of lags to compute. Must be < n - 1.
                   Default 40 (common for financial time series).
        """
        y = np.asarray(series, dtype=np.float64).flatten()
        if y.ndim != 1:
            raise ValueError("series must be a 1-D array")
        n = len(y)
        if n < 4:
            raise ValueError("series must have at least 4 observations")

        self._y = y
        self._n = n
        self._mean = np.mean(y)
        self._centered = y - self._mean

        if nlags >= n - 1:
            nlags = n - 2
        self._nlags = max(1, nlags)

        # Pre-compute ACF and PACF
        self._acf_values, self._acf_ci = self._compute_acf()
        self._pacf_values, self._pacf_ci = self._compute_pacf()
        self._ljung_box = self._compute_ljung_box()

    # ── ACF computation ─────────────────────────────────────────────────

    def _compute_acf(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute ACF with Bartlett confidence intervals.

        Returns:
            (acf_values, confidence_intervals) where confidence_intervals
            is a (nlags+1, 2) array of [lower, upper] bounds.
        """
        n = self._n
        nlags = self._nlags
        y = self._centered

        # Sample autocovariance
        acf = np.zeros(nlags + 1)
        acf[0] = 1.0  # ρ₀ = 1

        gamma_0 = np.sum(y ** 2) / n
        if gamma_0 <= 0:
            return acf, np.zeros((nlags + 1, 2))

        for k in range(1, nlags + 1):
            gamma_k = np.sum(y[k:] * y[:-k]) / n
            acf[k] = gamma_k / gamma_0

        # Confidence intervals using Bartlett's formula
        # Default: ±1.96 / sqrt(n) for white noise approximation
        ci = np.zeros((nlags + 1, 2))
        z = stats.norm.ppf(0.975)  # ≈ 1.96

        for k in range(1, nlags + 1):
            se = _bartlett_standard_error(acf, n, k)
            ci[k, 0] = -z * se
            ci[k, 1] = z * se

        return acf, ci

    # ── PACF computation ────────────────────────────────────────────────

    def _compute_pacf(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute PACF via Durbin-Levinson with confidence intervals.

        Returns:
            (pacf_values, confidence_intervals) where confidence_intervals
            is a (nlags+1, 2) array of [lower, upper] bounds.
        """
        n = self._n
        nlags = self._nlags

        pacf = _levinson_durbin(self._acf_values, nlags)

        # PACF confidence intervals: ±1.96 / sqrt(n)
        ci = np.zeros((nlags + 1, 2))
        z = stats.norm.ppf(0.975)
        se = 1.0 / np.sqrt(n)

        for k in range(1, nlags + 1):
            ci[k, 0] = -z * se
            ci[k, 1] = z * se

        return pacf, ci

    # ── Ljung-Box test ─────────────────────────────────────────────────

    def _compute_ljung_box(self, max_lag: int | None = None) -> dict:
        """Compute the Ljung-Box test for white noise.

        Q = n(n+2) * Σ_{k=1}^{m} ρ̂_k^2 / (n-k)
        Under H₀ of white noise, Q ~ χ²(m).

        Args:
            max_lag: Maximum lag for the test. Default: min(20, nlags).

        Returns:
            dict with keys: statistic, p_value, lags, is_white_noise.
        """
        if max_lag is None:
            max_lag = min(20, self._nlags)

        n = self._n
        acf = self._acf_values

        q_stat = 0.0
        for k in range(1, max_lag + 1):
            q_stat += acf[k] ** 2 / (n - k)

        q_stat *= n * (n + 2)

        p_value = 1.0 - stats.chi2.cdf(q_stat, df=max_lag)

        return {
            "statistic": float(q_stat),
            "p_value": float(p_value),
            "lags_tested": max_lag,
            "is_white_noise": p_value > 0.05,
        }

    # ── Order suggestion ────────────────────────────────────────────────

    @staticmethod
    def _find_cutoff(values: np.ndarray, ci: np.ndarray, max_lag: int) -> int:
        """Find the lag after which values stay within confidence bounds.

        A "cutoff" after lag p means: |value[p+1]| < CI and values stay
        mostly inside CI for several lags after.

        Returns:
            Suggested order (0 if no cutoff detected).
        """
        z = stats.norm.ppf(0.975)
        se = 1.0 / np.sqrt(len(values) if hasattr(values, '__len__') else 1)

        for lag in range(1, max_lag):
            # Check if value at lag+1 is inside bounds
            if abs(values[lag + 1]) < z * se:
                # Confirm: at least 3 of next 5 values are also inside
                count_inside = 0
                check_range = min(5, max_lag - lag)
                for j in range(1, check_range + 1):
                    if lag + j <= max_lag and abs(values[lag + j]) < z * se:
                        count_inside += 1
                if count_inside >= min(3, check_range):
                    return lag

        return 0

    # ── public API ─────────────────────────────────────────────────────

    def acf(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ACF values and confidence intervals.

        Returns:
            (values, conf_int) where values is (nlags+1,) array (ρ₀=1 at index 0),
            and conf_int is (nlags+1, 2) array of [lower, upper] bounds.
        """
        return self._acf_values.copy(), self._acf_ci.copy()

    def pacf(self) -> tuple[np.ndarray, np.ndarray]:
        """Return PACF values and confidence intervals.

        Returns:
            (values, conf_int) where values is (nlags+1,) array (φ₀₀=1 at index 0),
            and conf_int is (nlags+1, 2) array of [lower, upper] bounds.
        """
        return self._pacf_values.copy(), self._pacf_ci.copy()

    def suggested_ar_order(self) -> int:
        """Suggest AR order (p) based on PACF cutoff.

        Looks for the lag after which PACF values fall inside the
        confidence interval.
        """
        return self._find_cutoff(self._pacf_values, self._pacf_ci, self._nlags)

    def suggested_ma_order(self) -> int:
        """Suggest MA order (q) based on ACF cutoff.

        Looks for the lag after which ACF values fall inside the
        confidence interval.
        """
        return self._find_cutoff(self._acf_values, self._acf_ci, self._nlags)

    def is_white_noise(self, alpha: float = 0.05) -> bool:
        """Test if the series is white noise via Ljung-Box test.

        Args:
            alpha: Significance level (default 0.05).

        Returns:
            True if series appears to be white noise (fail to reject H₀).
        """
        return self._ljung_box["p_value"] > alpha

    def report(self) -> dict:
        """Generate a comprehensive ACF/PACF diagnostic report.

        Returns:
            dict with keys: nlags, n_obs, suggested_ar, suggested_ma,
                            is_white_noise, ljung_box, acf_summary, pacf_summary.
        """
        # Significant ACF lags (outside CI)
        significant_acf = []
        for k in range(1, self._nlags + 1):
            if self._acf_values[k] < self._acf_ci[k, 0] or self._acf_values[k] > self._acf_ci[k, 1]:
                significant_acf.append(k)

        # Significant PACF lags
        significant_pacf = []
        for k in range(1, self._nlags + 1):
            if self._pacf_values[k] < self._pacf_ci[k, 0] or self._pacf_values[k] > self._pacf_ci[k, 1]:
                significant_pacf.append(k)

        ar_order = self.suggested_ar_order()
        ma_order = self.suggested_ma_order()
        wn = self.is_white_noise()

        return {
            "nlags": self._nlags,
            "n_obs": self._n,
            "suggested_ar_order": ar_order,
            "suggested_ma_order": ma_order,
            "suggested_model": (
                f"ARMA({ar_order},{ma_order})"
                if ar_order > 0 or ma_order > 0
                else "White noise"
            ),
            "is_white_noise": wn,
            "ljung_box": {
                "statistic": round(self._ljung_box["statistic"], 4),
                "p_value": round(self._ljung_box["p_value"], 6),
                "lags_tested": self._ljung_box["lags_tested"],
            },
            "acf_significant_lags": significant_acf[:10],  # top 10
            "pacf_significant_lags": significant_pacf[:10],
        }
