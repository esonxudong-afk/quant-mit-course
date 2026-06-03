"""
Lo-MacKinlay Variance Ratio Test
==================================
Tests whether a price series follows a random walk against alternatives
of mean reversion (VR < 1) or trending/explosive behavior (VR > 1).

Reference:
- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow
  random walks: Evidence from a simple specification test.

Usage:
    vrt = VarianceRatioTest(prices)
    vr, pval = vrt.compute_vr(k=2)
    print(vrt.report())
"""

import numpy as np
from scipy import stats


class VarianceRatioTest:
    """Lo-MacKinlay variance ratio test for random walk hypothesis."""

    def __init__(self, prices: np.ndarray | list[float]):
        """
        Args:
            prices: 1-D array of price observations (levels, not returns).
                    Must have at least 3 observations for k=2.
        """
        prices_arr = np.asarray(prices, dtype=np.float64).flatten()
        if prices_arr.ndim != 1:
            raise ValueError("prices must be a 1-D array")
        if len(prices_arr) < 3:
            raise ValueError("prices must have at least 3 observations")
        if np.any(prices_arr <= 0):
            raise ValueError("prices must be strictly positive")

        self._raw = prices_arr
        self._log_prices = np.log(prices_arr)
        self._returns = np.diff(self._log_prices)  # log-returns
        self._n = len(self._returns)  # T = nq for k-period aggregation

    # ── core computation ────────────────────────────────────────────────

    def compute_vr(self, k: int = 2) -> tuple[float, float]:
        """Compute VR(k) and its p-value under the homoscedasticity assumption.

        VR(k) = Var(k-period return) / (k * Var(1-period return))

        Uses Lo-MacKinlay (1988) unbiased variance estimators:
        - σ²_a: unbiased 1-period variance
        - σ²_b(k): unbiased k-period variance (adjusted for overlapping)
        - VR(k) = σ²_b(k) / σ²_a

        Test statistic M₁(k) = (VR(k) - 1) / sqrt(φ̂)
        where φ̂ = 2*(2k-1)*(k-1) / (3k*nq) under homoscedasticity.

        Returns:
            (vr_statistic, p_value) — two-sided p-value.
        """
        self._validate_k(k)

        ret = self._returns
        nq = self._n  # T = nq, number of 1-period returns

        # Unbiased 1-period variance: σ²_a
        mu = np.mean(ret)
        sigma2_a = np.sum((ret - mu) ** 2) / (nq - 1)

        # k-period (overlapping) returns
        ret_k = self._k_period_returns(k)
        nk = len(ret_k)  # nq - k + 1
        mu_k = np.mean(ret_k)

        # Unbiased k-period variance (adjusted for overlap bias)
        # σ²_b(k) = 1/m * sum[(r_k(t) - k*μ)²], m = k*(nq - k + 1)*(1 - k/nq)
        m_factor = k * nk * (1 - k / nq)
        sigma2_b = np.sum((ret_k - k * mu) ** 2) / m_factor

        # Variance Ratio
        vr = sigma2_b / sigma2_a

        # Asymptotic variance of VR under homoscedasticity
        # φ(k) = 2*(2k-1)*(k-1) / (3k*nq)
        phi = (2 * (2 * k - 1) * (k - 1)) / (3 * k * nq)
        se = np.sqrt(phi) if phi > 0 else np.inf

        z_score = (vr - 1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return float(vr), float(p_value)

    def compute_vr_multi(self, k_list: list[int] | None = None) -> dict:
        """Compute VR for multiple aggregation intervals.

        Args:
            k_list: List of k values. Default: [2, 5, 10, 20].

        Returns:
            dict mapping k → (vr, p_value).
        """
        if k_list is None:
            k_list = [2, 5, 10, 20]
        return {k: self.compute_vr(k) for k in k_list}

    # ── diagnostic helpers ──────────────────────────────────────────────

    def is_random_walk(self, alpha: float = 0.05) -> bool:
        """Check if the series is consistent with a random walk."""
        _, p = self.compute_vr(k=2)
        return p >= alpha

    def is_mean_reverting(self, alpha: float = 0.05) -> bool:
        """Check for mean reversion: VR < 1 AND statistically significant."""
        vr, p = self.compute_vr(k=2)
        return vr < 1.0 and p < alpha

    def is_trending(self, alpha: float = 0.05) -> bool:
        """Check for trending/explosive behavior: VR > 1 AND statistically significant."""
        vr, p = self.compute_vr(k=2)
        return vr > 1.0 and p < alpha

    def report(self) -> dict:
        """Generate a diagnostic report."""
        vr, p = self.compute_vr(k=2)
        vr_multi = self.compute_vr_multi()

        interpretation = "random_walk"
        if vr < 1.0 and p < 0.05:
            interpretation = "mean_reverting"
        elif vr > 1.0 and p < 0.05:
            interpretation = "trending"

        return {
            "vr_k2": round(vr, 4),
            "p_value_k2": round(p, 4),
            "is_random_walk": self.is_random_walk(),
            "is_mean_reverting": self.is_mean_reverting(),
            "is_trending": self.is_trending(),
            "interpretation": interpretation,
            "multi_vr": {k: {"vr": round(v[0], 4), "p": round(v[1], 4)} for k, v in vr_multi.items()},
            "n_observations": len(self._raw),
        }

    # ── internals ───────────────────────────────────────────────────────

    def _validate_k(self, k: int):
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if k > self._n:
            raise ValueError(f"k ({k}) exceeds available returns ({self._n})")

    def _k_period_returns(self, k: int) -> np.ndarray:
        """Compute overlapping k-period log-returns."""
        log_p = self._log_prices
        nk = len(log_p) - k
        return log_p[k:] - log_p[:nk]


# ── test helpers ────────────────────────────────────────────────────────

def simulate_random_walk(n: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate a log-normal random walk price series."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.01, n - 1)
    log_prices = np.cumsum(log_returns)
    log_prices = np.insert(log_prices, 0, 0)
    return np.exp(log_prices + 4.0)  # keep prices positive


def simulate_mean_reverting(n: int = 1000, phi: float = 0.9, seed: int = 42) -> np.ndarray:
    """Generate a mean-reverting (stationary AR(1)) log-price series."""
    rng = np.random.default_rng(seed)
    log_p = np.zeros(n)
    noise = rng.normal(0, 0.01, n - 1)
    for t in range(1, n):
        log_p[t] = phi * log_p[t - 1] + noise[t - 1]
    return np.exp(log_p + 4.0)


def simulate_trending(n: int = 1000, drift: float = 1.0002, seed: int = 42) -> np.ndarray:
    """Generate a trending (momentum/explosive) price series."""
    rng = np.random.default_rng(seed)
    prices = np.zeros(n)
    prices[0] = 100.0
    for t in range(1, n):
        prices[t] = prices[t - 1] * drift + rng.normal(0, 0.01)
    return prices
