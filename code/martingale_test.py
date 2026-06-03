"""
Martingale Test
===============
Tests whether an equity curve (cumulative returns) behaves as a martingale,
indicating no exploitable alpha, or as a sub-martingale, indicating positive
expected returns.

Tests performed:
1. Variance Ratio test on the equity curve
2. Runs test on the return signs
3. Ljung-Box autocorrelation test

Usage:
    mt = MartingaleTest(equity_curve)
    print(mt.report())
"""

import numpy as np
from scipy import stats
from variance_ratio_test import VarianceRatioTest


class MartingaleTest:
    """Comprehensive martingale diagnostic for equity curves.

    A martingale: E[next | history] = current  → no predictable drift.
    A sub-martingale: E[next | history] > current → positive drift (alpha).
    """

    def __init__(self, equity_curve: np.ndarray | list[float]):
        """
        Args:
            equity_curve: 1-D array of cumulative equity values (e.g., NAV).
        """
        eq = np.asarray(equity_curve, dtype=np.float64).flatten()
        if eq.ndim != 1:
            raise ValueError("equity_curve must be a 1-D array")
        if len(eq) < 10:
            raise ValueError("equity_curve must have at least 10 observations")
        if np.any(eq <= 0):
            raise ValueError("equity_curve values must be strictly positive")

        self._equity = eq
        self._returns = np.diff(np.log(eq))  # log returns
        self._vr_tester = VarianceRatioTest(eq)
        self._n_returns = len(self._returns)

    # ── individual tests ────────────────────────────────────────────────

    def variance_ratio_test(self, k: int = 2) -> tuple[float, float]:
        """Apply VR test directly to the equity curve.

        If the equity curve is a martingale, VR(k) ≈ 1.
        If VR(k) > 1 and significant → trend persistence (sub-martingale).
        If VR(k) < 1 and significant → mean reversion.

        Args:
            k: Aggregation period (default 2).

        Returns:
            (vr_statistic, p_value).
        """
        return self._vr_tester.compute_vr(k)

    def runs_test(self) -> tuple[float, float]:
        """Wald-Wolfowitz runs test on return signs.

        Tests whether positive and negative returns appear randomly.
        Under the martingale null, signs are independent.

        Returns:
            (z_statistic, p_value).
        """
        signs = np.sign(self._returns)
        # Remove zeros
        signs = signs[signs != 0]
        if len(signs) < 2:
            raise ValueError("Insufficient non-zero returns for runs test")

        # Count runs
        runs = 1
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1]:
                runs += 1

        n_pos = int(np.sum(signs > 0))
        n_neg = int(np.sum(signs < 0))
        n_total = n_pos + n_neg

        if n_pos == 0 or n_neg == 0:
            raise ValueError("All returns have same sign; runs test not applicable")

        # Expected runs and std
        expected_runs = 1 + (2 * n_pos * n_neg) / n_total
        var_runs = ((2 * n_pos * n_neg) *
                    (2 * n_pos * n_neg - n_total)) / (n_total ** 2 * (n_total - 1))

        z = (runs - expected_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return float(z), float(p_value)

    def autocorrelation_test(self, max_lag: int = 10) -> dict:
        """Ljung-Box test for autocorrelation in returns.

        H₀: Returns are independently distributed (no autocorrelation).
        Under the martingale hypothesis, returns should not exhibit significant AC.

        Returns:
            dict with keys: statistic, p_value, lags_tested, and
            individual_lags: list of (lag, acf, p_value) tuples.
        """
        ret = self._returns
        T = len(ret)
        if max_lag >= T:
            max_lag = T - 2
        if max_lag < 1:
            raise ValueError("Not enough data for autocorrelation test")

        n_lags = min(max_lag, T // 5)
        if n_lags < 1:
            n_lags = 1

        # Ljung-Box Q statistic
        acf_values = []
        q_stat = 0.0
        for lag in range(1, n_lags + 1):
            rho = np.corrcoef(ret[:-lag], ret[lag:])[0, 1]
            if np.isnan(rho):
                rho = 0.0
            acf_values.append(rho)
            q_stat += (rho ** 2) / (T - lag)
        q_stat *= T * (T + 2)

        p_value = 1 - stats.chi2.cdf(q_stat, df=n_lags)

        # Individual lag p-values (Bartlett's approx)
        individual = []
        se = 1.0 / np.sqrt(T)
        for lag in range(1, n_lags + 1):
            rho = acf_values[lag - 1]
            z_score = rho / se
            lag_p = 2 * (1 - stats.norm.cdf(abs(z_score)))
            individual.append((lag, round(float(rho), 4), round(float(lag_p), 4)))

        return {
            "statistic": round(float(q_stat), 4),
            "p_value": round(float(p_value), 4),
            "lags_tested": n_lags,
            "individual_lags": individual,
        }

    # ── composite judgments ─────────────────────────────────────────────

    def is_martingale(self, alpha: float = 0.05) -> bool:
        """Composite test: is the equity curve consistent with a martingale?

        We check the structural properties:
        - VR test: does not reject random walk (variance grows linearly)
        - Autocorrelation: no significant serial dependence
        - Runs test: return signs are random

        A martingale has E[next|history] = current. If variance structure
        and independence hold, we classify as martingale even if sample
        mean drift happens to be non-zero (sampling error).
        """
        _, vr_p = self.variance_ratio_test(k=2)
        ac_result = self.autocorrelation_test(max_lag=10)
        try:
            _, runs_p = self.runs_test()
            runs_pass = runs_p >= alpha
        except (ValueError, RuntimeError):
            runs_pass = True  # Not enough variety; meaning ambiguous

        vr_pass = vr_p >= alpha
        ac_pass = ac_result["p_value"] >= alpha

        # If VR and autocorrelation both pass, it's structurally a martingale
        # (even if sample drift happens to be non-zero)
        return vr_pass and ac_pass and runs_pass

    def is_submartingale(self, alpha: float = 0.05) -> bool:
        """Check if the equity curve is a sub-martingale (positive drift / alpha).

        A sub-martingale has E[next | history] >= current with strict
        inequality. This manifests as:
        - Significant positive drift (t-test on mean returns)
        - OR significant trending behavior (VR > 1)

        The VR test is supplemented with a trend test on cumulative returns
        to capture persistent positive drift.
        """
        ret = self._returns
        T = len(ret)
        mean_ret = np.mean(ret)

        # Test 1: Is the drift significantly positive? (one-sided t-test)
        se_ret = np.std(ret, ddof=1) / np.sqrt(T)
        t_stat = mean_ret / se_ret if se_ret > 0 else 0.0
        drift_p = stats.t.sf(t_stat, df=T - 1)

        # Test 2: Is there autocorrelation persistence? (VR test)
        vr, vr_p = self.variance_ratio_test(k=2)
        trending = vr > 1.0 and vr_p < alpha

        # Test 3: Is total return positive? (simple check)
        total_ret = self._equity[-1] / self._equity[0] - 1

        # Sub-martingale: positive drift is significant OR trending
        has_positive_drift = drift_p < alpha and mean_ret > 0
        has_upward_trend = trending and total_ret > 0

        return has_positive_drift or has_upward_trend

    def report(self) -> dict:
        """Generate a comprehensive martingale diagnostic report."""
        # Run all tests
        vr, vr_p = self.variance_ratio_test(k=2)
        try:
            runs_z, runs_p = self.runs_test()
        except (ValueError, RuntimeError):
            runs_z, runs_p = 0.0, 0.5
        ac = self.autocorrelation_test(max_lag=10)

        mean_ret = float(np.mean(self._returns))
        total_ret = float(self._equity[-1] / self._equity[0] - 1)

        interpretation = "martingale"
        if self.is_submartingale():
            interpretation = "sub-martingale (positive alpha)"
        elif not self.is_martingale():
            # Check which tests failed
            if vr > 1.0 and vr_p < 0.05:
                interpretation = "trending / potentially sub-martingale"
            elif vr < 1.0 and vr_p < 0.05:
                interpretation = "mean-reverting / not martingale"
            else:
                interpretation = "non-martingale (autocorrelation or runs test rejects)"

        return {
            "total_return": round(float(total_ret), 6),
            "mean_log_return": round(mean_ret, 6),
            "n_observations": len(self._equity),
            "variance_ratio": round(vr, 4),
            "vr_p_value": round(vr_p, 4),
            "runs_test_z": round(runs_z, 4),
            "runs_test_p": round(runs_p, 4),
            "ljung_box_statistic": ac["statistic"],
            "ljung_box_p_value": ac["p_value"],
            "ljung_box_lags": ac["lags_tested"],
            "is_martingale": self.is_martingale(),
            "is_submartingale": self.is_submartingale(),
            "interpretation": interpretation,
        }


# ── test helpers ────────────────────────────────────────────────────────

def simulate_martingale(n: int = 1000, sigma: float = 0.01, seed: int = 42) -> np.ndarray:
    """Simulate a martingale equity curve: log returns ~ N(0, σ²)."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, sigma, n - 1)
    log_eq = np.cumsum(log_ret)
    log_eq = np.insert(log_eq, 0, 0)
    return np.exp(log_eq + 4.0)


def simulate_submartingale(n: int = 2000, mu: float = 0.001,
                           sigma: float = 0.01, seed: int = 42) -> np.ndarray:
    """Simulate a sub-martingale (positive drift) equity curve.

    Uses strong drift (mu=0.001 per period at sigma=0.01) so that
    over n=2000 periods expected total log return ≈ 2.0.
    """
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(mu, sigma, n - 1)
    log_eq = np.cumsum(log_ret)
    log_eq = np.insert(log_eq, 0, 0)
    return np.exp(log_eq + 4.0)
