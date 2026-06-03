"""
Bootstrap Non-Parametric Confidence Intervals
==============================================
Replaces the T-test approach from Chapter 1 because crypto returns are
non-normal (fat tails, skew). Bootstrap resampling gives distribution-free
confidence intervals for mean return and Sharpe ratio.

Supports:
  - Mean confidence interval (Bias-Corrected and Accelerated, BCa)
  - Sharpe ratio confidence interval
  - Bootstrap t-statistic distribution (non-parametric pivot)
  - Delta-method Sharpe standard error as sanity-check

Usage:
    bs = BootstrapStats(returns, n_bootstrap=10000)
    report = bs.report()
    # => {'mean': -0.001, 'mean_ci_lower': -0.005, 'mean_ci_upper': 0.002,
    #      'sharpe': -0.12, 'sharpe_ci_lower': -0.50, 'sharpe_ci_upper': 0.20,
    #      'significant_alpha': False}
"""

import numpy as np

# Optional: use scipy.stats for BCa acceleration
try:
    from scipy.stats import norm as scipy_norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class BootstrapStats:
    """Bootstrap non-parametric confidence intervals for mean and Sharpe ratio.

    Attributes:
        returns: 1-D numpy array of asset returns.
        n_bootstrap: Number of resampling iterations (default 10,000).
        seed: Random seed for reproducibility.
        annual_factor: Trading days for annualisation (default 252).
    """

    def __init__(self, returns: np.ndarray, n_bootstrap: int = 10000,
                 seed: int = 42, annual_factor: int = 252):
        """Initialise the bootstrap engine.

        Args:
            returns: 1-D array of return observations.
            n_bootstrap: Bootstrap iterations. ≥1000 recommended.
            seed: PRNG seed for reproducibility.
            annual_factor: Annualisation factor (252 = daily).

        Raises:
            ValueError: If returns is empty or n_bootstrap < 100.
        """
        returns = np.asarray(returns, dtype=float).ravel()
        if returns.size < 2:
            raise ValueError(f"Need at least 2 return observations, got {returns.size}")
        if n_bootstrap < 100:
            raise ValueError(f"n_bootstrap must be >= 100, got {n_bootstrap}")

        self.returns = returns
        self.n_obs = returns.size
        self.n_bootstrap = int(n_bootstrap)
        self.seed = int(seed)
        self.annual_factor = float(annual_factor)

        # Cached statistics
        self._sample_mean = float(np.mean(returns))
        self._sample_std = float(np.std(returns, ddof=1))
        self._sample_sharpe = (self._sample_mean / self._sample_std
                               * np.sqrt(self.annual_factor)
                               if self._sample_std > 0 else 0.0)

        # Pre-generate bootstrap samples once
        self._bootstrapped = False
        self._boot_means = None
        self._boot_sharpes = None

    def _generate_bootstrap(self):
        """Generate bootstrap replicates of mean and Sharpe ratio."""
        if self._bootstrapped:
            return
        rng = np.random.RandomState(self.seed)
        ret = self.returns
        boot_means = np.empty(self.n_bootstrap)
        boot_sharpes = np.empty(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            sample = rng.choice(ret, size=self.n_obs, replace=True)
            m = np.mean(sample)
            s = np.std(sample, ddof=1)
            boot_means[i] = m
            if s > 0:
                boot_sharpes[i] = m / s * np.sqrt(self.annual_factor)
            else:
                boot_sharpes[i] = 0.0

        self._boot_means = boot_means
        self._boot_sharpes = boot_sharpes
        self._bootstrapped = True

    # ── Mean CI  ──────────────────────────────────────────────────

    def mean_ci(self, alpha: float = 0.05) -> dict:
        """Compute percentile + BCa bootstrap CI for the mean.

        Args:
            alpha: Significance level (0.05 → 95% CI).

        Returns:
            dict with keys: method, lower, upper, alpha, sample_mean.
        """
        self._generate_bootstrap()
        boot = self._boot_means
        lower_percentile = np.percentile(boot, 100 * alpha / 2)
        upper_percentile = np.percentile(boot, 100 * (1 - alpha / 2))

        result = {
            "alpha": alpha,
            "sample_mean": self._sample_mean,
            "percentile_ci_lower": float(lower_percentile),
            "percentile_ci_upper": float(upper_percentile),
        }

        # BCa if scipy available
        if _HAS_SCIPY:
            try:
                lower_bca, upper_bca = self._bca_ci(boot, self.returns, alpha)
                result["bca_ci_lower"] = float(lower_bca)
                result["bca_ci_upper"] = float(upper_bca)
                result["method"] = "BCa"
            except Exception:
                result["bca_ci_lower"] = result["percentile_ci_lower"]
                result["bca_ci_upper"] = result["percentile_ci_upper"]
                result["method"] = "Percentile"
        else:
            result["method"] = "Percentile"

        # Add standard error
        result["bootstrap_se"] = float(np.std(boot, ddof=0))
        return result

    def _bca_ci(self, boot_dist: np.ndarray, orig_sample: np.ndarray,
                alpha: float) -> tuple:
        """BCa (Bias-Corrected and Accelerated) confidence interval."""
        z_alpha2 = scipy_norm.ppf(alpha / 2)
        z_1_alpha2 = scipy_norm.ppf(1 - alpha / 2)

        # Bias correction
        theta_hat = self._sample_mean
        prop_less = np.mean(boot_dist < theta_hat)
        z0 = scipy_norm.ppf(prop_less)

        # Acceleration (jackknife)
        n = orig_sample.size
        jack_means = np.empty(n)
        for i in range(n):
            jack_means[i] = np.mean(np.delete(orig_sample, i))
        jack_mean_global = np.mean(jack_means)
        num = np.sum((jack_mean_global - jack_means) ** 3)
        denom = 6 * (np.sum((jack_mean_global - jack_means) ** 2)) ** 1.5
        a = num / denom if denom > 0 else 0.0

        # Adjusted quantiles
        alpha1 = scipy_norm.cdf(z0 + (z0 + z_alpha2) / (1 - a * (z0 + z_alpha2)))
        alpha2 = scipy_norm.cdf(z0 + (z0 + z_1_alpha2) / (1 - a * (z0 + z_1_alpha2)))

        lower = float(np.percentile(boot_dist, 100 * alpha1))
        upper = float(np.percentile(boot_dist, 100 * alpha2))
        return lower, upper

    # ── Sharpe CI  ────────────────────────────────────────────────

    def sharpe_ci(self, alpha: float = 0.05) -> dict:
        """Compute bootstrap CI for the annualised Sharpe ratio.

        Args:
            alpha: Significance level.

        Returns:
            dict with keys: sample_sharpe, percentile_ci_lower,
                percentile_ci_upper, annual_factor.
        """
        self._generate_bootstrap()
        boot = self._boot_sharpes

        lower = np.percentile(boot, 100 * alpha / 2)
        upper = np.percentile(boot, 100 * (1 - alpha / 2))

        return {
            "sample_sharpe": self._sample_sharpe,
            "percentile_ci_lower": float(lower),
            "percentile_ci_upper": float(upper),
            "alpha": alpha,
            "annual_factor": int(self.annual_factor),
            "bootstrap_se": float(np.std(boot, ddof=0)),
        }

    # ── T-stat distribution  ──────────────────────────────────────

    def t_stat_distribution(self) -> np.ndarray:
        """Bootstrap t-statistic distribution (studentised pivot).

        Each bootstrap replicate:
            t*_i = sqrt(n) * (mean*_i - sample_mean) / std*_i

        Returns:
            1-D numpy array of bootstrap t-statistics.
        """
        self._generate_bootstrap()
        rng = np.random.RandomState(self.seed + 12345)  # different stream
        t_stats = np.empty(self.n_bootstrap)
        sqrt_n = np.sqrt(self.n_obs)
        ret = self.returns

        for i in range(self.n_bootstrap):
            sample = rng.choice(ret, size=self.n_obs, replace=True)
            m = np.mean(sample)
            s = np.std(sample, ddof=1)
            if s > 0:
                t_stats[i] = sqrt_n * (m - self._sample_mean) / s
            else:
                t_stats[i] = 0.0

        return t_stats

    # ── Full Report  ──────────────────────────────────────────────

    def report(self, alpha: float = 0.05) -> dict:
        """Generate a complete bootstrap statistics report.

        Returns:
            dict with mean CI, Sharpe CI, and significance flag.
                significant_alpha is True if the mean CI excludes zero
                (i.e. we reject H0: mean=0 at the given alpha).
        """
        mean_ci = self.mean_ci(alpha=alpha)
        sharpe_ci = self.sharpe_ci(alpha=alpha)

        # Significant if zero is NOT inside the CI
        mean_lower = mean_ci.get("bca_ci_lower", mean_ci["percentile_ci_lower"])
        mean_upper = mean_ci.get("bca_ci_upper", mean_ci["percentile_ci_upper"])
        significant_alpha = not (mean_lower <= 0 <= mean_upper)

        return {
            "n_observations": self.n_obs,
            "n_bootstrap": self.n_bootstrap,
            "annual_factor": int(self.annual_factor),
            "sample_mean": self._sample_mean,
            "sample_std": self._sample_std,
            "sample_sharpe": self._sample_sharpe,
            "mean_ci_method": mean_ci["method"],
            "mean_ci_lower": mean_lower,
            "mean_ci_upper": mean_upper,
            "sharpe_ci_lower": sharpe_ci["percentile_ci_lower"],
            "sharpe_ci_upper": sharpe_ci["percentile_ci_upper"],
            "significant_alpha": significant_alpha,
        }


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap statistics for financial returns")
    parser.add_argument("--returns", nargs="+", type=float,
                        help="Space-separated daily returns")
    parser.add_argument("--n", type=int, default=10000, help="Bootstrap iterations")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    args = parser.parse_args()

    if args.demo:
        print("=== Bootstrap Demo: Synthetic Normal Returns (true mean=0) ===\n")
        np.random.seed(42)
        fake_returns = np.random.normal(0, 0.02, size=200)
        bs = BootstrapStats(fake_returns, n_bootstrap=5000, seed=args.seed)
        r = bs.report(alpha=args.alpha)
        for k, v in r.items():
            print(f"  {k}: {v}")
        print(f"\n  significant_alpha is {r['significant_alpha']} "
              f"(false is GOOD — true mean is zero)")
    elif args.returns:
        bs = BootstrapStats(np.array(args.returns), n_bootstrap=args.n, seed=args.seed)
        r = bs.report(alpha=args.alpha)
        for k, v in r.items():
            print(f"  {k}: {v}")
    else:
        parser.print_help()
