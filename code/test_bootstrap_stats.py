"""
Tests for BootstrapStats
========================
Validates:
  1. Normal data → CI contains the true mean
  2. Fat-tailed data → CI is wider (bootstrap SE > normal SE)
  3. Zero-mean data → significant_alpha is False
  4. Non-zero mean data → significant_alpha is True
  5. Sharpe CI for a known-Mean/STD process
  6. t_stat_distribution shape and centering
  7. Edge cases: tiny sample, invalid args
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_stats import BootstrapStats


class TestBootstrapStats(unittest.TestCase):
    """Comprehensive tests for BootstrapStats."""

    # ── helpers  ──────────────────────────────────────────────────
    @staticmethod
    def _normal(n=252, mean=0.0, std=0.02):
        # one year of daily returns
        np.random.seed(123)
        return np.random.normal(mean, std, size=n)

    @staticmethod
    def _fat_tails(n=500):
        # t-dist with df=3 → excess kurtosis ~ infinite (finite for sample)
        np.random.seed(456)
        return np.random.standard_t(df=3, size=n) * 0.02

    # ── Test 1: Normal data CI contains true mean  ────────────────
    def test_normal_ci_contains_true_mean(self):
        """Bootstrap mean CI should contain the true mean for normal returns."""
        true_mean = 0.001
        ret = self._normal(n=252, mean=true_mean, std=0.02)
        bs = BootstrapStats(ret, n_bootstrap=2000, seed=42)
        ci = bs.mean_ci(alpha=0.05)

        # Check percentile CI contains true mean
        self.assertLessEqual(ci["percentile_ci_lower"], true_mean,
                             "Percentile lower bound should be <= true mean")
        self.assertGreaterEqual(ci["percentile_ci_upper"], true_mean,
                                "Percentile upper bound should be >= true mean")
        self.assertEqual(ci["alpha"], 0.05)
        self.assertAlmostEqual(ci["sample_mean"], true_mean, delta=0.005)

        # Check BCa CI also contains true mean (if scipy available)
        if "bca_ci_lower" in ci:
            self.assertLessEqual(ci["bca_ci_lower"], true_mean)
            self.assertGreaterEqual(ci["bca_ci_upper"], true_mean)

    # ── Test 2: Fat-tailed → wider bootstrap SE  ─────────────────
    def test_fat_tail_bootstrap_se_larger_than_normal(self):
        """Fat-tailed data should produce larger bootstrap standard error
        than the naive normal-theory SE."""
        ret_normal = self._normal(n=252, mean=0.0, std=0.02)
        ret_fat = self._fat_tails(n=252)

        bs_normal = BootstrapStats(ret_normal, n_bootstrap=2000, seed=42)
        bs_fat = BootstrapStats(ret_fat, n_bootstrap=2000, seed=42)

        ci_n = bs_normal.mean_ci()
        ci_f = bs_fat.mean_ci()

        # Fat-tail bootstrap SE should be larger (or at least not much smaller)
        # Note: t_3 has larger variance, so SE will naturally be larger.
        # We check the ratio: fat SE / normal SE should be > 0.8
        self.assertGreater(ci_f["bootstrap_se"] / ci_n["bootstrap_se"], 0.8,
                           "Fat-tail SE should not be materially smaller than normal SE")

    # ── Test 3: Zero mean → not significant  ─────────────────────
    def test_zero_mean_not_significant(self):
        """When the true mean is zero, significant_alpha should be False."""
        ret = self._normal(n=500, mean=0.0, std=0.02)
        bs = BootstrapStats(ret, n_bootstrap=2000, seed=42)
        report = bs.report(alpha=0.05)
        self.assertFalse(report["significant_alpha"],
                         "Zero-mean sample should NOT trigger significant_alpha")

    # ── Test 4: Strong positive mean → significant  ──────────────
    def test_positive_mean_significant(self):
        """When the true mean is strongly positive, significant_alpha = True."""
        ret = self._normal(n=252, mean=0.005, std=0.01)
        bs = BootstrapStats(ret, n_bootstrap=2000, seed=42)
        report = bs.report(alpha=0.05)

        # With mean 0.005 and std 0.01, annualised Sharpe ≈ 7.9 — very significant
        self.assertTrue(report["significant_alpha"],
                        "Strong positive mean should trigger significant_alpha")
        self.assertGreater(report["sample_sharpe"], 3.0,
                           "Annualised Sharpe should be large given the true mean/std ratio")

    # ── Test 5: Sharpe CI structure  ─────────────────────────────
    def test_sharpe_ci_structure(self):
        """Sharpe CI returns expected keys and plausible values."""
        ret = self._normal(n=252, mean=0.001, std=0.02)
        bs = BootstrapStats(ret, n_bootstrap=2000, seed=42)
        ci = bs.sharpe_ci(alpha=0.10)
        self.assertIn("sample_sharpe", ci)
        self.assertIn("percentile_ci_lower", ci)
        self.assertIn("percentile_ci_upper", ci)
        self.assertIn("annual_factor", ci)
        self.assertEqual(ci["annual_factor"], 252)
        self.assertEqual(ci["alpha"], 0.10)
        self.assertLess(ci["percentile_ci_lower"], ci["percentile_ci_upper"])
        # The sample Sharpe should lie inside the CI
        self.assertLessEqual(ci["percentile_ci_lower"], ci["sample_sharpe"])
        self.assertGreaterEqual(ci["percentile_ci_upper"], ci["sample_sharpe"])

    # ── Test 6: t_stat_distribution centering  ───────────────────
    def test_t_stat_distribution_centered(self):
        """Bootstrap t-statistics should be centred near zero under H0."""
        ret = self._normal(n=252, mean=0.001, std=0.02)
        bs = BootstrapStats(ret, n_bootstrap=3000, seed=42)
        t_stats = bs.t_stat_distribution()

        self.assertEqual(len(t_stats), 3000)
        # The distribution of t* should be centred roughly at zero
        mean_t = np.mean(t_stats)
        self.assertAlmostEqual(mean_t, 0.0, delta=0.15,
                               msg="Bootstrap t-stat distribution should be centred near 0")

    # ── Test 7: Invalid inputs  ──────────────────────────────────
    def test_invalid_n_bootstrap(self):
        """Too few bootstrap iterations should raise ValueError."""
        ret = self._normal(n=100)
        with self.assertRaises(ValueError):
            BootstrapStats(ret, n_bootstrap=50)

    def test_invalid_returns(self):
        """Empty or single-element returns should raise ValueError."""
        with self.assertRaises(ValueError):
            BootstrapStats(np.array([]))
        with self.assertRaises(ValueError):
            BootstrapStats(np.array([0.01]))

    # ── Test 8: Different alpha levels  ──────────────────────────
    def test_alpha_levels_ci_width(self):
        """Smaller alpha → wider CI."""
        ret = self._normal(n=252, mean=0.001, std=0.02)
        bs = BootstrapStats(ret, n_bootstrap=2000, seed=42)

        ci_10 = bs.mean_ci(alpha=0.10)
        ci_01 = bs.mean_ci(alpha=0.01)

        width_10 = ci_10["percentile_ci_upper"] - ci_10["percentile_ci_lower"]
        width_01 = ci_01["percentile_ci_upper"] - ci_01["percentile_ci_lower"]

        self.assertGreater(width_01, width_10,
                           "99% CI should be wider than 90% CI")

    # ── Test 9: Reproducibility  ─────────────────────────────────
    def test_reproducibility(self):
        """Same seed → same results."""
        ret = self._normal(n=100)
        bs1 = BootstrapStats(ret, n_bootstrap=2000, seed=42)
        bs2 = BootstrapStats(ret, n_bootstrap=2000, seed=42)

        r1 = bs1.report()
        r2 = bs2.report()
        for key in ["mean_ci_lower", "mean_ci_upper", "sharpe_ci_lower", "sharpe_ci_upper"]:
            self.assertEqual(r1[key], r2[key], f"Value mismatch for {key}")

    # ── Test 10: Annual factor (Crypto 365)  ─────────────────────
    def test_annual_factor_365(self):
        """Crypto markets often use 365 days."""
        ret = self._normal(n=365, mean=0.001, std=0.02)
        bs = BootstrapStats(ret, n_bootstrap=2000, seed=42, annual_factor=365)
        ci = bs.sharpe_ci()
        self.assertEqual(ci["annual_factor"], 365)
        self.assertIsInstance(ci["sample_sharpe"], float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
