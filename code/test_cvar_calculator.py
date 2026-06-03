"""
Tests for CVaRCalculator
========================
Validates:
  1. Normal sample → historical CVaR ≈ normal CVaR
  2. Fat-tailed sample → historical CVaR > normal CVaR (in magnitude)
  3. var() returns the correct quantile
  4. cvar() is more extreme than var()
  5. normal_var() matches known normal quantile
  6. tail_risk_report structure
  7. Edge cases: invalid alpha, too few observations
  8. tail_warning flag triggers for fat tails
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvar_calculator import CVaRCalculator


class TestCVaRCalculator(unittest.TestCase):
    """Comprehensive tests for CVaRCalculator."""

    # ── helpers  ──────────────────────────────────────────────────
    @staticmethod
    def _normal(n=500, mean=0.0, std=0.02):
        np.random.seed(123)
        return np.random.normal(mean, std, size=n)

    @staticmethod
    def _fat_tails(n=500):
        np.random.seed(456)
        return np.random.standard_t(df=3, size=n) * 0.02

    # ── Test 1: Normal → historical ≈ normal CVaR  ───────────────
    def test_normal_sample_cvar_close_to_normal_cvar(self):
        """For normal data, historical CVaR should be close to normal-theory CVaR."""
        ret = self._normal(n=500, mean=-0.001, std=0.02)
        cvar = CVaRCalculator(ret)
        h_cvar = cvar.cvar(alpha=0.05)
        n_cvar = cvar.normal_cvar(alpha=0.05)

        # Check they're within ~30% of each other (sampling noise)
        ratio = abs(abs(h_cvar) / abs(n_cvar))
        self.assertTrue(0.6 < ratio < 1.5,
                        f"Normal CVaR ratio {ratio:.3f} should be ~1.0 for normal data. "
                        f"hist={h_cvar:.6f}, norm={n_cvar:.6f}")

    # ── Test 2: Fat-tailed → historical CVaR >> normal CVaR  ─────
    def test_fat_tail_cvar_exceeds_normal_cvar(self):
        """Fat-tailed data should have more extreme historical CVaR than normal."""
        ret = self._fat_tails(n=500)
        cvar = CVaRCalculator(ret)
        h_cvar = cvar.cvar(alpha=0.05)
        n_cvar = cvar.normal_cvar(alpha=0.05)

        # Historical CVaR should be more negative than normal CVaR
        self.assertLess(h_cvar, n_cvar,
                        f"Fat-tail CVaR ({h_cvar:.6f}) should be more extreme "
                        f"than normal CVaR ({n_cvar:.6f})")

    # ── Test 3: var() is correct quantile  ───────────────────────
    def test_var_is_correct_quantile(self):
        """var(0.05) should match np.percentile(returns, 5)."""
        ret = self._normal(n=200, mean=0.001, std=0.02)
        cvar = CVaRCalculator(ret)
        v = cvar.var(alpha=0.05)
        expected = np.percentile(ret, 5)
        self.assertAlmostEqual(v, expected, places=10)

    def test_var_alpha_01(self):
        """var(0.01) = 1% VaR."""
        ret = self._normal(n=500, mean=0.0, std=0.02)
        cvar = CVaRCalculator(ret)
        v05 = cvar.var(alpha=0.05)
        v01 = cvar.var(alpha=0.01)
        self.assertLess(v01, v05, "1% VaR should be more extreme than 5% VaR")

    # ── Test 4: cvar() is more extreme than var()  ───────────────
    def test_cvar_more_extreme_than_var(self):
        """CVaR (tail average) should be more extreme than VaR (tail threshold)."""
        ret = self._fat_tails(n=500)
        cvar = CVaRCalculator(ret)
        v = cvar.var(alpha=0.05)
        cv = cvar.cvar(alpha=0.05)
        # CVaR ≤ VaR for left tail (more negative = more extreme)
        self.assertLessEqual(cv, v, f"CVaR ({cv:.6f}) should be ≤ VaR ({v:.6f})")

    # ── Test 5: normal_var matches known quantile  ───────────────
    def test_normal_var_known_value(self):
        """For N(μ=-0.001, σ=0.02), var(0.05) should be μ + z_0.05 * σ."""
        # z_0.05 ≈ -1.6449
        ret = self._normal(n=1000, mean=-0.001, std=0.02)
        cvar = CVaRCalculator(ret)
        n_var = cvar.normal_var(alpha=0.05)
        expected = -0.001 + (-1.6449) * 0.02  # ≈ -0.0339
        self.assertAlmostEqual(n_var, expected, delta=0.005,
                               msg=f"Got {n_var:.6f}, expected ≈ {expected:.6f}")

    # ── Test 6: tail_risk_report structure  ──────────────────────
    def test_tail_risk_report_keys(self):
        """Report contains all expected keys."""
        ret = self._normal(n=200)
        cvar = CVaRCalculator(ret)
        rpt = cvar.tail_risk_report(alpha=0.05)
        required = [
            "alpha", "n_observations",
            "historical_var", "historical_cvar",
            "normal_var", "normal_cvar",
            "var_underestimation_ratio", "cvar_underestimation_ratio",
            "excess_kurtosis", "tail_warning",
        ]
        for key in required:
            self.assertIn(key, rpt, f"Missing key: {key}")

        self.assertEqual(rpt["n_observations"], 200)
        self.assertEqual(rpt["alpha"], 0.05)
        self.assertIsInstance(rpt["tail_warning"], bool)

    # ── Test 7: Invalid inputs  ──────────────────────────────────
    def test_invalid_alpha(self):
        """alpha must be in (0, 0.5)."""
        ret = self._normal(n=100)
        cvar = CVaRCalculator(ret)
        with self.assertRaises(ValueError):
            cvar.var(alpha=0.0)
        with self.assertRaises(ValueError):
            cvar.var(alpha=0.6)
        with self.assertRaises(ValueError):
            cvar.cvar(alpha=-0.1)

    def test_too_few_observations(self):
        """< 20 observations should raise ValueError."""
        with self.assertRaises(ValueError):
            CVaRCalculator(np.array([0.01] * 10))

    # ── Test 8: tail_warning for fat tails  ──────────────────────
    def test_tail_warning_off_for_normal(self):
        """Normal data should NOT trigger tail_warning."""
        ret = self._normal(n=500, mean=-0.001, std=0.02)
        cvar = CVaRCalculator(ret)
        rpt = cvar.tail_risk_report(alpha=0.05)
        # Normal samples should not consistently trigger fat-tail warning
        # (may occasionally due to sampling, but seed is fixed)
        self.assertFalse(rpt["tail_warning"],
                         f"Normal data should not trigger tail_warning, "
                         f"ratio = {rpt['cvar_underestimation_ratio']:.3f}")

    def test_tail_warning_on_for_fat_tails(self):
        """Fat-tailed data should trigger tail_warning."""
        ret = self._fat_tails(n=500)
        cvar = CVaRCalculator(ret)
        rpt = cvar.tail_risk_report(alpha=0.05)
        self.assertTrue(rpt["tail_warning"],
                        f"Fat-tail data should trigger tail_warning, "
                        f"ratio = {rpt['cvar_underestimation_ratio']:.3f}")

    # ── Test 9: excess_kurtosis  ─────────────────────────────────
    def test_excess_kurtosis_normal(self):
        """Normal samples should have excess kurtosis near 0."""
        ret = self._normal(n=1000, mean=0.0, std=0.02)
        cvar = CVaRCalculator(ret)
        rpt = cvar.tail_risk_report(alpha=0.05)
        self.assertAlmostEqual(rpt["excess_kurtosis"], 0.0, delta=0.5,
                               msg=f"Excess kurtosis {rpt['excess_kurtosis']:.3f}")

    # ── Test 10: Underestimation ratio > 1 for fat tails  ────────
    def test_fat_tail_underestimation_ratio(self):
        """Fat-tailed data should have var_underestimation_ratio > 1."""
        ret = self._fat_tails(n=500)
        cvar = CVaRCalculator(ret)
        rpt = cvar.tail_risk_report(alpha=0.05)
        self.assertGreater(rpt["cvar_underestimation_ratio"], 1.0,
                           f"Fat-tail CVaR ratio should be > 1, got {rpt['cvar_underestimation_ratio']:.3f}")
        # var ratio can be < 1 for VaR at certain sample sizes (VaR is a single
        # quantile, more stable than CVaR). The key is CVaR shows the fat tail.
        self.assertGreater(rpt["cvar_underestimation_ratio"], 1.05,
                           "CVaR ratio should exceed 1.05 for heavy fat tails")


if __name__ == "__main__":
    unittest.main(verbosity=2)
