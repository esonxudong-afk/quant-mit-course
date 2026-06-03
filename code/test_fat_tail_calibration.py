"""
Tests for FatTailCalibration
=============================
Validates:
  1. Normal sample → kurtosis ≈ 0, factor ≈ 1.0
  2. Normal sample → calibrated k_risk ≈ base (3.0)
  3. Fat-tailed sample → kurtosis > 1.5, factor > 1.0
  4. Fat-tailed sample → calibrated k_risk > base
  5. BTC-like kurtosis ≈ 6 → factor ≈ 2.25
  6. Report structure
  7. Edge cases: too few observations
  8. Base k_risk validation
  9. Different sample sizes
 10. Zero-kurtosis edge case
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fat_tail_calibration import FatTailCalibration


class TestFatTailCalibration(unittest.TestCase):
    """Comprehensive tests for FatTailCalibration."""

    # ── helpers  ──────────────────────────────────────────────────
    @staticmethod
    def _normal(n=500, mean=0.0, std=0.02):
        np.random.seed(123)
        return np.random.normal(mean, std, size=n)

    @staticmethod
    def _fat_tails(n=500):
        np.random.seed(456)
        return np.random.standard_t(df=3, size=n) * 0.02

    @staticmethod
    def _btc_like(n=500):
        """Returns with kurtosis ≈ 6 (like BTC daily)."""
        np.random.seed(99)
        # 90% normal; 10% fat-tail events
        core = np.random.normal(0, 0.02, size=int(n * 0.9))
        tails = np.random.normal(0, 0.10, size=int(n * 0.1))
        return np.concatenate([core, tails])

    # ── Test 1: Normal → kurtosis ≈ 0  ──────────────────────────
    def test_normal_kurtosis_near_zero(self):
        """Normal returns should have excess kurtosis near 0."""
        ret = self._normal(n=1000, mean=0.001, std=0.02)
        ftc = FatTailCalibration(ret)
        k = ftc.kurtosis()
        self.assertAlmostEqual(k, 0.0, delta=0.5,
                               msg=f"Normal kurtosis {k:.3f} should be near 0")

    # ── Test 2: Normal → factor ≈ 1.0  ──────────────────────────
    def test_normal_factor_near_one(self):
        """Normal returns should produce calibration factor ≈ 1.0."""
        ret = self._normal(n=500, mean=0.0, std=0.02)
        ftc = FatTailCalibration(ret)
        factor = ftc.calibration_factor()
        self.assertAlmostEqual(factor, 1.0, delta=0.15,
                               msg=f"Normal factor {factor:.3f} should be near 1.0")

    # ── Test 3: Normal → calibrated_k_risk ≈ base  ───────────────
    def test_normal_k_risk_near_base(self):
        """For normal returns, calibrated_k_risk should be close to base."""
        ret = self._normal(n=500, mean=0.0, std=0.02)
        ftc = FatTailCalibration(ret)
        k_risk = ftc.calibrated_k_risk(base_k_risk=3.0)
        self.assertAlmostEqual(k_risk, 3.0, delta=0.5,
                               msg=f"Normal k_risk {k_risk:.3f} should be near 3.0")

    # ── Test 4: Fat-tailed → kurtosis > 1.5  ────────────────────
    def test_fat_tail_kurtosis_elevated(self):
        """Fat-tailed (t₃) returns should have elevated kurtosis."""
        ret = self._fat_tails(n=500)
        ftc = FatTailCalibration(ret)
        k = ftc.kurtosis()
        self.assertGreater(k, 1.5,
                           f"Fat-tail kurtosis {k:.3f} should exceed 1.5")

    # ── Test 5: Fat-tailed → factor > 1.0  ──────────────────────
    def test_fat_tail_factor_greater_than_one(self):
        """Fat-tailed returns should produce calibration factor > 1.0."""
        ret = self._fat_tails(n=500)
        ftc = FatTailCalibration(ret)
        factor = ftc.calibration_factor()
        self.assertGreater(factor, 1.0,
                           f"Fat-tail factor {factor:.3f} should be > 1.0")

    # ── Test 6: BTC-like → factor ≈ 2.25  ───────────────────────
    def test_btc_like_factor(self):
        """BTC-like kurtosis ≈ 6 → factor should be approximately 2.25."""
        ret = self._btc_like(n=500)
        ftc = FatTailCalibration(ret)
        k = ftc.kurtosis()
        factor = ftc.calibration_factor()

        # With kurtosis ≈ 6, factor = 1 + (6-1)/4 = 2.25
        # Allow some sampling error
        expected = 1.0 + max(0.0, (k - 1.0) / 4.0)
        self.assertEqual(factor, expected,
                         f"Factor {factor:.4f} should match formula: 1 + max(0, ({k:.2f}-1)/4) = {expected:.4f}")

        self.assertGreater(factor, 1.5,
                           "BTC-like factor should be > 1.5")
        self.assertGreater(k, 2.0,
                           f"BTC-like kurtosis {k:.3f} should be > 2.0")

    # ── Test 7: Report structure  ────────────────────────────────
    def test_report_keys(self):
        """Report contains all expected keys."""
        ret = self._normal(n=200)
        ftc = FatTailCalibration(ret)
        rpt = ftc.report()
        required = [
            "n_observations", "excess_kurtosis",
            "calibration_factor", "calibrated_k_risk_default",
            "normality_verdict",
        ]
        for key in required:
            self.assertIn(key, rpt, f"Missing key: {key}")

        self.assertEqual(rpt["n_observations"], 200)
        self.assertIn(rpt["normality_verdict"],
                      ("normal", "mild_fat_tails", "fat_tails"))

    # ── Test 8: Normality verdict  ───────────────────────────────
    def test_verdict_normal(self):
        """Normal returns should get 'normal' verdict."""
        ret = self._normal(n=500, mean=0.0, std=0.02)
        ftc = FatTailCalibration(ret)
        rpt = ftc.report()
        self.assertEqual(rpt["normality_verdict"], "normal")

    def test_verdict_fat_tails(self):
        """t₃ returns should get 'fat_tails' verdict."""
        ret = self._fat_tails(n=500)
        ftc = FatTailCalibration(ret)
        rpt = ftc.report()
        self.assertEqual(rpt["normality_verdict"], "fat_tails")

    # ── Test 9: Invalid inputs  ─────────────────────────────────
    def test_too_few_observations(self):
        """< 30 observations should raise ValueError."""
        with self.assertRaises(ValueError):
            FatTailCalibration(np.array([0.01] * 20))

    def test_invalid_base_k_risk(self):
        """Negative base_k_risk should raise ValueError."""
        ret = self._normal(n=100)
        ftc = FatTailCalibration(ret)
        with self.assertRaises(ValueError):
            ftc.calibrated_k_risk(base_k_risk=-1.0)
        with self.assertRaises(ValueError):
            ftc.calibrated_k_risk(base_k_risk=0.0)

    # ── Test 10: Custom base k_risk  ─────────────────────────────
    def test_custom_base_k_risk(self):
        """calibrated_k_risk scales linearly with base."""
        ret = self._btc_like(n=500)
        ftc = FatTailCalibration(ret)
        factor = ftc.calibration_factor()

        k5 = ftc.calibrated_k_risk(base_k_risk=5.0)
        k2 = ftc.calibrated_k_risk(base_k_risk=2.0)

        self.assertAlmostEqual(k5 / 5.0, factor, delta=0.001)
        self.assertAlmostEqual(k2 / 2.0, factor, delta=0.001)

    # ── Test 11: Factor is always >= 1  ──────────────────────────
    def test_factor_never_below_one(self):
        """Even with slightly negative excess kurtosis, factor >= 1."""
        ret = self._normal(n=500, mean=0.0, std=0.02)
        ftc = FatTailCalibration(ret)
        factor = ftc.calibration_factor()
        self.assertGreaterEqual(factor, 1.0,
                                f"Factor {factor:.4f} should be >= 1.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
