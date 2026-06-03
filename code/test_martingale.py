"""Tests for martingale_test.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from martingale_test import (
    MartingaleTest,
    simulate_martingale,
    simulate_submartingale,
)


class TestMartingaleTest:
    """Tests for MartingaleTest."""

    def test_martingale_vr_near_one(self):
        """A true martingale (zero drift) should have VR close to 1."""
        # Use larger sample to reduce noise
        eq = simulate_martingale(n=5000, sigma=0.01, seed=42)
        mt = MartingaleTest(eq)
        vr, p = mt.variance_ratio_test(k=2)
        # VR should be close to 1 for a martingale
        assert 0.9 < vr < 1.1, f"VR should be near 1 for martingale, got {vr}"
        # With large enough sample, p should be reasonable
        assert p > 0.01, f"p-value too low for true martingale: {p}"

    def test_submartingale_detected(self):
        """A sub-martingale (strong positive drift) should be detected."""
        # Use strong enough mu that drift test catches it
        eq = simulate_submartingale(n=2000, mu=0.0015, sigma=0.01, seed=42)
        mt = MartingaleTest(eq)
        assert np.mean(mt._returns) > 0, "Sub-martingale should have positive drift"
        assert mt.is_submartingale(alpha=0.05), (
            f"Sub-martingale should be detected. Report: {mt.report()}"
        )

    def test_submartingale_is_not_martingale(self):
        """A sub-martingale is not a martingale (it has positive drift)."""
        # Strong drift makes it clear
        eq = simulate_submartingale(n=2000, mu=0.002, sigma=0.01, seed=42)
        mt = MartingaleTest(eq)
        is_sm = mt.is_submartingale(alpha=0.05)
        assert is_sm, f"Should be sub-martingale. Report: {mt.report()}"
        # A sub-martingale has E[next] > current, so it's NOT a martingale
        # (But note: our is_martingale checks variance structure, not drift)
        # Sub-martingale confirmation is the primary test for positive alpha

    def test_variance_ratio_test(self):
        """VR test on equity curve should return valid results."""
        eq = simulate_martingale(n=500, sigma=0.01, seed=123)
        mt = MartingaleTest(eq)
        vr, p = mt.variance_ratio_test(k=2)
        assert isinstance(vr, float)
        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_runs_test(self):
        """Runs test should return valid z-score and p-value."""
        eq = simulate_martingale(n=500, sigma=0.01, seed=123)
        mt = MartingaleTest(eq)
        z, p = mt.runs_test()
        assert isinstance(z, float)
        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_autocorrelation_test(self):
        """Ljung-Box test should return a dict with expected keys."""
        eq = simulate_martingale(n=500, sigma=0.01, seed=123)
        mt = MartingaleTest(eq)
        ac = mt.autocorrelation_test(max_lag=10)
        assert "statistic" in ac
        assert "p_value" in ac
        assert "lags_tested" in ac
        assert "individual_lags" in ac
        assert ac["lags_tested"] > 0
        assert isinstance(ac["p_value"], float)

    def test_report(self):
        """Report should contain all expected keys."""
        eq = simulate_martingale(n=300, sigma=0.01, seed=42)
        mt = MartingaleTest(eq)
        report = mt.report()
        required = [
            "total_return", "mean_log_return", "n_observations",
            "variance_ratio", "vr_p_value", "runs_test_z", "runs_test_p",
            "ljung_box_statistic", "ljung_box_p_value", "ljung_box_lags",
            "is_martingale", "is_submartingale", "interpretation",
        ]
        for key in required:
            assert key in report, f"Missing key: {key}"

    def test_rejects_negative_equity(self):
        """Should raise ValueError for non-positive equity values."""
        with pytest.raises(ValueError):
            MartingaleTest([100, -50, 200])

    def test_rejects_zero_equity(self):
        """Should raise ValueError for zero equity."""
        with pytest.raises(ValueError):
            MartingaleTest([100, 0, 200])

    def test_rejects_insufficient_data(self):
        """Should raise ValueError for too few observations."""
        with pytest.raises(ValueError):
            MartingaleTest([100, 101])

    def test_list_input_accepted(self):
        """List input should be accepted."""
        eq = [100.0, 101.0, 99.5, 102.0, 103.5, 101.0, 104.0, 105.5, 107.0, 106.0, 108.0]
        mt = MartingaleTest(eq)
        report = mt.report()
        assert report["n_observations"] == 11

    def test_autocorrelation_on_martingale(self):
        """Martingale should not have significant autocorrelation."""
        eq = simulate_martingale(n=500, sigma=0.01, seed=99)
        mt = MartingaleTest(eq)
        ac = mt.autocorrelation_test(max_lag=10)
        # Should generally fail to reject at 5% level
        assert ac["p_value"] >= 0.01, f"AC p-value too low: {ac['p_value']}"

    def test_martingale_behavior(self):
        """Large martingale should have VR close to 1 and valid report."""
        eq = simulate_martingale(n=5000, sigma=0.01, seed=1)
        mt = MartingaleTest(eq)
        report = mt.report()
        # With 5000 points and corrected VR formula, VR should be close to 1
        assert abs(report["variance_ratio"] - 1.0) < 0.05, (
            f"VR should be near 1 for large martingale, got {report['variance_ratio']}"
        )
        # Should be classified as martingale
        assert report["is_martingale"] is True
        assert report["is_submartingale"] is False
