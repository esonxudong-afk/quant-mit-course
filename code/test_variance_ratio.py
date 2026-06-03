"""Tests for variance_ratio_test.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from variance_ratio_test import (
    VarianceRatioTest,
    simulate_random_walk,
    simulate_mean_reverting,
    simulate_trending,
)


class TestVarianceRatioTest:
    """Tests for the VarianceRatioTest class."""

    def test_random_walk_vr_near_one(self):
        """VR(k) should be close to 1 for a random walk."""
        prices = simulate_random_walk(n=2000, seed=42)
        vrt = VarianceRatioTest(prices)
        vr, p = vrt.compute_vr(k=2)
        # VR should be close to 1 (within ±5%)
        assert abs(vr - 1.0) < 0.05, (
            f"VR for RW should be near 1, got {vr}, p={p}"
        )

    def test_mean_reverting_vr_less_than_one(self):
        """VR should be < 1 and significant for strong mean-reverting series."""
        prices = simulate_mean_reverting(n=2000, phi=0.7, seed=42)
        vrt = VarianceRatioTest(prices)
        vr, p = vrt.compute_vr(k=2)
        assert vr < 1.0, f"Mean-reverting series should have VR < 1, got {vr}"
        assert p < 0.05, f"Mean reversion should be significant, p={p}"
        assert vrt.is_mean_reverting(alpha=0.05), "is_mean_reverting should be True"

    def test_mean_reverting_multi_k(self):
        """Mean reversion signal should strengthen with larger k."""
        prices = simulate_mean_reverting(n=2000, phi=0.7, seed=42)
        vrt = VarianceRatioTest(prices)
        multi = vrt.compute_vr_multi(k_list=[2, 5, 10])
        # VR should decrease with k for mean-reverting series
        assert multi[2][0] > multi[10][0], (
            f"VR should decrease with k: VR(2)={multi[2][0]}, VR(10)={multi[10][0]}"
        )

    def test_trending_detected(self):
        """VR test should detect trending behavior (VR > 1)."""
        prices = simulate_trending(n=1000, drift=1.0002, seed=42)
        vrt = VarianceRatioTest(prices)
        vr, p = vrt.compute_vr(k=2)
        assert vr > 1.0, f"Trending series should have VR > 1, got {vr}"

    def test_multi_period_vr(self):
        """Multi-period VR should return a dict with expected keys."""
        prices = simulate_random_walk(n=1000, seed=42)
        vrt = VarianceRatioTest(prices)
        result = vrt.compute_vr_multi(k_list=[2, 5, 10])
        assert set(result.keys()) == {2, 5, 10}
        for vr, p in result.values():
            assert isinstance(vr, float)
            assert isinstance(p, float)
            assert 0 <= p <= 1

    def test_report_contains_all_keys(self):
        """Report should contain expected diagnostic keys."""
        prices = simulate_random_walk(n=500, seed=123)
        vrt = VarianceRatioTest(prices)
        report = vrt.report()
        required = ["vr_k2", "p_value_k2", "is_random_walk",
                     "is_mean_reverting", "is_trending", "interpretation",
                     "multi_vr", "n_observations"]
        for key in required:
            assert key in report, f"Missing key: {key}"

    def test_rejects_negative_prices(self):
        """Should raise ValueError for non-positive prices."""
        with pytest.raises(ValueError):
            VarianceRatioTest([100, -50, 200])

    def test_rejects_zero_prices(self):
        """Should raise ValueError for zero prices."""
        with pytest.raises(ValueError):
            VarianceRatioTest([100, 0, 200])

    def test_rejects_insufficient_data(self):
        """Should raise ValueError for too few observations."""
        with pytest.raises(ValueError):
            VarianceRatioTest([100, 200])  # only 2

    def test_rejects_invalid_k(self):
        """Should raise ValueError for k < 2."""
        prices = simulate_random_walk(n=100, seed=1)
        vrt = VarianceRatioTest(prices)
        with pytest.raises(ValueError):
            vrt.compute_vr(k=1)

    def test_rejects_k_too_large(self):
        """Should raise ValueError for k > number of returns."""
        prices = simulate_random_walk(n=50, seed=1)
        vrt = VarianceRatioTest(prices)
        with pytest.raises(ValueError):
            vrt.compute_vr(k=100)

    def test_prices_as_list(self):
        """Should accept list input."""
        prices = [100.0, 101.0, 102.5, 103.0, 101.5, 100.0]
        vrt = VarianceRatioTest(prices)
        vr, p = vrt.compute_vr(k=2)
        assert isinstance(vr, float)

    def test_log_prices_vs_prices_consistency(self):
        """The test uses log prices internally for consistency."""
        np.random.seed(42)
        prices = np.exp(np.cumsum(np.random.randn(100) * 0.01) + 3)
        vrt = VarianceRatioTest(prices)
        vr, p = vrt.compute_vr(k=5)
        assert 0 < vr < 3, f"VR out of reasonable range: {vr}"

    def test_is_trending_requires_significant(self):
        """is_trending should require both VR > 1 and p < alpha."""
        # For a near-random-walk, is_trending should be False
        prices = simulate_random_walk(n=2000, seed=99)
        vrt = VarianceRatioTest(prices)
        assert not vrt.is_trending(alpha=0.01), (
            "Random walk should not be flagged as trending"
        )
