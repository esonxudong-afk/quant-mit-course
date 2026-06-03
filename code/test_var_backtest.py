"""
Tests for var_backtest.py — VaRBacktest (Kupiec & Christoffersen)

Coverage:
- Correctly calibrated VaR → Kupiec does NOT reject H₀
- Underestimated VaR → Kupiec REJECTS H₀
- Overestimated VaR (zero violations) → still valid (conservative)
- Christoffersen independence test
- Edge cases (short data, mismatched lengths, negative VaR, etc.)
"""

import numpy as np
import pytest
from var_backtest import VaRBacktest
from var_calculator import VaRCalculator


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def calibrated_data():
    """
    Generate returns and a correctly calibrated VaR series.
    Returns ~ N(0, 0.02); VaR fixed at Normal 95% level → should see ~5% violations.
    """
    rng = np.random.default_rng(54321)
    returns = rng.normal(0.0, 0.02, 2000)
    # Correct 95% VaR: σ * z_0.05 = 0.02 * 1.64485 ≈ 0.0329
    var = np.full(2000, 0.02 * 1.64485)
    return returns, var


@pytest.fixture
def underestimated_var():
    """
    VaR too small → violation rate >> α → Kupiec should reject.
    VaR set to σ * z_0.30 (only 70% coverage). Returns are N(0, 0.02).
    """
    rng = np.random.default_rng(99999)
    returns = rng.normal(0.0, 0.02, 2000)
    # 70% VaR → α=0.30, z_0.30 = -0.5244 → VaR = σ * 0.5244 = 0.0105
    var = np.full(2000, 0.02 * 0.5244)
    return returns, var


@pytest.fixture
def overestimated_var():
    """
    VaR too large → near-zero violations → conservative model.
    VaR set to σ * z_0.001 (99.9% coverage).
    """
    rng = np.random.default_rng(11111)
    returns = rng.normal(0.0, 0.02, 2000)
    var = np.full(2000, 0.02 * 3.09)  # ~99.9% VaR
    return returns, var


@pytest.fixture
def clustered_violations():
    """
    Artificially create clustered violations for Christoffersen test.
    Returns: alternating blocks of extreme negative returns (violations)
    and zero returns (no violations).
    """
    rng = np.random.default_rng(77777)
    T = 1000
    returns = np.zeros(T)
    var = np.full(T, 0.03)

    # Create 5 blocks of 10 violations each = 50 violations (5% rate)
    for block in range(5):
        start = 150 + block * 150
        returns[start : start + 10] = -0.05  # Loss > VaR → violation

    return returns, var


# ═══════════════════════════════════════════════════════════════════
# Kupiec tests
# ═══════════════════════════════════════════════════════════════════

class TestKupiecCalibrated:
    """Tests for correctly calibrated VaR."""

    def test_calibrated_var_not_rejected(self, calibrated_data):
        """Correctly calibrated VaR should NOT be rejected by Kupiec test at 5%."""
        returns, var = calibrated_data
        bt = VaRBacktest(returns, var, alpha=0.05)
        result = bt.kupiec_test()
        assert not result["reject_5pct"], (
            f"Calibrated VaR incorrectly rejected: p={result['p_value']:.4f}"
        )

    def test_violation_rate_near_alpha(self, calibrated_data):
        """Violation rate should be close to alpha for a calibrated model."""
        returns, var = calibrated_data
        bt = VaRBacktest(returns, var, alpha=0.05)
        rate = bt.violation_rate()
        # With 2000 obs, rate should be within ~2% of 0.05
        assert 0.02 < rate < 0.09, f"violation_rate={rate:.4f}"

    def test_expected_violations_equals_alpha_times_T(self, calibrated_data):
        bt = VaRBacktest(calibrated_data[0], calibrated_data[1], alpha=0.05)
        assert bt.expected_violations() == 100.0  # 0.05 * 2000

    def test_expected_violations_at_different_alpha(self, calibrated_data):
        returns, var = calibrated_data
        bt = VaRBacktest(returns, var, alpha=0.01)
        assert bt.expected_violations() == 20.0  # 0.01 * 2000


class TestKupiecRejection:
    """Tests where VaR is clearly mis-calibrated."""

    def test_underestimated_var_is_rejected(self, underestimated_var):
        """VaR too small → too many violations → Kupiec should reject."""
        returns, var = underestimated_var
        bt = VaRBacktest(returns, var, alpha=0.05)
        rate = bt.violation_rate()
        # With 70% coverage, violation rate should be ~30%
        assert rate > 0.15, f"Expected high violation rate, got {rate:.4f}"
        result = bt.kupiec_test()
        assert result["reject_5pct"], (
            f"Underestimated VaR should be rejected: p={result['p_value']:.4f}"
        )

    def test_overestimated_var_has_low_violation_rate(self, overestimated_var):
        """VaR too large → very few violations → conservative."""
        returns, var = overestimated_var
        bt = VaRBacktest(returns, var, alpha=0.05)
        rate = bt.violation_rate()
        assert rate < 0.02, f"Expected very low violation rate, got {rate:.4f}"


# ═══════════════════════════════════════════════════════════════════
# Christoffersen tests
# ═══════════════════════════════════════════════════════════════════

class TestChristoffersen:
    """Tests for the conditional coverage test."""

    def test_random_violations_not_clustered(self, calibrated_data):
        """Random (i.i.d.) violations should not show clustering."""
        returns, var = calibrated_data
        bt = VaRBacktest(returns, var, alpha=0.05)
        chris = bt.christoffersen_test()
        # Independence should NOT be rejected for random violations
        assert not chris["reject_ind_5pct"], (
            f"Independence rejected for random data: p_ind={chris['p_ind']:.4f}"
        )

    def test_clustered_violations_detected(self, clustered_violations):
        """Clustered violations should fail the independence test."""
        returns, var = clustered_violations
        bt = VaRBacktest(returns, var, alpha=0.05)
        chris = bt.christoffersen_test()
        # With clustered violations, independence should be rejected
        assert chris["reject_ind_5pct"], (
            f"Independence NOT rejected for clustered data: p_ind={chris['p_ind']:.4f}"
        )

    def test_transition_counts_sum_correctly(self, calibrated_data):
        """n00 + n01 + n10 + n11 = T - 1."""
        returns, var = calibrated_data
        bt = VaRBacktest(returns, var, alpha=0.05)
        chris = bt.christoffersen_test()
        tc = chris["transition_counts"]
        total = tc["n00"] + tc["n01"] + tc["n10"] + tc["n11"]
        assert total == bt.T - 1, f"Transitions sum to {total}, expected {bt.T - 1}"


# ═══════════════════════════════════════════════════════════════════
# Integration: VaRCalculator → VaRBacktest
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end: compute VaR, then backtest it."""

    def test_var_calculator_to_backtest(self):
        """Generate returns, compute VaR, backtest → calibrated VaR passes."""
        rng = np.random.default_rng(12345)
        returns = rng.normal(0.0, 0.02, 1500)

        # Use VaRCalculator to get VaR estimate
        calc = VaRCalculator(returns, position_value=1.0)
        var_estimate = calc.normal_var(0.05)

        # Create VaR series (constant)
        var_series = np.full(1500, var_estimate)

        # Backtest
        bt = VaRBacktest(returns, var_series, alpha=0.05)
        result = bt.kupiec_test()

        assert not result["reject_5pct"], (
            f"Calibrated VaR rejected in end-to-end: p={result['p_value']:.4f}"
        )

    def test_backtest_with_varying_var(self):
        """Backtest with a rolling-window VaR (simulated here with noise)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, 1000)

        # Simulate a rolling VaR: base + noise
        base_var = 0.02 * 1.64485
        var_series = np.full(1000, base_var) + rng.normal(0, 0.001, 1000)
        var_series = np.abs(var_series)  # Ensure non-negative

        bt = VaRBacktest(returns, var_series, alpha=0.05)
        k = bt.kupiec_test()
        # Should still be roughly calibrated
        assert not k["reject_5pct"], (
            f"Rolling VaR rejected: p={k['p_value']:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Input validation and edge cases."""

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            VaRBacktest(
                np.random.normal(0, 1, 100),
                np.random.normal(0, 1, 99),
            )

    def test_raises_on_short_series(self):
        with pytest.raises(ValueError, match="at least 20"):
            VaRBacktest(
                np.random.normal(0, 1, 10),
                np.full(10, 0.03),
            )

    def test_raises_on_invalid_alpha(self):
        r = np.random.normal(0, 1, 100)
        v = np.full(100, 0.03)
        with pytest.raises(ValueError, match="alpha"):
            VaRBacktest(r, v, alpha=0.6)
        with pytest.raises(ValueError, match="alpha"):
            VaRBacktest(r, v, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            VaRBacktest(r, v, alpha=-0.1)

    def test_raises_on_negative_var(self):
        r = np.random.normal(0, 1, 100)
        v = np.full(100, -0.01)
        with pytest.raises(ValueError, match="non-negative"):
            VaRBacktest(r, v)

    def test_no_violations_lr_computes(self):
        """Edge case: zero violations — LR should still be finite."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 500)
        # VaR so large there are no violations
        var = np.full(500, 1.0)
        bt = VaRBacktest(returns, var, alpha=0.05)
        k = bt.kupiec_test()
        # LR should be finite and p_value should be very small (model is wrong)
        assert np.isfinite(k["lr_stat"])
        assert k["reject_5pct"]  # 0 violations when expecting 5% → rejected

    def test_all_violations_lr_computes(self):
        """Edge case: all violations — LR should still be finite."""
        rng = np.random.default_rng(42)
        returns = -np.abs(rng.normal(0.0, 0.05, 500))  # All negative returns
        var = np.full(500, 0.0)  # Zero VaR → all violations
        bt = VaRBacktest(returns, var, alpha=0.05)
        k = bt.kupiec_test()
        assert np.isfinite(k["lr_stat"])
        assert k["reject_5pct"]

    def test_report_contains_all_keys(self, calibrated_data):
        returns, var = calibrated_data
        bt = VaRBacktest(returns, var, alpha=0.05)
        rpt = bt.report()
        expected_top = [
            "n_observations", "alpha", "expected_violations",
            "actual_violations", "violation_rate", "kupiec", "christoffersen",
        ]
        for k in expected_top:
            assert k in rpt, f"Missing key: {k}"
        assert "lr_stat" in rpt["kupiec"]
        assert "lr_cc" in rpt["christoffersen"]

    def test_2d_array_is_flattened(self, calibrated_data):
        """2-D arrays (e.g., (N,1)) should be accepted and flattened."""
        returns = calibrated_data[0].reshape(-1, 1)
        var = calibrated_data[1].reshape(-1, 1)
        bt = VaRBacktest(returns, var, alpha=0.05)
        # Should not raise
        assert bt.T == 2000
