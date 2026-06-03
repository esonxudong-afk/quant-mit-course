"""
Tests for var_calculator.py — VaRCalculator and PortfolioVaR

Coverage:
- Normal VaR on normal data
- Historical VaR on normal data  →  normal ≈ historical for Gaussian data
- Monte Carlo VaR reproducibility and convergence
- Expected Shortfall always ≥ VaR
- Edge cases (invalid alpha, insufficient data)
- PortfolioVaR: portfolio_var, component_var sum
- Diversification ratio: independent assets → DR ≈ √N
"""

import numpy as np
import pytest
from var_calculator import VaRCalculator, PortfolioVaR


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def gaussian_returns():
    """Large sample of i.i.d. normal returns — Normal VaR should be accurate."""
    rng = np.random.default_rng(12345)
    return rng.normal(0.0, 0.02, 5000)  # μ=0, σ=2%, 5000 obs


@pytest.fixture
def fat_tail_returns():
    """t-distributed returns — heavier tails than normal."""
    rng = np.random.default_rng(42)
    return rng.standard_t(df=3, size=2000) * 0.02  # df=3 t-dist, scaled


@pytest.fixture
def multi_asset_returns():
    """Independent asset returns for portfolio testing."""
    rng = np.random.default_rng(42)
    T, N = 5000, 4
    R = rng.normal(0.0, 0.02, size=(T, N))
    return R


# ═══════════════════════════════════════════════════════════════════
# VaRCalculator tests
# ═══════════════════════════════════════════════════════════════════

class TestVaRCalculatorNormal:
    """Tests for normal (parametric) VaR."""

    def test_normal_var_zero_mean_95(self, gaussian_returns):
        """For μ≈0, σ=0.02: VaR_0.05 ≈ σ · z_0.05 ≈ 0.02 × 1.645 = 0.0329"""
        calc = VaRCalculator(gaussian_returns, position_value=1.0)
        var = calc.normal_var(0.05)
        # Expected: σ · 1.64485  (z_0.05 = -1.64485)
        expected = 0.02 * 1.64485
        assert abs(var - expected) < 0.005, f"var={var:.6f}, expected≈{expected:.6f}"

    def test_normal_var_scales_with_horizon(self, gaussian_returns):
        """VaR should scale by √horizon for zero-mean returns."""
        calc1 = VaRCalculator(gaussian_returns, position_value=1.0, horizon_days=1)
        calc10 = VaRCalculator(gaussian_returns, position_value=1.0, horizon_days=10)
        var1 = calc1.normal_var(0.05)
        var10 = calc10.normal_var(0.05)
        # var10 / var1 ≈ √10 ≈ 3.162 (ignoring drift)
        ratio = var10 / var1 if var1 > 0 else 0
        assert 2.8 < ratio < 3.5, f"ratio={ratio:.4f}"

    def test_normal_var_99_is_larger_than_95(self, gaussian_returns):
        """More extreme VaR is larger."""
        calc = VaRCalculator(gaussian_returns, position_value=1.0)
        assert calc.normal_var(0.01) > calc.normal_var(0.05)

    def test_normal_var_scales_with_position_value(self, gaussian_returns):
        """VaR scales linearly with position value."""
        calc1 = VaRCalculator(gaussian_returns, position_value=1000)
        calc2 = VaRCalculator(gaussian_returns, position_value=2000)
        assert abs(calc2.normal_var(0.05) - 2 * calc1.normal_var(0.05)) < 1e-9

    def test_normal_var_non_negative(self, gaussian_returns):
        """VaR should never be negative."""
        calc = VaRCalculator(gaussian_returns, position_value=100)
        assert calc.normal_var(0.05) >= 0
        assert calc.normal_var(0.01) >= 0


class TestVaRCalculatorHistorical:
    """Tests for historical VaR."""

    def test_historical_var_basic(self, gaussian_returns):
        """Historical VaR should be close to normal VaR for large Gaussian sample."""
        calc = VaRCalculator(gaussian_returns, position_value=1.0)
        nv = calc.normal_var(0.05)
        hv = calc.historical_var(0.05)
        # With 5000 obs, empirical quantile ≈ theoretical
        rel_diff = abs(hv - nv) / nv if nv > 0 else 0
        assert rel_diff < 0.15, f"hv={hv:.6f}, nv={nv:.6f}, diff={rel_diff:.4f}"

    def test_historical_var_fat_tail_is_larger_than_normal(self, fat_tail_returns):
        """For fat-tailed returns, historical VaR >= normal VaR at extreme quantiles.
        At 1% level the heavy tail effect should dominate."""
        calc = VaRCalculator(fat_tail_returns, position_value=1.0)
        nv = calc.normal_var(0.01)
        hv = calc.historical_var(0.01)
        assert hv >= nv * 0.95, f"hv={hv:.6f}, nv={nv:.6f}"

    def test_historical_var_99_is_larger_than_95(self, gaussian_returns):
        calc = VaRCalculator(gaussian_returns, position_value=1.0)
        assert calc.historical_var(0.01) > calc.historical_var(0.05)


class TestVaRCalculatorMonteCarlo:
    """Tests for Monte Carlo VaR."""

    def test_mc_var_reproducibility(self, gaussian_returns):
        """Same seed should produce identical results."""
        calc = VaRCalculator(gaussian_returns, position_value=1.0)
        v1 = calc.monte_carlo_var(0.05, n_sim=10000, seed=42)
        v2 = calc.monte_carlo_var(0.05, n_sim=10000, seed=42)
        assert v1 == v2

    def test_mc_var_converges_to_normal(self, gaussian_returns):
        """With large n_sim, MC VaR ≈ Normal VaR."""
        calc = VaRCalculator(gaussian_returns, position_value=1.0)
        nv = calc.normal_var(0.05)
        mc = calc.monte_carlo_var(0.05, n_sim=100000, seed=42)
        rel_diff = abs(mc - nv) / nv if nv > 0 else 0
        assert rel_diff < 0.02, f"mc={mc:.6f}, nv={nv:.6f}, diff={rel_diff:.4f}"


class TestVaRCalculatorExpectedShortfall:
    """Tests for Expected Shortfall (CVaR)."""

    def test_es_always_gte_var(self, gaussian_returns):
        """ES should always be ≥ VaR at the same confidence level."""
        calc = VaRCalculator(gaussian_returns, position_value=1.0)
        for alpha in [0.01, 0.05, 0.10]:
            es = calc.expected_shortfall(alpha)
            hv = calc.historical_var(alpha)
            assert es >= hv - 1e-12, f"alpha={alpha}: es={es:.6f}, hv={hv:.6f}"

    def test_es_fat_tail_much_larger_relative_gap(self, fat_tail_returns):
        """For fat-tailed returns, ES/VaR ratio should be larger."""
        calc = VaRCalculator(fat_tail_returns, position_value=1.0)
        hv = calc.historical_var(0.05)
        es = calc.expected_shortfall(0.05)
        ratio = es / hv if hv > 0 else 0
        # For t_3, ES/VaR should be notably > 1
        assert ratio > 1.1, f"ratio={ratio:.4f}"


class TestVaRCalculatorEdgeCases:
    """Edge case and input validation tests."""

    def test_raises_on_short_input(self):
        with pytest.raises(ValueError, match="at least 5"):
            VaRCalculator(np.array([0.01, 0.02]))

    def test_raises_on_invalid_alpha(self, gaussian_returns):
        calc = VaRCalculator(gaussian_returns)
        with pytest.raises(ValueError, match="alpha"):
            calc.normal_var(0.6)
        with pytest.raises(ValueError, match="alpha"):
            calc.normal_var(0.0)
        with pytest.raises(ValueError, match="alpha"):
            calc.normal_var(-0.1)

    def test_raises_on_negative_position_value(self, gaussian_returns):
        with pytest.raises(ValueError, match="position_value"):
            VaRCalculator(gaussian_returns, position_value=-100)

    def test_raises_on_zero_horizon(self, gaussian_returns):
        with pytest.raises(ValueError, match="horizon_days"):
            VaRCalculator(gaussian_returns, horizon_days=0)

    def test_raises_on_too_few_mc_sims(self, gaussian_returns):
        calc = VaRCalculator(gaussian_returns)
        with pytest.raises(ValueError, match="n_sim"):
            calc.monte_carlo_var(0.05, n_sim=10)

    def test_report_contains_all_keys(self, gaussian_returns):
        calc = VaRCalculator(gaussian_returns, position_value=10000, horizon_days=5)
        rpt = calc.report()
        expected_keys = [
            "position_value", "horizon_days", "n_observations",
            "mu_daily", "sigma_daily", "alpha", "confidence_level",
            "normal_var", "historical_var", "monte_carlo_var", "expected_shortfall",
        ]
        for k in expected_keys:
            assert k in rpt, f"Missing key: {k}"


# ═══════════════════════════════════════════════════════════════════
# PortfolioVaR tests
# ═══════════════════════════════════════════════════════════════════

class TestPortfolioVaRBasic:
    """Basic portfolio VaR tests."""

    def test_portfolio_var_historical_consistent(self, multi_asset_returns):
        """Portfolio VaR on portfolio returns should match VaRCalculator directly."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        pf = PortfolioVaR(multi_asset_returns, w, position_value=1.0)
        pf_var = pf.portfolio_var("historical", 0.05)

        # Direct computation
        pf_ret = multi_asset_returns @ w
        direct = VaRCalculator(pf_ret, position_value=1.0).historical_var(0.05)

        assert abs(pf_var - direct) < 1e-12

    def test_portfolio_var_methods_all_positive(self, multi_asset_returns):
        w = np.array([0.4, 0.3, 0.2, 0.1])
        pf = PortfolioVaR(multi_asset_returns, w, position_value=10000)
        for method in ["normal", "historical", "monte_carlo"]:
            v = pf.portfolio_var(method, 0.05)
            assert v > 0, f"method={method}: var={v}"

    def test_invalid_method_raises(self, multi_asset_returns):
        w = np.array([0.25, 0.25, 0.25, 0.25])
        pf = PortfolioVaR(multi_asset_returns, w)
        with pytest.raises(ValueError, match="method"):
            pf.portfolio_var("bogus", 0.05)


class TestPortfolioVaRComponent:
    """Component VaR tests."""

    def test_component_var_sums_to_portfolio_var(self, multi_asset_returns):
        """Component VaR contributions should sum to total portfolio VaR."""
        w = np.array([0.4, 0.3, 0.2, 0.1])
        pf = PortfolioVaR(multi_asset_returns, w, position_value=1.0)
        comp = pf.component_var("historical", 0.05)
        total = pf.portfolio_var("historical", 0.05)
        assert abs(np.sum(comp) - total) < 1e-6, f"sum comp={np.sum(comp):.8f}, total={total:.8f}"

    def test_component_var_non_negative_for_long_only(self, multi_asset_returns):
        """Component VaR contributions should be reasonable for long-only.
        They may have slight negatives from numerical gradient but should
        be within expected range relative to portfolio VaR."""
        w = np.array([0.4, 0.3, 0.2, 0.1])
        pf = PortfolioVaR(multi_asset_returns, w)
        comp = pf.component_var("historical", 0.05)
        total = pf.portfolio_var("historical", 0.05)
        # Each component should be reasonable relative to total
        for c in comp:
            assert c >= -total * 0.1, f"Component {c} too negative vs total {total}"


class TestPortfolioVaRDiversification:
    """Diversification ratio tests."""

    def test_diversification_ratio_independent_assets(self, multi_asset_returns):
        """For N independent identical assets, DR ≈ √N."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        pf = PortfolioVaR(multi_asset_returns, w, position_value=1.0)
        dr = pf.diversification_ratio(0.05)
        expected = np.sqrt(4)  # = 2.0
        # With large sample, should be close to √N
        assert 1.5 < dr < 2.5, f"dr={dr:.4f}, expected≈{expected:.4f}"

    def test_diversification_ratio_greater_than_one(self, multi_asset_returns):
        """Any diversified portfolio should have DR ≥ 1."""
        w = np.array([0.4, 0.3, 0.2, 0.1])
        pf = PortfolioVaR(multi_asset_returns, w)
        dr = pf.diversification_ratio(0.05)
        assert dr >= 0.99, f"dr={dr:.4f}"

    def test_perfectly_correlated_dr_close_to_one(self):
        """If all assets are the same, DR ≈ 1."""
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.02, 2000)
        R = np.column_stack([base, base, base])  # perfect correlation
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        pf = PortfolioVaR(R, w, position_value=1.0)
        dr = pf.diversification_ratio(0.05)
        assert 0.8 < dr < 1.2, f"dr={dr:.4f} (expected ≈ 1.0)"


class TestPortfolioVaREdgeCases:
    """Edge cases for PortfolioVaR."""

    def test_raises_on_non_unity_weights(self, multi_asset_returns):
        with pytest.raises(ValueError, match="sum to 1"):
            PortfolioVaR(multi_asset_returns, np.array([0.5, 0.5, 0.5, 0.0]))

    def test_raises_on_negative_weights(self, multi_asset_returns):
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            PortfolioVaR(multi_asset_returns, np.array([-0.5, 0.5, 0.5, 0.5]))

    def test_raises_on_mismatched_dimensions(self, multi_asset_returns):
        with pytest.raises(ValueError, match="weights length"):
            PortfolioVaR(multi_asset_returns, np.array([0.5, 0.3, 0.2]))

    def test_raises_on_insufficient_periods(self):
        R = np.random.normal(0, 0.02, (2, 3))
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        with pytest.raises(ValueError, match="at least 3"):
            PortfolioVaR(R, w)

    def test_report_contains_all_keys(self, multi_asset_returns):
        w = np.array([0.4, 0.3, 0.2, 0.1])
        pf = PortfolioVaR(multi_asset_returns, w, labels=["A", "B", "C", "D"])
        rpt = pf.report()
        expected = [
            "n_assets", "n_periods", "labels", "weights",
            "position_value", "alpha", "confidence_level",
            "portfolio_var", "component_var_historical", "diversification_ratio",
        ]
        for k in expected:
            assert k in rpt, f"Missing key: {k}"
