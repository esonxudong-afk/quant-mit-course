"""
Tests for dynamic_spread.py — DynamicGridSpread class.

Coverage:
  - Initialisation & validation
  - Volatility calculation (normal, edge, error)
  - Optimal spread (normal, zero-vol floor, high-inv cap, error)
  - adjust_grid_params convenience method
  - regime_check (low, normal, high, crisis, negative error)
  - CLI argument parsing
"""

import math
import sys
import pytest
import subprocess

# Ensure the module is importable
sys.path.insert(0, ".")

from dynamic_spread import DynamicGridSpread, main


# ==================== Initialisation ====================

class TestInit:
    def test_defaults(self):
        ds = DynamicGridSpread()
        assert ds.k_risk == 3.0
        assert ds.max_inventory == 10
        assert ds.vol_lookback == 20

    def test_custom(self):
        ds = DynamicGridSpread(k_risk=5.0, max_inventory=20, vol_lookback=60)
        assert ds.k_risk == 5.0
        assert ds.max_inventory == 20
        assert ds.vol_lookback == 60

    def test_k_risk_zero_raises(self):
        with pytest.raises(ValueError, match="k_risk must be positive"):
            DynamicGridSpread(k_risk=0)

    def test_k_risk_negative_raises(self):
        with pytest.raises(ValueError, match="k_risk must be positive"):
            DynamicGridSpread(k_risk=-1.5)

    def test_max_inventory_zero_raises(self):
        with pytest.raises(ValueError, match="max_inventory must be positive"):
            DynamicGridSpread(max_inventory=0)

    def test_max_inventory_negative_raises(self):
        with pytest.raises(ValueError, match="max_inventory must be positive"):
            DynamicGridSpread(max_inventory=-5)

    def test_vol_lookback_too_small_raises(self):
        with pytest.raises(ValueError, match="vol_lookback must be >= 2"):
            DynamicGridSpread(vol_lookback=1)


# ==================== Volatility ====================

class TestVolatility:
    """compute_annual_volatility tests"""

    FLAT_30 = [100.0] * 30          # zero vol
    TREND_30 = list(range(100, 130))  # steady uptrend, low vol
    # ~30% annualised vol
    VOLATILE_30 = [
        100.0, 102.5, 99.0, 103.1, 97.5, 104.2, 98.0, 106.0, 95.5, 107.3,
        94.0, 108.5, 93.2, 109.0, 92.8, 110.1, 91.5, 111.3, 90.0, 112.5,
        89.5, 113.0, 88.0, 114.2, 87.5, 115.0, 86.0, 116.3, 85.5, 117.0,
    ]

    def test_zero_vol_flat_prices(self):
        ds = DynamicGridSpread()
        vol = ds.compute_annual_volatility(self.FLAT_30)
        assert vol == 0.0

    def test_trend_low_vol(self):
        ds = DynamicGridSpread()
        vol = ds.compute_annual_volatility(self.TREND_30)
        # Trend-only should produce very low annualised vol
        assert 0.0 < vol < 0.30

    def test_volatile_positive(self):
        ds = DynamicGridSpread()
        vol = ds.compute_annual_volatility(self.VOLATILE_30)
        assert vol > 0.0

    def test_exactly_lookback_prices(self):
        ds = DynamicGridSpread(vol_lookback=20)
        vol = ds.compute_annual_volatility(self.VOLATILE_30)
        assert vol > 0.0

    def test_insufficient_prices_raises(self):
        ds = DynamicGridSpread(vol_lookback=20)
        with pytest.raises(ValueError, match="Need at least 20"):
            ds.compute_annual_volatility([100.0, 101.0])

    def test_negative_price_raises(self):
        ds = DynamicGridSpread(vol_lookback=20)
        prices = [100.0] * 19 + [-1.0]
        with pytest.raises(ValueError, match="Price at index"):
            ds.compute_annual_volatility(prices)

    def test_zero_price_raises(self):
        ds = DynamicGridSpread(vol_lookback=20)
        prices = [100.0] * 19 + [0.0]
        with pytest.raises(ValueError, match="Price at index"):
            ds.compute_annual_volatility(prices)

    def test_numpy_style(self):
        """If numpy is available, cross-check against numpy.std"""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        ds = DynamicGridSpread(vol_lookback=20)
        prices = self.VOLATILE_30[-20:]
        window = prices
        log_rets = np.diff(np.log(window))
        np_vol = float(np.std(log_rets, ddof=1) * np.sqrt(252))
        our_vol = ds.compute_annual_volatility(prices)
        assert math.isclose(our_vol, np_vol, rel_tol=1e-9)


# ==================== Optimal Spread ====================

class TestOptimalSpread:
    def test_normal(self):
        """30% vol, inv=3/10 => expected spread ~7.4%"""
        ds = DynamicGridSpread(k_risk=3.0, max_inventory=10)
        bid, ask, spread = ds.optimal_spread(
            price=100.0, vol_annual=0.30, inventory=3, max_inventory=10
        )
        expected_delta = (0.30 / math.sqrt(252)) * 3.0 * (1.0 + 3 / 10)
        assert math.isclose(spread, expected_delta, rel_tol=1e-9)
        assert math.isclose(bid, 100.0 * (1.0 - spread / 2.0), rel_tol=1e-9)
        assert math.isclose(ask, 100.0 * (1.0 + spread / 2.0), rel_tol=1e-9)
        assert bid < 100.0 < ask
        assert 0.06 < spread < 0.09  # rough ballpark for 30% vol

    def test_zero_inventory(self):
        """With zero inventory, penalty term (1+|q|/Q_max) = 1.0"""
        ds = DynamicGridSpread(k_risk=3.0, max_inventory=10)
        bid, ask, spread = ds.optimal_spread(
            price=100.0, vol_annual=0.30, inventory=0, max_inventory=10
        )
        base_delta = (0.30 / math.sqrt(252)) * 3.0 * 1.0
        assert math.isclose(spread, base_delta, rel_tol=1e-9)
        assert bid < ask

    def test_max_inventory(self):
        """At max inventory (10/10), inv_ratio capped at 1 → penalty × 2"""
        ds = DynamicGridSpread(k_risk=3.0, max_inventory=10)
        bid, ask, spread = ds.optimal_spread(
            price=100.0, vol_annual=0.30, inventory=10, max_inventory=10
        )
        base_delta = (0.30 / math.sqrt(252)) * 3.0 * 2.0
        assert math.isclose(spread, base_delta, rel_tol=1e-9)

    def test_exceeds_max_inventory(self):
        """inventory=15, max=10 → ratio capped to 1, same as 10/10"""
        ds = DynamicGridSpread(k_risk=3.0, max_inventory=10)
        bid1, ask1, s1 = ds.optimal_spread(100, 0.30, 10, 10)
        bid2, ask2, s2 = ds.optimal_spread(100, 0.30, 15, 10)
        assert math.isclose(s1, s2, rel_tol=1e-9)

    def test_negative_inventory(self):
        """Short position: inv=-5 should be same spread as abs(5)"""
        ds = DynamicGridSpread(k_risk=3.0, max_inventory=10)
        _, _, s_long = ds.optimal_spread(100, 0.30, 5, 10)
        _, _, s_short = ds.optimal_spread(100, 0.30, -5, 10)
        assert math.isclose(s_long, s_short, rel_tol=1e-9)

    def test_zero_vol_floor(self):
        """Zero volatility => delta floored at 1%"""
        ds = DynamicGridSpread()
        bid, ask, spread = ds.optimal_spread(100, 0.0, 0, 10)
        assert spread == 0.01
        assert math.isclose(bid, 99.5, rel_tol=1e-9)
        assert math.isclose(ask, 100.5, rel_tol=1e-9)

    def test_price_zero_raises(self):
        ds = DynamicGridSpread()
        with pytest.raises(ValueError, match="price must be positive"):
            ds.optimal_spread(0, 0.30, 0, 10)

    def test_price_negative_raises(self):
        ds = DynamicGridSpread()
        with pytest.raises(ValueError, match="price must be positive"):
            ds.optimal_spread(-50, 0.30, 0, 10)

    def test_vol_negative_raises(self):
        ds = DynamicGridSpread()
        with pytest.raises(ValueError, match="vol_annual must be non-negative"):
            ds.optimal_spread(100, -0.1, 0, 10)

    def test_max_inventory_zero_raises(self):
        ds = DynamicGridSpread()
        with pytest.raises(ValueError, match="max_inventory must be positive"):
            ds.optimal_spread(100, 0.30, 0, 0)

    def test_bid_not_negative_even_large_spread(self):
        """Very high vol + high inv => huge spread; bid floor is 0"""
        ds = DynamicGridSpread(k_risk=10, max_inventory=10)
        bid, ask, spread = ds.optimal_spread(5.0, 5.0, 10, 10)
        assert bid >= 0.0
        assert ask > 0.0


# ==================== adjust_grid_params ====================

class TestAdjustGridParams:
    def test_basic(self):
        ds = DynamicGridSpread(k_risk=3.0, max_inventory=10)
        prices = [100.0 + i * 0.5 for i in range(30)]  # steady uptrend
        result = ds.adjust_grid_params(price=115.0, close_prices=prices, inventory=3)
        assert "bid" in result
        assert "ask" in result
        assert "spread_pct" in result
        assert "vol_annual" in result
        assert "inventory_ratio" in result
        assert result["bid"] < result["ask"]
        assert result["inventory_ratio"] == 0.3  # 3 / 10
        assert result["vol_annual"] >= 0.0

    def test_inventory_ratio_capped(self):
        ds = DynamicGridSpread(max_inventory=5)
        prices = [100.0] * 25
        result = ds.adjust_grid_params(price=100.0, close_prices=prices, inventory=10)
        assert result["inventory_ratio"] == 1.0

    def test_insufficient_prices(self):
        ds = DynamicGridSpread(vol_lookback=20)
        with pytest.raises(ValueError, match="Need at least 20"):
            ds.adjust_grid_params(price=100.0, close_prices=[100.0], inventory=0)


# ==================== regime_check ====================

class TestRegimeCheck:
    def test_low_vol(self):
        ds = DynamicGridSpread()
        assert ds.regime_check(0.0) == "low_vol"
        assert ds.regime_check(0.05) == "low_vol"
        assert ds.regime_check(0.14999) == "low_vol"

    def test_normal(self):
        ds = DynamicGridSpread()
        assert ds.regime_check(0.15) == "normal"
        assert ds.regime_check(0.20) == "normal"
        assert ds.regime_check(0.30) == "normal"
        assert ds.regime_check(0.34999) == "normal"

    def test_high_vol(self):
        ds = DynamicGridSpread()
        assert ds.regime_check(0.35) == "high_vol"
        assert ds.regime_check(0.45) == "high_vol"
        assert ds.regime_check(0.54999) == "high_vol"

    def test_crisis(self):
        ds = DynamicGridSpread()
        assert ds.regime_check(0.55) == "crisis"
        assert ds.regime_check(1.0) == "crisis"
        assert ds.regime_check(3.0) == "crisis"

    def test_negative_raises(self):
        ds = DynamicGridSpread()
        with pytest.raises(ValueError, match="vol_annual cannot be negative"):
            ds.regime_check(-0.01)


# ==================== CLI ====================

class TestCLI:
    def test_vol_flag(self):
        """python dynamic_spread.py --price 100 --vol 0.30 --inventory 3 --max-inv 10"""
        proc = subprocess.run(
            [sys.executable, "dynamic_spread.py",
             "--price", "100", "--vol", "0.30",
             "--inventory", "3", "--max-inv", "10"],
            capture_output=True, text=True, cwd=".",
        )
        assert proc.returncode == 0
        out = proc.stdout
        assert "Bid" in out
        assert "Ask" in out
        assert "Spread" in out
        assert "Regime" in out
        # Regime for 0.30 is "normal" (0.15 ≤ vol < 0.35)
        assert "normal" in out

    def test_prices_flag(self):
        prices_str = [str(100 + i) for i in range(25)]
        proc = subprocess.run(
            [sys.executable, "dynamic_spread.py",
             "--price", "100", "--prices"] + prices_str +
            ["--inventory", "0", "--max-inv", "10"],
            capture_output=True, text=True, cwd=".",
        )
        assert proc.returncode == 0, proc.stderr
        assert "Vol (annual)" in proc.stdout

    def test_regime_only(self):
        proc = subprocess.run(
            [sys.executable, "dynamic_spread.py",
             "--price", "100", "--vol", "0.55", "--regime-only"],
            capture_output=True, text=True, cwd=".",
        )
        assert proc.returncode == 0
        assert proc.stdout.strip() == "crisis"

    def test_missing_vol_and_prices(self):
        proc = subprocess.run(
            [sys.executable, "dynamic_spread.py",
             "--price", "100"],
            capture_output=True, text=True, cwd=".",
        )
        assert proc.returncode != 0


# ==================== Manual: Replicate usage example ====================

class TestUsageExample:
    """Replicate the example from the task description."""

    def test_example(self):
        ds = DynamicGridSpread(k_risk=3.0, max_inventory=10)

        # Generate ~30% annualised volatility over 30 data points
        import random
        random.seed(42)
        prices = [100.0]
        sigma_daily = 0.30 / math.sqrt(252)
        for _ in range(29):
            ret = random.gauss(0, sigma_daily)
            prices.append(prices[-1] * math.exp(ret))

        vol = ds.compute_annual_volatility(prices)
        assert vol > 0.0  # should be roughly ~0.30

        bid, ask, spread = ds.optimal_spread(
            price=100.0, vol_annual=0.30,
            inventory=3, max_inventory=10
        )
        # Rough check from task description
        assert math.isclose(bid, 96.44, rel_tol=0.02), f"bid={bid}"
        assert math.isclose(ask, 103.89, rel_tol=0.02), f"ask={ask}"
        assert math.isclose(spread, 0.0744, rel_tol=0.02), f"spread={spread}"

        regime = ds.regime_check(0.30)
        # 0.30 falls in normal (0.15 ≤ vol < 0.35)
        assert regime == "normal"
