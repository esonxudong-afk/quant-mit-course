"""Tests for garch_model.py"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from garch_model import GARCHModel, EWMASmoother, simulate_garch


# ---------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------

def assert_close(a, b, tol=0.20, msg=""):
    """Assert relative or absolute closeness."""
    a_f, b_f = float(a), float(b)
    denom = max(abs(b_f), 1e-10)
    rel_err = abs(a_f - b_f) / denom
    abs_err = abs(a_f - b_f)
    ok = (rel_err < tol) or (abs_err < 1e-6)
    assert ok, f"{msg}: got {a_f:.6f}, expected {b_f:.6f} (rel={rel_err:.4f}, abs={abs_err:.6f})"


# ---------------------------------------------------------------------
# Test 1: GARCH parameter recovery from simulated data
# ---------------------------------------------------------------------

def test_garch_parameter_recovery():
    """Fit GARCH on simulated data and verify α, β are recovered within tolerance."""
    returns, true_sigma = simulate_garch(
        n=2000, omega=0.05, alpha=0.10, beta=0.85, mu=0.0, seed=42
    )

    model = GARCHModel()
    model.fit(returns)

    r = model.report()
    assert r["converged"], "Model should converge on well-behaved data"

    # Parameter recovery — allow generous tolerance for finite sample
    assert_close(r["alpha"], 0.10, tol=0.50, msg="alpha recovery")
    assert_close(r["beta"], 0.85, tol=0.20, msg="beta recovery")
    assert_close(r["omega"], 0.05, tol=0.80, msg="omega recovery")


# ---------------------------------------------------------------------
# Test 2: Persistence (α+β) is positive
# ---------------------------------------------------------------------

def test_persistence_positive():
    """α + β must be > 0 for a well-specified GARCH on real-like data."""
    returns, _ = simulate_garch(
        n=1000, omega=0.05, alpha=0.15, beta=0.80, mu=0.0, seed=7
    )

    model = GARCHModel()
    model.fit(returns)

    p = model.persistence()
    assert p > 0, f"Persistence should be positive, got {p}"
    assert p < 1, f"Persistence should be < 1 for stationarity, got {p}"
    assert_close(p, 0.15 + 0.80, tol=0.25, msg="persistence recovery")


# ---------------------------------------------------------------------
# Test 3: Half-life calculation
# ---------------------------------------------------------------------

def test_half_life():
    """half_life = ln(0.5) / ln(α+β) should match known values."""
    returns, _ = simulate_garch(
        n=1000, omega=0.05, alpha=0.10, beta=0.85, mu=0.0, seed=99
    )

    model = GARCHModel()
    model.fit(returns)

    hl = model.half_life()
    p = model.persistence()

    assert p > 0, "Persistence must be positive for finite half-life"
    assert np.isfinite(hl), f"Half-life should be finite, got {hl}"
    assert hl > 0, f"Half-life should be positive, got {hl}"

    # Verify formula internally
    expected_hl = np.log(0.5) / np.log(p)
    assert_close(hl, expected_hl, tol=1e-6, msg="half_life formula consistency")

    # Persistence ~0.95 → half-life ≈ 13.5
    # With noise, we check rough magnitude
    assert 2 < hl < 200, f"Half-life {hl} should be in [2, 200] for typical daily data"


# ---------------------------------------------------------------------
# Test 4: EWMA vs RiskMetrics benchmark (λ=0.94)
# ---------------------------------------------------------------------

def test_ewma_riskmetrics_benchmark():
    """EWMA with λ=0.94 should produce volatility that is highly correlated
    with GARCH-estimated volatility on the same data."""
    returns, _ = simulate_garch(
        n=1000, omega=0.05, alpha=0.10, beta=0.85, mu=0.0, seed=123
    )

    model = GARCHModel()
    model.fit(returns)

    ewma = EWMASmoother(lam=0.94)
    sq = (returns - np.mean(returns)) ** 2
    ewma_vol = ewma.smooth(sq)
    garch_vol = model.conditional_volatility()

    # They should be positively correlated (both capture volatility clustering)
    corr = float(np.corrcoef(ewma_vol, garch_vol)[0, 1])
    assert corr > 0.5, f"EWMA and GARCH volatilities should be correlated, got r={corr:.4f}"

    # EWMA mean vol should be on the same order as GARCH
    ewma_mean = float(np.mean(ewma_vol))
    garch_mean = float(np.mean(garch_vol))
    ratio = ewma_mean / garch_mean
    assert 0.3 < ratio < 3.0, f"EWMA/GARCH mean vol ratio {ratio:.3f} out of [0.3, 3.0]"

    # compare_with_garch should return full dict
    comp = ewma.compare_with_garch(returns, model)
    assert "correlation" in comp
    assert "mae" in comp
    assert "rmse" in comp
    assert comp["correlation"] > 0.5


# ---------------------------------------------------------------------
# Test 5: Forecast converges to unconditional volatility
# ---------------------------------------------------------------------

def test_forecast_convergence():
    """Long-horizon forecast should converge to unconditional volatility."""
    returns, true_sigma = simulate_garch(
        n=1000, omega=0.05, alpha=0.10, beta=0.85, mu=0.0, seed=42
    )

    model = GARCHModel()
    model.fit(returns)

    # Short forecast
    f5 = model.forecast_volatility(steps=5)
    assert len(f5) == 5
    assert np.all(f5 > 0), "All volatility forecasts must be positive"

    # Long-horizon forecast should converge to unconditional vol
    f100 = model.forecast_volatility(steps=100)
    uncond_vol = model.report()["unconditional_vol"]

    # Last forecast should be close to unconditional vol
    assert_close(
        f100[-1], uncond_vol, tol=0.10,
        msg="long-horizon fcast vs unconditional vol"
    )


# ---------------------------------------------------------------------
# Test 6: Simulation consistency
# ---------------------------------------------------------------------

def test_simulate_garch_consistency():
    """Simulated GARCH data should have non-constant volatility."""
    returns, sigma = simulate_garch(n=500, seed=42)

    assert len(returns) == 500
    assert len(sigma) == 500
    assert np.all(sigma > 0), "All volatilities must be positive"

    # Returns should have variance clustering: squared returns autocorrelation
    # should be positive at lag 1 (a defining feature of GARCH)
    sq = (returns - np.mean(returns)) ** 2
    acf1 = float(np.corrcoef(sq[:-1], sq[1:])[0, 1])
    assert acf1 > 0.01, f"Squared returns should show positive autocorrelation, got {acf1:.4f}"


# ---------------------------------------------------------------------
# Test 7: Edge cases / input validation
# ---------------------------------------------------------------------

def test_input_validation():
    """Invalid inputs should raise clear errors."""
    model = GARCHModel()

    # Too few observations
    try:
        model.fit(np.random.randn(50))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # 2D returns
    try:
        model.fit(np.random.randn(100, 2))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Call methods before fitting
    unfitted = GARCHModel()
    try:
        unfitted.conditional_volatility()
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

    try:
        unfitted.persistence()
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

    try:
        unfitted.forecast_volatility()
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

    # EWMA: lam must be in (0, 1)
    try:
        EWMASmoother(lam=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        EWMASmoother(lam=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # simulate_garch: omega must be > 0
    try:
        simulate_garch(omega=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # simulate_garch: stationarity
    try:
        simulate_garch(alpha=0.6, beta=0.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------
# Test 8: Conditional volatility length matches input
# ---------------------------------------------------------------------

def test_conditional_volatility_shape():
    """Fitted sigma should have same length as returns."""
    returns, _ = simulate_garch(n=300, seed=123)
    model = GARCHModel()
    model.fit(returns)

    sigma = model.conditional_volatility()
    assert len(sigma) == 300
    assert sigma.ndim == 1
    assert np.all(sigma > 0)


# ---------------------------------------------------------------------
# Test 9: Log-likelihood improves with better model
# ---------------------------------------------------------------------

def test_log_likelihood_ordering():
    """A GARCH(1,1) should fit GARCH-generated data better than constant-vol model."""
    returns, _ = simulate_garch(
        n=800, omega=0.05, alpha=0.15, beta=0.80, mu=0.0, seed=7
    )

    model = GARCHModel()
    model.fit(returns)

    # Log-likelihood of the fitted GARCH
    ll_garch = model.log_likelihood_

    # Log-likelihood under constant variance (α=β=0, variance = sample var)
    eps = returns - np.mean(returns)
    const_var = float(np.var(eps, ddof=0))
    n = len(returns)
    ll_const = -0.5 * np.sum(np.log(const_var) + eps**2 / const_var)

    assert ll_garch > ll_const, (
        f"GARCH log-lik ({ll_garch:.1f}) should exceed constant-vol ({ll_const:.1f})"
    )
    # The improvement should be non-trivial for GARCH data
    assert ll_garch - ll_const > 10.0, "GARCH should substantially outperform constant vol"


# ---------------------------------------------------------------------
# Test 10: Report completeness
# ---------------------------------------------------------------------

def test_report_completeness():
    """report() should return all expected keys with valid values."""
    returns, _ = simulate_garch(n=500, seed=42)
    model = GARCHModel()
    model.fit(returns)

    r = model.report()

    expected_keys = {
        "omega", "alpha", "beta", "mu", "persistence",
        "half_life", "log_likelihood", "converged",
        "unconditional_vol", "n_observations",
    }
    missing = expected_keys - set(r.keys())
    extra = set(r.keys()) - expected_keys
    assert not missing, f"Missing keys: {missing}"
    assert not extra, f"Unexpected keys: {extra}"

    assert r["omega"] > 0, f"omega should be > 0, got {r['omega']}"
    assert r["alpha"] >= 0
    assert r["beta"] >= 0
    assert r["persistence"] < 1, "Must be stationary"
    assert r["unconditional_vol"] > 0
    assert r["n_observations"] == 500


# ---------------------------------------------------------------------
# Test 11: EWMA smooth method correctness (manual check)
# ---------------------------------------------------------------------

def test_ewma_smooth_correctness():
    """Manually verify EWMA recursion on known data."""
    sq = np.array([4.0, 1.0, 9.0, 0.25], dtype=np.float64)
    ewma = EWMASmoother(lam=0.90)

    result = ewma.smooth(sq)

    # Initial variance = mean of all squared returns = (4+1+9+0.25)/4 = 3.5625
    var0 = np.mean(sq)  # 3.5625
    expected = np.zeros(4)
    expected[0] = np.sqrt(var0)

    for t in range(1, 4):
        var = 0.9 * (expected[t - 1] ** 2) + 0.1 * sq[t - 1]
        expected[t] = np.sqrt(var)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------
# Test 12: Decreasing persistence → shorter half-life
# ---------------------------------------------------------------------

def test_persistence_half_life_relationship():
    """Higher persistence should mean longer half-life (monotonic)."""
    # Simulate two processes with different persistence
    returns_high, _ = simulate_garch(
        n=1000, omega=0.05, alpha=0.10, beta=0.88, mu=0.0, seed=1
    )
    returns_low, _ = simulate_garch(
        n=1000, omega=0.05, alpha=0.25, beta=0.50, mu=0.0, seed=1
    )

    m1 = GARCHModel()
    m1.fit(returns_high)
    m2 = GARCHModel()
    m2.fit(returns_low)

    # The high-persistence model should show larger persistence
    # and longer half-life than the low-persistence model
    assert m1.persistence() > m2.persistence(), (
        f"High-pers process ({m1.persistence():.4f}) should exceed "
        f"low-pers ({m2.persistence():.4f})"
    )
    assert m1.half_life() > m2.half_life(), (
        f"High-pers half-life ({m1.half_life():.1f}) should exceed "
        f"low-pers ({m2.half_life():.1f})"
    )


# ---------------------------------------------------------------------
# Test 13: Forecast monotonicity toward unconditional vol
# ---------------------------------------------------------------------

def test_forecast_monotonic():
    """For a stationary GARCH (α+β < 1), long-horizon forecasts should
    be smooth and approach unconditional vol monotonically from the
    current conditional vol level."""
    returns, _ = simulate_garch(
        n=1000, omega=0.05, alpha=0.10, beta=0.85, mu=0.0, seed=42
    )
    model = GARCHModel()
    model.fit(returns)

    f = model.forecast_volatility(steps=50)
    uncond = model.report()["unconditional_vol"]

    # Forecasts should all be positive
    assert np.all(f > 0)

    # The distance |f[h] - uncond| should generally decrease (not strictly
    # monotonic due to sqrt, but the squared variance should be)
    diff_last = abs(f[-1] - uncond)
    diff_first = abs(f[0] - uncond)
    assert diff_last < diff_first, (
        f"Forecast should converge toward unconditional vol: "
        f"|diff[0]|={diff_first:.6f}, |diff[-1]|={diff_last:.6f}"
    )
