"""Tests for factor_attribution.py"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factor_attribution import FactorAttribution


def make_single_factor_data(T=200, seed=42):
    """Generate data where return = alpha + beta * factor + noise."""
    rng = np.random.RandomState(seed)
    factor = rng.randn(T) * 0.02  # daily factor returns
    alpha = 0.001
    beta = 1.2
    noise = rng.randn(T) * 0.01
    returns = alpha + beta * factor + noise
    return returns, factor.reshape(-1, 1), ["MKT"], alpha, np.array([beta])


def make_multi_factor_data(T=200, seed=42):
    """Generate data with 3 orthogonal factors."""
    rng = np.random.RandomState(seed)
    # Generate orthogonal factors via QR of random normal data,
    # then scale to unit variance (std=1) and multiply by desired std
    raw = rng.randn(T, 3)
    Q, _ = np.linalg.qr(raw)
    # Q has unit-norm columns. Normalize to std=1 by multiplying by sqrt(T)
    factors = Q * np.sqrt(T) * 0.02  # each factor has std=0.02

    alpha = 0.001
    betas = np.array([1.2, -0.5, 0.8])
    noise = rng.randn(T) * 0.005
    returns = alpha + factors @ betas + noise

    return returns, factors, ["MKT", "SMB", "HML"], alpha, betas


def test_fit_single_factor():
    """Single factor regression should recover alpha and beta."""
    returns, factors, names, alpha_true, beta_true = make_single_factor_data(T=500)
    fa = FactorAttribution(returns, factors, names)
    result = fa.fit()

    assert abs(result["alpha"] - alpha_true) < 0.005, \
        f"Alpha: estimated={result['alpha']:.6f}, true={alpha_true}"
    assert abs(result["betas"][0] - beta_true[0]) < 0.1, \
        f"Beta: estimated={result['betas'][0]:.4f}, true={beta_true[0]:.4f}"
    assert result["r_squared"] > 0.5, "R² should be high for well-specified model"


def test_fit_multi_factor():
    """Multi-factor regression should work correctly."""
    returns, factors, names, alpha_true, beta_true = make_multi_factor_data(T=500)
    fa = FactorAttribution(returns, factors, names)
    result = fa.fit()

    assert abs(result["alpha"] - alpha_true) < 0.005
    for k in range(3):
        assert abs(result["betas"][k] - beta_true[k]) < 0.1, \
            f"Beta[{k}]: est={result['betas'][k]:.4f}, true={beta_true[k]:.4f}"
    assert result["adj_r_squared"] > 0.5


def test_factor_exposures():
    """factor_exposures should return correct betas."""
    returns, factors, names, _, beta_true = make_multi_factor_data(T=500)
    fa = FactorAttribution(returns, factors, names)
    fa.fit()
    exposures = fa.factor_exposures()
    assert exposures.shape == (3,)
    assert np.allclose(exposures, beta_true, atol=0.1)


def test_factor_contributions():
    """Factor contributions should sum to ~1.0 for fitted variance."""
    returns, factors, names, _, _ = make_multi_factor_data(T=500)
    fa = FactorAttribution(returns, factors, names)
    fa.fit()
    contribs = fa.factor_contributions()

    assert "contributions" in contribs
    assert "contributions_to_total" in contribs
    assert "residual_contribution" in contribs

    # Contributions to fitted variance should sum to ~1.0
    # Note: with correlated factors this may not sum exactly to 1
    fitted_sum = sum(contribs["contributions"])
    assert 0.8 < fitted_sum < 1.2, \
        f"Factor contributions to fitted variance should be near 1.0, got {fitted_sum:.4f}"

    # Contributions to total + residual should sum to ~1.0
    total_contrib = sum(contribs["contributions_to_total"]) + contribs["residual_contribution"]
    assert abs(total_contrib - 1.0) < 0.01, \
        f"Total contributions should sum to 1.0, got {total_contrib:.4f}"

    # Residual contribution should be reasonably small for well-specified model
    # With noise std=0.005 and factor-driven returns, residual should be minority
    assert contribs["residual_contribution"] < 0.8, \
        f"Residual contribution should not dominate, got {contribs['residual_contribution']:.4f}"


def test_orthogonal_factors_independent_betas():
    """With orthogonal factors, adding a factor should not change other betas."""
    T = 1000
    rng = np.random.RandomState(42)

    # Generate two orthogonal factors with meaningful variance
    raw = rng.randn(T, 2)
    Q, _ = np.linalg.qr(raw)
    factors_raw = Q * np.sqrt(T)  # std=1 each
    f1 = factors_raw[:, 0] * 0.02
    f2 = factors_raw[:, 1] * 0.02

    beta1 = 1.5
    beta2 = 0.8
    noise = rng.randn(T) * 0.005
    returns = beta1 * f1 + beta2 * f2 + noise

    # Fit with only f1
    fa1 = FactorAttribution(returns, f1.reshape(-1, 1), ["F1"])
    fa1.fit()
    beta1_single = fa1.factor_exposures()[0]

    # Fit with both f1 and f2
    fa2 = FactorAttribution(returns, np.column_stack([f1, f2]), ["F1", "F2"])
    fa2.fit()
    beta1_multi = fa2.factor_exposures()[0]

    # With orthogonal factors, beta1 should be the same
    assert abs(beta1_single - beta1_multi) < 0.05, \
        f"Orthogonal factors: beta1_single={beta1_single:.4f}, beta1_multi={beta1_multi:.4f}"


def test_single_factor_equals_capm():
    """Single factor regression should be equivalent to simple beta/alpha estimation."""
    returns, factors, names, alpha_true, beta_true = make_single_factor_data(T=500)

    fa = FactorAttribution(returns, factors, names)
    fa.fit()

    # Manual OLS
    X = np.column_stack([np.ones(len(returns)), factors])
    beta_ols = np.linalg.lstsq(X, returns, rcond=None)[0]

    assert abs(fa._alpha - beta_ols[0]) < 1e-10
    assert abs(fa.factor_exposures()[0] - beta_ols[1]) < 1e-10


def test_alpha_significance():
    """alpha_significance should return a t-statistic."""
    returns, factors, names, alpha_true, _ = make_multi_factor_data(T=500)
    fa = FactorAttribution(returns, factors, names)
    fa.fit()
    t_stat = fa.alpha_significance()

    # alpha=0.001 has weak signal relative to noise (0.005).
    # t-stat should be a reasonable number.
    assert isinstance(t_stat, float)
    assert abs(t_stat) < 50, f"Alpha t-stat should be finite, got {t_stat}"


def test_adj_r_squared():
    """adj_r_squared should be <= r_squared."""
    returns, factors, names, _, _ = make_multi_factor_data(T=100)
    fa = FactorAttribution(returns, factors, names)
    result = fa.fit()
    adj = fa.adj_r_squared()
    assert adj <= result["r_squared"] + 1e-10, \
        f"adj R² ({adj}) should be <= R² ({result['r_squared']})"
    assert 0.0 <= adj <= 1.0, f"adj R² should be in [0,1], got {adj}"


def test_report():
    """Report should be comprehensive."""
    returns, factors, names, _, _ = make_multi_factor_data(T=200)
    fa = FactorAttribution(returns, factors, names)
    rep = fa.report()

    assert "alpha" in rep
    assert "alpha_t_stat" in rep
    assert "alpha_p_value" in rep
    assert "factors" in rep
    assert len(rep["factors"]) == 3
    assert "r_squared" in rep
    assert "adj_r_squared" in rep
    assert "f_statistic" in rep

    # Each factor detail
    for fd in rep["factors"]:
        assert "name" in fd
        assert "beta" in fd
        assert "se" in fd
        assert "t_stat" in fd
        assert "p_value" in fd
        assert "contrib_to_fitted" in fd
        assert "contrib_to_total" in fd


def test_input_validation():
    """Input validation should catch bad inputs."""
    # Mismatched lengths
    try:
        FactorAttribution(np.array([1.0, 2.0]), np.array([[1.0], [2.0], [3.0]]))
        assert False, "Should raise"
    except ValueError:
        pass

    # Wrong number of factor names
    try:
        FactorAttribution(np.array([1.0, 2.0, 3.0]),
                          np.array([[1.0], [2.0], [3.0]]),
                          ["A", "B", "C"])
        assert False, "Should raise"
    except ValueError:
        pass


def test_call_before_fit():
    """Methods should raise RuntimeError if called before fit."""
    returns, factors, names, _, _ = make_multi_factor_data(T=100)
    fa = FactorAttribution(returns, factors, names)

    try:
        fa.factor_exposures()
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass

    try:
        fa.factor_contributions()
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass


def test_f_stat_significant():
    """F-stat should be significant for well-specified model."""
    returns, factors, names, _, _ = make_multi_factor_data(T=500)
    fa = FactorAttribution(returns, factors, names)
    result = fa.fit()
    assert result["f_pvalue"] < 0.001, \
        f"F-test should be significant, p={result['f_pvalue']}"
    assert result["f_statistic"] > 10


if __name__ == "__main__":
    tests = [
        test_fit_single_factor,
        test_fit_multi_factor,
        test_factor_exposures,
        test_factor_contributions,
        test_orthogonal_factors_independent_betas,
        test_single_factor_equals_capm,
        test_alpha_significance,
        test_adj_r_squared,
        test_report,
        test_input_validation,
        test_call_before_fit,
        test_f_stat_significant,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
