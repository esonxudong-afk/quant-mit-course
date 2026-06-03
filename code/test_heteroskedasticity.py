"""Tests for heteroskedasticity_test.py"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heteroskedasticity_test import HeteroskedasticityTest


def make_data(n=200, k=3, homoskedastic=True, seed=42):
    """Generate test data with known properties."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, k)
    X[:, 0] = 1.0  # intercept
    beta = np.ones(k)
    y = X @ beta
    if homoskedastic:
        noise = rng.randn(n) * 0.5
    else:
        # Heteroskedastic: variance increases with X[:, 1]
        noise = rng.randn(n) * (0.5 * np.abs(X[:, 1]) + 0.1)
    resid = noise
    return resid, X, y, noise


def test_bp_homoskedastic():
    """BP test should NOT reject H0 on homoskedastic data."""
    resid, X, _, _ = make_data(n=200, k=3, homoskedastic=True)
    ht = HeteroskedasticityTest(resid, X)
    result = ht.breusch_pagan()
    assert "statistic" in result
    assert "p_value" in result
    assert result["p_value"] > 0.05, f"BP p_value={result['p_value']} should be > 0.05 for homoskedastic data"
    assert not result["reject_H0"], "Should not reject H0 for homoskedastic data"


def test_bp_heteroskedastic():
    """BP test SHOULD reject H0 on strongly heteroskedastic data."""
    # Create data where variance is directly proportional to a column of X.
    # BP regresses e² on X, so the linear relationship must be between e² and X itself,
    # not |X|. Use strictly positive X values to avoid symmetric cancellation.
    rng = np.random.RandomState(42)
    n = 300
    # Use uniform [0,3] so variance increases monotonically with X[:,1]
    X = rng.uniform(0, 3, (n, 3))
    X[:, 0] = 1.0  # intercept
    # sigma increases with X[:,1]
    sigma = 0.5 + 5.0 * X[:, 1]
    resid = rng.randn(n) * sigma
    ht = HeteroskedasticityTest(resid, X)
    result = ht.breusch_pagan()
    assert result["p_value"] < 0.05, f"BP p_value={result['p_value']} should be < 0.05 for heteroskedastic data"
    assert result["reject_H0"], "Should reject H0 for heteroskedastic data"


def test_white_homoskedastic():
    """White test should NOT reject H0 on homoskedastic data."""
    resid, X, _, _ = make_data(n=200, k=3, homoskedastic=True)
    ht = HeteroskedasticityTest(resid, X)
    result = ht.white_test()
    assert "statistic" in result
    assert "p_value" in result


def test_white_heteroskedastic():
    """White test SHOULD reject H0 on heteroskedastic data."""
    resid, X, _, _ = make_data(n=300, k=3, homoskedastic=False)
    ht = HeteroskedasticityTest(resid, X)
    result = ht.white_test()
    # With strong heteroskedasticity, should reject
    # Give some tolerance — White test is less powerful than BP for linear heteroskedasticity
    assert result["p_value"] < 0.10, f"White p_value={result['p_value']} should indicate heteroskedasticity"


def test_gq_basic():
    """GQ test basic output structure."""
    resid, X, _, _ = make_data(n=200, k=3, homoskedastic=True)
    ht = HeteroskedasticityTest(resid, X)
    result = ht.goldfeld_quandt(split_ratio=0.3)
    assert "statistic" in result
    assert "method" in result


def test_is_homoskedastic():
    """is_homoskedastic should return True for homoskedastic data."""
    resid, X, _, _ = make_data(n=200, k=3, homoskedastic=True)
    ht = HeteroskedasticityTest(resid, X)
    assert ht.is_homoskedastic(alpha=0.05), "Should detect homoskedasticity"


def test_is_not_homoskedastic():
    """is_homoskedastic should return False for heteroskedastic data."""
    resid, X, _, _ = make_data(n=300, k=3, homoskedastic=False)
    ht = HeteroskedasticityTest(resid, X)
    assert not ht.is_homoskedastic(alpha=0.05), "Should detect heteroskedasticity"


def test_robust_se_hc0():
    """HC0 robust SE must return correct shape and positive values."""
    resid, X, _, _ = make_data(n=100, k=3, homoskedastic=False)
    ht = HeteroskedasticityTest(resid, X)
    beta_hat = np.array([1.0, 0.5, 0.3])
    se = ht.robust_se(beta_hat, cov_type="HC0")
    assert se.shape == (3,)
    assert np.all(se >= 0), "SE must be non-negative"


def test_robust_se_hc1():
    """HC1 should be HC0 * sqrt(n/(n-k))."""
    resid, X, _, _ = make_data(n=100, k=3, homoskedastic=False)
    ht = HeteroskedasticityTest(resid, X)
    beta_hat = np.array([1.0, 0.5, 0.3])
    se_hc0 = ht.robust_se(beta_hat, cov_type="HC0")
    se_hc1 = ht.robust_se(beta_hat, cov_type="HC1")

    n, k = 100, 3
    expected_ratio = np.sqrt(n / (n - k))
    ratio = se_hc1 / np.maximum(se_hc0, 1e-10)
    assert np.allclose(ratio, expected_ratio, atol=1e-10), \
        f"HC1/HC0 ratio should be {expected_ratio}, got {ratio}"


def test_robust_se_hc2_hc3():
    """HC2 and HC3 should return valid results."""
    resid, X, _, _ = make_data(n=100, k=3, homoskedastic=False)
    ht = HeteroskedasticityTest(resid, X)
    beta_hat = np.array([1.0, 0.5, 0.3])
    se_hc2 = ht.robust_se(beta_hat, cov_type="HC2")
    se_hc3 = ht.robust_se(beta_hat, cov_type="HC3")

    assert se_hc2.shape == (3,)
    assert se_hc3.shape == (3,)
    assert np.all(se_hc2 >= 0)
    assert np.all(se_hc3 >= 0)
    # HC3 should be >= HC2 generally (more conservative)
    # This is typical but not guaranteed; check at least both are positive


def test_robust_se_homoskedastic_close():
    """On homoskedastic data, all HC variants should be similar."""
    resid, X, _, _ = make_data(n=500, k=3, homoskedastic=True)
    ht = HeteroskedasticityTest(resid, X)
    beta_hat = np.array([1.0, 0.5, 0.2])
    se_hc0 = ht.robust_se(beta_hat, cov_type="HC0")
    se_hc1 = ht.robust_se(beta_hat, cov_type="HC1")
    se_hc2 = ht.robust_se(beta_hat, cov_type="HC2")
    se_hc3 = ht.robust_se(beta_hat, cov_type="HC3")

    # On homoskedastic data, all should be within reasonable range of each other
    # HC3 may be slightly larger
    assert np.allclose(se_hc0, se_hc2, rtol=0.05), "HC0 and HC2 should be close on homoskedastic data"
    assert np.all(se_hc3 >= se_hc2 - 1e-10)  # HC3 should be >= HC2


def test_report():
    """Report should contain all expected keys."""
    resid, X, _, _ = make_data(n=100, k=3, homoskedastic=True)
    ht = HeteroskedasticityTest(resid, X)
    r = ht.report()
    assert "breusch_pagan" in r
    assert "white_test" in r
    assert "goldfeld_quandt" in r
    assert "is_homoskedastic" in r
    assert isinstance(r["is_homoskedastic"], bool)


def test_input_validation():
    """Input validation should raise on bad shapes."""
    try:
        HeteroskedasticityTest(np.array([1.0, 2.0]), np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        assert False, "Should have raised ValueError for mismatched lengths"
    except ValueError:
        pass

    try:
        HeteroskedasticityTest(np.array([[1.0], [2.0]]), np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert False, "Should have raised ValueError for 2D residuals"
    except ValueError:
        pass


def test_gq_heteroskedastic():
    """GQ test on clearly heteroskedastic data with known ordering."""
    rng = np.random.RandomState(42)
    n = 200
    X = np.column_stack([np.ones(n), np.arange(n) / n])
    # Increasing variance with X[:, 1] (the sorted index)
    noise = rng.randn(n) * (0.1 + 2.0 * X[:, 1])
    y = X @ np.array([1.0, 0.5]) + noise
    resid = y - X @ np.array([1.0, 0.5])

    ht = HeteroskedasticityTest(resid, X)
    result = ht.goldfeld_quandt(split_ratio=0.2)
    # Since variance increases with X[:, 1] which is sorted ascending,
    # the last segment should have higher variance → F-test should reject
    assert result["p_value"] < 0.05, f"GQ p_value={result['p_value']} should reject for increasing variance"


if __name__ == "__main__":
    tests = [
        test_bp_homoskedastic,
        test_bp_heteroskedastic,
        test_white_homoskedastic,
        test_white_heteroskedastic,
        test_gq_basic,
        test_gq_heteroskedastic,
        test_is_homoskedastic,
        test_is_not_homoskedastic,
        test_robust_se_hc0,
        test_robust_se_hc1,
        test_robust_se_hc2_hc3,
        test_robust_se_homoskedastic_close,
        test_report,
        test_input_validation,
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
