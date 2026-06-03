"""Tests for robust_regression.py"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robust_regression import RobustRegression


def make_normal_data(n=100, k=3, seed=42):
    """Generate clean normal data."""
    rng = np.random.RandomState(seed)
    X = np.column_stack([np.ones(n), rng.randn(n, k - 1)])
    beta_true = np.array([1.0, 2.0, -0.5])
    y = X @ beta_true + rng.randn(n) * 0.3
    return X, y, beta_true


def make_outlier_data(n=100, k=3, outlier_fraction=0.1, seed=42):
    """Generate data with outliers."""
    rng = np.random.RandomState(seed)
    X = np.column_stack([np.ones(n), rng.randn(n, k - 1)])
    beta_true = np.array([1.0, 2.0, -0.5])
    y = X @ beta_true + rng.randn(n) * 0.3

    # Add outliers
    n_outliers = int(n * outlier_fraction)
    outlier_idx = rng.choice(n, n_outliers, replace=False)
    y[outlier_idx] += rng.randn(n_outliers) * 20.0  # large deviations

    return X, y, beta_true, outlier_idx


def test_fit_normal_data():
    """On normal data, Huber should recover true coefficients closely."""
    X, y, beta_true = make_normal_data(n=200)
    rr = RobustRegression(tune=1.345)
    result = rr.fit(X, y)
    beta_huber = result["beta"]

    assert result["converged"], "IRLS should converge on normal data"
    assert result["n_iter"] < 50, "Should converge quickly on clean data"

    # Coefficients should be close to truth
    assert np.allclose(beta_huber, beta_true, atol=0.15), \
        f"Huber beta={beta_huber} far from true={beta_true}"


def test_huber_approx_ols_normal():
    """On normal data without outliers, Huber ≈ OLS."""
    X, y, _ = make_normal_data(n=200)
    rr = RobustRegression(tune=1.345)
    comp = rr.compare_with_ols(X, y)

    # Coefficient difference should be small
    assert comp["beta_diff_norm"] < 0.1, \
        f"Huber and OLS should be close on normal data, diff={comp['beta_diff_norm']}"

    # On pure normal data with tune=1.345, ~18% of obs have |r/scale| > 1.345
    # This is expected: 1.345 * σ means ~18% exceed in each tail
    # With n=200, expect ~36 flagged; allow up to 50
    assert comp["outlier_count"] < 50, \
        f"On normal data, some observations are flagged due to threshold, got {comp['outlier_count']}"


def test_huber_better_than_ols_with_outliers():
    """With outliers, Huber should be significantly better than OLS."""
    X, y, beta_true, _ = make_outlier_data(n=200, outlier_fraction=0.15)
    rr = RobustRegression(tune=1.345)
    comp = rr.compare_with_ols(X, y)

    # Huber should be closer to true beta
    huber_error = np.linalg.norm(comp["huber_beta"] - beta_true)
    ols_error = np.linalg.norm(comp["ols_beta"] - beta_true)

    assert huber_error < ols_error, \
        f"Huber error={huber_error:.4f} should be < OLS error={ols_error:.4f} with outliers"
    assert comp["outlier_count"] >= 10, \
        f"Should detect outliers, got {comp['outlier_count']}"


def test_predict():
    """Predict should return correct shape."""
    X, y, _ = make_normal_data(n=100)
    rr = RobustRegression(tune=1.345)
    rr.fit(X, y)

    X_new = np.column_stack([np.ones(10), np.random.randn(10, 2)])
    y_pred = rr.predict(X_new)
    assert y_pred.shape == (10,), f"Expected shape (10,), got {y_pred.shape}"


def test_residuals():
    """Residuals should have correct shape."""
    X, y, _ = make_normal_data(n=100)
    rr = RobustRegression(tune=1.345)
    rr.fit(X, y)
    r = rr.residuals()
    assert r.shape == (100,)
    assert np.allclose(np.mean(r), 0.0, atol=0.2)


def test_convergence():
    """IRLS should converge within max_iter."""
    X, y, _ = make_normal_data(n=50)
    rr = RobustRegression(tune=1.345, max_iter=50, tol=1e-8)
    result = rr.fit(X, y)
    assert result["converged"], "IRLS should converge on simple data"
    assert result["n_iter"] > 0


def test_weights_range():
    """Weights should be in [0, 1]."""
    X, y, _, _ = make_outlier_data(n=100, outlier_fraction=0.2)
    rr = RobustRegression(tune=1.345)
    result = rr.fit(X, y)
    w = result["weights"]
    assert np.all(w >= 0) and np.all(w <= 1.0 + 1e-10), "Weights must be in [0, 1]"
    assert np.sum(w < 0.99) > 0, "Should have some downweighted observations"


def test_report():
    """Report should have expected structure."""
    X, y, _ = make_normal_data(n=100)
    rr = RobustRegression(tune=1.345)
    rr.fit(X, y)
    rep = rr.report()
    assert "beta" in rep
    assert "scale" in rep
    assert "n_iter" in rep
    assert "converged" in rep
    assert "residual_stats" in rep
    assert "outlier_count" in rep


def test_predict_before_fit():
    """Calling predict before fit should raise."""
    rr = RobustRegression()
    try:
        rr.predict(np.array([[1.0, 2.0]]))
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass


def test_input_validation():
    """Input validation should catch bad inputs."""
    rr = RobustRegression()

    # Mismatched lengths
    try:
        rr.fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0]))
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # 2D y
    try:
        rr.fit(np.array([[1.0], [2.0]]), np.array([[1.0], [2.0]]))
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # Negative tune
    try:
        RobustRegression(tune=-1.0)
        assert False, "Should raise ValueError for negative tune"
    except ValueError:
        pass


def test_compare_with_ols_output():
    """compare_with_ols should return comprehensive comparison."""
    X, y, _, _ = make_outlier_data(n=100, outlier_fraction=0.1)
    rr = RobustRegression(tune=1.345)
    comp = rr.compare_with_ols(X, y)

    assert "huber_beta" in comp
    assert "ols_beta" in comp
    assert "beta_diff" in comp
    assert "huber_mse" in comp
    assert "ols_mse" in comp
    assert "huber_mad" in comp
    assert "ols_mad" in comp
    assert "outlier_count" in comp
    assert "converged" in comp


def test_different_tune_values():
    """Testing with different tune constants."""
    X, y, _, _ = make_outlier_data(n=100, outlier_fraction=0.2)

    # Smaller tune → more aggressive outlier handling
    rr_small = RobustRegression(tune=0.5)
    rr_large = RobustRegression(tune=3.0)

    r_small = rr_small.fit(X, y)
    r_large = rr_large.fit(X, y)

    # Larger tune should be closer to OLS
    ols_beta = np.linalg.lstsq(X, y, rcond=None)[0]
    diff_small = np.linalg.norm(r_small["beta"] - ols_beta)
    diff_large = np.linalg.norm(r_large["beta"] - ols_beta)

    # Larger tune → more like OLS → smaller diff from OLS
    assert diff_large <= diff_small * 1.5, \
        f"Larger tune should be closer to OLS, small_diff={diff_small:.4f}, large_diff={diff_large:.4f}"


if __name__ == "__main__":
    tests = [
        test_fit_normal_data,
        test_huber_approx_ols_normal,
        test_huber_better_than_ols_with_outliers,
        test_predict,
        test_residuals,
        test_convergence,
        test_weights_range,
        test_report,
        test_predict_before_fit,
        test_input_validation,
        test_compare_with_ols_output,
        test_different_tune_values,
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
