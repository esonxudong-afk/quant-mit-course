"""
鲁棒回归模块 — Robust Regression via Huber's M-estimator

替代OLS，在存在异常值或肥尾分布时提供更可靠的回归估计。
使用 IRLS (Iteratively Reweighted Least Squares) 算法。

公式:
    ρ(u) = u²/2           if |u| ≤ k
           k|u| - k²/2    if |u| > k

    ψ(u) = ρ'(u) = u            if |u| ≤ k
                    k·sign(u)   if |u| > k

    w(u) = ψ(u)/u = 1          if |u| ≤ k
                     k/|u|     if |u| > k

    k = tune · MAD  (默认 tune=1.345, 95% efficiency at normal)

Usage:
    rr = RobustRegression(tune=1.345)
    result = rr.fit(X, y)
    y_pred = rr.predict(X_new)
    print(rr.report())
"""

import numpy as np
from scipy.stats import norm


class RobustRegression:
    """Huber 稳健回归

    Parameters
    ----------
    tune : float
        Huber tuning constant. Default 1.345 gives 95% asymptotic efficiency
        under normality while retaining robustness.

    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance for coefficient change.
    """

    def __init__(self, tune: float = 1.345, max_iter: int = 100, tol: float = 1e-6):
        if tune <= 0:
            raise ValueError("tune must be positive")
        self.tune = tune
        self.max_iter = max_iter
        self.tol = tol

        # Fitted results
        self._beta = None
        self._residuals_arr = None
        self._scale = None  # estimated scale (MAD-based)
        self._n_iter = None
        self._converged = False
        self._X = None
        self._y = None
        self._weights = None  # final IRLS weights
        self._n = None
        self._k = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Fit Huber robust regression via IRLS.

        Parameters
        ----------
        X : np.ndarray, shape (n, k)
            Design matrix (can include intercept column)
        y : np.ndarray, shape (n,)
            Response variable

        Returns
        -------
        dict with keys:
            beta, converged, n_iter, scale, weights, residuals
        """
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have same number of observations")

        n, k = X.shape
        self._n = n
        self._k = k
        self._X = X
        self._y = y

        # Step 1: OLS as initial estimate
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X) @ y

        # Pre-compute X'X inverse for IRLS
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)

        # IRLS loop
        converged = False
        for iteration in range(self.max_iter):
            residuals = y - X @ beta

            # Estimate scale using MAD of residuals
            # MAD = median(|r_i - median(r)|), scale = MAD / 0.6745 (for normal consistency)
            med_resid = np.median(residuals)
            mad = np.median(np.abs(residuals - med_resid))
            if mad < 1e-15:
                mad = 1e-15
            scale = mad / 0.6745
            k = self.tune * scale

            # Compute weights: w_i = ψ(r_i/scale) / (r_i/scale)
            standardized = residuals / scale
            w = np.ones(n)
            abs_std = np.abs(standardized)
            large_mask = abs_std > self.tune
            # Small residuals: w=1
            # Large residuals: w = tune / |standardized|
            w[large_mask] = self.tune / abs_std[large_mask]

            # IRLS update: β = (X'WX)⁻¹ X'Wy
            W = np.diag(w)
            try:
                XtWX = X.T @ W @ X
                XtWX_inv = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError:
                XtWX_inv = np.linalg.pinv(X.T @ W @ X)

            beta_new = XtWX_inv @ X.T @ W @ y

            # Check convergence
            change = np.max(np.abs(beta_new - beta))
            beta = beta_new

            if change < self.tol:
                converged = True
                break

        # Final residuals and scale
        residuals = y - X @ beta
        med_resid = np.median(residuals)
        mad = np.median(np.abs(residuals - med_resid))
        if mad < 1e-15:
            mad = 1e-15
        scale = mad / 0.6745

        self._beta = beta
        self._residuals_arr = residuals
        self._scale = scale
        self._n_iter = iteration + 1
        self._converged = converged

        # Final weights
        standardized = residuals / scale
        w = np.ones(n)
        abs_std = np.abs(standardized)
        large_mask = abs_std > self.tune
        w[large_mask] = self.tune / abs_std[large_mask]
        self._weights = w

        return {
            "beta": beta.copy(),
            "converged": converged,
            "n_iter": self._n_iter,
            "scale": scale,
            "weights": w.copy(),
            "residuals": residuals.copy(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted model.

        Parameters
        ----------
        X : np.ndarray, shape (m, k)
            Design matrix for prediction

        Returns
        -------
        np.ndarray of predicted values
        """
        if self._beta is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.atleast_2d(X)
        if X.shape[1] != self._k:
            raise ValueError(f"X has {X.shape[1]} columns, expected {self._k}")
        return X @ self._beta

    def residuals(self) -> np.ndarray:
        """Return residuals from fitted model."""
        if self._residuals_arr is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._residuals_arr.copy()

    def compare_with_ols(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compare Huber robust regression with OLS on the same data.

        Fits both models and reports coefficient differences.

        Returns
        -------
        dict with keys:
            huber_beta, ols_beta, beta_diff, huber_scale,
            ols_mse, huber_mse, huber_mad, ols_mad,
            outlier_weight_count
        """
        # Fit Huber
        self.fit(X, y)

        # Fit OLS
        try:
            ols_beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            ols_beta = np.linalg.pinv(X) @ y

        ols_resid = y - X @ ols_beta
        huber_resid = self._residuals_arr

        # MSE
        ols_mse = np.mean(ols_resid ** 2)
        huber_mse = np.mean(huber_resid ** 2)

        # MAD
        ols_mad = np.median(np.abs(ols_resid - np.median(ols_resid)))
        huber_mad_val = np.median(np.abs(huber_resid - np.median(huber_resid)))

        # Count downweighted observations (weight < 1)
        if self._weights is not None:
            outlier_count = int(np.sum(self._weights < 0.99))
        else:
            outlier_count = 0

        return {
            "huber_beta": self._beta.tolist(),
            "ols_beta": ols_beta.tolist(),
            "beta_diff": (self._beta - ols_beta).tolist(),
            "beta_diff_norm": float(np.linalg.norm(self._beta - ols_beta)),
            "huber_scale": float(self._scale),
            "ols_mse": float(ols_mse),
            "huber_mse": float(huber_mse),
            "ols_mad": float(ols_mad),
            "huber_mad": float(huber_mad_val),
            "outlier_count": outlier_count,
            "n_observations": self._n,
            "converged": self._converged,
            "n_iter": self._n_iter,
        }

    def report(self) -> dict:
        """Generate a summary report."""
        if self._beta is None:
            return {"error": "Model not fitted. Call fit() first."}

        return {
            "beta": self._beta.tolist(),
            "converged": self._converged,
            "n_iter": self._n_iter,
            "scale": float(self._scale),
            "n_observations": self._n,
            "n_parameters": self._k,
            "tune": self.tune,
            "residual_stats": {
                "mean": float(np.mean(self._residuals_arr)),
                "std": float(np.std(self._residuals_arr)),
                "mad": float(np.median(np.abs(self._residuals_arr - np.median(self._residuals_arr)))),
                "min": float(np.min(self._residuals_arr)),
                "max": float(np.max(self._residuals_arr)),
            },
            "outlier_count": int(np.sum(self._weights < 0.99)) if self._weights is not None else 0,
        }
