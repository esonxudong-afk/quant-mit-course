"""
多因子归因模块 — Factor Attribution

对币种/资产收益率做多因子暴露分析。
OLS多因子回归分解收益率到因子暴露、因子贡献和alpha。

Usage:
    fa = FactorAttribution(returns, factors, factor_names)
    result = fa.fit()
    exposures = fa.factor_exposures()
    contribs = fa.factor_contributions()
    print(fa.report())
"""

import numpy as np
from scipy.stats import t as t_dist


class FactorAttribution:
    """多因子归因分析

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        资产超额收益率序列（T个时间点）
    factors : np.ndarray, shape (T, K)
        K个因子的收益率序列
    factor_names : list of str
        因子名称列表
    """

    def __init__(self, returns: np.ndarray, factors: np.ndarray,
                 factor_names: list = None):
        returns = np.asarray(returns, dtype=np.float64)
        factors = np.asarray(factors, dtype=np.float64)

        if returns.ndim != 1:
            raise ValueError("returns must be 1D array")
        if factors.ndim == 1:
            factors = factors.reshape(-1, 1)
        if factors.ndim != 2:
            raise ValueError("factors must be 1D or 2D array")
        if len(returns) != factors.shape[0]:
            raise ValueError("returns and factors must have same number of time points")

        self._returns = returns
        self._factors = factors
        self._T = len(returns)
        self._K = factors.shape[1]

        if factor_names is None:
            self._factor_names = [f"Factor_{i+1}" for i in range(self._K)]
        else:
            if len(factor_names) != self._K:
                raise ValueError(f"factor_names length ({len(factor_names)}) "
                                 f"!= number of factors ({self._K})")
            self._factor_names = list(factor_names)

        # Results
        self._alpha = None
        self._betas = None
        self._residuals = None
        self._fitted_values = None
        self._r_squared = None
        self._adj_r_squared = None
        self._alpha_se = None
        self._betas_se = None
        self._f_stat = None
        self._f_pvalue = None
        self._fitted = False

    def fit(self) -> dict:
        """Fit multi-factor regression: r_t = α + Σ β_k·f_{k,t} + ε_t

        Uses OLS with intercept.

        Returns
        -------
        dict with summary statistics
        """
        T, K = self._T, self._K

        # Design matrix: [1, f_1, f_2, ..., f_K]
        X = np.column_stack([np.ones(T), self._factors])
        y = self._returns

        # OLS
        try:
            beta_all, residuals_sum, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            beta_all = np.linalg.pinv(X) @ y

        self._alpha = beta_all[0]
        self._betas = beta_all[1:]
        fitted = X @ beta_all
        self._fitted_values = fitted
        residuals = y - fitted
        self._residuals = residuals

        # Goodness of fit
        SS_res = np.sum(residuals ** 2)
        SS_tot = np.sum((y - np.mean(y)) ** 2)
        df_model = K
        df_resid = T - K - 1  # n - k - 1 (including intercept)

        if SS_tot > 1e-15 and df_resid > 0:
            self._r_squared = 1.0 - SS_res / SS_tot
        else:
            self._r_squared = 0.0

        self._adj_r_squared = 1.0 - (1.0 - self._r_squared) * (T - 1) / max(df_resid, 1)

        # Standard errors
        if df_resid > 0:
            sigma2 = SS_res / df_resid
        else:
            sigma2 = SS_res / max(T, 1)

        # Covariance matrix
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)

        cov_beta = sigma2 * XtX_inv
        se = np.sqrt(np.maximum(np.diag(cov_beta), 0))
        self._alpha_se = se[0]
        self._betas_se = se[1:]

        # F-test for overall significance
        if df_resid > 0 and self._r_squared > 0:
            self._f_stat = (self._r_squared / df_model) / ((1.0 - self._r_squared) / df_resid)
        else:
            self._f_stat = 0.0

        self._f_pvalue = 1.0 - f_dist_cdf(self._f_stat, df_model, df_resid) if df_resid > 0 else 1.0
        self._fitted = True

        return {
            "alpha": float(self._alpha),
            "betas": self._betas.tolist(),
            "factor_names": self._factor_names,
            "r_squared": float(self._r_squared),
            "adj_r_squared": float(self._adj_r_squared),
            "f_statistic": float(self._f_stat),
            "f_pvalue": float(self._f_pvalue),
            "n_observations": T,
            "n_factors": K,
        }

    def factor_exposures(self) -> np.ndarray:
        """Return factor exposures (β coefficients).

        Returns
        -------
        np.ndarray of shape (K,)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self._betas.copy()

    def factor_contributions(self) -> dict:
        """Compute each factor's contribution to total variance of fitted returns.

        For factor k:
            contribution = β_k² · Var(f_k) / Var(fitted)

        And also contribution to total return variance (including residual).

        Returns
        -------
        dict with keys:
            contributions: np.ndarray factor contributions to fitted variance
            contributions_to_total: np.ndarray factor contributions to total variance
            factor_names: list of factor names
            residual_contribution: residual share of total variance
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        fitted_var = np.var(self._fitted_values) if len(self._fitted_values) > 1 else 1e-15
        total_var = np.var(self._returns) if len(self._returns) > 1 else 1e-15

        contributions = np.zeros(self._K)
        contributions_to_total = np.zeros(self._K)

        for k in range(self._K):
            fk_var = np.var(self._factors[:, k]) if self._T > 1 else 1e-15
            contrib_fitted = (self._betas[k] ** 2) * fk_var
            contributions[k] = contrib_fitted / fitted_var if fitted_var > 1e-15 else 0.0
            contributions_to_total[k] = contrib_fitted / total_var if total_var > 1e-15 else 0.0

        if total_var > 1e-15:
            residual_contribution = 1.0 - np.sum(contributions_to_total)
            # May be slightly negative due to covariance; clip
            residual_contribution = max(0.0, residual_contribution)
        else:
            residual_contribution = 0.0

        return {
            "contributions": contributions.tolist(),
            "contributions_to_total": contributions_to_total.tolist(),
            "factor_names": self._factor_names,
            "residual_contribution": float(residual_contribution),
        }

    def alpha_significance(self) -> float:
        """α的t统计量

        H0: α = 0
        Returns t-statistic for alpha.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if self._alpha_se is None or self._alpha_se < 1e-15:
            return float(np.sign(self._alpha) * np.inf) if self._alpha != 0 else 0.0
        return float(self._alpha / self._alpha_se)

    def adj_r_squared(self) -> float:
        """Return adjusted R-squared."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return float(self._adj_r_squared)

    def report(self) -> dict:
        """Generate complete factor attribution report."""
        if not self._fitted:
            self.fit()

        alpha_t = self.alpha_significance()
        df_resid = self._T - self._K - 1
        if df_resid > 0:
            alpha_pvalue = 2.0 * (1.0 - t_dist.cdf(abs(alpha_t), df_resid))
        else:
            alpha_pvalue = 1.0

        # Individual factor t-stats
        factor_t_stats = []
        factor_p_values = []
        for k in range(self._K):
            se = self._betas_se[k] if self._betas_se[k] > 1e-15 else 1e-15
            t_val = self._betas[k] / se
            factor_t_stats.append(float(t_val))
            if df_resid > 0:
                factor_p_values.append(float(2.0 * (1.0 - t_dist.cdf(abs(t_val), df_resid))))
            else:
                factor_p_values.append(1.0)

        contribs = self.factor_contributions()

        # Build detailed factor info
        factors_detail = []
        for k in range(self._K):
            factors_detail.append({
                "name": self._factor_names[k],
                "beta": float(self._betas[k]),
                "se": float(self._betas_se[k]),
                "t_stat": factor_t_stats[k],
                "p_value": factor_p_values[k],
                "contrib_to_fitted": contribs["contributions"][k],
                "contrib_to_total": contribs["contributions_to_total"][k],
            })

        return {
            "alpha": float(self._alpha),
            "alpha_se": float(self._alpha_se),
            "alpha_t_stat": float(alpha_t),
            "alpha_p_value": float(alpha_pvalue),
            "factors": factors_detail,
            "r_squared": float(self._r_squared),
            "adj_r_squared": float(self._adj_r_squared),
            "f_statistic": float(self._f_stat),
            "f_pvalue": float(self._f_pvalue),
            "residual_contribution": contribs["residual_contribution"],
            "n_observations": self._T,
            "n_factors": self._K,
        }


def f_dist_cdf(x: float, df1: float, df2: float) -> float:
    """F-distribution CDF using scipy.stats.f if available, else approximation."""
    try:
        from scipy.stats import f as f_dist_scipy
        return f_dist_scipy.cdf(x, df1, df2)
    except ImportError:
        # Fallback approximation
        if x <= 0:
            return 0.0
        # Wilson-Hilferty approximation
        if df2 > 0:
            z = ((x ** (1.0 / 3.0)) * (1.0 - 2.0 / (9.0 * df2)) - (1.0 - 2.0 / (9.0 * df1))) / \
                np.sqrt(2.0 / (9.0 * df1) + (x ** (2.0 / 3.0)) * 2.0 / (9.0 * df2))
            from scipy.stats import norm
            return norm.cdf(z)
        return 0.0
