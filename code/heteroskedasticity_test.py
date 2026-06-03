"""
异方差检验模块 — Heteroskedasticity Test

检验收益率序列是否满足Gauss-Markov的恒定方差假设。
提供 BP检验、White检验、GQ检验 以及 异方差稳健标准误。

Usage:
    ht = HeteroskedasticityTest(residuals, X)
    result = ht.breusch_pagan()
    result = ht.white_test()
    result = ht.goldfeld_quandt()
    print(ht.report())
"""

import numpy as np
from scipy.stats import chi2, f


class HeteroskedasticityTest:
    """异方差检验

    Parameters
    ----------
    residuals : np.ndarray, shape (n,)
        OLS回归残差
    X : np.ndarray, shape (n, k)
        设计矩阵（含截距项）
    """

    def __init__(self, residuals: np.ndarray, X: np.ndarray):
        if residuals.ndim != 1:
            raise ValueError("residuals must be 1D array")
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if len(residuals) != X.shape[0]:
            raise ValueError("residuals and X must have same number of observations")

        self._residuals = residuals.astype(np.float64)
        self._X = X.astype(np.float64)
        self._n = len(residuals)
        self._k = X.shape[1]  # number of parameters (including intercept)
        self._e2 = self._residuals ** 2
        self._sigma2 = np.mean(self._e2)  # mean squared residual

    # ─── BP 检验 (Breusch-Pagan) ────────────────────────────────────────────

    def breusch_pagan(self) -> dict:
        """Breusch-Pagan 异方差检验

        H0: 同方差 (homoskedasticity)
        H1: 方差与自变量线性相关

        Procedure:
        1. 用 X 回归 e² 得到 R²
        2. LM = n * R² ~ χ²(k-1)

        Returns
        -------
        dict with keys: statistic, p_value, df, alpha_01, alpha_05, alpha_10, reject_H0
        """
        # Regress squared residuals on X
        try:
            beta_aux, _, _, _ = np.linalg.lstsq(self._X, self._e2, rcond=None)
        except np.linalg.LinAlgError:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "df": self._k - 1,
                "method": "Breusch-Pagan",
                "error": "Singular matrix",
            }

        e2_hat = self._X @ beta_aux
        ssr = np.sum((self._e2 - np.mean(self._e2)) ** 2)
        sse = np.sum((self._e2 - e2_hat) ** 2)
        r_squared_aux = 1.0 - sse / ssr if ssr > 1e-15 else 0.0
        r_squared_aux = max(0.0, min(r_squared_aux, 1.0))

        lm_stat = self._n * r_squared_aux
        df = self._k - 1
        p_value = 1.0 - chi2.cdf(lm_stat, df)

        return self._make_result(lm_stat, p_value, df, "Breusch-Pagan")

    # ─── White 检验 ─────────────────────────────────────────────────────────

    def white_test(self) -> dict:
        """White's General Test for Heteroskedasticity

        H0: 同方差
        H1: 异方差（不限定具体形式）

        辅助回归: e² ~ X + X² + X_cross_terms
        统计量: n * R² ~ χ²(df)

        Returns
        -------
        dict with keys: statistic, p_value, df, alpha_01, alpha_05, alpha_10, reject_H0
        """
        # Build auxiliary regressors: X columns, squared X columns, cross-terms
        # Exclude intercept column from squaring (if first column is all 1s)
        # We determine which column is intercept by checking near-constant

        n, k = self._X.shape

        # Determine intercept column index (if exists)
        intercept_col = -1
        for j in range(k):
            if np.allclose(self._X[:, j], 1.0, atol=1e-10):
                intercept_col = j
                break

        # Build auxiliary matrix Z
        aux_cols = []
        for j in range(k):
            aux_cols.append(self._X[:, j])  # original columns
        # Add squared terms for non-intercept columns
        for j in range(k):
            if j != intercept_col:
                aux_cols.append(self._X[:, j] ** 2)
        # Add cross terms for non-intercept columns (j < i)
        for j in range(k):
            if j == intercept_col:
                continue
            for i in range(j + 1, k):
                if i == intercept_col:
                    continue
                aux_cols.append(self._X[:, j] * self._X[:, i])

        Z = np.column_stack(aux_cols)
        df = Z.shape[1] - 1  # minus intercept

        # Need to handle potential near-singularity — use pinv for stability
        try:
            beta_aux, _, _, _ = np.linalg.lstsq(Z, self._e2, rcond=None)
        except np.linalg.LinAlgError:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "df": df,
                "method": "White",
                "error": "Singular matrix in auxiliary regression",
            }

        e2_hat = Z @ beta_aux
        ssr = np.sum((self._e2 - np.mean(self._e2)) ** 2)
        sse = np.sum((self._e2 - e2_hat) ** 2)
        r_squared_aux = 1.0 - sse / ssr if ssr > 1e-15 else 0.0
        r_squared_aux = max(0.0, min(r_squared_aux, 1.0))

        lm_stat = n * r_squared_aux
        p_value = 1.0 - chi2.cdf(lm_stat, max(df, 1))

        return self._make_result(lm_stat, p_value, max(df, 1), "White")

    # ─── Goldfeld-Quandt 检验 ────────────────────────────────────────────────

    def goldfeld_quandt(self, split_ratio: float = 0.3) -> dict:
        """Goldfeld-Quandt Test for Heteroskedasticity

        H0: 同方差
        H1: 方差随某个排序变量单调变化（方差递增）

        将样本按某个维度排序后去掉中间 split_ratio 部分，
        分别对前后两段做回归，比较残差方差。

        Procedure:
        1. 去掉中间 c = int(n * split_ratio) 个观测
        2. 分别对前后 (n-c)/2 个观测回归
        3. F = RSS2/df2 / (RSS1/df1) ~ F(df2, df1)

        Parameters
        ----------
        split_ratio : float
            去掉的中间观测比例，默认 0.3

        Returns
        -------
        dict
        """
        c = int(self._n * split_ratio)
        if c < 1:
            c = 1
        n1 = (self._n - c) // 2
        n2 = self._n - c - n1

        if n1 <= self._k or n2 <= self._k:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "df1": n1 - self._k,
                "df2": n2 - self._k,
                "method": "Goldfeld-Quandt",
                "error": "Insufficient degrees of freedom after split",
            }

        # First segment
        X1 = self._X[:n1, :]
        y1_resid2 = self._e2[:n1]

        # Last segment
        X2 = self._X[-n2:, :]
        y2_resid2 = self._e2[-n2:]

        # Compute RSS for each segment using squared residuals
        rss1 = np.sum(y1_resid2)
        rss2 = np.sum(y2_resid2)

        df1 = n1 - self._k
        df2 = n2 - self._k

        if df1 <= 0 or df2 <= 0:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "df1": df1,
                "df2": df2,
                "method": "Goldfeld-Quandt",
                "error": "Non-positive degrees of freedom",
            }

        if rss1 < 1e-15:
            rss1 = 1e-15
        if rss2 < 1e-15:
            rss2 = 1e-15

        # F = larger_variance / smaller_variance (one-sided test)
        var1 = rss1 / df1
        var2 = rss2 / df2

        if var2 > var1:
            f_stat = var2 / var1
            df_num, df_den = df2, df1
        else:
            f_stat = var1 / var2
            df_num, df_den = df1, df2

        p_value = 1.0 - f.cdf(f_stat, df_num, df_den)

        return {
            "statistic": float(f_stat),
            "p_value": float(p_value),
            "df1": df1,
            "df2": df2,
            "df_num": df_num,
            "df_den": df_den,
            "method": "Goldfeld-Quandt",
            "alpha_01": f_stat > f.ppf(0.99, df_num, df_den),
            "alpha_05": f_stat > f.ppf(0.95, df_num, df_den),
            "alpha_10": f_stat > f.ppf(0.90, df_num, df_den),
            "reject_H0": p_value < 0.05,
        }

    # ─── 综合判断 ───────────────────────────────────────────────────────────

    def is_homoskedastic(self, alpha: float = 0.05) -> bool:
        """综合判断是否同方差

        使用 BP 和 White 检验的p值做综合判断。
        如果两个检验都无法拒绝H0，则认为同方差。

        Parameters
        ----------
        alpha : float
            显著性水平

        Returns
        -------
        bool
        """
        bp = self.breusch_pagan()
        wt = self.white_test()

        bp_ok = bp.get("p_value", 0.0) > alpha
        wt_ok = wt.get("p_value", 0.0) > alpha

        return bp_ok and wt_ok

    # ─── 异方差稳健标准误 ────────────────────────────────────────────────────

    def robust_se(self, beta_hat: np.ndarray, cov_type: str = "HC1") -> np.ndarray:
        """异方差稳健标准误 (Heteroskedasticity-Consistent Standard Errors)

        计算 White / MacKinnon-White 异方差稳健协方差矩阵估计。

        HC0: White's original estimator
            V = (X'X)⁻¹ X' diag(e²) X (X'X)⁻¹
        HC1: HC0 * n / (n-k)  (最常用)
        HC2: e² / (1-h)  其中 h = diag(X (X'X)⁻¹ X')
        HC3: e² / (1-h)²  (最保守)

        Parameters
        ----------
        beta_hat : np.ndarray
            OLS估计的系数向量
        cov_type : str
            "HC0", "HC1", "HC2", or "HC3"

        Returns
        -------
        np.ndarray of standard errors for each coefficient
        """
        X = self._X
        e = self._residuals
        n, k = X.shape

        # X'X inverse
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)

        if cov_type == "HC0":
            # White's original
            omega = np.diag(e ** 2)
            V = XtX_inv @ X.T @ omega @ X @ XtX_inv

        elif cov_type == "HC1":
            # HC0 scaled by n/(n-k) — most common in practice
            V_hc0 = XtX_inv @ X.T @ np.diag(e ** 2) @ X @ XtX_inv
            V = V_hc0 * (n / (n - k))

        elif cov_type == "HC2":
            # Leverage-adjusted
            H = X @ XtX_inv @ X.T
            h = np.diag(H)
            h = np.clip(h, 0.0, 1.0 - 1e-15)  # ensure 1-h > 0
            w = e ** 2 / (1.0 - h)
            V = XtX_inv @ X.T @ np.diag(w) @ X @ XtX_inv

        elif cov_type == "HC3":
            # More conservative leverage adjustment
            H = X @ XtX_inv @ X.T
            h = np.diag(H)
            h = np.clip(h, 0.0, 1.0 - 1e-15)
            w = e ** 2 / ((1.0 - h) ** 2)
            V = XtX_inv @ X.T @ np.diag(w) @ X @ XtX_inv

        else:
            raise ValueError(f"Unknown cov_type: {cov_type}. Use HC0, HC1, HC2, or HC3.")

        se = np.sqrt(np.diag(V))
        # Handle numerical issues
        se = np.maximum(se, 0.0)
        return se

    # ─── 报告 ────────────────────────────────────────────────────────────────

    def report(self) -> dict:
        """生成完整的异方差检验报告"""
        bp = self.breusch_pagan()
        wt = self.white_test()
        gq = self.goldfeld_quandt()

        return {
            "n_observations": self._n,
            "n_parameters": self._k,
            "breusch_pagan": {
                "statistic": round(bp.get("statistic", np.nan), 4),
                "p_value": round(bp.get("p_value", np.nan), 4),
                "reject_H0": bp.get("reject_H0", None),
            },
            "white_test": {
                "statistic": round(wt.get("statistic", np.nan), 4),
                "p_value": round(wt.get("p_value", np.nan), 4),
                "reject_H0": wt.get("reject_H0", None),
            },
            "goldfeld_quandt": {
                "statistic": round(gq.get("statistic", np.nan), 4),
                "p_value": round(gq.get("p_value", np.nan), 4),
                "reject_H0": gq.get("reject_H0", None),
            },
            "is_homoskedastic": bool(self.is_homoskedastic()),
        }

    # ─── helpers ────────────────────────────────────────────────────────────

    def _make_result(self, statistic: float, p_value: float, df: int,
                     method: str) -> dict:
        """Wrap test results into a standard dict."""
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "df": df,
            "method": method,
            "alpha_01": p_value < 0.01,
            "alpha_05": p_value < 0.05,
            "alpha_10": p_value < 0.10,
            "reject_H0": p_value < 0.05,
        }
