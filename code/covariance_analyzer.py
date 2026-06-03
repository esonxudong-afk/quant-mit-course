"""
Covariance Analyzer — 协方差矩阵分析器

对持仓组合的收益率数据构建协方差矩阵，做特征值分解，输出风险归因。

Usage:
    analyzer = CovarianceAnalyzer()
    analyzer.fit(returns, symbols)
    print(analyzer.dominance_ratio())
    print(analyzer.to_dict())
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class CovarianceAnalyzer:
    """协方差矩阵分析器 — 风险归因与组合优化"""

    def __init__(self):
        self._returns: Optional[np.ndarray] = None   # n_assets × n_days
        self._symbols: List[str] = []
        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None
        self._shrinkage_delta: float = 0.0

    # ── fit ────────────────────────────────────────────────────────────

    def fit(self, returns: np.ndarray, symbols: list) -> "CovarianceAnalyzer":
        """
        输入收益率矩阵并计算协方差矩阵。

        Parameters
        ----------
        returns : np.ndarray, shape (n_assets, n_days)
            每行一个资产，每列一个交易日
        symbols : list of str
            资产名称列表，长度必须等于 n_assets
        """
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim != 2:
            raise ValueError(f"returns 必须是 2 维数组，得到 shape={returns.shape}")

        n_assets, n_days = returns.shape
        if len(symbols) != n_assets:
            raise ValueError(f"symbols 长度 {len(symbols)} 与资产数 {n_assets} 不匹配")

        self._returns = returns
        self._symbols = list(symbols)
        self._mean = np.mean(returns, axis=1, keepdims=True)

        # 中心化矩阵
        centered = returns - self._mean                # (n_assets, n_days)

        # 样本协方差：Σ = (X - X̄)(X - X̄)ᵀ / (n-1)
        if n_days > 1:
            self._cov = (centered @ centered.T) / (n_days - 1)
        else:
            # 单日数据：退化情形，只用中心化外积
            self._cov = centered @ centered.T
            self._cov = np.maximum(self._cov, 1e-15)  # 保证非负对角元

        # 缩水估计（资产数 > 天数时自动启用）
        if n_assets > n_days:
            self._apply_shrinkage()

        # 特征值分解
        self._decompose()

        return self

    # ── shrinkage ──────────────────────────────────────────────────────

    def _apply_shrinkage(self):
        """Ledoit-Wolf 风格缩水估计：Σ_shrunk = (1-δ)·Σ_sample + δ·diag(Σ_sample)"""
        diag_vars = np.diag(self._cov)
        total_var = np.sum(diag_vars)
        total_entries = np.sum(np.abs(self._cov))
        if total_entries > 1e-15:
            self._shrinkage_delta = total_var / total_entries
        else:
            self._shrinkage_delta = 0.0

        # 缩水
        shrunk = (1.0 - self._shrinkage_delta) * self._cov \
                 + self._shrinkage_delta * np.diag(diag_vars)
        self._cov = shrunk

    # ── cov_matrix ─────────────────────────────────────────────────────

    def cov_matrix(self) -> np.ndarray:
        """返回协方差矩阵 (n_assets × n_assets)"""
        if self._cov is None:
            raise RuntimeError("请先调用 fit()")
        return self._cov.copy()

    # ── eigen decomposition ────────────────────────────────────────────

    def _decompose(self):
        """内部特征值分解"""
        eigenvalues, eigenvectors = np.linalg.eigh(self._cov)

        # eigh 返回升序，我们翻转为降序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 半正定裁剪
        eigenvalues = np.clip(eigenvalues, 0.0, None)

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

    def eigen_decompose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        特征值分解。

        Returns
        -------
        eigenvalues : np.ndarray, shape (n_assets,)
            降序排列的特征值，所有值 ≥ 0
        eigenvectors : np.ndarray, shape (n_assets, n_assets)
            对应的特征向量（列向量），正交矩阵
        """
        if self._eigenvalues is None:
            raise RuntimeError("请先调用 fit()")
        return self._eigenvalues.copy(), self._eigenvectors.copy()

    # ── risk_decomposition ─────────────────────────────────────────────

    def risk_decomposition(self) -> Dict:
        """
        风险归因。

        Returns
        -------
        dict with:
            eigenvalues           : 特征值列表
            variance_share        : 每个特征值的方差占比
            cumulative_share      : 累积占比
            top_contributor       : 每个特征向量中最大权重的资产
        """
        if self._eigenvalues is None or self._eigenvectors is None:
            raise RuntimeError("请先调用 fit()")

        total = np.sum(self._eigenvalues)
        if total < 1e-15:
            total = 1.0

        shares = self._eigenvalues / total
        cum_shares = np.cumsum(shares)

        top_contributors = []
        for i in range(len(self._eigenvalues)):
            vec = self._eigenvectors[:, i]
            idx_max = int(np.argmax(np.abs(vec)))
            top_contributors.append({
                "asset": self._symbols[idx_max] if idx_max < len(self._symbols) else f"asset_{idx_max}",
                "weight": float(vec[idx_max]),
                "index": idx_max,
            })

        return {
            "eigenvalues": self._eigenvalues.tolist(),
            "variance_share": shares.tolist(),
            "cumulative_share": cum_shares.tolist(),
            "top_contributor": top_contributors,
        }

    # ── dominance_ratio ────────────────────────────────────────────────

    def dominance_ratio(self) -> Dict:
        """
        第一特征值占比 λ₁ / Σλᵢ，衡量分散化程度。

        Interpretation:
            > 80%  → 标的基本是一个篮子
            50-80% → 有分散但不够
            < 50%  → 较好的分散化

        Returns
        -------
        dict with keys: ratio, interpretation, threshold_80, threshold_50
        """
        if self._eigenvalues is None:
            raise RuntimeError("请先调用 fit()")

        total = np.sum(self._eigenvalues)
        if total < 1e-15:
            ratio = 1.0
        else:
            ratio = float(self._eigenvalues[0] / total)

        if ratio > 0.80:
            interp = "标的基本是一个篮子 (>80%)"
        elif ratio > 0.50:
            interp = "有分散但不够 (50-80%)"
        else:
            interp = "较好的分散化 (<50%)"

        return {
            "ratio": ratio,
            "interpretation": interp,
            "threshold_80": ratio > 0.80,
            "threshold_50": ratio > 0.50,
        }

    # ── min_variance_portfolio ─────────────────────────────────────────

    def min_variance_portfolio(self) -> np.ndarray:
        """
        最小方差组合权重（闭式解）。

        w_min = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1)

        Returns
        -------
        weights : np.ndarray, shape (n_assets,)
            归一化权重，sum = 1
        """
        if self._cov is None:
            raise RuntimeError("请先调用 fit()")

        n = self._cov.shape[0]
        ones = np.ones(n)

        # 如果协方差矩阵奇异，用伪逆
        try:
            inv_cov = np.linalg.inv(self._cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(self._cov)

        numerator = inv_cov @ ones                 # Σ⁻¹·1
        denominator = ones @ numerator             # 1ᵀ·Σ⁻¹·1

        if abs(denominator) < 1e-15:
            # 退化为等权
            return np.ones(n) / n

        return numerator / denominator

    # ── to_dict ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """
        完整输出字典。

        Returns
        -------
        dict with:
            cov_matrix         : 协方差矩阵
            eigenvalues        : 特征值
            eigenvectors       : 特征向量（逐列）
            risk_decomposition : 风险归因
            dominance          : 主导特征值分析
            min_variance_weights : 最小方差组合权重
            symbols            : 资产名称
            shrinkage_delta    : 缩水参数（0 表示未缩水）
        """
        risk = self.risk_decomposition()
        dom = self.dominance_ratio()
        mvp = self.min_variance_portfolio()

        return {
            "cov_matrix": self._cov.tolist(),
            "eigenvalues": self._eigenvalues.tolist(),
            "eigenvectors": [self._eigenvectors[:, i].tolist() for i in range(len(self._eigenvalues))],
            "risk_decomposition": risk,
            "dominance": dom,
            "min_variance_weights": mvp.tolist(),
            "symbols": self._symbols,
            "shrinkage_delta": self._shrinkage_delta,
        }
