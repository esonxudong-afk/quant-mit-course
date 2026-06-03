"""
PCA Factor Extractor — 主成分分析因子提取器

从多只股票收益率的 n×m 矩阵做 SVD 分解，提取前 k 个主成分，
输出方差解释率和因子载荷。

维度约定：行 = 股票 (assets), 列 = 日期 (time periods)

Usage:
    from pca_factor_extractor import PCAFactorExtractor

    extractor = PCAFactorExtractor(n_components=3)
    extractor.fit(returns_matrix)  # shape (n_stocks, n_days)

    print(extractor.get_variance_explained())
    print(extractor.get_factor_exposures(0))

    # 因子中性化
    neutral = extractor.neutralize(returns_matrix)
"""

import argparse
import os
import sys

import numpy as np


class PCAFactorExtractor:
    """
    基于 SVD 的 PCA 因子提取器。

    Parameters
    ----------
    n_components : int
        要提取的主成分（因子）数量，默认 3。
    """

    def __init__(self, n_components: int = 3):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self._k = n_components

        # 拟合后填充
        self._n_days: int = 0
        self._n_stocks: int = 0
        self._explained_variance_ratio: np.ndarray | None = None
        self._loadings: np.ndarray | None = None       # (n_stocks, k)
        self._factors: np.ndarray | None = None        # (n_days, k)
        self._singular_values: np.ndarray | None = None
        self._mean: np.ndarray | None = None           # (n_stocks,) 逐资产均值

    def fit(self, returns_matrix: np.ndarray) -> "PCAFactorExtractor":
        """
        对收益率矩阵拟合 PCA 模型。

        Parameters
        ----------
        returns_matrix : np.ndarray, shape (n_stocks, n_days)
            行=股票, 列=日期（时间序列观测）。

        Returns
        -------
        self
        """
        if returns_matrix.ndim != 2:
            raise ValueError("returns_matrix must be 2-dimensional")
        n_stocks, n_days = returns_matrix.shape
        if n_stocks < 1 or n_days < 1:
            raise ValueError("returns_matrix must have at least 1 stock and 1 day")
        if n_days < self._k:
            raise ValueError(
                f"n_components ({self._k}) cannot exceed n_days ({n_days})"
            )

        self._n_stocks = n_stocks
        self._n_days = n_days

        # 对每列（时间维）中心化：等价于每行（股票）减去自己的均值
        self._mean = returns_matrix.mean(axis=1, keepdims=True)  # (n_stocks, 1)
        centered = returns_matrix - self._mean

        # SVD: U (n_stocks, n_stocks), s (min(n_stocks,n_days),), Vt (n_days, n_days)
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)

        self._singular_values = s

        # 方差解释率
        s2 = s ** 2
        total_var = s2.sum()
        self._explained_variance_ratio = s2[:self._k] / total_var

        # 因子载荷: U[:, :k] @ diag(s[:k]) / sqrt(n_days)
        # 这样 loadings 是 (n_stocks, k)，每一列是一个因子的载荷向量
        self._loadings = U[:, :self._k] @ np.diag(s[:self._k]) / np.sqrt(n_days)

        # 因子收益: Vt[:k, :].T → (n_days, k)
        # 注意：Vt 行对应原始矩阵的列（日期维），所以 Vt[:k, :] 是 k×n_days
        self._factors = Vt[:self._k, :].T  # (n_days, k)

        return self

    def get_factor_exposures(self, stock_idx: int) -> np.ndarray:
        """
        返回某只股票对 k 个因子的暴露（载荷）。

        Parameters
        ----------
        stock_idx : int
            股票索引（0-based）。

        Returns
        -------
        np.ndarray, shape (k,)
        """
        self._check_fitted()
        if not (0 <= stock_idx < self._n_stocks):
            raise IndexError(f"stock_idx {stock_idx} out of range [0, {self._n_stocks})")
        return self._loadings[stock_idx]

    def get_variance_explained(self) -> np.ndarray:
        """
        每个因子的方差解释率。

        Returns
        -------
        np.ndarray, shape (k,)
        """
        self._check_fitted()
        return self._explained_variance_ratio

    def get_factors(self) -> np.ndarray:
        """
        因子时间序列（因子收益）。

        Returns
        -------
        np.ndarray, shape (n_days, k)
        """
        self._check_fitted()
        return self._factors

    def get_loadings(self) -> np.ndarray:
        """
        所有股票的因子载荷矩阵。

        Returns
        -------
        np.ndarray, shape (n_stocks, k)
        """
        self._check_fitted()
        return self._loadings

    def get_singular_values(self) -> np.ndarray:
        """
        返回所有奇异值。

        Returns
        -------
        np.ndarray
        """
        self._check_fitted()
        return self._singular_values

    def neutralize(
        self, returns: np.ndarray, factor_indices: list[int] | None = None
    ) -> np.ndarray:
        """
        因子中性化：用 OLS 将每只股票的收益回归到选定因子上，返回残差。

        残差收益率 = 原始收益率 - 因子可解释部分

        Parameters
        ----------
        returns : np.ndarray, shape (n_stocks, n_days)
            待中性化的收益率矩阵。
        factor_indices : list[int] or None
            要中性化的因子索引。None 表示所有因子。

        Returns
        -------
        np.ndarray, shape (n_stocks, n_days)
            因子中性化后的残差收益率。
        """
        self._check_fitted()

        if returns.shape != (self._n_stocks, self._n_days):
            raise ValueError(
                f"returns shape {returns.shape} does not match fitted shape "
                f"({self._n_stocks}, {self._n_days})"
            )

        if factor_indices is None:
            factor_indices = list(range(self._k))
        else:
            # 验证索引
            for idx in factor_indices:
                if not (0 <= idx < self._k):
                    raise ValueError(f"factor index {idx} out of range [0, {self._k})")

        # 提取选定的因子时间序列
        selected_factors = self._factors[:, factor_indices]  # (n_days, m)

        # 对每只股票分别做 OLS: returns[i] ~ selected_factors + intercept
        residuals = np.empty_like(returns)
        # 添加截距
        X = np.column_stack([np.ones(self._n_days), selected_factors])  # (n_days, m+1)

        # 使用正规方程做 OLS
        XtX_inv = np.linalg.pinv(X.T @ X)
        for i in range(self._n_stocks):
            y = returns[i, :]  # (n_days,)
            beta = XtX_inv @ X.T @ y
            residuals[i, :] = y - X @ beta

        return residuals

    def _check_fitted(self) -> None:
        if self._loadings is None:
            raise RuntimeError("PCAFactorExtractor has not been fitted yet. Call fit() first.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(
        description="PCA Factor Extractor — 从收益率矩阵提取主成分因子"
    )
    parser.add_argument(
        "--returns", required=True,
        help="收益率矩阵 .npy 文件路径 (shape: n_stocks x n_days)"
    )
    parser.add_argument(
        "--components", type=int, default=3,
        help="主成分数量 (default: 3)"
    )
    parser.add_argument(
        "--output", default="./output",
        help="输出目录 (default: ./output)"
    )
    args = parser.parse_args()

    returns_matrix = np.load(args.returns)

    extractor = PCAFactorExtractor(n_components=args.components)
    extractor.fit(returns_matrix)

    os.makedirs(args.output, exist_ok=True)

    # 保存方差解释率
    explained = extractor.get_variance_explained()
    np.save(os.path.join(args.output, "explained_variance.npy"), explained)
    print(f"方差解释率: {explained}")
    print(f"累积: {explained.sum():.4f}")

    # 保存因子载荷
    loadings = extractor.get_loadings()
    np.save(os.path.join(args.output, "factor_loadings.npy"), loadings)
    print(f"因子载荷形状: {loadings.shape}")

    # 保存因子时间序列
    factors = extractor.get_factors()
    np.save(os.path.join(args.output, "factor_returns.npy"), factors)
    print(f"因子收益形状: {factors.shape}")

    # 因子中性化
    neutral = extractor.neutralize(returns_matrix)
    np.save(os.path.join(args.output, "neutralized_returns.npy"), neutral)
    print(f"中性化收益形状: {neutral.shape}")

    print(f"\n所有输出已保存至: {args.output}")


if __name__ == "__main__":
    _main()
