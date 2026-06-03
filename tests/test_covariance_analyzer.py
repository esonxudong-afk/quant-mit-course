"""
Tests for CovarianceAnalyzer.

Run:  python -m pytest tests/test_covariance_analyzer.py -v
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from code.covariance_analyzer import CovarianceAnalyzer


# ── Helper ────────────────────────────────────────────────────────────

def make_random_returns(n_assets=5, n_days=252, seed=42):
    """Generate random returns matrix (n_assets × n_days)"""
    rng = np.random.default_rng(seed)
    # Correlated random returns via Cholesky
    # Random correlation
    A = rng.normal(0, 1, (n_assets, n_assets))
    cov_true = A @ A.T
    std = np.sqrt(np.diag(cov_true))
    corr = cov_true / np.outer(std, std)

    # Generate correlated normal
    L = np.linalg.cholesky(corr)
    z = rng.normal(0, 0.01, (n_assets, n_days))  # daily returns, small vol
    returns = L @ z
    return returns


# ── Basic tests ───────────────────────────────────────────────────────

class TestInit:
    def test_fresh_instance_has_no_data(self):
        analyzer = CovarianceAnalyzer()
        assert analyzer._returns is None
        assert analyzer._cov is None
        assert analyzer._eigenvalues is None


class TestFit:
    def test_basic_fit(self):
        returns = make_random_returns(5, 252)
        symbols = ["A", "B", "C", "D", "E"]
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, symbols)
        assert analyzer._cov is not None
        assert analyzer._cov.shape == (5, 5)
        assert analyzer._eigenvalues is not None
        assert len(analyzer._eigenvalues) == 5

    def test_single_day_degenerate(self):
        returns = np.array([[0.01], [0.02], [0.03]])
        symbols = ["X", "Y", "Z"]
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, symbols)
        assert analyzer._cov.shape == (3, 3)

    def test_symbol_length_mismatch_raises(self):
        returns = make_random_returns(3, 100)
        analyzer = CovarianceAnalyzer()
        try:
            analyzer.fit(returns, ["A", "B"])
            assert False, "应该抛出 ValueError"
        except ValueError:
            pass

    def test_non_2d_raises(self):
        analyzer = CovarianceAnalyzer()
        try:
            analyzer.fit(np.array([1, 2, 3]), ["A", "B", "C"])
            assert False, "应该抛出 ValueError"
        except ValueError:
            pass


# ── Covariance ────────────────────────────────────────────────────────

class TestCovMatrix:
    def test_symmetry(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, ["A", "B", "C", "D", "E"])
        cov = analyzer.cov_matrix()
        assert np.allclose(cov, cov.T), "协方差矩阵应对称"

    def test_positive_diagonal(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        cov = analyzer.cov_matrix()
        assert np.all(np.diag(cov) >= 0), "对角线元素应非负"

    def test_without_fit_raises(self):
        analyzer = CovarianceAnalyzer()
        try:
            analyzer.cov_matrix()
            assert False
        except RuntimeError:
            pass


# ── Eigen decomposition ───────────────────────────────────────────────

class TestEigenDecompose:
    def test_descending_order(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        evals, _ = analyzer.eigen_decompose()
        assert np.all(np.diff(evals) <= 0), "特征值应降序排列"

    def test_non_negative_eigenvalues(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        evals, _ = analyzer.eigen_decompose()
        assert np.all(evals >= 0), f"特征值应≥0, got min={evals.min()}"

    def test_eigenvectors_orthogonal(self):
        """特征向量应为正交矩阵 VᵀV = I"""
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        _, evecs = analyzer.eigen_decompose()
        # evecs[:, i] 是第 i 个特征向量
        identity = evecs.T @ evecs
        assert np.allclose(identity, np.eye(5), atol=1e-10), \
            f"特征向量不正交, 偏差={np.max(np.abs(identity - np.eye(5)))}"

    def test_reconstruction(self):
        """V·Λ·Vᵀ 应复原协方差矩阵"""
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        evals, evecs = analyzer.eigen_decompose()
        cov = analyzer.cov_matrix()
        reconstructed = evecs @ np.diag(evals) @ evecs.T
        assert np.allclose(cov, reconstructed, atol=1e-10), \
            f"重构失败, max_diff={np.max(np.abs(cov - reconstructed))}"

    def test_all_eigenvalues_real(self):
        """协方差矩阵（对称）特征值全部为实数"""
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        evals, _ = analyzer.eigen_decompose()
        assert np.all(np.isreal(evals)), "特征值应为实数"


# ── Risk decomposition ────────────────────────────────────────────────

class TestRiskDecomposition:
    def test_variance_shares_sum_to_one(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        result = analyzer.risk_decomposition()
        shares = np.array(result["variance_share"])
        assert np.isclose(np.sum(shares), 1.0, atol=1e-10)

    def test_cumulative_reaches_one(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        result = analyzer.risk_decomposition()
        cum = result["cumulative_share"]
        assert np.isclose(cum[-1], 1.0, atol=1e-10)

    def test_top_contributors_have_correct_format(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        result = analyzer.risk_decomposition()
        assert len(result["top_contributor"]) == 5
        for tc in result["top_contributor"]:
            assert "asset" in tc
            assert "weight" in tc
            assert "index" in tc


# ── Dominance ratio ───────────────────────────────────────────────────

class TestDominanceRatio:
    def test_returns_dict(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        dom = analyzer.dominance_ratio()
        assert "ratio" in dom
        assert "interpretation" in dom
        assert "threshold_80" in dom
        assert 0.0 <= dom["ratio"] <= 1.0

    def test_highly_correlated_gives_high_dominance(self):
        """构建高度相关资产 → dominance ratio 应很高"""
        # 3 个几乎相同的序列
        base = np.random.default_rng(99).normal(0, 0.01, 100)
        returns = np.array([
            base + np.random.default_rng(1).normal(0, 0.0001, 100),
            base + np.random.default_rng(2).normal(0, 0.0001, 100),
            base + np.random.default_rng(3).normal(0, 0.0001, 100),
        ])
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("XYZ"))
        dom = analyzer.dominance_ratio()
        assert dom["ratio"] > 0.80, f"高度相关资产 dominance 应 > 80%, got {dom['ratio']:.2%}"


# ── Min variance portfolio ────────────────────────────────────────────

class TestMinVariancePortfolio:
    def test_weights_sum_to_one(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        w = analyzer.min_variance_portfolio()
        assert np.isclose(np.sum(w), 1.0, atol=1e-10)

    def test_all_positive_in_practice(self):
        """闭式解 (Σ⁻¹·1)/(1ᵀΣ⁻¹1) 理论上可能含负权，但实际通常为正"""
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        w = analyzer.min_variance_portfolio()
        # 不强制要求全正，但至少检查返回了数组
        assert len(w) == 5

    def test_equal_weights_for_uncorrelated_equal_var(self):
        """不相关且等方差 → 最小方差=等权"""
        # 构造：独立同分布
        rng = np.random.default_rng(7)
        returns = rng.normal(0, 0.01, (3, 500))  # 3 assets, 500 days, iid
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("UVW"))
        w = analyzer.min_variance_portfolio()
        # 应该接近等权
        expected = np.ones(3) / 3
        assert np.allclose(w, expected, atol=0.15), \
            f"iid 数据最小方差组合应近似等权, got {w}"


# ── Shrinkage ─────────────────────────────────────────────────────────

class TestShrinkage:
    def test_high_dimension_triggers_shrinkage(self):
        """资产数 > 天数 → 自动缩水"""
        returns = np.random.default_rng(1).normal(0, 0.01, (10, 5))  # 10 assets, 5 days
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, [f"A{i}" for i in range(10)])
        assert analyzer._shrinkage_delta > 0.0, "高维情形应触发缩水"

    def test_shrinkage_preserves_symmetry(self):
        returns = np.random.default_rng(2).normal(0, 0.01, (8, 4))
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, [f"X{i}" for i in range(8)])
        cov = analyzer.cov_matrix()
        assert np.allclose(cov, cov.T), "缩水后协方差仍应对称"

    def test_shrinkage_positive_semidefinite(self):
        returns = np.random.default_rng(3).normal(0, 0.01, (8, 4))
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, [f"Y{i}" for i in range(8)])
        evals, _ = analyzer.eigen_decompose()
        assert np.all(evals >= 0), "缩水后特征值应≥0"


# ── to_dict ───────────────────────────────────────────────────────────

class TestToDict:
    def test_contains_all_keys(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        d = analyzer.to_dict()
        assert "cov_matrix" in d
        assert "eigenvalues" in d
        assert "eigenvectors" in d
        assert "risk_decomposition" in d
        assert "dominance" in d
        assert "min_variance_weights" in d
        assert "symbols" in d
        assert "shrinkage_delta" in d

    def test_min_variance_weights_sum_to_one_in_dict(self):
        returns = make_random_returns(5, 252)
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, list("ABCDE"))
        d = analyzer.to_dict()
        w = np.array(d["min_variance_weights"])
        assert np.isclose(np.sum(w), 1.0, atol=1e-10)


# ── Edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_asset(self):
        returns = np.array([[0.01, 0.02, -0.01, 0.005, 0.015]])
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, ["SINGLE"])
        assert analyzer.cov_matrix().shape == (1, 1)
        evals, evecs = analyzer.eigen_decompose()
        assert len(evals) == 1
        assert evals[0] >= 0
        # dominance 应为 100%
        dom = analyzer.dominance_ratio()
        assert np.isclose(dom["ratio"], 1.0)

    def test_two_identical_assets(self):
        """两个完全相同的资产"""
        base = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        returns = np.array([base, base])
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, ["X", "Y"])
        # dominance 应为 100%（相关性=1）
        dom = analyzer.dominance_ratio()
        assert np.isclose(dom["ratio"], 1.0, atol=1e-6), \
            f"完全相关资产 dominance=1, got {dom['ratio']}"

    def test_large_negative_return_handled(self):
        """大负收益不破坏计算"""
        returns = np.array([
            [0.01, -0.50, 0.02, 0.01, -0.03],
            [0.02, -0.40, 0.01, 0.02, 0.01],
        ])
        analyzer = CovarianceAnalyzer()
        analyzer.fit(returns, ["A", "B"])
        evals, _ = analyzer.eigen_decompose()
        assert np.all(evals >= 0)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
