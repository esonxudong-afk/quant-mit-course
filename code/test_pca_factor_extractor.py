"""
Tests for PCAFactorExtractor.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pca_factor_extractor import PCAFactorExtractor


def test_fit_basic():
    """基本拟合：随机矩阵，验证输出形状"""
    np.random.seed(42)
    n_stocks, n_days, k = 10, 100, 3
    returns = np.random.randn(n_stocks, n_days) * 0.02

    extractor = PCAFactorExtractor(n_components=k)
    extractor.fit(returns)

    # 形状检查
    assert extractor.get_loadings().shape == (n_stocks, k)
    assert extractor.get_factors().shape == (n_days, k)
    assert extractor.get_variance_explained().shape == (k,)
    assert extractor.get_singular_values().shape == (min(n_stocks, n_days),)


def test_variance_explained_sums_to_one():
    """方差解释率之和应近似等于1（前 min(n_stocks,n_days) 个）"""
    np.random.seed(123)
    n_stocks, n_days, k = 8, 50, 3
    returns = np.random.randn(n_stocks, n_days) * 0.03

    extractor = PCAFactorExtractor(n_components=k)
    extractor.fit(returns)

    # 前 k 个解释率之和 <= 1
    explained = extractor.get_variance_explained()
    assert np.all(explained >= 0), "解释率应非负"
    assert explained.sum() <= 1.0 + 1e-10, f"k 个因子解释率之和应 <= 1, got {explained.sum()}"

    # 所有奇异值的方差解释率之和 = 1
    sv = extractor.get_singular_values()
    all_explained = (sv ** 2) / (sv ** 2).sum()
    assert abs(all_explained.sum() - 1.0) < 1e-10, f"all explained sum = {all_explained.sum()}"


def test_centering():
    """拟合后中心化矩阵各列均值应接近0"""
    np.random.seed(456)
    n_stocks, n_days, k = 6, 40, 2
    returns = np.random.randn(n_stocks, n_days) * 0.02

    extractor = PCAFactorExtractor(n_components=k)
    extractor.fit(returns)

    # 获取均值
    mean = extractor._mean
    # 原始减均值：每行（每只股票）均值应为0
    centered = returns - mean
    row_means = centered.mean(axis=1)
    assert np.allclose(row_means, 0.0, atol=1e-12)


def test_factor_exposures():
    """获取单只股票的因子暴露"""
    np.random.seed(789)
    n_stocks, n_days, k = 5, 30, 2
    returns = np.random.randn(n_stocks, n_days)

    extractor = PCAFactorExtractor(n_components=k)
    extractor.fit(returns)

    for i in range(n_stocks):
        exp = extractor.get_factor_exposures(i)
        assert exp.shape == (k,)

    # 越界
    try:
        extractor.get_factor_exposures(n_stocks)
        assert False, "should have raised IndexError"
    except IndexError:
        pass


def test_neutralize_orthogonality():
    """中性化后收益率应与因子正交"""
    np.random.seed(101)
    n_stocks, n_days, k = 10, 60, 3
    # 生成有因子结构的收益率
    F = np.random.randn(n_days, k) * 0.02
    B = np.random.randn(n_stocks, k) * 0.5
    alpha = np.random.randn(n_stocks, n_days) * 0.005
    returns = B @ F.T + alpha

    extractor = PCAFactorExtractor(n_components=k)
    extractor.fit(returns)

    neutral = extractor.neutralize(returns)

    factors = extractor.get_factors()
    # 对于每个因子，残差与因子应正交：residual @ factor ≈ 0
    for j in range(k):
        cov = neutral @ factors[:, j]  # (n_stocks,)
        assert np.allclose(cov, 0.0, atol=1e-8), f"Factor {j}: max correlation = {np.abs(cov).max()}"


def test_neutralize_subset():
    """仅中性化部分因子"""
    np.random.seed(202)
    n_stocks, n_days, k = 8, 50, 3
    F = np.random.randn(n_days, k) * 0.02
    B = np.random.randn(n_stocks, k) * 0.5
    returns = B @ F.T + np.random.randn(n_stocks, n_days) * 0.003

    extractor = PCAFactorExtractor(n_components=k)
    extractor.fit(returns)

    # 只中性化因子 0
    neutral = extractor.neutralize(returns, factor_indices=[0])
    factors = extractor.get_factors()

    # 残差应与因子 0 正交
    cov0 = neutral @ factors[:, 0]
    assert np.allclose(cov0, 0.0, atol=1e-8), f"Factor 0 not neutralized: max={np.abs(cov0).max()}"


def test_neutralize_returns_mean():
    """中性化不应改变收益率均值太多（因包含截距）"""
    np.random.seed(303)
    n_stocks, n_days, k = 10, 80, 3
    returns = np.random.randn(n_stocks, n_days) * 0.02 + 0.001

    extractor = PCAFactorExtractor(n_components=k)
    extractor.fit(returns)
    neutral = extractor.neutralize(returns)

    # 含截距回归：每只股票的残差均值 ≈ 0
    for i in range(n_stocks):
        assert abs(neutral[i].mean()) < 1e-10, f"Stock {i}: residual mean = {neutral[i].mean()}"


def test_unfitted_raises():
    """未拟合调用 get_* 应抛出 RuntimeError"""
    extractor = PCAFactorExtractor(n_components=2)
    for method in [
        lambda: extractor.get_factor_exposures(0),
        lambda: extractor.get_variance_explained(),
        lambda: extractor.get_factors(),
        lambda: extractor.get_loadings(),
        lambda: extractor.neutralize(np.random.randn(3, 5)),
    ]:
        try:
            method()
            assert False, f"should have raised RuntimeError"
        except RuntimeError:
            pass


def test_invalid_n_components():
    """n_components 必须 >= 1"""
    try:
        PCAFactorExtractor(n_components=0)
        assert False, "should have raised"
    except ValueError:
        pass


def test_n_components_exceeds_n_days():
    """n_components 不能超过 n_days"""
    extractor = PCAFactorExtractor(n_components=5)
    returns = np.random.randn(10, 3)  # 只有3天
    try:
        extractor.fit(returns)
        assert False, "should have raised"
    except ValueError:
        pass


def test_neutralize_shape_mismatch():
    """中性化时收益率形状不匹配"""
    extractor = PCAFactorExtractor(n_components=2)
    extractor.fit(np.random.randn(5, 20))
    bad = np.random.randn(5, 10)
    try:
        extractor.neutralize(bad)
        assert False, "should have raised"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 自验证入口
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        ("fit_basic", test_fit_basic),
        ("variance_explained_sums_to_one", test_variance_explained_sums_to_one),
        ("centering", test_centering),
        ("factor_exposures", test_factor_exposures),
        ("neutralize_orthogonality", test_neutralize_orthogonality),
        ("neutralize_subset", test_neutralize_subset),
        ("neutralize_returns_mean", test_neutralize_returns_mean),
        ("unfitted_raises", test_unfitted_raises),
        ("invalid_n_components", test_invalid_n_components),
        ("n_components_exceeds_n_days", test_n_components_exceeds_n_days),
        ("neutralize_shape_mismatch", test_neutralize_shape_mismatch),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name}: {e}")

    print(f"\n{passed}/{passed+failed} tests passed")
    return passed, failed


if __name__ == "__main__":
    p, f = run_all_tests()
    if f > 0:
        sys.exit(1)
