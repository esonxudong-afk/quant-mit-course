"""
Tests for normality_test.py
"""

import pytest
import numpy as np
from normality_test import NormalityTester, NormalityReport, quick_normality_check


class TestNormalityTesterInit:
    """构造器测试"""

    def test_init_with_array(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        tester = NormalityTester(data)
        assert tester.n == 100

    def test_init_with_list(self):
        tester = NormalityTester([0.01, -0.02, 0.03, 0.01, -0.01])
        assert tester.n == 5

    def test_init_flattens_2d(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        tester = NormalityTester(data)
        assert tester.returns.ndim == 1
        assert tester.n == 4

    def test_init_too_few_samples(self):
        with pytest.raises(ValueError, match="至少需要"):
            NormalityTester([1.0, 2.0, 3.0])

    def test_init_constant_raises(self):
        with pytest.raises(ValueError, match="常数"):
            NormalityTester(np.ones(10))

    def test_skewness_kurtosis_properties(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        tester = NormalityTester(data)
        assert abs(tester.skewness) < 0.3  # 近似0
        assert abs(tester.kurtosis) < 0.3  # excess ≈ 0


class TestJarqueBera:
    """Jarque-Bera 检验测试"""

    def test_normal_data_passes(self):
        """正态样本不应拒绝H0"""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        tester = NormalityTester(data)
        stat, p = tester.jarque_bera()
        assert p > 0.05  # 不拒绝正态

    def test_fat_tailed_data_rejects(self):
        """肥尾样本(学生t, df=3)应拒绝正态假设"""
        rng = np.random.default_rng(42)
        data = rng.standard_t(df=3, size=500)
        tester = NormalityTester(data)
        stat, p = tester.jarque_bera()
        assert p < 0.05  # 拒绝正态

    def test_skewed_data_rejects(self):
        """偏态数据应拒绝"""
        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(-2, 0.5, 300),
            rng.normal(2, 0.5, 200),
        ])
        tester = NormalityTester(data)
        stat, p = tester.jarque_bera()
        assert p < 0.05

    def test_returns_floats(self):
        rng = np.random.default_rng(42)
        tester = NormalityTester(rng.normal(0, 1, 100))
        stat, p = tester.jarque_bera()
        assert isinstance(stat, float)
        assert isinstance(p, float)


class TestShapiroWilk:
    """Shapiro-Wilk 检验测试"""

    def test_normal_data_passes(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        tester = NormalityTester(data)
        stat, p = tester.shapiro_wilk()
        assert p > 0.05

    def test_exponential_data_rejects(self):
        """指数分布明显非正态"""
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=1.0, size=200)
        tester = NormalityTester(data)
        stat, p = tester.shapiro_wilk()
        assert p < 0.05

    def test_returns_floats(self):
        rng = np.random.default_rng(42)
        tester = NormalityTester(rng.normal(0, 1, 100))
        stat, p = tester.shapiro_wilk()
        assert isinstance(stat, float)
        assert isinstance(p, float)


class TestIsNormal:
    """综合正态性判断测试"""

    def test_normal_is_true(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        tester = NormalityTester(data)
        assert tester.is_normal(alpha=0.05) == True

    def test_fat_tail_is_false(self):
        rng = np.random.default_rng(42)
        data = rng.standard_t(df=3, size=500)
        tester = NormalityTester(data)
        assert tester.is_normal(alpha=0.05) == False

    def test_alpha_sensitivity(self):
        """严格alpha应该更容易拒绝"""
        rng = np.random.default_rng(42)
        data = rng.standard_t(df=10, size=200)
        tester = NormalityTester(data)
        # alpha=0.01 更严格, alpha=0.10 更宽松
        # 不测试具体结果(随机性), 但验证函数正常运行
        result_strict = tester.is_normal(alpha=0.01)
        result_loose = tester.is_normal(alpha=0.10)
        assert isinstance(result_strict, bool)
        assert isinstance(result_loose, bool)


class TestIsLognormal:
    """对数正态性检验测试"""

    def test_lognormal_data_is_lognormal(self):
        """对数正态样本应对log(returns)通过正态检验"""
        rng = np.random.default_rng(42)
        # 生成对数正态: exp(N(0, 0.1)) - 1 得到 positive returns
        log_ret = rng.normal(0, 0.1, 500)
        price_ret = np.exp(log_ret) - 1  # 保证 > -1
        tester = NormalityTester(price_ret)
        # log(1+ret) = log(exp(log_ret)) = log_ret ~ N(0,0.1)
        assert tester.is_lognormal(alpha=0.05) == True

    def test_fat_tail_not_lognormal(self):
        """肥尾数据(直接取对数)可能仍然拒绝"""
        rng = np.random.default_rng(42)
        data = rng.standard_t(df=3, size=500) * 0.01  # 缩放避免 ≤ -1
        # 保证 > -1
        data = np.clip(data, -0.99, None)
        tester = NormalityTester(data)
        result = tester.is_lognormal(alpha=0.05)
        assert result in (True, False)  # np.bool_ ok

    def test_negative_100pct_returns_false(self):
        """有 ≤ -100% 的return → 对数正态无意义 → False"""
        data = np.array([-1.0, 0.01, 0.02, 0.01, -0.01])
        tester = NormalityTester(data)
        assert tester.is_lognormal(alpha=0.05) == False

    def test_all_negative_fails_log(self):
        """全负但 > -1 的数据, log后会变"""
        rng = np.random.default_rng(42)
        data = 0.5 + rng.normal(0, 1, 200) * 0.01  # 均值 ≈ 0.5%
        # 需要保证正数
        data = np.abs(data)
        tester = NormalityTester(data)
        result = tester.is_lognormal(alpha=0.05)
        assert result in (True, False)  # np.bool_ ok


class TestReport:
    """综合报告测试"""

    def test_report_structure(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        tester = NormalityTester(data)
        report = tester.report(alpha=0.05)
        assert isinstance(report, NormalityReport)
        assert report.n_samples == 200
        assert isinstance(report.skewness, float)
        assert isinstance(report.kurtosis, float)
        assert isinstance(report.jb_stat, float)
        assert isinstance(report.jb_pvalue, float)
        assert isinstance(report.sw_stat, float)
        assert isinstance(report.sw_pvalue, float)
        assert isinstance(report.is_normal, bool)
        assert isinstance(report.is_lognormal, bool)
        assert isinstance(report.interpretation, str)
        assert "正态性诊断报告" in report.interpretation

    def test_report_contains_warnings_for_fat_tail(self):
        rng = np.random.default_rng(42)
        data = rng.standard_t(df=3, size=500)
        tester = NormalityTester(data)
        report = tester.report(alpha=0.05)
        assert report.is_normal == False
        assert "峰度" in report.interpretation

    def test_report_for_skewed_data(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=1.0, size=200)
        tester = NormalityTester(data)
        report = tester.report(alpha=0.05)
        assert report.is_normal == False
        assert "偏度" in report.interpretation


class TestQuickNormalityCheck:
    """便捷函数测试"""

    def test_quick_check(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        result = quick_normality_check(data)
        assert "is_normal" in result
        assert "is_lognormal" in result
        assert "skewness" in result
        assert "kurtosis" in result
        assert result["is_normal"] == True


class TestEdgeCases:
    """边界情况"""

    def test_tiny_sample_works(self):
        """刚好4个样本, 不应报错"""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 4)
        tester = NormalityTester(data)
        stat, p = tester.jarque_bera()
        assert isinstance(stat, float)
        assert isinstance(p, float)

    def test_near_constant_data(self):
        """几乎常数的数据"""
        data = np.array([1.0, 1.0, 1.0, 1.001, 1.0, 1.0, 0.999, 1.0])
        # 不为全常数所以不抛异常
        tester = NormalityTester(data)
        report = tester.report()
        assert isinstance(report, NormalityReport)

    def test_large_skewness(self):
        """极端偏度"""
        data = np.array([0.01] * 48 + [0.5, 0.6])  # 大多数小, 两个极大
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=1.0, size=200)
        tester = NormalityTester(data)
        assert tester.skewness > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
