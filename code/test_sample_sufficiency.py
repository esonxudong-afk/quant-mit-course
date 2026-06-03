"""
Tests for sample_sufficiency.py
"""

import pytest
import numpy as np
from sample_sufficiency import (
    SampleSufficiency,
    SufficiencyResult,
    sample_sufficiency_report,
)


class TestSampleSufficiencyInit:
    """构造器测试"""

    def test_default_init(self):
        ss = SampleSufficiency()
        assert ss.expected_return == 0.003
        assert ss.sigma == 0.03
        assert ss.confidence == 0.95

    def test_custom_init(self):
        ss = SampleSufficiency(expected_return=0.01, sigma=0.05, confidence=0.99)
        assert ss.expected_return == 0.01
        assert ss.sigma == 0.05
        assert ss.confidence == 0.99

    def test_negative_expected_return_raises(self):
        with pytest.raises(ValueError, match="expected_return"):
            SampleSufficiency(expected_return=-0.001)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            SampleSufficiency(sigma=0)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            SampleSufficiency(confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            SampleSufficiency(confidence=0)


class TestRequiredSamples:
    """required_samples 测试"""

    def test_baseline_case(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03)
        n = ss.required_samples(t_stat_target=2.0)
        # n = (0.03 * 2.0 / 0.003)^2 = (20)^2 = 400
        assert n == 400

    def test_higher_return_fewer_samples(self):
        ss = SampleSufficiency(expected_return=0.01, sigma=0.03)
        n = ss.required_samples(t_stat_target=2.0)
        # n = (0.03 * 2.0 / 0.01)^2 = 36
        assert n == 36

    def test_higher_sigma_more_samples(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.05)
        n = ss.required_samples(t_stat_target=2.0)
        # n = (0.05 * 2.0 / 0.003)^2 ≈ 1112
        assert n > 400

    def test_ceiling_behavior(self):
        ss = SampleSufficiency(expected_return=0.01, sigma=0.03)
        n = ss.required_samples(t_stat_target=1.0)
        # n = (0.03 * 1.0 / 0.01)^2 = 9
        assert n == 9

    def test_decimal_rounds_up(self):
        ss = SampleSufficiency(expected_return=0.002, sigma=0.03)
        n = ss.required_samples(t_stat_target=2.0)
        # n = (0.03 * 2.0 / 0.002)^2 = 900, exact
        assert n == 900

    def test_negative_t_stat_raises(self):
        ss = SampleSufficiency()
        with pytest.raises(ValueError, match="t_stat_target"):
            ss.required_samples(t_stat_target=-1.0)

    def test_extreme_small_expected_return(self):
        ss = SampleSufficiency(expected_return=0.0001, sigma=0.03)
        n = ss.required_samples(t_stat_target=2.0)
        # n = (0.03 * 2.0 / 0.0001)^2 = (600)^2 = 360000
        assert n == 360000


class TestCurrentSufficiency:
    """current_sufficiency 测试"""

    def test_large_n(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03)
        result = ss.current_sufficiency(n_trades=400)
        assert result["n_trades"] == 400
        assert result["is_sufficient"] == True
        # se = 0.03 / sqrt(400) = 0.0015
        assert abs(result["standard_error"] - 0.0015) < 1e-8
        # t_stat = 0.003 / 0.0015 = 2.0
        assert abs(result["t_stat"] - 2.0) < 1e-8

    def test_small_n_insufficient(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03)
        result = ss.current_sufficiency(n_trades=100)
        assert result["n_trades"] == 100
        assert result["is_sufficient"] == False
        # se = 0.03 / 10 = 0.003
        assert abs(result["standard_error"] - 0.003) < 1e-8
        # t_stat = 0.003 / 0.003 = 1.0
        assert abs(result["t_stat"] - 1.0) < 1e-8

    def test_zero_trades_raises(self):
        ss = SampleSufficiency()
        with pytest.raises(ValueError, match="n_trades"):
            ss.current_sufficiency(n_trades=0)

    def test_se_decreases_with_n(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03)
        r100 = ss.current_sufficiency(n_trades=100)
        r400 = ss.current_sufficiency(n_trades=400)
        assert r100["standard_error"] > r400["standard_error"]
        assert r100["t_stat"] < r400["t_stat"]


class TestSufficiencyReport:
    """sufficiency_report 测试"""

    def test_report_structure(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03)
        report = ss.sufficiency_report(n_trades=100)
        assert isinstance(report, SufficiencyResult)
        assert report.n_current == 100
        assert report.n_required == 400
        assert report.gap == 300
        assert report.is_sufficient == False
        assert "样本不足" in report.interpretation

    def test_report_sufficient(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03)
        report = ss.sufficiency_report(n_trades=500)
        assert report.is_sufficient == True
        assert report.gap == -100  # 有余
        assert "样本充分" in report.interpretation

    def test_report_exact_boundary(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03)
        report = ss.sufficiency_report(n_trades=400)
        assert report.is_sufficient == True

    # ── 你的币种数据验证 ──
    def test_crypto_scenario(self):
        """σ=3%, μ=0.3%, t_target=2.0 → n_min=400"""
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03)
        n = ss.required_samples(t_stat_target=2.0)
        assert n == 400

        # 179天, 假设每天2笔交易 → 358 trades
        report = ss.sufficiency_report(n_trades=358)
        assert report.is_sufficient == False
        assert report.gap == 42  # 差42笔

        # 验证t值
        assert abs(report.t_stat_current - (0.003 / (0.03 / np.sqrt(358)))) < 1e-6


class TestConvenienceFunction:
    """便捷函数测试"""

    def test_sample_sufficiency_report(self):
        result = sample_sufficiency_report(
            expected_return=0.003,
            sigma=0.03,
            n_trades=500,
        )
        assert isinstance(result, SufficiencyResult)
        assert result.is_sufficient == True


class TestConfidenceMapping:
    """不同置信度下的目标t值"""

    def test_95_confidence(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03, confidence=0.95)
        r = ss.current_sufficiency(n_trades=400)
        # 95% → z ≈ 1.96
        assert abs(r["t_stat_target"] - 1.96) < 0.01

    def test_99_confidence(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03, confidence=0.99)
        r = ss.current_sufficiency(n_trades=400)
        # 99% → z ≈ 2.576
        assert abs(r["t_stat_target"] - 2.576) < 0.01

    def test_90_confidence(self):
        ss = SampleSufficiency(expected_return=0.003, sigma=0.03, confidence=0.90)
        r = ss.current_sufficiency(n_trades=400)
        # 90% → z ≈ 1.645
        assert abs(r["t_stat_target"] - 1.645) < 0.01

    def test_higher_confidence_requires_more_samples(self):
        ss95 = SampleSufficiency(expected_return=0.003, sigma=0.03, confidence=0.95)
        ss99 = SampleSufficiency(expected_return=0.003, sigma=0.03, confidence=0.99)
        # 更高置信度 → 更难满足
        r95 = ss95.current_sufficiency(n_trades=400)
        r99 = ss99.current_sufficiency(n_trades=400)
        assert r95["is_sufficient"] == True
        assert r99["is_sufficient"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
