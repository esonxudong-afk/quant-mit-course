"""
Sample Sufficiency Calculator (LLN-based)
===========================================
基于大数定律(LLN)，计算策略统计显著性所需的最小样本量。

核心公式:
  n_min = (σ × t_target / μ)²

其中:
  - σ: 每笔交易收益率的标准差
  - μ: 期望收益率 (expected_return)
  - t_target: 目标t统计量 (通常2.0对应约95%置信度)

参考: Breuer et al. - Pitfalls of Trading Strategy Analysis
       Harvey et al. - ...and the Cross-Section of Expected Returns
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SufficiencyResult:
    """样本充分性结果"""
    n_current: int
    n_required: int
    gap: int                    # 正数=不足, 负数=有余
    is_sufficient: bool
    t_stat_current: float
    t_stat_target: float
    standard_error_current: float
    expected_return: float
    sigma: float
    interpretation: str


class SampleSufficiency:
    """基于大数定律的样本充分性检验器"""

    def __init__(
        self,
        expected_return: float = 0.003,
        sigma: float = 0.03,
        confidence: float = 0.95,
    ):
        """
        Parameters
        ----------
        expected_return : float
            期望每笔交易收益率 (μ), 默认0.3%
        sigma : float
            每笔交易收益率标准差 (σ), 默认3%
        confidence : float
            置信水平, 默认0.95
        """
        if expected_return <= 0:
            raise ValueError("expected_return must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")

        self.expected_return = expected_return
        self.sigma = sigma
        self.confidence = confidence

    def required_samples(self, t_stat_target: float = 2.0) -> int:
        """
        计算达到目标t统计量所需的最小样本量。

        n_min = (σ × t_target / μ)²

        Parameters
        ----------
        t_stat_target : float
            目标t统计量, 默认2.0

        Returns
        -------
        int : 所需最小交易数 (向上取整)
        """
        if t_stat_target <= 0:
            raise ValueError("t_stat_target must be positive")

        n_raw = (self.sigma * t_stat_target / self.expected_return) ** 2
        return int(np.ceil(n_raw))

    def current_sufficiency(self, n_trades: int) -> Dict[str, float]:
        """
        计算当前样本量下的统计量。

        Parameters
        ----------
        n_trades : int
            当前已执行的交易次数

        Returns
        -------
        dict with:
            - n_trades: 当前交易数
            - standard_error: 标准误 σ/√n
            - t_stat: 当前t统计量 = μ / SE
            - t_stat_target: 目标t统计量 (基于confidence算出的z值)
            - is_sufficient: 是否已达到充分性
        """
        if n_trades <= 0:
            raise ValueError("n_trades must be positive")

        sqrt_n = np.sqrt(n_trades)
        se = self.sigma / sqrt_n
        t_stat = self.expected_return / se

        # 从置信度反推目标t (双侧)
        from scipy.stats import norm
        target = norm.ppf((1 + self.confidence) / 2)

        return {
            "n_trades": n_trades,
            "standard_error": se,
            "t_stat": t_stat,
            "t_stat_target": target,
            "is_sufficient": abs(t_stat) >= target,
        }

    def sufficiency_report(self, n_trades: int) -> SufficiencyResult:
        """
        生成完整的样本充分性报告。

        Parameters
        ----------
        n_trades : int
            当前已执行的交易次数

        Returns
        -------
        SufficiencyResult
        """
        from scipy.stats import norm

        n_required = self.required_samples()
        gap = n_required - n_trades
        is_sufficient = n_trades >= n_required

        sqrt_n = np.sqrt(n_trades) if n_trades > 0 else 1.0
        se = self.sigma / sqrt_n
        t_stat = self.expected_return / se
        t_target = norm.ppf((1 + self.confidence) / 2)

        # 解读
        if is_sufficient:
            interpretation = (
                f"✅ 样本充分. 当前 {n_trades} 笔交易 ≥ 所需 {n_required} 笔.\n"
                f"   t={t_stat:.2f} ≥ 目标 t={t_target:.2f}, "
                f"标准误 SE={se:.4f}."
            )
        else:
            additional_needed = gap
            interpretation = (
                f"❌ 样本不足. 当前 {n_trades} 笔交易 < 所需 {n_required} 笔.\n"
                f"   缺口: {additional_needed} 笔.\n"
                f"   t={t_stat:.2f} < 目标 t={t_target:.2f}, "
                f"标准误 SE={se:.4f}.\n"
                f"   建议: 继续收集数据, 或检查策略收益是否被高估."
            )

        return SufficiencyResult(
            n_current=n_trades,
            n_required=n_required,
            gap=gap,
            is_sufficient=is_sufficient,
            t_stat_current=t_stat,
            t_stat_target=t_target,
            standard_error_current=se,
            expected_return=self.expected_return,
            sigma=self.sigma,
            interpretation=interpretation,
        )


# ── Convenience function ────────────────────────────────────────────

def sample_sufficiency_report(
    expected_return: float = 0.003,
    sigma: float = 0.03,
    n_trades: int = 100,
    confidence: float = 0.95,
) -> SufficiencyResult:
    """便捷函数: 一键生成样本充分性报告"""
    checker = SampleSufficiency(
        expected_return=expected_return,
        sigma=sigma,
        confidence=confidence,
    )
    return checker.sufficiency_report(n_trades)
