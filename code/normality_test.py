"""
Normality Tester — 收益率正态性与对数正态性检验
================================================

检验收益率序列是否服从正态分布或对数正态分布。
使用 Jarque-Bera 检验 (基于偏度和峰度) 和 Shapiro-Wilk 检验。

对于对冲基金/量化策略收益分析:
  - 如果收益呈正态 → 可以使用均值-方差框架
  - 如果收益呈肥尾 → 需要更高阶的风险度量 (VaR, CVaR)
  - 如果对数收益呈正态 → 适合对价格建模 (几何布朗运动)

参考:
  - Jarque & Bera (1987)
  - Shapiro & Wilk (1965)
  - Lo (2001) - Risk Management for Hedge Funds
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NormalityReport:
    """正态性检验综合报告"""
    n_samples: int
    skewness: float
    kurtosis: float              # excess kurtosis
    jb_stat: float
    jb_pvalue: float
    sw_stat: float
    sw_pvalue: float
    is_normal: bool
    is_lognormal: bool
    is_log_normal_sw: Optional[float] = None
    is_log_normal_jb: Optional[float] = None
    log_sw_stat: Optional[float] = None
    log_sw_pvalue: Optional[float] = None
    log_jb_stat: Optional[float] = None
    log_jb_pvalue: Optional[float] = None
    interpretation: str = ""


class NormalityTester:
    """收益分布正态性检验器"""

    def __init__(self, returns: np.ndarray):
        """
        Parameters
        ----------
        returns : np.ndarray
            收益率序列 (1D array). 如果全零或常数, 检验无意义.
        """
        returns = np.asarray(returns, dtype=float).ravel()

        if len(returns) < 4:
            raise ValueError("至少需要4个观测值")
        if np.all(returns == returns[0]):
            raise ValueError("常数序列无法检验正态性")

        self.returns = returns
        self.n = len(returns)

        # 预先计算偏度和峰度
        self._skewness = float(stats.skew(returns, bias=False))
        self._kurtosis = float(stats.kurtosis(returns, bias=False))  # excess

    @property
    def skewness(self) -> float:
        return self._skewness

    @property
    def kurtosis(self) -> float:
        """Excess kurtosis"""
        return self._kurtosis

    def jarque_bera(self) -> Tuple[float, float]:
        """
        Jarque-Bera 检验

        H0: 数据服从正态分布
        H1: 数据不服从正态分布

        JB = n/6 * (S² + (K-3)²/4) 其中K是峰度(excess + 3)

        Returns
        -------
        (jb_stat, p_value)
        """
        jb_stat, p_value = stats.jarque_bera(self.returns)
        return float(jb_stat), float(p_value)

    def shapiro_wilk(self) -> Tuple[float, float]:
        """
        Shapiro-Wilk 检验

        H0: 数据服从正态分布
        H1: 数据不服从正态分布

        注意: SW检验推荐样本量 3 ≤ n ≤ 5000

        Returns
        -------
        (sw_stat, p_value)
        """
        sw_stat, p_value = stats.shapiro(self.returns)
        return float(sw_stat), float(p_value)

    def is_normal(self, alpha: float = 0.05) -> bool:
        """
        综合判断收益率是否服从正态分布 (JB + SW 均为显著性).

        判断逻辑: 任一检验拒绝 → 认为非正态

        Parameters
        ----------
        alpha : float
            显著性水平, 默认 0.05

        Returns
        -------
        bool
        """
        _, jb_p = self.jarque_bera()
        _, sw_p = self.shapiro_wilk()
        return (jb_p >= alpha) and (sw_p >= alpha)

    def is_lognormal(self, alpha: float = 0.05) -> bool:
        """
        判断收益率是否服从对数正态分布。

        对 log(1 + returns) 做正态性检验。

        Parameters
        ----------
        alpha : float
            显著性水平, 默认 0.05

        Returns
        -------
        bool
        """
        # 对数收益率: log(1+r), 需保证 1+r > 0
        if np.any(self.returns <= -1.0):
            # 有回报 ≤ -100%, 对数正态无意义
            return False

        log_rets = np.log(1 + self.returns)
        # 检查是否常数
        if np.allclose(log_rets, log_rets[0]):
            return False

        _, jb_p = stats.jarque_bera(log_rets)
        _, sw_p = stats.shapiro(log_rets)

        return (jb_p >= alpha) and (sw_p >= alpha)

    def report(self, alpha: float = 0.05) -> NormalityReport:
        """
        生成完整的正态性诊断报告。

        Parameters
        ----------
        alpha : float
            显著性水平

        Returns
        -------
        NormalityReport
        """
        jb_stat, jb_p = self.jarque_bera()
        sw_stat, sw_p = self.shapiro_wilk()

        is_n = self.is_normal(alpha)
        is_ln = self.is_lognormal(alpha)

        # 对数正态检验细节
        log_sw_stat, log_sw_p = None, None
        log_jb_stat, log_jb_p = None, None
        if not np.any(self.returns <= -1.0):
            log_rets = np.log(1 + self.returns)
            if not np.allclose(log_rets, log_rets[0]):
                log_jb_stat, log_jb_p = stats.jarque_bera(log_rets)
                log_sw_stat, log_sw_p = stats.shapiro(log_rets)

        # 构建解读
        lines = [f"=== 正态性诊断报告 (n={self.n}) ==="]
        lines.append(f"偏度: {self._skewness:.4f}  (标准正态=0)")
        lines.append(f"超额峰度: {self._kurtosis:.4f}  (标准正态=0)")

        if self._kurtosis > 0.5:
            lines.append("  ⚠️  峰度 > 0.5 → 肥尾分布, 极端事件概率高于正态")
        if abs(self._skewness) > 0.5:
            lines.append(f"  ⚠️  偏度偏差 > 0.5 → 分布不对称")

        lines.append(f"\nJarque-Bera: stat={jb_stat:.4f}, p={jb_p:.4f}")
        lines.append(f"Shapiro-Wilk: stat={sw_stat:.4f}, p={sw_p:.4f}")

        if is_n:
            lines.append(f"\n✅ 不能拒绝正态假设 (α={alpha})")
        else:
            lines.append(f"\n❌ 拒绝正态假设 (α={alpha})")

        if is_ln:
            lines.append(f"✅ 不能拒绝对数正态假设 (α={alpha})")
        else:
            lines.append(f"❌ 拒绝对数正态假设 (α={alpha})")

        if self._kurtosis > 1.0:
            lines.append("\n建议: 肥尾分布应考虑 VaR/CVaR 或极值理论 (EVT).")
        if abs(self._skewness) > 1.0:
            lines.append("建议: 显著偏度应使用非对称风险度量.")

        return NormalityReport(
            n_samples=self.n,
            skewness=self._skewness,
            kurtosis=self._kurtosis,
            jb_stat=jb_stat,
            jb_pvalue=jb_p,
            sw_stat=sw_stat,
            sw_pvalue=sw_p,
            is_normal=is_n,
            is_lognormal=is_ln,
            log_sw_stat=log_sw_stat,
            log_sw_pvalue=log_sw_p,
            log_jb_stat=log_jb_stat,
            log_jb_pvalue=log_jb_p,
            interpretation="\n".join(lines),
        )


# ── Convenience functions ───────────────────────────────────────────

def quick_normality_check(returns: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """便捷函数: 快速正态性检查"""
    tester = NormalityTester(returns)
    return {
        "is_normal": tester.is_normal(alpha),
        "is_lognormal": tester.is_lognormal(alpha),
        "skewness": tester.skewness,
        "kurtosis": tester.kurtosis,
    }
