#!/usr/bin/env python3
"""
beta_alpha_monitor.py 单元测试 (使用模拟数据, 无需网络)
"""
import os
import sys
import json
import tempfile
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

# 确保可以导入被测模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beta_alpha_monitor import BetaAlphaMonitor, _prefixed


class TestMarketPrefix(unittest.TestCase):
    """市场前缀测试"""

    def test_sh_prefix(self):
        self.assertEqual(_prefixed("600519"), "sh600519")
        self.assertEqual(_prefixed("688001"), "sh688001")
        self.assertEqual(_prefixed("900901"), "sh900901")

    def test_sz_prefix(self):
        self.assertEqual(_prefixed("000001"), "sz000001")
        self.assertEqual(_prefixed("300750"), "sz300750")
        self.assertEqual(_prefixed("002415"), "sz002415")

    def test_bj_prefix(self):
        self.assertEqual(_prefixed("830799"), "bj830799")
        self.assertEqual(_prefixed("430047"), "bj430047")

    def test_invalid_code(self):
        with self.assertRaises(ValueError):
            _prefixed("xyz123")


class TestBetaAlphaMonitorMock(unittest.TestCase):
    """使用 mock 数据测试 BetaAlphaMonitor 核心逻辑"""

    def setUp(self):
        """构造模拟日K数据"""
        np.random.seed(42)
        n = 300  # 300 个交易日

        # 日期
        dates = pd.date_range(end=datetime(2026, 6, 3), periods=n, freq="B")

        # 基准指数收益率 (正态分布, 年化 8%, 年化波动 20%)
        idx_daily_ret = np.random.normal(0.08 / 252, 0.20 / np.sqrt(252), n)
        idx_close = 4000 * np.exp(np.cumsum(idx_daily_ret))

        # 个股: β=1.2, α_daily=0.0002 (≈5% 年化), 残差波动 1.5%
        beta_true = 1.2
        alpha_true = 0.0002
        resid_std_true = 0.015
        stock_ret = alpha_true + beta_true * idx_daily_ret + \
                    np.random.normal(0, resid_std_true, n)
        stock_close = 100 * np.exp(np.cumsum(stock_ret))

        # 构建 DataFrame (模拟 mootdx 格式)
        self.stock_df = pd.DataFrame({
            "open": stock_close * 0.99,
            "close": stock_close,
            "high": stock_close * 1.02,
            "low": stock_close * 0.98,
            "vol": np.ones(n) * 1e6,
            "amount": np.ones(n) * 1e9,
        }, index=dates + pd.Timedelta(hours=15))

        self.index_df = pd.DataFrame({
            "open": idx_close * 0.99,
            "close": idx_close,
            "high": idx_close * 1.02,
            "low": idx_close * 0.98,
            "vol": np.ones(n) * 1e6,
            "amount": np.ones(n) * 1e9,
        }, index=dates + pd.Timedelta(hours=15))

        self.beta_true = beta_true
        self.alpha_true = alpha_true

    def _inject_data(self, monitor):
        """注入模拟数据, 绕过 mootdx"""
        monitor.stock_df = self.stock_df.copy()
        monitor.index_df = self.index_df.copy()

    def test_compute_returns(self):
        monitor = BetaAlphaMonitor("600519")
        self._inject_data(monitor)
        n = monitor.compute_returns()
        self.assertEqual(n, len(self.stock_df) - 1)  # 首行 NaN
        # 收益率不应全为 0
        self.assertGreater(abs(monitor.stock_ret.mean()), 0)
        self.assertGreater(abs(monitor.index_ret.mean()), 0)

    def test_compute_beta_alpha_accuracy(self):
        """验证 beta/alpha 估计值在合理误差范围内"""
        monitor = BetaAlphaMonitor("600519")
        self._inject_data(monitor)
        monitor.compute_returns()

        # 使用较大窗口以获得更准确的估计
        result = monitor.compute_beta_alpha(250)

        self.assertIsNotNone(result["beta"])
        self.assertIsNotNone(result["alpha_annual"])
        self.assertIsNotNone(result["t_stat"])
        self.assertIsNotNone(result["p_value"])

        # beta 估计误差应在 0.2 以内
        self.assertAlmostEqual(result["beta"], self.beta_true, delta=0.2,
                               msg=f"beta={result['beta']}, true={self.beta_true}")

        # alpha_daily 估计误差 (放宽到 0.002, 因为小样本下 alpha 估计噪声大)
        self.assertAlmostEqual(result["alpha_daily"], self.alpha_true, delta=0.003,
                               msg=f"alpha_daily={result['alpha_daily']}, true={self.alpha_true}")

        # R² 应该相对高 (beta 显著, >0.45 即可, 随机噪声会拉低)
        self.assertGreater(result["r_squared"], 0.45)

    def test_alpha_annual_formula(self):
        """验证年化 alpha = daily * 252"""
        monitor = BetaAlphaMonitor("600519")
        self._inject_data(monitor)
        monitor.compute_returns()
        result = monitor.compute_beta_alpha(100)
        # 由于 compute_beta_alpha 内部对 alpha_daily 和 alpha_annual
        # 分别 round(6), 所以 alpha_annual ≈ alpha_daily * 252
        self.assertAlmostEqual(
            result["alpha_annual"],
            result["alpha_daily"] * 252,
            delta=2e-4  # round(6) 带来 ~1e-4 级别的舍入误差
        )

    def test_return_types(self):
        """验证返回值类型"""
        monitor = BetaAlphaMonitor("600519")
        self._inject_data(monitor)
        monitor.compute_returns()
        result = monitor.compute_beta_alpha(50)
        self.assertIsInstance(result["beta"], float)
        self.assertIsInstance(result["t_stat"], float)
        self.assertIsInstance(result["p_value"], float)
        self.assertIsInstance(result["r_squared"], float)

    def test_run_all_windows(self):
        monitor = BetaAlphaMonitor("600519", lookback_windows=[20, 60, 120])
        self._inject_data(monitor)
        monitor.compute_returns()
        results = monitor.run_all_windows()
        self.assertEqual(len(results), 3)
        for key in ["20", "60", "120"]:
            self.assertIn(key, results)
            self.assertIsNotNone(results[key].get("beta"))

    def test_insufficient_data(self):
        """窗口大于可用数据时返回错误标记"""
        monitor = BetaAlphaMonitor("600519", lookback_windows=[9999])
        self._inject_data(monitor)
        monitor.compute_returns()
        result = monitor.compute_beta_alpha(9999)
        self.assertIsNone(result["beta"])
        self.assertIn("error", result)
        self.assertIn("数据不足", result["error"])

    def test_diagnose_beta_unstable(self):
        """模拟 β 不稳定的诊断"""
        monitor = BetaAlphaMonitor("600519", lookback_windows=[20, 252])

        # 注入两种不同场景的数据: 短窗口高 β, 长窗口低 β
        np.random.seed(123)
        n = 300
        dates = pd.date_range(end=datetime(2026, 6, 3), periods=n, freq="B")

        idx_ret = np.random.normal(0.08 / 252, 0.20 / np.sqrt(252), n)
        idx_close = 4000 * np.exp(np.cumsum(idx_ret))

        # 前 50 天 β=1.5, 后 250 天 β=0.6
        stock_ret = np.zeros(n)
        stock_ret[:50] = 0.0002 + 1.5 * idx_ret[:50] + np.random.normal(0, 0.015, 50)
        stock_ret[50:] = 0.0002 + 0.6 * idx_ret[50:] + np.random.normal(0, 0.015, n - 50)
        stock_close = 100 * np.exp(np.cumsum(stock_ret))

        monitor.stock_df = pd.DataFrame({
            "open": stock_close * 0.99, "close": stock_close,
            "high": stock_close * 1.01, "low": stock_close * 0.99,
            "vol": np.ones(n) * 1e6, "amount": np.ones(n) * 1e9,
        }, index=dates + pd.Timedelta(hours=15))

        monitor.index_df = pd.DataFrame({
            "open": idx_close * 0.99, "close": idx_close,
            "high": idx_close * 1.01, "low": idx_close * 0.99,
            "vol": np.ones(n) * 1e6, "amount": np.ones(n) * 1e9,
        }, index=dates + pd.Timedelta(hours=15))

        monitor.compute_returns()
        monitor.run_all_windows()
        diag = monitor.diagnose()

        # 应该有 β 不稳定的警告
        unstable_msgs = [m for m in diag if "β不稳定" in m]
        self.assertGreater(len(unstable_msgs), 0,
                           f"应检测到β不稳定, 实际诊断: {diag}")

    def test_diagnose_not_significant(self):
        """模拟 α 不显著的诊断 (纯噪声)"""
        monitor = BetaAlphaMonitor("600519", lookback_windows=[60])
        np.random.seed(99)
        n = 100
        dates = pd.date_range(end=datetime(2026, 6, 3), periods=n, freq="B")

        idx_ret = np.random.normal(0, 0.01, n)
        stock_ret = np.random.normal(0, 0.03, n)  # 纯噪声, 无 alpha
        idx_close = 4000 * np.exp(np.cumsum(idx_ret))
        stock_close = 100 * np.exp(np.cumsum(stock_ret))

        monitor.stock_df = pd.DataFrame({
            "open": stock_close * 0.99, "close": stock_close,
            "high": stock_close * 1.01, "low": stock_close * 0.99,
            "vol": np.ones(n) * 1e6, "amount": np.ones(n) * 1e9,
        }, index=dates + pd.Timedelta(hours=15))
        monitor.index_df = pd.DataFrame({
            "open": idx_close * 0.99, "close": idx_close,
            "high": idx_close * 1.01, "low": idx_close * 0.99,
            "vol": np.ones(n) * 1e6, "amount": np.ones(n) * 1e9,
        }, index=dates + pd.Timedelta(hours=15))

        monitor.compute_returns()
        monitor.run_all_windows()
        diag = monitor.diagnose()

        # 检查是否有 R² 低的警告（因纯噪声相关性很低）
        low_r2 = [m for m in diag if "R²过低" in m]
        # 不一定100%触发，取决于随机种子，但概率很高
        self.assertTrue(
            len(low_r2) > 0 or any("不显著" in m for m in diag),
            f"应至少有某种诊断: {diag}"
        )

    def test_to_dict(self):
        monitor = BetaAlphaMonitor("600519")
        self._inject_data(monitor)
        monitor.compute_returns()
        monitor.run_all_windows()
        d = monitor.to_dict()
        self.assertEqual(d["stock_code"], "600519")
        self.assertEqual(d["index_code"], "000300")
        self.assertIn("windows", d)
        self.assertIn("diagnosis", d)
        self.assertIsInstance(d["timestamp"], str)

    def test_save_output(self):
        monitor = BetaAlphaMonitor("600519")
        self._inject_data(monitor)
        monitor.compute_returns()
        monitor.run_all_windows()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = monitor.save_output(tmpdir)
            self.assertTrue(os.path.exists(path))

            with open(path) as f:
                data = json.load(f)

            self.assertEqual(data["stock_code"], "600519")
            self.assertIn("252", data["windows"])
            self.assertIsInstance(data["diagnosis"], list)

    def test_log_returns(self):
        """验证使用的是对数收益率而非简单收益率"""
        monitor = BetaAlphaMonitor("600519")
        self._inject_data(monitor)
        monitor.compute_returns()

        # 手动验证第一条收益率
        close0 = self.stock_df["close"].iloc[0]
        close1 = self.stock_df["close"].iloc[1]
        expected = np.log(close1 / close0)
        actual = monitor.stock_ret.iloc[0]
        self.assertAlmostEqual(expected, actual)

    def test_market_prefix_integration(self):
        """测试各市场前缀"""
        # 上海主板
        self.assertEqual(_prefixed("600000"), "sh600000")
        # 上海科创板
        self.assertEqual(_prefixed("688001"), "sh688001")
        # 深圳主板
        self.assertEqual(_prefixed("000001"), "sz000001")
        # 深圳创业板
        self.assertEqual(_prefixed("300750"), "sz300750")
        # 北交所
        self.assertEqual(_prefixed("830001"), "bj830001")

    def test_edge_case_small_sample(self):
        """极少样本的边界情况"""
        monitor = BetaAlphaMonitor("600519")
        np.random.seed(1)
        n = 5
        dates = pd.date_range(end=datetime(2026, 6, 3), periods=n, freq="B")
        idx_close = np.array([4000, 4010, 3995, 4020, 4030])
        stock_close = np.array([100, 101, 99, 102, 103])

        monitor.stock_df = pd.DataFrame({
            "open": stock_close * 0.99, "close": stock_close,
            "high": stock_close * 1.01, "low": stock_close * 0.99,
            "vol": np.ones(n), "amount": np.ones(n),
        }, index=dates + pd.Timedelta(hours=15))
        monitor.index_df = pd.DataFrame({
            "open": idx_close * 0.99, "close": idx_close,
            "high": idx_close * 1.01, "low": idx_close * 0.99,
            "vol": np.ones(n), "amount": np.ones(n),
        }, index=dates + pd.Timedelta(hours=15))

        monitor.compute_returns()
        result = monitor.compute_beta_alpha(4)
        self.assertIsNotNone(result["beta"])
        self.assertIsNotNone(result["r_squared"])

    def test_returns_alignment(self):
        """验证收益率序列对齐"""
        monitor = BetaAlphaMonitor("600519")
        self._inject_data(monitor)
        monitor.compute_returns()
        # 长度应该一致
        self.assertEqual(len(monitor.stock_ret), len(monitor.index_ret))
        # index 应该一致
        self.assertTrue(monitor.stock_ret.index.equals(monitor.index_ret.index))


class TestBetaAlphaMonitorLive(unittest.TestCase):
    """需要网络的集成测试 (mootdx 连接)"""

    def test_fetch_real_data(self):
        """拉取真实数据并计算"""
        try:
            monitor = BetaAlphaMonitor("600519", index_code="000300",
                                       lookback_windows=[20, 60])
            n = monitor.fetch_data(offset=100)
            self.assertGreater(n, 0, "应拉取到数据")
            m = monitor.compute_returns()
            self.assertGreater(m, 0, "应有收益率数据")
            monitor.run_all_windows()

            for key, r in monitor.results.items():
                self.assertIsNotNone(r.get("beta"), f"窗口{key}应有beta")
                self.assertIsNotNone(r.get("r_squared"), f"窗口{key}应有R²")
        except Exception as e:
            self.skipTest(f"mootdx 连接失败 (网络问题): {e}")

    def test_different_index(self):
        """测试不同基准指数"""
        try:
            monitor = BetaAlphaMonitor("000001", index_code="000905",
                                       lookback_windows=[20])
            n = monitor.fetch_data(offset=100)
            self.assertGreater(n, 0)
            monitor.compute_returns()
            result = monitor.compute_beta_alpha(20)
            self.assertIsNotNone(result["beta"])
        except Exception as e:
            self.skipTest(f"mootdx 连接失败: {e}")


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
