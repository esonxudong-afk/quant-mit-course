#!/usr/bin/env python3
"""
Beta/Alpha Monitor — A股个股相对于基准指数的 Beta & Alpha 计算工具

用法:
    python beta_alpha_monitor.py 600519
    python beta_alpha_monitor.py 600519 --index 000905 --output ./output

数据源:
    mootdx (标准行情), 日K线 (frequency=9)

方法:
    OLS 回归: r_stock = α + β · r_index + ε
    收益率: r = ln(close_t / close_{t-1})

输出: JSON 包含各窗口 β, α, t-stat, p-value, R², 残差标准差, 残差自相关 + 诊断信息
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
from scipy import stats

# ── 市场前缀工具 ──────────────────────────────────────────────
_MARKET_PREFIX = {
    "6": "sh",   # 上海主板
    "0": "sz",   # 深圳主板
    "3": "sz",   # 深圳创业板
    "4": "bj",   # 北京交易所
    "8": "bj",   # 北京交易所
    "9": "sh",   # 上海
}


def _prefixed(code: str) -> str:
    """为股票代码添加市场前缀 (mootdx 个股需要前缀, 指数不需要)"""
    first = code[0]
    prefix = _MARKET_PREFIX.get(first)
    if prefix is None:
        raise ValueError(f"无法识别股票代码市场前缀: {code} (首位={first})")
    return f"{prefix}{code}"


# ── 主类 ─────────────────────────────────────────────────────


class BetaAlphaMonitor:
    """A股个股 Beta/Alpha 监控器

    从 mootdx 拉取日K线数据，计算个股相对于基准指数的 Beta & Alpha，
    支持多窗口 rolling OLS 回归 + 诊断。

    Parameters
    ----------
    stock_code : str
        个股代码 (不含市场前缀, 如 "600519")
    index_code : str
        基准指数代码 (不含市场前缀, 如 "000300" 沪深300)
    lookback_windows : list[int]
        回归窗口 (交易日天数), 默认 [20, 60, 120, 252]
    """

    def __init__(self, stock_code, index_code="000300",
                 lookback_windows=None):
        self.stock_code = stock_code
        self.index_code = index_code
        self.lookback_windows = lookback_windows or [20, 60, 120, 252]

        # 数据容器
        self.stock_df = None   # 个股日K DataFrame
        self.index_df = None   # 指数日K DataFrame
        self.stock_ret = None  # 个股对数收益率 Series (index: datetime)
        self.index_ret = None  # 指数对数收益率 Series

        # 结果
        self.results = {}      # {str(window): dict}

    # ── 数据获取 ───────────────────────────────────────────

    def fetch_data(self, offset=400):
        """通过 mootdx 获取个股+基准指数日K线

        Parameters
        ----------
        offset : int
            取多少条日K线记录 (默认 400, 约覆盖 1.5 年)
        """
        from mootdx.quotes import Quotes

        client = Quotes.factory(market="std")

        # 个股 — mootdx std market 的 bars() 不需要市场前缀
        self.stock_df = client.bars(symbol=self.stock_code, frequency=9,
                                    start=0, offset=offset)

        # 指数 (用 index_bars, 不需要前缀)
        self.index_df = client.index_bars(symbol=self.index_code,
                                          frequency=9,
                                          start=0, offset=offset)

        if self.stock_df is None or len(self.stock_df) == 0:
            raise RuntimeError(f"个股 {self.stock_code} 无数据")
        if self.index_df is None or len(self.index_df) == 0:
            raise RuntimeError(f"指数 {self.index_code} 无数据")

        # 按日期对齐 (取交集)
        common_idx = self.stock_df.index.intersection(self.index_df.index)
        self.stock_df = self.stock_df.loc[common_idx].sort_index()
        self.index_df = self.index_df.loc[common_idx].sort_index()

        return len(self.stock_df)

    # ── 收益率计算 ─────────────────────────────────────────

    def compute_returns(self):
        """计算对数收益率: r_t = ln(close_t / close_{t-1})"""
        if self.stock_df is None or self.index_df is None:
            raise RuntimeError("请先调用 fetch_data() 获取数据")

        self.stock_ret = np.log(
            self.stock_df["close"] / self.stock_df["close"].shift(1)
        ).dropna()
        self.index_ret = np.log(
            self.index_df["close"] / self.index_df["close"].shift(1)
        ).dropna()

        # 再次对齐
        common_idx = self.stock_ret.index.intersection(self.index_ret.index)
        self.stock_ret = self.stock_ret.loc[common_idx]
        self.index_ret = self.index_ret.loc[common_idx]

        return len(self.stock_ret)

    # ── 单窗口 OLS ─────────────────────────────────────────

    def compute_beta_alpha(self, window_days):
        """对最近 window_days 个交易日做 OLS 回归

        Parameters
        ----------
        window_days : int
            窗口大小 (交易日数)

        Returns
        -------
        dict 包含:
            beta, alpha_daily, alpha_annual,
            t_stat, p_value, r_squared,
            resid_std, resid_autocorr
        """
        if self.stock_ret is None or self.index_ret is None:
            raise RuntimeError("请先调用 compute_returns()")

        n_avail = len(self.stock_ret)
        if window_days > n_avail:
            return self._empty_result(window_days,
                                      reason=f"数据不足 (需要{window_days}, 可用{n_avail})")

        # 取最近 window_days 个交易日
        y = self.stock_ret.iloc[-window_days:].values  # 因变量
        X = self.index_ret.iloc[-window_days:].values  # 自变量

        # 去除 NaN / Inf
        mask = np.isfinite(y) & np.isfinite(X)
        y, X = y[mask], X[mask]
        n = len(y)

        if n < 3:
            return self._empty_result(window_days,
                                      reason=f"有效样本不足 (n={n})")

        # OLS: y = α + β·X
        X_design = np.column_stack([np.ones(n), X])  # [1, X]
        try:
            coeffs, residuals, rank, singular = np.linalg.lstsq(
                X_design, y, rcond=None
            )
        except np.linalg.LinAlgError:
            return self._empty_result(window_days, reason="linalg error")

        alpha_daily = float(coeffs[0])
        beta = float(coeffs[1])

        # 残差
        y_pred = X_design @ coeffs
        resid = y - y_pred

        # 残差标准差
        resid_std = float(np.std(resid, ddof=2))  # 无偏估计

        # R²
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # 标准误 & t 统计量 (H0: α=0)
        # SE(α) = σ · sqrt( (1/n) + (mean(X)² / Σ(X_i - mean(X))²) )
        mse = ss_res / (n - 2) if n > 2 else 0.0
        X_mean = np.mean(X)
        X_demean_sq = np.sum((X - X_mean) ** 2)
        if X_demean_sq > 0 and mse > 0:
            se_alpha = np.sqrt(mse * (1.0 / n + X_mean ** 2 / X_demean_sq))
            t_stat = float(alpha_daily / se_alpha)
            # 双尾 p-value, df = n-2
            p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 2))
        else:
            t_stat = 0.0
            p_value = 1.0

        # 残差自相关 (lag-1)
        if len(resid) >= 3:
            resid_autocorr = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
            if not np.isfinite(resid_autocorr):
                resid_autocorr = 0.0
        else:
            resid_autocorr = 0.0

        # 年化 alpha
        alpha_annual = float(alpha_daily * 252)

        return {
            "window": window_days,
            "n_samples": n,
            "beta": round(beta, 6),
            "alpha_daily": alpha_daily,
            "alpha_annual": alpha_annual,
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "r_squared": round(r_squared, 6),
            "resid_std": round(resid_std, 6),
            "resid_autocorr": round(resid_autocorr, 6),
        }

    def _empty_result(self, window_days, reason=""):
        return {
            "window": window_days,
            "n_samples": 0,
            "beta": None,
            "alpha_daily": None,
            "alpha_annual": None,
            "t_stat": None,
            "p_value": None,
            "r_squared": None,
            "resid_std": None,
            "resid_autocorr": None,
            "error": reason,
        }

    # ── 全窗口计算 ─────────────────────────────────────────

    def run_all_windows(self):
        """对所有预设窗口计算 Beta/Alpha"""
        for w in self.lookback_windows:
            self.results[str(w)] = self.compute_beta_alpha(w)
        return self.results

    # ── 诊断 ───────────────────────────────────────────────

    def diagnose(self):
        """基于多窗口结果生成诊断信息

        Returns
        -------
        list[str]  诊断字符串列表
        """
        diagnosis = []
        warnings_list = []

        # 辅助: 获取某窗口结果
        def _w(days):
            return self.results.get(str(days))

        # 1. β 不稳定性: 20d vs 252d
        r20 = _w(20)
        r252 = _w(252)
        if r20 and r252 and r20.get("beta") is not None and r252.get("beta") is not None:
            diff = abs(r20["beta"] - r252["beta"])
            if diff > 0.3:
                msg = (f"WARN: β不稳定(20d={r20['beta']:.4f}, "
                       f"252d={r252['beta']:.4f}, diff={diff:.4f})")
                diagnosis.append(msg)

        # 2. α 不显著: 对每个窗口检查 p > 0.05
        for key, r in self.results.items():
            if r.get("p_value") is not None and r["p_value"] > 0.05:
                msg = f"WARN: α不显著(p={r['p_value']:.4f}, {key}d窗口)"
                diagnosis.append(msg)

        # 3. R² 低
        for key, r in self.results.items():
            if r.get("r_squared") is not None and r["r_squared"] < 0.1:
                msg = (f"WARN: R²过低(R²={r['r_squared']:.4f}, "
                       f"{key}d窗口) → β分析可能不适用")
                diagnosis.append(msg)

        # 4. 残差自相关高
        for key, r in self.results.items():
            if r.get("resid_autocorr") is not None:
                ac = abs(r["resid_autocorr"])
                if ac > 0.2:
                    msg = (f"WARN: 残差自相关过高(|ρ|={ac:.4f}, "
                           f"{key}d窗口) → 模型遗漏因子")
                    diagnosis.append(msg)

        if not diagnosis:
            diagnosis.append("OK: 未检测到明显异常")

        return diagnosis

    # ── 输出 ───────────────────────────────────────────────

    def save_output(self, output_dir):
        """将结果输出为 JSON 文件到指定目录

        Parameters
        ----------
        output_dir : str
            输出目录路径

        Returns
        -------
        str  输出文件路径
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now(timezone.utc).isoformat()
        diag = self.diagnose()

        output = {
            "stock_code": self.stock_code,
            "index_code": self.index_code,
            "timestamp": timestamp,
            "windows": self.results,
            "diagnosis": diag,
        }

        filename = f"beta_alpha_{self.stock_code}_{self.index_code}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return filepath

    def to_dict(self):
        """返回结果的字典表示"""
        return {
            "stock_code": self.stock_code,
            "index_code": self.index_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "windows": self.results,
            "diagnosis": self.diagnose(),
        }


# ── CLI ─────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="A股 Beta/Alpha 监控器 — "
                    "计算个股相对基准指数的 Beta 和 Alpha"
    )
    parser.add_argument("stock", help="个股代码, 如 600519")
    parser.add_argument("--index", default="000300",
                        help="基准指数代码, 默认 000300 (沪深300)")
    parser.add_argument("--output", "-o", default="./output",
                        help="输出目录, 默认 ./output")
    parser.add_argument("--offset", type=int, default=400,
                        help="拉取日K条数, 默认 400")
    parser.add_argument("--windows", "-w", nargs="+", type=int,
                        default=[20, 60, 120, 252],
                        help="回归窗口 (交易日数), 默认 20 60 120 252")
    args = parser.parse_args()

    monitor = BetaAlphaMonitor(
        stock_code=args.stock,
        index_code=args.index,
        lookback_windows=args.windows,
    )

    try:
        print(f"📡 获取数据: {args.stock} vs {args.index} ...")
        n = monitor.fetch_data(offset=args.offset)
        print(f"   共 {n} 条对齐后的日K记录")

        print("📈 计算对数收益率 ...")
        m = monitor.compute_returns()
        print(f"   共 {m} 个有效交易日收益率")

        print("📊 计算 Beta/Alpha ...")
        monitor.run_all_windows()

        # 打印结果
        print()
        print(f"{'窗口':>6s}  {'N':>5s}  {'Beta':>8s}  {'α_annual':>10s}  "
              f"{'t-stat':>8s}  {'p-value':>8s}  {'R²':>8s}  {'resid_ac':>9s}")
        print("-" * 85)
        for key in sorted(monitor.results.keys(), key=int):
            r = monitor.results[key]
            if r.get("error"):
                print(f"{key:>6s}  {r['error']}")
                continue
            print(f"{key:>6s}  {r['n_samples']:>5d}  "
                  f"{r['beta']:>8.4f}  {r['alpha_annual']:>10.4f}  "
                  f"{r['t_stat']:>8.4f}  {r['p_value']:>8.4f}  "
                  f"{r['r_squared']:>8.4f}  {r['resid_autocorr']:>9.4f}")

        # 诊断
        print()
        print("🔍 诊断:")
        for d in monitor.diagnose():
            print(f"   {d}")

        # 保存
        if args.output:
            path = monitor.save_output(args.output)
            print(f"\n💾 已保存到: {path}")

    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
