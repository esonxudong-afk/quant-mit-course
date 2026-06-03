"""
Historical CVaR (Conditional Value at Risk) Calculator
=======================================================
Replaces the naive normal ±2σ stop-loss from the first chapter.
Crypto returns exhibit kurtosis ≈ 6, so the normal assumption
under-estimates tail risk by 2-3x.  This module implements:

  - Historical VaR (empirical quantile)
  - Historical CVaR (Expected Shortfall — mean of worst α returns)
  - Normal-theory VaR / CVaR (for comparison)
  - Tail-risk report: expected shortfall, underestimation ratio

Usage:
    cvar = CVaRCalculator(returns)
    cvar.var(0.05)              # 5% VaR → loss threshold
    cvar.cvar(0.05)             # 5% CVaR → average loss beyond VaR
    cvar.tail_risk_report()     # full comparison report
"""

import numpy as np

# Optional scipy for normal quantile
try:
    from scipy.stats import norm as scipy_norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class CVaRCalculator:
    """Historical and normal-theory VaR / CVaR for a return series.

    All VaR/CVaR values are in the same units as the input returns
    (typically daily log or simple returns).  Positive = gain, so
    VaR at α=0.05 is the *5th percentile* — a negative number
    representing a loss.  The report inverts signs for readability.

    Attributes:
        returns: 1-D numpy array of returns.
        n_obs: Number of observations.
    """

    def __init__(self, returns: np.ndarray):
        """Initialise the calculator.

        Args:
            returns: 1-D array of return observations.
                Day-aligned, can be log or simple returns.

        Raises:
            ValueError: If returns is empty or has < 20 observations
                (insufficient for meaningful tail estimation).
        """
        returns = np.asarray(returns, dtype=float).ravel()
        if returns.size < 20:
            raise ValueError(f"Need at least 20 observations for tail estimation, "
                             f"got {returns.size}")
        self.returns = returns
        self.n_obs = returns.size
        self._sorted = None

    @property
    def _sorted_returns(self):
        """Cached sorted returns (ascending)."""
        if self._sorted is None:
            self._sorted = np.sort(self.returns)
        return self._sorted

    # ── Historical (non-parametric)  ──────────────────────────────

    def var(self, alpha: float = 0.05) -> float:
        """Historical Value at Risk.

        The α-quantile of the return distribution.  If α=0.05,
        returns the 5th percentile — 95% confidence that losses
        will not exceed this threshold.

        Uses linear interpolation between order statistics (R-7
        method, same as numpy default).

        Args:
            alpha: Tail probability (0 < alpha < 0.5). Default 0.05.

        Returns:
            float: VaR value (may be negative = loss).
        """
        if not 0 < alpha < 0.5:
            raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
        return float(np.percentile(self.returns, 100 * alpha))

    def cvar(self, alpha: float = 0.05) -> float:
        """Historical Conditional VaR (Expected Shortfall).

        Average return across the worst α × 100% of observations.
        If returns are daily, this is the expected loss *given* that
        the loss exceeds the VaR threshold.

        Args:
            alpha: Tail probability. Default 0.05.

        Returns:
            float: CVaR (average of returns ≤ VaR at α).
        """
        if not 0 < alpha < 0.5:
            raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")

        var_val = self.var(alpha)
        # All returns at or below the VaR threshold
        tail = self.returns[self.returns <= var_val + 1e-12]

        if len(tail) == 0:
            # Degenerate case — fall back to a fraction of sorted
            n_tail = max(1, int(np.ceil(self.n_obs * alpha)))
            tail = self._sorted_returns[:n_tail]

        return float(np.mean(tail))

    # ── Normal-theory (parametric) ────────────────────────────────

    def normal_var(self, alpha: float = 0.05) -> float:
        """Parametric VaR assuming normality.

        VaR_N = μ̂ + z_α × σ̂  (z_α is the α-quantile of standard normal)

        Args:
            alpha: Tail probability.

        Returns:
            float: Normal-theory VaR.
        """
        if not 0 < alpha < 0.5:
            raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")

        mean = np.mean(self.returns)
        std = np.std(self.returns, ddof=1)

        if _HAS_SCIPY:
            z = scipy_norm.ppf(alpha)
        else:
            # Abramowitz & Stegun approximation for N^{-1}(p)
            z = _approx_norm_ppf(alpha)

        return float(mean + z * std)

    def normal_cvar(self, alpha: float = 0.05) -> float:
        """Parametric CVaR (Expected Shortfall) under normality.

        ES_N = μ̂ - σ̂ × φ(z_α) / α      where φ is the standard normal PDF.

        Args:
            alpha: Tail probability.

        Returns:
            float: Normal-theory CVaR (Expected Shortfall).
        """
        if not 0 < alpha < 0.5:
            raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")

        mean = np.mean(self.returns)
        std = np.std(self.returns, ddof=1)

        if _HAS_SCIPY:
            z = scipy_norm.ppf(alpha)
            pdf_z = scipy_norm.pdf(z)
        else:
            z = _approx_norm_ppf(alpha)
            pdf_z = _approx_norm_pdf(z)

        # hVaR = History VaR; normal CVaR = μ - σ * φ(z) / α
        es = mean - std * pdf_z / alpha
        return float(es)

    # ── Report ────────────────────────────────────────────────────

    def tail_risk_report(self, alpha: float = 0.05) -> dict:
        """Full tail-risk diagnostic report.

        Computes historical VaR and CVaR, compares with normal-theory
        counterparts, and quantifies the underestimation ratio.

        Args:
            alpha: Tail probability.

        Returns:
            dict with keys:
                - alpha
                - n_observations
                - historical_var, historical_cvar
                - normal_var, normal_cvar
                - var_underestimation_ratio: hist_var / normal_var
                - cvar_underestimation_ratio: hist_cvar / normal_cvar
                - kurtosis: sample excess kurtosis
                - tail_warning: True if historical CVaR exceeds normal
                  CVaR by > 20% (suggesting fat tails)
        """
        h_var = self.var(alpha)
        h_cvar = self.cvar(alpha)
        n_var = self.normal_var(alpha)
        n_cvar = self.normal_cvar(alpha)

        # Underestimation ratio: how many times the historical (true)
        # tail exceeds the normal-theory estimate.
        # For losses, both are negative; ratio > 1 means history is worse.
        var_ratio = abs(h_var / n_var) if n_var != 0 else float("inf")
        cvar_ratio = abs(h_cvar / n_cvar) if n_cvar != 0 else float("inf")

        # Excess kurtosis (normal = 0)
        ret = self.returns
        mean_ret = np.mean(ret)
        m2 = np.sum((ret - mean_ret) ** 2) / self.n_obs
        m4 = np.sum((ret - mean_ret) ** 4) / self.n_obs
        kurt = m4 / (m2 ** 2) - 3 if m2 > 0 else 0.0

        # Warning: fat tails if history CVaR exceeds normal CVaR by > 10%
        # (For losses, "exceed" means more negative → ratio > 1.1)
        # A 10% threshold is more sensitive than 20%; crypto often sees 2-3x.
        tail_warning = cvar_ratio > 1.1

        return {
            "alpha": alpha,
            "n_observations": self.n_obs,
            "historical_var": float(h_var),
            "historical_cvar": float(h_cvar),
            "normal_var": float(n_var),
            "normal_cvar": float(n_cvar),
            "var_underestimation_ratio": float(var_ratio),
            "cvar_underestimation_ratio": float(cvar_ratio),
            "excess_kurtosis": float(kurt),
            "tail_warning": tail_warning,
        }


# ── Pure-Python fallback for normal CDF / PDF / PPF ──────────────
import math

def _approx_norm_ppf(p: float) -> float:
    """Approximate standard normal quantile (no scipy dependency).

    Uses the rational approximation from Peter Acklam.
    Absolute error < 1.15e-9 for all p ∈ (0, 1).
    """
    # Constants for approximation
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        return q * num / den
    else:
        r = math.sqrt(-math.log(min(p, 1 - p))) if p > 0 else 10.0
        if q < 0:
            r = -r
        num = (((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]
        den = ((d[0] * r + d[1]) * r + d[2]) * r + d[3]
        return num / den


def _approx_norm_pdf(z: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Historical vs Normal CVaR tail-risk comparison"
    )
    parser.add_argument("--returns", nargs="+", type=float,
                        help="Space-separated daily returns")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Tail probability (default 0.05)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo comparing normal vs fat-tailed")
    args = parser.parse_args()

    if args.demo:
        print("=== CVaR Demo: Normal vs Fat-Tailed (t₃) Returns ===")
        np.random.seed(42)
        normal_ret = np.random.normal(0, 0.02, size=500)
        fat_ret = np.random.standard_t(df=3, size=500) * 0.02

        print("\n--- Normal Returns ---")
        c = CVaRCalculator(normal_ret)
        rpt = c.tail_risk_report(alpha=0.05)
        for k, v in rpt.items():
            print(f"  {k}: {v}")

        print("\n--- Fat-Tailed (t₃) Returns ---")
        c2 = CVaRCalculator(fat_ret)
        rpt2 = c2.tail_risk_report(alpha=0.05)
        for k, v in rpt2.items():
            print(f"  {k}: {v}")

        print(f"\n  Fat-tail CVaR understimation ratio: "
              f"{rpt2['cvar_underestimation_ratio']:.2f}x (should be >1.0)")
    elif args.returns:
        c = CVaRCalculator(np.array(args.returns))
        rpt = c.tail_risk_report(alpha=args.alpha)
        for k, v in rpt.items():
            print(f"  {k}: {v}")
    else:
        parser.print_help()
