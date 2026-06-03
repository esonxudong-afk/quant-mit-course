"""
VaR Backtest — Kupiec & Christoffersen Tests
=============================================

Tests whether a VaR model is correctly calibrated by comparing
actual exceedances against the expected violation rate.

Tests
-----
1. Kupiec (1995) POF test — unconditional coverage
   LR_uc = -2 ln[(1-α)^(T-x) · α^x / ((1-x/T)^(T-x) · (x/T)^x)]
   ~ χ²(1)

2. Christoffersen (1998) conditional coverage test
   Tests whether violations are independent (no clustering),
   and jointly tests for correct coverage.
   LR_cc = LR_uc + LR_ind ~ χ²(2)

Formulas
--------
Violation indicator: I_t = 1 if -r_t > VaR_t, else 0

Kupiec LR:
    LR_uc = -2 [ (T - x)ln(1-α) + x·ln(α)
                 - (T - x)ln(1 - x/T) - x·ln(x/T) ]
    where T = total periods, x = violation count

Christoffersen independence:
    n_ij = transitions from state i to state j (i, j ∈ {0, 1})
    π_0 = n_01 / (n_00 + n_01)
    π_1 = n_11 / (n_10 + n_11)
    π   = (n_01 + n_11) / T

    LR_ind = -2 [ (n_00+n_10)·ln(1-π) + (n_01+n_11)·ln(π)
                  - n_00·ln(1-π_0) - n_01·ln(π_0)
                  - n_10·ln(1-π_1) - n_11·ln(π_1) ]

Usage:
    returns = np.random.normal(0, 0.02, 1000)
    var_series = np.full(1000, 0.033)  # constant 95% VaR
    bt = VaRBacktest(returns, var_series, alpha=0.05)
    print(bt.report())
"""

import numpy as np
from scipy import stats


class VaRBacktest:
    """Backtest a VaR model using Kupiec and Christoffersen tests.

    Parameters
    ----------
    returns : np.ndarray (1-D)
        Realized return series (decimal). Loss is -return.
    var_series : np.ndarray (1-D)
        Forecast VaR series, same length as returns.
        Each entry is the VaR amount predicted for that period.
    alpha : float
        Expected violation rate (0.05 = 95% VaR).
    """

    def __init__(
        self,
        returns: np.ndarray,
        var_series: np.ndarray,
        alpha: float = 0.05,
    ):
        r = np.asarray(returns, dtype=np.float64).flatten()
        v = np.asarray(var_series, dtype=np.float64).flatten()

        if r.ndim != 1:
            raise ValueError("returns must be 1-D")
        if v.ndim != 1:
            raise ValueError("var_series must be 1-D")
        if len(r) != len(v):
            raise ValueError(
                f"returns and var_series must have same length: got {len(r)} vs {len(v)}"
            )
        if len(r) < 20:
            raise ValueError("need at least 20 observations")
        if not 0 < alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5)")
        if np.any(v < 0):
            raise ValueError("VaR values must be non-negative")

        self.returns = r
        self.var_series = v
        self.alpha = float(alpha)
        self.T = len(r)

        # Violation indicators: violation when -return > VaR (loss exceeds VaR)
        self.violations = (-r > v).astype(int)
        self.x = int(np.sum(self.violations))  # total violation count

    # ── violation metrics ──────────────────────────────────────────

    def violation_rate(self) -> float:
        """Actual violation rate: x / T."""
        return self.x / self.T

    def expected_violations(self) -> float:
        """Expected violation count: α · T."""
        return self.alpha * self.T

    # ── Kupiec (POF) test ──────────────────────────────────────────

    def kupiec_test(self) -> dict:
        """Kupiec Proportion of Failures (POF) test.

        H₀: violation rate = α.

        Returns
        -------
        dict with keys:
            lr_stat : float — likelihood ratio statistic
            p_value : float — p-value (χ²(1))
            reject : bool — True if H₀ rejected at 5%
            method : str — "Kupiec POF"
        """
        if self.x == 0:
            # Limit case: no violations — compute directly
            lr = -2 * self.T * np.log(1 - self.alpha)
        elif self.x == self.T:
            # All violations
            lr = -2 * self.T * np.log(self.alpha)
        else:
            p_hat = self.x / self.T
            # Log-likelihood under H0
            ll_0 = (self.T - self.x) * np.log(1 - self.alpha) + self.x * np.log(self.alpha)
            # Log-likelihood under H1
            ll_1 = (self.T - self.x) * np.log(1 - p_hat) + self.x * np.log(p_hat)
            lr = -2 * (ll_0 - ll_1)

        lr = max(lr, 0.0)
        p_value = 1 - stats.chi2.cdf(lr, df=1)

        return {
            "lr_stat": round(lr, 6),
            "p_value": round(p_value, 6),
            "reject_5pct": bool(p_value < 0.05),
            "method": "Kupiec POF",
        }

    # ── Christoffersen (conditional coverage) test ─────────────────

    def christoffersen_test(self) -> dict:
        """Christoffersen conditional coverage test.

        Tests jointly:
          1. Unconditional coverage (same as Kupiec)
          2. Independence of violations (no clustering)

        Returns
        -------
        dict with keys:
            lr_cc : float — conditional coverage LR (χ²(2))
            lr_ind : float — independence LR (χ²(1))
            p_cc : float — p-value for conditional coverage
            p_ind : float — p-value for independence
            reject_cc : bool — True if joint H₀ rejected at 5%
            reject_ind : bool — True if independence H₀ rejected at 5%
            method : str — "Christoffersen"
        """
        # Transition counts n_ij: state i → state j
        # n_00: no violation → no violation
        # n_01: no violation → violation
        # n_10: violation → no violation
        # n_11: violation → violation
        n00, n01, n10, n11 = 0, 0, 0, 0
        for t in range(len(self.violations) - 1):
            if self.violations[t] == 0 and self.violations[t + 1] == 0:
                n00 += 1
            elif self.violations[t] == 0 and self.violations[t + 1] == 1:
                n01 += 1
            elif self.violations[t] == 1 and self.violations[t + 1] == 0:
                n10 += 1
            elif self.violations[t] == 1 and self.violations[t + 1] == 1:
                n11 += 1

        # Total periods used for transition
        T_trans = n00 + n01 + n10 + n11

        # Escape NaN by adding a small pseudo-count when categories are empty
        eps = 1e-12

        # Unrestricted probabilities
        pi0 = (n01 + eps) / (n00 + n01 + 2 * eps)  # Prob of violation after state 0
        pi1 = (n11 + eps) / (n10 + n11 + 2 * eps)  # Prob of violation after state 1
        pi = (n01 + n11 + 2 * eps) / (T_trans + 4 * eps)  # Overall violation prob

        # Independence LR
        # lr_ind = -2 * (L(π) - L(π₀, π₁))
        ll_restricted = (
            (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        )
        ll_unrestricted = (
            n00 * np.log(1 - pi0) + n01 * np.log(pi0)
            + n10 * np.log(1 - pi1) + n11 * np.log(pi1)
        )
        lr_ind = max(-2 * (ll_restricted - ll_unrestricted), 0.0)

        # Conditional coverage LR = LR_uc + LR_ind
        kupiec = self.kupiec_test()
        lr_uc = kupiec["lr_stat"]
        lr_cc = lr_uc + lr_ind

        p_ind = 1 - stats.chi2.cdf(lr_ind, df=1)
        p_cc = 1 - stats.chi2.cdf(lr_cc, df=2)

        return {
            "lr_cc": round(lr_cc, 6),
            "lr_ind": round(lr_ind, 6),
            "p_cc": round(p_cc, 6),
            "p_ind": round(p_ind, 6),
            "reject_cc_5pct": bool(p_cc < 0.05),
            "reject_ind_5pct": bool(p_ind < 0.05),
            "transition_counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
            "method": "Christoffersen",
        }

    # ── report ─────────────────────────────────────────────────────

    def report(self) -> dict:
        """Comprehensive backtest report."""
        kupiec = self.kupiec_test()
        chris = self.christoffersen_test()

        return {
            "n_observations": self.T,
            "alpha": self.alpha,
            "expected_violations": round(self.expected_violations(), 2),
            "actual_violations": self.x,
            "violation_rate": round(self.violation_rate(), 6),
            "kupiec": kupiec,
            "christoffersen": chris,
        }
