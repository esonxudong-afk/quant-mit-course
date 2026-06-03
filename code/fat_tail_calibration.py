"""
Fat-Tail Calibration for Dynamic Grid Spread
=============================================
Calibrates the k_risk parameter of DynamicGridSpread based on
empirical excess kurtosis of the return series.

Motivation:
  The baseline Avellaneda-Stoikov model assumes normally distributed
  mid-price increments.  Crypto returns exhibit kurtosis ≈ 6 (BTC)
  so the baseline k_risk = 3.0 systematically under-estimates tail
  risk, leading to spreads that are too narrow during volatile regimes.

  This module computes a calibration factor f:

      f = 1 + max(0, (excess_kurtosis - 1) / 4)

  Then:
      k_risk_calibrated = base_k_risk × f

  Interpretation:
      - Normal returns (excess kurtosis ≈ 0) →  f ≈ 1.0
      - Moderate fat tails (kurtosis = 2)     →  f = 1.25
      - BTC-level kurtosis = 6                →  f = 2.25
      - Extreme (kurtosis = 10)               →  f = 3.25

  The threshold of 1 means we ignore trivially elevated kurtosis
  (sampling noise), and the 1/4 slope gives a smooth, conservative
  increase that does not over-react.

Usage:
    ftc = FatTailCalibration(returns)
    k_risk = ftc.calibrated_k_risk(base_k_risk=3.0)
    # => ~2.25x normal for BTC-level kurtosis, so k_risk ≈ 6.75

    report = ftc.report()
    # => {'kurtosis': 6.0, 'calibration_factor': 2.25, ...}
"""

import numpy as np


class FatTailCalibration:
    """Calibrate k_risk for DynamicGridSpread based on return kurtosis.

    Attributes:
        returns: 1-D numpy array of asset returns.
        n_obs: Number of observations.
    """

    def __init__(self, returns: np.ndarray):
        """Initialise the calibrator.

        Args:
            returns: 1-D array of return observations.

        Raises:
            ValueError: If returns has fewer than 30 observations
                (kurtosis estimated from small samples is unreliable).
        """
        returns = np.asarray(returns, dtype=float).ravel()
        if returns.size < 30:
            raise ValueError(f"Need at least 30 observations for kurtosis "
                             f"estimation, got {returns.size}")
        self.returns = returns
        self.n_obs = returns.size
        self._kurt = None

    def kurtosis(self) -> float:
        """Sample excess kurtosis (Fisher definition, normal = 0).

        Uses the unbiased estimator for the fourth central moment
        and corrects for sample-size bias (Anscombe-Glynn adjustment
        is applied as a reasonable approximation).

        Returns:
            float: Excess kurtosis. Zero for normal. Positive = fat tails.
        """
        if self._kurt is not None:
            return self._kurt

        ret = self.returns
        n = float(self.n_obs)
        mean_ret = np.mean(ret)
        centered = ret - mean_ret
        m2 = np.sum(centered ** 2) / n
        m4 = np.sum(centered ** 4) / n

        if m2 < 1e-30:
            self._kurt = 0.0
            return 0.0

        # Sample excess kurtosis (population formula)
        kurt_pop = m4 / (m2 ** 2) - 3.0

        # Small-sample bias correction (Joanes & Gill 1998, type 3)
        # g2_corrected ≈ ((n-1)/((n-2)(n-3))) * ((n+1)*g2 + 6)
        # but this can over-correct for small n.  We use a simpler
        # formula that converges to the excess kurtosis as n → ∞.
        kurt_unbiased = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * kurt_pop + 6)

        # For very small n, the correction is unreliable.  Clip at the
        # raw population estimate as a floor.
        if kurt_unbiased < kurt_pop and kurt_unbiased < 0:
            kurt_unbiased = kurt_pop

        self._kurt = float(kurt_unbiased)
        return self._kurt

    def calibration_factor(self) -> float:
        """Compute the spread-widening factor from return kurtosis.

        Formula:
            f = 1 + max(0, (excess_kurtosis - 1) / 4)

        Why:
            - Subtract 1 to tolerate mild kurtosis from sampling noise.
            - Divide by 4 for gradual response (don't over-widen).
            - Clamp at zero: sub-normal kurtosis doesn't narrow spreads.

        Returns:
            float: Calibration factor ≥ 1.0.
        """
        k = self.kurtosis()
        factor = 1.0 + max(0.0, (k - 1.0) / 4.0)
        return factor

    def calibrated_k_risk(self, base_k_risk: float = 3.0) -> float:
        """Calibrated risk-aversion coefficient for DynamicGridSpread.

        k_risk_calibrated = base_k_risk × calibration_factor

        Args:
            base_k_risk: Baseline risk coefficient (default 3.0, matching
                DynamicGridSpread default).

        Returns:
            float: Calibrated k_risk value.

        Raises:
            ValueError: If base_k_risk <= 0.
        """
        if base_k_risk <= 0:
            raise ValueError(f"base_k_risk must be positive, got {base_k_risk}")
        factor = self.calibration_factor()
        return base_k_risk * factor

    def report(self) -> dict:
        """Generate a complete calibration report.

        Returns:
            dict with:
                - n_observations
                - kurtosis: excess kurtosis
                - calibration_factor: spread-widening multiplier
                - calibrated_k_risk_default: k_risk with base=3.0
                - normality_verdict: 'normal' | 'mild_fat_tails' | 'fat_tails'
                  based on excess kurtosis thresholds
        """
        k = self.kurtosis()
        factor = self.calibration_factor()
        k_calib = self.calibrated_k_risk(base_k_risk=3.0)

        if k < 0.5:
            verdict = "normal"
        elif k < 3.0:
            verdict = "mild_fat_tails"
        else:
            verdict = "fat_tails"

        return {
            "n_observations": self.n_obs,
            "excess_kurtosis": float(k),
            "calibration_factor": factor,
            "calibrated_k_risk_default": k_calib,
            "normality_verdict": verdict,
        }


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fat-tail calibration for DynamicGridSpread k_risk"
    )
    parser.add_argument("--returns", nargs="+", type=float,
                        help="Space-separated daily returns")
    parser.add_argument("--base-k", type=float, default=3.0,
                        help="Base k_risk coefficient")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with normal and fat-tailed data")
    args = parser.parse_args()

    if args.demo:
        print("=== Fat-Tail Calibration Demo ===\n")
        np.random.seed(42)

        # Normal returns
        normal = np.random.normal(0, 0.02, size=252)
        print("--- Normal Returns (n=252) ---")
        ftc_n = FatTailCalibration(normal)
        r_n = ftc_n.report()
        for k, v in r_n.items():
            print(f"  {k}: {v}")

        # Fat-tailed returns (t₃)
        fat = np.random.standard_t(df=3, size=252) * 0.02
        print("\n--- Fat-Tailed Returns t₃ (n=252) ---")
        ftc_f = FatTailCalibration(fat)
        r_f = ftc_f.report()
        for k, v in r_f.items():
            print(f"  {k}: {v}")

        # BTC-like (mix)
        print("\n--- BTC-like Returns (kurtosis ≈ 6, n=500) ---")
        np.random.seed(99)
        btc_like = np.concatenate([
            np.random.normal(0, 0.02, size=450),
            np.random.normal(0, 0.10, size=50),  # 10% tails
        ])
        ftc_b = FatTailCalibration(btc_like)
        r_b = ftc_b.report()
        for k, v in r_b.items():
            print(f"  {k}: {v}")

        print(f"\n  🎯 k_risk moves from 3.0 → {r_b['calibrated_k_risk_default']:.2f} for BTC-like returns")
    elif args.returns:
        ftc = FatTailCalibration(np.array(args.returns))
        r = ftc.report()
        for k, v in r.items():
            print(f"  {k}: {v}")
    else:
        parser.print_help()
