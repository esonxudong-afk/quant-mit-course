"""
VaR Calculator — Value at Risk via Three Methods + Portfolio VaR
================================================================

Computes VaR using three approaches:
1. Normal (parametric, delta-normal) VaR
2. Historical simulation VaR
3. Monte Carlo simulation VaR

Also provides Expected Shortfall (CVaR) and a portfolio-level wrapper that
computes portfolio VaR, component (marginal) VaR, and the diversification ratio.

Formulas
--------
Normal VaR:
    VaR_α = -μ - σ · Φ⁻¹(α) · √T
    where μ = mean return, σ = std, Φ⁻¹ = inverse normal CDF, T = horizon.

Historical VaR:
    VaR_α = -percentile(returns, α · 100)

Monte Carlo VaR:
    Simulate n_sim paths from fitted normal/empirical distribution,
    compute the α-percentile of terminal P&L.

Expected Shortfall (CVaR):
    ES_α = -E[returns | returns ≤ -VaR_α]

Portfolio VaR:
    PF_VaR = apply method to weighted portfolio returns.

Component VaR:
    CVaR_i = w_i · β_i · PF_VaR
    where β_i = ∂PF_VaR / ∂w_i (approximated via numeric perturbation).

Diversification Ratio:
    DR = Σ indiv_VaR_i / portfolio_VaR
    DR > 1 indicates diversification benefit.

Usage:
    returns = np.random.normal(0, 0.02, 1000)
    calc = VaRCalculator(returns, position_value=1_000_000, horizon_days=10)
    print(calc.report())

    # Portfolio
    pf = PortfolioVaR(returns_matrix, weights, labels=['Stocks', 'Bonds'])
    print(pf.report())
"""

import numpy as np
from scipy import stats


class VaRCalculator:
    """Three-method VaR calculator for a single asset or portfolio.

    Parameters
    ----------
    returns : np.ndarray (1-D)
        Historical return series (decimal, e.g., 0.02 = 2%).
    position_value : float
        Current position / portfolio value. VaR is scaled to this amount.
    horizon_days : int
        Time horizon in days. Used to scale the volatility (√t rule).
    """

    def __init__(
        self,
        returns: np.ndarray,
        position_value: float = 1.0,
        horizon_days: int = 1,
    ):
        r = np.asarray(returns, dtype=np.float64).flatten()
        if r.ndim != 1:
            raise ValueError("returns must be a 1-D array")
        if len(r) < 5:
            raise ValueError("returns must have at least 5 observations")
        if position_value <= 0:
            raise ValueError("position_value must be positive")
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")

        self.returns = r
        self.position_value = float(position_value)
        self.horizon_days = int(horizon_days)
        self.mu = float(np.mean(r))
        self.sigma = float(np.std(r, ddof=1))
        self.N = len(r)

    # ── parametric VaR ─────────────────────────────────────────────

    def normal_var(self, alpha: float = 0.05) -> float:
        """Parametric (delta-normal) VaR.

        VaR_α = -(μ·T + σ · z_α · √T) · P

        Parameters
        ----------
        alpha : float
            Significance level (0.01=99% VaR, 0.05=95% VaR).

        Returns
        -------
        float
            VaR amount (always ≥ 0).
        """
        self._check_alpha(alpha)
        z = stats.norm.ppf(alpha)
        scaled_mu = self.mu * self.horizon_days
        scaled_sigma = self.sigma * np.sqrt(self.horizon_days)
        var = -(scaled_mu + scaled_sigma * z) * self.position_value
        return max(var, 0.0)

    # ── historical VaR ─────────────────────────────────────────────

    def historical_var(self, alpha: float = 0.05) -> float:
        """Historical simulation VaR — quantile of the return distribution.

        Uses the empirical α-quantile of historical returns, scaled by √horizon.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        float
            VaR amount (always ≥ 0).
        """
        self._check_alpha(alpha)
        q = np.quantile(self.returns, alpha)
        scaled_q = q * np.sqrt(self.horizon_days)
        var = -scaled_q * self.position_value
        return max(var, 0.0)

    # ── Monte Carlo VaR ────────────────────────────────────────────

    def monte_carlo_var(
        self, alpha: float = 0.05, n_sim: int = 10000, seed: int | None = 42
    ) -> float:
        """Monte Carlo simulation VaR.

        Simulates n_sim return paths assuming returns are i.i.d. normal
        with parameters estimated from the input data.

        Parameters
        ----------
        alpha : float
            Significance level.
        n_sim : int
            Number of simulation paths.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        float
            VaR amount (always ≥ 0).
        """
        self._check_alpha(alpha)
        if n_sim < 100:
            raise ValueError("n_sim must be >= 100")
        rng = np.random.default_rng(seed)
        sim_returns = rng.normal(
            loc=self.mu * self.horizon_days,
            scale=self.sigma * np.sqrt(self.horizon_days),
            size=n_sim,
        )
        q = np.quantile(sim_returns, alpha)
        var = -q * self.position_value
        return max(var, 0.0)

    # ── Expected Shortfall ─────────────────────────────────────────

    def expected_shortfall(self, alpha: float = 0.05) -> float:
        """Expected Shortfall (CVaR) — average loss beyond VaR.

        Computed from historical returns: mean of returns below the α-quantile.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        float
            ES amount (always ≥ 0, always ≥ VaR).
        """
        self._check_alpha(alpha)
        q = np.quantile(self.returns, alpha)
        tail = self.returns[self.returns <= q]
        if len(tail) == 0:
            return 0.0
        es_return = np.mean(tail)
        scaled_es = es_return * np.sqrt(self.horizon_days)
        es = -scaled_es * self.position_value
        return max(es, 0.0)

    # ── aggregate report ───────────────────────────────────────────

    def report(self, alpha: float = 0.05) -> dict:
        """Generate a summary report with all three VaR methods and ES."""
        self._check_alpha(alpha)
        return {
            "position_value": self.position_value,
            "horizon_days": self.horizon_days,
            "n_observations": self.N,
            "mu_daily": round(self.mu, 8),
            "sigma_daily": round(self.sigma, 8),
            "alpha": alpha,
            "confidence_level": f"{(1 - alpha) * 100:.0f}%",
            "normal_var": round(self.normal_var(alpha), 6),
            "historical_var": round(self.historical_var(alpha), 6),
            "monte_carlo_var": round(self.monte_carlo_var(alpha), 6),
            "expected_shortfall": round(self.expected_shortfall(alpha), 6),
        }

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _check_alpha(alpha: float) -> None:
        if not 0 < alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5)")


# ═══════════════════════════════════════════════════════════════════
# PortfolioVaR
# ═══════════════════════════════════════════════════════════════════

class PortfolioVaR:
    """Portfolio-level VaR with component decomposition.

    Parameters
    ----------
    returns_matrix : np.ndarray (T × N)
        Historical returns: T time periods, N assets.
    weights : np.ndarray (N,)
        Portfolio weights (should sum to 1).
    labels : list[str] | None
        Asset names for reporting.
    position_value : float
        Portfolio notional value.
    """

    def __init__(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        labels: list[str] | None = None,
        position_value: float = 1.0,
    ):
        R = np.asarray(returns_matrix, dtype=np.float64)
        if R.ndim != 2:
            raise ValueError("returns_matrix must be 2-D (T × N)")
        T, N = R.shape
        if T < 3:
            raise ValueError("returns_matrix must have at least 3 time periods")

        w = np.asarray(weights, dtype=np.float64).flatten()
        if len(w) != N:
            raise ValueError(f"weights length {len(w)} != columns {N}")
        if not np.isclose(np.sum(w), 1.0, atol=1e-6):
            raise ValueError("weights must sum to 1")
        if np.any(w < 0) or np.any(w > 1):
            raise ValueError("weights must be in [0, 1]")
        if position_value <= 0:
            raise ValueError("position_value must be positive")

        self.returns = R
        self.weights = w
        self.labels = labels or [f"Asset_{i}" for i in range(N)]
        self.position_value = float(position_value)
        self.T = T
        self.N = N

        # Pre-compute portfolio returns
        self.pf_returns = R @ w

    # ── portfolio VaR ──────────────────────────────────────────────

    def portfolio_var(self, method: str = "historical", alpha: float = 0.05) -> float:
        """Compute portfolio-level VaR.

        Parameters
        ----------
        method : str
            One of 'normal', 'historical' (default), 'monte_carlo'.
        alpha : float
            Significance level.

        Returns
        -------
        float
            Portfolio VaR amount.
        """
        calc = VaRCalculator(
            self.pf_returns,
            position_value=self.position_value,
        )
        if method == "normal":
            return calc.normal_var(alpha)
        elif method == "historical":
            return calc.historical_var(alpha)
        elif method == "monte_carlo":
            return calc.monte_carlo_var(alpha)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'normal', 'historical', or 'monte_carlo'.")

    # ── component VaR (marginal VaR contribution) ──────────────────

    def component_var(
        self, method: str = "historical", alpha: float = 0.05, epsilon: float = 1e-4
    ) -> np.ndarray:
        """Compute component VaR for each asset.

        Component VaR is derived from the gradient of portfolio VaR:

            CVaR_i = w_i · ∂VaR(pf) / ∂w_i

        The partial derivative is approximated via central differences:

            ∂VaR/∂w_i ≈ [VaR(w + ε·e_i) - VaR(w - ε·e_i)] / (2ε)

        After computing contributions, they are rescaled to sum exactly
        to the portfolio VaR (Euler's theorem for homogeneous functions).

        Parameters
        ----------
        method : str
            VaR method.
        alpha : float
            Significance level.
        epsilon : float
            Perturbation size for numerical gradient.

        Returns
        -------
        np.ndarray (N,)
            Component VaR for each asset. Sums to portfolio VaR.
        """
        base_var = self.portfolio_var(method, alpha)
        if base_var < 1e-14:
            return np.zeros(self.N)

        marginal = np.zeros(self.N)

        for i in range(self.N):
            # Central difference without re-normalizing
            # (treat as small absolute change to weight i)
            w_up = self.weights.copy()
            w_down = self.weights.copy()
            w_up[i] += epsilon
            w_down[i] -= epsilon

            # Build portfolio returns with perturbed weights
            pf_r_up = self.returns @ w_up
            pf_r_down = self.returns @ w_down

            calc_up = VaRCalculator(pf_r_up, position_value=self.position_value)
            calc_down = VaRCalculator(pf_r_down, position_value=self.position_value)

            if method == "normal":
                var_up = calc_up.normal_var(alpha)
                var_down = calc_down.normal_var(alpha)
            elif method == "historical":
                var_up = calc_up.historical_var(alpha)
                var_down = calc_down.historical_var(alpha)
            elif method == "monte_carlo":
                var_up = calc_up.monte_carlo_var(alpha)
                var_down = calc_down.monte_carlo_var(alpha)
            else:
                raise ValueError(f"Unknown method: {method}")

            marginal[i] = (var_up - var_down) / (2 * epsilon)

        # Component VaR = w_i * marginal
        comp_var = self.weights * marginal

        # Euler allocation: rescale to sum exactly to base_var
        total_comp = np.sum(comp_var)
        if np.abs(total_comp) > 1e-12:
            comp_var = comp_var * (base_var / total_comp)
        else:
            # Equal allocation as fallback
            comp_var = np.full(self.N, base_var / self.N)

        return comp_var

    # ── diversification ratio ──────────────────────────────────────

    def diversification_ratio(self, alpha: float = 0.05) -> float:
        """Compute the diversification ratio.

        DR = Σ individual_VaR_i / portfolio_VaR

        DR > 1 indicates diversification benefit.
        For N independent, identical assets: DR ≈ √N.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        float
            Diversification ratio (≥ 1 in well-diversified portfolios).
        """
        indiv_var_sum = 0.0
        for i in range(self.N):
            calc = VaRCalculator(self.returns[:, i], position_value=self.position_value)
            indiv_var_sum += self.weights[i] * calc.historical_var(alpha)

        pf_var = self.portfolio_var("historical", alpha)

        if pf_var < 1e-12:
            return 0.0

        return indiv_var_sum / pf_var

    # ── report ─────────────────────────────────────────────────────

    def report(self, alpha: float = 0.05) -> dict:
        """Generate a summary report for the portfolio."""

        methods = ["normal", "historical", "monte_carlo"]
        pf_vars = {}
        for m in methods:
            pf_vars[m] = round(self.portfolio_var(m, alpha), 6)

        comp = self.component_var("historical", alpha)
        dr = round(self.diversification_ratio(alpha), 4)

        return {
            "n_assets": self.N,
            "n_periods": self.T,
            "labels": self.labels,
            "weights": [round(float(w), 6) for w in self.weights],
            "position_value": self.position_value,
            "alpha": alpha,
            "confidence_level": f"{(1 - alpha) * 100:.0f}%",
            "portfolio_var": pf_vars,
            "component_var_historical": [round(float(c), 6) for c in comp],
            "diversification_ratio": dr,
        }
