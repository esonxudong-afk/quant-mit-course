"""
Markov Regime Detector
======================
Fits a hidden Markov model to price data to detect market regimes
based on volatility and ADX-like metrics. Provides state transition
matrix estimation and regime prediction.

Usage:
    mrd = MarkovRegimeDetector(n_states=2)
    mrd.fit(prices)
    states = mrd.compute_states(prices)
    print(mrd.report())
"""

import numpy as np
from scipy import stats


class MarkovRegimeDetector:
    """Two-regime (or multi-regime) Markov-switching detector.

    States are identified by volatility characteristics:
    - State 0: low volatility (quiet/trending)
    - State 1: high volatility (choppy/volatile)
    """

    def __init__(self, n_states: int = 2):
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        self.n_states = n_states
        self._transmat: np.ndarray | None = None      # n_states × n_states
        self._stationary: np.ndarray | None = None     # n_states
        self._state_means: np.ndarray | None = None    # feature means per state
        self._fitted = False

    # ── feature extraction ──────────────────────────────────────────────

    def _compute_volatility_feature(self, prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Compute rolling annualized volatility as state feature."""
        log_p = np.log(prices)
        ret = np.diff(log_p)
        if len(ret) < window:
            raise ValueError(f"Need at least {window + 1} prices for window={window}")
        vol = np.zeros(len(ret))
        # First window-1 values: use expanding window
        for i in range(window - 1):
            if i >= 1:
                vol[i] = np.std(ret[:i + 1], ddof=1)
            else:
                vol[i] = 0.0
        # Remaining: rolling window of size window
        for i in range(window - 1, len(ret)):
            vol[i] = np.std(ret[i - window + 1:i + 1], ddof=1)
        # Pad first element so length matches prices
        vol = np.concatenate([[vol[0]], vol])
        return vol

    def _compute_adx_like(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Compute a simplified ADX-like directional movement indicator."""
        if len(prices) < window + 1:
            raise ValueError(f"Need at least {window + 2} prices for ADX window={window}")
        diff = np.diff(prices)
        up_move = np.maximum(diff, 0)
        down_move = np.maximum(-diff, 0)

        # Wilder's smoothing
        atr = np.zeros(len(prices))
        atr[0] = np.mean(np.abs(diff[:window]))
        for i in range(1, len(prices) - 1):
            atr[i + 1] = (atr[i] * (window - 1) + np.abs(diff[i])) / window

        # Simple up/down smoothing
        plus_dm = np.zeros(len(prices))
        minus_dm = np.zeros(len(prices))
        plus_dm[1] = np.mean(up_move[:window])
        minus_dm[1] = np.mean(down_move[:window])
        for i in range(2, len(prices)):
            plus_dm[i] = (plus_dm[i - 1] * (window - 1) + up_move[i - 1]) / window
            minus_dm[i] = (minus_dm[i - 1] * (window - 1) + down_move[i - 1]) / window

        with np.errstate(divide='ignore', invalid='ignore'):
            pdi = 100 * plus_dm / np.where(atr > 0, atr, 1)
            mdi = 100 * minus_dm / np.where(atr > 0, atr, 1)
            dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) > 0, pdi + mdi, 1)
        dx = np.nan_to_num(dx)

        # ADX smoothing
        adx = np.zeros(len(prices))
        adx[0] = np.mean(dx[1:window + 1])
        for i in range(1, len(prices)):
            adx[i] = (adx[i - 1] * (window - 1) + dx[i]) / window

        return adx

    # ── model fitting ────────────────────────────────────────────────────

    def fit(self, prices: np.ndarray | list[float]):
        """Fit the Markov model by estimating the transition matrix from price data.

        This uses a simple 2-step approach:
        1. Discretize states based on volatility percentile thresholds.
        2. Count transitions between states to estimate transition probabilities.
        """
        prices_arr = np.asarray(prices, dtype=np.float64).flatten()
        if len(prices_arr) < 10:
            raise ValueError("Need at least 10 price observations to fit")
        if np.any(prices_arr <= 0):
            raise ValueError("prices must be strictly positive")

        # Use volatility as the feature to assign states
        vol = self._compute_volatility_feature(prices_arr, window=min(20, len(prices_arr) // 5))

        # Assign states: equal-frequency quantile bins
        states = self._assign_states_quantile(vol, self.n_states)

        # Estimate transition matrix
        self._transmat = self._estimate_transition_matrix(states, self.n_states)

        # Compute stationary distribution
        self._stationary = self._compute_stationary(self._transmat)

        self._fitted = True

    def compute_states(self, prices: np.ndarray | list[float], window: int = 20) -> np.ndarray:
        """Compute regime states for a price series.

        Args:
            prices: Price array.
            window: Rolling window size for volatility computation.

        Returns:
            1-D integer array of state assignments (0-based).
        """
        prices_arr = np.asarray(prices, dtype=np.float64).flatten()
        if len(prices_arr) < window + 1:
            raise ValueError(f"Need at least {window + 1} prices for window={window}")
        if np.any(prices_arr <= 0):
            raise ValueError("prices must be strictly positive")

        vol = self._compute_volatility_feature(prices_arr, window=window)
        return self._assign_states_quantile(vol, self.n_states)

    def transition_matrix(self) -> np.ndarray:
        """Return the estimated n_states × n_states transition matrix."""
        if not self._fitted:
            raise RuntimeError("Model must be fit() before accessing transition_matrix()")
        return self._transmat.copy()

    def stationary_distribution(self) -> np.ndarray:
        """Return the stationary distribution π (1-D, length n_states)."""
        if not self._fitted:
            raise RuntimeError("Model must be fit() before accessing stationary_distribution()")
        return self._stationary.copy()

    def predict_regime_prob(self, current_state: int) -> np.ndarray:
        """Given the current state, return probability distribution for next state.

        Args:
            current_state: Current regime state index (0 to n_states-1).

        Returns:
            1-D array of probabilities for each next state.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit() before calling predict_regime_prob()")
        if not (0 <= current_state < self.n_states):
            raise ValueError(f"current_state must be in [0, {self.n_states - 1}]")
        return self._transmat[current_state, :].copy()

    def report(self) -> dict:
        """Generate a diagnostic report."""
        if not self._fitted:
            return {"fitted": False}

        transmat = self._transmat
        stationary = self._stationary

        # Interpretation
        interpretation = []
        for i in range(self.n_states):
            diag = transmat[i, i]
            if diag > 0.7:
                interpretation.append(f"State {i}: sticky (self-transition {diag:.3f})")
            else:
                interpretation.append(f"State {i}: transient (self-transition {diag:.3f})")

        return {
            "fitted": True,
            "n_states": self.n_states,
            "transition_matrix": transmat.tolist(),
            "stationary_distribution": stationary.tolist(),
            "interpretation": interpretation,
        }

    # ── internals ───────────────────────────────────────────────────────

    def _assign_states_quantile(self, feature: np.ndarray, n_states: int) -> np.ndarray:
        """Assign states based on quantile thresholds of the feature."""
        states = np.zeros(len(feature), dtype=int)
        for s in range(1, n_states):
            threshold = np.quantile(feature, s / n_states)
            states[feature > threshold] = s
        return states

    def _estimate_transition_matrix(self, states: np.ndarray, n_states: int) -> np.ndarray:
        """Count state transitions to estimate P(s_t+1 | s_t)."""
        transmat = np.ones((n_states, n_states))  # Add-1 smoothing (Dirichlet prior)
        for t in range(len(states) - 1):
            transmat[states[t], states[t + 1]] += 1
        # Normalize rows
        row_sums = transmat.sum(axis=1, keepdims=True)
        return transmat / row_sums

    def _compute_stationary(self, transmat: np.ndarray) -> np.ndarray:
        """Compute stationary distribution via eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        # Find eigenvector for eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.abs(eigenvectors[:, idx].real)
        pi = pi / pi.sum()
        return pi


# ── test helpers ────────────────────────────────────────────────────────

def simulate_two_state_series(n: int = 1000, seed: int = 42) -> np.ndarray:
    """Simulate a price series with two volatility regimes.

    State 0: low volatility (σ=0.005), state 1: high volatility (σ=0.02).
    Transition matrix roughly: [[0.95, 0.05], [0.08, 0.92]].
    """
    rng = np.random.default_rng(seed)
    transmat_true = np.array([[0.95, 0.05],
                              [0.08, 0.92]])
    volatilities = [0.005, 0.02]
    state = 0
    prices = np.zeros(n)
    prices[0] = 100.0
    for t in range(1, n):
        state = rng.choice(2, p=transmat_true[state])
        prices[t] = prices[t - 1] * np.exp(rng.normal(0, volatilities[state]))
    return prices
