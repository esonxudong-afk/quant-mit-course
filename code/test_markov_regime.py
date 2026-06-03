"""Tests for markov_regime.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from markov_regime import (
    MarkovRegimeDetector,
    simulate_two_state_series,
)


class TestMarkovRegimeDetector:
    """Tests for MarkovRegimeDetector."""

    def test_fit_basic(self):
        """Basic fit on two-state simulated data."""
        prices = simulate_two_state_series(n=500, seed=42)
        mrd = MarkovRegimeDetector(n_states=2)
        mrd.fit(prices)
        tm = mrd.transition_matrix()
        assert tm.shape == (2, 2)
        # Each row should sum to ~1
        for i in range(2):
            assert np.isclose(np.sum(tm[i]), 1.0, atol=1e-10), f"Row {i} doesn't sum to 1"

    def test_stationary_distribution_sums_to_one(self):
        """Stationary distribution must sum to 1."""
        prices = simulate_two_state_series(n=500, seed=42)
        mrd = MarkovRegimeDetector(n_states=2)
        mrd.fit(prices)
        sd = mrd.stationary_distribution()
        assert np.isclose(np.sum(sd), 1.0, atol=1e-10), f"Sum is {np.sum(sd)}"
        assert len(sd) == 2

    def test_reasonable_transition_matrix(self):
        """Both regimes should be reasonably sticky."""
        prices = simulate_two_state_series(n=1000, seed=42)
        mrd = MarkovRegimeDetector(n_states=2)
        mrd.fit(prices)
        tm = mrd.transition_matrix()
        # Both diagonals should be >= 0.5 (each state more likely to stay)
        assert tm[0, 0] >= 0.5, f"State 0 should be sticky: {tm}"
        assert tm[1, 1] >= 0.5, f"State 1 should be sticky: {tm}"

    def test_compute_states(self):
        """compute_states should return array of state labels."""
        prices = simulate_two_state_series(n=500, seed=42)
        mrd = MarkovRegimeDetector(n_states=3)
        states = mrd.compute_states(prices, window=20)
        assert len(states) == len(prices)
        assert set(np.unique(states)).issubset({0, 1, 2})

    def test_predict_regime_prob(self):
        """predict_regime_prob should return a valid probability distribution."""
        prices = simulate_two_state_series(n=500, seed=42)
        mrd = MarkovRegimeDetector(n_states=2)
        mrd.fit(prices)
        probs = mrd.predict_regime_prob(0)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-10)
        assert len(probs) == 2
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_regime_prob_invalid_state(self):
        """Should raise ValueError for out-of-range state."""
        prices = simulate_two_state_series(n=200, seed=42)
        mrd = MarkovRegimeDetector(n_states=2)
        mrd.fit(prices)
        with pytest.raises(ValueError):
            mrd.predict_regime_prob(5)
        with pytest.raises(ValueError):
            mrd.predict_regime_prob(-1)

    def test_report(self):
        """Report should contain expected fields."""
        prices = simulate_two_state_series(n=300, seed=42)
        mrd = MarkovRegimeDetector(n_states=2)
        mrd.fit(prices)
        rep = mrd.report()
        assert rep["fitted"] is True
        assert "transition_matrix" in rep
        assert "stationary_distribution" in rep
        assert "interpretation" in rep
        assert len(rep["interpretation"]) == 2

    def test_not_fitted_raises(self):
        """Accessing results before fit should raise RuntimeError."""
        mrd = MarkovRegimeDetector(n_states=2)
        with pytest.raises(RuntimeError):
            mrd.transition_matrix()
        with pytest.raises(RuntimeError):
            mrd.stationary_distribution()
        with pytest.raises(RuntimeError):
            mrd.predict_regime_prob(0)

    def test_n_states_one(self):
        """Single state should work (trivially)."""
        prices = np.cumprod(1 + np.random.randn(100) * 0.01) * 100
        mrd = MarkovRegimeDetector(n_states=1)
        mrd.fit(prices)
        tm = mrd.transition_matrix()
        assert tm.shape == (1, 1)
        assert np.isclose(tm[0, 0], 1.0)
        sd = mrd.stationary_distribution()
        assert np.isclose(sd[0], 1.0)

    def test_rejects_negative_prices(self):
        """Should raise ValueError for non-positive prices."""
        with pytest.raises(ValueError):
            mrd = MarkovRegimeDetector(n_states=2)
            mrd.fit([100, -10, 200])

    def test_rejects_insufficient_data(self):
        """Should raise ValueError for too few observations."""
        with pytest.raises(ValueError):
            mrd = MarkovRegimeDetector(n_states=2)
            mrd.fit([100, 101, 102])  # only 3

    def test_list_input_accepted(self):
        """List input should be accepted."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
        mrd = MarkovRegimeDetector(n_states=2)
        mrd.fit(prices)
        assert mrd._fitted

    def test_compute_states_insufficient_window(self):
        """Should raise ValueError if prices < window+1."""
        prices = np.array([100.0, 101.0, 102.0])
        mrd = MarkovRegimeDetector(n_states=2)
        with pytest.raises(ValueError):
            mrd.compute_states(prices, window=20)

    def test_three_states(self):
        """Test with 3 states."""
        prices = simulate_two_state_series(n=500, seed=42)
        mrd = MarkovRegimeDetector(n_states=3)
        mrd.fit(prices)
        tm = mrd.transition_matrix()
        assert tm.shape == (3, 3)
        for i in range(3):
            assert np.isclose(np.sum(tm[i]), 1.0, atol=1e-10)
        sd = mrd.stationary_distribution()
        assert np.isclose(np.sum(sd), 1.0, atol=1e-10)
