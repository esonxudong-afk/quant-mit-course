"""GARCH(1,1) 波动率模型 + EWMA基准"""
import numpy as np
from scipy.optimize import minimize

class GARCHModel:
    def __init__(self):
        self._fitted = False

    def fit(self, returns, method='MLE'):
        r = np.asarray(returns, dtype=float)
        if len(r) < 20:
            raise ValueError("Need at least 20 observations")
        mu0 = np.mean(r)
        var0 = np.var(r)
        # (omega, alpha, beta, mu)
        x0 = np.array([var0*0.01, 0.1, 0.85, mu0])
        bounds = [(1e-10, var0), (0, 0.4), (0.5, 0.999), (None, None)]

        def nll(x):
            omega, alpha, beta, mu = x
            if alpha + beta >= 0.9999:
                return 1e10
            eps = r - mu
            sigma2 = np.zeros(len(r))
            sigma2[0] = var0
            for t in range(1, len(r)):
                sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
            sigma2 = np.maximum(sigma2, 1e-8)
            ll = -0.5 * np.sum(np.log(2*np.pi*sigma2) + (eps**2)/sigma2)
            return -ll

        res = minimize(nll, x0, bounds=bounds, method='SLSQP')
        self.omega_, self.alpha_, self.beta_, self.mu_ = res.x
        self._r = r
        self._sigma2 = np.zeros(len(r))
        self._sigma2[0] = var0
        eps = r - self.mu_
        for t in range(1, len(r)):
            self._sigma2[t] = self.omega_ + self.alpha_ * eps[t-1]**2 + self.beta_ * self._sigma2[t-1]
        self._sigma2 = np.maximum(self._sigma2, 1e-8)
        self._fitted = True
        self.log_likelihood_ = -res.fun
        self.converged_ = res.success
        return self

    def conditional_volatility(self):
        if not self._fitted:
            raise RuntimeError("fit first")
        return np.sqrt(self._sigma2)

    def forecast_volatility(self, steps=5):
        if not self._fitted:
            raise RuntimeError("fit first")
        sigma2 = self._sigma2[-1]
        forecasts = []
        for _ in range(steps):
            sigma2 = self.omega_ + (self.alpha_ + self.beta_) * sigma2
            forecasts.append(np.sqrt(sigma2))
        return np.array(forecasts)

    def persistence(self):
        if not self._fitted:
            raise RuntimeError("fit first")
        return self.alpha_ + self.beta_

    def half_life(self):
        p = self.persistence()
        if p <= 0 or p >= 1:
            return float('inf')
        return np.log(0.5) / np.log(p)

    def report(self):
        if not self._fitted:
            raise RuntimeError("fit first")
        return {
            "omega": round(self.omega_, 8),
            "alpha": round(self.alpha_, 4),
            "beta": round(self.beta_, 4),
            "mu": round(self.mu_, 6),
            "persistence": round(self.persistence(), 4),
            "half_life": round(self.half_life(), 1) if np.isfinite(self.half_life()) else "inf",
            "log_likelihood": round(self.log_likelihood_, 2),
            "converged": self.converged_,
            "unconditional_vol": round(np.sqrt(self.omega_ / (1 - self.persistence())), 6),
            "n_observations": len(self._r)
        }


class EWMASmoother:
    def __init__(self, lam=0.94):
        if not 0 < lam < 1:
            raise ValueError("lam in (0,1)")
        self.lam = lam

    def smooth(self, squared_returns):
        s2 = np.asarray(squared_returns, dtype=float)
        out = np.zeros(len(s2))
        out[0] = s2[0]
        for t in range(1, len(s2)):
            out[t] = (1 - self.lam) * s2[t] + self.lam * out[t-1]
        return np.sqrt(out)

    def compare_with_garch(self, returns, garch_model):
        r = np.asarray(returns, dtype=float)
        garch_vol = garch_model.conditional_volatility()
        ewma_vol = self.smooth(r**2)
        corr = np.corrcoef(garch_vol, ewma_vol)[0, 1]
        mae = np.mean(np.abs(garch_vol - ewma_vol))
        rmse = np.sqrt(np.mean((garch_vol - ewma_vol)**2))
        return {"correlation": round(corr, 4), "MAE": round(mae, 6), "RMSE": round(rmse, 6),
                "garch_mean": round(np.mean(garch_vol), 6), "ewma_mean": round(np.mean(ewma_vol), 6)}


def simulate_garch(n, omega=1e-5, alpha=0.1, beta=0.8, mu=0.0, seed=42):
    np.random.seed(seed)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    r = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(n):
        sigma2[t] = omega + alpha * (eps[t-1]**2 if t > 0 else sigma2[0]) + beta * (sigma2[t-1] if t > 0 else sigma2[0])
        eps[t] = np.sqrt(sigma2[t]) * np.random.randn()
        r[t] = mu + eps[t]
    return r
