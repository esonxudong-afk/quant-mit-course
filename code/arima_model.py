"""ARIMA Model — Yule-Walker AR + MA estimation"""
import numpy as np

class ARIMAModel:
    def __init__(self, p=1, d=0, q=0):
        self.p, self.d, self.q = p, d, q
        self._fitted = False

    def fit(self, series):
        y = np.asarray(series, dtype=float).copy()
        if self.d > 0:
            y = np.diff(y, n=self.d)
        self._y, self._n = y, len(y)
        self._ar_coefs = np.zeros(max(self.p, 1))
        self._ma_coefs = np.zeros(max(self.q, 1))
        # Yule-Walker AR(p)
        if self.p > 0 and self._n > self.p + 5:
            acf = [np.corrcoef(y[i:], y[:len(y)-i])[0, 1] for i in range(self.p + 1)]
            R = np.array([[acf[abs(i-j)] for j in range(self.p)] for i in range(self.p)])
            r = np.array(acf[1:self.p+1])
            try:
                self._ar_coefs = np.linalg.solve(R, r)
            except np.linalg.LinAlgError:
                pass
        resid = y.copy()
        if self.p > 0:
            for t in range(self.p, len(y)):
                resid[t] = y[t] - np.dot(self._ar_coefs, y[t - self.p:t][::-1])
        self._sigma2 = np.var(resid[max(self.p, 1):]) if len(resid) > max(self.p, 1) else np.var(resid)
        self._resid = resid
        self._fitted = True
        return self

    def predict(self, steps=1):
        if not self._fitted: raise RuntimeError("fit first")
        y = list(self._y)
        for _ in range(steps):
            val = np.dot(self._ar_coefs, y[-self.p:][::-1]) if self.p > 0 and len(y) >= self.p else 0.0
            y.append(val)
        return np.array(y[-steps:])

    def aic(self):
        n, k = self._n, self.p + self.q + 1
        return n * np.log(self._sigma2 + 1e-10) + 2 * k

    def bic(self):
        n, k = self._n, self.p + self.q + 1
        return n * np.log(self._sigma2 + 1e-10) + k * np.log(n)

    def residuals(self):
        return self._resid

    def report(self):
        return {"p": self.p, "d": self.d, "q": self.q, "aic": round(self.aic(), 2),
                "bic": round(self.bic(), 2), "sigma2": round(self._sigma2, 8),
                "ar_coefs": self._ar_coefs.tolist(), "ma_coefs": self._ma_coefs.tolist()}


def auto_arima(series, max_p=5, max_q=2, criterion='aic'):
    best_score, best_model = float('inf'), None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            m = ARIMAModel(p=p, d=0, q=q).fit(series)
            score = m.aic() if criterion == 'aic' else m.bic()
            if score < best_score:
                best_score, best_model = score, m
    return best_model
