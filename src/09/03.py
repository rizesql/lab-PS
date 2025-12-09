import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from src.lib import signal

N = 1000
t = np.linspace(0, N, N)
residuals = np.random.normal(0, scale=2, size=N)


def trend(t):
    return 0.00005 * t**2 + 0.02 * t + 12


def seasonal(t):
    return signal.Sin(A=4, f=50, phi=0)(t) + signal.Sin(A=1, f=204, phi=0)(t)


def time_series(t):
    return trend(t) + seasonal(t) + residuals


y = time_series(t)


class MA:
    def __init__(self, data: np.ndarray, q: int):
        self.data = data
        self.q = q

    def _residuals(self, params, _):
        mu, thetas = params[0], params[1:]

        return lfilter([1], np.r_[1, thetas], self.data - mu)

    def fit(self):
        res = least_squares(self._residuals, np.r_[np.mean(self.data), np.zeros(self.q)], args=(self.data,))

        self.mu = res.x[0]
        self.thetas = res.x[1:]
        self.errors = self._residuals(res.x, self.data)
        self.fitted = self.data - self.errors

        return self


q = 20

ma = MA(y, q).fit()
y_pred = ma.fitted


plt.plot(t, y, label="Observed Data")
plt.plot(t, y_pred, label=f"MA Trend (q={q})", linestyle="--")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("src/09/03.pdf", format="pdf")
plt.show()
