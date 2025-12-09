import itertools
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from src.lib import signal

warnings.filterwarnings("ignore")

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

search_space = np.linspace(0.01, 0.99, 100)

LIMIT = 20

grid = {
    "p": range(1, LIMIT + 1),
    "i": [0],
    "q": range(1, LIMIT + 1),
}

best_aic = float("inf")
best_params = None
best_fit = None

print(f"[{time.strftime('%H:%M:%S')}] Starting Grid Search for ARMA(p,q)...")
print(f"[{time.strftime('%H:%M:%S')}] Search Space: p=[1, {LIMIT}], q=[1, {LIMIT}]")
print("-" * 65)
print(f"{'Order (p,q)':<18} | {'Status':<10} | {'AIC':<12} | {'Time (s)':<10}")
print("-" * 65)

start_total = time.time()
for params in itertools.product(*grid.values()):
    start = time.time()

    try:
        model = ARIMA(y, order=params)
        res = model.fit()

        elapsed = time.time() - start

        print(f"ARMA({params})".ljust(18) + f" | {'OK':<10} | {res.aic:>12.2f} | {elapsed:>10.4f}")

        if res.aic < best_aic:
            best_aic = res.aic
            best_params = params
            best_fit = res
            print(f"   >>> New Best Model Found! (AIC: {best_aic:.2f})")
    except Exception:
        elapsed = time.time() - start
        status = "FAILED"
        print(f"ARMA({params})".ljust(18) + f" | {'FAILED':<10} | {'N/A':>12} | {elapsed:>10.4f}")

total_elapsed = time.time() - start_total

print("-" * 65)
print(f"[{time.strftime('%H:%M:%S')}] Grid Search Complete.")
print(f"Total Execution Time: {total_elapsed:.2f} seconds")
print(f"Best Configuration: ARMA{best_params}")
print(f"Best AIC: {best_aic:.4f}")
print("-" * 65)

y_pred = best_fit.fittedvalues

plt.plot(t, y, label="Observed Data")
plt.plot(t, y_pred, label=f"ARMA (p={best_params[0]}, q={best_params[2]})", linestyle="--")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("src/09/04.pdf", format="pdf")
plt.show()
