import itertools

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

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

search_space = np.linspace(0.01, 0.99, 100)


def optimize_params(func, data, grid, fixed_args=()):
    best_params = []
    best_pred = []
    min_err = float("inf")

    for params in itertools.product(*grid):
        pred, err = func(data, *params, *fixed_args)

        if err < min_err:
            min_err = err
            best_params = params
            best_pred = pred

    return best_pred, best_params, min_err


@jit(nopython=True)
def exp_smoothing(x, alpha):
    s = np.zeros_like(x)
    s[0] = x[0]

    for t in range(1, len(x)):
        s[t] = alpha * x[t] + (1 - alpha) * s[t - 1]

    err = np.sum((s[:-1] - x[1:]) ** 2)
    return s, err


pred, params, min_err = optimize_params(
    func=exp_smoothing,
    data=y,
    grid=[search_space],
)


print(f"Best alpha: {params[0]:.4f}")
print(f"Min SSE: {min_err:.4f}")

plt.plot(y, label="Observed")
plt.plot(pred, linestyle="--", label=rf"Smoothed ($\alpha$={params[0]:.2f})")

plt.legend()
plt.grid()

plt.savefig("src/09/02-simple.pdf", format="pdf")
plt.show()


@jit(nopython=True)
def double_exp_smoothing(x, alpha, beta):
    s = np.zeros_like(x)
    s[0] = x[0]

    b = np.zeros_like(x)
    b[0] = x[1] - x[0]

    for t in range(1, len(x)):
        s[t] = alpha * x[t] + (1 - alpha) * (s[t - 1] + b[t - 1])
        b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * b[t - 1]

    err = np.sum((s[:-1] + b[:-1] - x[1:]) ** 2)
    return s + b, err


pred, params, min_err = optimize_params(
    func=double_exp_smoothing,
    data=y,
    grid=[search_space, search_space],
)

print(f"Best alpha: {params[0]:.4f}")
print(f"Best beta: {params[1]:.4f}")
print(f"Min SSE: {min_err:.4f}")

plt.plot(y, label="Observed")
plt.plot(pred, linestyle="--", label=rf"Smoothed ($\alpha$={params[0]:.2f}, $\beta$={params[1]:.2f})")

plt.legend()
plt.grid()

plt.savefig("src/09/02-double.pdf", format="pdf")
plt.show()


@jit(nopython=True)
def triple_exp_smoothing(x, alpha, beta, gamma, L):
    s = np.zeros_like(x)
    b = np.zeros_like(x)
    c = np.zeros_like(x)

    s[0] = x[0]
    b[0] = (x[L] - x[0]) / L

    c[:L] = x[:L] - s[0]

    for t in range(L, len(x)):
        s[t] = alpha * (x[t] - c[t - L]) + (1 - alpha) * (s[t - 1] + b[t - 1])
        b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * b[t - 1]
        c[t] = gamma * (x[t] - s[t] - b[t - 1]) + (1 - gamma) * c[t - L]

    err = np.sum((s[L - 1 : -1] + b[L - 1 : -1] + c[:-L] - x[L:]) ** 2)
    return s + b + c, err


L = 20
pred, params, min_err = optimize_params(
    func=triple_exp_smoothing,
    data=y,
    grid=[search_space, search_space, search_space],
    fixed_args=(L,),
)

print(f"Best alpha: {params[0]:.4f}")
print(f"Best beta: {params[1]:.4f}")
print(f"Best gamma: {params[2]:.4f}")
print(f"Min SSE: {min_err:.4f}")

plt.plot(y, label="Observed")
plt.plot(
    pred,
    linestyle="--",
    label=rf"Smoothed ($\alpha$={params[0]:.2f}, $\beta$={params[1]:.2f}, $\gamma$={params[2]:.2f})",
)

plt.legend()
plt.grid()

plt.savefig("src/09/02-triple.pdf", format="pdf")
plt.show()
