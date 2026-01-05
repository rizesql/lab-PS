import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    y = np.load("time_series.npy")
except FileNotFoundError:
    print("Time series not found. Run `python -m src.10.01`")
    exit()


def design_matrix(ts, p):
    return sliding_window_view(ts, p)[:, ::-1]


def ar_fit(ts, p, m=None):
    X = design_matrix(ts[:-1], p)
    Y = ts[p:]

    m = len(Y) if m is None else min(m, len(Y))

    X = X[-m:]
    Y = Y[-m:]
    return np.linalg.solve(X.T @ X, X.T @ Y)


p = 50
theta_ar = ar_fit(y, p)
np.save("theta_ar.npy", theta_ar)

plt.plot(y, label="Observed")
plt.plot(design_matrix(y[:-1], p) @ theta_ar, label="Predicted", linestyle="--")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("src/10/02.pdf", format="pdf")
plt.show()
