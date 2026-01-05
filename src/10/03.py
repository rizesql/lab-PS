import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix
from numpy.lib.stride_tricks import sliding_window_view

from src.lib.l1regls import l1regls

try:
    y = np.load("time_series.npy")
except FileNotFoundError:
    print("Time series not found. Run `python -m src.10.01`")
    exit()


def design_matrix(ts: np.ndarray, p: int) -> np.ndarray:
    return sliding_window_view(ts, p)[:, ::-1]


p = 50


def greedy_ar(ts: np.ndarray, p: int, max_feat=10):
    X = design_matrix(ts[:-1], p)
    Y = ts[p:]

    _, n_features = X.shape

    residual = Y.copy()
    x_pred = np.zeros(n_features)

    indices = []
    for _ in range(max_feat):
        corr = X.T @ residual

        best_idx = np.argmax(np.abs(corr))
        if best_idx in indices:
            break

        indices.append(best_idx)

        X_selected = X[:, indices]
        pred_selected, _, _, _ = np.linalg.lstsq(X_selected, Y)

        residual = Y - (X_selected @ pred_selected)
        x_pred[indices] = pred_selected

    return x_pred


def l1_reg_ar(ts: np.ndarray, p: int, lambda_reg=50.0):
    X = design_matrix(ts[:-1], p)
    Y = ts[p:]
    scale = 1 / np.sqrt(lambda_reg)

    A_cvx, b_cvx = matrix(X * scale), matrix(Y * scale)
    x_pred = l1regls(A_cvx, b_cvx)

    return np.array(x_pred).flatten()


theta_greedy = greedy_ar(y, p)
theta_l1 = l1_reg_ar(y, p, 800.0)

np.save("theta_greedy.npy", theta_greedy)
np.save("theta_l1.npy", theta_l1)

fig, ax = plt.subplots(2)

ax[0].stem(theta_greedy, linefmt="--", markerfmt="o", label="Greedy")
ax[0].legend()
ax[0].grid()

ax[1].stem(theta_l1, linefmt="--", markerfmt="o", label="L1")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.savefig("src/10/03.pdf", format="pdf")
plt.show()
