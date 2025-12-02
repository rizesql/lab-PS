import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_squared_error

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

fig, ax = plt.subplots(4)

ax[0].plot(t, y)
ax[0].set_ylabel("Observed")
ax[0].grid()

ax[1].plot(t, trend(t))
ax[1].set_ylabel("Trend")
ax[1].grid()

ax[2].plot(t, seasonal(t))
ax[2].set_ylabel("Seasonal")
ax[2].grid()

ax[3].plot(residuals)
ax[3].set_ylabel("Residuals")
ax[3].set_xlabel("Time")
ax[3].grid()

plt.tight_layout()
plt.savefig("src/08/01-a.pdf", format="pdf")
plt.show()


def autocorr(x):
    x = x - np.mean(x)
    out = np.correlate(x, x, mode="full")
    return out[out.size // 2 :] / out[out.size // 2]


gamma = autocorr(y)

plt.plot(gamma)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid()

plt.tight_layout()
plt.savefig("src/08/01-b.pdf", format="pdf")
plt.show()


def design_matrix(ts, p):
    return sliding_window_view(ts, p)[:, ::-1]


def ols_fit(ts, p, m=None):
    X = design_matrix(ts[:-1], p)
    Y = ts[p:]

    m = len(Y) if m is None else min(m, len(Y))

    X = X[-m:]
    Y = Y[-m:]
    return np.linalg.solve(X.T @ X, X.T @ Y)


p = 2
x_pred = ols_fit(y, p)

plt.plot(y, label="Observed")
plt.plot(design_matrix(y[:-1], p) @ x_pred, label="Predicted", linestyle="--")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("src/08/01-c.pdf", format="pdf")
plt.show()


def grid_search(ts, ps, ms):
    best_score = float("inf")
    best_params = (0, 0)

    results = []

    test_size = int(len(ts) * 0.2)
    train_data, test_data = ts[:-test_size], ts[-test_size:]

    for p in ps:
        hist = np.concat([train_data[-p:], test_data[:-1]])
        X_test = design_matrix(hist, p)

        for m in ms:
            w = ols_fit(train_data, p, m)

            y_pred = X_test @ w
            rmse = np.sqrt(mean_squared_error(test_data, y_pred))

            m_label = m if m is not None else -1
            results.append({"p": p, "m": m_label, "rmse": rmse})

            if rmse < best_score:
                best_score = rmse
                best_params = (p, m)

    return ((best_params, best_score), pd.DataFrame(results))


p_grid = [1, 2, 5, 10, 20, 50, 80]
m_grid = [100, 200, 300, 400, 500, 600, 700, 800, 900, None]
(best_cfg, min_rmse), results = grid_search(y, p_grid, m_grid)

heatmap = results.pivot_table(index="p", columns="m", values="rmse")
col, row = heatmap.columns.get_loc(best_cfg[1]), heatmap.index.get_loc(best_cfg[0])

ax = sns.heatmap(heatmap, annot=True, fmt=".3f", cmap="crest")
ax.add_patch(
    Rectangle(
        (col, row),
        width=1,
        height=1,
        fill=False,
        edgecolor="blue",
        lw=4,
        clip_on=False,
    )
)

plt.tight_layout()
plt.savefig("src/08/01-d.pdf", format="pdf")
plt.show()
