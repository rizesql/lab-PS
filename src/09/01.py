import matplotlib.pyplot as plt
import numpy as np

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
plt.savefig("src/09/01.pdf", format="pdf")
plt.show()
