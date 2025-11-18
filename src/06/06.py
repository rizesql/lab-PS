import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from src.lib import time

period = 3 * time.Day // time.Hour
x_df = pd.read_csv("Train.csv")

start = 984
x = x_df.iloc[start : start + period]
N = len(x)
t = np.arange(N)

fig, ax = plt.subplots(4, 1, figsize=(14, 24))
fig.tight_layout(pad=5.0)

ax[0].set_title("(b) Rolling mean filtering")
ax[0].plot(t, x["Count"], label="Original signal (x)", linewidth=1.6)

for w in 5, 7, 9, 13, 17:
    m = np.convolve(x["Count"], np.ones(w), "same") / w
    ax[0].plot(t, m, label=f"Mean (w={w})", linewidth=0.8)

ax[0].set_xlabel("Time (h)")
ax[0].set_ylabel("Amplitude")
ax[0].legend()
ax[0].grid()

f_nyquist = 1 / 2.0
# vreau sa pastrez componenta cu perioada de 12 ore, pentru a surprinde evenimentele cu aceasta recurenta
f_cutoff = 1 / (0.5 * time.Day // time.Hour)
Wn = f_cutoff / f_nyquist

N_ord = 5
rp_db = 5

b_butt, a_butt = signal.butter(N_ord, Wn, btype="low")
y_butt = signal.filtfilt(b_butt, a_butt, x["Count"])

b_cheby, a_cheby = signal.cheby1(N_ord, rp_db, Wn, btype="low")
y_cheby = signal.filtfilt(b_cheby, a_cheby, x["Count"])

ax[1].set_title(f"(e) Filters Comparison (Order={N_ord}, Wn={Wn:.3f})")

ax[1].plot(t, x["Count"], label="Original signal (x)", linewidth=1.6)
ax[1].plot(t, y_butt, label="Butterworth filter")
ax[1].plot(t, y_cheby, label=f"Chebyshev I filter (rp={rp_db}dB)")

ax[1].set_xlabel("Time (h)")
ax[1].set_ylabel("Amplitude")
ax[1].legend()
ax[1].grid()


ax[2].set_title(f"Effect of Order (N) - (rp={rp_db}dB constant)")
ax[2].plot(t, x["Count"], label="Original", linewidth=1.6)

for ord in 2, 5, 10:
    b_butt, a_butt = signal.butter(ord, Wn, btype="low")
    y_butt = signal.filtfilt(b_butt, a_butt, x["Count"])
    ax[2].plot(t, y_butt, label=f"Butterworth filter (N={ord})")

    b_cheby, a_cheby = signal.cheby1(ord, rp_db, Wn, btype="low")
    y_cheby = signal.filtfilt(b_cheby, a_cheby, x["Count"])
    ax[2].plot(t, y_cheby, label=f"Order (N={ord})")

ax[2].legend()
ax[2].grid()
ax[2].set_xlabel("Time (h)")
ax[2].set_ylabel("Amplitude")

ax[3].set_title(f"Effect of Ripple (rp) - (Order={N_ord} constant)")
ax[3].plot(t, x["Count"], label="Original", linewidth=1.6)

for rp in 0.5, 5, 15:
    b_cheby, a_cheby = signal.cheby1(N_ord, rp, Wn, btype="low")
    y_cheby = signal.filtfilt(b_cheby, a_cheby, x["Count"])
    ax[3].plot(t, y_cheby, label=f"Ripple (rp={rp} dB)")

ax[3].legend()
ax[3].grid()
ax[3].set_xlabel("Time (h)")
ax[3].set_ylabel("Amplitude")

plt.savefig("src/06/06.pdf", format="pdf")
plt.show()
