import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lib import time

x = pd.read_csv("Train.csv")
N = len(x)

# a) cum 1Hz = 1/s, si 1h = 3600s, frecventa de esantionare = 1/3600Hz
f_sample = 1 / time.Hour

# b) daca avem un sample per ora, intervalul de timp esantionat are N ore
elapsed = (1 / f_sample) * N

# c) nyquist, f_sample / 2 > f_max
f_max = f_sample / 2

# d)
X = np.fft.fft(x["Count"])
X_mag = np.abs(X / N)
X_spec = X_mag[: N // 2]
f = f_sample * np.linspace(0, N / 2, N // 2) / N

# e) componenta continua este prima componenta din transformata Fourier
f_cont = X_spec[0]

X_spec[0] = 0

# f)
top_n = 5
f_top = X_spec.argpartition(-top_n)[-top_n:]
f_top_sorted = f_top[np.argsort(X_spec[f_top])][::-1]
print(f_top_sorted)
print(X_spec[f_top_sorted])

for idx in f_top_sorted:
    freq = f[idx]
    mag = X_spec[idx]

    period_hours = 1 / freq / time.Hour
    period_days = 1 / freq / time.Day
    period_weeks = 1 / freq / time.Week
    period_months = 1 / freq / time.Month
    period_years = 1 / freq / time.Year

    print(f"Index = {idx}, f = {freq:.8e} Hz, |X| = {mag:.4f}")
    print(f"  > Hours:   {period_hours:.2f}")
    print(f"  > Days:    {period_days:.2f}")
    print(f"  > Weeks:   {period_weeks:.2f}")
    print(f"  > Months:  {period_months:.3f}")
    print(f"  > Years:   {period_years:.4f}")
    print("--------------------------------------------------")


# g)
monday = 1056
samples_per_month = int(time.Month // time.Hour)


# i)

# --- Period Analysis for idx = 1, f = 0.00000002 Hz ---
#   > Hours:   18286.00
#   > Days:    761.92
#   > Weeks:   108.85
#   > Months:  25.030
#   > Years:   2.0861
# --------------------------------------------------
# --- Period Analysis for idx = 2, f = 0.00000003 Hz ---
#   > Hours:   9143.00
#   > Days:    380.96
#   > Weeks:   54.42
#   > Months:  12.515
#   > Years:   1.0430
# --------------------------------------------------
# --- Period Analysis for idx = 762, f = 0.00001158 Hz ---
#   > Hours:   24.00
#   > Days:    1.00
#   > Weeks:   0.14
#   > Months:  0.033
#   > Years:   0.0027
# --------------------------------------------------
# --- Period Analysis for idx = 3, f = 0.00000005 Hz ---
#   > Hours:   6095.33
#   > Days:    253.97
#   > Weeks:   36.28
#   > Months:  8.343
#   > Years:   0.6954
# --------------------------------------------------
# --- Period Analysis for idx = 109, f = 0.00000166 Hz ---
#  > Hours:   167.76
#  > Days:    6.99
#  > Weeks:   1.00
#  > Months:  0.230
#  > Years:   0.0191
# --------------------------------------------------

# Am analizat initial primele 4 componente, insa am observat ca dintre ele, singura componenta
# care nu au legatura cu trendul general, ci cu evenimente periodice ar fi fost numai 762,
# care are perioada zilnica.
# In plus, am mai luat in considerare o componenta, si aceasta are perioada saptamanala.

# Cum ciclul de 24 de ore are idx = 762, cel de 12 ore ar avea idx = 1524 (762 * 2) si doresc
# sa il pastrez. Asadar, tot ce este peste 1524 (approx 1550), voi considera ca zgomot si voi elimina.

idx_cutoff = 1550
X_filtered = X.copy()
X_filtered[idx_cutoff : N - idx_cutoff] = 0
x_filtered = np.fft.ifft(X_filtered)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))

ax[0].stem(X_spec, basefmt=" ")
ax[0].scatter(f_top_sorted, X_spec[f_top_sorted], color="red", label="Top frequencies", zorder=5)
ax[0].set_title("Fourier Spectrum (Magnitude)")
ax[0].set_xlabel("Frequency [Hz]")
ax[0].set_ylabel("|X(ω)|")
ax[0].grid()
ax[0].legend()

ax[1].plot(x["Count"][monday : monday + samples_per_month])
ax[1].set_title("Original Signal (≈1 month)")
ax[1].set_xlabel("Sample index (hourly)")
ax[1].set_ylabel("Traffic")
ax[1].grid()

ax[2].plot(x_filtered[monday : monday + samples_per_month], color="tab:blue", label="Filtered signal")
ax[2].set_title("Filtered Signal")
ax[2].set_xlabel("Sample index (hourly)")
ax[2].set_ylabel("Traffic (filtered)")
ax[2].grid()
ax[2].legend()

plt.tight_layout()
plt.savefig("src/05/01.pdf", format="pdf")
plt.show()
