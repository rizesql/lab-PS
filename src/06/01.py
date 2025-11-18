import matplotlib.pyplot as plt
import numpy as np

from src.lib import signal

B = 1
period = (-3.0, 3.0)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

for idx, f_sample in enumerate((1, 1.5, 2, 4)):
    row, col = divmod(idx, 2)

    t_cont = np.linspace(period[0], period[1], 4000)

    Ts = 1 / f_sample
    t = np.arange(np.floor(period[0] / Ts), np.ceil(period[1] / Ts) + 1) * Ts

    x = signal.Sinc(B)
    x_hat = signal.SincInterpolate(t, x(t), Ts)

    ax[row, col].set_title(f"$F_s$ = {f_sample} Hz")

    ax[row, col].plot(t_cont, x(t_cont), label=x)
    ax[row, col].plot(t_cont, x_hat(t_cont), label=x_hat, linestyle="--")

    ax[row, col].stem(t, x(t), basefmt=" ")

    ax[row, col].set_xlim(period)
    ax[row, col].axhline(0, color="black", linewidth=0.5)
    ax[row, col].set_xlabel("Timp(s)")
    ax[row, col].axvline(0, color="black", linewidth=0.5)
    ax[row, col].set_ylabel("Amplitudine")
    ax[row, col].grid()
    ax[row, col].legend()

plt.tight_layout()
plt.savefig("src/06/01.pdf", format="pdf")
plt.show()
