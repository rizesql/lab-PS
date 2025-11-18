import matplotlib.pyplot as plt
import numpy as np

from src.lib import signal, windows

Nw = 200

f_sample = 3000
duration = Nw / f_sample

t = np.arange(Nw) / f_sample
sig = signal.Sin(A=1, f=100, phi=0)

fig, ax = plt.subplots(2)

for idx, win in enumerate((windows.Rect(Nw), windows.Hanning(Nw))):
    ax[idx].plot(t, sig(t), "k--", label=sig, alpha=0.5)
    ax[idx].plot(t, sig(t) * win(), "r-", label=f"{sig} windowed")
    ax[idx].plot(t, win(), "b:", label=win, alpha=0.7)

    ax[idx].set_xlabel("Time(s)")
    ax[idx].set_ylabel("Amplitude")
    ax[idx].legend()
    ax[idx].grid()

plt.tight_layout()
plt.savefig("src/06/05.pdf", format="pdf")
plt.show()
