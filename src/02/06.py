import numpy as np
import matplotlib.pyplot as plt

from src.lib import signal


f_sample = 8
a = signal.Sin(A=1, f=f_sample / 2, phi=0)
b = signal.Sin(A=1, f=f_sample / 4, phi=0)
c = signal.Sin(A=1, f=0, phi=0)

t_cont = np.linspace(0, 1, f_sample * 8, endpoint=False)
t_disc = np.arange(0, 1, 1 / f_sample)

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharey=True)

for sig, ax in zip((a, b, c), axes):
    ax.plot(t_cont, sig(t_cont), label=sig)
    ax.stem(t_disc, sig(t_disc), linefmt="r-", label="Samples")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.savefig("src/02/06.pdf", format="pdf")
plt.show()
