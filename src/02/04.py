import numpy as np
import matplotlib.pyplot as plt

from src.lib import signal

duration_s = 1.0

t_cont = np.linspace(0, duration_s, int(duration_s * 8 * 8 * 20), endpoint=False)

x1 = signal.Sin(A=1, f=8, phi=0)
x2 = signal.Sawtooth(A=1, f=4.2, phi=0)

fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharey=True)

ax[0].plot(t_cont, x1(t_cont), label=x1)
ax[0].set_xlabel("Time(s)")
ax[0].set_ylabel("Amplitude")
ax[0].legend()
ax[0].grid()

ax[1].plot(t_cont, x2(t_cont), label=x2)
ax[1].set_xlabel("Time(s)")
ax[1].set_ylabel("Amplitude")
ax[1].legend()
ax[1].grid()

ax[2].plot(t_cont, x1(t_cont) + x2(t_cont), label=f"{x1} + {x2}")
ax[2].set_xlabel("Time(s)")
ax[2].set_ylabel("Amplitude")
ax[2].legend()
ax[2].grid()

plt.tight_layout()
plt.savefig("src/02/04.pdf", format="pdf")
plt.show()
