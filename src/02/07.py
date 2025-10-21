import numpy as np
import matplotlib.pyplot as plt

from src.lib import signal


f_wave = 20
f_sample = 1000
duration_s = 1.0

t_cont = np.arange(0, duration_s, 1 / f_sample)
sig = signal.Sin(A=1, f=f_wave, phi=0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(t_cont, sig(t_cont), label=sig)
ax1.stem(t_cont, sig(t_cont), label=f"Sample (f={f_sample}Hz)", linefmt="r-")
ax1.stem(t_cont[::4], sig(t_cont)[::4], label=f"Sample 1/4 (f={f_sample / 4}Hz)")
ax1.set_xlim(0, 1 / f_wave)
ax1.set_xlabel("Time(s)")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid()

ax2.plot(t_cont, sig(t_cont), label=sig)
ax2.stem(t_cont, sig(t_cont), label=f"Sample (f={f_sample}Hz)", linefmt="r-")
ax2.stem(t_cont[::16], sig(t_cont)[::16], label=f"Sample 1/16 (f={f_sample / 16}Hz)")
ax2.set_xlim(0, 1 / f_wave)
ax2.set_xlabel("Time(s)")
ax2.set_ylabel("Amplitude")
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.savefig("src/02/07.pdf", format="pdf")
plt.show()
