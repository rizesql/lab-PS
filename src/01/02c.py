import numpy as np
import matplotlib.pyplot as plt

from src.lib import signal


f_wave = 240
f_sample = f_wave * 8
duration_s = 1.0

# oversample the continuous plot by a factor of 20 to make it look smoother
t_cont = np.linspace(0, duration_s, int(duration_s * f_sample * 20), endpoint=False)
t_disc = np.arange(0, duration_s, 1 / f_sample)

c = signal.Sawtooth(A=1, f=f_wave, phi=0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(t_cont, c(t_cont), label=c)
ax1.stem(t_disc, c(t_disc), linefmt="r-", label="Samples")
ax1.set_xlabel("Time(s)")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid()

ax2.plot(t_cont, c(t_cont), label=c)
ax2.stem(t_disc, c(t_disc), linefmt="r-", label="Samples")
ax2.set_xlim(0, 1 / f_wave)
ax2.set_xlabel("Time(s)")
ax2.set_ylabel("Amplitude")
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.savefig("src/01/02c.pdf", format="pdf")
plt.show()
