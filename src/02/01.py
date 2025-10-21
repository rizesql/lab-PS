import numpy as np
import matplotlib.pyplot as plt

from src.lib import signal


f_wave = 3.5
f_sample = f_wave * 8
duration_s = 1.0

# oversample the continuous plot by a factor of 20 to make it look smoother
t_cont = np.linspace(0, duration_s, int(duration_s * f_sample * 20), endpoint=False)
t_disc = np.arange(0, duration_s, 1 / f_sample)

x1 = signal.Sin(A=2, f=f_wave, phi=0)
x2 = signal.Cos(A=2, f=f_wave, phi=-np.pi / 2)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].plot(t_cont, x1(t_cont), label=x1)
ax[0, 0].stem(t_disc, x1(t_disc), linefmt="r-", label="Samples")
ax[0, 0].set_xlabel("Time(s)")
ax[0, 0].set_ylabel("Amplitude")
ax[0, 0].legend()
ax[0, 0].grid()

ax[0, 1].plot(t_cont, x1(t_cont), label=x1)
ax[0, 1].stem(t_disc, x1(t_disc), linefmt="r-", label="Samples")
ax[0, 1].set_xlim(0, 1 / f_wave)
ax[0, 1].set_xlabel("Time(s)")
ax[0, 1].set_ylabel("Amplitude")
ax[0, 1].legend()
ax[0, 1].grid()

ax[1, 0].plot(t_cont, x1(t_cont), label=x2)
ax[1, 0].stem(t_disc, x1(t_disc), linefmt="r-", label="Samples")
ax[1, 0].set_xlabel("Time(s)")
ax[1, 0].set_ylabel("Amplitude")
ax[1, 0].grid()

ax[1, 1].plot(t_cont, x2(t_cont), label=x2)
ax[1, 1].stem(t_disc, x2(t_disc), linefmt="r-", label="Samples")
ax[1, 1].set_xlim(0, 1 / f_wave)
ax[1, 1].set_xlabel("Time(s)")
ax[1, 1].set_ylabel("Amplitude")
ax[1, 1].legend()
ax[1, 1].grid()

plt.tight_layout()
plt.savefig("src/02/01.pdf", format="pdf")
plt.show()
