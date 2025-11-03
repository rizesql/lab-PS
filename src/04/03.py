from src.lib import signal
import numpy as np
import matplotlib.pyplot as plt


f_wave = 17
f_sample = 35

sig = signal.Sin(A=1, f=f_wave, phi=0)
sig1 = signal.Sin(A=1, f=f_sample + 1, phi=0)
sig2 = signal.Sin(A=1, f=1, phi=0)

t_cont = np.linspace(0, 1, 400, endpoint=False)
t_disc = np.arange(0, 1, 1 / f_sample)

fig, ax = plt.subplots(3, 1, figsize=(6, 8))

ax[0].stem(t_disc, sig(t_disc), linefmt="r-", label="Samples")
ax[0].plot(t_cont, sig(t_cont), label=sig)
ax[0].legend()
ax[0].grid()

ax[1].stem(t_disc, sig(t_disc), linefmt="r-", label="Samples")
ax[1].plot(t_cont, sig1(t_cont), label=sig1)
ax[1].legend()
ax[1].grid()

ax[2].stem(t_disc, sig(t_disc), linefmt="r-", label="Samples")
ax[2].plot(t_cont, sig2(t_cont), label=sig2)
ax[2].legend()
ax[2].grid()

plt.savefig("src/04/03.pdf", format="pdf")
plt.show()
