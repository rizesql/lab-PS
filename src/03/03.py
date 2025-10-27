import matplotlib.pyplot as plt
import numpy as np

from src.lib import fourier, signal


def sig(t):
    sig1 = signal.Sin(A=1, f=10, phi=np.pi / 2)
    sig2 = signal.Sin(A=3, f=13.5, phi=0)
    sig3 = signal.Sin(A=1, f=7, phi=3 * np.pi / 4)

    return sig1(t) + sig2(t) + sig3(t)


f_sample = 100
t_disc = np.arange(0, 1, 1 / f_sample)
t_cont = np.linspace(0, 1.0, 32 * 120, endpoint=False)


samples = sig(t_disc)
ft = fourier.transform(samples)
assert np.allclose(ft, np.fft.fft(samples))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(t_cont, sig(t_cont))
ax[0].set_xlabel("Time(s)")
ax[0].set_ylabel("sig(t)")
ax[0].grid()

ax[1].stem(np.abs(ft))
ax[1].set_xlabel("Frequency(Hz)")
ax[1].set_ylabel(r"$|X(\omega)|$")
ax[1].grid()

plt.tight_layout()
plt.savefig("src/03/03.pdf", format="pdf")
plt.show()
