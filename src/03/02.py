import matplotlib.pyplot as plt
import numpy as np

from src.lib import complex, signal

f_sample = 128

sig = signal.Sin(A=1, f=6, phi=0)
t_cont = np.arange(0.0, 1.0, 1 / f_sample / 8)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(t_cont, sig(t_cont), label=sig)
ax[0].set_xlabel("Time(s)")
ax[0].set_ylabel("Amplitude")
ax[0].grid()

wound_sig = complex.wind(sig)

ax[1].plot(wound_sig(t_cont).real, wound_sig(t_cont).imag)
ax[1].set_xlabel("Real")
ax[1].set_ylabel("Imaginar")
ax[1].grid()

plt.tight_layout()
plt.savefig("src/03/02i.pdf", format="pdf")
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 8))

for idx, w in enumerate((1, 2, 5, 7)):
    row, col = idx // 2, idx % 2

    wound_sig = complex.wind(sig, freq=w)

    ax[row, col].plot(wound_sig(t_cont).real, wound_sig(t_cont).imag)
    ax[row, col].set_title(f"$\\omega$={w}")
    ax[row, col].set_xlabel("Real")
    ax[row, col].set_ylabel("Imaginar")
    ax[row, col].grid()


plt.tight_layout()
plt.savefig("src/03/02ii.pdf", format="pdf")
plt.show()
