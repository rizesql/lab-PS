import matplotlib.pyplot as plt
import numpy as np

from src.lib import fourier

N = 8
F = fourier.matrix(N)

fig, ax = plt.subplots(N, 2)
for idx in range(N):
    ax[idx, 0].plot(F[idx].real, marker="o")
    ax[idx, 1].plot(F[idx].imag, marker="o")

assert np.allclose(N * np.identity(N), F @ F.conj())

plt.savefig("src/03/01.pdf", format="pdf")
plt.show()
