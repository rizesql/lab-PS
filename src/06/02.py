import matplotlib.pyplot as plt
import numpy as np

N = 100
N_ITER = 4

x = np.zeros(N)
x[N // 4 : N // 4 + N // 2] = 1

xs = np.array([x])
x_curr = x

for _ in range(N_ITER - 1):
    convolved = np.convolve(x_curr, x_curr, mode="same")
    convolved /= np.max(convolved)

    xs = np.vstack([xs, convolved])
    x_curr = convolved


fig, ax = plt.subplots(N_ITER, 1, figsize=(8, 2 * N_ITER))

for idx, sig in enumerate(xs):
    ax[idx].plot(sig)
    ax[idx].grid()
    ax[idx].set_xlim(0, N)
    ax[idx].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig("src/06/02.pdf", format="pdf")
plt.show()
