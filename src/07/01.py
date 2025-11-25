import matplotlib.pyplot as plt
import numpy as np

N = 128
f_max = 8

ns = np.linspace(0, f_max, num=N, endpoint=False)


def x1(ns):
    return np.sin(2 * np.pi * ns[:, np.newaxis] + 3 * np.pi * ns)


def x2(ns):
    return np.sin(4 * np.pi * ns[:, np.newaxis]) + np.cos(6 * np.pi * ns)


for idx, x in enumerate((x1(ns), x2(ns))):
    fig, ax = plt.subplots(1, 2)

    Y = np.fft.fft2(x)
    Y_db = 20 * np.log10(np.abs(Y))

    ax[0].imshow(x, cmap="gray")

    im = ax[1].imshow(Y_db)
    fig.colorbar(im, ax=ax[1])

    plt.tight_layout()
    plt.savefig(f"src/07/01-x{idx}.pdf", format="pdf")
    plt.show()


def Y1(N):
    Y = np.zeros((N, N))
    Y[0, 5] = Y[0, N - 5] = 1
    return Y


def Y2(N):
    Y = np.zeros((N, N))
    Y[5, 0] = Y[N - 5, 0] = 1
    return Y


def Y3(N):
    Y = np.zeros((N, N))
    Y[5, 5] = Y[N - 5, N - 5] = 1
    return Y


for idx, Y in enumerate((Y1(N), Y2(N), Y3(N))):
    Y_db = 20 * np.log10(np.abs(Y))
    x = np.abs(np.fft.ifft2(Y))

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(x, cmap="gray")

    im = ax[1].imshow(Y_db)
    fig.colorbar(im, ax=ax[1])

    plt.tight_layout()
    plt.savefig(f"src/07/01-Y{idx}.pdf", format="pdf")
    plt.show()
