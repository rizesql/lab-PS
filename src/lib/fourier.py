import numpy as np


def matrix(N: int):
    omega = -2j * np.pi / N
    k = np.arange(N)
    n = np.arange(N)

    return np.exp(omega * np.outer(k, n))


def dft(sig):
    N = len(sig)
    return matrix(N) @ sig


def fft(sig):
    N = int(sig.size)
    if N == 1:
        return sig

    k = np.arange(N // 2)
    omega = np.exp(-2j * np.pi * k / N)

    even, odd = fft(sig[::2]), fft(sig[1::2])

    return np.concat([even + (omega * odd), even - (omega * odd)])
