import numpy as np


def matrix(N: int):
    omega = -2j * np.pi / N
    k = np.arange(N)
    n = np.arange(N)

    return np.exp(omega * np.outer(k, n))


def transform(sig):
    N = len(sig)
    return matrix(N) @ sig
