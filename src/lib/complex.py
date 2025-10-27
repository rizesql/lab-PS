import numpy as np


def phasor(t, freq=1.0):
    return np.exp(-2j * np.pi * t * freq)


def wind(sig, freq=1.0):
    def wound_sig(t):
        return sig(t) * phasor(t, freq)

    return wound_sig
