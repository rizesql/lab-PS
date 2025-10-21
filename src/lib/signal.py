import numpy as np

type f32 = float | np.float32


class Sin:
    def __init__(self, A: f32, f: f32, phi: f32):
        self.A = A
        self.f = f
        self.phi = phi

    def __call__(self, t: np.ndarray):
        return self.A * np.sin(2 * np.pi * self.f * t + self.phi)

    def __str__(self):
        params = ""

        if self.A != 1.0:
            params += f", A={self.A}"

        if self.phi != 0.0:
            params += f", phi={self.phi:.2f}"

        return f"Sine wave(f={self.f}Hz{params})"


class Cos:
    def __init__(self, A: f32, f: f32, phi: f32):
        self.A = A
        self.f = f
        self.phi = phi

    def __call__(self, t: np.ndarray):
        return self.A * np.cos(2 * np.pi * self.f * t + self.phi)

    def __str__(self):
        params = ""

        if self.A != 1.0:
            params += f", A={self.A}"

        if self.phi != 0.0:
            params += f", phi={self.phi:.2f}"

        return f"Cosine wave(f={self.f}Hz{params})"


class Sawtooth:
    def __init__(self, A: f32, f: f32, phi: f32):
        self.A = A
        self.f = f
        self.phi = phi

    def __call__(self, t: np.ndarray):
        return self.A * 2 * self.f * np.mod(t, 1 / self.f) - 1 + self.phi

    def __str__(self):
        params = ""

        if self.A != 1.0:
            params += f", A={self.A}"

        if self.phi != 0.0:
            params += f", phi={self.phi:.2f}"

        return f"Sawtooth wave(f={self.f}Hz{params})"


class Square:
    def __init__(self, A: f32, f: f32, phi: f32):
        self.A = A
        self.f = f
        self.phi = phi

    def __call__(self, t: np.ndarray):
        return self.A * np.sign(np.sin(2 * np.pi * self.f * t + self.phi))

    def __str__(self):
        params = ""

        if self.A != 1.0:
            params += f", A={self.A}"

        if self.phi != 0.0:
            params += f", phi={self.phi:.2f}"

        return f"Square wave(f={self.f}Hz{params})"


def gamma(sig, noise_sig, snr: f32):
    return np.sqrt(np.linalg.norm(sig) ** 2 / (snr * np.linalg.norm(noise_sig) ** 2))
