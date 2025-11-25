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


class Sinc:
    def __init__(self, B: f32):
        self.B = B

    def __call__(self, t: np.ndarray):
        return np.sinc(self.B * t) ** 2

    def __str__(self):
        return f"$sinc^2$ wave(B={self.B})"


class SincInterpolate:
    def __init__(self, t_samples: np.ndarray, x_samples: np.ndarray, Ts: f32):
        self.t_samples = t_samples
        self.x_samples = x_samples
        self.Ts = Ts

    def __call__(self, t: np.ndarray):
        arg = (t[None, :] - self.t_samples[:, None]) / self.Ts

        return np.sum(self.x_samples[:, None] * np.sinc(arg), axis=0)

    def __str__(self):
        return f"Sinc interpolation(Ts={self.Ts:.2f}, N={len(self.x_samples)})"


def gamma(sig, noise_sig, snr: f32):
    return np.sqrt(np.linalg.norm(sig) ** 2 / (snr * np.linalg.norm(noise_sig) ** 2))


def snr(original: np.ndarray, other: np.ndarray):
    sig_pow = np.mean(original**2)
    noise_pow = np.mean((original - other) ** 2)

    if noise_pow == 0:
        return float("inf")

    return 10 * np.log10(sig_pow / noise_pow)
