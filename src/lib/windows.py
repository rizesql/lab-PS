import numpy as np


class Rect:
    def __init__(self, N: int):
        self.N = N
        self.data = np.ones(N)

    def __call__(self) -> np.ndarray:
        return self.data

    def __str__(self) -> str:
        return f"Rectangular(N={self.N})"

    def __len__(self) -> int:
        return self.N


class Hanning:
    def __init__(self, N: int):
        self.N = N
        self.data = self._generate()

    def _generate(self) -> np.ndarray:
        if self.N == 1:
            return np.array([1.0])

        n = np.arange(self.N)
        return 0.5 * (1 - np.cos(2 * np.pi * n / (self.N - 1)))

    def __call__(self) -> np.ndarray:
        return self.data

    def __str__(self) -> str:
        return f"Hanning(N={self.N})"

    def __len__(self) -> int:
        return self.N
