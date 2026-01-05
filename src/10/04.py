import numpy as np


def companion(coeffs: np.ndarray):
    N = len(coeffs)
    return np.column_stack(
        (
            np.vstack((np.zeros(N - 1), np.eye(N - 1))),
            -coeffs,
        )
    )


def roots_companion(coeffs):
    return np.linalg.eigvals(companion(coeffs))


N = 5
coeffs = np.random.rand(N)

roots = roots_companion(coeffs)
print(roots)
