import matplotlib.pyplot as plt
import numpy as np

try:
    theta_ar = np.load("theta_ar.npy")
    theta_greedy = np.load("theta_greedy.npy")
    theta_l1 = np.load("theta_l1.npy")

    models = {
        "AR(50) Standard": theta_ar,
        "AR(50) Greedy": theta_greedy,
        "AR(50) L1": theta_l1,
    }
except FileNotFoundError:
    print("Thetas not found. Run `python -m src.10.01`, then `python -m src.10.02` and `python -m src.10.03`")
    exit()


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


def is_stationary(x_pred: np.ndarray):
    coeffs = -x_pred[::-1]
    coeffs = np.append(coeffs, 1.0)

    roots = roots_companion(coeffs)
    min_abs = np.min(np.abs(roots))
    return min_abs > 1.0, roots


t = np.linspace(0, 2 * np.pi, 200)
plt.plot(np.cos(t), np.sin(t), "k--", label="Unit Circle")

colors = ["b", "g", "r"]
markers = ["o", "x", "^"]


for i, (name, theta) in enumerate(models.items()):
    stationary, roots = is_stationary(theta)
    if len(roots) > 0:
        plt.scatter(roots.real, roots.imag, label=name, alpha=0.7, c=colors[i], marker=markers[i])

    print(f"Model: {name}")
    print(f"  Stationary: {'YES' if stationary else 'NO'}")
    print("-" * 30)

plt.title("Roots of Characteristic Polynomial")
plt.axhline(0, color="black", lw=0.5)
plt.axvline(0, color="black", lw=0.5)
plt.legend()
plt.grid(linestyle=":")
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.tight_layout()
plt.savefig("src/10/05.pdf", format="pdf")
plt.show()
