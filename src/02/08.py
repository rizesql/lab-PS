import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

alpha = np.linspace(-np.pi / 2, np.pi / 2, endpoint=False)

sin = np.sin(alpha)
taylor = alpha
taylor_err = np.abs(sin - taylor)

pade = (alpha - 7 / 60 * alpha**3) / (1 + alpha**2 / 20)
pade_err = np.abs(sin - pade)

fig = plt.figure(figsize=(8, 8))

gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

ax1.plot(alpha, sin, label="$sin(\\alpha)$", linewidth=3, color="blue", alpha=0.8)
ax1.plot(alpha, taylor, label="Taylor: $y = \\alpha$", linestyle="--", color="red")
ax1.plot(alpha, pade, label="Padé", linestyle="-.", color="green")
ax1.set_xlabel("$\\alpha$ (radiani)")
ax1.set_ylabel("Valoare")
ax1.legend()
ax1.grid()

ax2.plot(alpha, taylor_err, label="Eroare Taylor", linestyle="--", color="red")
ax2.plot(alpha, pade_err, label="Eroare Padé", linestyle="-.", color="green")
ax2.set_title("Eroare absoluta")
ax2.set_ylabel("Eroare")
ax2.legend()
ax2.grid()

ax3.plot(alpha, taylor_err, label="Eroare Taylor", linestyle="--", color="red")
ax3.plot(alpha, pade_err, label="Eroare Padé", linestyle="-.", color="green")
ax3.set_yscale("log")
ax3.set_title("Eroare absoluta (scara logaritmica)")
ax3.set_ylabel("Eroare")
ax3.legend()
ax3.grid()

plt.tight_layout()
plt.savefig("src/02/08.pdf", format="pdf")
plt.show()
