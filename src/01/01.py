import numpy as np
import matplotlib.pyplot as plt

from src.lib import signal


x = signal.Cos(A=1, f=260, phi=np.pi / 3)
y = signal.Cos(A=1, f=140, phi=-np.pi / 3)
z = signal.Cos(A=1, f=60, phi=np.pi / 3)

frame = (0, 0.03)
step = 0.0005
sample_freq = int(200 * (frame[1] - frame[0]))

t = np.arange(frame[0], frame[1], step)
sampled_t = np.linspace(frame[0], frame[1], sample_freq)

fig, axes = plt.subplots(3)

for idx, sig in enumerate((x, y, z)):
    axes[idx].plot(t, sig(t), label=sig)
    axes[idx].stem(sampled_t, sig(sampled_t), linefmt="r-", label="Samples")

    axes[idx].legend()
    axes[idx].grid()

plt.savefig("src/01/01.pdf", format="pdf")
plt.show()
