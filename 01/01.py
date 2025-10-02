import numpy as np
import matplotlib.pyplot as plt

type f32 = float | np.float32


def cos_sig(A: f32, f: f32, phi: f32):
    def signal(t):
        return A * np.cos(2 * np.pi * f * t + phi)

    return signal


x = cos_sig(A=1, f=260, phi=np.pi / 3)
y = cos_sig(A=1, f=140, phi=-np.pi / 3)
z = cos_sig(A=1, f=60, phi=np.pi / 3)

frame = (0, 0.03)
step = 0.0005
sample_freq = int(200 * (frame[1] - frame[0]))

t = np.arange(frame[0], frame[1], step)
sampled_t = np.linspace(frame[0], frame[1], sample_freq)

fig, axes = plt.subplots(3)

for idx, sig in enumerate((x, y, z)):
    axes[idx].plot(t, sig(t))
    axes[idx].stem(sampled_t, sig(sampled_t))

plt.show()
