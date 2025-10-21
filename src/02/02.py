import numpy as np
import matplotlib.pyplot as plt

from src.lib import signal

f_wave = 2.4
f_sample = f_wave * 8
duration_s = 1.0

# oversample the continuous plot by a factor of 20 to make it look smoother
t_cont = np.linspace(0, duration_s, int(duration_s * f_sample * 20), endpoint=False)
noise_cont = np.random.normal(0, duration_s, int(duration_s * f_sample * 20))

sigs = [signal.Sin(A=1, f=f_wave, phi=np.pi * p / 4) for p in range(4)]

fig, ax = plt.subplots(5, 1, figsize=(8, 8))

for sig in sigs:
    ax[0].plot(t_cont, sig(t_cont), label=sig)
ax[0].set_xlabel("Time(s)")
ax[0].set_ylabel("Amplitude")
ax[0].legend()
ax[0].grid()


for idx, snr in enumerate((0.1, 1.0, 10.0, 100.0)):
    x = sigs[0](t_cont)
    gamma = signal.gamma(x, noise_cont, snr)
    noised_x_cont = x + gamma * noise_cont

    ax[idx + 1].plot(t_cont, x, label=sigs[0])
    ax[idx + 1].plot(t_cont, x + gamma * noise_cont, label=f"Noised {sigs[0]}, gamma={gamma}")
    ax[idx + 1].set_xlabel("Time(s)")
    ax[idx + 1].set_ylabel("Amplitude")
    ax[idx + 1].legend()
    ax[idx + 1].grid()

plt.tight_layout()
plt.savefig("src/02/02.pdf", format="pdf")
plt.show()
