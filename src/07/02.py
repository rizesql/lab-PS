import matplotlib.pyplot as plt
from scipy import datasets

from src.lib import filters, signal

f = datasets.face(gray=True).astype(float)

target_snr = 15.0
radius = 100
f_filtered = f.copy()
snr = float("inf")

while radius > 1:
    temp_img = filters.low_pass(f, radius)
    temp_snr = signal.snr(f, temp_img)

    if temp_snr < target_snr:
        break

    f_filtered = temp_img
    snr = temp_snr

    radius -= 2

radius += 2

print(f"Final Radius: {radius}")
print(f"Final SNR: {snr:.2f} dB")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(f, cmap="gray")
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(f_filtered, cmap="gray")
ax[1].set_title(f"Filtered (Radius={radius})\nSNR: {snr:.2f} dB")
ax[1].axis("off")

plt.tight_layout()
plt.savefig("src/07/02.pdf", format="pdf")
plt.show()
