import matplotlib.pyplot as plt
import numpy as np
from scipy import datasets

from src.lib import filters, signal

raton_beton = datasets.face(gray=True).astype(np.float32)

pixel_noise = 200
noise = np.random.randint(-pixel_noise, high=pixel_noise + 1, size=raton_beton.shape)
raton_zgomoton = np.clip(raton_beton + noise, 0, 255)

snr_init = signal.snr(raton_beton, raton_zgomoton)

for filter, param, search_range in (
    (filters.mean, "kernel_size", range(3, 20, 2)),
    (filters.low_pass, "radius", range(10, 101, 5)),
):
    best_val, best_img, history = filters.grid_search(
        noisy_img=raton_zgomoton,
        clean_img=raton_beton,
        filter=filter,
        param=param,
        search_range=search_range,
        metric=signal.snr,
    )
    best_snr = history["scores"].max()

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(f"{filter.__name__}")

    ax[0].imshow(raton_zgomoton, cmap="gray")
    ax[0].set_title(f"Unfiltered image (SNR={snr_init:.2f})")
    ax[0].axis("off")

    ax[1].imshow(best_img, cmap="gray")
    ax[1].set_title(f"Best result ({param}={best_val}, SNR={best_snr:.2f})")
    ax[1].axis("off")

    ax[2].plot(history["vals"], history["scores"], marker="o")
    ax[2].axvline(best_val, color="r", linestyle="--", label=f"Best {param}={best_val}")
    ax[2].set_xlabel(param)
    ax[2].set_ylabel("Metric score (SNR)")
    ax[2].set_title(f"Tuning {param}")
    ax[2].legend()
    ax[2].grid()

    plt.tight_layout()
    plt.savefig(f"src/07/03-{filter.__name__}.pdf", format="pdf")
    plt.show()
