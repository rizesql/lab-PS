import numpy as np
from scipy.fft import dctn

from jpeg import image
from jpeg.tables import BLOCK, BLOCK_2D, ZIGZAG_ORDER, QTable

type Chan = np.ndarray[tuple[int, int, int], np.dtype[np.int16]]


def quantize(chan: image.Chan[image.YCbCr], inv_q: QTable) -> Chan:
    blocks = image.blockwise(chan, BLOCK)
    centered = blocks.astype(float) - 128
    dcted = dctn(centered, axes=(-2, -1), norm="ortho")

    quantized = (dcted * inv_q).round().astype(np.int16)

    zigzagged = quantized.reshape(quantized.shape[:2] + (BLOCK_2D,))[..., ZIGZAG_ORDER]
    return zigzagged


def scale_q_table(q_table: np.ndarray, quality: int):
    q = np.clip(quality, 1, 100)
    scale = 5000 / q if q < 50 else 200 - q * 2

    table = np.floor((q_table * scale + 50) / 100).clip(1, 255)

    return table, 1.0 / table
