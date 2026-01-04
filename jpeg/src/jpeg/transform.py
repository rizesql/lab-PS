import numpy as np
from scipy.fft import dctn, idctn

from jpeg import image
from jpeg.image import YCbCr
from jpeg.tables import BLOCK, BLOCK_2D, FLAT_BLOCK, UNZIGZAG_ORDER, ZIGZAG_ORDER, QTable

type Chan = np.ndarray[tuple[int, int, int], np.dtype[np.int16]]


def scale_q_table(q_table: np.ndarray, quality: int):
    q = np.clip(quality, 1, 100)
    scale = 5000 / q if q < 50 else 200 - q * 2

    table = np.floor((q_table * scale + 50) / 100).clip(8, 255)

    return table, 1 / table


def quantize(chan: image.Chan[image.YCbCr], inv_q: QTable) -> Chan:
    blocks = image.blockwise(chan, BLOCK)
    centered = blocks.astype(float) - 128
    dcted = dctn(centered, axes=(-2, -1), norm="ortho")

    quantized = (dcted * inv_q).round().astype(np.int16)

    zigzagged = quantized.reshape(quantized.shape[:2] + (FLAT_BLOCK,))[..., ZIGZAG_ORDER]
    return zigzagged


def dequantize(chan: Chan, q_table: QTable) -> image.Chan[YCbCr]:
    unzigzagged = chan[..., UNZIGZAG_ORDER].reshape(chan.shape[:2] + BLOCK_2D)

    dequantized = unzigzagged * q_table

    idcted = idctn(dequantized, axes=(-2, -1), norm="ortho")
    blocks = (idcted + 128).clip(0, 255).astype(np.uint8)

    H, W = blocks.shape[:2]
    blocks = blocks.swapaxes(1, 2).reshape(H * BLOCK, W * BLOCK)

    return blocks.clip(0, 255).astype(np.uint8)
