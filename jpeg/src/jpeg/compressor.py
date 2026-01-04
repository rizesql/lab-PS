import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from jpeg import huffman, image, transform
from jpeg.image import RGB
from jpeg.tables import BLOCK, Q_CHROMA, Q_LUM, QTable

logger = logging.getLogger(__name__)


@dataclass
class JPEGMeta:
    height: int
    width: int

    q_lum: QTable
    q_chroma: QTable

    tables: huffman.Tables


class JPEGData(NamedTuple):
    Y: huffman.BlockStream
    Cb: huffman.BlockStream
    Cr: huffman.BlockStream


class Options(NamedTuple):
    q_lum: QTable = Q_LUM
    q_chroma: QTable = Q_CHROMA

    tables: huffman.TablesProducer = huffman.std_tables


def compress(img: image.Image[RGB], quality: int = 50, opts: Options = Options()) -> tuple[JPEGMeta, JPEGData]:
    H, W = img.shape[:2]
    logger.info(f"Starting compression for image {W}x{H}")

    ycbcr = image.rgb_to_ycbcr(img)
    padded = image.pad(ycbcr, block_size=2 * BLOCK)
    logger.info("Converted to YCbCr and padded")

    Y, Cb, Cr = image.split_chans(padded)
    Cb, Cr = image.subsample(Cb), image.subsample(Cr)
    logger.info("Chroma subsampling complete")

    q_lum, q_lum_inv = transform.scale_q_table(opts.q_lum, quality)
    q_chroma, q_chroma_inv = transform.scale_q_table(opts.q_chroma, quality)

    Y = transform.quantize(Y, q_lum_inv)
    Cb = transform.quantize(Cb, q_chroma_inv)
    Cr = transform.quantize(Cr, q_chroma_inv)
    logger.info("DCT and Quantization complete")

    Y, Cb, Cr = huffman.prepare_mcu_blocks(Y, Cb, Cr)
    logger.info("DCT and Quantization complete")

    tables = opts.tables(Y, Cb + Cr)
    logger.info("Tables complete")

    Y = huffman.encode(Y, tables.dc_lum.lookup, tables.ac_lum.lookup)
    Cb = huffman.encode(Cb, tables.dc_chroma.lookup, tables.ac_chroma.lookup)
    Cr = huffman.encode(Cr, tables.dc_chroma.lookup, tables.ac_chroma.lookup)
    logger.info("Huffman encoding complete")

    return (
        JPEGMeta(
            height=H,
            width=W,
            q_lum=q_lum,
            q_chroma=q_chroma,
            tables=tables,
        ),
        JPEGData(Y, Cb, Cr),
    )


def calc_mse(img: image.Image[RGB], quality: int, opts: Options = Options()) -> float:
    H, W = img.shape[:2]
    logger.info(f"Starting compression for image {W}x{H}")

    ycbcr = image.rgb_to_ycbcr(img)
    padded = image.pad(ycbcr, block_size=2 * BLOCK)
    logger.info("Converted to YCbCr and padded")

    Y, Cb, Cr = image.split_chans(padded)
    Cb, Cr = image.subsample(Cb), image.subsample(Cr)
    logger.info("Chroma subsampling complete")

    q_lum, q_lum_inv = transform.scale_q_table(opts.q_lum, quality)
    q_chroma, q_chroma_inv = transform.scale_q_table(opts.q_chroma, quality)

    Y = transform.quantize(Y, q_lum_inv)
    Cb = transform.quantize(Cb, q_chroma_inv)
    Cr = transform.quantize(Cr, q_chroma_inv)

    Y = transform.dequantize(Y, q_lum)
    Cb = transform.dequantize(Cb, q_chroma)
    Cr = transform.dequantize(Cr, q_chroma)

    Cb, Cr = image.upsample(Cb), image.upsample(Cr)

    restored = image.merge_chans(Y, Cb, Cr)
    restored = image.ycbcr_to_rgb(restored)
    restored = restored[:H, :W, :]

    err = np.mean((img.astype(np.float32) - restored.astype(np.float32)) ** 2)
    return float(err)


def find_opt_quality(img: image.Image[RGB], target_mse: float, opts: Options = Options()) -> int:
    low, high = 1, 100
    best = 100

    logger.info(f"Searching for optimal quality with target MSE <= {target_mse:.2f}")

    iteration = 0
    while low <= high:
        iteration += 1
        mid = low + (high - low) // 2
        mse = calc_mse(img, mid, opts)

        logger.info(f"Iter {iteration}: Quality={mid}, MSE={mse:.2f}")

        if mse <= target_mse:
            best = mid
            high = mid - 1
        else:
            low = mid + 1

    logger.info(f"Converged to Quality={best}")
    return best
