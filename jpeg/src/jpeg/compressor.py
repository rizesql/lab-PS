import logging
from dataclasses import dataclass
from typing import NamedTuple

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


def compress(img: image.Image[RGB], opts: Options = Options()) -> tuple[JPEGMeta, JPEGData]:
    H, W = img.shape[:2]
    logger.info(f"Starting compression for image {W}x{H}")

    ycbcr = image.rgb_to_ycbcr(img)
    padded = image.pad(ycbcr, block_size=2 * BLOCK)
    logger.info("Converted to YCbCr and padded")

    Y, Cb, Cr = image.split_chans(padded)
    Cb, Cr = image.subsample(Cb), image.subsample(Cr)
    logger.info("Chroma subsampling complete")

    q_lum, q_lum_inv = transform.scale_q_table(opts.q_lum, quality=50)
    q_chroma, q_chroma_inv = transform.scale_q_table(opts.q_chroma, quality=50)

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
