import logging
import struct
import time
import typing

import numpy as np

from jpeg.bit_writer import StreamingBitWriter
from jpeg.compressor import JPEGData, JPEGMeta
from jpeg.huffman import BitsVals
from jpeg.tables import ZIGZAG_ORDER, QTable

logger = logging.getLogger(__name__)


def to_file(f: typing.BinaryIO, meta: JPEGMeta, data: JPEGData):
    _marker(f, 0xD8)

    _marker(f, 0xE0)
    f.write(struct.pack(">H5sBBBHHBB", 16, b"JFIF\x00", 1, 1, 1, 72, 72, 0, 0))
    _dqt(f, _format_q_table(meta.q_lum), _format_q_table(meta.q_chroma))
    _sof(f, meta.height, meta.width)

    _dht(f, meta.tables.dc_lum, 0, 0)
    _dht(f, meta.tables.ac_lum, 1, 0)
    _dht(f, meta.tables.dc_chroma, 0, 1)
    _dht(f, meta.tables.ac_chroma, 1, 1)

    _sos(f)

    _pack_bitstream(f, data)

    _marker(f, 0xD9)


def _marker(f: typing.BinaryIO, marker: int):
    f.write(bytes([0xFF, marker]))


def _format_q_table(table: QTable):
    return table.flatten()[ZIGZAG_ORDER].astype(np.uint8).tobytes()


def _dqt(f: typing.BinaryIO, q_lum: bytes, q_chroma: bytes):
    _marker(f, 0xDB)
    f.write(struct.pack(">H", 132))

    f.write(bytes([0]))
    f.write(q_lum)

    f.write(bytes([1]))
    f.write(q_chroma)


def _sof(f: typing.BinaryIO, height: int, width: int):
    _marker(f, 0xC0)

    f.write(struct.pack(">H", 17))
    f.write(struct.pack(">BHH", 8, height, width))
    f.write(bytes([3]))
    f.write(bytes([1, 0x22, 0]))
    f.write(bytes([2, 0x11, 1]))
    f.write(bytes([3, 0x11, 1]))


def _dht(f: typing.BinaryIO, bv: BitsVals, t_class: int, t_id: int):
    _marker(f, 0xC4)

    length = 2 + 1 + 16 + len(bv.vals)
    f.write(struct.pack(">H", length))

    info = (t_class << 4) | t_id
    f.write(bytes([info]))

    f.write(bv.bits.tobytes())
    f.write(bv.vals.tobytes())


def _sos(f: typing.BinaryIO):
    _marker(f, 0xDA)
    f.write(struct.pack(">H", 12))
    f.write(bytes([3]))

    f.write(bytes([1, 0x00]))
    f.write(bytes([2, 0x11]))
    f.write(bytes([3, 0x11]))

    f.write(bytes([0, 63, 0]))


def _pack_bitstream(f: typing.BinaryIO, data: JPEGData):
    writer = StreamingBitWriter(f)

    Y, Cb, Cr = data.Y, data.Cb, data.Cr

    buffer: list[np.ndarray] = []
    buf_append = buffer.append
    y_grouper = zip(*[iter(Y)] * 4)
    BATCH_SIZE = 2048

    count = 0
    t_start = time.perf_counter()

    for y_blocks, cb_block, cr_block in zip(y_grouper, Cb, Cr):
        buf_append(y_blocks[0])
        buf_append(y_blocks[1])
        buf_append(y_blocks[2])
        buf_append(y_blocks[3])
        buf_append(cb_block)
        buf_append(cr_block)

        count += 6

        if count >= BATCH_SIZE:
            writer.write(np.concatenate(buffer))
            buffer.clear()

            t_end = time.perf_counter()
            logger.info(f"Processed batch of {count} blocks in {t_end - t_start:.4f}s")

            count = 0
            t_start = time.perf_counter()

    if buffer:
        writer.write(np.concatenate(buffer))
        t_end = time.perf_counter()
        logger.info(f"Processed final batch of {len(buffer)} blocks in {t_end - t_start:.4f}s")

    writer.flush()
