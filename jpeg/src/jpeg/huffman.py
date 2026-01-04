from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numba import njit

from jpeg import transform
from jpeg.tables import (
    AC_CHROMA_BITS,
    AC_CHROMA_VALS,
    AC_LUM_BITS,
    AC_LUM_VALS,
    BLOCK_2D,
    DC_CHROMA_BITS,
    DC_CHROMA_VALS,
    DC_LUM_BITS,
    DC_LUM_VALS,
)

type FlatChan = np.ndarray[tuple[int, int], np.dtype[np.int16]]


def prepare_mcu_blocks(
    Y: transform.Chan, Cb: transform.Chan, Cr: transform.Chan
) -> tuple[FlatChan, FlatChan, FlatChan]:
    y = _ensure_Y_in_mcu_order(Y)

    return (
        y.reshape(-1, BLOCK_2D),
        Cb.reshape(-1, BLOCK_2D),
        Cr.reshape(-1, BLOCK_2D),
    )


def _ensure_Y_in_mcu_order(chan: transform.Chan):
    H, W = chan.shape[:2]
    ret = chan.reshape(H // 2, 2, W // 2, 2, BLOCK_2D).swapaxes(1, 2)

    return ret


type BlockBits = np.ndarray
type BlockStream = Iterator[BlockBits]


# Range of DCT coefficients (Standard JPEG is -2048 to 2047)
_V_OFFSET = 2048
_V_RANGE = np.arange(-_V_OFFSET, _V_OFFSET + 1)
_ABS_V = np.abs(_V_RANGE)

_SIZE_LUT = np.zeros_like(_ABS_V, dtype=np.uint8)
_SIZE_LUT[_ABS_V > 0] = np.floor(np.log2(_ABS_V[_ABS_V > 0])).astype(np.uint8) + 1

# B. JPEG Value Table
# Logic: x if x > 0 else (x - 1) & mask
_VAL_LUT = np.where(_V_RANGE > 0, _V_RANGE, (_V_RANGE - 1) & ((1 << _SIZE_LUT) - 1))
_VAL_LUT[_V_RANGE == 0] = 0


@njit()
def _dc_encode(diffs: np.ndarray, table: BlockBits):
    indices = diffs + _V_OFFSET
    cats = _SIZE_LUT[indices]
    vals = _VAL_LUT[indices]

    for cat, val in zip(cats, vals):
        arr = np.empty((2 if cat > 0 else 1, 2), dtype=np.int64)

        arr[0, :] = table[cat, :]
        if cat > 0:
            arr[1, 0], arr[1, 1] = val, cat

        yield arr


@njit()
def _ac_encode(chan: np.ndarray, table: BlockBits):
    val_lut, size_lut, offset = _VAL_LUT, _SIZE_LUT, _V_OFFSET

    eob_code, eob_len = table[0x00, :]
    pad_code, pad_len = table[0xF0, :]

    empty_block = np.array([[eob_code, eob_len]], dtype=np.int64)

    for block in chan[:, 1:]:
        nz_idx = np.flatnonzero(block)
        if nz_idx.size == 0:
            yield empty_block.copy()
            continue

        vals = block[nz_idx]

        runs = np.empty_like(nz_idx)
        runs[0] = nz_idx[0]
        runs[1:] = np.diff(nz_idx) - 1

        res_list = []
        for run, val in zip(runs, vals):
            while run >= 16:
                res_list.append(pad_code)
                res_list.append(pad_len)
                run -= 16

            idx = val + offset
            size = size_lut[idx]

            symbol = (run << 4) | size
            res_list.extend(table[symbol])

            if size > 0:
                res_list.append(val_lut[idx])
                res_list.append(size)

        if nz_idx[-1] < 62:
            res_list.append(eob_code)
            res_list.append(eob_len)

        arr = np.array(res_list, dtype=np.int64).reshape(-1, 2)
        yield arr


def encode(chan: FlatChan, dc_table: BlockBits, ac_table: BlockBits) -> BlockStream:
    dc_diffs = np.diff(chan[:, 0], prepend=0)
    dc_enc = _dc_encode(dc_diffs, dc_table)

    ac_enc = _ac_encode(chan, ac_table)

    for dc, ac in zip(dc_enc, ac_enc):
        yield np.concatenate((dc, ac))


@dataclass
class BitsVals:
    bits: np.ndarray
    vals: np.ndarray

    def __post_init__(self):
        self.lookup: BlockBits = _derive_lookup_table(self)


def _derive_lookup_table(src: BitsVals) -> np.ndarray:
    counts = src.bits.astype(np.int64)
    lengths = np.repeat(np.arange(1, 17), counts)
    base_codes = np.zeros(16, dtype=np.int64)

    code = 0
    for i in range(16):
        base_codes[i] = code
        code = (code + counts[i]) << 1

    bases = np.repeat(base_codes, counts)

    group_starts = np.repeat(np.cumsum(counts) - counts, counts)
    offsets = np.arange(len(src.vals)) - group_starts

    final_codes = bases + offsets

    lut = np.zeros((256, 2), dtype=np.int64)

    lut[src.vals] = np.stack((final_codes, lengths), axis=1)

    return lut


@dataclass
class Tables:
    dc_lum: BitsVals
    dc_chroma: BitsVals
    ac_lum: BitsVals
    ac_chroma: BitsVals


type TablesProducer = Callable[[FlatChan, FlatChan], Tables]


def std_tables(lum: FlatChan, chroma: FlatChan) -> Tables:
    return Tables(
        dc_lum=BitsVals(bits=DC_LUM_BITS, vals=DC_LUM_VALS),
        dc_chroma=BitsVals(bits=DC_CHROMA_BITS, vals=DC_CHROMA_VALS),
        ac_lum=BitsVals(bits=AC_LUM_BITS, vals=AC_LUM_VALS),
        ac_chroma=BitsVals(bits=AC_CHROMA_BITS, vals=AC_CHROMA_VALS),
    )


def optimized_tables(lum: FlatChan, chroma: FlatChan) -> Tables:
    raise NotImplementedError("optimized_tables is not implemented yet")
