```py
# huffman.py
from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterator

import numpy as np

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


type BlockBits = list[tuple[int, int]]
type BlockStream = Iterator[BlockBits]


# Range of DCT coefficients (Standard JPEG is -2048 to 2047)
_V_OFFSET = 2048
_V_RANGE = np.arange(-_V_OFFSET, _V_OFFSET + 1)

_ABS_V = np.abs(_V_RANGE)
_SIZE_LUT = np.zeros_like(_ABS_V, dtype=np.uint8)
_SIZE_LUT[_ABS_V > 0] = np.floor(np.log2(_ABS_V[_ABS_V > 0])).astype(np.uint8) + 1

# B. JPEG Value Table (Replaces _to_bits)
# Logic: x if x > 0 else (x - 1) & mask
_VAL_LUT = np.where(_V_RANGE > 0, _V_RANGE, (_V_RANGE - 1) & ((1 << _SIZE_LUT) - 1))
_VAL_LUT[_V_RANGE == 0] = 0


def _dc_encode(chan: FlatChan, table: BlockBits) -> BlockStream:
    diffs = np.diff(chan[:, 0], prepend=0)

    indices = diffs + _V_OFFSET

    cats = _SIZE_LUT[indices]
    vals = _VAL_LUT[indices]

    for cat, val in zip(cats.tolist(), vals.tolist()):
        code_huff, len_huff = table[cat]

        if cat > 0:
            yield [(code_huff, len_huff), (val, cat)]
        else:
            yield [(code_huff, len_huff)]


def _ac_encode(chan: FlatChan, table: BlockBits) -> BlockStream:
    size_lut = _SIZE_LUT
    val_lut = _VAL_LUT
    offset = _V_OFFSET

    for block in chan[:, 1:]:
        nz_idx = np.flatnonzero(block)
        if nz_idx.size == 0:
            yield [table[0x00]]
            continue

        runs = np.diff(nz_idx, prepend=-1) - 1
        vals = block[nz_idx]

        bits: BlockBits = []
        for run, val in zip(runs.tolist(), vals.tolist()):
            while run >= 16:
                bits.append(table[0xF0])
                run -= 16

            idx = val + offset
            size = size_lut[idx]

            symbol = (run << 4) | size
            bits.append(table[symbol])

            if size > 0:
                bits.append((val_lut[idx], size))

        if nz_idx[-1] < 62:
            bits.append(table[0x00])

        yield bits


def encode(chan: FlatChan, dc_table: BlockBits, ac_table: BlockBits) -> BlockStream:
    dc_enc = _dc_encode(chan, dc_table)
    ac_enc = _ac_encode(chan, ac_table)

    for dc, ac in zip(dc_enc, ac_enc):
        yield dc + ac


def _to_bits(val, size):
    if size == 0:
        return 0

    if val > 0:
        return val

    return (val - 1) & ((1 << size) - 1)


@dataclass
class BitsVals:
    bits: list[int]
    vals: list[int]

    def __post_init__(self):
        self.lookup: BlockBits = _derive_lookup_table(self)


def _derive_lookup_table(src: BitsVals) -> BlockBits:
    lut: BlockBits = [(0, 0)] * 256

    code = 0
    val_idx = 0

    for length in range(1, 17):
        for _ in range(src.bits[length - 1]):
            symbol = src.vals[val_idx]
            lut[symbol] = (code, length)

            code += 1
            val_idx += 1

        code <<= 1

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


def optimized_tables(lum: FlatChan, Chroma: FlatChan) -> Tables:
    raise NotImplementedError("optimized_tables is not implemented yet")

```

```py
# huffman

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


type BlockBits = list[tuple[int, int]] | np.ndarray
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


def _dc_encode(chan: FlatChan, table: BlockBits) -> list[np.ndarray]:
    diffs = np.diff(chan[:, 0], prepend=0)
    table_arr = np.array(table, dtype=np.int64)
    return _dc_encode_jit(diffs, table_arr, _VAL_LUT, _SIZE_LUT, _V_OFFSET)


@njit()
def _dc_encode_jit(diffs, table_arr, val_lut, size_lut, offset):
    n = len(diffs)
    out_list = []

    indices = diffs + offset
    cats = size_lut[indices]
    vals = val_lut[indices]

    for idx in range(n):
        cat = cats[idx]
        val = vals[idx]

        code_huff = table_arr[cat, 0]
        len_huff = table_arr[cat, 1]

        if cat > 0:
            arr = np.empty((2, 2), dtype=np.int64)
            arr[0, 0] = code_huff
            arr[0, 1] = len_huff
            arr[1, 0] = val
            arr[1, 1] = cat
        else:
            arr = np.empty((1, 2), dtype=np.int64)
            arr[0, 0] = code_huff
            arr[0, 1] = len_huff

        out_list.append(arr)

    return out_list


def _ac_encode(chan: FlatChan, table: BlockBits) -> list[np.ndarray]:
    table_arr = np.array(table, dtype=np.int64)
    return _ac_encode_jit(chan, table_arr, _VAL_LUT, _SIZE_LUT, _V_OFFSET)


@njit()
def _ac_encode_jit(chan, table_arr, val_lut, size_lut, offset):
    n_blocks = chan.shape[0]
    out_list = []

    # Pre-allocate for EOB case
    eob_arr = np.empty((1, 2), dtype=np.int64)
    eob_arr[0, 0] = table_arr[0x00, 0]
    eob_arr[0, 1] = table_arr[0x00, 1]

    # Pre-allocate for padding (run >= 16)
    pad_code = table_arr[0xF0, 0]
    pad_len = table_arr[0xF0, 1]

    # Pre-allocate EOB values
    eob_code = table_arr[0x00, 0]
    eob_len = table_arr[0x00, 1]

    for i in range(n_blocks):
        block = chan[i, 1:]
        nz_idx = np.flatnonzero(block)

        if nz_idx.size == 0:
            out_list.append(eob_arr.copy())
            continue

        res_list = []

        runs = np.empty_like(nz_idx)
        runs[0] = nz_idx[0]
        runs[1:] = nz_idx[1:] - nz_idx[:-1]

        vals = block[nz_idx]

        n_vals = len(vals)
        for j in range(n_vals):
            run = runs[j]
            val = vals[j]

            while run >= 16:
                res_list.append(pad_code)
                res_list.append(pad_len)
                run -= 16

            idx = val + offset
            size = size_lut[idx]

            symbol = (run << 4) | size
            res_list.append(table_arr[symbol, 0])
            res_list.append(table_arr[symbol, 1])

            if size > 0:
                res_list.append(val_lut[idx])
                res_list.append(size)

        if nz_idx[-1] < 62:
            res_list.append(eob_code)
            res_list.append(eob_len)

        # Convert res_list to array
        arr_len = len(res_list) // 2
        arr = np.empty((arr_len, 2), dtype=np.int64)
        for k in range(arr_len):
            arr[k, 0] = res_list[2 * k]
            arr[k, 1] = res_list[2 * k + 1]

        out_list.append(arr)

    return out_list


def encode(chan: FlatChan, dc_table: BlockBits, ac_table: BlockBits) -> BlockStream:
    dc_enc = _dc_encode(chan, dc_table)
    ac_enc = _ac_encode(chan, ac_table)

    for dc, ac in zip(dc_enc, ac_enc):
        yield np.concatenate((dc, ac))


@dataclass
class BitsVals:
    bits: list[int]
    vals: list[int]

    def __post_init__(self):
        self.lookup: BlockBits = _derive_lookup_table(self)


def _derive_lookup_table(src: BitsVals) -> BlockBits:
    lut: BlockBits = [(0, 0)] * 256

    code = 0
    val_idx = 0

    for length in range(1, 17):
        for _ in range(src.bits[length - 1]):
            symbol = src.vals[val_idx]
            lut[symbol] = (code, length)

            code += 1
            val_idx += 1

        code <<= 1

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


def optimized_tables(lum: FlatChan, Chroma: FlatChan) -> Tables:
    raise NotImplementedError("optimized_tables is not implemented yet")

```
