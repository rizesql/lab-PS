from typing import BinaryIO

import numpy as np
from numba import jit


class StreamingBitWriter:
    def __init__(self, fp: BinaryIO, buffer_size: int = 65536):
        self.fp = fp
        self.buf = 0
        self.cnt = 0

        self.out_buffer = bytearray()
        self.buffer_size = buffer_size

    def write(self, val: int, nbits: int):
        self.buf = (self.buf << nbits) | (val & ((1 << nbits) - 1))
        self.cnt += nbits

        while self.cnt >= 8:
            self.cnt -= 8
            byte = (self.buf >> self.cnt) & 0xFF

            self.out_buffer.append(byte)

            if byte == 0xFF:
                self.out_buffer.append(0x00)

            if len(self.out_buffer) >= self.buffer_size:
                self._flush_buffer_to_disk()

    def bulk_write(self, instructions: list[tuple[int, int]] | np.ndarray):
        if len(instructions) == 0:
            return

        if self.out_buffer:
            self._flush_buffer_to_disk()

        if isinstance(instructions, np.ndarray):
            data = instructions
        else:
            data = np.array(instructions, dtype=np.int64)

        vals = data[:, 0]
        nbits = data[:, 1]

        chunk, self.buf, self.cnt = _pack_bulk_jit(vals, nbits, self.buf, self.cnt)

        chunk = chunk.tobytes()
        if b"\xff" in chunk:
            chunk = chunk.replace(b"\xff", b"\xff\x00")

        self.fp.write(chunk)

    def flush(self):
        if self.cnt > 0:
            pad = 8 - self.cnt
            self.buf = (self.buf << pad) | ((1 << pad) - 1)
            byte = self.buf & 0xFF

            self.out_buffer.append(byte)
            if byte == 0xFF:
                self.out_buffer.append(0x00)

            self.cnt = 0
            self.buf = 0

        self._flush_buffer_to_disk()

    def _flush_buffer_to_disk(self):
        if self.out_buffer:
            self.fp.write(self.out_buffer)
            self.out_buffer.clear()


@jit(nopython=True)
def _pack_bulk_jit(vals, nbits, buf, cnt):
    n_items = len(vals)

    max_bytes = n_items * 2 + 16
    out = np.empty(max_bytes, dtype=np.uint8)
    idx = 0

    for i in range(n_items):
        v = vals[i]
        n = nbits[i]

        buf = (buf << n) | v
        cnt += n

        while cnt >= 8:
            cnt -= 8
            out[idx] = (buf >> cnt) & 0xFF
            idx += 1

    return out[:idx], buf, cnt
