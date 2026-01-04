from typing import BinaryIO

import numpy as np
from numba import njit


class StreamingBitWriter:
    def __init__(self, fp: BinaryIO, buffer_size: int = 65536):
        self.fp = fp
        self.buf = 0
        self.cnt = 0

        self.out_buffer = bytearray()
        self.buffer_size = buffer_size

    def write(self, data: np.ndarray):
        if len(data) == 0:
            return

        if self.out_buffer:
            self._flush_out_buffer()

        vals, nbits = data[:, 0], data[:, 1]

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

        self._flush_out_buffer()

    def _flush_out_buffer(self):
        if self.out_buffer:
            self.fp.write(self.out_buffer)
            self.out_buffer.clear()


@njit()
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
