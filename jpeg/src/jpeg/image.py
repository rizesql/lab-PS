from typing import Generic, Literal, TypeVar, overload

import numpy as np

from jpeg.tables import BLOCK


class RGB: ...


class YCbCr: ...


ColorChan = TypeVar("ColorChan")


class Image(np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.uint8]], Generic[ColorChan]): ...


class Chan(np.ndarray[tuple[int, int], np.dtype[np.uint8]], Generic[ColorChan]): ...


type ImageLike[T] = Image[T] | Chan[T]


def pad[T: ImageLike](img: T, block_size=BLOCK) -> T:
    H, W = img.shape[:2]
    pad_h, pad_w = (-H) % block_size, (-W) % block_size

    padding = [(0, pad_h), (0, pad_w)]
    padding.extend([(0, 0)] * (img.ndim - 2))

    return np.pad(img, padding, mode="edge").view(img.__class__)  # ty:ignore[invalid-return-type]


def split_chans[T](img: Image[T]) -> tuple[Chan[T], Chan[T], Chan[T]]:
    return (
        img[..., 0].view(Chan),
        img[..., 1].view(Chan),
        img[..., 2].view(Chan),
    )  # ty:ignore[invalid-return-type]


def subsample[T](chan: Chan[T], factor=2) -> Chan[T]:
    H, W = chan.shape
    return chan.reshape(H // factor, factor, W // factor, factor).mean(axis=(1, 3)).astype(np.uint8)


class Block(
    np.ndarray[tuple[int, int, int, int, Literal[3]], np.dtype[np.uint8]],
    Generic[ColorChan],
): ...


class ChanBlock(
    np.ndarray[tuple[int, int, int, int], np.dtype[np.uint8]],
    Generic[ColorChan],
): ...


@overload
def blockwise[T, B: int](img: Image[T], block_size: B) -> Block[T]: ...
@overload
def blockwise[T, B: int](img: Chan[T], block_size: B) -> ChanBlock[T]: ...
def blockwise[B: int](img, block_size: B):
    H, W = img.shape[:2]

    shape = (H // block_size, block_size, W // block_size, block_size, *img.shape[2:])

    return img.reshape(shape).swapaxes(1, 2)


_ycbcr_coeffs = np.array(
    [
        [0.299, -0.168736, 0.5],
        [0.587, -0.331264, -0.418688],
        [0.114, 0.5, -0.081312],
    ],
    dtype=np.float32,
)
_ycbcr_shift = np.array([0, 128, 128], dtype=np.float32)


def rgb_to_ycbcr(src: Image[RGB]) -> Image[YCbCr]:
    ret = (src @ _ycbcr_coeffs) + _ycbcr_shift
    return ret.clip(0, 255).astype(np.uint8).view(Image)
