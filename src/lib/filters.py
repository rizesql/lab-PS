import numpy as np


def mean(im, kernel_size=3):
    offset = kernel_size // 2
    window_shape = (kernel_size, kernel_size)

    im_padded = np.pad(im, pad_width=offset, mode="constant", constant_values=0)

    kernel = np.ones(window_shape) / (kernel_size**2)
    windows = np.lib.stride_tricks.sliding_window_view(im_padded, window_shape=window_shape)

    return np.sum(windows * kernel, axis=(-1, -2))


def low_pass(im, radius=50):
    row, col = im.shape
    y, x = np.ogrid[:row, :col]

    dist = (x - (col // 2)) ** 2 + (y - (row // 2)) ** 2

    F = np.fft.fft2(im)
    F_filtered = np.fft.fftshift(F)
    F_filtered[dist > radius**2] = 0

    F_ishift = np.fft.ifftshift(F_filtered)
    return np.abs(np.fft.ifft2(F_ishift))


def grid_search(noisy_img, clean_img, filter, param, search_range, metric):
    best_score = -float("inf")
    best_val = None
    best_img = None

    history = []

    for val in search_range:
        kwargs = {param: val}

        temp_img = filter(noisy_img, **kwargs)
        score = metric(clean_img, temp_img)

        if score > best_score:
            best_score = score
            best_val = val
            best_img = temp_img

        history.append((val, score))

    history_arr = np.array(history, dtype=[("vals", np.float64), ("scores", np.float64)])

    return best_val, best_img, history_arr
