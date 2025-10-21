import numpy as np
import matplotlib.pyplot as plt

type Coord = tuple[int, int]


def neighbours(src: np.ndarray, coord: Coord):
    rows, cols = src.shape

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for offset in offsets:
        x = coord[0] + offset[0]
        y = coord[1] + offset[1]

        if 0 <= x < rows and 0 <= y < cols:
            yield src[x, y]


def game_of_life(src: np.ndarray):
    next_gen = src.copy()

    for idx, cell in np.ndenumerate(src):
        alive_neighbours = sum(neighbours(src, idx))
        next_gen[idx] = 1 if alive_neighbours == 3 or (cell == 1 and alive_neighbours == 2) else 0

    return next_gen


fig, ax = plt.subplots(2, 1, figsize=(4, 8))

gen = np.random.randint(0, 2, size=(16, 16))
ax[0].set_title("Gen 0")
ax[0].imshow(gen, cmap="binary")

gen = game_of_life(gen)
ax[1].set_title("Gen 1")
ax[1].imshow(gen, cmap="binary")

plt.tight_layout()
plt.savefig("src/01/02f.pdf", format="pdf")
plt.show()
