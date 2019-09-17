import numba as nb
from numba import jit
import numpy as np
import time

DRAW_MAX_SIGMA = 3


def image_2d(x, y, binsx, binsy, bin_size):
    im = np.zeros((binsy, binsx), dtype=np.int32)
    i_x = np.floor(x / bin_size).astype(np.int32)
    i_y = np.floor(y / bin_size).astype(np.int32)
    n = x.shape[0]
    hist_2d(i_x, i_y, n, im)
    return im

    
@jit(nopython=True, nogil=True)
def hist_2d(i_x, i_y, n, grid):

    for i in range(n):
        x = i_x[i]
        y = i_y[i]
        grid[y, x] += 1

@jit(nopython=True, nogil=True)
def hist_3d(i_x, i_y, i_z, n, grid):

    for i in range(n):
        x = i_x[i]
        y = i_y[i]
        z = i_z[i]
        grid[y, x, z] += 1


@jit(nopython=True, nogil=True)
def gaussian_2d(x, y, px, py, binsx, binsy, bin_size, min_blur_width=0):

    image = np.zeros((binsy, binsx), dtype=np.int32)
    x = x / bin_size
    y = y / bin_size
    print(min_blur_width / bin_size)
    blur_width = np.maximum(px / bin_size, min_blur_width)
    print(blur_width)
    blur_height = np.maximum(py / bin_size, min_blur_width)
    sy = (blur_height + blur_width) / 2
    sx = sy
    for x_, y_, sx_, sy_ in zip(x, y, sx, sy):
        max_y = DRAW_MAX_SIGMA * sy_
        i_min = np.int32(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = np.int32(y_ + max_y + 1)
        if i_max > binsy:
            i_max = binsy
        max_x = DRAW_MAX_SIGMA * sx_
        j_min = np.int32(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = np.int32(x_ + max_x) + 1
        if j_max > binsx:
            j_max = binsx
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                image[i, j] += np.exp(
                    -(
                        (j - x_ + 0.5) ** 2 / (2 * sx_ ** 2)
                        + (i - y_ + 0.5) ** 2 / (2 * sy_ ** 2)
                    )
                ) / (2 * np.pi * sx_ * sy_)
    return image
