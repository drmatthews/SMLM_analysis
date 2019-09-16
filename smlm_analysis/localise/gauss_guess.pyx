cimport cython
cimport libc.math as math
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def guess_sigma(double[:, ::1] img, int size):
    
    cdef:
        double size_ = <double>size
        double size_half_ = size_/ 2.0
        int size_half = <int>size_half_
        double sum_deviation_y = 0.0
        double sum_deviation_x = 0.0
        double sum_y = 0.0
        double sum_x = 0.0
        int i, d2
        double sx, sy

    for i in range(size):
        d2 = (i - size_half) * (i - size_half)
        sum_deviation_y += img[i, size_half] * <double>d2
        sum_deviation_x += img[size_half, i] * <double>d2
        sum_y += img[i, size_half]
        sum_x += img[size_half, i]
    sy = math.sqrt(sum_deviation_y / sum_y)
    sx = math.sqrt(sum_deviation_x / sum_x)

    if not math.isfinite(sy):
        sy = 0.01
    if not math.isfinite(sx):
        sx = 0.01
    return sy, sx


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sum_and_center_of_mass(double[:, ::1] img, int size):
    cdef:
        double x = 0.0
        double y = 0.0
        double tot = 0.0
        int i, j

    for i in range(size):
        for j in range(size):
            x += img[i, j] * i
            y += img[i, j] * j
            tot += img[i, j]
    x /= tot
    y /= tot
    return tot, y, x