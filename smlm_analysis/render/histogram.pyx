cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport exp, pi

@cython.boundscheck(False)
@cython.wraparound(False)
def hist_2d(int[::1] i_x, int[::1] i_y, int n, int[:,::1] grid):

    cdef int i, j, x, y

    for i in range(n):
        x = i_x[i]
        y = i_y[i]
        grid[y, x] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
def hist_3d(int[::1] i_x, int[::1] i_y, int[::1] i_z, int n, int[:,:,::1] grid):

    cdef int i, j, x, y, z

    for i in range(n):
        x = i_x[i]
        y = i_y[i]
        z = i_z[i]
        grid[y, x, z] += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_2d(double[::1] x, double[::1] y,
                double[::1] sx, double[::1] sy,
                int binsx, int binsy, double[:,::1] grid):
    cdef:
        int DRAW_MAX_SIGMA = 3
        double x_, y_, sx_, sy_, amp, ii, jj, xx_, yy_
        int i, j, i_min, i_max, j_min, j_max

    for x_, y_, sx_, sy_ in zip(x, y, sx, sy):
        max_y = DRAW_MAX_SIGMA * sy_
        i_min = <int>(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = <int>(y_ + max_y + 1)
        if i_max > binsy:
            i_max = binsy
        max_x = DRAW_MAX_SIGMA * sx_
        j_min = <int>(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = <int>(x_ + max_x) + 1
        if j_max > binsx:
            j_max = binsx
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                ii = <double>i
                jj = <double>j
                xx_ = <double>x_
                yy_ = <double>y_
                amp = exp(
                    -(
                        (jj - xx_ + 0.5) ** 2 / (2 * sx_ ** 2)
                        + (ii - yy_ + 0.5) ** 2 / (2 * sy_ ** 2)
                    )
                ) / (2 * pi * sx_ * sy_)
                grid[i, j] += amp