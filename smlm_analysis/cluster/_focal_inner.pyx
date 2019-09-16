cimport cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _density(int[::1] i_x, int[::1] i_y, int[:,::1] grid):

    cdef:
        int i, j, m, n
        int xsize = grid.shape[1]
        int ysize = grid.shape[0]
        int[:, ::1] density = np.zeros((ysize, xsize), dtype=np.int32)

    for i in range(1, ysize - 1):
        for j in range(1, xsize - 1):
            for m in range(-1, 2):
                for n in range(-1, 2):
                    density[i, j] += grid[i + m, j + n]

    return density

@cython.boundscheck(False)
@cython.wraparound(False)
def density_map(int[::1] i_x, int[::1] i_y, int[:,::1] grid, int minL):

    cdef:
        int i, j, m, n
        int xsize = grid.shape[1]
        int ysize = grid.shape[0]
        int[:, ::1] density = np.zeros((ysize, xsize), dtype=np.int32)
        int[:, ::1] core_map = np.zeros((ysize, xsize), dtype=np.int32)

    density = _density(i_x, i_y, grid)

    for i in range(i_x.shape[0]):
        if density[i_y[i], i_x[i]] > minL:
            core_map[i_y[i], i_x[i]] = 1

    return np.array(core_map)


@cython.boundscheck(False)
@cython.wraparound(False)
def density_map_at_minl(int[::1] i_x, int[::1] i_y, int[:,::1] grid, int minL):

    cdef:
        int i, j, m, n
        int xsize = grid.shape[1]
        int ysize = grid.shape[0]
        int[:, ::1] density = np.zeros((ysize, xsize), dtype=np.int32)
        int[:, ::1] core_map = np.zeros((ysize, xsize), dtype=np.int32)

    density = _density(i_x, i_y, grid)

    for i in range(i_x.shape[0]):
        if density[i_y[i], i_x[i]] == minL:
            core_map[i_y[i], i_x[i]] = 1

    return np.array(core_map)