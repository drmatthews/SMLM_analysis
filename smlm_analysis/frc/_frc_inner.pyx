  # distutils: language = c++

cimport cython
from libcpp.vector cimport vector
from libc.math cimport sqrt
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_frc(double complex[:,::1] im1_fft, double complex[:,::1] im2_fft):

    cdef:
        int sizex = im1_fft.shape[0]
        int sizey = im1_fft.shape[1]
        int cx = sizex / 2
        int cy = sizey / 2
        int dx, dy, i, j, q
        double g1, g2, g1g2
        double[:, ::1] im1_fft_sqr = np.ascontiguousarray(np.real(im1_fft * np.conj(im1_fft)))
        double[:, ::1] im2_fft_sqr = np.ascontiguousarray(np.real(im2_fft * np.conj(im2_fft)))
        double[:, ::1] im1_im2 = np.ascontiguousarray(np.real(im1_fft * np.conj(im2_fft)))

        double[::1] frc = np.zeros(sizex, dtype=np.float64)
        int[::1] frc_counts = np.zeros(sizex, dtype=np.int32)

    for i in range(sizey):
        dy = i - cy
        for j in range(sizex):
            dx = j - cx
            q = <int>(sqrt(dx*dx + dy*dy) + 0.5)
            g1 = im1_fft_sqr[i, j]
            g2 = im2_fft_sqr[i, j]
            g1g2 = im1_im2[i, j]
            frc[q] += (g1g2 / sqrt(g1*g2))
            frc_counts[q] += 1

    max_q = <int>(min([sizex, sizey])/2)
    frc = frc[:max_q]
    frc_counts[:max_q]

    return frc, frc_counts