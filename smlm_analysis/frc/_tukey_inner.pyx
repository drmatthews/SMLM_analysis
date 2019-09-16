cimport cython
from libc.math cimport sqrt, cos, trunc
from libc.math cimport pi as PI
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef ctukey(double alpha, double N):
    cdef:
        double[::1] w = np.zeros(<int>N)
        double alphaN = (alpha * (N - 1)) / 2
        double n
        int i

    for i in range(<int>N):
        n = <double>i
        if n >= 0.0 and n < alphaN:
            w[i] = (1.0 + cos(PI*((n / alphaN) - 1))) / 2.0
        elif n >= alphaN and n <= (N - 1) * (1 - alpha / 2.0):
            w[i] = 1.0
        elif (n > (N - 1) * (1 - alpha / 2.0) and n < (N - 1)):
            w[i] = (1.0 + cos(PI*((n / alphaN) - (2 / alpha) + 1))) / 2.0

    return w

@cython.boundscheck(False)
@cython.wraparound(False)
def tukey_2d(double alpha, int width, double[:, ::1] base):

    cdef:
        int radius, t, i, j
        double w = <double>width
        double[::1] x = np.linspace(-w/2.0, w/2.0, width)
        double[::1] y = np.linspace(-w/2.0, w/2.0, width)
        double[::1] tuk = np.zeros(width)
        double[::1] tuk_half = np.zeros(<int>(w / 2.0))
        int s = (width / 2) - 1

    tuk = ctukey(alpha, w)
    for t in range(width/2):
        tuk_half[t] = tuk[s]
        s += 1

    for i in range(width):
        for j in range(width):
            radius = <int>trunc(sqrt((x[j] * x[j]) + (y[i] * y[i])))
            if radius <= (width / 2):
                base[i, j] = tuk_half[radius]