cimport cython
cimport libc.math as math
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def linear_filter(double[:, ::1] img, double[:, ::1] kernel):
    """
    Linear convolution filter

    :param img: spot cut from input image
    :type img: numpy array
    :param kernel_size: size of kernel matrix
    :type kernel_size: int
    """

    cdef int kh = kernel.shape[0]
    cdef int kw = kernel.shape[1]
    cdef double kh_ = <double>kh
    cdef double kw_ = <double>kw
    cdef double margin_ = (math.fmax(kh, kw) - 1) / 2.0
    cdef int margin = <int>margin_
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int ii, jj, kk, ll, l, k
    cdef double tot

    padded = np.zeros(
        (img.shape[0] + 2 * margin,
         img.shape[1] + 2 * margin)
    )
    cdef double [:, ::1] padded_view = padded
    
    filtered = np.zeros_like(img)
    cdef double [:, ::1] filtered_view = filtered

    padded_view[margin:-margin, margin:-margin] = img
    for ii in range(height):
        for jj in range(width):
            tot = 0.0
            for k, kk in enumerate(range(ii, ii + kh)):
                for l, ll in enumerate(range(jj, jj + kw)):
                    tot += padded_view[kk, ll] * kernel[k, l]

            filtered_view[ii, jj] = tot
    return filtered

@cython.boundscheck(False)
@cython.wraparound(False)
def maximum_filter(double[:, ::1] img, int kernel_size):
    """
    Non linear convolution filter

    :param img: spot cut from input image
    :type img: numpy array
    :param kernel_size: size of kernel matrix
    :type kernel_size: int
    """
    cdef int kh = kernel_size
    cdef int kw = kernel_size
    cdef double kh_ = <double>kh
    cdef double kw_ = <double>kw
    cdef double margin_ = (math.fmax(kh_, kw_) - 1) / 2.0
    cdef int margin = <int>margin_
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int ii, jj, kk, ll
    cdef double max_val, val

    padded = np.zeros(
        (img.shape[0] + 2 * margin,
         img.shape[1] + 2 * margin)
    )
    cdef double [:, ::1] padded_view = padded
    
    filtered = np.zeros_like(img)
    cdef double [:, ::1] filtered_view = filtered

    padded_view[margin:-margin, margin:-margin] = img
   
    for ii in range(height):
        for jj in range(width):
            max_val = 0.0
            for kk in range(ii, ii + kh):
                for ll in range(jj, jj + kw):
                    val = padded_view[kk, ll]
                    max_val = math.fmax(max_val, val)
            filtered_view[ii, jj] = max_val
    return filtered

