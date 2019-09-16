import math

import numpy as np
from numba import jit
from matplotlib import pyplot as plt

from . import convolution


class ConvolutionFilter:
    """
    Perform convolution filtering - 
    wrapper around Cythonised function
    """
    def __init__(self, kernel, method='linear'):
        self.create_kernel(kernel)
        self.method = method

    def create_kernel(self, kernel):
        kernelX = np.reshape(kernel,(1,kernel.shape[0]))
        kernelY = np.reshape(kernel,(kernel.shape[0],1))
        self.kernel = np.outer(kernelX, kernelY)

    def filter_image(self, image):
        if self.method == 'linear':
            return convolution.linear_filter(image, self.kernel)
        elif self.method == 'nonlinear_max':
            return convolution.maximum_filter(image, self.kernel.shape[0])


class WaveletFilter:
    """
    Abstraction of 'a-trous' wavelet filter -
    uses Haar or BSpline as kernel
    """
    def __init__(self, plane, spline_order, spline_scale, nsamples):
        self.plane = plane
        self.order = spline_order
        self.scale = spline_scale
        self.nsamples = nsamples
        self.kernel = self.get_kernel()
        self.mfilter = ConvolutionFilter(self.kernel)

    def get_kernel(self):
        nsamples = self.nsamples
        samples = np.array(range(nsamples))
        for i in range(nsamples):
            samples[i] = i - nsamples // 2

        plane = self.plane
        spline = self.bspline_blender(samples)
        if plane == 1:
            return spline
        else:
            step = 2**(plane-1)
            n = (step * (nsamples - 1)) + 1
            kernel = np.zeros(n)
            for i in range(spline.shape[0]):
                kernel[i*step] = spline[i]
            return kernel

    def filter_image(self, image):
        return self.mfilter.filter_image(image)


    def bspline_blender(self, samples):
        t = np.add(np.divide(samples, self.scale), float(self.order) / 2.0)
        return self.normalize(self.N(self.order, t))

    def N(self, k, t):
        if k <= 1:
            return self.haar(t)
        else:
            res = np.zeros((t.shape[0],))
            for i, el in enumerate(t):
                ti = np.array([t[i],])
                Nt = self.N(k - 1, ti)
                Nt_1 = self.N(k - 1, ti - 1)
                res[i] = ti / (k - 1) * Nt[0] + (k - ti) / (k - 1) * Nt_1[0]
            return res

    def haar(self, t):
        res = np.array(range(t.shape[0]))
        for i in range(t.shape[0]):
            if (t[i] >= 0) and (t[i] < 1):
                res[i] = 1.0
            else:
                res[i] = 0.0
        return res        

    def normalize(self, arr):
        return np.divide(arr, np.sum(arr))        


class UniformFilter:
    """
    Abstraction of a uniform filter -
    mainly for creating kernel
    """
    def __init__(self, nsamples):
        self.nsamples = nsamples
        self.kernel = self.get_kernel()
        self.mfilter = ConvolutionFilter(self.kernel)

    def get_kernel(self):
        nsamples = self.nsamples
        return np.ones(nsamples) / nsamples

    def filter_image(self, image):
        return self.mfilter.filter_image(image)


class MaximumFilter:
    """
    Abstraction of a maximum filter -
    mainly for creating kernel
    """
    def __init__(self, nsamples):
        self.nsamples = nsamples
        self.kernel = self.get_kernel()
        self.mfilter = ConvolutionFilter(self.kernel, method='nonlinear_max')

    def get_kernel(self):
        nsamples = self.nsamples
        return np.ones(nsamples)

    def filter_image(self, image):
        return self.mfilter.filter_image(image)


class CompoundWaveletFilter:
    """
    Compund Wavelet filter -
    Filter with Haar, then BSpline and subtract
    """
    def __init__(self, spline_order, spline_scale):
        order = float(spline_order)
        scale = float(spline_scale)
        self.nsamples = int(2.0 * math.ceil(order * scale / 2.0) - 1.0)
        self.f1 = WaveletFilter(1, order, scale, self.nsamples)
        self.f2 = WaveletFilter(2, order, scale, self.nsamples)

    def filter_image(self, image):
        v1 = self.f1.filter_image(image)
        v2 = self.f2.filter_image(v1)        
        self.result_f1 = np.subtract(image, v1)
        self.result_f2 = np.subtract(image, v2)
        self.result = np.subtract(v1, v2)
        return self.result        


class CompoundUniformFilter:
    """
    Compound uniform filter -
    Filter image with small uniform (mean)
    kernel then kernel with size 2*N+1 and
    subtract
    """
    def __init__(self, nsamples):
        n1 = nsamples
        n2 = 2 * nsamples + 1
        self.f1 = UniformFilter(n1)
        self.f2 = UniformFilter(n2)

    def filter_image(self, image):
        v1 = self.f1.filter_image(image)
        v2 = self.f2.filter_image(image)
        self.result_f1 = np.subtract(image, v1)
        self.result = np.subtract(v1, v2)
        return self.result



if __name__=="__main__":
    fpath = ".\\test_data\\z_7448_647_Calib.nd2"
    import pims
    from nd2reader import ND2Reader
    from tifffile import imsave
    movie = pims.open(fpath)
    # movie.bundle_axes = 'cyx'
    frame = movie[100].astype(np.float64)
    cwf = CompoundWaveletFilter(3, 2.0)
    filtered = cwf.filter_image(frame)

    cuf = CompoundUniformFilter(3)
    ufilt = cuf.filter_image(frame)

    mf = MaximumFilter(7)
    mfilt = mf.filter_image(ufilt)
    mask = ufilt == mfilt


    from matplotlib import pyplot as plt
    # kernelX = np.reshape(cwf.f1.kernel, (1, cwf.f1.kernel.shape[0]))
    # kernelY = np.reshape(cwf.f1.kernel, (cwf.f1.kernel.shape[0], 1))
    # kernel = np.outer(kernelX, kernelY)
    # plt.figure()
    # plt.imshow(kernel)
    plt.figure()    
    plt.imshow(ufilt)
    plt.show()
    plt.imshow(mfilt)
    plt.show()