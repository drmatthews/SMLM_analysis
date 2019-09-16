import os

import numpy as np
import scipy
from scipy.optimize import leastsq

from matplotlib import pyplot as plt

from ..utils.parameters import SMLMParameters
from . import gauss_guess
from . import convolution
from .mle import mle

"""
Notes:

Fitting modules are incomplete. LS fitting needs work:
1. Integrated fitting (using erf) is not working (is photon conversion required?).
2. The code probably needs re-writing. A PSF (currently Peak) class is required. Keep
details of symmetric and ellipcatical models there?
3. Need to look at background estimation - is a variance map required for NSTORM5 data
(since a CMOS camera is used for acquisition)?
"""
    
class Fitter:
    def __init__(self, tol, max_iter):
        self.tol = tol
        self.max_iter = max_iter

    def initialise(self, img):
        tot, y, x = self.sum_and_center_of_mass(img)
        bg = np.min(self.mean_filter(img))     
        photons = tot - self.size * self.size * bg
        photons_sane = np.maximum(1.0, photons)
        sy, sx = self.guess_sigma(img - bg)
        return x, y, photons_sane, bg, sx, sy

    def initial_guess(self, img):
        # subclasses should override this
        pass

    def guess_sigma(self, img):
        return gauss_guess.guess_sigma(img, self.size)

    def sum_and_center_of_mass(self, img):
        return gauss_guess.sum_and_center_of_mass(img, self.size)

    def mean_filter(self, img):
        uniform = np.ones((3,3)) * (1/9)
        return convolution.linear_filter(img, uniform)    


class GaussMLEFitter(Fitter):
    """
    Base class for MLE. Methods wrap Cythonized functions
    """
    
    def fit(self, img):
        self.size, _ = img.shape
        guess = self.initial_guess(img)
        mle_params, likelihood, CRLB, iterations = (
            mle(img, self.size, guess, self.max_iter, self.tol)
        )
        self.fit_params = mle_params
        self.likelihood = likelihood
        self.CRLB = CRLB
        self.iterations = iterations
        self.fit_params = np.append(self.fit_params, [iterations])
        self.good = True
        if iterations == 1000:
            self.good = False
        self.fit_params = np.append(self.fit_params, [self.good])


class MLECircular(GaussMLEFitter):

    def initial_guess(self, img):
        guess = np.zeros(5, dtype=np.float64)
        guess[0], guess[1], guess[2], guess[3], sx, sy = self.initialise(img)
        guess[4] = (sx + sy) / 2
        return guess


class MLEElliptical(GaussMLEFitter):

    def initial_guess(self, img):
        guess = np.zeros(6, dtype=np.float64)
        guess[0], guess[1], guess[2], guess[3], guess[4], guess[5] = self.initialise(img)
        return guess


class GaussLSFitter(Fitter):
    def check_fit(self, params, infodict, success):
        good = True
        # params are [x, y, N, bg, S]
        if (success < 1) or (success > 4):
            good = False

        # if x or y are -ve the fit is bad
        if params[0] < 0.0 or params[1] < 0.0:
            good = False

        # N is -ve fit is bad
        if params[2] < 0.0:
            good is False

        # if bg is -ve fit is bad
        if params[3] < 0.0:
            good = False

        if params[2] / params[3] < 0.3:
            good = False

        # if sx is too small or too wide fit is bad
        if params[4] < 0.0 or params[4] > 5.0:
            good = False

        if params.shape[0] == 6:
            # if sy is too small or too wide fit is bad
            if params[5] < 0.0 or params[5] > 5.0:
                good = False        
        self.good = good
        self.iterations = infodict['nfev']

        self.fit_params = np.append(self.fit_params, [self.iterations])
        self.fit_params = np.append(self.fit_params, good)


    def fit(self, data):
        """
        Non-weighted least squares fit -
        Levenburg-Marquadt
        """
        self.size, _ = data.shape
        params = self.initial_guess(data)
        result = params
        fn = self.fit_func
        errorfunction = lambda p: np.ravel(fn(*p)(*np.indices(data.shape)) - data)
        [result, cov_x, infodict, mesg, success] = (
            leastsq(
                errorfunction, params,
                full_output=1, maxfev=self.max_iter,
                ftol=1e-2, xtol=1e-2
            )
        )
        self.fit_params = result
        err = errorfunction(result)
        err = scipy.sum(err * err)

        self.check_fit(result, infodict, success)


class LSElliptical(GaussLSFitter):   

    def initial_guess(self, img):
        guess = np.zeros(6, dtype=np.float64)
        guess[0], guess[1], guess[2], guess[3], guess[4], guess[5] = self.initialise(img)
        return guess

    def fit_func(self, x0, y0, p, bg, sx, sy):
        wx = 2.0 * sx
        wy = 2.0 * sy
        return lambda x,y: bg + p * np.exp(-(((x - x0) / wx)**2 + ((y - y0) / wy)**2) * 2)


class LSCircular(GaussLSFitter):    

    def initial_guess(self, img):
        guess = np.zeros(5, dtype=np.float64)
        guess[0], guess[1], guess[2], guess[3], sx, sy = self.initialise(img)
        guess[4] = (sx + sy) / 2
        return guess

    def fit_func(self, x0, y0, p, bg, w):
        """
        Symmetric Gaussian fit
        """
        return lambda x,y: bg + p * np.exp(-(((x - x0) / w)**2 + ((y - y0) / w)**2) * 2)

    # def fit_func(self, x0, y0, N, bg, s):
    #     """
    #     Integrated symmetric gaussian
    #     This is not working at present - last tried 230719
    #     """
    #     def f(x, y):
    #         # sqrt2s = math.sqrt(2) * s
    #         sqrt2s = math.sqrt(0.5 / (s**2))
    #         dx = x - x0
    #         dy = y - y0
    #         errdifx = erf((dx + 0.5) * sqrt2s) - erf((dx - 0.5) * sqrt2s)
    #         errdify = erf((dy + 0.5) * sqrt2s) - erf((dy - 0.5) * sqrt2s)
    #         # return value according to ThunderSTORM
    #         return bg + 0.25 * N * errdifx * errdify
    #     return f


# class GaussPeakFinder(PeakFinder):
#     def __init__(self, parameters):
#         super().__init__(parameters)

#     def _refine(self, candidates, frame):
#         method = self.params.iter_method
#         psf_shape = self.params.psf
#         max_iter = self.params.max_iter
#         tol = self.params.tolerance
#         is_3d = self.params.is_3d
#         camera_pixel = self.params.camera_pixel
#         photon_conversion = self.params.photon_conversion
#         window = self.params.window
#         fit_radius = int((window - 1) / 2.0)
#         peaks = []
#         for cand in candidates:
#             peak = GaussPeak(
#                 is_3d, window,
#                 frame.frame_no, camera_pixel,
#                 photon_conversion,
#                 method=method, psf_shape=psf_shape,
#                 max_iter=max_iter, tol=tol
#             )
            
#             # print(coord)
#             # cut the candidate out of the image
#             reference = cand - fit_radius
#             img = np.ascontiguousarray(
#                 frame[
#                     cand[0] - fit_radius: cand[0] + fit_radius + 1,
#                     cand[1] - fit_radius: cand[1] + fit_radius + 1
#             ])
#             peak.fit(img, reference)
#             peaks.append(peak)

#         return peaks

#     def locate(self, image):
#         candidates = self.find(image)
#         peaks = self._refine(candidates, image)
#         # print(time.time() - start)
#         return peaks


if __name__=='__main__':
    movie_path = ".\\test_data\\z_7448_647_Calib.nd2"
    # movie_path = ".\\test_data\\tubulin647 2d.nd2"
    import pims
    from nd2reader import ND2Reader
    from tifffile import imsave
    movie = pims.open(movie_path)
    # movie.bundle_axes = 'cyx'
    # print(movie[0].shape)
    # imsave(".\\test_data\\frame_10000.tif", movie[10000])
    smlm_params = SMLMParameters()
    smlm_params.frame_width = 512
    smlm_params.frame_height = 512
    smlm_params.camera_pixel = 160.0
    smlm_params.is_3d = False
    smlm_params.max_iter = 1000
    smlm_params.tolerance = 1e-6
    smlm_params.iter_method = 'ls'
    smlm_params.psf = 'circular'
    smlm_params.threshold_method = 'std'
    smlm_params.window = 5
    movie_dir = os.path.dirname(movie_path)
    movie_filename = os.path.basename(movie_path)
    basename = os.path.splitext(movie_filename)[0]
    smlm_params.locs_path = os.path.join(movie_dir, basename + '.h5')

    gpf = GaussPeakFinder(smlm_params)
    frame = movie[100].astype(np.float64)
    peaks = gpf.locate(frame)
    print(len(peaks))
    from matplotlib import pyplot as plt

    plt.figure()
    plt.imshow(frame)
    for peak in peaks:
        if peak.good:
            plt.plot(peak.y/160.0, peak.x/160.0, 'rx')
    plt.show()

    # p = PhasorPeak(7, 0)

