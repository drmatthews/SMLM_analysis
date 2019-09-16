import os
from optparse import OptionParser
import csv

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy
from scipy.optimize import leastsq
from numba import jit
from tifffile import imread

from . import convolution as conv

##
# helpers - jitted for speed-up
##
@jit(nopython=True)
def _guess_sigma(img, size):
    """
    Guess width of spot

    :param img: spot cut from input image
    :type img: numpy array
    :param size: size of the cut region
    :type size: int
    """
    size_half = int(size/2)
    sum_deviation_y = 0.0
    sum_deviation_x = 0.0
    sum_y = 0.0
    sum_x = 0.0
    for i in range(size):
        d2 = (i - size_half)**2
        sum_deviation_y += img[i, size_half] * d2
        sum_deviation_x += img[size_half, i] * d2
        sum_y += img[i, size_half]
        sum_x += img[size_half, i]
    sy = np.sqrt(sum_deviation_y / sum_y)
    sx = np.sqrt(sum_deviation_x / sum_x)
    if ~np.isfinite(sy):
        sy = 0.01
    if ~np.isfinite(sx):
        sx = 0.01
    return sy, sx

@jit(nopython=True)
def _sum_and_center_of_mass(img, size):
    """
    Estimate total intensity and center position
    of spot in cut region

    :param img: spot cut from input image
    :type img: numpy array
    :param size: size of the cut region
    :type size: int    
    """
    x = 0.0
    y = 0.0
    tot = 0.0
    for i in range(size):
        for j in range(size):
            x += img[i, j] * i
            y += img[i, j] * j
            tot += img[i, j]
    x /= tot
    y /= tot
    return tot, y, x

@jit(nopython=True)
def _uniform(kernel_size):
    """
    Kernel for uniform convolution filter
    :param kernel_size: size of kernel matrix
    :type kernel_size: int
    """
    kernel = np.ones((kernel_size, kernel_size))
    kh, kw = kernel.shape
    kernel = kernel / (kh * kw)
    return kernel

@jit(nopython=True)
def _convolution_filter(img, kernel):
    """
    Linear convolution filter

    :param img: spot cut from input image
    :type img: numpy array
    :param kernel_size: size of kernel matrix
    :type kernel_size: int      
    """

    kh, kw = kernel.shape
    if kh < 3 or kh < 3:
        raise ValueError('Minimum size for kernel is 3 pixels')

    if kh > img.shape[0] or kw > img.shape[1]:
        raise ValueError('Kernel size exceeds image size')

    margin = int((max(kh, kw) - 1) / 2)
    padded = np.zeros(
        (img.shape[0] + 2 * margin,
         img.shape[1] + 2 * margin)
    )
    padded[margin:-margin, margin:-margin] = img
    filtered = np.zeros_like(img)

    height, width = img.shape
    for i in range(height):
        for j in range(width):
            filtered[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)

    return filtered

@jit(nopython=True)
def _nonlinear_filter(img, kernel_size, filter='maximum'):
    """
    Non linear convolution filter

    :param img: spot cut from input image
    :type img: numpy array
    :param kernel_size: size of kernel matrix
    :type kernel_size: int      
    :param filter: the type of filtering to perform
    :type filter: str
    """
    kh, kw = kernel_size, kernel_size
    if kh < 3 or kh < 3:
        raise ValueError('Minimum size for kernel is 3 pixels')

    if kh > img.shape[0] or kw > img.shape[1]:
        raise ValueError('Kernel size exceeds image size')
    
    margin = int((kernel_size - 1) / 2)
    padded = np.zeros((img.shape[0] + 2 * margin, img.shape[1] + 2 * margin))
    padded[margin:-margin, margin:-margin] = img
    filtered = np.zeros_like(img)

    height, width = img.shape    
    for i in range(height):
        for j in range(width):
            if filter == 'maximum':
                filtered[i, j] = np.max(padded[i:i+kh, j:j+kw])
            elif filter == 'minimum':
                filtered[i, j] = np.min(padded[i:i+kh, j:j+kw])
            elif filter == 'median':
                filtered[i, j] = np.median(padded[i:i+kh, j:j+kw])
            else:
                raise ValueError(
                    'Filter should be "maximum", "minimum" or "median"'
                )
    return filtered

def _smooth(img, kernel_size=3, filter='uniform'):
    """
    Smooth an input image using a convolution filter

    :param img: spot cut from input image
    :type img: numpy array
    :param kernel_size: size of kernel matrix
    :type kernel_size: int
    :param filter: the type of filtering to perform
    :type filter: str
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel size should be an odd number')

    if 'uniform' in filter:
        filtered = _convolution_filter(img, _uniform(kernel_size))
    elif 'maximum' in filter:
        filtered = _nonlinear_filter(img, kernel_size)
    return filtered

@jit(nopython=True)
def _distance_matrix(coords):
    rows = coords.shape[0]
    cols = coords.shape[1]
    dist_mat = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            dist = 0
            for k in range(cols):
                dist += (coords[i, k] - coords[j, k])**2

            dist_mat[i, j] = dist

    return np.sqrt(dist_mat)

@jit(nopython=True)
def _remove_overlaps(coords, min_dist):
    dist = _distance_matrix(coords)
    return np.where(dist[:, 1] >= min_dist)
##
# end helpers
##

class LSFitter:
    """
    Least squares fitter - uses scipy.optimize.leastsq
    """
    def __init__(self, max_iter):
        """
        constructor

        :param max_iter: maximumm number of iterations
        :type max_iter: int
        """      
        self.tol = 1e-4
        self.max_iter = max_iter
        self.iterations = 0
        self.fit_params = np.zeros((5,), dtype=np.float32)
        self.good = True

    def initial_guess(self, img):
        """
        Guess initial fit parameters [x,y,N,bg,sx,sy]

        :param img: spot cut from input image
        :type img: numpy array        
        """
        guess = np.zeros(6, dtype=np.float32)

        tot, y, x = self.sum_and_center_of_mass(img)
        bg = np.min(self.mean_filter(img))
        intensity = tot - self.size * self.size * bg
        intensity = np.maximum(1.0, intensity)
        sy, sx = self.guess_sigma(img - bg)

        guess[0] = x
        guess[1] = y
        guess[2] = intensity
        guess[3] = bg
        guess[4] = sx
        guess[5] = sy

        return guess

    def guess_sigma(self, img):
        """
        Guess width of spot.
        Wrapper for jitted funtion.

        :param img: spot cut from input image
        :type img: numpy array    
        """
        return _guess_sigma(img, self.size)

    def sum_and_center_of_mass(self, img):
        """
        Guess total intensity and center position of spot.
        Wrapper for jitted funtion.

        :param img: spot cut from input image
        :type img: numpy array
        """
        return _sum_and_center_of_mass(img, self.size)

    def mean_filter(self, img):
        """
        Filter the cut region
        Wrapper for jitted funtion.

        :param img: spot cut from input image
        :type img: numpy array
        """        
        uniform = np.ones((3,3)) * (1/9)
        return _convolution_filter(img, uniform)

    def check_fit(self, params, infodict, success):
        """
        Check with the least squared fitting worked

        :param params: parameters determined by fit
        :type params: list
        :param infodict: information returned by leastsq
        :type infodict: dict
        :param success: did the fit work
        :type success: int
        """
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
        if params[4] < 1.0 or params[4] > 1.8:
            good = False

        # if sy is too small or too wide fit is bad
        if params[5] < 1.0 or params[5] > 1.8:
            good = False

        # if ratio of sx:sy > 1.2 fit is bad (non-circular)
        # if (params[4] / params[5]) > 1.2 or (params[5] / params[4]) > 1.2:
        #     good = False

        self.good = good
        self.iterations = infodict['nfev']

        self.fit_params = np.append(self.fit_params, [self.iterations])
        self.fit_params = np.append(self.fit_params, good)

    def fit(self, data, fit_func):
        """
        Non-weighted least squares fit -
        Levenburg-Marquadt

        :param data: spot cut from input image
        :type data: numpy array
        :param fit_func: the fit model
        :type fit_func: function
        """
        self.size, _ = data.shape
        params = self.initial_guess(data)
        result = params
        errfunc = lambda p: np.ravel(fit_func(*p)(*np.indices(data.shape)) - data)
        [result, _, info, _, success] = (
            leastsq(errfunc, params, full_output=1, maxfev=self.max_iter)
        )
        self.fit_params = result
        err = errfunc(result)
        err = scipy.sum(err * err)

        self.check_fit(result, info, success)

class Molecule:
    """
    A representation of bright spot in the image
    """
    def __init__(self, frame):
        self.frame = frame
        self.x = None
        self.y = None
        self.sx = None
        self.sy = None
        self.N = None
        self.bg = None
        self.rect = None

    def model(self, x0, y0, N, bg, sx, sy):
        """
        2D elliptical Gaussian representation of spot

        :param x0: x-position of spot
        :type x0: float
        :param y0: y-position of spot
        :type y0: float
        :param N: intensity of spot
        :type bg: background intensity
        :param sx: width of spot in x-direction
        :type sx: float
        :param sy: width of spot in y-direction
        :type sy: float                        
        """
        wx = 2.0 * sx
        wy = 2.0 * sy
        return lambda x,y: bg + N * np.exp(-(((x - x0) / wx)**2 \
                           + ((y - y0) / wy)**2) * 2)

    def fit(self, img, reference, max_iter, window):
        """
        Fit the model to the spot

        :param img: spot cut from input image
        :type img: numpy array
        :param reference: reference position of spot in input image
        :type reference: numpy array
        :param max_iter: maximum number of iterations for fitting
        :type max_iter: int
        """
        gf = LSFitter(max_iter)
        gf.fit(img, self.model)
        fit_params = gf.fit_params
        self.x = fit_params[0] + reference[0]
        self.y = fit_params[1] + reference[1]
        self.N = fit_params[2]
        self.bg = fit_params[3]
        self.sx = fit_params[4]
        self.sy = fit_params[5]
        self.iterations = fit_params[-2]
        self.good = fit_params[-1]

        self.rect = [
            self.y - (window - 1) / 2,
            self.x - (window - 1) / 2,
            (window - 1),
            (window - 1)
        ]

def save_molecules(molecule_path, molecules):
    with open(molecule_path, 'w', newline='') as csvfile:
        mol_writer = csv.writer(csvfile, delimiter=',')
        
        for mol in molecules:
            mol_writer.writerow(
                [mol.x, mol.y,
                 mol.N, mol.bg,
                 mol.sx, mol.sy,
                 mol.frame]
            )

def locate(image, threshold, window):
    """
    Use a uniform filter to do local background subtraction
    as in Huang et al Biomed. Optics Express
    Vol. 2, Issue 5, pp. 1377-1393 (2011)

    The initial positions are then refined by fitting a
    2D Elliptical Gaussian model to a region cut out of
    the input image. Fixed kernel sizes are used for 
    background subtraction and maximum finding.

    :param image: input microscope image
    :type image: numpy array
    :param threshold: intensity for spot identification
    :type threshold: float or None
    """
    # background subtract
    ufilt = (
        _smooth(image, 3, filter='uniform') - \
        _smooth(image, 7, filter='uniform')
    )
    plt.figure()
    plt.imshow(ufilt)
    plt.show()
    # find maxima
    mfilt = _smooth(ufilt, 7, filter='maximum')
    mask = ufilt == mfilt

    # retain features above threshold
    if threshold is not None:
            mask &= ufilt > threshold
    else:
        mask &= ufilt > 2 * np.std(ufilt)

    coords = np.nonzero(mask)
    if np.any(coords):
        # remove the ones near the edge
        # of the image
        coords = np.column_stack(coords)
        coords = coords[::-1]
        shape = np.array(image.shape)
        margin = [10, 10]
        near_edge = np.any(
            (coords < margin) | (coords > (shape - margin - 1)), 1
        )
        coords = coords[~near_edge]
        # remove any that are overlapping
        keep = _remove_overlaps(coords, window / 2)
        return coords[keep]
        # return coords
    else:
        return np.array([])

def find_spots(image, frame, threshold, window,
                  max_iter, show_plot=False):
    # identify spots
    spots = locate(image, threshold, window)

    # if there are any spots, refine their positions
    if np.any(spots):
        molecules = []
        for spot in spots:
            mol = Molecule(frame)
            fit_radius = int((window - 1) / 2.0)
            reference = spot - fit_radius
            roi = image[
                spot[0] - fit_radius: spot[0] + fit_radius + 1,
                spot[1] - fit_radius: spot[1] + fit_radius + 1
            ]
            mol.fit(roi, reference, max_iter, window)
            if mol.good:
                molecules.append(mol)

    if molecules and show_plot:
        figure, ax = plt.subplots(1)
        # ax.plot(spots[:,1], spots[:,0], 'rx')
        ax.imshow(image)
        for mol in molecules:
            rect = Rectangle(
                (mol.rect[0], mol.rect[1]),
                mol.rect[2],
                mol.rect[3],
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
        plt.show()

    return molecules

def run_analysis(filepath, threshold, window, max_iter):

    if os.path.exists and filepath.endswith('tif'):
        # read image
        input_img = imread(filepath)
        if input_img.ndim == 3:
            num_frames = input_img.shape[0]

            molecules = []
            for frame in range(num_frames):
                image = input_img[frame, :, :].astype(np.float32)
                
                molecules += find_spots(image, frame, threshold,
                                        window, max_iter)
        else:
            image = input_img.astype(np.float32)
            frame = 0
            molecules = find_spots(image, frame, threshold,
                                   window, max_iter, show_plot=True)            
        if molecules:
            # write them to file
            basename = os.path.splitext(filepath)[0]
            molecule_path = basename + '.csv'
            save_molecules(molecule_path, molecules)

def test_movie_frame(filepath, frame, threshold, window, max_iter):
    if os.path.exists and filepath.endswith('tif'):
        # ideally we wouldn't read the entire movie
        input_img = imread(filepath)

        if input_img.ndim == 3:
            try:
                image = input_img[frame, :, :].astype(np.float32)
                molecules = find_spots(image, frame, threshold, window,
                                       max_iter, show_plot=True)
            except:
                print('Frame not found in movie')
        else:
            raise ValueError('Can only run test on movie')

if __name__=='__main__':

    parser = OptionParser(
        usage='Usage: %prog [options] <challenge>'
    )
    parser.add_option(
        '-t', '--threshold',
        metavar='THRESHOLD', dest='threshold',
        type='int', default=None,
        help=('Provide a threshold for spot finding.'
             '2*std(filtered_image) is used otherwise.')
    )
    parser.add_option(
        '-w', '--window',
        metavar='WINDOW', dest='window',
        type='int', default=7,
        help=('number of pixels for 2D curve fitting region')
    )

    parser.add_option(
        '-i', '--iterations',
        metavar='ITERATIONS', dest='max_iter',
        type='int', default=500,
        help=('Max number of iterations for curve fitting')
    )

    parser.add_option(
        '-f', '--test-frame',
        metavar='TEST-FRAME', dest='frame',
        type='int', default=None,
        help=('Frame number in movie to test')
    )
    
    (opts, args) = parser.parse_args()
    try:
        filepath = args[0]
        if opts.frame is not None:
            test_movie_frame(filepath, opts.frame, opts.threshold, opts.window, opts.max_iter)
        else:
            run_analysis(filepath, opts.threshold, opts.window, opts.max_iter)
    except IndexError:
        parser.error('Could not process file - check inputs')
