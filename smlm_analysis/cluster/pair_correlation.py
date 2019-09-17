import numpy as np
from math import pi,floor
import scipy.optimize
from scipy.fftpack import fft2,ifft2,fftshift
from scipy.spatial.distance import cdist


# Least Squares fitting
def fit_function_LS(data, params, x, fn):
    result = params
    errorfunction = lambda p: fn(*p)(x) - data # noqa
    good = True
    [result, cov_x, infodict, mesg, success] = (
        scipy.optimize.leastsq(
            errorfunction, params, full_output=1, maxfev=500
        )
    )
#     result = (
#         scipy.optimize.least_squares(
#             errorfunction, params, bounds=(0, np.inf)
#         )
#     )
#     err = errorfunction(result.x)
    err = errorfunction(result)
    err = scipy.sum(err * err)
#     if (result.success < 1) or (result.success > 4):
    if (success < 1) or (success > 4):
        print("Fitting problem!", success, mesg)
        good = False
    return [result, cov_x, infodict, good]
#     return result


def psf(a, b):
    amp = 1 / 4.0 / pi / a**2 / b
    return lambda x: 1 + amp * np.exp(-x**2 / 4.0 / a**2)


def exponential(a, b):
    return lambda x: 1 + a * np.exp(-x / b)


def exponential_cosine(a, b, c, d):
    return lambda x: a + b*np.exp(-x/c)*np.cos((pi*x)/2/d)


def exponential_gaussian(a, b, c, d, e):
    rho = 0.0149
    sig = 0.60297151
    amp = 1/ 4.0 / pi / a**2 / b
    return lambda x: amp * np.exp(-x**2 / 4.0 / a**2) + c * np.exp(-x / d) + e


def fit_psf(data, x, params):
    return fit_function_LS(data, params, x, psf)


def fit_exponential(data, x, params):
    return fit_function_LS(data, params, x, exponential)


def fit_exponential_cosine(data, x, params):
    return fit_function_LS(data, params, x, exponential_cosine)


def fit_exponential_gaussian(data, x, params):
    return fit_function_LS(data, params, x, exponential_gaussian)


def init_curve(params, x_range, fit_func):
    curve = fit_func(*params)(x_range)
    return curve

def fit_correlation(data, x_range, solver, guess=None):
    result, cov_x, infodict, good = solver(data[0, 1:], x_range[1:], guess)
#     result = solver(data, x_range, guess)
    if solver.__name__ == 'fit_exponential':
        fit_func = exponential
    if solver.__name__ == 'fit_exponential_cosine':
        fit_func = exponential_cosine
    if solver.__name__ == 'fit_exponential_gaussian':
        fit_func = exponential_gaussian
    if solver.__name__ == 'fit_psf':
        fit_func = psf
    curve = init_curve(result, x_range, fit_func)
    return (curve, result)
#     curve = init_curve(result.x, x_range, fit_func)
#     return (curve, result.x)


def cart2pol(x, y, z):
    theta = np.arctan2(x, y)
    r = np.sqrt(x**2+y**2)
    v = z
    return (theta, r, v)


def pc_corr(image1=None, image2=None, region=None, rmax=100):
    """
    This is a generalised version of Matlab code from the
    publication:

    "Correlation Functions Quantify Super-Resolution Images
    and Estimate Apparent Clustering Due to Over-Counting"
    Veatch et al, PlosONE, DOI: 10.1371/journal.pone.0031457

    This paper should be referenced in any publication that
    results from the use of this function

    """
    if (image1 is None) or (image2 is None):
        raise 'Input at least one image to calculate correlation'

    if region is None:
        raise 'must create an roi'
    else:
        # roi will be [xmin,xmax,ymin,ymax]
        if not isinstance(region, list):
            if region.ndim == 2:
                mask = region
            else:
                raise 'region should be a list or a 2D numpy array'
        else:
            mask = np.zeros(image1.shape, dtype='float64')
            mask[region[0]:region[1], region[2]:region[3]] = 1.0
    from matplotlib import pyplot as plt

    N1 = np.sum(image1 * mask)  # number of particles within mask
    N2 = np.sum(image2 * mask)  # number of particles within mask
    I1 = image1.astype('float64')       # convert to double
    I2 = image2.astype('float64')
    L1 = I1.shape[0] + rmax  # size of fft2 (for zero padding)
    L2 = I1.shape[1] + rmax  # size of fft2 (for zero padding)

    A = float(np.sum(mask))      # area of mask
    # Normalization for correct boundary conditions
    NP = np.real(fftshift(ifft2(np.absolute(fft2(mask, (L1, L2)))**2)))
    with np.errstate(divide='ignore', invalid='ignore'):
        G1 = (
            A**2 / N1 / N2 *
            np.real(fftshift(ifft2(fft2(I1*mask, (L1, L2)) *
                    np.conj(fft2(I2*mask, (L1, L2))))))/NP
        )

    G1[np.isnan(G1)] = 0.0
    row = floor(float(L1) / 2) - rmax
    col = floor(float(L2) / 2) - rmax
    w = h = 2 * rmax + 1

    G = G1[row:row + h, col:col + w]   # only return valid part of G
    # map to x positions with center x=0
    xvals = (
        np.ones((1, 2 * rmax + 1)).T * np.linspace(-rmax, rmax, 2 * rmax + 1)
    )
    # map to y positions with center y=0
    yvals = (
        np.outer(
            np.linspace(-rmax, rmax, 2 * rmax + 1), np.ones((1, 2 * rmax + 1))
        )
    )
    zvals = G
    # convert x, y to polar coordinates
    theta, r, v = cart2pol(xvals, yvals, zvals)
    Ar = np.reshape(r, (1, (2 * rmax + 1)**2))
    Avals = np.reshape(v, (1, (2 * rmax + 1)**2))
    ind = np.argsort(Ar, axis=1)
    rr = np.sort(Ar, axis=1)
    vv = Avals[:, ind[0, :]]
    # the radii you want to extract
    rbins = [i for i in range(int(floor(np.max(rr))))]
    # bin by radius
    bin = np.digitize(rr[0, :], rbins, right=True)
    g = np.ones((1, rmax + 1))
    dg = np.ones((1, rmax + 1))
    for j in range(0, rmax + 1):
        m = bin == j
        # the number of pixels in that bin
        n2 = np.sum(m)
        if n2 == 0:
            continue
        else:
            # the average G values in this bin
            g[:, j] = np.sum(m * vv) / n2
            # the variance of the mean
            dg[:, j] = np.sqrt(np.sum(m * (vv - g[:, j])**2)) / n2

    r = np.arange(rmax+1)
    G[rmax+1, rmax+1] = 0
    return (g, r)
