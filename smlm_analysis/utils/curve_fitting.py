from math import pi, sqrt
import numpy as np
import scipy.optimize


def least_squares(data, params, x, fn):
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

def pd_(a, b, c, d, e, f):
    return lambda x: (
        (x / (2 * a*a)) * np.exp((-1) * x*x / (4 * a*a)) * e +
        (d / (c * sqrt(pi * 2))) * np.exp(-0.5 * ((x - b) / c) *
        ((x - b) / c)) + f * x
    )

def pd(x, a1, a2, a3, sig, w, xc):
    y = (
        a1 * ((x / 2 * sig**2) * np.exp(-(x**2) / (4 * sig**2))) +
        a2 * ((1 / sqrt(2 * pi * w**2)) * np.exp(-(x - xc)**2 / (2 * w**2))) +
        a3 * x
    )
    return y

def fit_psf(data, x, params):
    return least_squares(data, params, x, psf)


def fit_exponential(data, x, params):
    return least_squares(data, params, x, exponential)


def fit_exponential_cosine(data, x, params):
    return least_squares(data, params, x, exponential_cosine)


def fit_exponential_gaussian(data, x, params):
    return least_squares(data, params, x, exponential_gaussian)

def fit_pd(data, x, params):
    return least_squares(data, params, x, pd)    


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

def fit_adj_nn(data, x_range, guess=None):
    # result, cov_x, infodict, good = fit_pd(data, x_range, guess)
    fit_func = pd
    result, cov = scipy.optimize.curve_fit(fit_func, x_range, data, guess)
    print(result)
    curve = pd(x_range, *result)
    return (curve, result)
