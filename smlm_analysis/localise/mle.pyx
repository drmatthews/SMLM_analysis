cimport cython
cimport libc.math as math
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _erf(double x, double mu, double sigma):
    
    cdef double sq_norm, d, f

    sq_norm = 0.70710678118654757 / sigma       # sq_norm = sqrt(0.5/sigma**2)
    d = x - mu
    f = 0.5 * (math.erf((d + 0.5) * sq_norm) - \
               math.erf((d - 0.5) * sq_norm))  
    return f

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef (double, double) _dgauss(double x, double mu, double sigma, double N, double PSFc):

    cdef double a, b, d, dudt, d2udt2

    d = x - mu
    a = math.exp(-0.5 * ((d + 0.5) / sigma)**2)
    b = math.exp(-0.5 * ((d - 0.5) / sigma)**2)
    dudt = -N * PSFc * (a - b) / (math.sqrt(2.0 * math.pi) * sigma)
    d2udt2 = -N * ((d + 0.5) * a - (d - 0.5) * b) \
                * PSFc / (math.sqrt(2.0 * math.pi) * sigma**3)
    return dudt, d2udt2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef (double, double) _dgauss_1d_sigma(double x, double mu, double sigma, double N, double PSFc):
    
    cdef double ax, bx, dudt, d2udt2

    ax = math.exp(-0.5 * ((x + 0.5 - mu) / sigma)**2)
    bx = math.exp(-0.5 * ((x - 0.5 - mu) / sigma)**2)
    dudt = -N * (ax * (x + 0.5 - mu) - bx * (x - 0.5 - mu))\
            * PSFc / (math.sqrt(2.0 * math.pi) * sigma**2)
    d2udt2 = -2.0 * dudt / sigma - N * \
                (ax * (x + 0.5 - mu)**3 - bx * (x - 0.5 - mu)**3)\
                * PSFc / (math.sqrt(2.0 * math.pi) * sigma**5)
    return dudt, d2udt2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef (double, double) _dgauss_2d_sigma(double x, double y, double mu, double nu,
                     double sigma, double N, double PSFx, double PSFy):

    cdef double dSx, ddSx, dSy, ddSy, dudt, d2udt2

    dSx, ddSx = _dgauss_1d_sigma(x, mu, sigma, N, PSFy)
    dSy, ddSy = _dgauss_1d_sigma(y, nu, sigma, N, PSFx)    
    dudt = dSx + dSy
    d2udt2 = ddSx + ddSy
    return dudt, d2udt2

@cython.boundscheck(False)
@cython.cdivision(True)
def mle(double[:, ::1] img, int size, double[::1] guess, double max_iter, double tol):
    
    # guess is [x, y, N, bg, S]
    cdef int i, ii, jj, kk, ll, iterations
    cdef double old_x, old_y, old_sx, old_sy, model
    cdef double PSFx, PSFy, cf, df, update, data, Div
    cdef double likelihood
    cdef int n_params = guess.shape[0]
    cdef double [::1] mle_params

    GAMMA_ = np.array([1.0, 1.0, 0.5, 1.0, 1.0, 1.0])
    cdef double [::1] GAMMA = GAMMA_

    max_step_ = np.zeros(n_params, dtype=np.float64)
    cdef double [::1] max_step = max_step_    

    # Memory allocation (we do that outside of the loops
    # to avoid huge delays in threaded code):
    dudt_ = np.zeros(n_params, dtype=np.float64)
    cdef double [::1] dudt = dudt_

    d2udt2_ = np.zeros(n_params, dtype=np.float64)
    cdef double [::1] d2udt2 = d2udt2_

    numerator_ = np.zeros(n_params, dtype=np.float64)
    cdef double [::1] numerator = numerator_
    denominator_ = np.zeros(n_params, dtype=np.float64)
    cdef double [::1] denominator = denominator_

    M_ = np.zeros((n_params, n_params), dtype=np.float64)
    cdef double [:, ::1] M = M_

    Minv_ = np.zeros((n_params, n_params), dtype=np.float64)
    cdef double [:, ::1] Minv = Minv_

    CRLB_ = np.zeros(n_params, dtype=np.float64)
    cdef double [::1] CRLB = CRLB_

    max_step[0:2] = guess[4]
    for i in range(2,4):
        max_step[i] = 0.1 * guess[i]
    
    for i in range(4, n_params):
        max_step[i] = 0.2 * guess[i]

    if n_params == 5:
        old_x = guess[0]
        old_y = guess[1]
    else:
        old_x = guess[0]
        old_y = guess[1]
        old_sx = guess[4]
        old_sy = guess[5]

    kk = 0
    while kk < max_iter:
        kk += 1

        numerator[:] = 0.0
        denominator[:] = 0.0

        for ii in range(size):
            for jj in range(size):
                PSFx = _erf(ii, guess[0], guess[4])
                PSFy = _erf(jj, guess[1], guess[-1])


                # Derivatives
                dudt[0], d2udt2[0] = _dgauss(
                    ii, guess[0], guess[4], guess[2], PSFy
                )
                dudt[1], d2udt2[1] = _dgauss(
                    jj, guess[1], guess[4], guess[2], PSFx
                )
                dudt[2] = PSFx * PSFy
                d2udt2[2] = 0.0
                dudt[3] = 1.0
                d2udt2[3] = 0.0
                if n_params == 5:
                    dudt[4], d2udt2[4] = _dgauss_2d_sigma(
                        ii, jj, guess[0], guess[1],
                        guess[4], guess[2], PSFx, PSFy
                    )
                else:
                    dudt[4], d2udt2[4] = _dgauss_1d_sigma(
                        ii, guess[0], guess[4], guess[2], PSFy
                    )
                    dudt[5], d2udt2[5] = _dgauss_1d_sigma(
                        jj, guess[1], guess[5], guess[2], PSFx
                    )

                model = guess[2] * dudt[2] + guess[3]
                
                cf = df = 0.0
                data = img[ii, jj]
                if model > 10e-3:
                    cf = data / model - 1
                    df = data / model**2
                cf = math.fmin(cf, 10e4)
                df = math.fmin(df, 10e4)

                for ll in range(n_params):
                    numerator[ll] += cf * dudt[ll]
                    denominator[ll] += cf * d2udt2[ll] - df * dudt[ll]**2

        # The update
        for ll in range(n_params):
            if denominator[ll] == 0.0:
                if numerator[ll] * max_step[ll] < 0.0:
                    update = -1.0
                else:
                    update = 1.0
            else:
                update = math.fmin(
                    math.fmax(
                        numerator[ll] / denominator[ll], -max_step[ll]
                    ),
                    max_step[ll]
                )
            if kk < 5:
                update *= GAMMA[ll]
            guess[ll] -= update

        # Other constraints
        guess[2] = math.fmax(guess[2], 1.0)
        guess[3] = math.fmax(guess[3], 0.01)
        guess[4] = math.fmax(guess[4], 0.01)
        if n_params == 5:
            guess[4] = math.fmin(guess[4], size)
        else:
            guess[4] = math.fmax(guess[4], 0.01)
            guess[5] = math.fmax(guess[5], 0.01)

        # Check for convergence
        if n_params == 5:
            if ((math.fabs(old_x - guess[0]) < tol) and
                (math.fabs(old_y - guess[1]) < tol)):
                break
            else:
                old_x = guess[0]
                old_y = guess[1]
        else:
            if math.fabs(old_x - guess[0]) < tol:
                if math.fabs(old_y - guess[1]) < tol:
                    if math.fabs(old_sx - guess[4]) < tol:
                        if math.fabs(old_sy - guess[5]) < tol:
                            break
            old_x = guess[0]
            old_y = guess[1]
            old_sx = guess[4]
            old_sy = guess[5]                

    mle_params = np.asarray(guess)
    iterations = kk

    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    for ii in range(size):
        for jj in range(size):
            PSFx = _erf(ii, guess[0], guess[4])
            PSFy = _erf(jj, guess[1], guess[4])
            model = guess[3] + guess[2] * PSFx * PSFy

            # Calculating derivatives
            dudt[0], d2udt2[0] = _dgauss(
                ii, guess[0], guess[4], guess[2], PSFy
            )
            dudt[1], d2udt2[1] = _dgauss(
                jj, guess[1], guess[4], guess[2], PSFx
            )
            if n_params == 5:
                dudt[4], d2udt2[4] = _dgauss_2d_sigma(
                    ii, jj, guess[0], guess[1],
                    guess[4], guess[2], PSFx, PSFy
                )
            else:
                dudt[4], d2udt2[4] = _dgauss_1d_sigma(
                    ii, guess[0], guess[4], guess[2], PSFy
                )
                dudt[5], d2udt2[5] = _dgauss_1d_sigma(
                    jj, guess[1], guess[5], guess[2], PSFx
                )

            dudt[2] = PSFx * PSFy
            dudt[3] = 1.0

            # Building the Fisher Information Matrix
            model = guess[3] + guess[2] * dudt[2]
            for kk in range(n_params):
                for ll in range(kk, n_params):
                    M[kk, ll] += dudt[ll] * dudt[kk] / model
                    M[ll, kk] = M[kk, ll]

            # LogLikelihood
            if model > 0:
                data = img[ii, jj]
                if data > 0:
                    Div += data * math.log(model) - model\
                         - data * math.log(data) + data
                else:
                    Div += -model

    likelihood = Div

    # Matrix inverse (CRLB=F^-1)
    Minv = np.linalg.pinv(M)
    for kk in range(n_params):
        CRLB[kk] = Minv[kk, kk]

    return mle_params, likelihood, CRLB_, iterations