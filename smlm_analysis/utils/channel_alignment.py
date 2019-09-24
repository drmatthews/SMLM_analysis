"""
A collection of functions for aligning the channels in a multi-channel
localisation microscopy dataset.

Note
-----
Three methods are accessible here:

1. Image alignment using cross-correlation. A Gaussian function is
fitted to the cross correlation of two images to find the shift. The
final shift is determined using redundancy.

2. Image alignment by determine the displacement of the mean position
of linked localisations. KDTrees are used to associate the mean
positions across the channels. Displacements are calculated for image
pairs (e.g. (0,1), (0,2), (1,2) for 3 channels) and the final drift is
calculated using redundancy.

3. The classic method of using an affine transform (in this case only
the translation is used). Here the mean position of linked
localisations are associated across the channels using KDTrees and
the skimage AffineTransform class is used to calculate the shift.
"""
import numpy as np
from skimage import transform as tf
from numpy.fft import fft2, ifft2, fftshift
import scipy.optimize
from scipy.spatial import cKDTree

from ..render.rendering import image_2d


###
# functions used in redundant cross correlation
###
def near_neighbours(pointsA, pointsB, radius):
    """
    Determine the indices of the near neighbours of pointsA
    in pointsB.
    """

    treeA = cKDTree(pointsA)
    treeB = cKDTree(pointsB)
    # indices of all neighbours of treeA in treeB
    indices = treeA.query_ball_tree(treeB, radius)

    # occassionally there will be multiple elements
    # in one tree which will have neighbours in the
    # other or there will be no neighbours. Remove
    # the duplicates and set no neighbours to -1.
    ind = np.ones(len(indices), dtype=np.int64) * -1
    for i, idx in enumerate(indices):
        if len(idx) == 0:
            # no neighbours
            continue
        if len(idx) == 1:
            # one neighbour
            ind[i] = idx[0]
        elif len(idx) > 1:
            # multiple neighbours - the correct one
            # is found by determining the one with the
            # minimum distance
            dist = []
            for el in idx:
                dist.append(treeA.query(treeB.data[el], k=1)[0])
            min_dist, min_dist_idx = min(
                (val, idx) for (idx, val) in enumerate(dist)
            )
            ind[i] = idx[min_dist_idx]
    
    return ind


def fit(data, params, fit_func):
    """
    Non-weighted least squares fit -
    Levenburg-Marquadt
    """
    size, _ = data.shape
    result = params
    fn = fit_func
    errorfunction = (
        lambda p: np.ravel(fn(*p)(*np.indices(data.shape)) - data)
    )
    [result, cov_x, infodict, mesg, success] = (
        scipy.optimize.leastsq(
            errorfunction, params,
            full_output=1, ftol=1e-2, xtol=1e-2
        )
    )
    fit_params = result
    err = errorfunction(result)
    err = scipy.sum(err * err)

    return result


def symmetric_gaussian(bg, N, y0, x0, w):
    """
    2D symmetric Gaussian function for fitting
    cross correlation peak
    """
    return (
        lambda x,y: bg + N * np.exp(-(((x - x0) / w)**2 \
                    + ((y - y0) / w)**2) * 2)
    )


def fit_symmetric_gaussian(data, sigma=1.0):
    """
    Perform least squares fitting of Gaussian function
    to cross correlation peak.

    Data is assumed centered on the gaussian
    and of size roughly 2x the width.
    """
    params = [np.min(data),
              np.max(data),
              0.5 * data.shape[0],
              0.5 * data.shape[1],
              2.0 * sigma]
    return fit(data, params, symmetric_gaussian)


def xcorr(A, B):
    """
    Calculate cross correlation of two images.
    """
    fftA = fft2(A)
    fftB = fft2(B)
    norm = np.sqrt(A.size)
    return fftshift(
        np.real(
            ifft2(
                fftA * np.conj(fftB)
            )
        )
    )


def estimate_maxima(corr, window=48, padding=10):
    """
    Estimate maxima by upsampling the cross correlation
    using an FFT. Translated from Matlab code published
    in 
    """

    # crop out the correlation peak
    boxsize = corr.shape[0]
    sy = boxsize / 2
    sx = sy
    offsetx = np.floor(sx - window/2)
    offsety = np.floor(sy - window/2)
    if offsetx < 0:
        offsetx = 0
    elif offsetx > boxsize:
        offsetx = boxsize - window

    if offsety < 0:
        offsety = 0
    elif offsety > boxsize:
        offsety = boxsize - window

    offsetx = int(offsetx)
    offsety = int(offsety)
    crop = corr[offsety : offsety + window + 1,
                offsetx : offsetx + window + 1]

    # Fourier interpolation, zero-padding
    data = fftshift(np.real(fft2(crop)))

    padsize = window * padding
    padoffset = int((padsize / 2) - (window / 2))
    padded = np.zeros((padsize, padsize))
    padded[padoffset : padoffset + window + 1,
           padoffset : padoffset + window + 1] = data
    data = np.abs(ifft2(padded))
    maxima = np.unravel_index(np.argmax(np.abs(data)), data.shape)
    x = np.ceil((maxima[0] / padding) + offsetx)
    y = np.ceil((maxima[1] / padding) + offsety)
    return (x.astype(int), y.astype(int))


def redundancy(dx, dy, n_channels=3):
    """
    Work out the redundancy (A in r = AD) and solve the inverse
    problem to calculate the shift.
    """
    n_pairs = int(n_channels * (n_channels - 1) / 2.0)
    rij_x = np.zeros(n_pairs, dtype=np.float32)
    rij_y = np.zeros(n_pairs, dtype=np.float32)
    A = np.zeros((n_pairs, n_channels - 1), dtype=np.float32)
    p = 0
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
            print(i, j)
            print(dx[i, j])
            rij_x[p] = dx[i, j]
            rij_y[p] = dy[i, j]
            A[p, i: j] = 1.0
            p += 1

    pinv_A = np.linalg.pinv(A)
    dx = np.dot(pinv_A, rij_x)
    dy = np.dot(pinv_A, rij_y)
    
    dx = np.insert(np.cumsum(dx), 0, 0)
    dy = np.insert(np.cumsum(dy), 0, 0)
    return (dx, dy)


def image_offset(A, B):
    """
    Calculate the offset between two images by fitting the peak of
    the cross correlation of the two images.
    """
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in A.shape])
    corr = xcorr(A, B)
    maxima = estimate_maxima(corr, window=48, padding=10)
    fit_radius = 5
    roi = corr[maxima[1] - fit_radius: maxima[1] + fit_radius + 1,
               maxima[0] - fit_radius: maxima[0] + fit_radius + 1]

    result = fit_symmetric_gaussian(roi, 2.0)
    dx = result[2] + maxima[0] - fit_radius - midpoints[1]
    dy = result[3] + maxima[1] - fit_radius - midpoints[0]
    return (dx, dy)


def rcc(locs, frame_width, frame_height, camera_pixel, scale=2, n_channels=3):
    """
    Determine shift between channels using redundant cross correlation.
    """
    bins = [frame_width * scale, frame_height * scale]
    bin_size = float(camera_pixel / scale)
    dx = np.zeros((n_channels, n_channels))
    dy = np.zeros((n_channels, n_channels))
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
            im_base = image_2d(locs[i]['x'].values, locs[i]['y'].values, bins[1], bins[0], bin_size)
            im = image_2d(locs[j]['x'].values, locs[j]['y'].values, bins[1], bins[0], bin_size)
            shift_x, shift_y = image_offset(im_base, im)
            dx[i, j] = shift_x / float(scale)
            dy[i, j] = shift_y / float(scale)
            
    return redundancy(dx, dy)


###
# use the mean position of linked localisations to find channel shift
# also makes use of redundancy
###
def shift(pos, nn_radius, axis='x', n_channels=3):
    """
    Mean displacement of track centers (pos) for pairs channels.
    KDTrees are used to associate track centers across the channels.
    """
    d = np.zeros((n_channels, n_channels))
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
            nn = near_neighbours(pos[i], pos[j], nn_radius)
            comA = pos[i][nn !=-1]
            comB = pos[j].iloc[nn[nn != -1]]
            d[i, j] = np.nanmean(
                [cj - ci for ci, cj in zip(comA[axis], comB[axis])]
            )
    return d


###
# align channels using affine tranform
# this uses coordinates of markers in the image plane
###
def affine_transform(src, dst):
    """
    Wrapper around skimage transform
    """
    tform = tf.AffineTransform()
    tform.estimate(src, dst)
    A = tform.params.copy()
    # reg = tform.inverse(unreg)
    # A, reg, rank, s = np.linalg.lstsq(pad(unreg), pad(base))
    A[np.abs(A) < 1e-10] = 0  # set really small values to zero
    # return (base, reg, A)
    return (tform, A)


def inverse_transform(coords, tform):
    """
    Wrapper around skimage AffineTransform.inverse
    """
    return tform.inverse(coords)

