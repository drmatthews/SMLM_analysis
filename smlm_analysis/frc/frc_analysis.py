import os
import pandas as pd
import numpy as np
import skimage.draw
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
from tiffile import imread

from ..utils.locs_hdfstore import LocsHDFStore
from ._frc_inner import calc_frc
from ._tukey_inner import tukey_2d
from ..utils.render.histogram import hist_2d

###
# Needs refactoring - 
# provide a single entry point with a flag which selects
# whether to stream or not
###


def intersection(x1, y1, x2, y2):
    # finds a single crossing point
#      y = a*x + b
    a1 = (y1[1] - y1[0]) / (x1[1] - x1[0])
    a2 = (y2[1] - y2[0]) / (x2[1] - x2[0])
    b1 = y1[0] - a1 * x1[0]
    b2 = y2[0] - a2 * x2[0]
    xi = (b2 - b1) / (a1 - a2)
    yi = a1 * xi + b1
    return (xi, yi)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def frc_in_memory(locs_path=None, image1=None, image2=None,
                  blocksize=500, scale=10, camera_pixel=160,
                  camera_format=(256,256), show_plot=True):

    """
    Fourier ring correlation analysis of localisation microscopy data.
    Can calculate FRC on localisation tables exported from NSTORM or
    ThunderSTORM or an image saved in TIF format (must provide 2 images).
    This does not yet work on ROIs but LocsTable class does support this
    as long as the ROIs come from ImageJ.

    Works by splitting the input dataset into two. This is done as in [1].
    Blocks of localisations are randomly assigned to each half dataset
    and the FRC is calculated as in equation 1 of [1]. Actual frc calculation
    is done in cythonized function in _frc_inner.pyx.
    This assumes that repeated localisations have already been merged.

    This version dumps all localisations to memory and calculates the 2D
    histogram followed by the frc. If memory is an issue (the dataset is
    particularly big or your machine doesn't have much RAM) use the streaming
    option below.

    [1] Measuring image resolution in optical nanoscopy, 
    Robert P J Nieuwenhuizen, et al. Nature Methods volume 10,
    pages 557–562 (2013), https://doi.org/10.1038/nmeth.2448

    """
    
    if locs_path:
        with LocsHDFStore(locs_path) as lhdf:

            num_frames = lhdf.n_frames
            num_blocks = num_frames // blocksize
            subset = np.ones(num_blocks)
            subset[:num_blocks // 2] = 0
            np.random.shuffle(subset)
            rand_subset = subset.astype(int)

            first_half = []
            second_half = []
            start = 0
            end = blocksize - 1
            for i, s in enumerate(rand_subset):
                if s == 0:
                    # need a LocsTable method for slicing based on frame column
                    first_half.append(lhdf.table[(lhdf.table.frame >= float(start)) & (lhdf.table.frame <= float(end))])
                if s == 1:
                    second_half.append(lhdf.table[(lhdf.table.frame >= float(start)) & (lhdf.table.frame <= float(end))])
                start += blocksize
                end += blocksize

            first_half_df = pd.concat(first_half)
            second_half_df = pd.concat(second_half)
            first_points = np.array((first_half_df['x'].values, first_half_df['y'].values)).T
            second_points = np.array((second_half_df['x'].values, second_half_df['y'].values)).T

            bins = [f * scale for f in camera_format] # num pixels in reconstructed image
            nm_scale = float(camera_pixel / scale)

            # this bit is repeated in several locations - integrate
            # into rendering module
            i_x = np.floor(first_points[:, 0]/nm_scale).astype(np.int32)
            i_y = np.floor(first_points[:, 1]/nm_scale).astype(np.int32)
            n1 = first_points.shape[0]
            im1 = np.zeros(bins, dtype=np.int32)
            hist_2d(i_x, i_y, n1, im1)
            
            i_x = np.floor(second_points[:, 0]/nm_scale).astype(np.int32)
            i_y = np.floor(second_points[:, 1]/nm_scale).astype(np.int32)
            n2 = second_points.shape[0]
            im2 = np.zeros(bins, dtype=np.int32)
            hist_2d(i_x, i_y, n2, im2)

    elif image1 and image2:
        _, ext1 = os.path.splitext(image1)
        _, ext2 = os.path.splitext(image2)
        if (
            (('tif' in ext1.lower()) or ('tiff' in ext1.lower())) and
            (('tif' in ext2.lower()) or ('tiff' in ext2.lower()))
        ):
            im1 = imread(image1)
            im2 = imread(image2)
    else:
        print("To calculate the FRC you need a localisation table or two images in tif format")

    # note win2d is a buffer
    win2d = np.zeros((im1.shape[0], im1.shape[1]), dtype=np.float64)
    tukey_2d(1.0 / 4.0, im1.shape[0], win2d)

    im1 = im1 * win2d
    im2 = im2 * win2d

    im1_fft = np.fft.fftshift(np.fft.fft2(im1))
    im2_fft = np.fft.fftshift(np.fft.fft2(im2))

    frc, frc_counts = calc_frc(im1_fft, im2_fft)

    for i in range(frc.size):
        if (frc_counts[i] > 0):
            frc[i] = frc[i]/float(frc_counts[i])
        else:
            frc[i] = 0.0

    xvals = np.arange(frc.size)
    xvals = xvals/(float(im1_fft.shape[0]) * camera_pixel * (1.0/float(scale)))

    thresh = np.ones(xvals.shape[0]) * (1 / 7)
    smooth_frc_out = smooth(frc, 19)
    centers = xvals

    # this finds only the first crossing point - 
    # probably should deal with every crossing up to Nyquist
    ind = np.where((smooth_frc_out - thresh) < 0.0)[0][0]
    x1 = np.array((centers[ind - 1], centers[ind]))
    y1 = np.array((smooth_frc_out[ind - 1], smooth_frc_out[ind]))
    x2 = x1
    y2 = np.array((thresh[ind - 1], thresh[ind]))

    inters = intersection(x1, y1, x2, y2)
    
    if show_plot:
        plt.figure()
        plt.scatter(xvals, frc, s=4, color='black')
        plt.plot(inters[0], inters[1], 'g*', markersize=12)
        plt.plot(xvals, thresh, 'k-', lw=2)
        plt.xlim([xvals[0], xvals[-1]])
        plt.ylim([-0.2,1.2])
        plt.xlabel("Spatial Frequency (nm-1)")
        plt.ylabel("Correlation")
        # plt.savefig('frc_test.png')
        plt.show()


def frc_stream(fpath=None, blocksize=500, scale=10,
               camera_pixel=160, camera_format=(256,256),
               show_plot=True):

    """
    Streams localisation coordinates from disk and builds
    histogram on-the-fly. Find an im-memory version of this
    above.

    Follows [1].

    [1] Measuring image resolution in optical nanoscopy, 
    Robert P J Nieuwenhuizen, et al. Nature Methods volume 10,
    pages 557–562 (2013), https://doi.org/10.1038/nmeth.2448
    """
    
    if fpath:
        with LocsHDFStore(fpath) as lhdf:
            num_frames = lhdf.n_frames
            num_blocks = num_frames // blocksize
            subset = np.ones(num_blocks)
            subset[:num_blocks // 2] = 0
            np.random.shuffle(subset)
            rand_subset = subset.astype(int)

            bins = [f * scale for f in camera_format] # num pixels in reconstructed image
            nm_scale = float(camera_pixel / scale)
            im1 = np.zeros(bins, dtype=np.int32)
            im2 = np.zeros(bins, dtype=np.int32)

            start = 0
            end = blocksize - 1
            for i, s in enumerate(rand_subset):
                if s == 0:
                    locs = lhdf.get_in_frame_range(start=start, stop=end)
                    x = locs['x'].values
                    y = locs['y'].values
                    i_x = np.floor(x / nm_scale).astype(np.int32)
                    i_y = np.floor(y / nm_scale).astype(np.int32)
                    n1 = x.shape[0]
                    hist_2d(i_x, i_y, n1, im1)
                if s == 1:
                    locs = lhdf.get_in_frame_range(start=start, stop=end)
                    x = locs['x'].values
                    y = locs['y'].values
                    i_x = np.floor(x / nm_scale).astype(np.int32)
                    i_y = np.floor(y / nm_scale).astype(np.int32)
                    n2 = x.shape[0]
                    hist_2d(i_x, i_y, n2, im2)
                start += blocksize
                end += blocksize
    else:
        return None

    # note win2d is a buffer
    win2d = np.zeros((im1.shape[0], im1.shape[1]), dtype=np.float64)
    tukey_2d(1.0 / 4.0, im1.shape[0], win2d)

    im1 = im1 * win2d
    im2 = im2 * win2d

    im1_fft = np.fft.fftshift(np.fft.fft2(im1))
    im2_fft = np.fft.fftshift(np.fft.fft2(im2))

    frc, frc_counts = calc_frc(im1_fft, im2_fft)
    
    for i in range(frc.size):
        if (frc_counts[i] > 0):
            frc[i] = frc[i] / float(frc_counts[i])
        else:
            frc[i] = 0.0

    xvals = np.arange(frc.size)
    xvals = xvals / (float(im1_fft.shape[0]) * camera_pixel * (1.0 / float(scale)))
    frc = np.real(frc)

    thresh = np.ones(xvals.shape[0]) * (1 / 7)
    smooth_frc_out = smooth(frc, 19)
    centers = xvals

    # this finds only the first crossing point - 
    # probably should deal with every crossing up to Nyquist
    ind = np.where((smooth_frc_out - thresh) < 0.0)[0][0]
    x1 = np.array((centers[ind - 1], centers[ind]))
    y1 = np.array((smooth_frc_out[ind - 1], smooth_frc_out[ind]))
    x2 = x1
    y2 = np.array((thresh[ind - 1], thresh[ind]))

    inters = intersection(x1, y1, x2, y2)

    if show_plot:
        plt.figure()
        plt.scatter(xvals, frc, s = 4, color = 'black')
        plt.plot(inters[0], inters[1], 'g*', markersize=12)
        plt.plot(xvals, thresh, 'k-', lw=2)
        plt.xlim([xvals[0], xvals[-1]])
        plt.ylim([-0.2,1.2])
        plt.xlabel("Spatial Frequency (nm-1)")
        plt.ylabel("Correlation")
        # plt.savefig('frc_test.png')
        plt.show()

if __name__=='__main__':
    f = '.\\test_data\\tubulin647 2d.h5'
    # f = '.\\test_data\\ts_small.h5'
    in1 = '.\\test_data\\in1.tif'
    in2 = '.\\test_data\\in2.tif'
    frc_stream(f, blocksize=500, camera_pixel=16e3/100, show_plot=True)
    frc_in_memory(f, camera_pixel=16e3/100, show_plot=True)  

    