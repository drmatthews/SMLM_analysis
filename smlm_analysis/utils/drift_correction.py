import time

import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import scipy.optimize
from scipy.interpolate import interp1d
from scipy import interpolate
from matplotlib import pyplot as plt
import lmfit
import numba
import pandas as pd
import tqdm

from .locs_hdfstore import LocsHDFStore
from .parameters import SMLMParameters
from ..render.histogram import hist_2d, hist_3d


class DriftCorrectionBase:
    """
    Base class for drift correction -
    encapsulates making image histograms
    and calculation of image offsets.

    Drift can be corrected using the direct cross correlation
    method or the redundant method (see subclasses).

    Z drift is corrected by making xz and yz projections of
    the 3D histogram.
    """
    def __init__(self, locs_hdf, params, notebook):
        self.lhdf = locs_hdf
        self.num_frames = locs_hdf.n_frames
        width = params.frame_width
        height = params.frame_height
        scale = params.scale
        self.scale = scale
        self.camera_pixel = params.camera_pixel
        self.bins = [width * scale, height * scale]
        self.xy_scale = float(params.camera_pixel / scale)
        self.z_step = params.z_step
        self.blocksize = params.blocksize
        self.num_blocks = self.num_frames // self.blocksize
        # just in case the user forgets to set an
        # appropriate blocksize
        if self.num_blocks < 5:
            self.num_blocks = 5
            self.blocksize = self.num_frames // self.num_blocks

        self.pbar = tqdm.tqdm
        if notebook:
            self.pbar = tqdm.tqdm_notebook
        self.driftx = None
        self.drifty = None
        self.driftz = None

    def bin_locs(self, locs, is_3d=False, z_project=None):
        """
        Make a 2D histogram of localisations (an image) which 
        can be used to calculated cross correlation. A 2x upscale
        compared to raw data works well.
        NB. This could be moved to rendering module.
        """
        x = locs['x'].values
        y = locs['y'].values
        i_x = np.floor(x / self.xy_scale).astype(np.int32)
        i_y = np.floor(y / self.xy_scale).astype(np.int32)
        n = x.shape[0]
        bins = self.bins
    
        if is_3d:
            if not locs['z'].any(axis=0):
                raise ValueError('No non zero z values in dataset')

            z = locs['z'].values
            
            z_min = locs['z'].min()
            z_max = locs['z'].max()
            z_bins = int((z_max - z_min) / self.z_step)
            bins.append(z_bins)

            i_z = np.floor((z - z_min) / self.z_step).astype(np.int32)

        im = np.zeros(bins, dtype=np.int32)
        if is_3d:
            hist_3d(i_x, i_y, i_z, n, im)
        else:
            hist_2d(i_x, i_y, n, im)

        if is_3d and z_project:
            if z_project == 'z':
                im = self._z_projection(im)
            elif z_project == 'xz':
                im = self._xz_projection(im)
            elif z_project == 'yz':
                im = self._yz_projection(im)
            else:
                raise ValueError(
                    'z_project should be one of `z`, `xz` or `yz`'
                )

        return im
        
    def _z_projection(self, im):
        """
        Maximum intensity projection
        """
        return np.max(im, axis=-1)

    def _xz_projection(self, im):
        """
        xz view of 3D histogram
        """
        return np.max(im, axis=1)

    def _yz_projection(self, im):
        """
        yz view of 3D histogram
        """        
        return np.max(im, axis=0)

    def _interpolate_drift(self, xvals, yvals):
        """
        Interpolate drift data to produce a drift value
        for each camera frame in the original movie.
        """
        drift_pol = (
            interpolate.InterpolatedUnivariateSpline(xvals, yvals, k=3)
        )
        t_inter = np.arange(self.num_frames)
        final_drift = drift_pol(t_inter)
        return final_drift

    def _symmetric_gaussian(self, bg, N, y0, x0, w):
        """
        2D symmetric Gaussian function for fitting
        cross correlation peak
        """
        return (
            lambda x,y: bg + N * np.exp(-(((x - x0) / w)**2 \
                        + ((y - y0) / w)**2) * 2)
        )

    def _fit(self, data, params, fit_func):
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


    def _fit_symmetric_gaussian(self, data, sigma=1.0):
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
        return self._fit(data, params, self._symmetric_gaussian)

    def _xcorr(self, A, B):
        """
        Calculate correlation of two images.
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

    def _estimate_maxima(self, corr, window=48, padding=10):
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

    def image_offset(self, A, B):
        """
        Calculate the offset between two images by fitting the peak of
        the cross correlation of the two images.
        """
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in A.shape])
        corr = self._xcorr(A, B)
        maxima = self._estimate_maxima(corr, window=48, padding=10)
        fit_radius = 5
        roi = corr[maxima[1] - fit_radius: maxima[1] + fit_radius + 1,
                   maxima[0] - fit_radius: maxima[0] + fit_radius + 1]

        result = self._fit_symmetric_gaussian(roi, 2.0)
        dx = result[2] + maxima[0] - fit_radius - midpoints[1]
        dy = result[3] + maxima[1] - fit_radius - midpoints[0]
        return (dx, dy)

    def apply_correction(self):
        """
        Apply the calculated drift correction and write
        corrected localisations to file.

        Note that the whole table is read to memory, grouped
        according to frame number, corrected and then the whole
        corrected table is written back to file. This is done
        for speed considerations but will potentially use a
        lot of memory.
        """
        lhdf = self.lhdf
        locs = lhdf.table
        current_key = lhdf.key
        metadata = lhdf.metadata
        key = '/temp_table'
        i = 0
        corrected = []
        print('')
        desc = 'Applying correction'
        for fid, frame in self.pbar(
            locs.groupby('frame'), desc=desc, total=lhdf.n_frames):

            cf = frame.copy()
            xc = frame['x'].values - self.driftx[i] * self.camera_pixel
            yc = frame['y'].values - self.drifty[i] * self.camera_pixel
            cf.loc[:, 'x'] = xc
            cf.loc[:, 'y'] = yc
            if 'z' in frame:
                zc = frame['z'].values - self.driftz[i] * self.camera_pixel
                cf.loc[:, 'z'] = zc
            i += 1
            corrected.append(cf)

        print('')
        print('Writing to file...')
        lhdf.write_locs(pd.concat(corrected), key=key)
        lhdf.remove_table(current_key)
        lhdf.rename_table(key, current_key[1:])
        lhdf.write_metadata(metadata, key=current_key)


class DCCDriftCorrection(DriftCorrectionBase):
    """
    Direct cross correlation drift correction.

    The dataset is split into time blocks, images are rendered
    for each block and each block is cross correlated to the time
    zero block.
    """
    def __init__(self, locs_hdf, params, correct_z=False, notebook=False):
        super().__init__(locs_hdf, params, notebook)
        self.correct_z = correct_z

    def calculate_shift(self, is_3d=False, z_project=None):
        """
        Calculate the pixel offsets using cross correlation for
        each time block by correlating to the time zero block.
        """
        base_locs = self.lhdf.get_in_frame_range(
            start=0, stop=self.blocksize - 1
        )
        im_base = self.bin_locs(base_locs, is_3d=is_3d, z_project=z_project)
        base_x, base_y = self.image_offset(im_base, im_base)

        start = self.blocksize
        stop = start + self.blocksize - 1
        mid = self.blocksize // 2
        t = [mid]
        dx = [base_x]
        dy = [base_y]

        desc = 'Determining xy drift'
        if is_3d:
            desc = 'Determining z drift'

        for block in self.pbar(range(1, self.num_blocks), desc=desc):
            locs = self.lhdf.get_in_frame_range(start=start, stop=stop)
            im = self.bin_locs(locs, is_3d=is_3d, z_project=z_project)
            shift_x, shift_y = self.image_offset(im, im_base)

            start += self.blocksize
            stop += self.blocksize
            mid += self.blocksize
            t.append(mid)
            dx.append(shift_x / float(self.scale))
            dy.append(shift_y / float(self.scale))

        nt = np.array(t)
        driftx = self._interpolate_drift(nt, np.array(dx))
        drifty = self._interpolate_drift(nt, np.array(dy))

        return (driftx, drifty, nt)

    def run(self, show_plot=False):
        """
        Run drift correction by direct cross correlation.
        """
        self.driftx, self.drifty, time = self.calculate_shift()
        if show_plot:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot(np.arange(self.num_frames), self.driftx)
            plt.plot(np.arange(self.num_frames), self.drifty)
            plt.show()

        self.driftz = np.zeros_like(self.driftx)
        if self.correct_z:

            _, drifty_xz, time = self.calculate_shift(is_3d=True, z_project='xz')
            _, drifty_yz, _ = self.calculate_shift(is_3d=True, z_project='yz')

            self.driftz = np.mean(np.hstack(drifty_xz, drifty_yz), axis=1)
            if show_plot:
                plt.figure()
                plt.plot(np.arange(self.num_frames), self.driftz)
                plt.show()

        self.apply_correction()
        return (self.driftx, self.drifty, self.driftz)


class RCCDriftCorrection(DriftCorrectionBase):
    """
    Redundant cross correlation drift correction.

    The data is split into time blocks and each block i is
    correlated to the next block j in the series. The correction
    is treated as a set of N × (N – 1) / 2 (where N is the number of
    time blocks) overdetermined equations to solve r = AD (where r is
    the shift that maximises the cross correlation and D is the drift).

    This is effectively a Python version of Matlab code written for
    Y. Wang et al "Localization events-based sample drift correction for
    localization microscopy with redundant cross-correlation algorithm",
    Opt Express. 2014 Jun 30; 22(13): 15982–15991. 
    """
    def __init__(self, locs_hdf, params, rmax=0.2, correct_z=False, notebook=False):
        super().__init__(locs_hdf, params, notebook)
        self.correct_z = correct_z
        self.rmax = rmax

    def calculate_shift(self, is_3d=False, z_project=None):
        """
        Determine the pixel shift using the cross correlation for
        the N × (N – 1) / 2 pairs of time blocks.
        """
        num_pairs = int(self.num_blocks * (self.num_blocks - 1) / 2.0)
        dx = np.zeros((self.num_blocks, self.num_blocks))
        dy = np.zeros((self.num_blocks, self.num_blocks))

        desc = 'Determining xy drift'
        if is_3d:
            desc = 'Determining x drift'

        with self.pbar(total=num_pairs, desc=desc) as progress:
            for i in range(self.num_blocks - 1):
                for j in range(i + 1, self.num_blocks):
                    start_i = i * self.blocksize
                    stop_i = (i + 1) * self.blocksize
                    base_locs = self.lhdf.get_in_frame_range(start=start_i, stop=stop_i)
                    im_base = self.bin_locs(base_locs, is_3d=is_3d, z_project=z_project)

                    start_j = j * self.blocksize
                    stop_j = (j + 1) * self.blocksize
                    locs = self.lhdf.get_in_frame_range(start=start_j, stop=stop_j)
                    im = self.bin_locs(locs, is_3d=is_3d, z_project=z_project)

                    shift_x, shift_y = self.image_offset(im_base, im)
                    dx[i, j] = shift_x / float(self.scale)
                    dy[i, j] = shift_y / float(self.scale)
                    progress.update()
        return (dx, dy)

    def redundancy(self, dx, dy):
        """
        Work out the redundancy (A in r = AD) and solve the inverse
        problem to calculate the drift.
        """
        # this part is borrowed from:
        # https://github.com/ZhuangLab/storm-analysis/

        n_pairs = int(self.num_blocks * (self.num_blocks - 1) / 2.0)
        rij_x = np.zeros(n_pairs, dtype=np.float32)
        rij_y = np.zeros(n_pairs, dtype=np.float32)
        A = np.zeros((n_pairs, self.num_blocks - 1), dtype=np.float32)
        p = 0
        for i in range(self.num_blocks - 1):
            for j in range(i + 1, self.num_blocks):
                rij_x[p] = dx[i, j]
                rij_y[p] = dy[i, j]
                A[i, i: j] = 1.0
                p += 1

        # Calculate drift (pass1). 
        # dx and dy contain the optimal offset between
        # sub image i and sub image i+1 in x/y.
        pinv_A = np.linalg.pinv(A)
        dx = np.dot(pinv_A, rij_x)
        dy = np.dot(pinv_A, rij_y)

        # Calculate errors.
        err_x = np.dot(A, dx) - rij_x
        err_y = np.dot(A, dy) - rij_y

        err_d = np.sqrt(err_x * err_x + err_y * err_y)
        arg_sort_err = np.argsort(err_d)

        # Remove bad values.
        j = len(arg_sort_err) - 1
        while (j > 0) and (err_d[arg_sort_err[j]] > self.rmax):
            index = arg_sort_err[j]
            delA = np.delete(A, index, 0)
            if (np.linalg.matrix_rank(delA, tol=1.0) == (self.num_blocks - 1)):
                # print(j, "removing", index, "with error", err_d[index])
                A = delA
                rij_x = np.delete(rij_x, index, 0)
                rij_y = np.delete(rij_y, index, 0)
                err_d = np.delete(err_d, index, 0)
                arg_sort_err[(arg_sort_err > index)] -= 1
            # else:
            #     print("not removing", index, "with error", err_d[index])
            j -= 1

        return (A, rij_x, rij_y)

    def determine_drift(self, A, rij_x, rij_y):
        """
        Do the final calculation of the drift
        after the errors have been removed and 
        interpolate to the number of frames in
        the original movie.
        """
        # Calculate drift (pass2). 
        pinv_A = np.linalg.pinv(A)
        dx = np.dot(pinv_A, rij_x)
        dy = np.dot(pinv_A, rij_y)

        # Integrate to get final drift.
        driftx = np.zeros((dx.size))
        drifty = np.zeros((dy.size))
        for i in range(dx.size):
            driftx[i] = np.sum(dx[0:i])
            drifty[i] = np.sum(dy[0:i])
      
        edges = np.linspace(
            0, self.num_frames, self.num_blocks, endpoint=False
        )
        centers = []
        for i in range(edges.shape[0] - 1):
            centers.append((edges[i + 1] + edges[i]) / 2.0)

        # Create spline for interpolation.
        driftx = self._interpolate_drift(centers, driftx)
        drifty = self._interpolate_drift(centers, drifty)

        return (driftx, drifty) 


    def run(self, show_plot=False):
        """
        Run drift correction by redundant cross correlation.
        """
        dx, dy = self.calculate_shift()
        A, rij_x, rij_y = self.redundancy(dx, dy)
        self.driftx, self.drifty = self.determine_drift(A, rij_x, rij_y)

        # Plot XY drift.
        if show_plot:
            from matplotlib import pyplot as plt

            x = np.arange(self.num_frames)
            plt.plot(x, self.driftx, color = 'blue')
            plt.plot(x, self.drifty, color = 'red')
            plt.show()

        self.driftz = np.zeros_like(self.driftx)
        if self.correct_z:
            _, dy_xz, time = self.calculate_shift(is_3d=True, z_project='xz')
            _, dy_yz, _ = self.calculate_shift(is_3d=True, z_project='yz')

            dz = np.mean(np.hstack(dy_xz, dy_yz), axis=1)

            _, dz = self.calculate_shift()
            A, rij_z = self.redundancy(dz, dz)
            self.driftz, _ = self.determine_drift(A, rij_z, rij_z)

            # Plot Z drift.
            if show_plot:
                z = np.arange(self.num_frames)
                plt.plot(z, self.driftz, color='black')
                plt.show()

        self.apply_correction()
        return (self.driftx, self.drifty, self.driftz)


def load_nstorm_drift_file(fpath):
    data = np.loadtxt(fpath)
    plt.figure()
    plt.plot(data[:, 1])
    plt.plot(data[:, 2])
    plt.show()

def load_picasso_drift_file(fpath):
    data = np.loadtxt(fpath, skiprows=1)
    plt.figure()
    plt.plot(data[:, 0])
    plt.plot(data[:, 1])
    plt.show()

if __name__=='__main__':

    locs_path = '.\\test_data\\tubulin647 2d.h5'
    # load_nstorm_drift_file('.\\test_data\\tubulin647 2d_drift.txt')
    # load_picasso_drift_file('.\\test_data\\tubulin647 2d_locs_190909_140850_drift.txt')
    params = SMLMParameters()
    params.frame_width = 256
    params.frame_height = 256
    params.camera_pixel = 160.0
    params.scale = 2
    params.blocksize = 1000
    params.z_step = 100.0
    bins = [params.frame_width * params.scale, params.frame_height * params.scale]
    lhdf = LocsHDFStore(locs_path)
    # dcc = DCCDriftCorrection(lhdf, params)
    # dcc.run(show_plot=True)
    rcc = RCCDriftCorrection(lhdf, params)
    dx, dy, _ = rcc.run(show_plot=True)
    lhdf.close()