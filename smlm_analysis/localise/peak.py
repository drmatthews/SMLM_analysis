from math import pi, sqrt

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
import tqdm

from . import filters
from . import gaussian
from . import phasor

from ..utils.locs_hdfstore import LocsHDFStore

NUM_COLUMNS = 9
COLUMNS = ['x', 'y', 'z', 'wx', 'wy', 'uncertainty', 'frame', 'intensity', 'detections']

'''
uncertainty calculation from thunderstorm
        /**
         * Returns lateral uncertainty in nanometers.
         *
         * Thompson, et al. 2002; corrected by Mortensen, et al. 2010 (16/9 instead of 1 to not underestimate)
         * compensation for EM gain worked out by Quan, et al. 2010
         */
        public static double uncertaintyXY(Molecule molecule) throws UncertaintyNotApplicableException {
            double psfSigma2;
            if(molecule.hasParam(LABEL_SIGMA)) {    // 2D (symmetric Gaussian)
                psfSigma2 = molecule.getParam(LABEL_SIGMA, Units.NANOMETER)  * molecule.getParam(LABEL_SIGMA, Units.NANOMETER);
            } else if(molecule.hasParam(LABEL_SIGMA1) && molecule.hasParam(LABEL_SIGMA2)) { // 3D astigmatism (elliptic Gaussian)
                psfSigma2 = molecule.getParam(LABEL_SIGMA1, Units.NANOMETER)  * molecule.getParam(LABEL_SIGMA2, Units.NANOMETER);
            } else {
                throw new UncertaintyNotApplicableException("Missing parameter - `sigma`!");
            }
            double gain, readout;
            if (CameraSetupPlugIn.getIsEmGain()) {  // EMCCD
                gain = 2.0; // correction factor by Quan
                readout = 0.0;
            } else {    // CCD or sCMOS
                gain = 1.0;
                readout = CameraSetupPlugIn.getReadoutNoise();
            }

            double pixelSize = CameraSetupPlugIn.getPixelSize();
            double psfPhotons = molecule.getParam(LABEL_INTENSITY, Units.PHOTON) * CameraSetupPlugIn.getQuantumEfficiency();
            double bkgStd = molecule.getParam(LABEL_BACKGROUND, Units.PHOTON) * CameraSetupPlugIn.getQuantumEfficiency() + readout;
            double tau = 0.0;

            String fittingMethod = null;
            MeasurementProtocol protocol = IJResultsTable.getResultsTable().getMeasurementProtocol();
            if (protocol.analysisEstimator instanceof EllipticGaussianEstimatorUI) {    // 3D? elliptic Gauss (astigmatism)
                fittingMethod = ((EllipticGaussianEstimatorUI) protocol.analysisEstimator).getMethod();
                DaostormCalibration cal = ((EllipticGaussianEstimatorUI) protocol.analysisEstimator).getDaoCalibration();
                double l2 = abs(cal.getC1() * cal.getC2());
                double d2 = abs(cal.getD1() * cal.getD2());
                tau = 2.0 * PI * bkgStd*bkgStd * (psfSigma2*(1.0 + l2/d2) + pixelSize*pixelSize/12.0) / (psfPhotons * pixelSize*pixelSize);
            } else if (protocol.analysisEstimator instanceof SymmetricGaussianEstimatorUI) {    // 2D? Gauss or IntGauss
                fittingMethod = ((SymmetricGaussianEstimatorUI) protocol.analysisEstimator).getMethod();
                tau = 2.0 * PI * bkgStd*bkgStd * (psfSigma2 + pixelSize*pixelSize/12.0) / (psfPhotons * pixelSize*pixelSize);
            }

            if (fittingMethod != null) {
                if (fittingMethod.equals(SymmetricGaussianEstimatorUI.MLE)
                 || fittingMethod.equals(SymmetricGaussianEstimatorUI.WLSQ)) {
                    // Note: here we don't distinguish between MLE and WLSQ, however, there is a difference!
                    //       For details, see supplementary note for Mortensen 2010, Eq. (46), which shows an extra offset-dependent term!
                    return sqrt((gain * psfSigma2 + pixelSize*pixelSize/12.0) / psfPhotons * (1.0 + 4.0*tau + sqrt(2.0*tau/(1 + 4.0*tau))));
                } else if (fittingMethod.equals(SymmetricGaussianEstimatorUI.LSQ)) {
                    return sqrt((gain * psfSigma2 + pixelSize*pixelSize/12.0) / psfPhotons * (16.0/9.0 + 4.0*tau));
                }
            }
            throw new UncertaintyNotApplicableException("Unsupported fitting method! Was the measurement protocol loaded properly?");
        }

        /**
         * Returns axial uncertainty in nanometers.
         *
         * Here we always assume MLE fit --> the math has been worked out by Rieger, et al. 2014.
         * When we distinguish between (W)LSQ and MLE, we could work out the math for LSQ in a similar fashion
         * as Thompson, et al. 2002 and/or Mortensen, et al. 2010.
         */
        public static double uncertaintyZ(Molecule molecule) throws UncertaintyNotApplicableException {
            MeasurementProtocol protocol = IJResultsTable.getResultsTable().getMeasurementProtocol();
            if (!(protocol.analysisEstimator instanceof EllipticGaussianEstimatorUI)
                    || !(molecule.hasParam(LABEL_SIGMA1) && molecule.hasParam(LABEL_SIGMA2))) {
                throw new UncertaintyNotApplicableException("Axial uncertainty cannot be calculated for 2D estimate (missing sigma1, sigma2)!");
            }

            double gain, readout;
            if (CameraSetupPlugIn.getIsEmGain()) {  // EMCCD
                gain = 2.0; // correction factor by Quan
                readout = 0.0;
            } else {    // CCD or sCMOS
                gain = 1.0;
                readout = CameraSetupPlugIn.getReadoutNoise();
            }
            double pixelSize = CameraSetupPlugIn.getPixelSize();
            double psfPhotons = molecule.getParam(LABEL_INTENSITY, Units.PHOTON) * CameraSetupPlugIn.getQuantumEfficiency();
            double bkgStd = molecule.getParam(LABEL_BACKGROUND, Units.PHOTON) * CameraSetupPlugIn.getQuantumEfficiency() + readout;
            double psfSigma1 = molecule.getParam(LABEL_SIGMA1, Units.NANOMETER);
            double psfSigma2 = molecule.getParam(LABEL_SIGMA2, Units.NANOMETER);
            double zCoord = molecule.hasParam(LABEL_Z_REL)
                    ? molecule.getParam(LABEL_Z_REL, Units.NANOMETER)
                    : molecule.getParam(LABEL_Z, Units.NANOMETER);

            DaostormCalibration cal = ((EllipticGaussianEstimatorUI) protocol.analysisEstimator).getDaoCalibration();
            double l2 = abs(cal.getC1() * cal.getC2());
            double d2 = abs(cal.getD1() * cal.getD2());
            double tau = 2.0 * PI * bkgStd*bkgStd * (psfSigma1*psfSigma2*(1.0 + l2/d2) + pixelSize*pixelSize/12.0) / (psfPhotons * pixelSize*pixelSize);
            double zLimit = sqrt(l2 + d2);  // singularity in CRLB - do not evaluate at positions beyond
            if (abs(zCoord) >= zLimit) return Double.POSITIVE_INFINITY;
            //
            double compensation = (sqrt(gain * psfSigma1*psfSigma1 + pixelSize*pixelSize/12.0) / psfSigma1
                                +  sqrt(gain * psfSigma2*psfSigma2 + pixelSize*pixelSize/12.0) / psfSigma2)
                                / 2.0;  // finite pixel size and em gain compensation
            //
            double stdSigma;  // method-dependent parameter
            String fittingMethod = ((EllipticGaussianEstimatorUI) protocol.analysisEstimator).getMethod();
            if (fittingMethod != null) {
                if (fittingMethod.equals(SymmetricGaussianEstimatorUI.MLE)
                        || fittingMethod.equals(SymmetricGaussianEstimatorUI.WLSQ)) {
                    // Note: here we don't distinguish between MLE and WLSQ, however, there is a difference!
                    //       For details, see supplementary note for Mortensen 2010, Eq. (46), which shows an extra offset-dependent term!
                    stdSigma = sqrt(1 + 8.0 * tau + sqrt(9.0 * tau / (1.0 + 4.0 * tau))) * compensation / sqrt(psfPhotons);

                } else if (fittingMethod.equals(SymmetricGaussianEstimatorUI.LSQ)) {
                    stdSigma = sqrt(1 + 8.0 * tau) * compensation / sqrt(psfPhotons);
                } else {
                    throw new UncertaintyNotApplicableException("Unsupported (unknown) fitting method! Was the measurement protocol loaded properly?");
                }
            } else {
                throw new UncertaintyNotApplicableException("Unsupported (empty) fitting method! Was the measurement protocol loaded properly?");
            }
            //
            double Fsq = 4.0 * l2 * zCoord*zCoord / sqr(l2 + d2 + zCoord*zCoord);
            double stdF = sqrt(1.0 - Fsq) * stdSigma;
            double stdZ = stdF * sqr(l2 + d2 + zCoord*zCoord) / (2.0 * sqrt(l2) * (l2 + d2 - zCoord*zCoord));
            return stdZ;
        }
'''
class Peak:
    def __init__(self, is_3d, camera_pixel, photon_conversion):

        self.x = None
        self.y = None
        self.z = 0.0
        self.t = None
        self.wx = None
        self.wy = None
        self.uncertainty = None
        self.intensity = None
        self.bg = None
        self.frame_no = None
        self.is_3d = is_3d
        self.camera_pixel = camera_pixel
        self.photon_conversion = photon_conversion

    def calculate_tau_z(self):
        N = self.intensity
        a = 160.0 # camera pixel size - need to pass in cam parameters somehow
        l2 = 1 # what is this?
        d2 = 1 # what is this?
        sig = (self.wx + self.wy) / 2.0
        tz = 2 * pi * self.bg \
             * (sig**2 * (1 + (l2 / d2))+ (a**2 / 12)) \
             / N * a**2
        return tz

    def uncertainty_ls(self):
        N = self.intensity
        a = 160.0 # camera pixel size again
        sig = (self.wx + self.wy) / 2.0
        print(sig)
        t = 2 * pi * self.bg \
            * (sig**2 + (a**2 / 12)) \
            / N * a**2
        print(t)
        dx2 = (sig**2 + (a**2 / 12)) / N \
              * (16 / 9 + 4 * t)
        print(dx2)
        return sqrt(dx2)

    def uncertainty_mle(self):
        N = self.intensity
        a = 160.0 # camera pixel size again
        sig = (self.wx + self.wy) / 2.0

        t = 2 * pi * self.bg \
             * (sig**2 + (a**2 / 12)) \
             / N * a**2

        dx2 = (sig**2 + (a**2 / 12)) / N \
              * (16 / 9 + 4 * t + sqrt(2 * t / (1 + 4 * t)))
        return sqrt(dx2)

    def uncertainty_z(self):
        l = 1
        l2 = 1
        d2 = 1
        N = self.intensity

        a = 160.0 # camera pixel size - need to pass in cam parameters somehow
        sig = (self.wx + self.wy) / 2.0

        t = 2 * pi * self.bg \
             * (sig**2 * (1 + (l2 / d2))+ (a**2 / 12)) \
             / N * a**2

        dz2 = (l2 + d2 / 2 * l * N) \
              * sqrt(1 + 8 * t + sqrt(9 * t / (1 + 4 * t)))
        return sqrt(dz2)

    # def _to_photons(self, img, baseline, sensitivity, gain, qe):
    #     # at the moment a simply conversion factor is being used
    #     # instead of this
    #     img = np.float32(img)
    #     return (img - baseline) * sensitivity / (gain * qe)        
    def _to_photons(self, img):
        return img * self.photon_conversion

    def _to_nm(self, val):
        return val * self.camera_pixel

    def fit(self):
        # subclasses must override
        pass


class PhasorPeak(Peak):
    def __init__(self, is_3d, window,
                 camera_pixel, photon_conversion,
                 method=None, psf_shape=None,
                 max_iter=None, tol=None):

        super().__init__(is_3d, camera_pixel, photon_conversion)
        self.window = window

    def fit(self, img, reference, frame_no):

        pf = phasor.PhasorFitter(self.is_3d, self.window)
        pf.fit(img, reference)

        results = pf.fit_params
        self.x = self._to_nm(results[0] + reference[0])
        self.y = self._to_nm(results[1] + reference[1])
        self.wx = self._to_nm(results[2])
        self.wx = self._to_nm(results[3])

        # self.x = results[0] + reference[0]
        # self.y = results[1] + reference[1]
        # self.wx = results[2]
        # self.wx = results[3]
        self.bg = results[4]
        self.intensity = results[5]
        self.good = results[6]
        self.frame_no = frame_no       


class GaussPeak(Peak):
    def __init__(self, is_3d, window,
                 camera_pixel, photon_conversion,
                 method='mle', psf_shape='elliptical',
                 max_iter='100', tol=1e-3):

        super().__init__(is_3d, camera_pixel, photon_conversion)
        self.method = method
        self.psf_shape = psf_shape
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, img, reference, frame_no):

        if self.method == 'mle' or self.method == 'ls':
            if self.is_3d and self.psf_shape == 'elliptical':
                if self.method == 'ls':
                    fitter = gaussian.LSElliptical
                elif self.method == 'mle':
                    fitter = gaussian.MLEElliptical
            elif not self.is_3d and self.psf_shape == 'circular':
                if self.method == 'ls':
                    fitter = gaussian.LSCircular
                elif self.method == 'mle':
                    fitter = gaussian.MLECircular
            else:
                raise ValueError('3D fitting is only applicable with an elliptical psf')
        else:
            raise ValueError('method not applicable')

        gf = fitter(self.tol, self.max_iter)
        gf.fit(img)

        fit_params = gf.fit_params
        self.x = self._to_nm(fit_params[0] + reference[0])
        self.y = self._to_nm(fit_params[1] + reference[1])
        self.intensity = self._to_photons(fit_params[2])
        self.bg = fit_params[3]
        self.wx = self._to_nm(fit_params[4])
        if fit_params.shape == 5:
            self.wy = self.wx
        else:
            self.wy = self._to_nm(fit_params[5])
        self.iterations = fit_params[-2]
        self.good = fit_params[-1]
        self.frame_no = frame_no

        # if 'mle' in self.method:
        #     # pass cam parameters here?
        #     self.uncertainty = self.uncertainty_mle()
        # elif 'ls' in self.method:
        #     self.uncertainty = self.uncertainty_ls()

        # if self.is_3d:
        #     # look up z position from calibration
        #     self.uncertainty_z = self.uncertainty_z()
    

class PeakFinder:
    def __init__(self, parameters, notebook):
        self.params = parameters
        self.notebook = notebook

    def _find_overlapping(self, points):
        tree = KDTree(points, leaf_size=2)
        dist, ind = tree.query(points[:], k=2)
        rmax = self.params.window
        return np.where(dist[:, 1] >= rmax)    

    def _estimate(self, frame):

        # compound filter
        if 'wavelet' in self.params.locate_method:
            order = self.params.wavelet_spline_order
            scale = self.params.wavelet_spline_scale
            cf = filters.CompoundWaveletFilter(order, scale)
            filtered = cf.filter_image(frame.astype(np.float64))
            samples = order
        elif 'uniform' in self.params.locate_method:
            samples = self.params.samples
            cf = filters.CompoundUniformFilter(samples)
            filtered = cf.filter_image(frame.astype(np.float64))
        else:
            raise ValueError('Image filtering method not set')
        print(samples)
        # maximum filter
        mf = filters.MaximumFilter(2 * samples + 1)
        mfilt = mf.filter_image(filtered)

        if 'std' in self.params.threshold_method:
            threshold = np.std(cf.result_f1)
        elif 'manual' in self.params.threshold_method:
            threshold = self.params.threshold
    
        # find local maxima
        mask = filtered == mfilt
        mask &= filtered > threshold
    
        coords = np.nonzero(mask)
        if np.any(coords):
            # remove the ones near the edge
            # of the image
            coords = np.column_stack(coords)
            coords = coords[::-1]
            shape = np.array(frame.shape)
            margin = [10, 10]
            near_edge = np.any(
                (coords < margin) | 
                (coords > (shape - margin - 1)), 1
            )
            coords = coords[~near_edge]
            # remove any that are overlapping
            # keep = _remove_overlaps(coords, window / 2)
            # return coords[keep]
            return coords
        else:
            return np.array([])

    def _refine(self, image, candidates, peak_cls):
        peaks = []
        for pos in candidates:
            method, psf_shape, max_iter, tol = None, None, None, None
            if self.params.refine_method == 'gauss':
                if (hasattr(self.params, 'iter_method') and
                    hasattr(self.params, 'psf')):
        
                    method = self.params.iter_method
                    psf_shape = self.params.psf
                    max_iter = self.params.max_iter
                    tol = self.params.tol
                else:
                    raise ValueError('Gaussian fitting parameters not supplied')
            
            peak = peak_cls(
                self.params.is_3d,
                self.params.window,
                self.params.camera_pixel,
                self.params.photon_conversion,
                method=method,
                psf_shape=psf_shape,
                max_iter=max_iter,
                tol=tol
            ) 
            fit_radius = int((self.params.window - 1) / 2.0)
            reference = pos - fit_radius
            img = image[
                pos[0] - fit_radius: pos[0] + fit_radius + 1,
                pos[1] - fit_radius: pos[1] + fit_radius + 1
            ] * self.params.photon_conversion

            peak.fit(img, reference, image.frame_no)
            if peak.good:
                peaks.append(peak)
            # peaks.append(pos)
        return peaks   

    def locate(self, frame):

        if frame.ndim == 3:
            channel = self.params.channel
            image = frame[channel]
        else:
            image = frame
        
        candidates = self._estimate(image)

        if np.any(candidates):
            if hasattr(self.params, 'refine_method'):
                if 'phasor' in self.params.refine_method:
                    peak_cls = PhasorPeak
        
                elif 'gauss' in self.params.refine_method:
                    peak_cls = GaussPeak
                else:
                    raise ValueError('Unknown position refinement method')
                     
            return self._refine(frame, candidates, peak_cls)
        else:
            return None

    def peaks_to_array(self, peaks):
        if peaks:
            locs = np.zeros((len(peaks), NUM_COLUMNS))
            for p, peak in enumerate(peaks):
                locs[p, 0] = peak.x
                locs[p, 1] = peak.y
                locs[p, 2] = peak.z
                locs[p, 3] = peak.wx
                locs[p, 4] = peak.wy
                locs[p, 5] = peak.uncertainty
                locs[p, 6] = peak.frame_no
                locs[p, 7] = peak.intensity
                locs[p, 8] = 1

            return locs
        else:
            return np.array([])

    def write_localisations(self, localisations, localisations_path):
        pbar = tqdm.tqdm
        if self.notebook:
            pbar = tqdm.tqdm_notebook

        with LocsHDFStore(localisations_path, mode='write') as lhdf:
            lhdf.clear_locs()
            empty_frames = sum([1 for l in localisations if not np.any(l)])
            n_frames = len(localisations) - empty_frames
            n_locs = 0
            collect_locs = []
            for peaks in localisations:
                locs_arr = self.peaks_to_array(peaks)
                if np.any(locs_arr):
                    collect_locs.append(locs_arr)
                    n_locs += len(peaks)
                else:
                    continue

            locs = np.vstack(collect_locs)
            lhdf.write_locs(locs)
            metadata = {}
            metadata['n_locs'] = n_locs
            metadata['n_frames'] = n_frames
            metadata['source'] = self.params.source
            lhdf.write_metadata(metadata)

    def run(self, movie):
        pbar = tqdm.tqdm
        if self.notebook:
            pbar = tqdm.tqdm_notebook

        start = self.params.start_frame
        stop = self.params.stop_frame
        assert(start <= stop)
        func = self.locate
        args = [frame for frame in movie[start:stop]]
        with Pool(processes=cpu_count() - 1) as pool:
            max_ = stop - start
            desc = 'Localising molecules'
            results = list(
                pbar(pool.imap(func, args), total=max_, desc=desc)
            )
        print('')
        self.write_localisations(results, self.params.locs_path)
        num_molecules = sum([len(r) for r in results if r is not None])
        return num_molecules