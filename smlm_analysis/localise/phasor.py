import os
from math import pi, sqrt
import time

import numpy as np
from matplotlib import pyplot as plt


"""
Note that in ThunderSTORM implementation they normalise the
sub image intensity to the total intensity (in the sub image)
before doing the Fourier transform:

        //First calculate total intensity of matrix, then set each point, normalized by total intensity
        for (int y = 0; y < img.size_x; y++) {
            for (int x = 0; x < img.size_x; x++) {
                totalint += img.values[x+y*img.size_x];
            }
	}
        for (int y = 0; y < img.size_x; y++) {
            for (int x = 0; x < img.size_x; x++) {
                axy[x][y] = img.values[x+y*img.size_x]/totalint;
            }
	}

"""

"""
Snippet from ThunderSTORM version for calculating z position

        if (Astigmatism) {
            // Old astigmatism calibration based on 2 curves
            //Calculate subparts of the 3rd-factor function (done with Wolfram Alpha)
            //double div = abslengthx/abslengthy;
            //double part1 = a1*c1*c1*a2*div-2*a1*c1*a2*c2*div-a1*b1+a1*a2*c2*c2*div+a1*b2*div+b1*a2*div-a2*b2*div*div;
            //double zpos1 = (-1*Math.sqrt(part1)+a1*c1-a2*c2*div)/(a1-a2*div);
            //double zpos2 = (Math.sqrt(part1)+a1*c1-a2*c2*div)/(a1-a2*div);
            //double zpos3 = (c1*c1*a2*div+b1-a2*c2*c2*div+b2*div)/(2*a2*div*(c1-c2));
            //Choose the correct zpos - the one that is closest to 0.
            //zpos3 is chosen if div lies below the curve - error preventing
            //if(div == a1/a2){zpos = zpos3;}
            //else if (Math.abs(zpos2) < Math.abs(zpos1))
            //{
            //    zpos = zpos2;
            //}
            //else{
            //    zpos = zpos1;
            //}
            //Prevent NaN-errors
            //if(Double.isNaN(zpos)){zpos=0;}
            //End old astigmatism calibration 
            
            //New astigmatism calibration: If magn1>magn2, then use sigma1Cali, otherwise sigma2Cali (i.e. 'linear' piece of calibration curves)
            //ATM assuming ThunderSTORM calibration is used!
            //Check if ThunderSTORM calibration is used - if not, don't give any Zpos and warn the user
            if (calname.equals("ThunderSTORM")){
                //If clearly R1, use R1
                //double b = b1;
                //double a = a1;
                if (AmplitudeX/AmplitudeY > 1){//THIS Is the positive part in Christophe's data!
                    if (c1>c2){
                        zpos = -1*Math.sqrt(((AmplitudeX/AmplitudeY)-b1)/a1)+c1;
                    }else{
                        zpos = Math.sqrt(((AmplitudeX/AmplitudeY)-b2)/a2)-c2;
                    }
                }else{ 
                    if (c1>c2){
                        zpos = Math.sqrt(((AmplitudeY/AmplitudeX)-b2)/a2)+c2;
                    }else{
                        zpos = -1*Math.sqrt(((AmplitudeY/AmplitudeX)-b1)/a1)-c1;
                    }
                }
                //Prevent NaN-errors
                if(Double.isNaN(zpos)){zpos=0;}

            }else{
                zpos = 0;
                IJ.log("Please use the ThunderSTORM defocus curve option for calibration curves if using pSMLM-3D!");
            }
"""

"""
Aperture photometry snippet from ThunderSTORM version


        //Get background and signal levels
        # size of signal array
        int totsignalarraysize = 0;
        
        # the total signal
        totsignal = 0;

        # total number of non zero bg pixels
        int totbgnonzeros = 0;

        # total number of zero bg pixels
        int totbgzeros = 0;

        # build an array of bg
        double totbgarray[] = new double[axy.length*axy.length];
        for (int i = 0; i < axy.length; i++) { 
            for (int j = 0; j < axy.length; j++) { 
                //Noise
                if (noisearray[i][j] > 0){
                    totbgarray[i*axy.length+j] = axy[i][j]*totalint;
                    totbgnonzeros += 1;
                }else{
                    totbgzeros +=1;
                }
                //Signal
                if (signalarray[i][j] > 0){
                    totsignal += axy[i][j]*totalint;
                    totsignalarraysize += 1;
                }
            }
        }
        //Get Xth percentile
        Arrays.sort(totbgarray);
        int arraypercentile = (int) Math.ceil(totbgnonzeros*.56);
        double xthpercentile = totbgarray[arraypercentile + totbgzeros];
        
        //Get value for intensity, all > 0
        double intensity = Math.max(0,totsignal - xthpercentile * totsignalarraysize);
        double background = (xthpercentile);
        int id = 0;
        double backgroundstdarr[] = new double[totbgarray.length-(totbgzeros)];
        for (int i = totbgzeros; i < totbgarray.length; i++){
            backgroundstdarr[id] = totbgarray[i];654
            id+=1;
        }
        backgroundstd = getStdDev(backgroundstdarr);
        double[] parameters = new double[]{xpos, ypos, zpos, intensity, background, backgroundstd, abslengthx, abslengthy};
"""


NUM_COLUMNS = 7
COLUMNS = ['x', 'y', 'z', 'uncertainty', 'frame', 'intensity', 'detections']


class PhasorFitter:
    """Use the first phasor of the FFT of the image to find the position
    of the bright spot in the image.

    Reference:
        K. J. A. Martens et al. "Phasor based single-molecule localization
        microscopy in 3D (pSMLM-3D): An algorithm for MHz localization rates
        using standard CPUs", J. Chem. Phys. 148, 123311 (2018);
        https://doi.org/10.1063/1.5005899
    """
    def __init__(self, is_3d, window):
        self.fit_params = np.zeros(7, dtype=np.float64)
        self.aperture(window)
        self.is_3d = is_3d

    def aperture(self, window):
        centersize = window - 2
        ringsize = 2
        start = time.time()
        self.noise = np.zeros((window, window), dtype=np.int32)
        self.signal = np.zeros((window, window), dtype=np.int32)
        self.npixels_noise = 0
        for i in range(window):
            for j in range(window):
                dist = sqrt(
                    abs(i - (window - 1) / 2.0) * abs(i - (window - 1) / 2.0) + 
                    abs(j - (window - 1) / 2.0) * abs(j - (window - 1) / 2.0)
                )
                if dist > (centersize + 1) / 2.0:
                    if dist <= (ringsize + (centersize + 1) / 2.0):
                        self.noise[i, j] = 1
                        self.npixels_noise += 1
                if dist <= (centersize + 1) / 2.0:
                    self.signal[i, j] = 1

    def fit(self, img, reference):
        img = np.divide(img, np.sum(img))
        n = 2 * img.shape[0] + 1 
        im_fft = np.fft.fft2(img)
        im_pow = np.abs(np.fft.fftshift(im_fft))**2

        # plt.figure()
        # plt.imshow(np.log10(np.abs(im_fft)**2).astype(np.int64), interpolation='nearest')
        # plt.show()

        # Get the size of the matrix
        window_pixel_size = img.shape[0]

        # Calculate the angle of the X-phasor from the first
        # Fourier coefficient in X

        ang_x = np.angle(im_fft[1, 0])

        # Correct the angle
        if ang_x > 0:
            ang_x = ang_x - 2 * pi

        # Normalize the angle by 2pi and the amount of pixels of the ROI
        position_x = (np.abs(ang_x) / (2 * pi / window_pixel_size))

        # Calculate the angle of the Y-phasor from the first
        # Fourier coefficient in Y
        ang_y = np.angle(im_fft[0, 1])

        # Correct the angle
        if ang_y > 0:
            ang_y = ang_y - 2 * pi

        # Normalize the angle by 2pi and the amount of pixels of the ROI
        position_y = (np.abs(ang_y) / (2 * pi / window_pixel_size))


        # Calculate the magnitude of the X and Y phasors by taking the absolute
        # value of the first Fourier coefficient in X and Y
        # are these PSF widths?
        magnitude_x = np.abs(im_fft[1, 0])
        magnitude_y = np.abs(im_fft[0, 1])

        if self.is_3d:
            # uncalibrated ratio
            self.z = magnitude_y / magnitude_x

        if False:
            plt.figure()
            plt.imshow(img, interpolation='nearest', origin='lower')
            plt.plot(position_y, position_x, 'kx')
            plt.show()

        self.fit_params[0] = position_x
        self.fit_params[1] = position_y
        self.fit_params[2] = magnitude_x
        self.fit_params[3] = magnitude_y

        # # now use aperture to determine the intensity and the bg
        signal_masked = np.ma.masked_where(self.signal == 0, img)
        iraw = np.ma.sum(signal_masked)
        
        noise_masked = np.ma.masked_where(self.noise == 0, img)
        # self.bg = np.percentile(noise_mask, 56) * self.npixels_noise
        bg = np.ma.median(noise_masked) * self.npixels_noise
        self.fit_params[4] = bg
        self.fit_params[5] = iraw - bg
        # good = True
        self.fit_params[6] = True


if __name__=='__main__':
    from ..utils.parameters import SMLMParameters

    import pims
    from nd2reader import ND2Reader
    from tifffile import imsave

    movie_path = ".\\test_data\\tetra_speck_beads_time_lapse.nd2"
    # movie_path = ".\\test_data\\tubulin647 2d.nd2"

    movie = pims.open(movie_path)
    movie.bundle_axes = 'cyx'
    print(movie[0].shape)
    # imsave(".\\test_data\\frame_10000.tif", movie[10000])
    smlm_params = SMLMParameters()
    smlm_params.frame_width = 256
    smlm_params.frame_height = 256
    smlm_params.camera_pixel = 160.0
    smlm_params.start_frame = 0
    smlm_params.stop_frame = 1#movie.sizes['t']
    smlm_params.type = '2d'
    smlm_params.spline_order = 3
    smlm_params.spline_scale = 2
    smlm_params.threshold_method = 'std'
    smlm_params.window = 5
    movie_dir = os.path.dirname(movie_path)
    movie_filename = os.path.basename(movie_path)
    basename = os.path.splitext(movie_filename)[0]
    smlm_params.locs_path = os.path.join(movie_dir, basename + '_channel_0.h5')

    pf = PhasorPeakFinder(smlm_params)
    frame = movie[0]
    peaks = pf.locate(frame[0,...])

    from matplotlib import pyplot as plt

    plt.figure()
    plt.imshow(frame[0,...])
    for peak in peaks:
        plt.plot(peak.y, peak.x, 'rx')
    plt.show()

    # p = PhasorPeak(7, 0)
