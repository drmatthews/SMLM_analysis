import os
import errno
import argparse
import csv

import numpy as np
from multiprocessing import Pool, cpu_count
from matplotlib import pyplot as plot

import tqdm
import pims
import nd2reader

from ..utils.parameters import parse_parameters
from ..localise import filters


class Estimate:
    """
    Estimate the position of bright spots in a fluorescence
    image using a compound uniform filter. Spots are retained
    if their grey level value is above the threshold.
    """
    def __init__(self, samples, threshold_method, threshold):
        """
        Initialise the position estimator.

        Parameters
        ----------
        samples : int
            Size of kernel used in filtering.
        threshold_method : str
            Threshold method (either `std` or `manual`).
        threshold : float, optional
            Intensity threshold for position estimation - to be used if
            threshold_method is `manual`.            

        """
        self.threshold_method = threshold_method
        self.threshold = threshold
        self.cf = filters.CompoundUniformFilter(samples)
        self.mf = filters.MaximumFilter(2 * samples + 1)

    def run(self, image):
        """
        Perform position estimation on input image.

        Parameters
        ----------
        image : numpy array
            A numpy array representation of an image with float64
            dtype.

        Returns
        -------
        coords : numpy array
            An array of coordinates of bright spots in the input image.
        """
        # use compound filter
        cfilt = self.cf.filter_image(image)

        # use maximum filter
        mfilt = self.mf.filter_image(cfilt)

        # find local maxima
        if 'std' in self.threshold_method:
            threshold = np.std(self.cf.result_f1)

        elif 'manual' in self.threshold_method:
            threshold = self.threshold

        else:
            raise ValueError(
                'Method for thresholding should be `manual` or `std`'
            )

        # retain pixels above threshold
        mask = cfilt == mfilt
        mask &= cfilt > threshold

        coords = np.nonzero(mask)
        if np.any(coords):
            # remove the ones near the edge
            # of the image
            coords = np.column_stack(coords)
            coords = coords[::-1]
            shape = np.array(image.shape)
            margin = [10, 10]
            near_edge = np.any(
                (coords < margin) | 
                (coords > (shape - margin - 1)), 1
            )
            return coords[~near_edge]
        else:
            return np.array([])


def save_positions(path, results):
    """
    Save estimated positions to csv.

    Parameters
    ----------
    path : str
        The path to the csv file.
    results : list
        A list of numpy arrays of esitimated coordinates.
    """
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["x [px]", "y [px]", "Frame"])

        for i, positions in enumerate(results):
            if np.any(positions):
                for coord in positions:
                    writer.writerow(
                        [coord[1], coord[0], i]
                    )

    
def process(movie, start, stop, samples,
            threshold_method, threshold, notebook):
    """
    Run the position estimation using a multi-processing pool.

    Parameters
    ----------
    movie : pims instance
        A multi-frame image sequence.
    samples : int
        The size in pixels of the kernel in the compound filter.
    threshold_method : str
        Either `std` or `manual` [if `std` the threshold is etimated
        from the filtered image].
    threshold : float
        Grey level above which bright spots are retained.
    notebook : bool
        True if the script is run from a Jupyter notebook.

    Returns
    -------
    results : list
        A list of numpy arrays representing the coordinates of the
        bright spots found in each image of the movie.
    """
    pbar = tqdm.tqdm
    if notebook:
        pbar = tqdm.tqdm_notebook

    assert(start <= stop)

    estimator = Estimate(samples, threshold_method, threshold)
    func = estimator.run
    args = [frame.astype(np.float64) for frame in movie[start:stop]]
    with Pool(processes=cpu_count() - 1) as pool:
        max_ = stop - start
        desc = 'Estimating'
        results = list(
            pbar(pool.imap(func, args), total=max_, desc=desc)
        )
    return results


def analyse_movie(movie_path, parameters_path, notebook=True):
    """
    Estimate the position of bright spots in each frame of ND2 movie.

    Parameters
    ----------
    movie_path : str
        Path to movie in ND2 format to be processed.
    parameters_path : str
        Path to file in yaml format holding the paramters for position
        estimation.
    notebook : bool
        True if the script is run from a Jupyter notebook.

    Raises
    ------
    FileNotFoundError
        If `movie_path` does not exist.
    FileNotFoundError
        If `parameters_path` does not exist.
    ValueError
        If the movie being analysed is not in ND2 format.
    """
    if not os.path.exists(movie_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), movie_path
        )        
    
    if not os.path.exists(parameters_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), parameters_path
        )

    if movie_path.endswith('nd2'):
        movie = pims.open(movie_path)
        if 'c' in movie.sizes:
            # it is a multichannel movie
            movie.bundle_axes = 'cyx'
    else:
        raise ValueError('ND2 format movies only')

    # parse the parameters
    parameters = parse_parameters(parameters_path)
    start = parameters.start_frame
    stop = parameters.stop_frame
    samples = parameters.samples
    threshold_method = parameters.threshold_method
    threshold = parameters.threshold

    # run the analysis
    try:
        results = process(
            movie, start, stop, samples,
            threshold_method, threshold, notebook
        )

        if results:
            # write them to file
            basename = os.path.splitext(movie_path)[0]
            results_path = basename + '.csv'
            save_positions(results_path, results)
    except KeyboardInterrupt:
        print("Execution stopped")


def test_frame(movie_path, parameters_path, frame_no):
    """
    Estimate the position of bright spots in a single
    frame of an ND2 movie.

    Parameters
    ----------
    movie_path : str
        Path to movie in ND2 format to be processed.
    parameters_path : str
        Path to file in yaml format holding the paramters for position
        estimation.
    frame_no : int
        The index of the frame to process.

    Raises
    ------
    FileNotFoundError
        If `movie_path` does not exist.
    FileNotFoundError
        If `parameters_path` does not exist.
    ValueError
        If the movie being analysed is not in ND2 format.
    """    
    if not os.path.exists(movie_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), movie_path
        )        
    
    if not os.path.exists(parameters_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), parameters_path
        )

    if movie_path.endswith('nd2'):
        movie = pims.open(movie_path)
        if 'c' in movie.sizes:
            movie.bundle_axes = 'cyx'
    else:
        raise ValueError(
            'Movie format not supported - ND2 format files only'
        )

    if frame_no > 0 and frame_no < len(movie):
        frame = movie[frame_no]

        parameters = parse_parameters(parameters_path)
        samples = parameters.samples
        threshold_method = parameters.threshold_method
        threshold = parameters.threshold

        try:

            estimator = Estimate(samples, threshold_method, threshold)
            results = estimator.run(frame.astype(np.float64))

            if np.any(results):
                from matplotlib import pyplot as plt

                plt.figure()
                plt.imshow(frame)
                for result in results:
                    plt.plot(
                        result[1], result[0], 'rx'
                    )
                plt.show()
        except:
            print("Estimation failed")


if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description='Single molecule position estimation'
    )
    parser.add_argument(
        'movie_path',
        help='Full path to movie being processed'
    )
    parser.add_argument(
        'params_path',
        help='Full path to associated parameters file (yaml)'
    )
    parser.add_argument(
        '-f', '--frame_no', dest='frame_no', default=None,
        help='Frame number in movie on which to test the analysis'
    )    
    args = parser.parse_args()

    movie_path = args.movie_path
    params_path = args.params_path

    if args.frame_no:
        test_frame(movie_path, params_path, int(args.frame_no))
    else:
        analyse_movie(movie_path, params_path, notebook=False)