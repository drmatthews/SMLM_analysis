# steps
# 1. detect candidate featuures
# 2. refine the positions
# 3. save the positions

# options to accept
# 1. whether to wavelet or uniform filter
# 2. whether to use phasors to estimate positions?
# 3. whether to use a manual threshold or 2*std
# 4. whether to use phasor or iterative fitting
# 5. if using iterative whether to use MLE (default) or least squares
# 6. is it 3D or not

# how is the going to be called?
import os
import argparse

import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import threading
import tqdm

import pims
from nd2reader import ND2Reader

from ..localise import filters
from ..localise import zfitting
from ..localise.peak import PeakFinder
from ..utils.locs_hdfstore import LocsHDFStore
from ..utils.parameters import get_parameters
from ..utils.linking import link
from ..utils.drift_correction import RCCDriftCorrection

pbar = tqdm.tqdm


def linking(params, notebook):
    """
    Merge repeated localisations
    """
    locs_path = params.locs_path
    start = params.start_frame
    stop = params.stop_frame
    radius = params.radius
    max_gap = params.max_gap
    max_length = params.max_length
    link(locs_path, start, stop, radius=radius,
         max_gap=max_gap, max_length=max_length,
         notebook=notebook)


def drift_correction(params, notebook):
    """
    Correct xy and possibly z drift
    """
    with LocsHDFStore(params.locs_path) as lhdf:
        rcc = RCCDriftCorrection(lhdf, params, notebook=notebook)
        rcc.run()


def zfit(params, notebook):
    """
    Determine z positions for 3D datasets
    """
    if params.is_zcalibration:
        folder = os.path.dirname(params.movie_path)
        basename = os.path.splitext(
            os.path.basename(params.movie_path)
        )
        zcal_name = basename + '_zcalib.yaml'
        zcal_path = os.path.join(folder, zcal_name)
        zfitting.calibrate(
            params.locs_path, zcal_path,
            params.z_step, params.zcal_type,
            notebook
        )
    else:
        zfitting.fitz(
            params.locs_path, params.zcal_path,
            params.z_step, params.zcal_type,
            notebook
        )


def run(movie, params, notebook):

    if notebook:
        pbar = tqdm.tqdm_notebook

    pf = PeakFinder(params, notebook)
    num_molecules = pf.run(movie)

    if num_molecules > 0:
        if params.is_3d:
            zfit(params, notebook)
        
        if not params.is_zcalibration:
            drift_correction(params, notebook)
            linking(params, notebook)
    else:
        print('No molecules found in movie')


def analyse_movie(movie_path, parameters_path, notebook=False):
    if os.path.exists(movie_path) and os.path.exists(parameters_path):
        # if the path exists
        if notebook:
            pbar = tqdm.tqdm_notebook
        # check file extension and source from parameters
        # if it is tiff - use tifffile
        # else use pims
        if movie_path.endswith('nd2'):
            movie = pims.open(movie_path)
            if 'c' in movie.sizes:
                # it is a multichannel movie
                movie.bundle_axes = 'cyx'
        elif movie_path.endswith('tif') or movie_path.endswith('tiff'):
            pass

        # parse the parameters
        parameters = get_parameters(parameters_path)

        # after loading movie run the analysis
        try:
            results = run(movie, parameters, notebook)
        except KeyboardInterrupt:
            print("Something went wrong")
    else:
        raise ValueError('movie or parameters file not found')


def test_frame(movie_path, parameters_path, frame_no):
    if os.path.exists(movie_path) and os.path.exists(parameters_path):
        # if the path exists
        # check file extension and source from parameters
        # if it is tiff - use tifffile
        # else use pims
        if movie_path.endswith('nd2'):
            movie = pims.open(movie_path)
            if 'c' in movie.sizes:
                # it is a multichannel movie
                movie.bundle_axes = 'cyx'
        elif movie_path.endswith('tif') or movie_path.endswith('tiff'):
            pass

        if frame_no > 0 and frame_no < len(movie):
            frame = movie[frame_no]
            # parse the parameters
            parameters = get_parameters(parameters_path)
            camera_pixel = parameters.camera_pixel

            pf = PeakFinder(parameters, notebook)
            results = pf.locate(frame)

            if np.any(results):
                from matplotlib import pyplot as plt

                plt.figure()
                plt.imshow(frame)
                for mol in results:
                    plt.plot(mol.y/camera_pixel, mol.x/camera_pixel, 'rx')
                plt.show()
        else:
            raise ValueError('Frame not in movie')
    else:
        raise ValueError('movie or parameters file not found')


if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description='Single molecule localisation'
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
        '-n', '--notebook', action='store_true',
        help='Is the analysis run from the cmd line or a notebook'
    )
    parser.add_argument(
        '-f', '--frame_no', dest='frame_no', default=None,
        help='Frame number in movie on which to test the analysis'
    )    
    args = parser.parse_args()

    movie_path = args.movie_path
    params_path = args.params_path
    notebook = args.notebook

    if args.frame_no:
            test_frame(movie_path, params_path, int(args.frame_no))
    else:
        analyse_movie(movie_path, params_path, notebook)


