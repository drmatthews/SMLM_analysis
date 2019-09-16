import os
import yaml
import param


class SMLMParameters(param.Parameterized):
    # movie
    source = param.ObjectSelector(
        default='nstorm',
        objects=['nstorm', 'elyra', 'homemade'],
        doc='the software that was used to create the movie'
    )
    # channel
    channel = param.Integer(
        0, bounds=(0, 3),
        doc="Which channel to process"
    )    
    # is this a 3D acquisition
    is_3d = param.Boolean(
        default=False, bounds=(0,1),
        doc="The type of acquisition - 2D or 3D"
    )
    # if it is, what is the step size
    z_step = param.Number(
        50.0, bounds=(50.0, 100.0),
        doc="z step size for 3D calibration"
    )
    # are we doing a z calibration
    is_zcalibration = param.Boolean(
        default=False, bounds=(0,1),
        doc="Are we making a z calibration?"
    )
    zcal_type = param.ObjectSelector(
        default='fourth_poly',
        objects=['fourth_poly', 'zdefocus'],
        doc="The curve fit method for 3D calibration"
    )
    start_frame = param.Integer(
        0, bounds=(0,None),
        doc="Movie frame on which to start analysis"
    )
    stop_frame = param.Integer(
        1, bounds=(1,None),
        doc="Movie frame on which to stop analysis"
    )
    camera_pixel = param.Number(
        160, doc="Camera pixel size in image plane in nm"
    )
    frame_width = param.Integer(
        256, doc="Raw image width in pixels"
    )
    frame_height = param.Integer(
        256, doc="Raw image height in pixels"
    )
    photon_conversion = param.Number(
        0.45, doc="Factor for converting ADC to photons"
    )
    # locate
    locate_method = param.ObjectSelector(
        default='uniform',
        objects=['wavelet', 'uniform', 'dog'],
        doc=("The method using for preprocessing when"
             "estimating molecule positions")
    )
    # intensity thresholding for object identification
    threshold_method = param.ObjectSelector(
        default='std',
        objects=['std', 'manual', 'something'],
        doc="The thresholding method - 'std' or 'manual'"
    )
    threshold = param.Number(
        0.0, bounds=(0.0, None), doc="Threshold in ADU"
    )
    # locate
    samples = param.Integer(
        3, bounds=(1, 5),
        doc="Number of samples for uniform and maximum filters"
    )
    # wavelet parameters
    wavelet_spline_order = param.Integer(
        3, bounds=(1, 5),
        doc="Spline order for wavelet filter (default is 3.0)"
    )
    wavelet_spline_scale = param.Integer(
        2, bounds=(1,3),
        doc="Spline scale for wavelet filter (default is 2.0)"
    )
    # refine 
    # localisation method used
    refine_method = param.ObjectSelector(
        default='gauss',
        objects=['gauss', 'phasor'],
        doc="The iterative fitting method"
    )
    iter_method = param.ObjectSelector(
        default='mle',
        objects=['mle', 'ls'],
        doc="The type of gaussian fitting"
    )    
    # PSF parameters
    psf = param.ObjectSelector(
        default='circular',
        objects=['circular', 'elliptical'],
        doc="The (Gaussian) modelled PSF shape"
    )
    # iterative 'fitting' parameters
    max_iter = param.Integer(
        500, bounds=(500, 5000),
        doc="Maximum number of iterations to attempt convergance"
    )
    tol = param.Number(
        1e-6, doc="tolerance for curve fitting or log likelihood determination"
    )
    # fitting window (used for phasor or any iterative method)
    window = param.Integer(
        5, bounds=(5, 11),
        doc="Window for molecule localisation (needs to be an odd number)"
    )
    # drift correction
    blocksize = param.Integer(
        1000, bounds=(500, 5000),
        doc="Number of camera frames to be grouped for drift correction"
    )
    # linking
    radius = param.Number(
        0.5, bounds=(0.5, 1.0),
        doc="Search radius for linking in pixels"
    )
    max_gap = param.Integer(
        3, bounds=(0, 10),
        doc="Maximum number of frames a peak can disappear when linking"
    )
    max_length = param.Integer(
        0, bounds=(0, 1000),
        doc="Maximum length of link (0 is unlimited)"
    )
    # reconstruction
    scale = param.Integer(
        5, bounds=(2, 20),
        doc="The upscaling factor to determine frame size in reconstructed image"
    )
    z_pix = param.Number(
        100.0, bounds=(50.0, 200.0),
        doc="Voxel size in z direction (nm) for 3D rendering"
    )
    # io
    movie_path = param.Filename()
    locs_path = param.String()
    zcal_path = param.Filename()

def get_parameters(parameters_path):
    params = SMLMParameters()
    if os.path.exists(parameters_path):
        with open(parameters_path, 'r') as param_file:
            data = yaml.load(param_file, Loader=yaml.FullLoader)

        for k, v in data.items():
            for param in v:
                if hasattr(params, param):
                    setattr(params, param, data[k][param])
    else:
        raise IOError("Parameters file does not exist")

    return params


if __name__=="__main__":
    ppath = "test_data\\tubulin647 2d.yaml"
    params = get_parameters(ppath)
    print(params.locate_method)