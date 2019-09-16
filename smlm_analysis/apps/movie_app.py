import param
import numpy as np
import holoviews as hv
from holoviews import opts

from nd2reader import ND2Reader

from ..utils.parameters import SMLMParameters
from ..detection.phasor import PhasorPeakFinder


class MovieExplorer(param.Parameterized):

    channel = param.Selector(default=0)
    frame = param.Integer(default=0, bounds=(0, None))
    
    def __init__(self, movie, **params):
        super(MovieExplorer, self).__init__(**params)
        self.movie = movie
        self.frame_shape = movie.frame_shape

        self.smlm_params = SMLMParameters()
        self.smlm_params.frame_width = movie.frame_shape[0]
        self.smlm_params.frame_height = movie.frame_shape[1]
        self.smlm_params.camera_pixel = movie.metadata['pixel_microns']
        self.smlm_params.start = 0
        self.smlm_params.stop = movie.sizes['t']
        self.smlm_params.type = '2d'
        self.smlm_params.spline_order = 3
        self.smlm_params.spline_scale = 2
        self.smlm_params.threshold_method = 'std'
        self.smlm_params.fit_radius = 2

        self.param['channel'].objects = [c for c in range(movie.sizes['c'])]
        self.param['frame'].bounds = (0, movie.sizes['t'] - 1)

    def detect_features(self, img):
        pf = PhasorPeakFinder(self.smlm_params)
        peaks = pf.locate(img)
        return self.create_boxes(peaks)
    
    def create_boxes(self, peaks):
        coords = []
        for peak in peaks:
            coords.append((peak.y, peak.x))
        window = int((2 * self.smlm_params.fit_radius) + 1)
        polys = hv.Polygons([
            hv.Box(x, y, (window, window))
            for x, y in coords
        ])
        polys.opts(line_color='yellow', line_width=1, fill_alpha=0.0)
        return polys
        
    @param.depends('frame', 'channel')
    def view(self):
        f_num = self.frame
        channel = self.channel

        frame = self.movie.get_frame(f_num)
        img = frame[channel]

        ds = hv.Dataset(
            (
                np.arange(img.shape[0]),
                np.arange(img.shape[1]),
                img
            ),
            ['x', 'y'], 'Fluorescence'
        )
        im = ds.to(hv.Image, ['x', 'y'])
        im.opts(width=500, height=500, cmap='viridis')

        boxes = self.detect_features(img)
        return hv.Overlay([im * boxes])
