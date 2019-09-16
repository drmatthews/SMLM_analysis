import panel as pn
pn.extension()

import numpy as np
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

from nd2reader import ND2Reader
from smlm_analysis.utils.movie_reader import ND2Movie
from smlm_analysis.utils.parameters import SMLMParameters
from smlm_analysis.detection import phasor


path = 'test_data\\tubulin647 2d.nd2'
movie = ND2Reader(path)

def timeseries(frame):
    ds = hv.Dataset(
        (np.arange(movie.frame_shape[0]), np.arange(movie.frame_shape[1]), movie[frame]),
        ['x', 'y'], 'Fluorescence'
    )
    im = ds.to(hv.Image, ['x', 'y'])
    return im.opts(width=500, height=500, cmap='viridis')

dmap = hv.DynamicMap(timeseries, kdims=['frame'])
dmap.redim.range(frame=(0, movie.sizes['t'] - 1))

pn.Row(SMLMParameters.param)