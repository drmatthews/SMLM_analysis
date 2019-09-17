import cProfile
import re
import pstats
from pstats import SortKey

from .linking import link

lpath = '.\\test_data\\tetra_speck_beads_time_lapse_0.h5'
cProfile.run('link(lpath, 0, 100, radius=0.5 * 160.0)', 'link_stats')

p = pstats.Stats('link_stats')
p.strip_dirs().sort_stats(-1).print_stats()