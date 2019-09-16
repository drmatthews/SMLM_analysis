# run using 'python setup.py build_ext --inplace'
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


setup(
    package_dir={'smlm_analysis\\localise': ''},
	include_dirs = [numpy.get_include(), "."],
    ext_modules = cythonize(["smlm_analysis\\localise\\mle.pyx", 
                             "smlm_analysis\\localise\\gauss_guess.pyx",
                             "smlm_analysis\\localise\\convolution.pyx"], annotate=True)
)