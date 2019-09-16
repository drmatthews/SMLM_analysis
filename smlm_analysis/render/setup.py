try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


setup(
    package_dir={'smlm_analysis\\rendering': ''},
	include_dirs = [numpy.get_include(), "."],
    ext_modules = cythonize("smlm_analysis\\rendering\\histogram.pyx", annotate=True)
)