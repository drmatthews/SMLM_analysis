try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


setup(
    package_dir={'smlm_analysis\\frc': ''},
	include_dirs = [numpy.get_include(), "."],
    ext_modules = cythonize(["smlm_analysis\\frc\\_frc_inner.pyx",
                             "smlm_analysis\\frc\\_tukey_inner.pyx"],
                             annotate=False)
)