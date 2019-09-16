import setuptools
from Cython.Build import cythonize
import numpy


setuptools.setup(
      name='SMLM_analysis',
      version='0.1',
      description='Collection of Python files for processing SMLM data',
      url='http://github.com/drmatthews/SMLM_analysis',
      author='Dan Matthews',
      author_email='dr.dan.matthews@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(exclude=['docs', 'test_data']),
	include_dirs = [numpy.get_include(), "."],
      ext_modules = cythonize(
            ["smlm_analysis\\localise\\mle.pyx", 
             "smlm_analysis\\localise\\gauss_guess.pyx",
             "smlm_analysis\\localise\\convolution.pyx"
             "smlm_analysis\\render\\histogram.pyx"],
             annotate=False
      ),
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3"
     ],
     python_requires='>=3.6'
)