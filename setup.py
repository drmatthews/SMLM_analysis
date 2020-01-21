"""
Setup mainly to compile Cython extensions but
also used to install the majority of required
packages and modules.

To use this both numpy and cython have to already
be installed. This obviously needs fixing as installation
is currently a two-step process.
"""

import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
from pathlib import Path

CLUSTER = Path("smlm_analysis/cluster")
LOCALISE = Path("smlm_analysis/localise")
RENDER = Path("smlm_analysis/render")


# extensions = [
#     Extension(
#         "smlm_analysis.localise.mle",
#         [str(LOCALISE / "mle.pyx")]
#     ),
#     Extension(
#         "smlm_analysis.localise.gauss_guess",
#         [str(LOCALISE / "gauss_guess.pyx")]
#     ),
#     Extension(
#         "smlm_analysis.localise.convolution",
#         [str(LOCALISE / "convolution.pyx")]
#     ),
#     Extension(
#         "smlm_analysis.render.histogram", 
#         [str(RENDER / "histogram.pyx")]
#     ),
#     Extension(
#         "smlm_analysis.cluster._focal_inner",
#         [str(CLUSTER / "_focal_inner.pyx")]
#     ),
#     Extension(
#         "smlm_analysis.cluster._voronoi_inner",
#         [str(CLUSTER / "_voronoi_inner.pyx")]
#     )
# ]

setuptools.setup(
    name='SMLM_analysis',
    version='0.1',
    description='Collection of Python files for processing SMLM data',
    url='http://github.com/drmatthews/SMLM_analysis',
    author='Dan Matthews',
    author_email='dr.dan.matthews@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(exclude=['docs', 'test_data']),
    install_requires = [
        'numpy',
        'cython',
        'scipy',
        'pandas',
        'tqdm',
        'matplotlib'
    ],
    include_dirs = [np.get_include(), "."],
    ext_modules = cythonize(
          [str(LOCALISE / "mle.pyx"), 
           str(LOCALISE / "gauss_guess.pyx"),
           str(LOCALISE / "convolution.pyx"),
           str(RENDER / "histogram.pyx"),
           str(CLUSTER / "_focal_inner.pyx"),
           str(CLUSTER / "_voronoi_inner.pyx")],
           annotate=False
    ),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6'
)