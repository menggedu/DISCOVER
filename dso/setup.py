from distutils.core import setup
import os
from setuptools import dist

dist.Distribution().fetch_build_eggs(['Cython', 'numpy<=1.19'])

import numpy
from Cython.Build import cythonize

required = [
    "pytest",
    "cython",
    "numpy<=1.19",
    "tensorflow==1.14",
    "numba==0.53.1",
    "sympy",
    "pandas",
    "scikit-learn",
    "click",
    "deap",
    "pathos",
    "seaborn",
    "progress",
    "commentjson",
    "PyYAML",
    # "torch==1.9.1+cu111",
    # "torchaudio==0.9.1",
    # "torchvision==0.10.1+cu111",
    # "tqdm==4.64.1",
    # "xarray==0.20.2",
    # "tensorboard==1.15.0",
    # "pyDOE==0.3.8",
    # "numba==0.53.1"
]

extras = {
    "Numeric": [],
    "PINN":[]
}
extras['all'] = list(set([item for group in extras.values() for item in group]))

setup(  name='dso',
        version='1.0dev',
        description='Deep symbolic PDE identification',
        author='mgd',
        packages=['dso'],
        setup_requires=["numpy", "Cython"],
        ext_modules=cythonize([os.path.join('dso','cyfunc.pyx')]), 
        include_dirs=[numpy.get_include()],
        install_requires=required,
        extras_require=extras
        )
