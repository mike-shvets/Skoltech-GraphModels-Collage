from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import numpy

extensions = [Extension('graph_cut',
  sources=['graph_cut.pyx', 'src/graph.cpp', 'src/maxflow.cpp'],
  language='c++',
  include_dirs=['src', numpy.get_include()])]

setup(
    ext_modules = cythonize(extensions)
)
