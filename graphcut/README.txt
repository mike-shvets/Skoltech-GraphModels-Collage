This is a module to calculate min-cut.
Works in Python 2.7.9, 3.4.2

Requires Cython for compilation!

How to use the module:

1) Compile the C extension
python setup.py build_ext --inplace

Note: we provide a compiled extension for Python 2.7.9, Windows x64: graph_cut.pyd

2) Copy the following files to your working directory:
graph_cut.pyd / graph_cut.so

3) Use the module. See example_graph_cut.py for an example, and graph_cut.graph_cut function docs for help.
