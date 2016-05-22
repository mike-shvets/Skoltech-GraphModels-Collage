"""
Python interface to Vladimir Kolmogorov implementation of min-cut algorithm downloadable from:
http://www.cs.ucl.ac.uk/staff/V.Kolmogorov/software.html

The algorithm is described in:
Yuri Boykov and Vladimir Kolmogorov, 'An Experimental Comparison of Min-Cut/Max-Flow Algorithms for
Energy Minimization in Vision', IEEE Transactions on Pattern Analysis and Machine Intelligence, vol.
26, no. 9, pp. 1124-1137, Sept. 2004.

Usage:
import graph_cut
(cut, labels) = graph_cut.graph_cut(terminal_weights, edge_weights)

MATLAB version by Anton Osokin, firstname.lastname@gmail.com, Spring 2011 
Python version by Michael Figurnov, firstname@lastname.ru, March 2015
"""
import numpy as np
cimport numpy as np

cdef extern from "src/graph.h":
  ctypedef int node_id
  cdef enum termtype:
    SOURCE = 0
    SINK = 1
  cdef cppclass Graph[captype, tcaptype, flowtype]:
    Graph(int,int) except +
    node_id add_node()
    void add_edge(node_id, node_id, captype, captype)
    void add_tweights(node_id, tcaptype, tcaptype)
    flowtype maxflow()
    termtype what_segment(node_id)


def graph_cut(np.ndarray[double, ndim=2] terminal_weights not None, np.ndarray[double, ndim=2] edge_weights not None):
  """
  Args:
    term_weights - the edges connecting the source and the sink with regular nodes (np.array, dtype: float, shape: [numNodes, 2])
      term_weights[i, 0] is the weight of the edge connecting the source with node #i
      term_weights[i, 1] is the weight of the edge connecting node #i with the sink

      num_nodes is determined from the size of term_weights.

    edge_weights - the edges connecting regular nodes with each other (np.array, dtype: float, shape: [numNodes, 4])
      edge_weights[i, 2] connects node #edge_weights[i, 0] to node #edge_weights[i, 1]
      edge_weights[i, 3] connects node #edge_weights[i, 1] to node #edge_weights[i, 0]

  Outputs:
    cut - the minimum cut value (type float)
    labels - a numpy vector of length numNodes, where labels[i] is 0 or 1 if node #i belongs to S (source) or T (sink) respectively.
  """
  cdef int num_nodes = terminal_weights.shape[0]
  cdef int num_edges = edge_weights.shape[0]

  cdef double v_from, v_to, weight_from, weight_to

  if num_nodes < 1:
    raise ValueError("The number of nodes is not positive")
  
  if terminal_weights.shape[1] != 2:
    raise ValueError("The first parameter is not of size #nodes x 2")

  if edge_weights.shape[1] != 4:
    raise ValueError("The first parameter is not of size #edges x 2")

  for i in range(num_edges):
    v_from, v_to, weight_from, weight_to = edge_weights[i, 0], edge_weights[i, 1], edge_weights[i, 2], edge_weights[i, 3]
    if not float.is_integer(v_from) or not float.is_integer(v_to):
      raise ValueError("Error in pairwise terms array: wrong vertex index")
    if not 0 <= v_from < num_nodes or not 0 <= v_to < num_nodes:
      raise ValueError("Error in pairwise terms array: wrong vertex index")
    if weight_from < 0 or weight_to < 0:
      raise ValueError("Error in pairwise terms array: negative edge weight")

  cdef Graph[double, double, double] *g = new Graph[double, double, double](num_nodes, num_edges)
  
  for i in range(num_nodes):
    g.add_node()
    g.add_tweights(i, terminal_weights[i, 0], terminal_weights[i, 1])

  for i in range(num_edges):
    v_from, v_to, weight_from, weight_to = edge_weights[i, 0], edge_weights[i, 1], edge_weights[i, 2], edge_weights[i, 3]
    g.add_edge(<int>v_from, <int>v_to, weight_from, weight_to)

  cut = g.maxflow()
  cdef np.ndarray[np.int32_t, ndim=1] labels = np.zeros(shape=num_nodes, dtype=np.int32)
  for i in range(num_nodes):
    labels[i] = g.what_segment(i)
  
  del g

  return (cut, labels)
