import numpy as np
import graph_cut

# source, sink
terminal_weights = np.array([[16,0],[13,0],[0,20],[0,4]], dtype=float)

# From, To, Capacity, Rev_Capacity
edge_weights = np.array([[0,1,10,4], [0,2,12,0], [1,2,0,9], [1,3,14,0], [2,3,0,7]], dtype=float)

(cut, labels) = graph_cut.graph_cut(terminal_weights, edge_weights)
print (cut, labels)
