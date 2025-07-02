# File: src/clustering/distance_metrics/graph_structural.py

import numpy as np
from src.clustering.base_metric import DistanceMetric
import networkx as nx
from scipy.linalg import eigvals

class GraphEditDistance(DistanceMetric):
    def __call__(self, x, y):
        # Simplified GED - in production use dedicated graph libraries
        # Inputs should be adjacency matrices
        G1 = nx.from_numpy_array(x)
        G2 = nx.from_numpy_array(y)
        
        # Approximate edit distance
        return abs(len(G1.nodes) - len(G2.nodes)) + abs(len(G1.edges) - len(G2.edges))

class SpectralDistance(DistanceMetric):
    def __call__(self, x, y):
        # Compare eigenvalue spectra of adjacency matrices
        eig_x = np.sort(np.abs(eigvals(x)))
        eig_y = np.sort(np.abs(eigvals(y)))
        
        # Pad with zeros if different sizes
        max_len = max(len(eig_x), len(eig_y))
        eig_x_padded = np.pad(eig_x, (0, max_len - len(eig_x)))
        eig_y_padded = np.pad(eig_y, (0, max_len - len(eig_y)))
        
        return np.linalg.norm(eig_x_padded - eig_y_padded)