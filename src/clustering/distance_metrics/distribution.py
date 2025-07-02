# File: src/clustering/distance_metrics/distribution.py

import numpy as np
from src.clustering.base_metric import DistanceMetric
from scipy.stats import entropy

class EMD(DistanceMetric):
    def __init__(self, X=None):
        # Simplified implementation - for production use pyemd
        self.bins = np.linspace(0, 1, X.shape[1]) if X is not None else None
        
    def __call__(self, x, y):
        # Earth Mover's Distance approximation
        return np.sum(np.abs(np.cumsum(x) - np.cumsum(y)))

class Hellinger(DistanceMetric):
    def __call__(self, x, y):
        p = x / np.sum(x)
        q = y / np.sum(y)
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

class KLDivergence(DistanceMetric):
    def __call__(self, x, y):
        p = x / np.sum(x)
        q = y / np.sum(y)
        # Add epsilon to avoid log(0)
        return entropy(p + 1e-10, q + 1e-10)

class JensenShannon(DistanceMetric):
    def __call__(self, x, y):
        p = x / np.sum(x)
        q = y / np.sum(y)
        m = 0.5 * (p + q)
        return 0.5 * (entropy(p, m) + entropy(q, m))

class Bhattacharyya(DistanceMetric):
    def __call__(self, x, y):
        p = x / np.sum(x)
        q = y / np.sum(y)
        bc = np.sum(np.sqrt(p * q))
        return -np.log(bc) if bc > 0 else float('inf')