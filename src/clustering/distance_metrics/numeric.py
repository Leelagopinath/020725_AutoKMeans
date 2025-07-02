# File: src/clustering/distance_metrics/numeric.py

import numpy as np
from scipy.spatial.distance import euclidean, cityblock, chebyshev, braycurtis, canberra
from scipy.stats import pearsonr
from src.clustering.base_metric import DistanceMetric

class Euclidean(DistanceMetric):
    def __call__(self, x, y):
        return euclidean(x, y)

class Manhattan(DistanceMetric):
    def __call__(self, x, y):
        return cityblock(x, y)

class Minkowski(DistanceMetric):
    def __init__(self, p=2):
        self.p = p
        
    def __call__(self, x, y):
        return np.linalg.norm(x - y, ord=self.p)

class Chebyshev(DistanceMetric):
    def __call__(self, x, y):
        return chebyshev(x, y)

class Cosine(DistanceMetric):
    def __call__(self, x, y):
        dot = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return 1 - dot / (norm_x * norm_y) if norm_x > 0 and norm_y > 0 else 0.0

class Mahalanobis(DistanceMetric):
    def __init__(self, X=None):
        if X is not None:
            self.cov = np.cov(X.T)
            try:
                self.inv_cov = np.linalg.inv(self.cov)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                self.inv_cov = np.linalg.pinv(self.cov)
        else:
            self.inv_cov = None
            
    def __call__(self, x, y):
        if self.inv_cov is None:
            raise ValueError("Covariance matrix not initialized")
        diff = x - y
        return np.sqrt(diff.T @ self.inv_cov @ diff)

class Pearson(DistanceMetric):
    def __call__(self, x, y):
        r, _ = pearsonr(x, y)
        return 1 - r

class BrayCurtis(DistanceMetric):
    def __call__(self, x, y):
        return braycurtis(x, y)

class Canberra(DistanceMetric):
    def __call__(self, x, y):
        return canberra(x, y)