# File: src/clustering/distance_metrics/time_series.py

import numpy as np
from src.clustering.base_metric import DistanceMetric
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class DTW(DistanceMetric):
    def __call__(self, x, y):
        # Dynamic Time Warping
        x_flat = x.flatten()
        y_flat = y.flatten()
        distance, _ = fastdtw(x_flat, y_flat, dist=euclidean)
        return distance

class FastDTW(DistanceMetric):
    def __call__(self, x, y):
        # Same as DTW but with radius parameter
        x_flat = x.flatten()
        y_flat = y.flatten()
        distance, _ = fastdtw(x_flat, y_flat, radius=5, dist=euclidean)
        return distance

class ERP(DistanceMetric):
    def __init__(self, gap_penalty=0.5):
        self.gap_penalty = gap_penalty
        
    def __call__(self, x, y):
        # Edit Distance with Real Penalty
        x_flat = x.flatten()
        y_flat = y.flatten()
        n, m = len(x_flat), len(y_flat)
        dtw = np.zeros((n+1, m+1))
        
        for i in range(1, n+1):
            dtw[i][0] = dtw[i-1][0] + abs(x_flat[i-1] - self.gap_penalty)
        for j in range(1, m+1):
            dtw[0][j] = dtw[0][j-1] + abs(y_flat[j-1] - self.gap_penalty)
            
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(x_flat[i-1] - y_flat[j-1])
                dtw[i][j] = min(
                    dtw[i-1][j] + abs(x_flat[i-1] - self.gap_penalty),
                    dtw[i][j-1] + abs(y_flat[j-1] - self.gap_penalty),
                    dtw[i-1][j-1] + cost
                )
        return dtw[n][m]

class SBD(DistanceMetric):
    def __call__(self, x, y):
        # Shape-Based Distance (Normalized Cross-Correlation)
        x_flat = x.flatten()
        y_flat = y.flatten()
        ncc = np.correlate(x_flat, y_flat, mode='full')
        return 1 - np.max(ncc) / np.sqrt(np.sum(x_flat**2) * np.sum(y_flat**2))

class Frechet(DistanceMetric):
    def __call__(self, x, y):
        # Fréchet distance for curves
        from scipy.spatial.distance import cdist
        p = x.reshape(-1, 1)
        q = y.reshape(-1, 1)
        ca = np.full((len(p), len(q)), -1.0)
        
        def _c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            d = cdist(p[i:i+1], q[j:j+1], 'euclidean')[0][0]
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(_c(0, j-1), d)
            elif j == 0:
                ca[i, j] = max(_c(i-1, 0), d)
            else:
                ca[i, j] = max(min(_c(i-1, j), _c(i-1, j-1), _c(i, j-1)), d)
            return ca[i, j]
        
        return _c(len(p)-1, len(q)-1)