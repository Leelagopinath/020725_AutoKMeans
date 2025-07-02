# File: src/clustering/base_metric.py

import numpy as np
from abc import ABC, abstractmethod

class DistanceMetric(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate distance between two vectors"""
        pass

    def pairwise_distance(self, X: np.ndarray) -> np.ndarray:
        """Calculate pairwise distance matrix (optional)"""
        n = len(X)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = self(X[i], X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix

    def centroid(self, points: np.ndarray) -> np.ndarray:
        """Compute centroid for a set of points (default: mean)"""
        return np.mean(points, axis=0)

    @property
    def name(self) -> str:
        return self.__class__.__name__