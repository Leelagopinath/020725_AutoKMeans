# File: src/clustering/distance_metrics/mixed_type.py

import numpy as np
from src.clustering.base_metric import DistanceMetric
from sklearn.preprocessing import MinMaxScaler

class Gower(DistanceMetric):
    def __init__(self, X=None, categorical=None):
        if X is not None:
            self.ranges = np.ptp(X, axis=0)
            self.categorical = categorical if categorical else np.zeros(X.shape[1], dtype=bool)
        else:
            self.ranges = None
            self.categorical = None
            
    def __call__(self, x, y):
        if self.ranges is None:
            raise ValueError("Gower metric not initialized with data ranges")
            
        total = 0.0
        for i in range(len(x)):
            if self.ranges[i] > 0 and not self.categorical[i]:
                # Numeric feature
                total += abs(x[i] - y[i]) / self.ranges[i]
            else:
                # Categorical feature
                total += 1 if x[i] != y[i] else 0
        return total / len(x)

class HEOM(DistanceMetric):
    def __init__(self, X=None, categorical=None):
        if X is not None:
            self.ranges = np.ptp(X, axis=0)
            self.categorical = categorical if categorical else np.zeros(X.shape[1], dtype=bool)
        else:
            self.ranges = None
            self.categorical = None
            
    def __call__(self, x, y):
        if self.ranges is None:
            raise ValueError("HEOM metric not initialized with data ranges")
            
        total = 0.0
        for i in range(len(x)):
            if self.ranges[i] > 0 and not self.categorical[i]:
                # Numeric feature
                total += ((x[i] - y[i]) / self.ranges[i]) ** 2
            else:
                # Categorical feature
                total += 1 if x[i] != y[i] else 0
        return np.sqrt(total)

class KPrototypes(DistanceMetric):
    def __init__(self, X=None, categorical=None, gamma=0.5):
        self.gamma = gamma
        if X is not None:
            self.numeric_ranges = np.ptp(X[:, ~categorical], axis=0) if any(~categorical) else None
            self.categorical = categorical
        else:
            self.numeric_ranges = None
            self.categorical = None
            
    def __call__(self, x, y):
        if self.numeric_ranges is None or self.categorical is None:
            raise ValueError("K-Prototypes metric not properly initialized")
            
        num_distance = 0.0
        cat_distance = 0.0
        num_idx = 0
        
        for i, is_cat in enumerate(self.categorical):
            if not is_cat:
                # Numeric feature
                if self.numeric_ranges[num_idx] > 0:
                    num_distance += ((x[i] - y[i]) / self.numeric_ranges[num_idx]) ** 2
                num_idx += 1
            else:
                # Categorical feature
                cat_distance += 1 if x[i] != y[i] else 0
                
        return np.sqrt(num_distance) + self.gamma * cat_distance

class HVDM(DistanceMetric):
    def __init__(self, X=None, y=None, categorical=None):
        # Requires class labels for VDM part
        self.X = X
        self.y = y
        self.categorical = categorical
        self.numeric_ranges = np.ptp(X[:, ~categorical], axis=0) if any(~categorical) else None
        
    def __call__(self, x, y):
        if self.X is None or self.y is None:
            raise ValueError("HVDM requires training data with labels")
            
        total = 0.0
        num_idx = 0
        classes = np.unique(self.y)
        
        for i, is_cat in enumerate(self.categorical):
            if not is_cat:
                # Numeric feature
                if self.numeric_ranges[num_idx] > 0:
                    total += ((x[i] - y[i]) / (4 * self.numeric_ranges[num_idx])) ** 2
                num_idx += 1
            else:
                # Categorical feature - VDM part
                vdm = 0.0
                for c in classes:
                    p_x = np.mean(self.y[self.X[:, i] == x[i]] == c)
                    p_y = np.mean(self.y[self.X[:, i] == y[i]] == c)
                    vdm += (p_x - p_y) ** 2
                total += vdm
                
        return np.sqrt(total)