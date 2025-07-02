# File: src/clustering/distance_metrics/universal.py

import zlib
import numpy as np
from src.clustering.base_metric import DistanceMetric

class NCD(DistanceMetric):
    def __call__(self, x, y):
        # Convert arrays to bytes
        x_bytes = x.tobytes()
        y_bytes = y.tobytes()
        xy_bytes = x_bytes + y_bytes
        
        Cx = len(zlib.compress(x_bytes))
        Cy = len(zlib.compress(y_bytes))
        Cxy = len(zlib.compress(xy_bytes))
        
        return (Cxy - min(Cx, Cy)) / max(Cx, Cy)