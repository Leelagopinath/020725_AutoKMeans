# File: tests/test_engine.py
import numpy as np
from src.clustering.engine import cluster_data

def test_cluster_data_euclidean():
    # Simple 2-cluster data
    X = np.array([[0], [0], [10], [10]])
    labels, info = cluster_data(X, metric='euclidean', n_clusters=2, max_iter=10)
    assert set(labels) == {0, 1}
    assert info['n_clusters'] == 2
    assert 'inertia_history' in info['history']