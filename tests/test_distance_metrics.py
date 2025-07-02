# File: tests/test_distance_metrics.py

import numpy as np
import pytest
from src.clustering.distance_metrics import get_metric_by_name

@pytest.fixture
def sample_data():
    return {
        "numeric1": np.array([1.0, 2.0]),
        "numeric2": np.array([3.0, 4.0]),
        "binary1": np.array([1, 0, 1, 0]),
        "binary2": np.array([1, 1, 0, 0]),
        "dist1": np.array([0.3, 0.7]),
        "dist2": np.array([0.6, 0.4]),
        "ts1": np.array([[1, 2], [3, 4]]),
        "ts2": np.array([[5, 6], [7, 8]]),
        "graph1": np.array([[0, 1], [1, 0]]),
        "graph2": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
    }

def test_euclidean(sample_data):
    metric = get_metric_by_name("Euclidean")
    dist = metric(sample_data["numeric1"], sample_data["numeric2"])
    assert pytest.approx(dist, 0.001) == 2.828

def test_cosine(sample_data):
    metric = get_metric_by_name("Cosine")
    dist = metric(sample_data["numeric1"], sample_data["numeric2"])
    assert pytest.approx(dist, 0.001) == 0.016

def test_mahalanobis(sample_data):
    X = np.vstack([sample_data["numeric1"], sample_data["numeric2"]])
    metric = get_metric_by_name("Mahalanobis", X=X)
    dist = metric(sample_data["numeric1"], sample_data["numeric2"])
    assert dist > 0

def test_jaccard(sample_data):
    metric = get_metric_by_name("Jaccard")
    dist = metric(sample_data["binary1"], sample_data["binary2"])
    assert pytest.approx(dist, 0.001) == 0.5

def test_hellinger(sample_data):
    metric = get_metric_by_name("Hellinger")
    dist = metric(sample_data["dist1"], sample_data["dist2"])
    assert pytest.approx(dist, 0.001) == 0.365

def test_dtw(sample_data):
    metric = get_metric_by_name("DTW")
    dist = metric(sample_data["ts1"], sample_data["ts2"])
    assert dist > 0

def test_ncd(sample_data):
    metric = get_metric_by_name("NCD")
    dist = metric(sample_data["binary1"], sample_data["binary2"])
    assert 0 <= dist <= 1

# Add more tests as needed