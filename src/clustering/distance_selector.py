# File: src/clustering/distance_selector.py
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine, correlation
import tslearn.metrics as tsm
import gower
import pyemd
import Levenshtein

_CATEGORIES = {
    'Numeric / Vector': ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'correlation'],
    'Binary / Categorical': ['hamming', 'jaccard', 'dice', 'matching', 'levenshtein'],
    # ... other categories ...
}

def get_categories():
    return list(_CATEGORIES.keys())

def get_supported_metrics(category):
    return _CATEGORIES.get(category, [])

def is_euclidean(metric):
    return metric == 'euclidean'