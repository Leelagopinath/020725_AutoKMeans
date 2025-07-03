# File: src/clustering/distance_selector.py
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine, correlation
import tslearn.metrics as tsm
import gower
import pyemd
import Levenshtein

_CATEGORIES = {
     'Numeric / Vector': ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 
                         'mahalanobis', 'cosine', 'pearson', 'braycurtis', 'canberra'],
    'Binary / Categorical': ['hamming', 'jaccard', 'tanimoto', 'dice', 
                             'simple_matching', 'levenshtein', 'damerau_levenshtein',
                             'jaro_winkler', 'ochiai'],
    'Distribution / Histogram': ['emd', 'hellinger', 'kl_divergence', 
                                 'jensen_shannon', 'bhattacharyya'],
    'Sequence / Time-Series': ['dtw', 'fastdtw', 'erp', 'shape_based', 
                               'frechet', 'lcss'],
    'Mixed-Type': ['gower', 'heom', 'k_prototypes', 'hvdm'],
    'Graph & Structure': ['graph_edit', 'spectral'],
    'Universal / Compression': ['ncd']
}

def get_categories():
    return list(_CATEGORIES.keys())

def get_supported_metrics(category):
    return _CATEGORIES.get(category, [])

def is_euclidean(metric):
    return metric == 'euclidean'


def is_sklearn_metric(metric_name):
    """Check if a metric is natively supported by scikit-learn"""
    sklearn_metrics = [
        'euclidean', 'manhattan', 'chebyshev', 'minkowski', 
        'cosine', 'correlation', 'braycurtis', 'canberra'
    ]
    return metric_name.lower() in sklearn_metrics