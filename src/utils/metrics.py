# File: src/utils/metrics.py
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

def compute_metrics(X, labels):
    return {
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels)
    }