import time
import numpy as np

from src.clustering.engine import run_clustering
from src.clustering.distance_selector import get_supported_metrics, is_sklearn_metric

def benchmark_category(X, category, n_clusters, max_iter):
    """
    For a given X and category name, run clustering with every metric in that
    category. Return a list of dicts:
      [{"metric": str, "silhouette": float, "time_ms": float}, ...]
    """
    results = []
    metrics = get_supported_metrics(category)

    for m in metrics:
        start = time.perf_counter()
        # Get full result from run_clustering
        result = run_clustering(
            X, 
            metric_name=m,  # Correct parameter name
            n_clusters=n_clusters, 
            max_iter=max_iter
        )
        elapsed = (time.perf_counter() - start) * 1000  # ms

        # Use silhouette score from engine results
        score = result.get('silhouette', -1)  # Default to -1 if missing
        
        results.append({
            "metric": m,
            "silhouette": score,
            "time_ms": elapsed
        })

    return results