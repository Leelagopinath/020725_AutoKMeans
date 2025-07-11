#/Users/leelagopinath/Desktop/020725_AutoKMeans/src/clustering/benchmark.py

import time
import numpy as np

from src.clustering.engine import run_clustering
from src.clustering.distance_selector import get_supported_metrics, is_sklearn_metric
from .distance_selector import get_supported_metrics as _orig_get_supported_metrics
from .distance_selector import get_categories

def benchmark_category(X, category, n_clusters, max_iter):
    # Use the same mapping as in app.py
    CATEGORY_MAP = {
        "Numeric/Vector": "Numeric / Vector",
        "Binary/Categorical": "Binary / Categorical",
        "Distribution/Histogram": "Distribution / Histogram",
        "Sequence/Time-Series": "Sequence / Time-Series",
        "Mixed-Type": "Mixed-Type",
        "Graph & Structure": "Graph & Structure",
        "Universal/Compression": "Universal / Compression-Based"
    }
    internal_cat = CATEGORY_MAP.get(category, category)

    metrics = get_supported_metrics(internal_cat)
    
    """
    For a given X and category name, run clustering with every metric in that
    category. Return a list of dicts:
      [{"metric": str, "silhouette": float, "time_ms": float}, ...]
    """

    # If still empty, try alternative names
    if not metrics:
        alt_names = [
            internal_cat.replace(" Measures", ""),
            internal_cat.replace("/", " / "),
            internal_cat.replace("-", " ")
        ]
        for name in alt_names:
            metrics = get_supported_metrics(name)
            if metrics:
                break
    
    # Final fallback to all metrics
    if not metrics:
        all_metrics = []
        for cat in get_categories():
            all_metrics.extend(get_supported_metrics(cat) or [])
        metrics = all_metrics

    results = []
    

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