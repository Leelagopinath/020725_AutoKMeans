# File: src/clustering/engine.py

import numpy as np
from sklearn.metrics import silhouette_score
from src.clustering.base_metric import DistanceMetric
from src.clustering.distance_metrics import get_metric_by_name
from src.utils.logger import get_logger

logger = get_logger(__name__)

class KMeansEngine:
    def __init__(self, n_clusters=3, max_iter=100, metric_name="Euclidean", 
                 random_state=42, X=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric_name = metric_name
        self.random_state = random_state
        
        try:
            # Initialize metric with dataset if needed
            self.metric = get_metric_by_name(metric_name, X=X)
            logger.info(f"Initialized metric: {metric_name}")
        except Exception as e:
            logger.error(f"Metric initialization failed: {str(e)}")
            raise ValueError(f"Could not initialize metric '{metric_name}': {str(e)}")
    
    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]
    
    def _assign_clusters(self, X, centroids):
        labels = np.zeros(X.shape[0], dtype=int)
        for i, point in enumerate(X):
            min_dist = float('inf')
            for k, centroid in enumerate(centroids):
                dist = self.metric(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = k
        return labels
    
    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                # Use metric-specific centroid calculation if available
                if hasattr(self.metric, 'centroid'):
                    centroids[k] = self.metric.centroid(cluster_points)
                else:
                    centroids[k] = np.mean(cluster_points, axis=0)
        return centroids
    
    def _calculate_inertia(self, X, labels, centroids):
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            for point in cluster_points:
                inertia += self.metric(point, centroids[k]) ** 2
        return inertia
    
    def fit_predict(self, X):
        centroids = self._initialize_centroids(X)
        history = []
        labels = None
        
        for i in range(self.max_iter):
            # Assign points to clusters
            new_labels = self._assign_clusters(X, centroids)
            
            # Check for empty clusters
            unique_labels = np.unique(new_labels)
            if len(unique_labels) < self.n_clusters:
                # Reinitialize empty clusters
                missing = set(range(self.n_clusters)) - set(unique_labels)
                for k in missing:
                    centroids[k] = X[np.random.randint(0, len(X))]
                continue
                
            # Update centroids
            new_centroids = self._update_centroids(X, new_labels)
            
            # Calculate inertia
            inertia = self._calculate_inertia(X, new_labels, new_centroids)
            history.append(inertia)
            
            # Check convergence (no label change or centroid movement)
            if labels is not None and np.array_equal(labels, new_labels):
                break
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
                
            labels = new_labels
            centroids = new_centroids
        
        # Final inertia calculation
        inertia = self._calculate_inertia(X, labels, centroids)
        
        # Calculate silhouette score
        try:
            # Handle custom metrics
            if hasattr(self.metric, 'pairwise_distance'):
                dist_matrix = self.metric.pairwise_distance(X)
                silhouette = silhouette_score(dist_matrix, labels, metric='precomputed')
            else:
                silhouette = silhouette_score(X, labels, metric='euclidean')
        except Exception as e:
            logger.warning(f"Silhouette calculation failed: {str(e)}")
            silhouette = -1  # Indicate failure
        
        return {
            "labels": labels,
            "centroids": centroids,
            "inertia": inertia,
            "silhouette": silhouette,
            "n_iter": i + 1,
            "history": history
        }