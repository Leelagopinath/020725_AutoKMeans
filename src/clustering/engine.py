
# File: src/clustering/engine.py
import numpy as np
from sklearn.metrics import silhouette_score
from src.clustering.distance_metrics import get_metric_by_name
from src.utils.logger import get_logger
from scipy.spatial import distance as sdist

logger = get_logger(__name__)

def normalize_metric_name(name: str) -> str:
    """Normalize metric name for case-insensitive matching"""
    return name.strip().lower().replace(' ', '_')

class KMeansEngine:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        metric_name="euclidean",
        random_state: int = 42,
        X: np.ndarray = None
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        # Normalize and resolve metric
        normalized = normalize_metric_name(metric_name)
        raw_metric = None
        try:
            # Try custom registry
            raw_metric = get_metric_by_name(normalized, X=X)
            # Test signature
            raw_metric(np.zeros(X.shape[1]), np.zeros(X.shape[1]))
            self.metric = raw_metric
            logger.info(f"Initialized custom metric: {normalized}")
        except Exception as e:
            # Fallback to scipy.spatial.distance
            alias_map = {
                'manhattan': 'cityblock',
                'pearson': 'correlation'
            }
            key = alias_map.get(normalized, normalized)
            if hasattr(sdist, key):
                self.metric = getattr(sdist, key)
                logger.info(f"Fell back to scipy metric: {key}")
            else:
                msg = f"Could not initialize metric '{metric_name}' (normalized: '{normalized}')"
                logger.error(msg)
                raise ValueError(msg) from e
            if hasattr(sdist, normalized):
                self.metric = getattr(sdist, normalized)
                logger.info(f"Fell back to scipy metric: {normalized}")
            else:
                msg = f"Could not initialize metric '{metric_name}' (normalized: '{normalized}')"
                logger.error(msg)
                raise ValueError(msg)

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        labels = np.zeros(X.shape[0], dtype=int)
        for i, point in enumerate(X):
            dists = [self.metric(point, c) for c in centroids]
            labels[i] = int(np.argmin(dists))
        return labels

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            pts = X[labels == k]
            if len(pts) > 0:
                if hasattr(self.metric, 'centroid'):
                    centroids[k] = self.metric.centroid(pts)
                else:
                    centroids[k] = pts.mean(axis=0)
        return centroids

    def _calculate_inertia(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> float:
        inertia = 0.0
        for k in range(self.n_clusters):
            for pt in X[labels == k]:
                inertia += self.metric(pt, centroids[k])**2
        return inertia

    def fit_predict(self, X: np.ndarray) -> dict:
        centroids = self._initialize_centroids(X)
        labels = None
        last_assigned_labels = None  # Track last assigned labels
        history = []
        
        for i in range(self.max_iter):
            new_labels = self._assign_clusters(X, centroids)
            last_assigned_labels = new_labels  # Always track last assignment
            
            # Handle empty clusters
            unique = set(new_labels)
            if len(unique) < self.n_clusters:
                missing = set(range(self.n_clusters)) - unique
                for k in missing:
                    centroids[k] = X[np.random.randint(len(X))]
                continue  # Skip convergence check after fixing centroids

            new_centroids = self._update_centroids(X, new_labels)
            inertia = self._calculate_inertia(X, new_labels, new_centroids)
            history.append(inertia)

            # Check convergence
            if labels is not None and np.array_equal(labels, new_labels):
                labels = new_labels
                centroids = new_centroids
                break
            if np.allclose(centroids, new_centroids, atol=1e-4):
                labels = new_labels
                centroids = new_centroids
                break

            labels, centroids = new_labels, new_centroids

        # Ensure labels are always set
        if labels is None:
            labels = last_assigned_labels  # Use last assignment
            # Centroids already set during empty cluster handling

        # Final inertia and silhouette
        inertia = self._calculate_inertia(X, labels, centroids)
        sil = -1.0
        try:
            if hasattr(self.metric, 'silhouette_score'):
                sil = self.metric.silhouette_score(X, labels)
            else:
                sil = silhouette_score(X, labels, metric=self.metric)
        except Exception:
            try:
                sil = silhouette_score(X, labels, metric='euclidean')
            except Exception:
                logger.warning("Silhouette score calculation failed, using -1")

        return {
            'labels': labels,
            'centroids': centroids,
            'inertia': inertia,
            'silhouette': sil,
            'n_iter': i+1,
            'history': history
        }


def run_clustering(
    X: np.ndarray,
    n_clusters: int = 3,
    max_iter: int = 100,
    metric_name: str = 'euclidean',
    random_state: int = 42
) -> dict:
    engine = KMeansEngine(
        n_clusters=n_clusters,
        max_iter=max_iter,
        metric_name=metric_name,
        random_state=random_state,
        X=X
    )
    return engine.fit_predict(X)
