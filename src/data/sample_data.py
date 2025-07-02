# File: src/data/sample_data.py

import numpy as np
from sklearn.datasets import (
    load_iris,
    make_blobs,
    make_moons,
    make_circles,
    make_classification
)
from sklearn.preprocessing import StandardScaler


def get_sample_data(option: str):
    """Return sample datasets for demonstration"""
    # Iris Dataset: all 4 features
    if option == "Iris Dataset":
        iris = load_iris()
        X = iris.data
        info = {
            "name": "Iris Dataset",
            "description": "Classic flower dataset with 4 features",
            "points": len(X)
        }
        return X, info

    # Simple random blobs
    elif option == "Random Blobs":
        X, _ = make_blobs(
            n_samples=150,
            centers=3,
            n_features=2,
            cluster_std=1.0,
            random_state=42
        )
        info = {
            "name": "Random Blobs",
            "description": "Synthetic clustered data points",
            "points": len(X)
        }
        return X, info

    # Two moons: crescent shapes
    elif option == "Two Moons":
        X, _ = make_moons(
            n_samples=150,
            noise=0.1,
            random_state=42
        )
        X = StandardScaler().fit_transform(X)
        info = {
            "name": "Two Moons",
            "description": "Crescent-shaped clusters",
            "points": len(X)
        }
        return X, info

    # Concentric circles
    elif option == "Circles":
        X, _ = make_circles(
            n_samples=150,
            noise=0.05,
            factor=0.5,
            random_state=42
        )
        X = StandardScaler().fit_transform(X)
        info = {
            "name": "Circles",
            "description": "Concentric circular clusters",
            "points": len(X)
        }
        return X, info

    # Classification-type clusters
    elif option == "Classification":
        X, _ = make_classification(
            n_samples=150,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42
        )
        info = {
            "name": "Classification Blobs",
            "description": "Clusters from a classification generator",
            "points": len(X)
        }
        return X, info

    # Anisotropic blobs: non-spherical
    elif option == "Anisotropic Blobs":
        X, _ = make_blobs(
            n_samples=150,
            centers=3,
            n_features=2,
            random_state=42
        )
        # apply anisotropic transformation
        transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
        X = X.dot(transformation)
        X = StandardScaler().fit_transform(X)
        info = {
            "name": "Anisotropic Blobs",
            "description": "Elliptical clusters via linear transformation",
            "points": len(X)
        }
        return X, info

    # Default fallback
    else:
        # Default to Iris Dataset
        return get_sample_data("Iris Dataset")
