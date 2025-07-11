# File: src/visualization/cluster_plot.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Add this at the top of the file

def plot_clusters_2d(X, labels, centroids):
    """
    Create 2D visualization of clusters
    
    Args:
        X: Input data (DataFrame or array)
        labels: Cluster assignments
        centroids: Cluster centroids
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to array if DataFrame
    if isinstance(X, pd.DataFrame):
        X_vals = X.values
        col_names = X.columns.tolist()
    else:
        X_vals = X
        col_names = ['Feature 1', 'Feature 2']
    
    # Scatter plot with colors
    scatter = ax.scatter(X_vals[:, 0], X_vals[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolor='k')
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    
    # Add labels and legend
    ax.set_xlabel(col_names[0] if len(col_names) > 0 else 'Feature 1')
    ax.set_ylabel(col_names[1] if len(col_names) > 1 else 'Feature 2')
    ax.set_title('Cluster Visualization')
    ax.legend()
    
    return fig