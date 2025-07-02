# File: src/visualization/cluster_plot.py

import matplotlib.pyplot as plt
import numpy as np

def plot_clusters_2d(X, labels, centroids):
    plt.figure(figsize=(10, 6))
    
    # Plot clusters
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            s=50, color=colors[i],
            edgecolor='k', alpha=0.7,
            label=f'Cluster {label}'
        )
    
    # Plot centroids
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        s=250, marker='*',
        color='gold', edgecolor='k',
        label='Centroids'
    )
    
    plt.title('Cluster Visualization', fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()