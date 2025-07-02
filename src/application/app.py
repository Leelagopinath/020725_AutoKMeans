# File: src/application/app.py

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
                
from src.data.loader import load_preprocessed_data
from src.data.sample_data import get_sample_data
from src.clustering.engine import KMeansEngine
from src.visualization.cluster_plot import plot_clusters_2d
from src.visualization.convergence_plot import plot_convergence_history
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


# Initialize
config = load_config()
logger = get_logger(__name__)
st.set_page_config(page_title="K-Means Automation", layout="wide")

def main():
    st.title("Production Grade K-Means Clustering")
    st.markdown("Universal Automation Platform with 40+ Distance Measures for Advanced Data Analysis")
    
    # Data Loading Section
    st.sidebar.header("Data Input")
    data_option = st.sidebar.radio(
        "Data Source:",
        ["Sample Dataset", "Preprocessed Dataset"]
    )
    
    if data_option == "Sample Dataset":
        dataset_option = st.sidebar.selectbox(
            "Choose Sample Dataset:",
            ["Iris Dataset (2D)", "Random Blobs", "Two Moons"]
        )
        X, dataset_info = get_sample_data(dataset_option)
    else:
        dataset_files = [f for f in os.listdir("data/preprocessed") if f.endswith(".csv")]
        if not dataset_files:
            st.warning("No preprocessed datasets found. Using sample data instead.")
            X, dataset_info = get_sample_data("Iris Dataset (2D)")
        else:
            dataset_option = st.sidebar.selectbox(
                "Choose Preprocessed Dataset:",
                dataset_files
            )
            X, dataset_info = load_preprocessed_data(os.path.join("data/preprocessed", dataset_option))
    
    st.sidebar.markdown(f"**{dataset_info['name']}**")
    st.sidebar.markdown(dataset_info["description"])
    st.sidebar.markdown(f"**Points:** {len(X)}")
    st.sidebar.markdown(f"**Dimensions:** {X.shape[1]}")
    
    # Display sample data
    with st.expander("Data Preview"):
        if isinstance(X, pd.DataFrame):
            st.dataframe(X.head())
        else:
            st.write(pd.DataFrame(X[:5], columns=[f"Feature {i+1}" for i in range(X.shape[1])]))
    
    # Distance Measure Selection
    st.header("Distance Measure Selection")
    metric_category = st.selectbox(
        "Select Metric Category:",
        [
            "Numeric/Vector Measures", 
            "Binary/Categorical Measures",
            "Distribution/Histogram Measures",
            "Sequence/Time-Series Measures",
            "Mixed-Type Measures",
            "Graph & Structure Measures",
            "Universal/Compression-Based"
        ]
    )
    
    # Metric selection based on category
    metric_name = select_metric(metric_category)
    
    # Clustering Parameters
    st.header("Clustering Parameters")
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Number of Clusters (K)", 2, 20, 3)
    with col2:
        max_iter = st.slider("Max Iterations", 10, 1000, 100)
    
    # Run Clustering
    run_button = st.button("Run K-Means Clustering", type="primary")
    
    if run_button:
        with st.spinner(f"Clustering with {metric_name}..."):
            try:
                engine = KMeansEngine(
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    metric_name=metric_name,
                    X=X  # Pass data for metric initialization
                )
                results = engine.fit_predict(X)
                
                display_results(results, metric_name, n_clusters, max_iter)
                
                # Visualization
                if X.shape[1] >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Cluster Visualization")
                        fig = plot_clusters_2d(X, results['labels'], results['centroids'])
                        st.pyplot(fig)
                        
                    with col2:
                        st.subheader("Convergence History")
                        fig = plot_convergence_history(results['history'])
                        st.pyplot(fig)
                else:
                    st.warning("Cluster visualization requires at least 2 dimensions")
                
                # Detailed Analysis
                st.subheader("Detailed Analysis")
                with st.expander("Cluster Assignments"):
                    for i in range(min(10, len(X))):
                        st.markdown(f"**Point {i+1}:** {X[i]} → Cluster {int(results['labels'][i])}")
                
                with st.expander("Cluster Centroids"):
                    for i, centroid in enumerate(results['centroids']):
                        st.markdown(f"**Cluster {i+1}:** {centroid}")
                
                # Export functionality
                if st.button("Export Results"):
                    export_results(X, results)
                    
            except Exception as e:
                st.error(f"Clustering failed: {str(e)}")
                logger.exception("Clustering error")

def select_metric(category):
    metrics = {
        "Numeric/Vector Measures": [
            "Euclidean", "Manhattan", "Minkowski", "Chebyshev", 
            "Mahalanobis", "Cosine", "Pearson", "Bray-Curtis", "Canberra"
        ],
        "Binary/Categorical Measures": [
            "Hamming", "Jaccard", "Tanimoto", "Dice", 
            "Simple Matching", "Levenshtein", "Damerau-Levenshtein", "Jaro-Winkler", "Ochiai"
        ],
        "Distribution/Histogram Measures": [
            "EMD", "Hellinger", "KL Divergence", "Jensen-Shannon", "Bhattacharyya"
        ],
        "Sequence/Time-Series Measures": [
            "DTW", "FastDTW", "ERP", "SBD", "Fréchet", "LCSS"
        ],
        "Mixed-Type Measures": [
            "Gower", "HEOM", "K-Prototypes", "HVDM"
        ],
        "Graph & Structure Measures": [
            "GED", "Spectral Distance"
        ],
        "Universal/Compression-Based": [
            "NCD"
        ]
    }
    return st.selectbox(f"Select {category.split('/')[0]} Metric:", metrics[category])

def display_results(results, metric_name, n_clusters, max_iter):
    st.success("✅ Clustering Completed Successfully!")
    st.subheader("Clustering Results")
    
    # Create metrics display
    col1, col2, col3 = st.columns(3)
    col1.metric("Distance Measure", metric_name)
    col2.metric("Clusters", n_clusters)
    col3.metric("Iterations", results['n_iter'])
    
    col1, col2 = st.columns(2)
    col1.metric("Inertia", f"{results['inertia']:.4f}")
    col2.metric("Silhouette Score", f"{results['silhouette']:.4f}")
    
    # Interpretation
    st.markdown("**Interpretation:**")
    st.markdown(f"- **Inertia:** Lower values indicate tighter clusters (current: {results['inertia']:.2f})")
    st.markdown(f"- **Silhouette Score:** Ranges from -1 to 1. Higher values indicate better separation")
    if results['silhouette'] > 0.5:
        st.success("Strong cluster structure (Silhouette > 0.5)")
    elif results['silhouette'] > 0.25:
        st.info("Reasonable cluster structure (Silhouette > 0.25)")
    else:
        st.warning("Weak cluster structure. Consider fewer clusters or different metric")

def export_results(X, results):
    # Create a DataFrame with original data and cluster assignments
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
    else:
        df = X.copy()
    
    df['Cluster'] = results['labels']
    
    # Add centroid information
    for i, centroid in enumerate(results['centroids']):
        centroid_df = pd.DataFrame([centroid], columns=df.columns[:-1])
        centroid_df['Cluster'] = f"Centroid_{i+1}"
        df = pd.concat([df, centroid_df], ignore_index=True)
    
    # Save to CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='clustering_results.csv',
        mime='text/csv'
    )
    st.success("Results exported successfully!")

if __name__ == "__main__":
    main()