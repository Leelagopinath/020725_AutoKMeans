import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Add root to path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.loader import load_preprocessed_data
from src.data.sample_data import get_sample_data
from src.clustering.engine import KMeansEngine
from src.clustering.benchmark import benchmark_category
from src.clustering.distance_selector import get_supported_metrics, is_sklearn_metric
from src.visualization.cluster_plot import plot_clusters_2d
from src.visualization.convergence_plot import plot_convergence_history
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.data_type_detector import detect_data_type

# Initialize
config = load_config()
logger = get_logger(__name__)
st.set_page_config(page_title="K-Means Automation", layout="wide")


def main():
    st.title("Production Grade K-Means Clustering")
   

    # ------------------ Data Input ------------------
    st.sidebar.header("Data Input")
    data_option = st.sidebar.selectbox(
        "Select Data Input Method:",
        ["Manual Upload", "Sample Datasets", "Manual Entry"]
    )

    X = None
    dataset_info = {}

    if data_option == "Sample Datasets":
        dataset_option = st.sidebar.selectbox(
            "Choose Sample Dataset:",
            ["Iris Dataset", "Random Blobs", "Two Moons"]
        )
        X, dataset_info = get_sample_data(dataset_option)

    elif data_option == "Manual Upload":
        uploaded_file = st.sidebar.file_uploader("Upload Csv,Txt,Xcel,Json File", type=["csv","txt","xls", "xlsx", "json"])
        if uploaded_file:
            try:
                name = uploaded_file.name.lower()
                if name.endswith(('.csv', '.txt')):
                    X = pd.read_csv(uploaded_file)
                elif name.endswith(('.xls', '.xlsx')):
                    X = pd.read_excel(uploaded_file)
                elif name.endswith('.json'):
                    X = pd.read_json(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    X = None
                if X is not None:
                    dataset_info = {"name": uploaded_file.name, "description": "User uploaded dataset"}
            except Exception as e:
                st.error(f"Failed to load file: {e}")
                X = None
        else:
            st.warning("Please upload a data file.")

    else:  # Manual Entry
        num_rows = st.sidebar.number_input("Number of rows", min_value=1, max_value=100, value=5)
        num_features = st.sidebar.number_input("Number of features", min_value=1, max_value=10, value=2)
        st.sidebar.write("Enter your data:")
        data_input = []
        for i in range(num_rows):
            row = [st.sidebar.number_input(f"Row {i+1} - Feature {j+1}", key=f"input_{i}_{j}")
                   for j in range(num_features)]
            data_input.append(row)
        X = pd.DataFrame(data_input, columns=[f"Feature {i+1}" for i in range(num_features)])
        dataset_info = {"name": "Manual Entry", "description": "User manually entered data"}

    # Display dataset info
    if X is not None:
        st.sidebar.markdown(f"**{dataset_info.get('name','Dataset')}**")
        st.sidebar.markdown(dataset_info.get('description',''))
        st.sidebar.markdown(f"**Points:** {len(X)}")
        st.sidebar.markdown(f"**Dimensions:** {X.shape[1]}")
        with st.expander("Data Preview"):
            df_preview = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            st.dataframe(df_preview.head())
    else:
        return

    # -------------- Distance Measure Selection & Mode --------------
    st.header("Distance Measure Selection")
    mode = st.radio("Mode:", ["Manual", "AI Smart"], index=0, horizontal=True)

    if mode == "Manual":
        metric_category = st.selectbox(
            "Select Metric Category:",
            [
                "Numeric/Vector Measures", "Binary/Categorical Measures",
                "Distribution/Histogram Measures", "Sequence/Time-Series Measures",
                "Mixed-Type Measures", "Graph & Structure Measures",
                "Universal/Compression-Based"
            ]
        )
        CATEGORY_MAP = {
        "Numeric/Vector Measures": "Numeric/Vector Measures",
        "Binary/Categorical Measures": "Binary/Categorical Measures",
        "Distribution/Histogram Measures": "Distribution/Histogram Measures",
        "Sequence/Time-Series Measures": "Sequence/Time-Series Measures",
        "Mixed-Type Measures": "Mixed-Type Measures",
        "Graph & Structure Measures": "Graph & Structure Measures",
        "Universal/Compression-Based": "Universal/Compression-Based"
    }
    internal_key = CATEGORY_MAP.get(metric_category, metric_category)
    metrics_list = get_supported_metrics(internal_key)
    if not metrics_list:
        st.error(f"No metrics found for category '{metric_category}'")
        
        return
        metric_name = st.selectbox("Select Metric:", get_supported_metrics(metric_category))
        st.markdown(f"**Using:** `{metric_name}`")
        
    else:
        st.markdown("### 🤖 AI Smart Method Selection")
        category, confidence = detect_data_type(X)
        st.markdown(f"**Data Type Detected:** {category}  •  {confidence*100:.0f}% confidence")
        metrics_available = get_supported_metrics(category)
        st.markdown(f"_{len(metrics_available)} methods will be tested_")
        if not metrics_available:
            st.error("No methods defined for this data type.")
            return

        
        
        n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 20, 3)
        max_iter = st.sidebar.slider("Max Iterations", 10, 1000, 100)
        if st.button("🚀 Run AI Smart Selection"):
            with st.spinner("Benchmarking methods…"):
                results = benchmark_category(X, category, n_clusters, max_iter)
            if not results:
                st.error("No metrics available to benchmark for this data category.")
                return
            best = max(results, key=lambda r: r["silhouette"])
            metric = best['metric']

            # Summary
            col1, col2 = st.columns([2,3])
            with col1:
                st.success("✅ Analysis Complete!")
                st.markdown(f"🏆 **Best Method:** `{best['metric']}`  •  Score: **{best['silhouette']:.3f}**")
            with col2:
                st.markdown("#### 📊 Top 3 Methods")
                for i, r in enumerate(sorted(results, key=lambda x: x['silhouette'], reverse=True)[:3],1):
                    st.write(f"{i}. `{r['metric']}` — {r['silhouette']:.3f}")
            with st.expander("⏱️ Processing Times (ms)"):
                times_df = pd.DataFrame(results)[['metric','time_ms']]
                times_df['time_ms'] = times_df['time_ms'].round().astype(int).astype(str)+"ms"
                st.dataframe(times_df.set_index('metric'))
            metric_name = best['metric']
        else:
            st.info("Click the button to automatically select the best metric.")
            return

    # -------------- Clustering Parameters & Execution --------------
    st.header("Clustering Parameters")
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Number of Clusters (K)", 2, 20, 3)
    with col2:
        max_iter = st.slider("Max Iterations", 10, 1000, 100)

    if st.button("🚀 Run K-Means Clustering"):
        with st.spinner(f"Clustering with {metric_name}…"):
            try:
                engine = KMeansEngine(n_clusters=n_clusters, max_iter=max_iter,
                                      metric_name=metric_name, X=X)
                results = engine.fit_predict(X)
                display_results(results, metric_name, n_clusters)

                # Visualization
                if X.shape[1] >= 2:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Cluster Visualization")
                        fig = plot_clusters_2d(X, results['labels'], results['centroids'])
                        st.pyplot(fig)
                    with c2:
                        st.subheader("Convergence History")
                        fig = plot_convergence_history(results['history'])
                        st.pyplot(fig)
                # Detailed Analysis
                st.subheader("Detailed Analysis")
                with st.expander("Cluster Assignments"):
                    for i in range(min(10, len(X))):
                        st.write(f"Point {i+1}: {X.iloc[i].tolist()} → Cluster {results['labels'][i]}")
                with st.expander("Cluster Centroids"):
                    for idx, ctr in enumerate(results['centroids'],1):
                        st.write(f"Cluster {idx}: {ctr}")
                # Export
                export_results(X, results)
            except Exception as e:
                st.error(f"Clustering failed: {e}")
                logger.exception(e)


def display_results(results, metric_name, n_clusters):
    st.success("✅ Clustering Completed Successfully!")
    st.subheader("Clustering Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Distance Measure", metric_name)
    c2.metric("Clusters", n_clusters)
    c3.metric("Iterations", results['n_iter'])
    c1, c2 = st.columns(2)
    c1.metric("Inertia", f"{results['inertia']:.4f}")
    c2.metric("Silhouette Score", f"{results['silhouette']:.4f}")
    st.markdown("**Interpretation:**")
    if results['silhouette'] > 0.5:
        st.success("Strong cluster structure")
    elif results['silhouette'] > 0.25:
        st.info("Reasonable cluster structure")
    else:
        st.warning("Weak cluster structure; consider different metric/k")


def export_results(X, results):
    df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
    df['Cluster'] = results['labels']
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", data=csv,
                       file_name='clusters.csv', mime='text/csv')

if __name__ == "__main__":
    main()