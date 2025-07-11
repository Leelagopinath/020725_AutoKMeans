import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.sample_data import get_sample_data
from src.clustering.engine import KMeansEngine
from src.clustering.benchmark import benchmark_category
from src.clustering.distance_selector import get_supported_metrics,is_sklearn_metric
from src.clustering.distance_selector import get_supported_metrics as _orig_get
from src.clustering.distance_selector import get_categories
from src.visualization.cluster_plot import plot_clusters_2d
from src.visualization.convergence_plot import plot_convergence_history
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.data_type_detector import detect_data_type

def get_supported_metrics(category):
    # 1) Try exact match
    metrics = _orig_get(category)
    if metrics:
        return metrics

    # 2) Otherwise, normalize both sides (strip punctuation/spaces, lowercase)
    def normalize(s: str) -> str:
        return ''.join(ch for ch in s.lower() if ch.isalnum())

    target = normalize(category)
    for real in get_categories():
        if normalize(real) == target:
            return _orig_get(real)
        
     # Try common variations
    variations = [
        category.replace(" Measures", ""),
        category.replace("/", " / "),
        category.replace("-", " ")
    ]
    
    for var in variations:
        metrics = _orig_get(var)
        if metrics:
            return metrics

    # 3) No match
    return []

# Initialize
config = load_config()
logger = get_logger(__name__)
st.set_page_config(page_title="K-Means Automation", layout="wide")

CATEGORY_MAPPING = {
    "Numeric/Vector Measures": "Numeric / Vector",
    "Binary/Categorical Measures": "Binary / Categorical",
    "Distribution/Histogram Measures": "Distribution / Histogram",
    "Sequence/Time-Series Measures": "Sequence / Time-Series",
    "Mixed-Type Measures": "Mixed-Type",
    "Graph & Structure Measures": "Graph & Structure",
    "Universal/Compression-Based": "Universal / Compression"
}


def main():
    st.title("Production Grade K-Means Clustering")
    st.markdown("Universal Automation Platform with 40+ Distance Measures for Advanced Data Analysis")

    # ─── Data Input ─────────────────────────────────────────────────────────────
    st.sidebar.header("Data Input")
    data_option = st.sidebar.selectbox("Select Data Input Method:",
                                       ["Manual Upload", "Sample Datasets", "Manual Entry"])

    X = None
    dataset_info = {}

    if data_option == "Sample Datasets":
        dataset_option = st.sidebar.selectbox("Choose Sample Dataset:",
                                              ["Iris Dataset", "Random Blobs", "Two Moons"])
        X, dataset_info = get_sample_data(dataset_option)

    elif data_option == "Manual Upload":
        uploaded = st.sidebar.file_uploader("Upload data file (csv, txt, xls, xlsx, json)",
                                            type=["csv", "txt", "xls", "xlsx", "json"])
        if not uploaded:
            st.warning("Please upload a data file.")
            return
        name = uploaded.name.lower()
        try:
            if name.endswith((".csv", ".txt")):
                X = pd.read_csv(uploaded)
            elif name.endswith((".xls", ".xlsx")):
                X = pd.read_excel(uploaded)
            else:
                X = pd.read_json(uploaded)
            dataset_info = {"name": uploaded.name, "description": "User uploaded dataset"}
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

    else:  # Manual Entry
        rows = st.sidebar.number_input("Number of rows", 1, 100, 5)
        cols = st.sidebar.number_input("Number of features", 1, 10, 2)
        st.sidebar.write("Enter your data:")
        data = [
            [st.sidebar.number_input(f"R{i+1}C{j+1}", key=f"cell_{i}_{j}") for j in range(cols)]
            for i in range(rows)
        ]
        X = pd.DataFrame(data, columns=[f"Feature {j+1}" for j in range(cols)])
        dataset_info = {"name": "Manual Entry", "description": "User manually entered data"}

    if X is None:
        return

    # Show dataset info + preview
    st.sidebar.markdown(f"**{dataset_info.get('name', 'Dataset')}**")
    st.sidebar.markdown(dataset_info.get("description", ""))
    st.sidebar.markdown(f"**Points:** {len(X)}  •  **Dims:** {X.shape[1]}")
    with st.expander("Data Preview"):
        dfp = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        st.dataframe(dfp.head())

    # ─── Distance Measure Selection ─────────────────────────────────────────────
    st.header("Distance Measure Selection")
    mode = st.radio("Mode:", ["Manual", "AI Smart"], index=0, horizontal=True)

    # 1) Manual
    if mode == "Manual":
        display_categories = [
            "Numeric/Vector Measures", "Binary/Categorical Measures",
            "Distribution/Histogram Measures", "Sequence/Time-Series Measures",
            "Mixed-Type Measures", "Graph & Structure Measures",
            "Universal/Compression-Based"
        ]
        chosen_cat = st.selectbox("Select Metric Category:", display_categories)
        internal_cat = CATEGORY_MAPPING.get(chosen_cat)

        metrics_list = get_supported_metrics(internal_cat)

        if not metrics_list:
            metrics_list = get_supported_metrics(chosen_cat)
        
        # Try fetching metrics with the full key, else strip ' Measures'
        metrics_list = get_supported_metrics(internal_cat)
        if not metrics_list:
            alt_key = chosen_cat.replace(" Measures", "")
            metrics_list = get_supported_metrics(alt_key)

        if not metrics_list:
            st.error(f"No metrics found for category '{chosen_cat}'")
            return

        metric_name = st.selectbox("Select Metric:", metrics_list)
        st.markdown(f"**Using:** `{metric_name}`")

    # 2) AI Smart
    # 2) AI Smart
    else:
        st.markdown("### 🤖 AI Smart Method Selection")

        try:
            # Handle both return types from detect_data_type
            detection_result = detect_data_type(X)
            
            if isinstance(detection_result, tuple) and len(detection_result) == 2:
                detected_label, confidence = detection_result
            else:
                # Handle case where it returns just a string
                detected_label = detection_result
                confidence = 1.0  # Default confidence
                
            st.markdown(f"**Data Type Detected:** {detected_label}  •  {confidence*100:.0f}% confidence")

            DETECT_MAP = {
                "Numeric/Vector": "Numeric/Vector Measures",
                "Numeric/Vector Measures": "Numeric/Vector Measures",
                "Binary/Categorical": "Binary/Categorical Measures",
                "Distribution/Histogram": "Distribution/Histogram Measures",
                "Sequence/Time-Series": "Sequence/Time-Series Measures",
                "Mixed-Type": "Mixed-Type Measures",
                "Graph & Structure": "Graph & Structure Measures",
                "Universal/Compression": "Universal/Compression-Based",
            }

            internal_key = DETECT_MAP.get(detected_label)

            if not internal_key:
                st.error(f"Could not map detected type '{detected_label}' to an internal category")
                return

            candidates = get_supported_metrics(internal_key) or \
             get_supported_metrics(detected_label) or \
             get_supported_metrics(internal_key.replace("/", " / ").replace(" Measures", ""))

            if not candidates:
                # Try to find any metrics that might work
                all_metrics = []
                for cat in get_categories():
                    all_metrics.extend(_orig_get(cat) or [])
                
                if all_metrics:
                    st.warning("No category-specific metrics found. Trying all available metrics...")
                    candidates = all_metrics
                else:
                    st.error("No clustering metrics available at all!")
                    return

            st.markdown(f"_{len(candidates)} methods will be tested_")
            
            k = st.sidebar.slider("Number of Clusters (K)", 2, 20, 3)
            mi = st.sidebar.slider("Max Iterations", 10, 1000, 100)

            if st.button("🚀 Run AI Smart Selection"):
                with st.spinner("Benchmarking methods…"):
                    results = benchmark_category(X, internal_key, k, mi)

                if not results:
                    st.error("No metrics available to benchmark for this data category.")
                    return

                best = max(results, key=lambda r: r["silhouette"])
                metric_name = best["metric"]

                st.success("✅ Analysis Complete!")
                st.markdown(f"🏆 **Best Method Selected:** `{metric_name}`  •  Score: **{best['silhouette']:.3f}**")

                st.markdown("#### 📊 Top 3 Methods")
                for idx, r in enumerate(sorted(results, key=lambda x: x["silhouette"], reverse=True)[:3], 1):
                    st.write(f"{idx}. `{r['metric']}` — {r['silhouette']:.3f}")

                with st.expander("⏱️ Processing Times (ms)"):
                    df_t = pd.DataFrame(results)[["metric", "time_ms"]]
                    df_t["time_ms"] = df_t["time_ms"].round().astype(int).astype(str) + "ms"
                    st.dataframe(df_t.set_index("metric"))
            else:
                return
                
        except Exception as e:
            st.error(f"Data type detection failed: {str(e)}")
            return

    # ─── Clustering Parameters & Execution ────────────────────────────────────
    st.header("Clustering Parameters")
    c1, c2 = st.columns(2)
    with c1:
        n_clusters = st.slider("Number of Clusters (K)", 2, 20, 3)
    with c2:
        max_iter = st.slider("Max Iterations", 10, 1000, 100)

    if st.button("🚀 Run K-Means Clustering"):
        with st.spinner(f"Clustering with {metric_name}…"):
            try:
                # Normalize metric_name for engine
                engine = KMeansEngine(
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    metric_name=metric_name,
                    X=X
)
                results = engine.fit_predict(X)
                display_results(results, metric_name, n_clusters)

                # Visualization
                if X.shape[1] >= 2:
                    v1, v2 = st.columns(2)
                    with v1:
                        st.subheader("Cluster Visualization")
                        st.pyplot(plot_clusters_2d(X, results["labels"], results["centroids"]))
                    with v2:
                        st.subheader("Convergence History")
                        st.pyplot(plot_convergence_history(results["history"]))

                # Detailed Analysis & Export
                st.subheader("Detailed Analysis")
                with st.expander("Cluster Assignments"):
                    for i in range(min(10, len(X))):
                        # Handle both DataFrame and array types
                        if hasattr(X, 'iloc'):
                            point_data = X.iloc[i].tolist()
                        else:
                            point_data = X[i].tolist()
                        st.write(f"Point {i+1}: {point_data} → Cluster {results['labels'][i]}")

                with st.expander("Cluster Centroids"):
                    for idx, ctr in enumerate(results["centroids"], 1):
                        st.write(f"Cluster {idx}: {ctr}")

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
    c3.metric("Iterations", results["n_iter"])

    m1, m2 = st.columns(2)
    m1.metric("Inertia", f"{results['inertia']:.4f}")
    m2.metric("Silhouette Score", f"{results['silhouette']:.4f}")

    st.markdown("**Interpretation:**")
    if results["silhouette"] > 0.5:
        st.success("Strong cluster structure")
    elif results["silhouette"] > 0.25:
        st.info("Reasonable cluster structure")
    else:
        st.warning("Weak cluster structure; consider a different metric or K")


def export_results(X, results):
    df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    df["Cluster"] = results["labels"]
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", data=csv_bytes,
                       file_name="clustering_results.csv", mime="text/csv")


if __name__ == "__main__":
    main()
