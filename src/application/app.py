# File: src/application/app.py

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
from src.clustering.distance_selector import get_supported_metrics as _orig_get
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
    st.download_button(
        "📥 Download Results as CSV",
        data=csv_bytes,
        file_name="clustering_results.csv",
        mime="text/csv",
        key="export_csv"
    )


def main():
    st.title("Production Grade K-Means Clustering")
    

    # ─── Data Input ────────────────────────────────────────────────────────
    st.sidebar.header("Data Input")
    data_option = st.sidebar.selectbox(
        "Select Data Input Method:",
        ["Manual Upload", "Sample Datasets", "Manual Entry"],
        key="data_option"
    )

    X = None
    dataset_info = {}

    if data_option == "Sample Datasets":
        dataset_option = st.sidebar.selectbox(
            "Choose Sample Dataset:",
            ["Iris Dataset", "Random Blobs", "Two Moons"],
            key="dataset_option"
        )
        X, dataset_info = get_sample_data(dataset_option)

    elif data_option == "Manual Upload":
        uploaded = st.sidebar.file_uploader(
            "Upload data file (csv, txt, xls, xlsx, json)",
            type=["csv", "txt", "xls", "xlsx", "json"],
            key="file_uploader"
        )
        if uploaded:
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
        else:
            st.warning("Please upload a data file.")
            st.stop()

    else:  # Manual Entry
        rows = st.sidebar.number_input("Number of rows", 1, 100, 5, key="me_rows")
        cols = st.sidebar.number_input("Number of features", 1, 10, 2, key="me_cols")
        st.sidebar.write("Enter your data:")
        data = [
            [
                st.sidebar.number_input(f"R{i+1}C{j+1}", key=f"cell_{i}_{j}")
                for j in range(cols)
            ]
            for i in range(rows)
        ]
        X = pd.DataFrame(data, columns=[f"Feature {j+1}" for j in range(cols)])
        dataset_info = {"name": "Manual Entry", "description": "User manually entered data"}

    # Preview
    st.sidebar.markdown(f"**{dataset_info.get('name', 'Dataset')}**")
    st.sidebar.markdown(dataset_info.get("description", ""))
    st.sidebar.markdown(f"**Points:** {len(X)}  •  **Dims:** {X.shape[1]}")
    with st.expander("Data Preview", expanded=True):
        dfp = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        st.dataframe(dfp.head(), key="data_preview")

    # ─── Distance Measure Selection ─────────────────────────────────────────
    st.header("Distance Measure Selection")
mode = st.radio(
    "Mode:",
    ["Manual", "AI Smart"],
    index=0,
    horizontal=True,
    key="selection_mode"
)

# Reset the other mode when switching
if 'last_mode' not in st.session_state:
    st.session_state.last_mode = mode

if st.session_state.last_mode != mode:
    # Mode changed - reset the other mode's state
    if mode == "Manual":
        st.session_state.ai_metric = None
    else:
        st.session_state.manual_metric = None
    st.session_state.last_mode = mode

# Initialize session state variables
if "manual_metric" not in st.session_state:
    st.session_state.manual_metric = None
if "ai_metric" not in st.session_state:
    st.session_state.ai_metric = None

# 1) Manual Mode - Simplified workflow
if mode == "Manual":
    st.markdown("### 🔧 Manual Metric Selection")
    categories = [
        "Numeric/Vector Measures", "Binary/Categorical Measures",
        "Distribution/Histogram Measures", "Sequence/Time-Series Measures",
        "Mixed-Type Measures", "Graph & Structure Measures",
        "Universal/Compression-Based"
    ]
    chosen_cat = st.selectbox(
        "Select Metric Category:",
        categories,
        key="manual_category"
    )
    
    metrics_list = get_supported_metrics(CATEGORY_MAPPING[chosen_cat])
    st.session_state.manual_metric = st.selectbox(
        "Select Metric:", 
        metrics_list, 
        key="manual_metric"
    )
    st.markdown(f"**Using:** {st.session_state.manual_metric}")

# 2) AI Smart Mode - Complete workflow
else:
    st.markdown("### 🤖 AI Smart Method Selection")
    detection = detect_data_type(X)
    if isinstance(detection, tuple):
        detected_label, confidence = detection
    else:
        detected_label, confidence = detection, 1.0

    st.markdown(f"**Data Type Detected:** {detected_label}  •  {confidence*100:.0f}% confidence")

    # Map detector labels to distance_selector keys
    DETECT_MAP = {
        "Numeric/Vector":             "Numeric / Vector",
        "Numeric/Vector Measures":    "Numeric / Vector",
        "Binary/Categorical":         "Binary / Categorical",
        "Binary/Categorical Measures":"Binary / Categorical",
        "Distribution/Histogram":     "Distribution / Histogram",
        "Distribution/Histogram Measures":"Distribution / Histogram",
        "Sequence/Time-Series":       "Sequence / Time‑Series",
        "Sequence/Time‑Series Measures":"Sequence / Time‑Series",
        "Mixed-Type":                 "Mixed‑Type",
        "Mixed‑Type Measures":        "Mixed‑Type",
        "Graph & Structure":          "Graph & Structure",
        "Graph & Structure Measures": "Graph & Structure",
        "Universal/Compression":      "Universal / Compression",
        "Universal/Compression-Based":"Universal / Compression",
    }
    
    internal_key = DETECT_MAP.get(detected_label)
    if not internal_key:
        st.error(f"Could not map '{detected_label}' to a known category")
        st.stop()

    candidates = get_supported_metrics(internal_key)
    st.markdown(f"_{len(candidates)} methods will be tested_")
    if not candidates:
        st.error("No methods defined for this data type.")
        st.stop()

    # AI-specific parameters in sidebar
    k  = st.sidebar.slider("Number of Clusters (K)", 2, 20, 3, key="ai_k")
    mi = st.sidebar.slider("Max Iterations", 10, 1000, 100, key="ai_mi")

    if st.button("🚀 Run AI Smart Selection", key="ai_run"):
        with st.spinner("Benchmarking methods…"):
            results = benchmark_category(X, internal_key, k, mi)

        if not results:
            st.error("No metrics available to benchmark for this data category.")
            st.stop()

        # Show top 3 methods
        top3 = sorted(results, key=lambda r: r["silhouette"], reverse=True)[:3]
        st.success("✅ Analysis Complete!")
        st.markdown(f"🏆 **Best Method:** `{top3[0]['metric']}` — {top3[0]['silhouette']:.3f}")
        st.markdown("#### 📊 Top 3 Methods")
        for idx, r in enumerate(top3, start=1):
            st.write(f"{idx}. `{r['metric']}` — {r['silhouette']:.3f}")

        # Let user pick one of those three
        choice = st.selectbox(
            "Select method for clustering:",
            [r["metric"] for r in top3],
            key="ai_choice"
        )
        st.session_state.ai_metric = choice

    # Show selected AI metric if available
    if st.session_state.ai_metric:
        st.markdown(f"**Selected AI Metric:** `{st.session_state.ai_metric}`")
    else:
        st.info("Click the button to automatically select the best metric.")

# Get current metric based on active mode
current_metric = st.session_state.manual_metric if mode == "Manual" else st.session_state.ai_metric

# ─── Clustering Parameters & Execution ───────────────────────────────────
st.header("Clustering Parameters")
c1, c2 = st.columns(2)
with c1:
    n_clusters = st.slider(
        "Number of Clusters (K)",
        2, 20, 3,
        key="cluster_k"
    )
with c2:
    max_iter = st.slider(
        "Max Iterations",
        10, 1000, 100,
        key="cluster_iter"
    )

# Only show Run button in AI mode when metric is selected
# Manual mode runs automatically without button
if mode == "AI Smart" and current_metric:
    run_cluster = st.button("🚀 Run K-Means Clustering", key="cluster_run")
elif mode == "Manual" and current_metric:
    run_cluster = True
else:
    run_cluster = False

# Execute clustering when:
# - Manual mode has metric selected (automatic run)
# - AI mode has metric + button pressed
if run_cluster and current_metric:
    with st.spinner(f"Clustering with {current_metric}…"):
        try:
            engine = KMeansEngine(
                n_clusters=n_clusters,
                max_iter=max_iter,
                metric_name=current_metric,
                X=X
            )
            results = engine.fit_predict(X)
            display_results(results, current_metric, n_clusters)

            # Visualization
            if X.shape[1] >= 2:
                v1, v2 = st.columns(2)
                with v1:
                    st.subheader("Cluster Visualization")
                    st.pyplot(plot_clusters_2d(X, results["labels"], results["centroids"]))
                with v2:
                    st.subheader("Convergence History")
                    st.pyplot(plot_convergence_history(results["history"]))

            # Detailed Analysis
            st.subheader("Detailed Analysis")
            with st.expander("Cluster Assignments"):
                for i in range(min(10, len(X))):
                    row = X.iloc[i].tolist() if hasattr(X, "iloc") else X[i].tolist()
                    st.write(f"Point {i+1}: {row} → Cluster {results['labels'][i]}")
            with st.expander("Cluster Centroids"):
                for idx, ctr in enumerate(results["centroids"], 1):
                    st.write(f"Cluster {idx}: {ctr}")

            export_results(X, results)

        except Exception as e:
            st.error(f"Clustering failed: {e}")
            logger.exception(e)

# Show warnings if no metric selected
if not current_metric:
    if mode == "Manual":
        st.warning("Please select a distance metric for Manual mode clustering")
    else:
        st.warning("Please run AI Smart Selection and choose a metric")


if __name__ == "__main__":
    main()
