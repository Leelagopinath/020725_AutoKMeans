import streamlit as st
import pandas as pd

from src.utils.data_type_detector import detect_data_type
from src.clustering.benchmark import benchmark_category

def show():
    """
    Streamlit page: AI‑Smart Method Selection.
    Expects in st.session_state:
      - X: DataFrame or 2D array of features
      - n_clusters: int
      - max_iter: int
    Sets:
      - session_state['metric_name'] to the chosen best metric
    """
    st.header("🤖 AI Smart Method Selection")

    X = st.session_state.get("X", None)
    n_clusters = st.session_state.get("n_clusters", 3)
    max_iter = st.session_state.get("max_iter", 300)

    if X is None:
        st.warning("No data loaded yet!")
        return

    # 1) Detect data type
    category, confidence = detect_data_type(X)
    st.markdown(f"**Data Type Detected:** {category}  •  {confidence*100:.0f}% confidence")

    # 2) Run benchmarking when user clicks
    if st.button("🚀 Run AI Smart Selection"):
        with st.spinner("Benchmarking all methods in category…"):
            results = benchmark_category(X, category, n_clusters, max_iter)

        # 3) Choose best
        best = max(results, key=lambda r: r["silhouette"])

        # 4) Display summary UI
        col1, col2 = st.columns([2, 3])
        with col1:
            st.success("✅ Analysis Complete!")
            st.markdown(
                f"🏆 **Best Method:** `{best['metric']}`  •  "
                f"Score: **{best['silhouette']:.3f}**"
            )
        with col2:
            st.markdown("#### 📊 Top 3 Methods")
            top3 = sorted(results, key=lambda r: r["silhouette"], reverse=True)[:3]
            for i, r in enumerate(top3, start=1):
                st.write(f"{i}. `{r['metric']}` — {r['silhouette']:.3f}")

        # 5) Show timing details
        with st.expander("⏱️ Processing Times (ms)"):
            df_times = pd.DataFrame(results)[["metric", "time_ms"]]
            df_times["time_ms"] = df_times["time_ms"].round().astype(int).astype(str) + "ms"
            st.dataframe(df_times.set_index("metric"))

        # 6) Store best metric for downstream
        st.session_state["metric_name"] = best["metric"]
    else:
        st.info("Click ▶️ to automatically select the best distance metric.")
