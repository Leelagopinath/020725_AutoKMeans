# File: src/application/pages/clustering.py
import streamlit as st
from src.clustering.engine import cluster_data
from sklearn.decomposition import PCA

st.title("3️⃣ K-Means Clustering")

df = st.session_state['df']
metric = st.session_state['distance_metric']
features = st.multiselect("Select features for clustering", st.session_state['features'], default=st.session_state['features'])

k = st.slider("Number of clusters (k)", 2, 20, 3)
max_iter = st.slider("Max iterations", 10, 500, 300)

if st.button("Run Clustering"):
    labels, info = cluster_data(df[features], metric, k, max_iter)
    st.session_state['labels'] = labels
    st.session_state['cluster_info'] = info
    st.success("Clustering complete")
    st.write(info)