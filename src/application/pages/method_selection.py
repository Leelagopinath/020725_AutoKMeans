# File: src/application/pages/method_selection.py
import streamlit as st
from src.clustering.distance_selector import get_categories, get_supported_metrics

st.title("2️⃣ Method Selection")

df = st.session_state.get('df')
if df is None:
    st.error("No data found. Please upload first.")
    st.stop()

cats = get_categories()
category = st.selectbox("Select distance category:", cats)
metric = st.selectbox("Select distance metric:", get_supported_metrics(category))

# Save choice
st.session_state['distance_metric'] = metric
st.session_state['features'] = df.select_dtypes('number').columns.tolist()

# Navigation hint
st.write("Click Next to continue to clustering.")