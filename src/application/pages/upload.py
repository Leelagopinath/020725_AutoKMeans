# File: src/application/pages/upload.py
import streamlit as st
from src.data.loader import load_dataset, list_sample_datasets

st.title("1️⃣ Data Upload")

# Option: sample datasets
samples = list_sample_datasets()
choice = st.selectbox("Or choose a sample dataset:", ["-- Upload File --"] + samples)

if choice != "-- Upload File --":
    df = load_dataset(choice, sample=True)
    st.success(f"Loaded sample: {choice}")
elif uploaded := st.file_uploader("Upload preprocessed data (CSV/Parquet)"):
    df = load_dataset(uploaded)
    st.success("Dataset loaded")
else:
    st.stop()

st.write("Data preview:")
st.dataframe(df.head())

# Store in session
st.session_state['df'] = df