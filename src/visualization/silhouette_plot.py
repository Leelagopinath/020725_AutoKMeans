# File: src/visualization/silhouette_plot.py
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import streamlit as st

def plot_silhouette(X, labels):
    score = silhouette_score(X, labels)
    sample_sil = silhouette_samples(X, labels)
    fig, ax = plt.subplots()
    ax.hist(sample_sil, bins=30)
    ax.axvline(score, color='red', linestyle='--')
    ax.set_title(f"Silhouette Plot (avg={score:.2f})")
    st.pyplot(fig)