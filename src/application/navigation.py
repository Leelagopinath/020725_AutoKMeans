# File: src/application/navigation.py
import streamlit as st

def show_navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Clustering", "Results", "Documentation"])
    return page