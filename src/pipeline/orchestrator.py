# File: src/pipeline/orchestrator.py
# (Optional) demonstrates sequential pipeline control
def run_pipeline():
    import streamlit as st
    pages = ["upload", "method_selection", "clustering", "results"]
    st.sidebar.title("Pipeline")
    choice = st.sidebar.radio("Stage", pages)
    if choice == 'upload':
        import src.application.pages.upload
    elif choice == 'method_selection':
        import src.application.pages.method_selection
    elif choice == 'clustering':
        import src.application.pages.clustering
    else:
        import src.application.pages.results