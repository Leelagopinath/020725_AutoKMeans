# File: src/application/navigation.py

def show_navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Clustering", "Results", "Documentation"])
    return page