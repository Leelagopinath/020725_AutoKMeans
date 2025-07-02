import streamlit as st
from src.visualization.cluster_plot import plot_scatter
from src.visualization.silhouette_plot import plot_silhouette
from src.visualization.convergence_plot import plot_convergence

st.title("4️⃣ Results & Visualization")

df = st.session_state['df']
labels = st.session_state['labels']
info = st.session_state['cluster_info']
features = st.session_state['features']

# Attach labels
df['Cluster'] = labels

st.write("### Clustered Data Preview")
st.dataframe(df.head())

st.write("### Scatter Plot")
plot_scatter(df[features], labels)

st.write("### Silhouette Plot")
plot_silhouette(df[features], labels)

st.write("### Convergence History")
plot_convergence(info['history'])

st.download_button("Download clustered data", df.to_csv(index=False), "clustered.csv")