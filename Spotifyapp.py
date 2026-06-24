import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Spotify Song Clustering",
    page_icon="🎵",
    layout="wide"
)

# Load Model
pipeline = joblib.load("spotify_clustering_pipeline.pkl")

# Title
st.title("🎵 Spotify Song Clustering Application")

st.markdown("""
This application predicts the cluster of a song based on its audio features using a trained K-Means Clustering model.
""")

# Sidebar Inputs
st.sidebar.header("Enter Song Features")

popularity = st.sidebar.slider("Popularity", 0, 100, 50)

duration_ms = st.sidebar.number_input(
    "Duration (ms)",
    min_value=10000,
    max_value=500000,
    value=200000
)

danceability = st.sidebar.slider(
    "Danceability",
    0.0,
    1.0,
    0.5
)

energy = st.sidebar.slider(
    "Energy",
    0.0,
    1.0,
    0.5
)

loudness = st.sidebar.slider(
    "Loudness",
    -60.0,
    5.0,
    -10.0
)

speechiness = st.sidebar.slider(
    "Speechiness",
    0.0,
    1.0,
    0.1
)

acousticness = st.sidebar.slider(
    "Acousticness",
    0.0,
    1.0,
    0.5
)

instrumentalness = st.sidebar.slider(
    "Instrumentalness",
    0.0,
    1.0,
    0.0
)

liveness = st.sidebar.slider(
    "Liveness",
    0.0,
    1.0,
    0.2
)

valence = st.sidebar.slider(
    "Valence",
    0.0,
    1.0,
    0.5
)

tempo = st.sidebar.slider(
    "Tempo",
    0.0,
    250.0,
    120.0
)

# Input DataFrame
input_data = pd.DataFrame([[
    popularity,
    duration_ms,
    danceability,
    energy,
    loudness,
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    valence,
    tempo
]], columns=[
    'popularity',
    'duration_ms',
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
])

# Display Input Data
st.subheader("📊 Input Features")

st.dataframe(input_data)

# Cluster Names
cluster_names = {
    0: "High Energy Segment",
    1: "Popular Music Segment",
    2: "Dance Music Segment",
    3: "Acoustic Segment",
    4: "Balanced Audio Segment"
}

cluster_descriptions = {
    0: "Songs with high energy, loudness and tempo.",
    1: "Popular songs with strong audience appeal.",
    2: "Highly danceable songs suitable for playlists and parties.",
    3: "Acoustic and soft songs with lower intensity.",
    4: "Songs with balanced audio characteristics."
}

# Prediction
if st.button("Predict Cluster"):

    cluster = pipeline.named_steps['kmeans'].predict(
        pipeline.named_steps['scaler'].transform(input_data)
    )[0]

    st.success(f"Predicted Cluster: {cluster}")

    st.info(
        f"🎧 Segment Name: {cluster_names.get(cluster,'Unknown')}"
    )

    st.write(
        cluster_descriptions.get(cluster,'No description available.')
    )

# Project Details
with st.expander("📖 About This Project"):

    st.write("""
    Dataset: Spotify Tracks Dataset

    Algorithms Used:
    - K-Means Clustering
    - Hierarchical Clustering
    - DBSCAN

    Evaluation Metrics:
    - Silhouette Score
    - Davies-Bouldin Index
    - Elbow Method

    Final Model Selected:
    - K-Means Clustering
    """)

# Footer
st.markdown("---")
