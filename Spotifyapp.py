import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Spotify Song Clustering",
    page_icon="🎵",
    layout="wide"
)

# =========================
# SPOTIFY THEME
# =========================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #121212;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #191414;
}

/* Titles */
h1, h2, h3 {
    color: #1DB954 !important;
}

/* Text */
p, div, label {
    color: white !important;
}

/* Button */
.stButton > button {
    background-color: #1DB954;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #1ed760;
    color: black;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: #191414;
    border-radius: 10px;
    padding: 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #191414;
}

/* Expander */
.streamlit-expanderHeader {
    color: #1DB954 !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
pipeline = joblib.load("spotify_clustering_pipeline.pkl")

# =========================
# TITLE
# =========================
st.title("🎵 Spotify Song Clustering Application")

st.markdown("""
Discover song segments using Machine Learning clustering based on Spotify audio features.
""")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("🎧 Enter Song Features")

popularity = st.sidebar.slider("Popularity", 0, 100, 50)

duration_ms = st.sidebar.number_input(
    "Duration (ms)",
    min_value=10000,
    max_value=600000,
    value=200000
)

danceability = st.sidebar.slider(
    "Danceability",
    0.0, 1.0, 0.5
)

energy = st.sidebar.slider(
    "Energy",
    0.0, 1.0, 0.5
)

loudness = st.sidebar.slider(
    "Loudness",
    -60.0, 5.0, -10.0
)

speechiness = st.sidebar.slider(
    "Speechiness",
    0.0, 1.0, 0.1
)

acousticness = st.sidebar.slider(
    "Acousticness",
    0.0, 1.0, 0.5
)

instrumentalness = st.sidebar.slider(
    "Instrumentalness",
    0.0, 1.0, 0.0
)

liveness = st.sidebar.slider(
    "Liveness",
    0.0, 1.0, 0.2
)

valence = st.sidebar.slider(
    "Valence",
    0.0, 1.0, 0.5
)

tempo = st.sidebar.slider(
    "Tempo",
    0.0, 250.0, 120.0
)

# =========================
# INPUT DATAFRAME
# =========================
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

# =========================
# DISPLAY INPUTS
# =========================
st.subheader("📊 Input Features")

st.dataframe(input_data)

# =========================
# CLUSTER DETAILS
# =========================
cluster_names = {
    0: "High Energy Segment",
    1: "Popular Music Segment",
    2: "Dance Music Segment",
    3: "Acoustic Segment",
    4: "Balanced Audio Segment"
}

cluster_descriptions = {
    0: "Songs with high energy and tempo.",
    1: "Popular songs with strong audience appeal.",
    2: "Highly danceable songs suitable for parties.",
    3: "Acoustic and softer songs.",
    4: "Balanced songs with mixed audio characteristics."
}

# =========================
# PREDICTION
# =========================
if st.button("🎯 Predict Cluster"):

    scaled_data = pipeline.named_steps['scaler'].transform(input_data)

    cluster = pipeline.named_steps['kmeans'].predict(
        scaled_data
    )[0]

    st.success(
        f"Predicted Cluster: {cluster}"
    )

    st.info(
        f"🎵 Segment: {cluster_names[cluster]}"
    )

    st.write(
        cluster_descriptions[cluster]
    )

# =========================
# ABOUT PROJECT
# =========================
with st.expander("📖 About Project"):

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

    Final Model:
    - K-Means Clustering
    """)

# =========================
# FOOTER
# =========================
st.markdown("---")
