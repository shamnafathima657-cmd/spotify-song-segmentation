import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Spotify Song Clustering",
    page_icon="🎵",
    layout="wide"
)

# ----------------------------------------------------
# Custom CSS — Dark theme with green accents
# ----------------------------------------------------
st.markdown("""
<style>
    /* Overall background */
    .stApp {
        background-color: #0e1117;
        color: #f0f0f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #1f2530;
    }

    /* Title */
    .app-title {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 2.4rem;
        font-weight: 800;
        color: #1ED760;
        margin-bottom: 0px;
    }

    .app-subtitle {
        color: #c9c9c9;
        font-size: 1.05rem;
        margin-top: 4px;
        margin-bottom: 10px;
    }

    hr {
        border: none;
        border-top: 1px solid #1ED760;
        opacity: 0.5;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #1ED760;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* Predict button */
    div.stButton > button {
        background-color: #1ED760;
        color: #0e1117;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.5em;
        font-size: 1rem;
    }
    div.stButton > button:hover {
        background-color: #17b552;
        color: #0e1117;
    }

    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #1f2530;
        border-radius: 8px;
    }

    /* Info / success boxes */
    div[data-testid="stSuccess"] {
        background-color: rgba(30, 215, 96, 0.1);
        border: 1px solid #1ED760;
        border-radius: 8px;
    }
    div[data-testid="stInfo"] {
        background-color: rgba(30, 215, 96, 0.08);
        border: 1px solid #1ED760;
        border-radius: 8px;
    }

    /* About project box */
    .about-box {
        background-color: #131722;
        border: 1px solid #1ED760;
        border-radius: 10px;
        padding: 18px 22px;
        color: #e6e6e6;
        line-height: 1.6;
    }
    .about-box b {
        color: #1ED760;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9a9a9a;
        font-size: 0.9rem;
        margin-top: 30px;
        padding-bottom: 15px;
    }
    .footer span {
        color: #1ED760;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
pipeline = joblib.load("spotify_clustering_pipeline.pkl")

# ----------------------------------------------------
# Title
# ----------------------------------------------------
st.markdown('<div class="app-title">🎵 Spotify Song Clustering</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">This application predicts the cluster of a song '
    'based on its audio features using a trained K-Means clustering model.</div>',
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------
st.sidebar.header("Enter Song Features")

popularity = st.sidebar.slider("Popularity (0 - 100)", 0, 100, 50)

duration_ms = st.sidebar.number_input(
    "Duration (ms)",
    min_value=10000,
    max_value=500000,
    value=200000
)

danceability = st.sidebar.slider("Danceability (0 - 1)", 0.0, 1.0, 0.5)
energy = st.sidebar.slider("Energy (0 - 1)", 0.0, 1.0, 0.5)
loudness = st.sidebar.slider("Loudness (-60 - 5)", -60.0, 5.0, -10.0)
speechiness = st.sidebar.slider("Speechiness (0 - 1)", 0.0, 1.0, 0.1)
acousticness = st.sidebar.slider("Acousticness (0 - 1)", 0.0, 1.0, 0.5)
instrumentalness = st.sidebar.slider("Instrumentalness (0 - 1)", 0.0, 1.0, 0.0)
liveness = st.sidebar.slider("Liveness (0 - 1)", 0.0, 1.0, 0.2)
valence = st.sidebar.slider("Valence (0 - 1)", 0.0, 1.0, 0.5)
tempo = st.sidebar.slider("Tempo (0 - 250)", 0.0, 250.0, 120.0)

# ----------------------------------------------------
# Input DataFrame
# ----------------------------------------------------
input_data = pd.DataFrame([[
    popularity, duration_ms, danceability, energy, loudness,
    speechiness, acousticness, instrumentalness, liveness, valence, tempo
]], columns=[
    'popularity', 'duration_ms', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
])

# ----------------------------------------------------
# Display Input Data
# ----------------------------------------------------
st.markdown('<div class="section-header">📊 Input Features</div>', unsafe_allow_html=True)
st.dataframe(input_data, use_container_width=True)

# ----------------------------------------------------
# Cluster Names & Descriptions
# ----------------------------------------------------
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

# ----------------------------------------------------
# Helper: simulate cluster distribution for the chart
# (replace this with your real dataset's cluster counts if available)
# ----------------------------------------------------
@st.cache_data
def get_cluster_distribution(_pipeline, n_samples=10000, seed=42):
    rng = np.random.default_rng(seed)
    sample = pd.DataFrame({
        'popularity': rng.uniform(0, 100, n_samples),
        'duration_ms': rng.uniform(10000, 500000, n_samples),
        'danceability': rng.uniform(0, 1, n_samples),
        'energy': rng.uniform(0, 1, n_samples),
        'loudness': rng.uniform(-60, 5, n_samples),
        'speechiness': rng.uniform(0, 1, n_samples),
        'acousticness': rng.uniform(0, 1, n_samples),
        'instrumentalness': rng.uniform(0, 1, n_samples),
        'liveness': rng.uniform(0, 1, n_samples),
        'valence': rng.uniform(0, 1, n_samples),
        'tempo': rng.uniform(0, 250, n_samples),
    })
    labels = _pipeline.named_steps['kmeans'].predict(
        _pipeline.named_steps['scaler'].transform(sample)
    )
    counts = pd.Series(labels).value_counts().sort_index()
    return counts

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
predicted_cluster = None

if st.button("🎵 Predict Cluster"):
    predicted_cluster = pipeline.named_steps['kmeans'].predict(
        pipeline.named_steps['scaler'].transform(input_data)
    )[0]

    st.success(f"Predicted Cluster: {predicted_cluster}")
    st.info(f"🎧 Segment Name: {cluster_names.get(predicted_cluster, 'Unknown')}")
    st.write(cluster_descriptions.get(predicted_cluster, 'No description available.'))

# ----------------------------------------------------
# Cluster Distribution Chart
# ----------------------------------------------------
st.markdown('<div class="section-header">📈 Cluster Distribution</div>', unsafe_allow_html=True)

counts = get_cluster_distribution(pipeline)

bar_colors = ["#1ED760"] * len(counts)
if predicted_cluster is not None and predicted_cluster in counts.index:
    bar_colors[list(counts.index).index(predicted_cluster)] = "#0e1117"

fig = go.Figure(
    data=[
        go.Bar(
            x=[str(i) for i in counts.index],
            y=counts.values,
            marker_color="#1ED760",
            marker_line_color="#1ED760",
            marker_line_width=1,
        )
    ]
)
fig.update_layout(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="#f0f0f0",
    xaxis_title="Cluster",
    yaxis_title="Count",
    xaxis=dict(gridcolor="#1f2530"),
    yaxis=dict(gridcolor="#1f2530"),
    margin=dict(l=20, r=20, t=20, b=20),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# Project Details
# ----------------------------------------------------
st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
<b>Dataset:</b> Spotify Tracks Dataset<br><br>
<b>Algorithms Used:</b> K-Means Clustering, Hierarchical Clustering, DBSCAN<br><br>
<b>Evaluation Metrics:</b> Silhouette Score, Davies-Bouldin Index, Elbow Method<br><br>
<b>Final Model:</b> K-Means Clustering
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
