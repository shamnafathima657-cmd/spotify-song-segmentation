import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Spotify Song Personality Finder",
    page_icon="🎵",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

pipeline = joblib.load("spotify_clustering_pipeline.pkl")

# ==========================
# SPOTIFY CSS
# ==========================
st.markdown("""
<style>

/* Main App */
.stApp{
    background-color:#000000;
    color:white;
}

/* Header */
header{
    background-color:#000000 !important;
}

[data-testid="stHeader"]{
    background-color:#000000 !important;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#121212;
    border-right:1px solid #1DB954;
}

/* Sidebar Text */
[data-testid="stSidebar"] *{
    color:white !important;
}

/* Headings */
h1,h2,h3{
    color:#1DB954 !important;
}

/* Button */
.stButton > button{
    background-color:#1DB954;
    color:black;
    font-weight:bold;
    border:none;
    border-radius:12px;
    height:55px;
    width:100%;
}

.stButton > button:hover{
    background-color:#1ED760;
}

/* Slider Track */
.stSlider [data-baseweb="slider"] > div > div{
    background:#1DB954 !important;
}

/* Slider Handle */
.stSlider [role="slider"]{
    background-color:#1DB954 !important;
}

/* Number Input */
input{
    background-color:#121212 !important;
    color:white !important;
}

/* Card */
.card{
    background:#121212;
    padding:20px;
    border-radius:15px;
    border:1px solid #1DB954;
}

/* Slider value labels (0, 100, 0.00, 1.00, etc.) */
.stSlider span{
    color: white !important;
}

/* Current slider value (50, 0.50, etc.) */
.stSlider div[data-testid="stThumbValue"]{
    color: white !important;
    font-weight: bold;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label{
    color: white !important;
}

/* Number input text (Duration) */
section[data-testid="stSidebar"] input{
    color: white !important;
    background-color:#1a1a1a !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================
# SIDEBAR
# ==========================

st.sidebar.markdown("# 🎧 Song Features")

popularity = st.sidebar.slider(
    "Popularity",
    0,100,50
)

duration_ms = st.sidebar.number_input(
    "Duration (ms)",
    min_value=10000,
    max_value=500000,
    value=200000
)

danceability = st.sidebar.slider(
    "Danceability",
    0.0,1.0,0.50
)

energy = st.sidebar.slider(
    "Energy",
    0.0,1.0,0.50
)

loudness = st.sidebar.slider(
    "Loudness",
    -60.0,5.0,-10.0
)

speechiness = st.sidebar.slider(
    "Speechiness",
    0.0,1.0,0.10
)

acousticness = st.sidebar.slider(
    "Acousticness",
    0.0,1.0,0.50
)

instrumentalness = st.sidebar.slider(
    "Instrumentalness",
    0.0,1.0,0.00
)

liveness = st.sidebar.slider(
    "Liveness",
    0.0,1.0,0.20
)

valence = st.sidebar.slider(
    "Valence",
    0.0,1.0,0.50
)

tempo = st.sidebar.slider(
    "Tempo",
    0.0,250.0,120.0
)

# ==========================
# INPUT DATA
# ==========================

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

# ==========================
# HEADER
# ==========================

st.title("🎵 Spotify Song Personality Finder")

st.write("""
Discover the personality of a song using Spotify audio features.
Adjust the song attributes and see which music style best matches your song.
""")

st.markdown("---")
# ==========================
# FEATURE CARDS
# ==========================
st.markdown("## 🎧 Song Features Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div style="
        background:#121212;
        padding:20px;
        border-radius:15px;
        border:1px solid #1DB954;
        text-align:center;">
        <h4 style="color:#1DB954;">Popularity</h4>
        <h2 style="color:white;">{popularity}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="
        background:#121212;
        padding:20px;
        border-radius:15px;
        border:1px solid #1DB954;
        text-align:center;">
        <h4 style="color:#1DB954;">Danceability</h4>
        <h2 style="color:white;">{danceability:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="
        background:#121212;
        padding:20px;
        border-radius:15px;
        border:1px solid #1DB954;
        text-align:center;">
        <h4 style="color:#1DB954;">Energy</h4>
        <h2 style="color:white;">{energy:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style="
        background:#121212;
        padding:20px;
        border-radius:15px;
        border:1px solid #1DB954;
        text-align:center;">
        <h4 style="color:#1DB954;">Tempo</h4>
        <h2 style="color:white;">{tempo:.0f}</h2>
    </div>
    """, unsafe_allow_html=True)
    
# ==========================
# CLUSTER NAMES
# ==========================

cluster_names = {
    0: "🔥 Power Beats",
    1: "🌟 Chart Toppers",
    2: "💃 Dance Vibes",
    3: "🌿 Chill Acoustic",
    4: "🎶 Everyday Mix"
}

cluster_descriptions = {
    0: "High-energy songs perfect for workouts and intense playlists.",
    1: "Popular songs loved by a wide audience.",
    2: "Danceable tracks ideal for parties and celebrations.",
    3: "Relaxing acoustic songs with a calm atmosphere.",
    4: "Balanced songs suitable for everyday listening."
}

# ==========================
# PREDICTION BUTTON
# ==========================

if st.button("🎵 Analyze My Song"):

    cluster = pipeline.predict(input_data)[0]

    st.success("Song analyzed successfully!")

    st.markdown("## 🎯 Your Song Personality")

    cluster_name = cluster_names.get(cluster, "Unknown Cluster")
    cluster_desc = cluster_descriptions.get(cluster, "No description available.")

    st.markdown(
        f"""
        <div style="
            background-color:#121212;
            border:2px solid #1DB954;
            border-radius:20px;
            padding:25px;
            margin-top:15px;
        ">

        <h2 style="
            color:#1DB954;
            margin-bottom:15px;
        ">
            {cluster_name}
        </h2>

        <p style="
            color:white;
            font-size:20px;
            line-height:1.8;
        ">
            {cluster_desc}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown("---")
st.subheader("📊 Music Segment Distribution")

cluster_counts = [22389, 7651, 7339, 30180, 38458]

cluster_labels = [
    "Power Beats",
    "Chart Toppers",
    "Dance Vibes",
    "Chill Acoustic",
    "Everyday Mix"
]

fig, ax = plt.subplots(figsize=(8,4))

ax.bar(
    cluster_labels,
    cluster_counts,
    color="#1DB954"
)
ax.set_facecolor("#121212")
fig.patch.set_facecolor("#000000")

ax.set_title(
    "Spotify Song Segments",
    color="white"
)

ax.set_xlabel(
    "Cluster",
    color="white"
)

ax.set_ylabel(
    "Number of Songs",
    color="white"
)

ax.tick_params(colors="white")

for spine in ax.spines.values():
    spine.set_color("white")

st.pyplot(fig)
st.markdown("---")

st.markdown("""
<div style="
background:#121212;
padding:25px;
border-radius:15px;
border:1px solid #1DB954;
">

<h3 style="color:#1DB954;">
ℹ️ About This Project
</h3>

<p style="color:white;">
This application groups Spotify songs into music segments using Machine Learning clustering techniques.
</p>

<p style="color:white;">
Audio features such as danceability, energy, loudness, tempo, acousticness and popularity are used to identify similar song patterns.
</p>

<p style="color:white;">
The project compares K-Means, Hierarchical Clustering and DBSCAN, with K-Means selected as the final model for deployment.
</p>

</div>
""", unsafe_allow_html=True)