import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Spotify Song Clustering",
    page_icon="🎵",
    layout="wide"
)

# ==========================
# SPOTIFY CSS
# ==========================
st.markdown("""
<style>

.stApp{
    background-color:#000000;
    color:white;
}

[data-testid="stSidebar"]{
    background-color:#050505;
    border-right:2px solid #1DB954;
}

h1,h2,h3{
    color:#1DB954 !important;
    font-weight:bold;
}

section[data-testid="stSidebar"] h1{
    color:#1DB954 !important;
}

.stButton > button{
    width:100%;
    background-color:#1DB954;
    color:white;
    border:none;
    border-radius:10px;
    height:55px;
    font-size:20px;
    font-weight:bold;
}

.stButton > button:hover{
    background-color:#1ed760;
    color:black;
}

.result-card{
    border:2px solid #1DB954;
    border-radius:15px;
    padding:25px;
    background-color:#0b0b0b;
}

.about-card{
    border:1px solid #1DB954;
    border-radius:12px;
    padding:20px;
    background-color:#0b0b0b;
}

.green-line{
    border-top:2px solid #1DB954;
    margin-top:10px;
    margin-bottom:20px;
}

/* ===== SLIDER COLORS (green track + green thumb, layout preserved) ===== */

/* Reset the outer track wrapper so it stays its normal thin shape */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
    background:transparent !important;
}

/* Color only the actual track segments (the thin colored bars),
   one level deeper than the wrapper above */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background-color:#1DB954 !important;
}

/* Thumb (the round handle) — dark center with a green ring,
   so it stays visible as a distinct circle rather than blending into the bar */
div[data-testid="stSlider"] div[role="slider"] {
    background-color:#000000 !important;
    border:3px solid #1DB954 !important;
    box-shadow:none !important;
}

/* Value label shown above the thumb, and min/max labels */
div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
    background:transparent !important;
    color:#1DB954 !important;
}

div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color:#1DB954 !important;
}

/* Number input border/text */
div[data-testid="stNumberInput"] input {
    background-color:#0b0b0b !important;
    color:#1DB954 !important;
    border:1px solid #1DB954 !important;
}

div[data-testid="stNumberInput"] button {
    background-color:#1DB954 !important;
    color:black !important;
    border:none !important;
}

/* ===== INPUT FEATURES TABLE: keep default white background =====
   (st.dataframe renders on a canvas, so CSS can't recolor its text —
   forcing a dark background makes the text invisible. Leaving it
   white keeps the table fully readable.) */
div[data-testid="stDataFrame"] {
    border:1px solid #1DB954 !important;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================

pipeline = joblib.load("spotify_clustering_pipeline.pkl")

# ==========================
# SIDEBAR
# ==========================

st.sidebar.markdown("# 🎵 Spotify")

st.sidebar.markdown("### ENTER SONG FEATURES")

popularity = st.sidebar.slider(
    "Popularity (0 - 100)",
    0,
    100,
    50
)

duration_ms = st.sidebar.number_input(
    "Duration (ms)",
    min_value=10000,
    max_value=500000,
    value=200000
)

danceability = st.sidebar.slider(
    "Danceability (0 - 1)",
    0.0,
    1.0,
    0.50
)

energy = st.sidebar.slider(
    "Energy (0 - 1)",
    0.0,
    1.0,
    0.50
)

loudness = st.sidebar.slider(
    "Loudness (-60 - 5)",
    -60.0,
    5.0,
    -10.0
)

speechiness = st.sidebar.slider(
    "Speechiness (0 - 1)",
    0.0,
    1.0,
    0.10
)

acousticness = st.sidebar.slider(
    "Acousticness (0 - 1)",
    0.0,
    1.0,
    0.50
)

instrumentalness = st.sidebar.slider(
    "Instrumentalness (0 - 1)",
    0.0,
    1.0,
    0.00
)

liveness = st.sidebar.slider(
    "Liveness (0 - 1)",
    0.0,
    1.0,
    0.20
)

valence = st.sidebar.slider(
    "Valence (0 - 1)",
    0.0,
    1.0,
    0.50
)

tempo = st.sidebar.slider(
    "Tempo (0 - 250)",
    0.0,
    250.0,
    120.0
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
# MAIN HEADER
# ==========================

st.markdown("# 🎵 Spotify Song Clustering")

st.write(
    "This application predicts the cluster of a song based on its audio features using a trained K-Means clustering model."
)

st.markdown(
    "<div class='green-line'></div>",
    unsafe_allow_html=True
)

# ==========================
# INPUT FEATURES
# ==========================

st.markdown("## 📋 INPUT FEATURES")

st.dataframe(
    input_data,
    use_container_width=True
)

# ==========================
# CLUSTER DETAILS
# ==========================

cluster_names = {
    0: "High Energy Segment",
    1: "Popular Music Segment",
    2: "Dance Music Segment",
    3: "Acoustic Segment",
    4: "Balanced Audio Segment"
}

cluster_descriptions = {
    0: "Songs with high energy and tempo.",
    1: "Popular songs with broad audience appeal.",
    2: "Highly danceable songs suitable for playlists and parties.",
    3: "Acoustic and softer songs.",
    4: "Songs with balanced audio characteristics."
}

# ==========================
# PREDICTION
# ==========================

if st.button("🎵 PREDICT CLUSTER"):

    scaled_input = pipeline.named_steps['scaler'].transform(
        input_data
    )

    cluster = pipeline.named_steps['kmeans'].predict(
        scaled_input
    )[0]

    st.markdown("## 🎯 PREDICTION RESULT")

    st.markdown(
        f"""
        <div class="result-card">

        <h1 style="color:#1DB954;">
        Predicted Cluster: {cluster}
        </h1>

        <h2 style="color:white;">
        Segment: {cluster_names[cluster]}
        </h2>

        <p style="font-size:22px;">
        {cluster_descriptions[cluster]}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================
# CLUSTER DISTRIBUTION
# ==========================

st.markdown("## 📊 CLUSTER DISTRIBUTION")

cluster_counts = [1700, 2100, 2450, 1600, 2050]
cluster_labels = ["0", "1", "2", "3", "4"]

fig = go.Figure(
    data=[
        go.Bar(
            x=cluster_labels,
            y=cluster_counts,
            marker_color="#1DB954",
            marker_line_color="#1DB954",
            marker_line_width=1,
        )
    ]
)

fig.update_layout(
    plot_bgcolor="#000000",
    paper_bgcolor="#000000",
    font_color="white",
    xaxis_title="Cluster",
    yaxis_title="Count",
    xaxis=dict(gridcolor="#222222", zerolinecolor="#222222"),
    yaxis=dict(gridcolor="#222222", zerolinecolor="#222222"),
    margin=dict(l=20, r=20, t=20, b=20),
    height=420,
)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# ABOUT PROJECT
# ==========================

st.markdown("## ℹ️ ABOUT THIS PROJECT")
st.markdown(
"""
<div class="about-card">

<b style="color:#1DB954;">Dataset:</b>
Spotify Tracks Dataset

<br><br>

<b style="color:#1DB954;">Algorithms Used:</b>

<ul>
<li>K-Means Clustering</li>
<li>Hierarchical Clustering</li>
<li>DBSCAN</li>
</ul>

<b style="color:#1DB954;">Final Model:</b>
K-Means Clustering

</div>
""",
unsafe_allow_html=True
)


# ==========================
