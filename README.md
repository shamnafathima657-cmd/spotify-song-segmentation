# spotify-song-segmentation
An unsupervised machine learning project that clusters Spotify songs based on audio features using K-Means, Hierarchical Clustering, and DBSCAN. Includes EDA, cluster evaluation, PCA visualization, model pipeline development, and Streamlit deployment.


#  Spotify Song Clustering using Unsupervised Machine Learning

##  Project Overview

This project applies Unsupervised Machine Learning techniques to discover hidden patterns in Spotify songs. Songs are grouped into meaningful clusters based on their audio characteristics such as danceability, energy, loudness, acousticness, tempo, and popularity.

Three clustering algorithms were implemented and compared:

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

The clusters were evaluated using Silhouette Score, Davies-Bouldin Index, and the Elbow Method. K-Means was selected as the final model due to its balanced and interpretable clustering performance.

---

##  Objectives

- Understand and analyze the Spotify dataset.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Select and scale relevant features.
- Apply multiple clustering algorithms.
- Evaluate clustering performance.
- Visualize clusters using PCA.
- Build an automated machine learning pipeline.
- Deploy the final clustering model using Streamlit.

---

##  Dataset Features

The following audio features were used for clustering:

- Popularity
- Duration (ms)
- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- SciPy
- Joblib
- Streamlit

---

##  Project Workflow

1. Data Loading
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis
4. Feature Selection
5. Feature Scaling
6. K-Means Clustering
7. Hierarchical Clustering
8. DBSCAN
9. Cluster Evaluation
10. PCA Visualization
11. Cluster Profiling
12. Pipeline Development
13. Model Export
14. Streamlit Deployment

---

##  Evaluation Results

| Algorithm | Silhouette Score | Davies-Bouldin Index |
|------------|-----------------|----------------------|
| K-Means | 0.1404 | 1.8913 |
| Hierarchical | 0.1097 | 2.0783 |
| DBSCAN | 0.2365 | 2.1752 |

### Final Model Selection

K-Means Clustering

K-Means was selected because it produced balanced and interpretable clusters with the lowest Davies-Bouldin Index.

---

##  Cluster Visualization

Principal Component Analysis (PCA) was used to reduce dimensionality and visualize the discovered clusters.

Cluster Segments:

- High Energy Segment
- Popular Music Segment
- Dance Music Segment
- Acoustic Segment
- Balanced Audio Segment

---

##  Streamlit Application

The Streamlit application allows users to:

- Enter song audio features
- Predict the cluster assignment
- View cluster descriptions
- Interact with the clustering model

#### Run locally:

streamlit run Spotifyapp.py

#### Streamlit cloud community link:

https://spotify-song-segmentation-kmsirm4qj4nf7rrfvnk7ok.streamlit.app/

## Key Insights
- Songs were successfully grouped based on audio characteristics.
- K-Means produced the most balanced clusters.
- Hierarchical Clustering provided useful similarity relationships.
- DBSCAN detected outliers but created highly imbalanced clusters.
- The final model can be used for music segmentation and recommendation systems.

## Conclusion

This project successfully identified natural song groupings within the Spotify dataset using unsupervised learning techniques. After comparing K-Means, Hierarchical Clustering, and DBSCAN, K-Means was selected as the final model due to its superior cluster quality and interpretability. The developed Streamlit application enables interactive cluster prediction and analysis for new songs.
