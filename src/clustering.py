# src/clustering.py
"""
Clustering module: determining optimal K (bounded between 2 and 6), training KMeans, and distinct cluster profiling.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, List, Dict, Any
import logging
import streamlit as st

logger = logging.getLogger(__name__)

def find_optimal_k(X: np.ndarray, max_k: int = 6, random_state: int = 42) -> Tuple[int, List[float], List[float], List[int]]:
    """
    Determine the optimal number of clusters using Silhouette score & Elbow method.
    Strictly caps max_k at 6 (or lower based on dataset size N) to ensure practical OTT business segmentation.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    max_k : int, optional
        Maximum number of clusters to test (capped at 6 max).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[int, List[float], List[float], List[int]]
        optimal_k, inertias, silhouette_scores, k_values.
    """
    n_samples = X.shape[0]
    
    # Cap max_k to 6 max for business audience segmentation, or n_samples - 1 if dataset is tiny
    upper_bound = min(6, n_samples - 1)
    effective_max_k = max(2, min(max_k, upper_bound))

    if n_samples < 3:
        raise ValueError(f"Dataset has only {n_samples} samples. At least 3 samples are required for clustering.")

    inertias = []
    silhouette_scores = []
    k_range = list(range(2, effective_max_k + 1))

    progress_bar = st.progress(0, text="Evaluating cluster quality...")
    for i, k in enumerate(k_range):
        progress_bar.progress((i + 1) / len(k_range), text=f"Evaluating K={k}...")

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)

        try:
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
            logger.info(f"K={k}, Inertia={kmeans.inertia_:.2f}, Silhouette={score:.4f}")
        except ValueError as e:
            logger.warning(f"Silhouette calculation failed for K={k}: {e}")
            silhouette_scores.append(-1.0)

    progress_bar.empty()

    # Find k with highest silhouette score
    valid_scores = [s for s in silhouette_scores if s > -1.0]
    if valid_scores:
        best_idx = int(np.argmax(silhouette_scores))
        optimal_k = k_range[best_idx]
    else:
        # Fallback to K=3 if silhouette fails
        optimal_k = min(3, len(k_range) + 1)

    logger.info(f"Selected Optimal K: {optimal_k}")
    return optimal_k, inertias, silhouette_scores, k_range

def train_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans:
    """Train a KMeans model with specified number of clusters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    return kmeans

def assign_clusters(df: pd.DataFrame, kmeans: KMeans, X_scaled: np.ndarray) -> pd.DataFrame:
    """Assign cluster labels to dataset."""
    labels = kmeans.predict(X_scaled)
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    return df_with_clusters

def generate_cluster_profiles(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Analyze each cluster and generate distinct, human-readable cluster profiles.
    Ensures zero duplicate cluster names.
    """
    profiles = {}
    
    # Identify numerical & categorical columns dynamically
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['Cluster'], errors='ignore').tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Filter out ID columns from statistics display
    numeric_cols = [c for c in numeric_cols if not any(k in c.lower() for k in ['id', 'index', 'uuid'])]
    categorical_cols = [c for c in categorical_cols if not any(k in c.lower() for k in ['name', 'id', 'user'])]

    # Population overall statistics for relative comparisons
    pop_stats = {col: df[col].mean() for col in numeric_cols}

    cluster_stats_list = []
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster_id]
        
        stats = {
            'cluster_id': cluster_id,
            'size': len(cluster_df),
            'percentage': (len(cluster_df) / len(df)) * 100,
            'numeric_means': {col: cluster_df[col].mean() for col in numeric_cols},
            'categorical_modes': {col: (cluster_df[col].mode()[0] if not cluster_df[col].empty else 'N/A') for col in categorical_cols},
            'categorical_dist': {col: cluster_df[col].value_counts().head(3).to_dict() for col in categorical_cols}
        }
        cluster_stats_list.append((cluster_id, stats))

    # Generate unique cluster names using relative profiling
    used_names = set()
    for cluster_id, stats in cluster_stats_list:
        base_name = create_relative_cluster_name(stats, pop_stats)
        
        # Ensure name uniqueness across all clusters
        final_name = base_name
        suffix = 2
        while final_name in used_names:
            final_name = f"{base_name} ({suffix})"
            suffix += 1

        used_names.add(final_name)
        stats['name'] = final_name
        profiles[cluster_id] = stats

    return profiles

def create_relative_cluster_name(stats: Dict[str, Any], pop_stats: Dict[str, float]) -> str:
    """
    Generate intuitive name based on relative deviation from overall population averages.
    """
    num_means = stats['numeric_means']
    cat_modes = stats['categorical_modes']

    # Check primary watch time / engagement
    watch_col = next((c for c in num_means if 'watch' in c.lower() or 'time' in c.lower()), None)
    days_col = next((c for c in num_means if 'days' in c.lower() or 'login' in c.lower() or 'recency' in c.lower()), None)
    genre_col = next((c for c in cat_modes if 'genre' in c.lower() or 'category' in c.lower() or 'type' in c.lower()), None)
    sub_col = next((c for c in cat_modes if 'sub' in c.lower() or 'plan' in c.lower() or 'tier' in c.lower()), None)

    mode_genre = cat_modes.get(genre_col, '') if genre_col else ''
    
    # Engagement tier relative to population
    engagement_label = ""
    if watch_col and watch_col in pop_stats:
        ratio = num_means[watch_col] / (pop_stats[watch_col] + 1e-5)
        if ratio >= 1.25:
            engagement_label = "Binge Viewers"
        elif ratio <= 0.75:
            engagement_label = "Casual Streamers"
        else:
            engagement_label = "Regular Audience"
    else:
        engagement_label = "Viewers"

    # Recency status relative to population
    recency_label = ""
    if days_col and days_col in pop_stats:
        ratio = num_means[days_col] / (pop_stats[days_col] + 1e-5)
        if ratio >= 1.3:
            recency_label = "At-Risk "

    # Combine into readable segment name
    if mode_genre:
        name = f"{recency_label}{engagement_label} ({mode_genre})"
    elif sub_col and cat_modes.get(sub_col):
        name = f"{recency_label}{engagement_label} - {cat_modes[sub_col]} Plan"
    else:
        name = f"{recency_label}{engagement_label} - Segment {stats['cluster_id'] + 1}"

    return name