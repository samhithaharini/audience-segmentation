# src/visualization.py
"""
Plotly visualization functions for the dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_elbow_curve(k_range, inertias):
    """
    Elbow curve: Inertia vs. number of clusters.
    """
    fig = px.line(
        x=k_range, y=inertias,
        markers=True,
        labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'},
        title='Elbow Method for Optimal K'
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#1E293B',
        paper_bgcolor='#1E293B',
        font_color='#F8FAFC'
    )
    return fig

def plot_cluster_sizes(profiles):
    """
    Bar chart of cluster sizes (number of users per segment).
    """
    df = pd.DataFrame([
        {'Cluster': f"Group {i+1}: {info['name']}", 'Size': info['size']}
        for i, info in profiles.items()
    ])
    fig = px.bar(
        df, x='Cluster', y='Size',
        color='Size',
        color_continuous_scale='Blues',
        title='Audience Segment Sizes'
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#1E293B',
        paper_bgcolor='#1E293B',
        font_color='#F8FAFC',
        xaxis_tickangle=-30
    )
    return fig