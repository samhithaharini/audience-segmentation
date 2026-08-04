# app.py
"""
Audience Segmentation Dashboard
A machine learning application for OTT user clustering and personalized recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import get_dataset_info
from src.preprocessing import preprocess_pipeline
from src.clustering import find_optimal_k, train_kmeans, assign_clusters, generate_cluster_profiles
from src.recommendation import get_recommendations
from src.visualization import plot_elbow_curve, plot_cluster_sizes
from src.utils import save_pickle

logging.basicConfig(level=logging.INFO)

MODEL_DIR = 'models'
OUTPUT_DIR = 'outputs'
DATA_DIR = 'data'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ========== REQUIRED COLUMNS ==========
REQUIRED_COLUMNS = {
    'User_ID', 'Name', 'Age', 'Country', 'Subscription_Type',
    'Watch_Time_Hours', 'Favorite_Genre', 'Last_Login'
}

# ========== SESSION STATE ==========
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'kmeans' not in st.session_state:
    st.session_state.kmeans = None
if 'cluster_profiles' not in st.session_state:
    st.session_state.cluster_profiles = None
if 'optimal_k' not in st.session_state:
    st.session_state.optimal_k = None
if 'segmented_df' not in st.session_state:
    st.session_state.segmented_df = None
if 'inertias' not in st.session_state:
    st.session_state.inertias = None
if 'k_range' not in st.session_state:
    st.session_state.k_range = None

st.set_page_config(
    page_title="Audience Segmentation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== THEME (Slate Indigo) ==========
def apply_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0F172A !important;
            color: #F8FAFC !important;
            font-family: 'Outfit', sans-serif !important;
        }

        [data-testid="stSidebar"] {
            background-color: #1E293B !important;
            border-right: 1px solid #334155 !important;
        }

        .info-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .stButton > button {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.3rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(59, 130, 246, 0.5) !important;
        }

        [data-testid="stMetricValue"] {
            font-weight: 700 !important;
            color: #38BDF8 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-weight: 500 !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #0F172A !important;
            border-color: #334155 !important;
            color: #F8FAFC !important;
            border-radius: 8px !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background: #1E293B;
            color: #94A3B8;
            border-radius: 8px;
            padding: 8px 18px;
            border: 1px solid #334155;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #3B82F6 !important;
            color: #FFFFFF !important;
            border-color: #3B82F6 !important;
        }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ========== SIDEBAR ==========
def sidebar_navigation():
    st.sidebar.markdown("<h2 style='color:#38BDF8;'>🎬 CinemaPulse</h2>", unsafe_allow_html=True)
    st.sidebar.caption("Audience Segmentation")
    st.sidebar.markdown("---")
    pages = [
        "🏠 Home",
        "📂 Upload Dataset",
        "🧹 Preprocessing",
        "🤖 Audience Segmentation",
        "👥 Segment Profiles",
        "🎬 Recommendations",
        "⬇ Export Results"
    ]
    return st.sidebar.radio("Navigation", pages)

# ========== PAGES ==========
def home_page():
    st.title("🎬 Audience Segmentation Dashboard")
    st.markdown("##### Discover viewer patterns and deliver relevant content.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>📂 Upload</h4>
            <p>Load user activity CSV.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>🤖 Segment</h4>
            <p>Cluster viewers automatically.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='info-card'>
            <h4>🎬 Recommend</h4>
            <p>Get tailored content suggestions.</p>
        </div>
        """, unsafe_allow_html=True)

def upload_page():
    st.title("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Validate columns
            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                st.error(f"❌ Dataset is missing required column(s): {', '.join(missing)}. Please upload a file with all required columns.")
                st.session_state.df_original = None  # reject dataset
            else:
                st.session_state.df_original = df
                st.success(f"✅ Dataset loaded successfully ({df.shape[0]:,} rows × {df.shape[1]} columns)")
                st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        sample_path = os.path.join(DATA_DIR, 'netflix_users.csv')
        if os.path.exists(sample_path):
            st.info("No custom file uploaded. You can load the sample dataset to test.")
            if st.button("Load Sample Dataset"):
                try:
                    df = pd.read_csv(sample_path)
                    # Validate sample too
                    missing = REQUIRED_COLUMNS - set(df.columns)
                    if missing:
                        st.error(f"Sample dataset is missing columns: {', '.join(missing)}")
                    else:
                        st.session_state.df_original = df
                        st.success(f"Sample dataset loaded ({df.shape[0]:,} rows × {df.shape[1]} columns)")
                        st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading sample: {e}")

    if st.session_state.df_original is not None:
        df = st.session_state.df_original
        info = get_dataset_info(df)
        st.subheader("Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{info['shape'][0]:,}")
        c2.metric("Columns", info['shape'][1])
        c3.metric("Duplicates", info['duplicates'])

def preprocessing_page():
    st.title("🧹 Preprocessing")
    df = st.session_state.df_original
    if df is None:
        st.warning("Please upload a valid dataset first (Upload Dataset tab).")
        return

    if st.button("Run Preprocessing"):
        with st.spinner("Processing..."):
            try:
                df_clean, X_scaled, scaler, preprocessor = preprocess_pipeline(df)
                st.session_state.df_processed = df_clean
                st.session_state.X_scaled = X_scaled
                st.session_state.scaler = scaler
                st.session_state.preprocessor = preprocessor
                st.success("✅ Preprocessing completed.")
                c1, c2 = st.columns(2)
                c1.metric("Clean Records", df_clean.shape[0])
                c2.metric("Features", X_scaled.shape[1])
                st.dataframe(df_clean.head(10), use_container_width=True)
                save_pickle(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
                save_pickle(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))
            except Exception as e:
                st.error(f"Preprocessing error: {e}")

def segmentation_page():
    st.title("🤖 Audience Segmentation")
    X_scaled = st.session_state.X_scaled
    if X_scaled is None:
        st.warning("Please run preprocessing first.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        strategy = st.radio("Mode", ["Automatic", "Manual"])
        manual_k = 4
        if strategy == "Manual":
            manual_k = st.slider("Number of Clusters (K)", 2, 6, 4)

    with c2:
        if st.button("Train Segmentation Model"):
            with st.spinner("Clustering..."):
                try:
                    if strategy == "Automatic":
                        optimal_k, inertias, sil_scores, k_range = find_optimal_k(X_scaled, max_k=6)
                        target_k = optimal_k
                        st.session_state.inertias = inertias
                        st.session_state.k_range = k_range
                    else:
                        target_k = manual_k
                        st.session_state.inertias = None
                        st.session_state.k_range = None

                    st.session_state.optimal_k = target_k
                    kmeans = train_kmeans(X_scaled, n_clusters=target_k)
                    st.session_state.kmeans = kmeans
                    save_pickle(kmeans, os.path.join(MODEL_DIR, 'kmeans.pkl'))

                    df_clean = st.session_state.df_processed
                    df_segmented = assign_clusters(df_clean, kmeans, X_scaled)
                    st.session_state.segmented_df = df_segmented
                    df_segmented.to_csv(os.path.join(OUTPUT_DIR, 'segmented_users.csv'), index=False)

                    profiles = generate_cluster_profiles(df_segmented)
                    st.session_state.cluster_profiles = profiles

                    st.success(f"✅ Formed **{target_k}** distinct audience segments!")
                except Exception as e:
                    st.error(f"Segmentation error: {e}")

    # --- Display ONE dynamic graph ---
    if st.session_state.segmented_df is not None and st.session_state.cluster_profiles is not None:
        st.markdown("---")
        st.subheader("📊 Segmentation Insight")

        if strategy == "Automatic" and st.session_state.inertias is not None and st.session_state.k_range is not None:
            # Show Elbow curve
            fig = plot_elbow_curve(st.session_state.k_range, st.session_state.inertias)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show Cluster sizes (always useful for manual mode)
            fig = plot_cluster_sizes(st.session_state.cluster_profiles)
            st.plotly_chart(fig, use_container_width=True)

def profiles_page():
    st.title("👥 Segment Profiles")
    profiles = st.session_state.cluster_profiles
    if profiles is None:
        st.warning("Please train segmentation first.")
        return

    for cluster_id, stats in profiles.items():
        with st.expander(f"Group {cluster_id + 1}: {stats['name']} ({stats['size']:,} users)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Averages")
                for key, val in stats['numeric_means'].items():
                    st.metric(key.replace('_', ' '), f"{val:.2f}")
            with col2:
                st.markdown("#### Dominant Traits")
                for key, val in stats['categorical_modes'].items():
                    st.write(f"• **{key.replace('_', ' ')}:** `{val}`")

def recommendations_page():
    st.title("🎬 Recommendations")
    profiles = st.session_state.cluster_profiles
    if profiles is None:
        st.warning("Please train segmentation first.")
        return

    cluster_ids = sorted(profiles.keys())
    selected = st.selectbox("Select Segment", cluster_ids, format_func=lambda x: f"Group {x+1}: {profiles[x]['name']}")
    if selected is not None:
        stats = profiles[selected]
        recs = get_recommendations(stats['name'], stats)
        st.subheader(f"Recommended for {stats['name']}")
        for title in recs:
            st.markdown(f"🎬 {title}")

def download_page():
    st.title("⬇ Export Results")
    df = st.session_state.segmented_df
    if df is None:
        st.warning("No segmented data available.")
        return
    csv = df.to_csv(index=False)
    st.download_button("Download Segmented CSV", data=csv, file_name='segmented_users.csv', mime='text/csv')

def main():
    page = sidebar_navigation()
    if page == "🏠 Home":
        home_page()
    elif page == "📂 Upload Dataset":
        upload_page()
    elif page == "🧹 Preprocessing":
        preprocessing_page()
    elif page == "🤖 Audience Segmentation":
        segmentation_page()
    elif page == "👥 Segment Profiles":
        profiles_page()
    elif page == "🎬 Recommendations":
        recommendations_page()
    elif page == "⬇ Export Results":
        download_page()

if __name__ == "__main__":
    main()