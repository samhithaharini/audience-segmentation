# OTT Audience Segmentation & Personalized Recommendation System

A production-ready Machine Learning MVP for clustering OTT users based on viewing behavior and generating personalized content recommendations. Built with Streamlit, Scikit-learn, and Plotly.

---

## 🎯 Problem Statement

OTT platforms (Netflix, Amazon Prime, etc.) need to understand their diverse user base to deliver personalized content. This project:
- Groups users into meaningful audience segments using **unsupervised clustering** (KMeans).
- Generates **segment-specific content recommendations** using a rule-based engine.
- Provides an **interactive dashboard** for data exploration, model training, and recommendation delivery.

---

## 📊 Dataset

The dataset contains user-level information:
- `User_ID`, `Name` (identifiers, dropped before training)
- `Age`, `Country`, `Subscription_Type`, `Watch_Time_Hours`, `Favorite_Genre`, `Last_Login`

**Feature Engineering:**
- `Last_Login` → `Days_Since_Last_Login` (recency of activity)
- Optional: `Age_Group`, `Engagement_Level`, `Watch_Time_Category` (based on analysis)

---

## 🧠 Methodology

### 1. Data Preprocessing
- Handle missing values & duplicates
- Encode categorical features (One‑Hot Encoding for nominal, Ordinal for Subscription_Type if needed)
- Scale numerical features with `StandardScaler` (required for KMeans)

### 2. Clustering (Unsupervised)
- **KMeans** selected for:
  - Scalability & interpretability
  - Clear cluster centroids for business labeling
- Optimal **K** determined via:
  - **Elbow Method** (inertia)
  - **Silhouette Score** (cohesion & separation)
- Model saved as `kmeans.pkl` and `scaler.pkl`

### 3. Cluster Profiling
Each cluster is analyzed to generate:
- Business‑friendly name (e.g., *Premium Binge Watchers*)
- Average age, preferred genre, watch time, subscription type, country distribution, recent activity

### 4. Recommendation Engine
Rule‑based mapping from cluster profiles to curated content lists:
- Example: *Action Lovers* → John Wick, Extraction, Avengers, etc.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Upload Dataset] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Clustering KMeans]
    D --> E[Optimal K via Elbow & Silhouette]
    E --> F[Assign Segments]
    F --> G[Cluster Profiling]
    G --> H[Recommendation Engine]
    H --> I[Interactive Dashboard]