# 🎬 CinemaPulse – Audience Segmentation for Personalized OTT Content Recommendation

##  Project Overview

CinemaPulse is a Machine Learning-based web application that segments OTT users into different audience groups based on their viewing behavior, watch time, subscription type, preferred genre, and user activity. Using **K-Means Clustering**, the application identifies similar user groups and provides personalized content recommendations through an interactive Streamlit dashboard.

---

##  Problem Statement

OTT platforms serve users with diverse viewing preferences. Delivering the same recommendations to every user reduces engagement and retention.

The objective of this project is to automatically segment users based on their behavior and preferences, enabling personalized content recommendations and better user engagement.

---

##  Solution Overview

The application performs the following tasks:

- Load the OTT user dataset
- Clean and preprocess the data
- Perform feature engineering
- Encode categorical features
- Scale numerical features
- Determine the optimal number of clusters
- Train a K-Means clustering model
- Segment users into audience groups
- Generate personalized content recommendations
- Visualize results through an interactive Streamlit dashboard

---

##  Features

- User-friendly Streamlit interface
- Automatic data preprocessing
- Feature engineering
- Audience segmentation using K-Means
- Elbow Method & Silhouette Score for optimal clusters
- Interactive visualizations
- Audience profile analysis
- Personalized content recommendations
- Download segmented dataset

---

##  Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Model Serialization | Joblib |
| Version Control | Git & GitHub |

---

##  Project Structure

```text
CinemaPulse/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── netflix_users.csv
│
├── models/
│   ├── kmeans.pkl
│   └── scaler.pkl
│
├── outputs/
│   └── segmented_users.csv
│
└── src/
    ├── data_loader.py
    ├── preprocessing.py
    ├── clustering.py
    ├── recommendation.py
    ├── visualization.py
    └── utils.py
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/CinemaPulse.git
```

### 2. Navigate to the project

```bash
cd CinemaPulse
```

### 3. Create a virtual environment (Optional)

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  How to Run

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

##  Machine Learning Workflow

```
Dataset
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Encoding & Scaling
   ↓
K-Means Clustering
   ↓
Audience Segmentation
   ↓
Personalized Recommendations
```

---

##  Future Enhancements

- Real-time recommendation system
- Hybrid recommendation engine
- User authentication
- Cloud deployment (AWS/Azure/GCP)
- Deep Learning-based segmentation
- Collaborative filtering
- REST API integration

---
