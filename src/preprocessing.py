# src/preprocessing.py
"""
Data preprocessing module: dynamic column detection, smart imputation, encoding, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Dynamically identify column types in any uploaded dataset.

    Returns
    -------
    dict with keys: 'id_cols', 'date_cols', 'numerical_cols', 'categorical_cols'
    """
    id_cols = []
    date_cols = []
    numerical_cols = []
    categorical_cols = []

    for col in df.columns:
        col_clean = str(col).strip()
        col_lower = col_clean.lower()

        # Check for ID / Name columns
        if any(keyword in col_lower for keyword in ['id', 'user_id', 'userid', 'name', 'customer_id', 'index', 'uuid']):
            id_cols.append(col)
            continue

        # Check for date / time columns
        if any(keyword in col_lower for keyword in ['date', 'login', 'time_stamp', 'timestamp', 'created', 'joined', 'last_seen']):
            # Verify if converting to datetime works for at least some non-null values
            try:
                sample_parsed = pd.to_datetime(df[col].dropna().head(10), errors='coerce')
                if sample_parsed.notna().sum() > 0:
                    date_cols.append(col)
                    continue
            except Exception:
                pass

        # Check numerical vs categorical
        if pd.api.types.is_numeric_dtype(df[col]):
            # If unique values are very low (e.g. <= 3 for integer codes), treat as categorical, else numerical
            if df[col].nunique() <= 3 and not pd.api.types.is_float_dtype(df[col]):
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            # String / object / category
            categorical_cols.append(col)

    return {
        'id_cols': id_cols,
        'date_cols': date_cols,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }

def handle_missing_values(df: pd.DataFrame, column_info: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Smart missing value imputation:
    - Numerical columns -> median
    - Categorical columns -> mode or 'Unknown'
    - Date columns -> drop rows only if date is null and critical
    """
    df = df.copy()

    for col in column_info['numerical_cols']:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed missing values in '{col}' with median: {median_val}")

    for col in column_info['categorical_cols']:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Imputed missing values in '{col}' with mode: {mode_val}")

    return df

def feature_engineering(df: pd.DataFrame, date_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Perform feature engineering:
    - Convert date column (e.g. 'Last_Login') to recency feature 'Days_Since_Last_Login'.
    Returns updated DataFrame and updated list of numerical feature names.
    """
    df = df.copy()
    engineered_num_cols = []

    for date_col in date_cols:
        try:
            parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
            if parsed_dates.notna().any():
                ref_date = parsed_dates.max()
                recency_col = f"Days_Since_{date_col.replace(' ', '_')}"
                df[recency_col] = (ref_date - parsed_dates).dt.days.fillna(0)
                engineered_num_cols.append(recency_col)
                logger.info(f"Engineered recency feature '{recency_col}' from '{date_col}'")
        except Exception as e:
            logger.warning(f"Could not compute recency for column '{date_col}': {e}")

    return df, engineered_num_cols

def preprocess_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, ColumnTransformer]:
    """
    Complete dynamic preprocessing pipeline:
        1. Clean column names & remove duplicates
        2. Detect column roles (IDs, dates, numerical, categorical)
        3. Feature engineering (recency calculation)
        4. Smart missing value imputation
        5. One-hot encoding for categorical attributes
        6. Standard scaling for all numerical features

    Returns
    -------
    Tuple:
        - df_clean: Processed DataFrame (with feature columns + cluster metadata)
        - X_scaled: Scaled feature matrix ready for clustering
        - scaler: Fitted StandardScaler
        - preprocessor: Fitted ColumnTransformer (or encoder info)
    """
    df_clean = df.copy()

    # 1. Remove exact duplicate rows
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 2. Dynamic column classification
    column_info = detect_column_types(df_clean)

    # 3. Feature engineering on date columns
    df_clean, engineered_num_cols = feature_engineering(df_clean, column_info['date_cols'])
    column_info['numerical_cols'].extend(engineered_num_cols)

    # 4. Smart missing value imputation
    df_clean = handle_missing_values(df_clean, column_info)

    # Filter columns to use in clustering (exclude ID and original date columns)
    num_cols = [c for c in column_info['numerical_cols'] if c in df_clean.columns]
    cat_cols = [c for c in column_info['categorical_cols'] if c in df_clean.columns]

    # Ensure we have at least some features
    if not num_cols and not cat_cols:
        raise ValueError("No suitable numerical or categorical features found for clustering.")

    # 5. One-Hot Encode Categorical Features
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(df_clean[cat_cols])
        encoded_feature_names = encoder.get_feature_names_out(cat_cols)
    else:
        cat_encoded = np.empty((len(df_clean), 0))
        encoded_feature_names = np.array([])
        encoder = None

    # Extract numerical features matrix
    if num_cols:
        num_matrix = df_clean[num_cols].values
    else:
        num_matrix = np.empty((len(df_clean), 0))

    # Combine categorical encoded matrix + numerical matrix
    X_combined = np.hstack([cat_encoded, num_matrix]) if cat_cols and num_cols else (cat_encoded if cat_cols else num_matrix)

    # 6. Scale combined feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    preprocessor = {
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'encoder': encoder,
        'feature_names': list(encoded_feature_names) + list(num_cols)
    }

    return df_clean, X_scaled, scaler, preprocessor