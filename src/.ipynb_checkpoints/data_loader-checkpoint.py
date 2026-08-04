# src/data_loader.py
"""
Data loading and validation module.
"""

import pandas as pd
from typing import Optional

EXPECTED_COLUMNS = [
    'User_ID', 'Name', 'Age', 'Country', 'Subscription_Type',
    'Watch_Time_Hours', 'Favorite_Genre', 'Last_Login'
]

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the OTT user dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    ValueError
        If the dataset is missing required columns.
    """
    df = pd.read_csv(filepath)
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    return df

def preview_dataset(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the first n rows of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n : int, optional
        Number of rows to preview, by default 5.

    Returns
    -------
    pd.DataFrame
        First n rows.
    """
    return df.head(n)

def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Generate dataset summary information.

    Returns
    -------
    dict
        Contains shape, column dtypes, missing counts, and duplicate rows.
    """
    return {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }