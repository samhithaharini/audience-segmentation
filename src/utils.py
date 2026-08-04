# src/utils.py
"""
Utility functions for the OTT Audience Segmentation project.
"""

import os
import pickle
from typing import Any

def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save a Python object to a pickle file.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    filepath : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    filepath : str
        Source file path.

    Returns
    -------
    Any
        Deserialized object.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)