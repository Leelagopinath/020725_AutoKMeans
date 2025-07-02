# File: src/data/loader.py

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_preprocessed_data(file_path):
    """Load a preprocessed dataset with metadata"""
    df = pd.read_csv(file_path)
    name = os.path.basename(file_path).replace(".csv", "")
    
    # Extract metadata if available
    description = "Preprocessed dataset"
    if "description" in df.columns:
        description = df.iloc[0]["description"]
        df = df.drop(columns=["description"])
    
    # Convert to numpy array
    X = df.values
    
    return X, {
        "name": name,
        "description": description,
        "points": len(X),
        "dims": X.shape[1]
    }

def preprocess_data(X):
    """Robust preprocessing pipeline"""
    # Handle missing values
    if isinstance(X, pd.DataFrame):
        X = X.fillna(X.mean())
    
    # Convert to numpy if needed
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    # Scale numerical features
    if X.dtype.kind in 'iuf':  # Integer, unsigned integer, float
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    return X