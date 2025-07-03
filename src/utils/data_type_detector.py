import numpy as np
import pandas as pd

def detect_data_type(X):
    """
    Inspect X and return a (category, confidence) tuple.
    Categories:
      - Numeric/Vector Measures
      - Binary/Categorical Measures
      - Distribution/Histogram Measures
      - Sequence/Time-Series Measures
      - Mixed-Type Measures
      - Graph & Structure Measures
      - Universal/Compression-Based
    """
    # Ensure DataFrame
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Simple heuristic: if all numeric → Numeric/Vector
    if all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        # time‑series if index is datetime
        if isinstance(X.index, pd.DatetimeIndex):
            return "Sequence/Time-Series Measures", 0.90
        return "Numeric/Vector Measures", 0.95

    # If mixture of types → Mixed‑Type
    if any(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes) and \
       any(pd.api.types.is_object_dtype(dtype) for dtype in X.dtypes):
        return "Mixed-Type Measures", 0.85

    # If all object/string → Binary/Categorical
    if all(pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype)
           for dtype in X.dtypes):
        return "Binary/Categorical Measures", 0.90

    # Fallback to Distribution
    return "Distribution/Histogram Measures", 0.70
