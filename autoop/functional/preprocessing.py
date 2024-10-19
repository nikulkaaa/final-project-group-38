from typing import List, Tuple
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_features(features: List[Feature], dataset: Dataset) -> List[Tuple[str, np.ndarray, dict]]:
    """Preprocess features.
    Args:
        features (List[Feature]): List of features.
        dataset (Dataset): Dataset object.
    Returns:
        List[str, Tuple[np.ndarray, dict]]: List of preprocessed features. Each ndarray of shape (N, ...)
    """
    results = []
    raw = dataset.read()
    for feature in features:
        if feature.type == "categorical":
            encoder = OneHotEncoder()
            data = encoder.fit_transform(raw[feature.name].values.reshape(-1, 1)).toarray()
            aritfact = {"type": "OneHotEncoder", "encoder": encoder.get_params()}
            results.append((feature.name, data, aritfact))
        if feature.type == "numerical":
            scaler = StandardScaler()
            data = scaler.fit_transform(raw[feature.name].values.reshape(-1, 1))
            artifact = {"type": "StandardScaler", "scaler": scaler.get_params()}
            results.append((feature.name, data, artifact))
    # Sort for consistency
    results = list(sorted(results, key=lambda x: x[0]))
    return results
