from typing import List, Tuple
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from io import StringIO


def preprocess_features(
        features: List[Feature],
        dataset: Dataset,
        one_hot_encode_target: bool = False
        ) -> List[Tuple[str, np.ndarray, dict]]:
    """Preprocess features.

    Args:
        features (List[Feature]): List of features.
        dataset (Dataset): Dataset object.
        one_hot_encode_target (bool): Whether to
        one-hot encode the target feature.

    Returns:
        List[Tuple[str, np.ndarray, dict]]: List of preprocessed features.
        Each tuple contains feature name,
        data array, and an artifact dictionary.
    """
    results = []
    raw_data = dataset.read()
    if isinstance(raw_data, bytes):
        raw = pd.read_csv(StringIO(raw_data.decode('utf-8')))
    else:
        raw = raw_data

    for feature in features:
        if feature.type == "categorical":
            if one_hot_encode_target:
                # Use integer encoding for the target feature
                data = raw[feature.name].astype(
                    'category'
                ).cat.codes.values.reshape(-1, 1)
                artifact = {"type": "LabelEncoding"}
                results.append((feature.name, data, artifact))
            else:
                encoder = OneHotEncoder(sparse_output=False)
                data = encoder.fit_transform(
                    raw[feature.name].values.reshape(-1, 1)
                ).toarray()
                artifact = {"type": "OneHotEncoder", "encoder": encoder}
                results.append((feature.name, data, artifact))

        elif feature.type == "numerical" or (feature.is_target and not one_hot_encode_target):
            # Process target as numerical
            data = raw[feature.name].values.reshape(-1, 1)
            artifact = {"type": "None", "scaler": None}
            results.append((feature.name, data, artifact))

    # Sort for consistency
    results = list(sorted(results, key=lambda x: x[0]))
    return results
