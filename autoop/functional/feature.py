from typing import List
import pandas as pd
import io
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Decode the dataset data from bytes, interpret it as a CSV, and determine
    the type of each feature (column).

    Args:
        dataset (Dataset): A Dataset object that includes data encoded as bytes
        representing a CSV.

    Returns:
        List[Feature]: A list of Feature objects with names and detected types
        (numerical or categorical).
    """
    # Decode the byte data into a DataFrame
    if isinstance(dataset, bytes):
        # Bytes to string
        data_string = io.StringIO(dataset.decode('utf-8'))
        # Read the string as a DataFrame
        df = pd.read_csv(data_string)
    else:
        raise ValueError("Dataset data is not in the expected bytes format.")

    features = []
    for column in df.columns:
        # Determine the type of each column based on its dtype
        if pd.api.types.is_numeric_dtype(df[column]):
            unique_values = df[column].dropna().unique()
            # Check if the feature is binary
            if set(unique_values).issubset({0, 1}):
                feature_type = 'categorical'
            else:
                feature_type = 'numerical'
        elif pd.api.types.is_categorical_dtype(df[column]) or df[
                column].dtype == object:
            feature_type = 'categorical'
        else:
            feature_type = 'unknown'  # Fallback for any unexpected dtype

        # Create a new Feature object and append it to the list
        features.append(Feature(name=column, type=feature_type))

    return features
