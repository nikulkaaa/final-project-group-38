from typing import List, Dict, Any
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """The class for the pipeline."""
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """Constructor for the pipeline."""
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and (
                model.type != "classification"):
            raise ValueError(
                "Model type must be classification for categorical")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature")

    def __str__(self) -> str:
        """Magic str method."""
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """Returns the model"""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Used to get the artifacts generated during the
        pipeline execution to be saved
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data)))
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        # Pass a flag to indicate if the target should be one-hot encoded
        is_classification = (self._target_feature.type == 'categorical')
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset, one_hot_encode_target=is_classification)[0]

        self._register_artifact(target_feature_name, artifact)

        # Preprocess input features without additional flags for one-hot encoding
        input_results = preprocess_features(self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)

        # Assign input and output vectors
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact) in input_results]

        # Ensure input vectors have matching number of rows
        if len(self._input_vectors) > 0:
            num_rows = self._input_vectors[0].shape[0]
            for vector in self._input_vectors:
                if vector.shape[0] != num_rows:
                    raise ValueError("Input vectors have different number of rows.")
        
        # Log the shape for debugging
        if self._output_vector.shape[1] > 1 and not is_classification:
            raise ValueError(f"Unexpected output vector shape {self._output_vector.shape} for regression.")

    def _split_data(self) -> None:
        # Compact the input vectors into a single 2D array
        input_matrix = self._compact_vectors(self._input_vectors)
        target_vector = np.array(self._output_vector)

        # Define the splitting index based on the split ratio
        split_index = int(self._split * len(target_vector))

        # Split the data into training and testing sets
        self._train_X, self._test_X = input_matrix[
            :split_index], input_matrix[split_index:]
        self._train_y, self._test_y = target_vector[
            :split_index], target_vector[split_index:]

    @property
    def train_X(self) -> np.array:
        """Returns the observations."""
        return self._train_X

    @property
    def train_y(self) -> np.array:
        """Returns the ground truth."""
        return self._train_y

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        # Convert each vector to a 2D array if it is 1D
        reshaped_vectors = [vector.reshape(-1, 1)
                            if vector.ndim == 1 else vector
                            for vector in vectors]

        result = np.concatenate(reshaped_vectors, axis=1)
        return result

    def _train(self) -> None:
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        # Evaluate on the test set
        test_X = self._compact_vectors(self._test_X)
        test_Y = self._test_y
        test_predictions = self._model.predict(test_X)

        self._test_metrics_results = []
        for metric in self._metrics:
            result = metric(test_predictions, test_Y)
            self._test_metrics_results.append((metric, result))

        # Evaluate on the training set
        train_X = self._compact_vectors(self._train_X)
        train_Y = self._train_y
        train_predictions = self._model.predict(train_X)

        self._train_metrics_results = []
        for metric in self._metrics:
            result = metric(train_predictions, train_Y)
            self._train_metrics_results.append((metric, result))

        # Store the predictions separately for further use if needed
        self._train_predictions = train_predictions
        self._test_predictions = test_predictions

    def execute(self) -> Dict[str, Any]:
        """Function to execute the pipeline processes."""
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "train_metrics": self._train_metrics_results,
            "test_metrics": self._test_metrics_results,
            "train_predictions": self._train_predictions,
            "test_predictions": self._test_predictions,
        }
