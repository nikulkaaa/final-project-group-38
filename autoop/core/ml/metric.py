from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "accuracy",
    "average_precision",
    "log_loss",
    "mean_squared_error",
    "r_squared",
    "mean_absolute_error"
]   # add the names (in strings) of the metrics you implement


def get_metric(name: str) -> "Metric":
    """
    Return a metric instance given its str name.

    Factory function to get a metric by name.
    """

    metric_classes = {
        "accuracy": Accuracy,
        "average_precision": AveragePrecision,
        "log_loss": LogLoss,
        "mean_squared_error": MeanSquaredError,
        "r_squared": RSquared,
        "mean_absolute_error": MeanAbsoluteError
    }
    if name in metric_classes:
        return metric_classes[name]()
    else:
        error_message = (
            f"Metric '{name}' not found. "
            f"Available metrics: {', '.join(METRICS)}")
        raise ValueError(error_message)


class Metric(ABC):
    """
    Abstract Base class for all metrics.

    Metrics take ground truth and prediction as input and
    return a real number
    """
    @abstractmethod
    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """Abstract call method."""
        pass

    def evaluate(self, predictions: np.ndarray,
                            ground_truth: np.ndarray) -> float:
        """Alias for calling the metric as a function."""
        return self.__call__(predictions, ground_truth)

# Metrics for Classification


class Accuracy(Metric):
    """Class to measure the accuracy of predictions made by the model."""

    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """Calculates and returns the accuracy of predictions."""
        return np.sum(predictions == ground_truth) / len(predictions)


class AveragePrecision(Metric):
    """Class to calculate the Average Precision (AP) from prediction scores."""
    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """Calculates the average precision score."""
        # Sort scores and corresponding truth values
        sorted_indices = np.argsort(predictions)[::-1]
        ground_truth_sorted = ground_truth[sorted_indices]

        # Create an array of cumulative sums of the true labels
        cumulative_true = np.cumsum(ground_truth_sorted)

        # Calculate precision at each threshold
        precision_at_t = cumulative_true / (
            np.arange(len(ground_truth_sorted)) + 1)

        # Calculate recall at each threshold
        recall_at_t = cumulative_true / np.sum(ground_truth_sorted)

        # Calculate the changes in recall
        # prepend 0 to have the same length as precision
        recall_change = np.diff(recall_at_t, prepend=0)  

        # Calculate average precision as the sum of
        # products of precision and recall change
        average_precision = np.sum(precision_at_t * recall_change)
        return average_precision


class LogLoss(Metric):
    """Class to calculate the logarithmic loss for classification."""
    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """Calculates and returns the logarithmic loss to
        captrue confidence in predictions."""
        # Clip predictions to prevent log(0) and ensure numerical stability
        eps = 1e-15
        predictions = np.clip(predictions, eps, 1 - eps)

        # Convert ground truth to match the shape of predictions
        ground_truth = np.eye(predictions.shape[1])[ground_truth]

        # Return calculated Logarithmic Loss
        return -np.mean(np.sum(ground_truth * np.log(predictions), axis=1))


# Metrics for Regression
class MeanSquaredError(Metric):
    """Class to show the Mean Squared Error
    between predicted and actual values."""

    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """Calculates and returns the Mean Squared Error
        between predicted and actual values."""
        return np.sum((predictions - ground_truth)**2) / len(predictions)


class RSquared(Metric):
    """Class to show how well predictions approximate
    actual values based on the R-squared statistic."""
    def __call__(self, predictions: np.ndarray,
                            ground_truth: np.ndarray) -> float:
        """Calculates and returns the Mean Squared 
        Error between predicted and actual values."""
        # Calculate the sum of squared residuals
        ss_res = np.sum((ground_truth - predictions)**2)

        # Calculate the total sum of squares
        ss_total = np.sum((ground_truth - np.mean(ground_truth))**2)

        # Return the calculated R Squared statistic
        return 1 - (ss_res / ss_total)


class MeanAbsoluteError(Metric):
    """Class to calculate the Mean Absolute Error (MAE)
    between predicted and actual values."""
    def __call__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """Calculates and returns the Mean Absolute Error
        between predictions and ground truth values."""
        return np.mean(np.abs(predictions - ground_truth))
