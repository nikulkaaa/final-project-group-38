from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
] # add the names (in strings) of the metrics you implement

def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    raise NotImplementedError("To be implemented.")

class Metric(...):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number

    def __call__(self):
        raise NotImplementedError("To be implemented.")

# add here concrete implementations of the Metric class
    