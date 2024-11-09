
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.metric import Metric, get_metric
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
from copy import deepcopy
from collections import Counter
from pydantic import BaseModel, field_validator, PrivateAttr, Field


class Model(BaseModel, ABC):
    """Abstract Base Class for a Model."""

    _is_trained: bool = PrivateAttr(default=False)

    @property
    def is_trained(self) -> bool:
        """Get the _is_trained private attribute."""
        return deepcopy(self._is_trained)

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        """Change the is_trained value."""
        self._is_trained = value

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        pass

# Models for Classification: knn, DecisionTreeClassifier, MLP classifier

class KNearestNeighbors(Model):
    """Model for the K Nearest Neighbours classification model"""
    _model: KNeighborsClassifier = PrivateAttr()
    type: str = Field("classification")

    def __init__(self, **kwargs) -> None:
        """Constructor for the K Nearest Neighbours classifier."""
        super().__init__()
        # Instantiate the DecisionTreeClassifier in the private attribute
        self._model = KNeighborsClassifier(**kwargs)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Use the built-in fit method from the KNearestNeighbours
        class from scikit-learn.

        Calculate the parameters and stores them in the parameters dictionary.
        """
        # Fit the Lasso model to the provided data
        self._model.fit(observations.T, ground_truth)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained K Nearest Neighbours Classifier."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self._model.predict(X.T)

class DecisionTreeClassifierModel(Model):
    """Model for the decision tree classification model"""

    _tree: DecisionTreeClassifier = PrivateAttr()
    type: str = Field("classification")

    def __init__(self, **kwargs) -> None:
        """Constructor for the decision tree classifier."""
        super().__init__()
        # Instantiate the DecisionTreeClassifier in the private attribute
        self._tree = DecisionTreeClassifier(**kwargs)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Use the built-in fit method from the DecisionTreeClassifier
        class from scikit-learn.

        Calculate the parameters and stores them in the parameters dictionary.
        """
        # Fit the Lasso model to the provided data
        self._tree.fit(observations.T, ground_truth)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained Decision Tree Classifier."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self._tree.predict(X.T)


class MLPClassifierModel(Model):
    """A classification model based on MLP."""
    type: str = Field("classification")

    def __init__(self, **kwargs) -> None:
        """Constructor for the MLP Classifier."""
        super().__init__()
        # Instantiate the MLPClassifier and store it in the private attribute
        self._model = MLPClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the MLP Classifier on the given data."""
        self._model.fit(X.T, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained MLP Classifier."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self._model.predict(X.T)

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric: Metric) -> float:
        """Evaluate the model on the given data using the specified metric."""
        predictions = self.predict(X.T)
        return metric(predictions, y)


# Models for Regression: mulitiple variable regression, lasso

class MultipleLinearRegression(Model):
    """
    A model implementing the regularized version of linear regression.

    It uses the MLR Wrapper model from scikit-learn for its calculations.
    """
    type: str = Field("regression")

    def __init__(self) -> None:
        """Initialize the MLR Wrapper model."""
        super().__init__()
        self._model = LinearRegression()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Use the built-in fit method from the MLR class from scikit-learn.

        Calculate the parameters and stores them in the parameters dictionary.
        """
        # Fit the Lasso model to the provided data
        self._model.fit(observations.T, ground_truth)

        # Store the coefficients and intercept in the _parameters dictionary
        self._parameters = {
            "coefficients": self._model.coef_,
            "intercept": self._model.intercept_,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions from observations.

        It uses the built-in predict function of the MLR model.
        """
        # Make predictions using the Lasso model
        return self._model.predict(observations.T)

class LassoModel(Model):
    """
    A model implementing the regularized version of linear regression.

    It uses the Lasso model from scikit-learn for its calculations.
    """
    type: str = Field("regression")

    def __init__(self) -> None:
        """Initialize the LassoWrapper model."""
        super().__init__()
        self._lasso = Lasso()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Use the built-in fit method from the Lasso class from scikit-learn.

        Calculate the parameters and stores them in the parameters dictionary.
        """
        # Fit the Lasso model to the provided data
        self._lasso.fit(observations.T, ground_truth)

        # Store the coefficients and intercept in the _parameters dictionary
        self._parameters = {
            "coefficients": self._lasso.coef_,
            "intercept": self._lasso.intercept_,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions from observations.

        It uses the built-in predict function of the Lasso model.
        """
        # Make predictions using the Lasso model
        return self._lasso.predict(observations.T)


class RadiusNeighborsModel(Model):
    """
    A model implementing regression based on neighbors within a
    fixed radius. (Radius Neighbors Regressor)

    It uses the RadiusNeighborsRegressor model from scikit-learn
    for its calculations.
    """
    _model = RadiusNeighborsRegressor()
    type: str = Field("regression")

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the RadiusNeighborsRegressor model to the provided data.
        """
        self._model.fit(observations.T, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the fitted RadiusNeighborsRegressor model.
        """
        return self._model.predict(observations.T)
