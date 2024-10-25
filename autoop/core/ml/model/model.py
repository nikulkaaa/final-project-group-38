
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.metric import Metric, get_metric
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsRegressor
import numpy as np
from copy import deepcopy
from typing import Literal
from collections import Counter
from pydantic import BaseModel, field_validator, PrivateAttr, Field

class Model(BaseModel, ABC):

    _is_trained : bool = PrivateAttr(default = False)

    @property
    def is_trained(self):
        """Get the _is_trained private attribute."""
        return deepcopy(self._is_trained)
    
    @is_trained.setter
    def is_trained(self, value):
        """Change the is_trained value."""
        self._health = value
        return self._health

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        pass

# Models for Classification: knn, DecisionTreeClassifier

class KNearestNeighbors(Model):
    """
    Implement the K-Nearest Neighbors algorithm.

    The algorithm predicts the class of a given data point,
    based on the majority class of its nearest neighbors.

    Attributes:
        k: An integer storing the number of nearest neighbors,
          to consider for classification
    """
    k: int = Field(default = 3)

    @field_validator('k')
    @classmethod
    def validate_k(cls, k: int):
        """Validate k, if it complies to conditions."""
        if not isinstance(k, int):
            raise ValueError('k must be an integer')
        if k <= 0:
            raise ValueError('k must be greater than zero')
        if k % 2 == 0:
            raise ValueError('k must be an odd number')

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        """
        Fits the given dataset in the model's internal parameters.

        It ensures that the observations and ground truth are valid
        numpy arrays, and that the dataset is correctly structured
        with samples in the row dimension and features in the
        column dimension.
        """
        # Check if the inputs are in form of np.nd array
        if not isinstance(observations, np.ndarray) or not isinstance(
            ground_truth, np.ndarray
        ):
            raise TypeError("Inputs must be of type numpy.ndarray")

        # Check if the number of samples is in the row dimension,
        #  while the variables are in the column dimension
        if observations.shape[0] < observations.shape[1]:
            raise ValueError(
                "Row (sample) and column (variable) dimensions don't align."
            )


    def predict(self, observations: np.ndarray):
        """
        Predict the distance for each observation.

        This method calculates the distance between the observation
        and all training data points stored in the model,
        finds the `k` nearest neighbors,
        and assigns the most common label to the prediction.

        It returns predictions for each of the observations
        """
        predictions = []

        for observation in observations:
            # STEP 1: calculate distance between an observation and every point
            distances = np.linalg.norm(
                self._parameters["observations"] - observation, axis=1
            )

            # STEP 2: sort the array of the distances and take first
            # k → have the closest ones at the start
            k_indices = np.argsort(distances)[: self.k]

            # STEP 3: check the label aka ground truth of those points
            k_nearest_labels = [
                self._parameters["ground_truth"][i] for i in k_indices
            ]

            # now we have k = 3, 3 labels inside an array #

            # STEP 4: take the most common label and return to the caller
            most_common = Counter(k_nearest_labels).most_common(1)

            # STEP 5: add prediction to the list of predictions
            predictions.append(most_common[0][0])
        return np.array(predictions)

class DecisionTreeClassifierModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        # Instantiate the DecisionTreeClassifier and store it in the private attribute
        self._tree = DecisionTreeClassifier(**kwargs)
    
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Use the built-in fit method from the DecisionTreeClassifier class from scikit-learn.

        Calculate the parameters and stores them in the parameters dictionary.
        """
        # Fit the Lasso model to the provided data
        self._tree.fit(observations, ground_truth)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained Decision Tree Classifier."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict(X)

class MLPClassifierModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        # Instantiate the MLPClassifier and store it in the private attribute
        self._model = MLPClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the MLP Classifier on the given data."""
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained MLP Classifier."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric: Metric) -> float:
        """Evaluate the model on the given data using the specified metric."""
        predictions = self.predict(X)
        return metric(predictions, y)
    

# Models for Regression: mulitiple variable regression, lasso

class MultipleLinearRegression(Model):
    """
    A multiple linear regression model.

    Generates a prediction based on a number of observations, using a model
    trained to minimize the difference between the actual and target values.
    """

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Calculate and store the parameters."""
        # Check if the inputs are in the form of np.nd array
        if not isinstance(observations, np.ndarray) or not isinstance(
            ground_truth, np.ndarray
        ):
            raise TypeError("Inputs must be of type numpy.ndarray")

        # Check if the number of samples is in the row dimension,
        # while the variables are in the column dimension
        if observations.shape[0] < observations.shape[1]:
            raise ValueError(
                "Row (sample) and column (variable) dimensions don't align."
            )

        # Add a column of ones to the observations matrix
        # to account for the intercept (bias) term
        n_samples = observations.shape[0]
        x_bias = np.hstack(
            [observations, np.ones((n_samples, 1))]
        )  # Add bias column (n, p+1)

        # x_transpose_x is the augmented transpose matrix observations,
        # dotted with the augmented observations
        x_transpose_x = np.dot(x_bias.T, x_bias)

        # Check if the matrix x_transpose_x can be inverted (determinant)
        if np.linalg.det(x_transpose_x) == 0:
            raise ValueError("Matrix is singular and cannot be inverted")

        # Invert matrix x_transpose_x and calculate weights
        xtx_inv = np.linalg.inv(x_transpose_x)
        xty = np.dot(x_bias.T, ground_truth)
        weight = np.dot(xtx_inv, xty)

        self._parameters = {"Weight": weight}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Make predictions for the provided observations."""
        # Check if the input is and nd array
        if not isinstance(observations, np.ndarray):
            raise TypeError("Inputs must be of type numpy.ndarray")

        # Get the weight value from the parameters dictionary
        weights = self._parameters.get("Weight")
        # Add a column of ones for bias
        n_samples = observations.shape[0]
        x_bias = np.hstack([observations, np.ones((n_samples, 1))])

        # Multiply the matrices, x for observations
        # and w for weights store in the parameters dictionary,
        # and return these predictions
        return np.dot(x_bias, weights)


class LassoModel(Model):
    """
    A model implementing the regularized version of linear regression.

    It uses the Lasso model from scikit-learn for its calculations.
    """

    def __init__(self):
        """Initialize the LassoWrapper model."""
        super().__init__()
        self._lasso = sklearn.linear_model.Lasso()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Use the built-in fit method from the Lasso class from scikit-learn.

        Calculate the parameters and stores them in the parameters dictionary.
        """
        # Fit the Lasso model to the provided data
        self._lasso.fit(observations, ground_truth)

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
        return self._lasso.predict(observations)


class RadiusNeighborsModel(Model):
    """
    A model implementing regression based on neighbors within a fixed radius. (Radius Neighbors Regressor)

    It uses the RadiusNeighborsRegressor model from scikit-learn for its calculations.
    """
    _model = RadiusNeighborsRegressor()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the RadiusNeighborsRegressor model to the provided data.
        """
        self._model.fit(observations, ground_truth)
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the fitted RadiusNeighborsRegressor model.
        """
        return self._model.predict(observations)
    



