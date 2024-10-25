from autoop.core.ml.model.model import Model
import numpy as np
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