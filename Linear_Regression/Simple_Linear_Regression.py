import numpy as np
from sklearn.linear_model import LinearRegression

class My_LinearRegression:
    """
    A custom linear regression class that trains a model and provides prediction functionality.
    """

    def __init__(self, X, y):
        """
        Initializes the class with training data.

        Args:
          X (np.ndarray): A 2D array representing the feature matrix. Features 
          y (np.ndarray): A 1D array representing the target vector. Targets 
        """

        self.X = np.array(X, dtype=float)
        self.y = np.array(y, dtype=float)

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if self.y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("The number of samples in X and y must match.")

        self.reg = LinearRegression()

    def create_and_train(self):
        """
        Trains the linear regression model using the stored data.

        Returns:
          tuple: A tuple containing the learned coefficients and intercept.
          The coeffients are values for each feature, and the intercept is the bias term.
        """

        self.reg.fit(self.X, self.y)
        return self.reg.coef_, self.reg.intercept_

    def predict_sample(self, sample):
        """
        Predicts the target value for a new sample.

        Args:
          sample (np.ndarray or list): A 1D array or list representing the new sample to predict for.

        Returns:
          float: The predicted target value for the given sample.
        """

        sample = np.array(sample, dtype=float)

        if sample.ndim != 1:
            raise ValueError("Sample must be a 1D array or list.")
        if sample.shape[0] != self.X.shape[1]:
            raise ValueError("Sample size must match the number of features in X.")

        return self.reg.predict(sample.reshape(1, -1))[0]

# This code creates a linear regression model to estimate house prices based on square meters.
# Training: It uses historical data (square meters and prices) to calculate:

# Square Meters = Independent Variable (what you already know, the data you base your guess on).
# House Price = Dependent Variable (what you want to predict, based on square meters).
# Linear Regression: The model learns the relationship between square meters and house prices.
# Coefficient: How much the price increases per additional square meter.
# Intercept: The base price when square meters are 0.
# Prediction: For a new house, the model calculates the estimated price using the formula:
# Price = Coefficient * SquareMeters + Intercept
# Example: If the coefficient is 200 and the intercept is 50,000, for 80 square meters, the price would be:
# Price = 200 * 80 + 50,000 = 66,000