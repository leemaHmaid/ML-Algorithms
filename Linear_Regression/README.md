# Linear Regression

This repository contains an implementation of the Linear Regression model in Python. Linear Regression is a fundamental machine learning algorithm used for regression tasks. It aims to find the best-fit line that minimizes the mean squared error between the predicted and actual values.

## Repository Structure

- `class.py`: Contains the implementation of the `LinearRegression` class, which encapsulates the logic and functionality of the Linear Regression model.
- `data.py`: Generates synthetic data for training and testing purposes. You can modify this file to use your own datasets.
- `training.py`: Provides an example usage of the `LinearRegression` class, including model training and evaluation.

## Usage

### `class.py`

The `LinearRegression` class provides the following methods:

- `__init__(self, lr, n_epochs)`: Initializes the Linear Regression model with the learning rate (`lr`) and the number of epochs (`n_epochs`).
- `add_ones(self, x)`: Adds a column of ones to the input feature matrix `x`.
- `linear_function(self, x)`: Computes the linear function using the learned weights.
- `mse_loss(self, x, y_true)`: Calculates the mean squared error loss between the predicted and true values.
- `gradient(self, x, y)`: Computes the gradient of the loss function.
- `predict(self, X)`: Predicts the values for new data `X`.
- `evaluate(self, X, y)`: Evaluates the model's performance using mean squared error loss.
- `fit(self, x, y)`: Fits the model to the training data `x` and corresponding targets `y`.
- `accuracy(self, y_true, y_pred)`: Calculates the accuracy of the model's predictions (Note: This method is not applicable for regression tasks).

### `data.py`

The `data.py` file generates synthetic data for training and testing purposes. You can modify the file to use your own datasets. It currently includes functions to generate linear relationships between features and targets with added noise.

### `training.py`

The `training.py` file provides an example usage of the Linear Regression model. It includes data generation, model training, and evaluation. You can modify this file to suit your specific requirements.

