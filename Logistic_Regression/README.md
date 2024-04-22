# Logistic Regression

This folder contains the implementation of the Logistic Regression model in Python. Logistic Regression is a popular classification algorithm that models the relationship between input features and categorical targets using the logistic function.

## folder Structure

- `class.py`: Contains the implementation of the `LogisticRegression` class, which encapsulates the logic and functionality of the Logistic Regression model.
- `data.py`: Generates synthetic data for training and testing purposes. You can modify this file to use your own datasets.
- `training.py`: Provides an example usage of the Logistic Regression model, including model training and evaluation.
## Usage

The `LogisticRegression` class provides the following methods:

- `__init__(self, lr, n_epochs)`: Initializes the Logistic Regression model with the learning rate (`lr`) and the number of epochs (`n_epochs`).
- `add_ones(self, x)`: Adds a column of ones to the input feature matrix `x`.
- `sigmoid(self, x)`: Computes the sigmoid function.
- `predict_proba(self, X)`: Predicts the probabilities of the positive class for new data `X`.
- `predict(self, X, threshold=0.5)`: Predicts the class labels for new data `X` based on a given threshold.
- `evaluate(self, X, y)`: Evaluates the model's performance using accuracy and the F1 score.
- `fit(self, x, y)`: Fits the model to the training data `x` and corresponding targets `y`.