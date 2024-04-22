# Machine Learning Fundamentals Implemetations



This repository contains Four Files : Linear regression model , logistic regression model and neural network implementation from scratch and a PCA implementation. This README file provides a structured and summarized explanation of both implementations.

## Linear Regression

The linear_regression folder implements a simple linear regression algorithm. It provides functionalities to fit the model to training data, make predictions on new data, and evaluate the model's performance using metrics such as mean squared error (MSE) and R-squared.


## Logistic Regression

The logistic_regression folder implements logistic regression, a popular algorithm for binary classification problems. It includes functions to fit the model, predict class labels, and assess performance using metrics like accuracy, precision, recall, and F1-score.

## Neural Network Implementation

The neural network implementation is a code file that demonstrates how to create a basic neural network model from scratch. The purpose of this implementation is to provide a clear understanding of the fundamental concepts behind neural networks without relying on external libraries or frameworks.
Neural networks are computational models inspired by the human brain. They consist of interconnected nodes, called neurons, organized in layers. The neural network implementation in this repository focuses on a simple feedforward neural network, which means the information flows in one direction, from the input layer to the output layer.
The code file contains functions to initialize the network, train it using a training dataset, and make predictions on new data. The training process involves adjusting the weights and biases of the network based on the error between predicted and actual outputs. The implementation uses basic mathematical operations, such as matrix multiplication and activation functions, to perform forward propagation and backpropagation, which are essential steps in training a neural network.
This implementation serves as a starting point for understanding the underlying concepts of neural networks. It is not optimized for large-scale or complex problems, but rather focuses on simplicity
and educational purposes.

![The-NN-used-for-detection-A-sigmoid-activation-function-is-used-for-the-hidden-and](https://github.com/leemaHmaid/ML-Algorithms/assets/52715254/e0f9df94-e1c4-46a8-8805-9e564df489ed)

## Structure :
We have two files:
1- NN_scratch : In this file the class of the model is created, the file contains the model class and sigmoid activation function ,and the traing functions 
2- DataSet : In this file the data set generationg function is implememnted 

## Neural Network

The `neural_network.py` file contains the implementation of a feedforward neural network using the sigmoid activation function. The neural network class (`NeuralNetwork`) is defined with methods for initialization, forward and backward passes, parameter updates, and prediction.


## Dataset:
The `dataset.py` file contains functions for generating synthetic data for training and testing the neural network. The generateData() function generates the data, and the trainTest() function splits the data into training and testing sets.
Other utility files, such as sigmoid.py, may be required for the functionality of the neural network  

## Install required dependencies:
before running the code make sure you installed the required dependencied such as :

-numpy
-matplotlib
-sklearn
-pandas



## Principal Component Analysis (PCA)

The PCA implementation in this repository demonstrates a popular dimensionality reduction technique used in machine learning and data analysis. PCA aims to transform a high-dimensional dataset into a lower-dimensional representation while preserving the essential information.
PCA is particularly useful when dealing with datasets that contain a large number of features or variables. It helps in identifying the most significant patterns and reducing the data's complexity, making subsequent analysis more efficient.
The code file for PCA contains functions to perform the following steps:
Standardization: The input data is standardized to have zero mean and unit variance, ensuring that all features contribute equally to the analysis.
Covariance Matrix Computation: The covariance matrix is calculated based on the standardized data. It describes the relationships between the different features.
Eigenvalue Decomposition: The covariance matrix is decomposed into its eigenvectors and eigenvalues, which represent the principal components and their corresponding importance.
Dimensionality Reduction: The eigenvectors are ranked by their corresponding eigenvalues, and a subset of them is selected to form the principal components. These components capture the most significant information in the data.
Data Transformation: The original data is projected onto the selected principal components, resulting in a lower-dimensional representation.


## PCA
The pca.py file contains the implementation of the Principal Component Analysis (PCA) algorithm. The PCA class is defined with methods for fitting the model to input data and transforming the data using the fitted model.
