import numpy as np

def sigmoid(z):
    sigFunction = 1/(1+np.exp(-z))
    return sigFunction


def d_sigmoid(z):
   dsigmoid = sigmoid(z)*(1-sigmoid(z))
   return dsigmoid
