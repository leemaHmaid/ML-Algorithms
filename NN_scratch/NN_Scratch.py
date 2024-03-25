import numpy as np
import matplotlib.pyplot as plt
from dataset import trainTest
from dataset import *
from sigmoid import *
from math import *

class NeuralNetwork:
  def __init__(self,h0,h1,h2):
    self.W1 = np.random.randn(h1,h0)*0.01
    self.b1 = np.zeros((h1,h2))
    self.W2 = np.random.randn(h2,h1)*0.01
    self.b2 = np.zeros((1,1))

 
  def forward_pass(self,X):
     Z1 = self.W1.dot(X) + self.b1
     A1 = sigmoid(Z1)
     Z2 =  self.W2.dot(A1)+ self.b2     
     A2 = sigmoid(Z2)
     return A2, Z2, A1, Z1
  
  def backward_pass(self,X,Y, A2, Z2, A1, Z1):

    m = Y.shape[1]
    dA2 = -(Y/A2) + ((1-Y)/(1-A2))
    dZ2 = A2-Y
    dA1 =  np.dot(self.W2.T,dZ2)
    dZ1 = dA1*d_sigmoid(Z1) #
    dW1 = (1/m)*np.dot(dZ1, X.T)
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    return dW1 , dW2, db1 ,db2
  
  def update(self,dW1, dW2, db1, db2, alpha ):

    self.W1 = self.W1 - alpha*dW1
    self.W2 = self.W2 - alpha*dW2
    self.b1 = self.b1 - alpha*db1
    self.b2 = self.b2 - alpha*db2

    return self.W1, self.W2, self.b1, self.b2

  def predict(self,X):

    A2,_,_,_= self.forward_pass(X)
    predictions = (A2 >= 0.5).astype(int)
    return predictions
  

#Evaluation

def loss(y_pred, Y):
  m = Y.shape[1]
  nloss = -(1/m) * np.sum(Y*np.log(y_pred) + (1-Y)*np.log(1-y_pred))
  return  nloss

"""## Accuracy"""

def accuracy(y_pred, y):

  accuracy = np.sum(y_pred == y)/y.shape[1]

  return accuracy



"""## Training loop"""
model = NeuralNetwork(h0=2,h1=10,h2=1)
X_train , Y_train , X_test, Y_test = trainTest()
alpha = 0.1
n_epochs = 10000
train_loss = []
test_loss = []
for i in range(n_epochs):
  ## forward pass
  A2, Z2, A1, Z1  =  model.forward_pass(X_train)
  ## backward pass
  dW1, dW2, db1, db2= model.backward_pass(X_train,Y_train, A2, Z2, A1, Z1)
  ## update parameters
  W1, W2, b1, b2 = model.update(dW1, dW2, db1, db2, alpha )

 
  ## plot boundary
  if i %1000 == 0:
     print(f"training loss after {i} epochs is {loss(A2, Y_train)}")
     AT2, ZT2, AT1, ZT1 = model.forward_pass(X_test)
     print(f"test loss after {i} epochs is {loss(AT2, Y_test)}")

## plot train et test losses

 

y_pred = model.predict(X_train)
train_accuracy = accuracy(y_pred, Y_train)
print ("train accuracy :", train_accuracy)

y_pred = model.predict(X_test)
test_accuracy = accuracy(y_pred, Y_test)
print ("test accuracy :", test_accuracy)



