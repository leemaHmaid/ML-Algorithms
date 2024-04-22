import numpy as np
class LinearRegression:
    def __init__(self,lr ,n_epochs) :
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_losses = []
        self.w = None

    def add_ones(self,x):
        ones =  np.ones((x.shape[0],1))
        return np.hstack((ones,x))

    def linear_function(self, x ):
        return x.dot(self.w)

    def mse_loss(self, x,ytrue ):
        ypred = self.linear_function(x)
        return np.mean(np.square(ytrue-ypred))

    def gradient(self , x , y ):
        ypred = self.linear_function(x)
        return -2*(np.dot(x, y - ypred))

    def predict(self, X):
            X = self.add_ones(X)  
            return self.linear_function(X, self.w)
    def evaluate(self, X, y):
        predictions = self.predict(X)
        loss = self.mse_loss(y, predictions)
        return loss
    def fit(self,x,y):
        x = self.add_ones(x)

        # reshape y if needed
        y=y.reshape(-1,1)

        # Initialize w to zeros vector >>> (x.shape[1])
        self.w = np.zeros((x.shape[1],1))

        for epoch in range(self.n_epochs):

        #compute the gradient
            gradient = self.gradient( x , y )

            #update rule
            self.w -= self.lr*gradient

            #Compute and append the training loss in a list
            loss = self.mse_loss(x , y)
            self.train_losses.append(loss)

            if epoch%1000 == 0:
                print(f'loss for epoch {epoch}  : {loss}')


    def accuracy(self,y_true, y_pred):
        acc = np.mean((y_pred == y_true))*100
        return acc

