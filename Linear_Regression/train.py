from data import X_train , Y_train ,X_test, Y_test
from Class import LinearRegression


# Fit Training data
model =  LinearRegression(0.01,n_epochs=10000)
model.fit(X_train,Y_train)

ypred_train = model.predict(X_train)
acc = model.accuracy(Y_train,ypred_train)
print(f"The training accuracy is: {acc}")
print(" ")

ypred_test = model.predict(X_test)
acc = model.accuracy(Y_test,ypred_test)
print(f"The test accuracy is: {acc}")