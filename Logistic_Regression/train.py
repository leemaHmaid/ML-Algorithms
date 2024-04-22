from data import X_train , y_train , X_test , y_test
from main import LogisticRegression


# Fit Training data
model = LogisticRegression(0.01,n_epochs=10000)
model.fit(X_train,y_train)

ypred_train = model.predict(X_train)
acc = model.accuracy(y_train,ypred_train)
print(f"The training accuracy is: {acc}")
print(" ")

ypred_test = model.predict(X_test)
acc = model.accuracy(y_test,ypred_test)
print(f"The test accuracy is: {acc}")