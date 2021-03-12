import numpy as np
from aux_methods import accuracy, kfoldcv, train_test_split
from sklearn.datasets import load_iris
from MLP import MLP
from random import random
iris = load_iris()
#X, Y = iris_data.drop(['Class'], axis=1).values, iris_data['Class']
X,Y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

kfold = kfoldcv(X, Y, k=5)


#for kdata, klabel in kfold:
#    print(kdata, klabel)

#print("Perceptron classification accuracy", accuracy(y_test, predictions))
mlp = MLP(4, [8], 3)
inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
targets = np.array([[i[0] + i[1]] for i in inputs])

# train
mlp.fit(X_train, y_train, 100, 0.085)

input = np.array([0.3, 0.1])
target = np.array([0.4])

# get a prediction
y_pred = mlp.forward_process(X_test)
accuracy(y_test, y_pred)
#print()
#print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))