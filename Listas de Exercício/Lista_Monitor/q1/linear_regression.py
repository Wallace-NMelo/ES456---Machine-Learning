import numpy as np


class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        # init parameters
        n_samples, n_features = np.shape(X_train)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X_train, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(np.transpose(X_train), (y_predicted - y_train))
            db = (1 / n_samples) * np.sum(y_predicted - y_train)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        y_approx = np.dot(X_test, self.weights) + self.bias
        return y_approx
