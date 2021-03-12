import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self.step_func
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape

        self.weights = np.random.uniform(low=-1, high=1, size=n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in Y])

        for iter in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def step_func(self, x):
        return np.where(x > 0, 1, 0)
