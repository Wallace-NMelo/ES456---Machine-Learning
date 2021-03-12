import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, clusters, max_iter=5):
        self.clusters, self.max_iter = clusters, max_iter

    def fit(self, X):

        self.shape = X.shape
        self.n, self.m = self.shape

        self.phi, self.weights = np.ones(self.clusters) / self.clusters, np.ones(self.shape) / self.clusters

        random_row = np.random.randint(low=0, high=self.n, size=self.clusters)
        self.mu = [X[row_index, :] for row_index in random_row]
        self.sigma = [np.cov(X.T) for _ in range(self.clusters)]

        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)

    def e_step(self, X):

        self.weights = self.pred_prob(X)
        self.phi = self.weights.mean(axis=0)

    def m_step(self, X):

        for i in range(self.clusters):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T,
                                   aweights=(weight / total_weight).flatten(),
                                   bias=True)

    def pred_prob(self, X):
        likelihood = np.zeros((self.n, self.clusters))
        for i in range(self.clusters):
            distribution = multivariate_normal(mean=self.mu[i], cov=self.sigma[i])
            likelihood[:, i] = distribution.pdf(X)

        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    def predict(self, X):
        weights = self.pred_prob(X)
        return np.argmax(weights, axis=1)
