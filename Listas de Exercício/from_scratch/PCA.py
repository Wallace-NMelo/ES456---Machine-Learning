import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance
        cov = np.cov(X.T)
        #
        ei_values, ei_vectors = np.linalg.eig(cov)
        # sort elements
        ei_vectors = ei_vectors.T
        idxs = np.argsort(ei_values)[::-1]
        ei_values = ei_values[idxs]
        ei_vectors = ei_vectors[idxs]

        # store
        self.components = ei_vectors[0:self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)
