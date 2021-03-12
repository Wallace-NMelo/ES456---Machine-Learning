import numpy as np
from random import randrange


def mean(array):
    return np.around(array.sum(axis=0) / array.shape[0], decimals=3)


def cov_matrix(array):
    array -= mean(array)
    return np.dot(array.T, array) / array.shape[0]


def label_prob(Y):
    labels = []
    lab_prob = []
    for unique_ in Y.unique():
        prob = np.around(len(Y[Y == unique_]) / len(Y) * 100, decimals=3)
        labels.append(unique_)
        lab_prob.append(prob)
    return labels, lab_prob


# Split a dataset into a train and test set
def train_test_split(data, target, test_size=0.60):
    data_train = list()
    train_target = list()
    train_size = int((1 - test_size) * len(data))
    dataset_copy = list(data)
    target_copy = list(target)
    while len(data_train) < train_size:
        index = randrange(len(dataset_copy))
        data_train.append(dataset_copy.pop(index))
        train_target.append(target_copy.pop(index))

    return data_train, dataset_copy, train_target, target_copy


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true) * 100
    return accuracy


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) * 100


def kfoldcv(data, target, k=10):
    dataset_cp, target_cp = list(data), list(target)
    size, dim = np.shape(data)
    subset_size = int(size / k)
    data_train, target_train, k_folds = np.empty((0, dim), float), [], []
    for _ in range(k):
        for _ in range(subset_size):
            idx = randrange(len(dataset_cp))
            data_train = np.vstack((data_train, dataset_cp.pop(idx)))
            target_train.append(target_cp.pop(idx))
        k_folds.append((data_train, target_train))
        data_train, target_train = np.empty((0, dim), float), []

    return k_folds
