import numpy as np
from random import randrange


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


# Accuracy of predicted label
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy * 100


def mean_squared_error(y_true, y_pred):
    return np.average((y_true - y_pred) ** 2) * 100


def higher_dim(X):
    # Get unique input
    unique = np.array(X.drop_duplicates().reset_index(drop=True))
    # Dimension of new Array
    dim = len(unique)
    X_ = np.array(X)
    new_X = np.zeros((X.shape[0], dim), dtype=np.int)
    number = 1
    for idx, unique_ in enumerate(unique):
        array = np.arange(idx, idx + dim)
        new_X[(X_ == unique_).all(axis=1)] = array
        number *= 2
    return new_X
