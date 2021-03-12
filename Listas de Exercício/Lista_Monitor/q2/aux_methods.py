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


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return np.around(accuracy, decimals=4)*100


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)