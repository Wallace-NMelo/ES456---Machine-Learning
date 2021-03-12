# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:36:06 2020

@author: arthur
"""

import numpy as np
import pandas as pd
from KNN import knn
import matplotlib.pyplot as plt
from pathlib import Path


# Path of current directory
current_path = Path.cwd()

# load the dataset
with open(current_path.joinpath('dataset/wine.data')) as f:
    dataset = [i.strip().split(',') for i in f.readlines()]
    data = pd.DataFrame(dataset, dtype=np.float32)

def standardization(X_train, X_test):
    X_train = ((X_train - X_train.min())/(X_train.max() - X_train.min()))
    X_test = ((X_test - X_test.min())/(X_test.max() - X_test.min()))
    return X_train, X_test

np.random.seed(2)
accuracies = []

train = data.sample(frac=0.8)
test = data.drop(train.index)
k_neighbors = np.arange(1, 12, 1)

for k in k_neighbors:
    X_train, y_train = train.drop(columns=[0]), train.iloc[:, 0]
    X_test, y_test = test.drop(columns=[0]), test.iloc[:, 0]
    X_train, X_test = standardization(X_train, X_test)
    c = knn(X_train, y_train, k=k)
    pred = c.fit(X_test)
    acc = np.mean(pred == (y_test.to_numpy()))
    accuracies.append(acc)

plt.figure(figsize=(20,20))
plt.grid(color='gray', linestyle=':', linewidth=1)
plt.plot(k_neighbors, accuracies, color='green', marker='o')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Testing the accuracy for different values of K')

