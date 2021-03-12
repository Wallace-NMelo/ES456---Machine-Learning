# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:36:06 2020

@author: arthur
"""

import numpy as np
import pandas as pd
from collections import Counter

class knn:
    def __init__(self, X_train, y_train, k):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def predict(self, test):
        new_X_train = self.X_train.copy()

        new_X_train['distances'] = np.linalg.norm(new_X_train.values - test.values, axis=1)
        new_X_train = pd.concat([new_X_train, self.y_train], axis=1)
        new_X_train = new_X_train.sort_values('distances', ascending=True)
        k_neighbors = new_X_train.head(self.k)

        labels = k_neighbors.loc[:, 0]
        # create counter of labels
        majority_count = Counter(labels)
        return majority_count.most_common(1).pop()[0]

    def fit(self, X_test):
        predictions = np.zeros(X_test.shape[0])
        X_test.reset_index(drop=True, inplace=True)
        for index, row in X_test.iterrows():
            predictions[index] = self.predict(row)

        return predictions


