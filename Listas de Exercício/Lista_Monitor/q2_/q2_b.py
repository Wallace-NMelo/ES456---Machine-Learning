# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:06:58 2020

@author: arthur
"""

import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path


# Path of current directory
current_path = Path.cwd()

# load the dataset
with open(current_path.joinpath('dataset/wine.data')) as f:
    dataset = [i.strip().split(',') for i in f.readlines()]
    data = pd.DataFrame(dataset, dtype=np.float32)

X_train, y_train = data.drop(columns=[0]), data.iloc[:, 0]
X_train = ((X_train - X_train.min()) / (X_train.max() - X_train.min()))


class k_meanspp:
    '''
    data: multidimensional data
    k: number of clusters
    max_iter: number of iterations

    '''

    def __init__(self, data, k=3, max_iter=20):
        self.k = k
        self.max_iter = max_iter
        np.random.seed(0)
        if isinstance(data, np.ndarray):
            self.X = data
        else:
            self.X = data.to_numpy()

        # initialize centers
        centers = []
        # take c1
        centers.append(self.X[np.random.choice(range(self.X.shape[0])), :].tolist())
        # take a new center
        for i in range(1, k):
            D = self.short_dis(self.X, centers)
            if i == 0:
                pdf = D / np.sum(D)
                new_center = self.X[np.random.choice(range(self.X.shape[0]), replace=False,
                                                     p=pdf.flatten())]
                centers.append(new_center.tolist())
            else:
                dist_min = np.min(D, axis=1)
                pdf = dist_min / np.sum(dist_min)
                new_center = self.X[np.random.choice(range(self.X.shape[0]), replace=False,
                                                     p=pdf.flatten())]
            centers.append(new_center.tolist())
        self.centers = np.array(centers)
        # print('Initial centers: ' + str(self.centers))

    def short_dis(self, data, centers):
        '''
        Shortest distance from a data point to the closest center
        we have already chosen
        '''
        distance = np.sum((np.array(centers) - data[:, None, :]) ** 2, axis=2)
        return distance

    def fit(self, targets):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = self.centers[i]

        for it in range(self.max_iter):
            self.r = {}  # categorization

            # expectation
            for j in range(self.k):
                self.r[j] = []
            for x in self.X:
                # euclidian distance
                J = [np.linalg.norm(x - self.centroids[c]) for c in self.centroids]
                index = J.index(min(J))
                self.r[index].append(x)

            # optmization
            for x_categ in self.r:
                # new centroid for the cluster
                self.centroids[x_categ] = np.mean(self.r[x_categ], axis=0)

            if it == self.max_iter - 1:
                print('Centros que convergiram:')
                for k in range(self.k):
                    print('c{}: {}\n'.format(k + 1, self.centroids[k]))

        for k in range(0, self.k):
            idx1 = np.where(targets == float(k + 1))[0][0]
            idx2 = np.where(targets == float(k + 1))[0][-1]
            xm = np.mean(self.X[idx1:idx2 + 1], axis=0)
            a = abs(list(self.centroids.values()) - xm)
            center = np.where(a == a.min(axis=0))[0]
            indices = Counter(center).most_common(1)[0][0]
            print('Centro mais proximo do vinho {}: \n{}'.format(k + 1, self.centroids[indices]))


clf = k_meanspp(data, k=3)
clf.fit(y_train)
