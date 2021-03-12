# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:10:25 2020

@author: arthur
"""

import numpy as np

def mse(data, y_pred, y):
    m, n = data.shape
    #evaluate on the whole training set
    J = np.sum(1/m * (y_pred - y) ** 2)
    return J


class MLP():

    def __init__(self, shape, size_of_layers):
        self.shape = shape
        #samples and features
        self.n, self.m = self.shape
        self.layers = size_of_layers
        #initialize parameters
        self.params = {}
        self.params['w'] = [np.random.randn(y, x) * 0.01 for x, y in zip(self.layers[:-1], self.layers[1:])]
        self.params['b'] = [np.zeros((layer, 1)) for layer in self.layers[1:]]

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-self.Z))

    def sigmoid_prime(self, Z):
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def forward(self, X):
        """
        This funtion performs the forward propagation method using
        activation from previous layer:

        Parameters:
            X: input data coming into the layer from prev layer

        Return:
            Z: Linear function
            a: Activation function
        """

        for i in range(0, len(self.layers)-1):
            if i == 0:
                self.Z[i] = np.dot(self.params['w'][i], X.to_numpy().T) + self.params['b'][i]
                self.a[i] = self.relu(self.Z[i])
            else:
                self.Z[i] = np.dot(self.params['w'][i], self.a[i-1]) + self.params['b'][i]
                self.a[i] = self.relu(self.Z[i])
            #print(f'layer {i+1}:')
            #print(f'Z shape: {self.Z[i].shape}')
            #print(f'a shape: {self.a[i].shape}')
            #print('W shape: {}\n'.format(self.params['w'][i].shape))



    def backward(self, y):
        '''
        This function performs the backward propagation methog using
        '''
        #grad_table is a data structure that will store the derivatives that have been computed
        self.grad_table = {}
        self.grad_table['db'] = [np.zeros(b.shape) for b in self.params['b']]
        self.grad_table['dw'] = [np.zeros(w.shape) for w in self.params['w']]
        #dJ/da_2
        da2 = (self.a[-1] - y.to_numpy())
        upstream = da2 * np.heaviside(self.Z[-1], 0)
        #the last layer
        self.grad_table['dw'][-1] = np.dot(upstream, self.a[-2].T)
        self.grad_table['db'][-1] = upstream

        for l in range(2, len(self.layers)-1):
            upstream = (np.dot(self.params['w'][-l+1].T, upstream) *\
                        np.heaviside(self.Z[-l], 0))

            self.grad_table['dw'][-l] = upstream * self.a[-l-1].T
            self.grad_table['db'][-l] = upstream


    def fit(self, epochs, data, batch_size, l_r):
        #initialize linear function
        self.Z = [np.zeros((layer, batch_size)) for layer in self.layers[1:]]
        self.a = [np.zeros((layer, batch_size)) for layer in self.layers[1:]]

        n_samples = self.m

        for i in range(epochs):
            print(f'epoch {i}/{epochs}:')
            data = data.sample(frac=1)
            batches = [data[k:k+batch_size] for k in range(0, n_samples, batch_size)]
            for batch in batches:
                self.forward(batch.drop(columns=['target']))
                #cost = mse(batch, self.a[-1], batch['target'].to_numpy())
                #print(cost)
                self.backward(batch['target'])
                self.params['w'] = [w - (l_r/len(batch)) * dw
                                   for w, dw in zip(self.params['w'], self.grad_table['dw'])]
                self.params['b'] = [b - (l_r/len(batch)) * db
                                   for b, db in zip(self.params['b'], self.grad_table['db'])]

