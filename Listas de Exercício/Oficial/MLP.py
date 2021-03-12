import numpy as np
import math
from aux_methods import mean_squared_error


class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        # Initiate radom weights
        self.weights = [np.random.randn(x, y) for y, x in zip(self.layers[:-1], self.layers[1:])]
        self.bias = [np.zeros((layer, 1)) for layer in self.layers[1:]]
        # Create activations
        self.activation = [np.zeros(layer_) for layer_ in self.layers]
        # Create Derivatives
        self.derivatives = [np.zeros(layer_) for layer_ in self.layers[:-1]]

    def forward_process(self, inputs):
        activation = self.relu(inputs)[:,np.newaxis]
        self.activation[0] = activation
        for i, w in enumerate(self.weights):
            # calculate net inputs
            net_inputs = np.dot(w, activation)
            # calculate activation
            activation = self.relu(net_inputs)
            self.activation[i + 1] = activation
        return activation

    def back_propagate(self, error):

        for i in reversed(range(len(self.derivatives))):
            # get activation for previous layer
            activation = self.activation[i + 1]
            # apply sigmoid derivative function
            delta = np.array(error * self.sig_d(activation))
            # reshape delta has to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T
            # get activations for current layer
            current_activations = self.activation[i]
            # reshape activations as to have them as a 2d column matrix
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_re)
            # backpropogate the next error
            error = np.dot(self.weights[i].T, delta)
        return error

    def gradient_descent(self, learningRate=1):
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = np.transpose(self.derivatives[i])
            weights += derivatives * learningRate

    def fit(self, inputs, targets, epochs, lerning_rate, verbose=False):

        for epoch in range(epochs):
            sum_error = 0
            for j, (input, target) in enumerate(zip(inputs, targets)):
                # forward propagation
                output = self.forward_process(input)
                error = target - output
                # back propagation
                self.back_propagate(error)
                # apply gradient descent
                self.gradient_descent(lerning_rate)
                # report error
                sum_error += mean_squared_error(target, output)
            if verbose:
                print("Error : {0} at epoch {1}".format(sum_error,epoch))

    def sig(self, x):
        return 1 / (1 + (np.exp(-self.x)))

    def relu(self, x):
        x[x < 0] = 0
        return x

    def sig_d(self, x):
        return x * (1.0 - x)
