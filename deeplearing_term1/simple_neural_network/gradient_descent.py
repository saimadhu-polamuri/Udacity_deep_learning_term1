#! usr/bin/env python2

# Required Python Packages

import numpy as np


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of the Sigmoid function
    :param x:
    :return:
    """
    return sigmoid(x) * (1 - sigmoid(x))


def main():

    learning_rate = 0.5
    x = np.array([1, 2, 3, 4])
    y = np.array(0.5)

    # Initial weights
    w = np.array([0.5, -0.5, 0.3, 0.1])

    # Calculate the linear combination (h) value
    h = np.dot(x, w)
    # Calculate the network output with the h value
    nn_output = sigmoid(h)
    # Calculate the error (Actual - nn_output)
    error = y - nn_output
    # Calculate the output gradient
    output_gradient = sigmoid_derivative(h)
    # Calculate the error term
    error_term = error * output_gradient
    # Gradient descent
    del_w = learning_rate * error_term * x

    print('Neural Network output:')
    print(nn_output)
    print('Amount of Error:')
    print(error)
    print('Change in Weights:')
    print(del_w)

if __name__ == "__main__":
    main()
