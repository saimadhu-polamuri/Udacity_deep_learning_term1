#! usr/bin/env python2

# Required Python Packages
import numpy as np

# Constants
INPUTS = np.array([0.7, -0.3])
WEIGHTS = np.array([0.1, 0.8])
BIAS = -0.1


def sigmoid(x):
    """
    Calculate the sigmoid output
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def main():

    # Calculate sigmoid input
    sigmoid_input = np.dot(INPUTS, WEIGHTS) + BIAS
    print "Sigmoid Input :: {}".format(sigmoid_input)
    sigmoid_output = sigmoid(sigmoid_input)
    print "Sigmoid Output :: {}".format(sigmoid_output)


if __name__ == "__main__":
    main()