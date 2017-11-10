#! /usr/bin/env python2

# Required Python Packages
import numpy as np


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def main():

    # Network size
    input_size = 4
    hidden_size = 3
    output_size = 2

    np.random.seed(42)
    # Make some fake data
    x = np.random.randn(4)
    x = np.array(x, ndmin=2)

    weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(input_size, hidden_size))
    weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(hidden_size, output_size))

    # Make a forward pass through the network
    hidden_layer_in = np.dot(x, weights_input_to_hidden)
    hidden_layer_out = sigmoid(hidden_layer_in)
    print "Hidden layer output :: ", hidden_layer_out
    print "weights_hidden_to_output :: {}".format(weights_hidden_to_output)
    output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
    output_layer_out = sigmoid(output_layer_in)

    print "Output-layer Output :: {}".format(output_layer_out)


if __name__ == "__main__":
    main()