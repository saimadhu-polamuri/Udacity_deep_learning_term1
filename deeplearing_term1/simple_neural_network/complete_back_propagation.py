#! /usr/bin/env python2

# Required Python Packages
import numpy as np
import pandas as pd

# Paths
DATA_PATH = "Inputs/binary.csv"

# Random seed
np.random.seed(21)


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def main():
    """

    :return:
    """
    # Load the admissions data
    admissions = pd.read_csv(DATA_PATH)

    # Make dummy variables for rank
    data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
    data = data.drop('rank', axis=1)

    # Split off random 10% of the data for testing
    np.random.seed(21)
    sample = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
    data, test_data = data.ix[sample], data.drop(sample)

    # Split into features and targets
    features, targets = data.drop('admit', axis=1), data['admit']
    features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

    # Hyperparameters
    n_hidden = 2  # number of hidden units
    epochs = 10    # Actual 900
    learning_rate = 0.005

    n_records, n_features = features.shape
    last_loss = None
    # Initialize weights
    weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                            size=(n_features, n_hidden))
    weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                             size=n_hidden)

    for e in range(epochs):
        del_w_input_hidden = np.zeros(weights_input_hidden.shape)
        del_w_hidden_output = np.zeros(weights_hidden_output.shape)

        for x, y in zip(features.values, targets):
            ## Forward pass ##
            # Calculate the output
            hidden_input = np.dot(x, weights_input_hidden)
            hidden_output = sigmoid(hidden_input)
            output_layer_input = np.dot(hidden_output, weights_hidden_output)
            output = sigmoid(output_layer_input)
            print "Actual output :: {}, Predicted output :: {}".format(y, output)

            ## Backward pass ##
            # Calculate the network's prediction error
            error = y - output

            # Calculate error term for the output unit
            output_error_term = error * output * (1 - output)

            ## propagate errors to hidden layer

            # Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(output_error_term, weights_hidden_output)

            # Calculate the error term for the hidden layer
            hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

            # Update the change in weights
            del_w_hidden_output += output_error_term * hidden_output
            del_w_input_hidden += hidden_error_term * x[:, None]

        # Update weights
        weights_input_hidden += (learning_rate * del_w_input_hidden) / float(n_records)
        weights_hidden_output += (learning_rate * del_w_hidden_output) / float(n_records)

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            hidden_output = sigmoid(np.dot(x, weights_input_hidden))
            out = sigmoid(np.dot(hidden_output,
                                 weights_hidden_output))
            loss = np.mean((out - targets) ** 2)

            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss

    # Calculate accuracy on test data
    hidden = sigmoid(np.dot(features_test, weights_input_hidden))
    out = sigmoid(np.dot(hidden, weights_hidden_output))
    predictions = out > 0.5
    accuracy = np.mean(predictions == targets_test)
    print("Prediction accuracy: {:.3f}".format(accuracy))



if __name__ == "__main__":
    main()