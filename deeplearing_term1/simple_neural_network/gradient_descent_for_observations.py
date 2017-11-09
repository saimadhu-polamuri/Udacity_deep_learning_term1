#! /usr/bin/env python2

# Required Python Packages
import numpy as np
import pandas as pd


# Paths
DATA_PATH = "Inputs/binary.csv"


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def get_sigmoid_derivative(sigmoid_output):
    """
    Calculate sigmoid derivative with sigmoid output value
    :param sigmoid_output:
    :return:
    """

    return sigmoid_output * (1 - sigmoid_output)


def main():
    """

    :return:
    """
    admissions_data = pd.read_csv(DATA_PATH)
    print "Observation :: ", len(admissions_data.index)

    # Make dummy variables for rank
    data = pd.concat([admissions_data, pd.get_dummies(admissions_data['rank'], prefix='rank')], axis=1)
    data = data.drop('rank', axis=1)

    # Standarize features
    for field in ['gre', 'gpa']:
        mean, std = data[field].mean(), data[field].std()
        data.loc[:, field] = (data[field] - mean) / std

    # Split off random 10% of the data for testing
    np.random.seed(42)
    sample = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
    data, test_data = data.ix[sample], data.drop(sample)

    # Split into features and targets
    features, targets = data.drop('admit', axis=1), data['admit']
    features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)

    # Neural Network hyperparameters
    epochs = 1000
    learning_rate = 0.5

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):

            # Loop through all records, x is the input, y is the target

            # Note: We haven't included the h variable from the previous
            #       lesson. You can add it if you want, or you can calculate
            #       the h together with the output

            # Calculate the output
            # Calculate the linear combination of inputs and weights
            h = np.dot(x, weights)
            output = sigmoid(h)

            # Calculate the error
            error = y - output

            # Calculate the error term
            output_gradient = get_sigmoid_derivative(output)
            error_term = error * output_gradient

            # Calculate the change in weights for this sample
            #       and add it to the total weight change
            del_w += learning_rate * error_term * x

        # TODO: Update weights using the learning rate and the average change in weights
        weights += del_w

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss

    # Calculate accuracy on test data
    tes_out = sigmoid(np.dot(features_test, weights))
    predictions = tes_out > 0.5
    accuracy = np.mean(predictions == targets_test)
    print("Prediction accuracy: {:.3f}".format(accuracy))


if __name__ == "__main__":
    main()