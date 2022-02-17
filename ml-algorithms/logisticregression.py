"""
Logistic Regression implementation

Equations and notation adapted from - http://cs230.stanford.edu/fall2018/section_files/section3_soln.pdf
"""

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, X, y, learning_rate=0.01, num_iterations=10000):
        self.lr = learning_rate
        self.X = X  # Shape = n x m
        self.y = y  # Shape = 1 x m
        self.num_iterations = num_iterations
        self.m = X.shape[1]  # Number of data points
        self.n = X.shape[0]  # Number of features for each data point

        self.weights = np.zeros((self.n, 1))  # Initialize the weights to zeros
        self.bias = 0  # Initialize the bias to 0

    def train(self):
        for it in range(self.num_iterations):
            z = np.dot(self.weights.T, self.X) + self.bias
            y_hat = self.sigmoid(z)
            loss = self.loss(y_hat)

            # Backprop
            dw = 1/self.m * np.dot(self.X, (y_hat - self.y).T)
            db = 1/self.m * np.sum(y_hat - self.y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

            if it % 200 == 0:
                print(f"Loss at iteration {it} = {loss}")

    def predict(self, x_test):
        z = np.dot(self.weights.T, x_test) + self.bias
        y_hat = self.sigmoid(z)
        return y_hat > 0.5

    def loss(self, y_hat):
        return -np.sum((self.y*np.log(y_hat)) + (1-self.y)*np.log(1-y_hat))

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))


if __name__ == "__main__":
    X, y = make_blobs(n_samples=100, centers=2, n_features=2)
    # Transpose X to comply with the rest of the dimensions
    X = X.T

    regression = LogisticRegression(X, y)
    regression.train()

    y_predicted = regression.predict(X).reshape(-1)
    fig, _ = plt.subplots(1, 2)
    fig.axes[0].scatter(X[0, :][y == 0], X[1, :][y == 0], color="r")
    fig.axes[0].scatter(X[0, :][y == 1], X[1, :][y == 1], color="b")

    fig.axes[1].scatter(X[0, :][y_predicted == 0], X[1, :][y_predicted == 0], color="r")
    fig.axes[1].scatter(X[0, :][y_predicted == 1], X[1, :][y_predicted == 1], color="b")
    plt.show()

