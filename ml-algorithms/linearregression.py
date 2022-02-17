"""
Linear Regression Implementation
"""

import numpy as np


class LinearRegressionGradientDescent:
    def __init__(self, X, y, learning_rate=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.num_iterations = num_iterations
        # Append a column of ones to the input vector to account for the bias term
        self.X = np.append(np.ones((1, X.shape[1])), X, axis=0)  # Shape = n x m
        self.y = y  # Shape = 1 x m
        self.m = self.X.shape[1]  # Number of data points
        self.n = self.X.shape[0]  # Number of features for each data point
        # Initialize the weights with ones/zeros
        self.weights = np.ones((self.n, 1))  # Shape = n x 1

    def get_cost(self, y_hat):
        # Mean square Error = (y_hat - y)**2
        return (1/self.m)*np.sum(np.power((y_hat-self.y), 2))

    def train(self):
        for it in range(self.num_iterations):
            y_hat = np.dot(self.weights.T, self.X)
            cost = self.get_cost(y_hat)

            # Gradient Descent
            dCdW = (2/self.m) * np.dot(self.X, (y_hat - self.y).T)
            self.weights -= self.lr * dCdW
            if it % 200 == 0:
                print(f"Cost at {it} iteration:{cost}")

    def predict(self, x_test):
        return np.dot(self.weights.T, x_test)



if __name__ == "__main__":
    X = np.random.rand(2, 500)
    y = 3 * X[0, :] + 4 * X[1, :] + 5 + np.random.randn(1, 500) * 0.1
    regression = LinearRegressionGradientDescent(X, y)
    regression.train()
    print(regression.weights)
