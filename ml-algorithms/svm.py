import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class SVM:
    def __init__(self, X, y, lambda_param=0.1, learning_rate=0.001, num_iterations=1000):
        self.X = X
        self.y = np.where(y <= 0, -1, 1)
        self.m, self.n = X.shape  # m is the number of samples and n is the number of features
        self.num_iterations = num_iterations
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.weights = np.zeros((self.n, 1))
        self.bias = 0

    def train(self):
        for it in range(self.num_iterations):
            for point_idx, point in enumerate(self.X):
                point = point.reshape((-1, 1))
                y_hat = self.y[point_idx]*(np.dot(point.T, self.weights) - self.bias)
                if y_hat >= 1:
                    self.weights -= self.lr*(2*self.lambda_param*self.weights)
                else:
                    self.weights -= self.lr*(2*self.lambda_param*self.weights - self.y[point_idx]*point)
                    self.bias -= self.lr*self.y[point_idx]

    def predict(self, x_test):
        y_hat = (np.dot(x_test, self.weights) - self.bias)
        return np.sign(y_hat)

    def visualize_svm(self):
        x = np.linspace(min(self.X[:, 0]), max(self.X[:, 0]), 100)
        y = (-self.weights[0]*x + self.bias)/self.weights[1]
        plt.plot(x, y)
        plt.plot(x, y+1/self.weights[1])
        plt.plot(x, y-1/self.weights[1])
        plt.scatter(self.X[:, 0][self.y == -1], self.X[:, 1][self.y == -1], color="r")
        plt.scatter(self.X[:, 0][self.y == 1], self.X[:, 1][self.y == 1], color="b")
        plt.show()

if __name__ == "__main__":
    X, y = make_blobs(n_samples=100, centers=2, n_features=2)
    svm = SVM(X, y)
    svm.train()
    print(svm.weights, svm.bias)
    svm.visualize_svm()

