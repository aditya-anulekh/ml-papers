import numpy as np
from sklearn.datasets import make_blobs


class KNN:
    def __init__(self, X, y, k=3):
        """
        Init method for K Nearest Neighbors
        :param X: np.array
        :param y: np.array
        :param k: int
        """
        self.X = X
        self.y = y
        self.k = k

    def get_distance(self, x_test):
        # Distance is going to be a matrix of dimensions (num of test points x num of training points)
        distances = np.zeros((len(x_test), self.X.shape[0]))
        for i, point in enumerate(x_test):
            distances[i, :] = np.sqrt(np.sum((self.X - point) ** 2, axis=1))
        return distances

    def predict(self, x_test):
        y_pred = np.zeros(len(x_test))
        distances = self.get_distance(x_test)
        predictions = self.y[distances.argsort(axis=1)][:, :self.k]
        for i in range(len(x_test)):
            y_pred[i] = np.argmax(np.bincount(predictions[i]))

        return y_pred


if __name__ == "__main__":
    X, y = make_blobs(n_samples=100, centers=2, n_features=2)
    knn = KNN(X, y)
    print(knn.predict(X))
