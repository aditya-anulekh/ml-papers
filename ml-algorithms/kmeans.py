import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


class KMeans:
    def __init__(self, X, num_clusters, num_iterations=10):
        self.X = X
        self.num_clusters = num_clusters
        self.classes = np.array(range(self.num_clusters))
        self.num_iterations = num_iterations
        self.m = X.shape[0]  # Number of data points
        self.n = X.shape[1]  # Number of features
        self.centroids = np.zeros((self.num_clusters, self.n))
        self.predictions = np.zeros(self.m, dtype=int)

        for i in range(self.num_clusters):
            self.centroids[i, :] = X[np.random.choice(range(self.m))]

    def __str__(self):
        # Override builtin __str__ function to print information about clusters
        info = f"There are {self.num_clusters} clusters with centroids at \n"
        info += "\n".join([str(i) for i in self.centroids])

        return info

    def get_distance(self, point1, point2):
        dist = sum((point1 - point2) ** 2)
        return np.sqrt(dist)

    def get_centroid(self):
        # Iterate through all the clusters and calculate the mean
        self.centroids = np.array(
            [np.mean(self.X[self.predictions == i], axis=0) for i in self.classes]
        )

    def train(self):
        self.predictions = self.assign_cluster()
        for it in range(self.num_iterations):
            self.get_centroid()
            self.predictions = self.assign_cluster()
            if it % 2 == 0:
                print(self.__str__())
                error = 0
                for point_idx, point in enumerate(self.X):
                    error += self.get_distance(point, self.centroids[int(self.predictions[point_idx])])
                print(f"Error at {it} iteration = {error}")

    def assign_cluster(self):
        labels = []
        for point_idx, point in enumerate(self.X):
            labels.append(np.argmin([self.get_distance(point, centroid) for centroid in self.centroids]))

        return np.array(labels)

    def plot_clusters(self):
        colors = iter(cm.rainbow(np.linspace(0, 1, self.num_clusters)))
        for i in self.classes:
            plt.scatter(X[:, 0][self.predictions == i], X[:, 1][self.predictions == i], color=next(colors))
        plt.show()


if __name__ == "__main__":
    # X, y, centers = make_blobs(n_samples=300, centers=3, n_features=2, return_centers=True)
    # print(centers)
    data = pd.read_csv("data/mickey.csv", comment="#", sep=" ", header=None)
    X = np.array(list(zip(data[0], data[1])))
    kmeans = KMeans(X, num_clusters=3)
    kmeans.train()
    kmeans.plot_clusters()
