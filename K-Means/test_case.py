import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-6, output_dir='kmeans_gif'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.image_files = []

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for i in range(self.max_iter):
            self.labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X)
            self._plot_intermediate_stage(X, i)
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids
        self._create_gif()

    def _assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X):
        return np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])

    def predict(self, X):
        return self._assign_clusters(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def _plot_intermediate_stage(self, X, iteration):
        plt.scatter(X[:, 0], X[:, 1], c=self.labels, s=50, cmap='viridis', alpha=0.6)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=50, c='white', marker='D', edgecolors="black")
        plt.title(f'Iteration {iteration + 1}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(f'{self.output_dir}/kmeans_{iteration + 1}.png')
        self.image_files.append(f'{self.output_dir}/kmeans_{iteration + 1}.png')
        plt.close()

    def _create_gif(self):
        images = []
        for filename in self.image_files:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'{self.output_dir}/kmeans.gif', images, duration=1, fps=2, loop=0)
        for filename in self.image_files:
            os.remove(filename)

### Helper Functions for Final Visualization

def plot_clusters(X, labels, centroids, title='K-Means Clustering'):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=50, c='white', marker='D', edgecolors="black")
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

### Example Usage

# Generate sample data
from sklearn.datasets import make_blobs
# create simulated data for examples
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, 
                  shuffle=False, random_state=0)

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(X)

# Plot the final clusters
# plot_clusters(X, labels, kmeans.centroids)
