import numpy as np
from collections import Counter

# Function to calculate the Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Function to calculate the Manhattan distance
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class KNN:
    def __init__(self, k=3, distance_fn=euclidean_distance):
        self.k = k  # Number of neighbors
        self.distance_fn = distance_fn  # Distance function (default: Euclidean)

    def fit(self, X, y):
        # Store the training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Predict the class for each sample in X
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances from x to all training samples
        distances = [self.distance_fn(x, x_train) for x_train in self.X_train]
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
