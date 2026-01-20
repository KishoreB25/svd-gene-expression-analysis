# src/ml/clustering.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans(X, n_clusters, random_state=42):
    """
    Runs KMeans clustering.

    Parameters:
    X : ndarray
        Reduced feature matrix (SVD space)
    n_clusters : int

    Returns:
    labels : ndarray
        Cluster assignments
    model : KMeans
    """
    model = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state
    )
    labels = model.fit_predict(X)
    return labels, model


def compute_silhouette(X, labels):
    """
    Computes silhouette score.
    """
    return silhouette_score(X, labels)
