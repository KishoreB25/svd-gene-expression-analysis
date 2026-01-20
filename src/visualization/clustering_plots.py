# src/visualization/clustering_plots.py

import matplotlib.pyplot as plt

def plot_clusters_2d(X_reduced, cluster_labels, save_path=None):
    """
    2D cluster visualization in SVD space.
    """

    plt.figure(figsize=(6,5))
    scatter = plt.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.7
    )
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("KMeans Clusters in SVD Space")
    plt.colorbar(scatter, label="Cluster ID")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
