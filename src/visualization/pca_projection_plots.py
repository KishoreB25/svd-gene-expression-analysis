# src/visualization/pca_projection_plots.py

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_pca_2d(X, labels, save_path=None):
    """
    2D PCA projection for comparison with SVD.
    """

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(6,5))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7
    )
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("2D Projection using PCA")
    plt.colorbar(scatter, label="Class Label")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
