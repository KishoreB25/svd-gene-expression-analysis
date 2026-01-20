# src/visualization/svd_plots.py

import matplotlib.pyplot as plt
import numpy as np

def plot_singular_values(singular_values, save_path=None):
    """
    Scree plot of singular values.
    """

    plt.figure(figsize=(6,4))
    plt.plot(
        range(1, len(singular_values) + 1),
        singular_values,
        marker="o"
    )
    plt.xlabel("Component Index")
    plt.ylabel("Singular Value")
    plt.title("Singular Value Spectrum (Truncated SVD)")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_variance_explained(svd_model, save_path=None):
    """
    Cumulative variance explained plot.
    """

    cumulative_variance = np.cumsum(
        svd_model.explained_variance_ratio_
    )

    plt.figure(figsize=(6,4))
    plt.plot(
        cumulative_variance,
        marker="o"
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("Cumulative Variance Explained by SVD")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
